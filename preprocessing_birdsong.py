import os
import numpy as np
import pandas as pd
import librosa
import pywt
import scipy.signal as sg
from scipy.signal import butter, filtfilt, hilbert
from scipy.fftpack import fft, ifft
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import logging
import soundfile as sf
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)

# === PARAMETRY ===
SELECTED_SPECIES = ["houfin", "comwax", "calqua"]
BANDPASS_FILTER_TYPE = 'butter'   # 'savgol', 'butter', 'fft', 'wavelet'
DENOISING_FILTER_TYPE = 'savgol'  # 'savgol', 'wavelet', 'hilbert'
FEATURE_TYPE = 'mfcc'    # 'mfcc', 'mfe', 'wavelet', 'lpc', 'pca'
N_MFCC = 13  # Number of MFCC coefficients to extract
# RESAMPLE_RATE = 20000
MAX_SEGMENTS_PER_FILE = 10  # Maximum number of segments to process from each file
SEGMENT_LENGTH_MS = 512  # Length of each segment in milliseconds
SEGMENT_DISTANCE_MS = 400  # Minimum distance between segments in milliseconds
BANDPASS_LOW_FREQ = 1000  # Lower cutoff frequency for bandpass filter (Hz)
BANDPASS_HIGH_FREQ = 8000  # Upper cutoff frequency for bandpass filter (Hz)
BANDPASS_ORDER = 5  # Order of the Butterworth filter
PEAK_HEIGHT = 0.1  # Minimum height of peaks (normalized)
PEAK_PROMINENCE = 0.1  # Minimum prominence of peaks (normalized)
STANDARDIZE_FEATURES = True  # Whether to standardize features before saving

CSV_SAVE_DIR = Path("data_generation/data") / FEATURE_TYPE / BANDPASS_FILTER_TYPE / DENOISING_FILTER_TYPE
CSV_SAVE_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_CSV_FILE = (
    f"features-{FEATURE_TYPE}"
    f"_bandpass-{BANDPASS_FILTER_TYPE}"
    f"_denoise-{DENOISING_FILTER_TYPE}"
    f"{'_mfcc'+str(N_MFCC) if FEATURE_TYPE == 'mfcc' else ''}"
    f"_seg{SEGMENT_LENGTH_MS}ms"
    f"_dist{SEGMENT_DISTANCE_MS}ms"
    f"_maxseg{MAX_SEGMENTS_PER_FILE}"
    f"_bp{BANDPASS_LOW_FREQ}-{BANDPASS_HIGH_FREQ}Hz"
    f"{'_order' + str(BANDPASS_ORDER) if BANDPASS_FILTER_TYPE == 'butter' else ''}"
    f"_peak{PEAK_HEIGHT}"
    f"_prom{PEAK_PROMINENCE}.csv"
)

EXPORT_CSV_PATH = CSV_SAVE_DIR / EXPORT_CSV_FILE

SAMPLE_SAVE_DIR = Path("audio/processed_samples")
SAMPLE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

def save_audio_sample(signal, sr, filename, subdir):
    """Save audio sample to a file."""
    try:
        save_path = SAMPLE_SAVE_DIR / subdir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)  # Create all parent directories
        sf.write(str(save_path), signal, sr)
        logging.info(f"Saved audio sample to {save_path}")
    except Exception as e:
        logging.error(f"Error saving audio sample {filename} to {subdir}: {e}")

# === FILTRY ===
def butter_filter(data, cutoff, fs, filter_type='high', order=BANDPASS_ORDER):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data)

def wavelet_denoise(signal, wavelet='db1'):
    coeffs = pywt.wavedec(signal, wavelet)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(denoised, wavelet)

def fft_bandpass(signal, sr, freq_min=BANDPASS_LOW_FREQ, freq_max=BANDPASS_HIGH_FREQ):
    n = len(signal)
    fft_vals = fft(signal)
    freqs = np.fft.fftfreq(n, d=1/sr)
    fft_vals[(freqs < freq_min) | (freqs > freq_max)] = 0
    return np.real(ifft(fft_vals))

def apply_bandpass_filter(signal, sr, method='butter'):
    if method == 'butter':
        filtered = butter_filter(signal, BANDPASS_LOW_FREQ, sr, filter_type='high')
        filtered = butter_filter(filtered, BANDPASS_HIGH_FREQ, sr, filter_type='low')
        return filtered
    elif method == 'fft':
        return fft_bandpass(signal, sr)
    return signal

def apply_denoising_filter(signal, method='wavelet'):
    if method == 'savgol':
        return savgol_filter(signal, 101, 3)
    elif method == 'wavelet':
        return wavelet_denoise(signal)
    elif method == 'hilbert':
        analytic_signal = np.abs(hilbert(signal))
        return analytic_signal
    return signal

# === SEGMENTACJA ===
def segment_signal(signal, sr, window_size_ms=SEGMENT_LENGTH_MS, distance_ms=SEGMENT_DISTANCE_MS):
    # Convert window size and distance from milliseconds to samples
    window_size = int(sr * window_size_ms / 1000)
    distance = int(sr * distance_ms / 1000)
    
    # Normalize signal for better peak detection
    normalized_signal = signal / np.max(np.abs(signal))
    
    # Find peaks with more parameters
    peaks, properties = sg.find_peaks(normalized_signal, 
                                    distance=distance,
                                    height=PEAK_HEIGHT,  # Minimum height of peaks
                                    prominence=PEAK_PROMINENCE)  # Minimum prominence of peaks
    
    logging.info(f"Found {len(peaks)} peaks in signal of length {len(signal)}")
    logging.info(f"Window size: {window_size} samples ({window_size_ms}ms at {sr}Hz)")
    logging.info(f"Peak detection parameters: height={PEAK_HEIGHT}, prominence={PEAK_PROMINENCE}")
    
    segments = []
    for i, p in enumerate(peaks):
        start = max(0, p - window_size // 2)
        end = start + window_size
        if end <= len(signal):
            segment = signal[start:end]
            if len(segment) == window_size:  # Only add if segment has correct length
                segments.append(segment)
                logging.debug(f"Segment {i}: {len(segment)} samples ({len(segment)*1000/sr:.1f}ms)")
            else:
                logging.warning(f"Segment {i} has incorrect length: {len(segment)} (expected {window_size})")
        else:
            logging.warning(f"Peak {i} at position {p} would create segment beyond signal length")
    
    logging.info(f"Created {len(segments)} valid segments")
    return segments

def extract_mfe(signal, sr, n_mels=40, frame_stride=0.01):
    segment_len = int(sr * SEGMENT_LENGTH_MS / 1000)  # Convert ms to samples
    n_fft = min(segment_len, len(signal))
    if n_fft % 2 != 0:  # Ensure n_fft is even
        n_fft -= 1
    
    hop_length = min(int(sr * frame_stride), len(signal) // 4)
    if hop_length == 0:
        hop_length = 1
    
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft,
                                                     hop_length=hop_length, n_mels=n_mels,
                                                     power=2.0)
    log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
    scaler = StandardScaler()
    log_mel_norm = scaler.fit_transform(log_mel.T).T
    return log_mel_norm.T

# === CECHY ===
def extract_features(signal, sr, feature_type):
    if feature_type == 'mfcc':
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
        # Calculate statistics for each coefficient across time
        mfcc_stats = np.array([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.min(mfccs, axis=1),
            np.max(mfccs, axis=1)
        ])
        # Flatten the statistics into a single feature vector
        features = mfcc_stats.flatten()
        logging.debug(f"Extracted {N_MFCC} MFCC coefficients with 4 statistics each")
        return features
    elif feature_type == 'mfe':
        return extract_mfe(signal, sr)
    elif feature_type == 'wavelet':
        coeffs = pywt.wavedec(signal, 'db1')
        return np.hstack([np.mean(c) for c in coeffs])
    elif feature_type == 'lpc':
        return librosa.lpc(signal, order=13)
    elif feature_type == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=5)
        return pca.fit_transform(signal.reshape(-1, 1)).flatten()
    return signal

# === PRZETWARZANIE ===
def export_features_to_csv(filtered_data):
    rows = []
    label_names = filtered_data.features["ebird_code"].names
    processed_species = set()  # Track which species we've already saved samples for
    all_features = []  # Store all features for standardization
    feature_names = None
    
    for example in filtered_data:
        label = label_names[example['ebird_code']]
        if SELECTED_SPECIES and label not in SELECTED_SPECIES:
            continue

        filepath = example['filepath']
        filename = os.path.basename(filepath)
        base_filename = os.path.splitext(filename)[0]

        try:
            logging.info(f"Processing file: {filepath}")
            signal, sr = librosa.load(filepath, sr=None)
            
            # Save samples for the first file of each species
            if label not in processed_species:
                # Save original sample
                save_audio_sample(signal, sr, f"{base_filename}_original.wav", f"{label}/original")
                
                # Apply bandpass filter
                bandpass_signal = apply_bandpass_filter(signal, sr, method=BANDPASS_FILTER_TYPE)
                save_audio_sample(bandpass_signal, sr, f"{base_filename}_bandpass.wav", f"{label}/bandpass")

                # Apply denoising filter
                filtered_signal = apply_denoising_filter(bandpass_signal, DENOISING_FILTER_TYPE)
                save_audio_sample(filtered_signal, sr, f"{base_filename}_denoised.wav", f"{label}/denoised")

                # Save segments
                segments = segment_signal(filtered_signal, sr)
                if segments:
                    segment_count = 0
                    for i, seg in enumerate(segments):
                        if len(seg) > 0 and segment_count < 3:  # Save only first 3 non-empty segments
                            save_audio_sample(seg, sr, f"{base_filename}_segment_{i}.wav", f"{label}/segments")
                            segment_count += 1
                            logging.debug(f"Saved segment {i} of length {len(seg)} samples ({len(seg)*1000/sr:.1f}ms)")
                
                processed_species.add(label)
            else:
                # For subsequent files, just process without saving samples
                bandpass_signal = apply_bandpass_filter(signal, sr, method=BANDPASS_FILTER_TYPE)
                filtered_signal = apply_denoising_filter(bandpass_signal, DENOISING_FILTER_TYPE)
                segments = segment_signal(filtered_signal, sr)

            # if RESAMPLE_RATE:
            #     bandpass_signal = librosa.resample(bandpass_signal, orig_sr=sr, target_sr=RESAMPLE_RATE)
            #     sr = RESAMPLE_RATE

            if not segments:
                logging.warning(f"No segments found in file: {filepath}")
                continue

            # Process segments for feature extraction (limited to MAX_SEGMENTS_PER_FILE)
            valid_segments = [seg for seg in segments if len(seg) > 0]
            if len(valid_segments) > MAX_SEGMENTS_PER_FILE:
                logging.info(f"Limiting segments from {len(valid_segments)} to {MAX_SEGMENTS_PER_FILE} for file: {filepath}")
                valid_segments = valid_segments[:MAX_SEGMENTS_PER_FILE]

            for i, seg in enumerate(valid_segments):
                features = extract_features(seg, sr, FEATURE_TYPE)
                
                # Create feature names based on MFCC coefficients and their statistics
                if FEATURE_TYPE == 'mfcc':
                    if feature_names is None:
                        feature_names = []
                        for stat in ['mean', 'std', 'min', 'max']:
                            for coef in range(N_MFCC):
                                feature_names.append(f'mfcc_{coef}_{stat}')
                    row = {name: val for name, val in zip(feature_names, features)}
                else:
                    if feature_names is None:
                        feature_names = [f"f{j}" for j in range(len(features))]
                    row = {f"f{j}": val for j, val in enumerate(features)}
                
                row["Diagnosis"] = label
                row["Segment"] = i
                row["SourceFile"] = filename
                rows.append(row)
                all_features.append(features)

        except Exception as e:
            logging.error(f"Error processing {filepath}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Standardize features if requested
    if STANDARDIZE_FEATURES and all_features:
        logging.info("Standardizing features...")
        scaler = StandardScaler()
        
        # Get feature columns (excluding metadata)
        feature_cols = [col for col in df.columns if col not in ['Diagnosis', 'Segment', 'SourceFile']]
        
        # Fit and transform features
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        # Save scaler parameters
        scaler_params = pd.DataFrame({
            'mean': scaler.mean_,
            'scale': scaler.scale_
        })
        scaler_params.to_csv('feature_scaler_params.csv', index=False)
        logging.info("Saved scaler parameters to feature_scaler_params.csv")
    
    # Save to CSV
    df.to_csv(EXPORT_CSV_PATH, index=False)
    logging.info(f"Saved features to: {EXPORT_CSV_PATH}")
    
    # Print feature statistics
    if STANDARDIZE_FEATURES:
        logging.info("\nFeature statistics after standardization:")
        logging.info(f"Mean: {df[feature_cols].mean().mean():.3f}")
        logging.info(f"Std: {df[feature_cols].std().mean():.3f}")

# === URUCHOMIENIE ===
if __name__ == "__main__":
    from datasets import load_dataset
    # Wczytanie tylko podzbioru 'UHH'
    dataset = load_dataset("DBD-research-group/BirdSet", name="UHH", trust_remote_code=True)
    train_data = dataset["train"]

    export_features_to_csv(train_data)
