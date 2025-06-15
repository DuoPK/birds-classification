import os
import logging
from pathlib import Path
import librosa
import soundfile as sf
from preprocessing_birdsong import (
    SELECTED_SPECIES, BANDPASS_FILTER_TYPE, RESAMPLE_RATE,
    apply_bandpass_filter, apply_denoising_filter, segment_signal,
    SEGMENT_LENGTH_MS, SEGMENT_DISTANCE_MS
)

# === LOGGING CONFIGURATION ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('filtered_samples.log'),
        logging.StreamHandler()
    ]
)

# === SAMPLE SAVING CONFIGURATION ===
FILTERED_SAMPLES_DIR = Path("filtered_samples")
FILTERED_SAMPLES_DIR.mkdir(exist_ok=True)

def save_audio_sample(signal, sr, filename, subdir):
    """Save audio sample to a file."""
    try:
        save_path = FILTERED_SAMPLES_DIR / subdir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(save_path), signal, sr)
        logging.info(f"Saved audio sample to {save_path}")
    except Exception as e:
        logging.error(f"Error saving audio sample {filename} to {subdir}: {e}")

def process_and_save_samples(dataset):
    """Process and save samples for each species with different denoising filters."""
    label_names = dataset.features["ebird_code"].names
    processed_species = set()
    
    # Available denoising filters
    denoising_filters = ['savgol', 'wavelet', 'hilbert']
    
    logging.info(f"Starting processing. Will save samples to: {FILTERED_SAMPLES_DIR}")
    logging.info(f"Processing species: {SELECTED_SPECIES}")
    
    for example in dataset:
        label = label_names[example['ebird_code']]
        if label not in SELECTED_SPECIES or label in processed_species:
            continue

        filepath = example['filepath']
        filename = os.path.basename(filepath)
        base_filename = os.path.splitext(filename)[0]

        try:
            logging.info(f"\nProcessing file for species {label}: {filepath}")
            signal, sr = librosa.load(filepath, sr=None)
            
            # Save original sample
            save_audio_sample(signal, sr, f"{base_filename}_original.wav", f"{label}/original")
            logging.info(f"Saved original sample for {label}")
            
            # Apply bandpass filter
            bandpass_signal = apply_bandpass_filter(signal, sr, method=BANDPASS_FILTER_TYPE)
            save_audio_sample(bandpass_signal, sr, f"{base_filename}_bandpass.wav", f"{label}/bandpass")
            logging.info(f"Saved bandpass filtered sample for {label}")

            # Try each denoising filter
            for denoising_type in denoising_filters:
                logging.info(f"\nProcessing {denoising_type} filter for {label}")
                # Apply denoising filter
                filtered_signal = apply_denoising_filter(bandpass_signal, method=denoising_type)
                
                # Save full filtered file
                save_audio_sample(filtered_signal, sr, 
                                f"{base_filename}_{denoising_type}_full.wav", 
                                f"{label}/{denoising_type}")
                logging.info(f"Saved full {denoising_type} filtered file for {label}")

                # Save segments
                segments = segment_signal(filtered_signal, sr)
                if segments:
                    segment_count = 0
                    for i, seg in enumerate(segments):
                        if len(seg) > 0 and segment_count < 3:
                            save_audio_sample(seg, sr, 
                                            f"{base_filename}_{denoising_type}_segment_{i}.wav", 
                                            f"{label}/{denoising_type}/segments")
                            segment_count += 1
                            logging.info(f"Saved segment {i} for {denoising_type} filter")
                    if segment_count < 3:
                        logging.warning(f"Found only {segment_count} valid segments for {denoising_type} filter")
                else:
                    logging.warning(f"No segments found for {denoising_type} filter")

            processed_species.add(label)
            logging.info(f"\nCompleted processing for species: {label}")
            logging.info(f"Processed species so far: {processed_species}")

        except Exception as e:
            logging.error(f"Error processing {filepath}: {e}")

    logging.info(f"\nFinished processing all samples. Processed species: {processed_species}")
    logging.info(f"All files saved in: {FILTERED_SAMPLES_DIR}")

if __name__ == "__main__":
    from datasets import load_dataset
    # Load only 'UHH' subset
    dataset = load_dataset("DBD-research-group/BirdSet", name="UHH", trust_remote_code=True)
    train_data = dataset["train"]
    
    process_and_save_samples(train_data)
    logging.info("Finished processing all samples")
