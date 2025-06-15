import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from pathlib import Path
import sys

# === PARAMETRY ===
INPUT_CSV = "filtered_features.csv"
OUTPUT_DIR = Path("processed_features")
NORMALIZATION_TYPE = "standard"  # "standard", "minmax", "robust"

# === LOGGING CONFIGURATION ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_preparation.log'),
        logging.StreamHandler()
    ]
)

def aggregate_features(df):
    """Aggregate MFCC features per segment using statistical functions."""
    feature_cols = [col for col in df.columns if col.startswith('f')]
    group_cols = ['Diagnosis', 'Segment', 'SourceFile']
    
    if not feature_cols:
        raise ValueError("No feature columns found in the dataframe")
    
    logging.info(f"Found {len(feature_cols)} feature columns")
    logging.info(f"Grouping by: {group_cols}")

    def compute_aggregates(group):
        stats = {}
        for col in feature_cols:
            values = group[col].values
            stats[f'{col}_mean'] = np.mean(values)
            stats[f'{col}_std'] = np.std(values)
            stats[f'{col}_min'] = np.min(values)
            stats[f'{col}_max'] = np.max(values)
            stats[f'{col}_median'] = np.median(values)
            stats[f'{col}_q25'] = np.percentile(values, 25)
            stats[f'{col}_q75'] = np.percentile(values, 75)
        return pd.Series(stats)

    grouped = df.groupby(group_cols)
    aggregated_df = grouped.apply(compute_aggregates).reset_index()
    
    logging.info(f"Aggregated features shape: {aggregated_df.shape}")
    return aggregated_df

def normalize_features(df, method='standard'):
    """Normalize features using specified scaling method."""
    feature_cols = [col for col in df.columns if col.startswith('f')]
    metadata_cols = ['Diagnosis', 'Segment', 'SourceFile']

    if not feature_cols:
        raise ValueError("No feature columns found in the dataframe")

    X = df[feature_cols]
    metadata = df[metadata_cols]

    if method == 'standard':
        scaler = StandardScaler()
        logging.info("Using StandardScaler (zero mean, unit variance)")
    elif method == 'minmax':
        scaler = MinMaxScaler()
        logging.info("Using MinMaxScaler (scale to [0,1] range)")
    elif method == 'robust':
        scaler = RobustScaler()
        logging.info("Using RobustScaler (robust to outliers)")
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    X_normalized = scaler.fit_transform(X)
    normalized_df = pd.DataFrame(X_normalized, columns=feature_cols)
    normalized_df = pd.concat([metadata, normalized_df], axis=1)

    return normalized_df, scaler

def save_scaler_params(scaler, method, output_dir):
    """Save essential scaler parameters."""
    try:
        params = {}
        if method == 'standard':
            params = {'mean': scaler.mean_, 'scale': scaler.scale_}
        elif method == 'minmax':
            params = {
                'min': scaler.min_,
                'scale': scaler.scale_,
                'data_min': scaler.data_min_,
                'data_max': scaler.data_max_
            }
        elif method == 'robust':
            params = {'center': scaler.center_, 'scale': scaler.scale_}

        for name, value in params.items():
            np.save(output_dir / f"{method}_{name}.npy", value)
            logging.info(f"Saved {method} scaler parameter: {name}")
    except Exception as e:
        logging.error(f"Error saving scaler parameters: {e}")
        raise

def main():
    try:
        # Verify input file exists
        if not Path(INPUT_CSV).exists():
            raise FileNotFoundError(f"Input file {INPUT_CSV} not found")

        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Load and verify data
        logging.info(f"Loading input CSV: {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV)
        logging.info(f"Loaded {len(df)} samples with {len(df.columns)-3} features")

        # Aggregate features
        logging.info("Aggregating MFCC features per segment")
        aggregated_df = aggregate_features(df)

        # Normalize features
        logging.info(f"Normalizing features using {NORMALIZATION_TYPE} normalization")
        normalized_df, scaler = normalize_features(aggregated_df, method=NORMALIZATION_TYPE)

        # Save results
        output_file = OUTPUT_DIR / f"normalized_{NORMALIZATION_TYPE}_aggregated_features.csv"
        normalized_df.to_csv(output_file, index=False)
        logging.info(f"Saved normalized aggregated features to: {output_file}")

        # Save scaler parameters
        save_scaler_params(scaler, NORMALIZATION_TYPE, OUTPUT_DIR)

        # Print summary
        logging.info("\nFeature preparation summary:")
        logging.info(f"Input samples: {len(df)}")
        logging.info(f"Output samples: {len(normalized_df)}")
        logging.info(f"Number of features: {len([col for col in normalized_df.columns if col.startswith('f')])}")
        logging.info("\nClass distribution:")
        logging.info(normalized_df['Diagnosis'].value_counts())

    except Exception as e:
        logging.error(f"Error in feature preparation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
