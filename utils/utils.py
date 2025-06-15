import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config import RANDOM_STATE


def load_dataset(dataset_path: str):
    """Load and validate dataset"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError("Dataset is empty")

    if 'Diagnosis' not in df.columns or 'SourceFile' not in df.columns:
        raise ValueError("Dataset must contain 'Diagnosis' and 'SourceFile' columns")

    # Drop metadata columns
    metadata_cols = ['Diagnosis', 'Segment', 'SourceFile']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    return df, feature_cols


def split_data_by_source(df, feature_cols, test_size=0.2, random_state=RANDOM_STATE):
    """Split data based on source files to prevent data leakage"""
    # Get unique source files
    source_files = df['SourceFile'].unique()
    if len(source_files) < 2:
        raise ValueError("Not enough unique source files for splitting")

    # Split source files into train and test
    train_files, test_files = train_test_split(
        source_files,
        test_size=test_size,
        random_state=random_state,
        stratify=df.groupby('SourceFile')['Diagnosis'].first()
    )

    # Split data based on source files
    train_data = df[df['SourceFile'].isin(train_files)]
    test_data = df[df['SourceFile'].isin(test_files)]

    if train_data.empty or test_data.empty:
        raise ValueError("Empty dataset after splitting")

    # Prepare feature matrices and labels
    X_train = train_data[feature_cols]
    y_train = train_data['Diagnosis']
    X_test = test_data[feature_cols]
    y_test = test_data['Diagnosis']

    if X_train.empty or X_test.empty:
        raise ValueError("Empty feature matrix after splitting")

    return X_train, X_test, y_train, y_test, train_files, test_files


def find_csv_paths(
        base_dir="data_generation/data",
        feature_type=None,
        bandpass_filter_type=None,
        denoising_filter_type=None
):
    """Returns a list of paths to CSV files that meet the given filters."""
    paths = []
    for p in Path(base_dir).rglob("*.csv"):
        parts = p.parts
        if feature_type and feature_type not in parts:
            continue
        if bandpass_filter_type and bandpass_filter_type not in parts:
            continue
        if denoising_filter_type and denoising_filter_type not in parts:
            continue
        paths.append(str(p))
    return paths
