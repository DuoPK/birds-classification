from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from data_generation.csv_generator import CSV
from data_generation.scaller import DataScaler

LABEL_COL = "Diagnosis"
MAPPING_DIR = Path("data/label_mappings")
MAPPING_DIR.mkdir(parents=True, exist_ok=True)
NEAR_ZERO_TOL = 1e-2
ZERO_ROW_THRESHOLD = 0.60


def _encode_labels(df, save_name: str):
    if LABEL_COL not in df.columns:
        print(f"[WARN] Kolumna '{LABEL_COL}' nie znaleziona – pomijam kodowanie etykiet.")
        return df

    if pd.api.types.is_numeric_dtype(df[LABEL_COL]):
        return df

    le = LabelEncoder()
    df[LABEL_COL] = le.fit_transform(df[LABEL_COL])

    mapping = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
    mapping_path = MAPPING_DIR / f"{save_name}_label_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    return df


def _drop_sparse_rows(df, tol: float = NEAR_ZERO_TOL, threshold: float = ZERO_ROW_THRESHOLD):
    num_cols = df.select_dtypes(include=[np.number]).columns.difference([LABEL_COL])
    if not len(num_cols):
        return df

    zero_frac = (df[num_cols].abs() < tol).sum(axis=1) / len(num_cols)
    mask_keep = zero_frac < threshold
    removed = (~mask_keep).sum()
    if removed:
        print(f"[INFO] Usunięto {removed} wierszy z >= {int(threshold * 100)}% zerowych wartości.")
    return df[mask_keep].reset_index(drop=True)


def data_generate(input_path: str = "data/filtered_features.csv", scaling: str = "minmax"):
    print(f"\n[START] Skalowanie: {scaling}")
    data_scaler = DataScaler()

    df = pd.read_csv(input_path)

    df = _drop_sparse_rows(df)

    if scaling == "minmax":
        df_scaled = data_scaler.minmax_scaler(df)
    else:
        df_scaled = data_scaler.standardize_data(df)

    df_encoded = _encode_labels(df_scaled, save_name=f"clean_{scaling}")

    output_path = f"data/clean_{scaling}.csv"
    CSV.save(df_encoded, output_path)
    print(f"[DONE] Zapisano przetworzone dane: {output_path}")


if __name__ == "__main__":
    data_generate(scaling="minmax")
    data_generate(scaling="standard")
