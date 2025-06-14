from pathlib import Path
import json

from sklearn.preprocessing import LabelEncoder

from data_generation.csv_generator import CSV
from data_generation.filler import DataFiller
from data_generation.scaller import DataScaler


LABEL_COL = "Diagnosis"
MAPPING_DIR = Path("data/label_mappings")
MAPPING_DIR.mkdir(parents=True, exist_ok=True)


def _encode_labels(df, save_name: str):
    import pandas as pd

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



def data_generate(fill_na: str = "mean", one_hot_encoding: bool = True, scaling: str = "minmax", info: bool = False):
    cleaner = DataFiller("data/filtered_features.csv")
    data_scaler = DataScaler()
    print(f"Fill NA with {fill_na}, one hot encoding={one_hot_encoding}, {scaling} scaling")

    if fill_na == "mean":
        df_intermediate = cleaner.fill_na_with_mean(info=info)
    elif fill_na == "median":
        df_intermediate = cleaner.fill_na_with_median(info=info)
    else:
        df_intermediate = cleaner.fill_na_with_impute(info=info)

    CSV.save(df_intermediate, f"data/fill_na/{fill_na}.csv")

    if scaling == "minmax":
        df_scaled = data_scaler.minmax_scaler(df_intermediate)
    else:
        df_scaled = data_scaler.standardize_data(df_intermediate)


    df_encoded = _encode_labels(df_scaled, save_name=f"{fill_na}_{scaling}")


    CSV.save(df_encoded, f"data/{fill_na}_{scaling}.csv")


if __name__ == "__main__":
    # MinMax
    data_generate(fill_na="mean", one_hot_encoding=True, scaling="minmax", info=True)
    data_generate(fill_na="median", one_hot_encoding=True, scaling="minmax", info=True)
    data_generate(fill_na="impute", one_hot_encoding=True, scaling="minmax", info=True)

    # Standard
    data_generate(fill_na="mean", one_hot_encoding=True, scaling="standard")
    data_generate(fill_na="median", one_hot_encoding=True, scaling="standard")
    data_generate(fill_na="impute", one_hot_encoding=True, scaling="standard")
