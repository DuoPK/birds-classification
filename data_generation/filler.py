import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import numpy as np

def print_changed_values(original_df, df):
    changed_cells = []
    for column in df.columns:
        null_mask = original_df[column].isnull()
        changed_indices = df.index[null_mask]
        for idx in changed_indices:
            diagnosis = df.at[idx, 'Diagnosis'] if 'Diagnosis' in df.columns else 'Brak diagnozy'
            changed_cells.append((idx, column, df.at[idx, column], diagnosis))

    print("Lista imputowanych wartości:")
    for idx, column, value, diagnosis in changed_cells:
        print(f"Indeks: {idx}, Cecha: {column}, Nowa wartość: {value}, Diagnoza: {diagnosis}")

class DataFiller:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.feature_cols = [col for col in self.df.columns if col.startswith("f")]

    def convert_zeros_to_nan(self, columns=None):
        df_updated = self.df.copy()
        if columns is None:
            columns = self.feature_cols
        df_updated[columns] = df_updated[columns].replace(0, np.nan)
        self.df = df_updated

    def fill_na_with_mean(self, info=False):
        df_filled = self.df.copy()
        for col in self.feature_cols:
            if col in df_filled.columns:
                mean_value = df_filled[col].mean()
                df_filled[col] = df_filled[col].fillna(round(mean_value, 4))
        if info:
            print_changed_values(self.df, df_filled)
        return df_filled

    def fill_na_with_median(self, info=False):
        df_filled = self.df.copy()
        for col in self.feature_cols:
            if col in df_filled.columns:
                median_value = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(round(median_value, 4))
        if info:
            print_changed_values(self.df, df_filled)
        return df_filled

    def fill_na_with_impute(self, info=False):
        self.convert_zeros_to_nan()

        df_filled = self.df.copy()

        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
        df_imputed = pd.DataFrame(imputer.fit_transform(df_filled[self.feature_cols]),
                                  columns=self.feature_cols,
                                  index=df_filled.index)

        df_filled[self.feature_cols] = df_imputed

        if info:
            print_changed_values(self.df, df_filled)
        return df_filled
