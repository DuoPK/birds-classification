import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge

from data_generation.utils import print_changed_values
from features import int_features, binary_features, categorical_features, range_features


class DataFiller:
    int_columns = int_features + binary_features + list(categorical_features.keys())

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.numeric_cols = list(range_features.keys()) + binary_features + list(categorical_features.keys())
        if "Ethnicity" in self.numeric_cols:
            self.numeric_cols.remove("Ethnicity")

    def fill_na_with_mean(self, info=False):
        df_filled = self.df.copy()
        for col in self.numeric_cols:
            if col in df_filled.columns:
                mean_value = df_filled[col].mean()
                if col in DataFiller.int_columns:
                    mean_value = round(mean_value)
                else:
                    mean_value = round(mean_value, 4)
                df_filled[col] = df_filled[col].fillna(mean_value)
        if "Ethnicity" in df_filled.columns:
            moda = df_filled["Ethnicity"].mode()[0]
            df_filled["Ethnicity"] = df_filled["Ethnicity"].fillna(moda)
        if info:
            print_changed_values(self.df, df_filled)
        return df_filled

    def fill_na_with_median(self, info=False):
        df_filled = self.df.copy()
        for col in self.numeric_cols:
            if col in df_filled.columns:
                median_value = df_filled[col].median()
                if col in DataFiller.int_columns:
                    median_value = round(median_value)
                else:
                    median_value = round(median_value, 4)
                df_filled[col] = df_filled[col].fillna(median_value)
        if "Ethnicity" in df_filled.columns:
            moda = df_filled["Ethnicity"].mode()[0]
            df_filled["Ethnicity"] = df_filled["Ethnicity"].fillna(moda)
        if info:
            print_changed_values(self.df, df_filled)
        return df_filled

    def fill_na_with_impute(self, info=False):
        df_filled = self.df.copy()

        # --- Etap 1: Imputacja cech przy użyciu IterativeImputer (BayesianRidge) ---
        # Wyłączamy Ethnicity, który będzie imputowany oddzielnie.
        features_for_imputation = [col for col in df_filled.columns if col != "Ethnicity"]
        imputer_full = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
        df_imputed = pd.DataFrame(imputer_full.fit_transform(df_filled[features_for_imputation]),
                                  columns=features_for_imputation, index=df_filled.index)
        # Uaktualniamy cechy.
        df_filled.update(df_imputed)

        # --- Etap 2: Imputacja zmiennej Ethnicity przy użyciu RandomForestClassifier ---
        if "Ethnicity" in df_filled.columns:
            mask_train = df_filled["Ethnicity"].notna()
            mask_missing = df_filled["Ethnicity"].isna()

            if mask_missing.sum() > 0 and mask_train.sum() > 0:
                X_train = df_filled.loc[mask_train, features_for_imputation]
                y_train = df_filled.loc[mask_train, "Ethnicity"].astype(int)
                X_missing = df_filled.loc[mask_missing, features_for_imputation]

                clf = RandomForestClassifier(random_state=42)
                clf.fit(X_train, y_train)
                predicted = clf.predict(X_missing)
                df_filled.loc[mask_missing, "Ethnicity"] = predicted

        # --- Etap 3: Postprocessing pozostałych zmiennych ---
        # Przycięcie zmiennych numerycznych do określonych zakresów.
        for col, (low, high) in range_features.items():
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].clip(lower=low, upper=high)

        # Zaokrąglenie i konwersja zmiennych całkowitych.
        int_cols = df_filled.columns.intersection(int_features)
        df_filled[int_cols] = df_filled[int_cols].round()

        # Progowanie zmiennych binarnych – wartości > 0.5 przypisujemy do 1, inaczej 0.
        binary_cols = df_filled.columns.intersection(binary_features)
        df_filled[binary_cols] = (df_filled[binary_cols] > 0.5).astype(float)

        # Przycięcie zmiennych kategorycznych do dozwolonych wartości.
        for col, allowed_values in categorical_features.items():
            if col in df_filled.columns:
                allowed_min = min(allowed_values)
                allowed_max = max(allowed_values)
                df_filled[col] = df_filled[col].round()
                df_filled[col] = df_filled[col].clip(lower=allowed_min, upper=allowed_max)

        if info:
            print_changed_values(self.df, df_filled)
        return df_filled
