class DataScaler:

    def get_f_columns(self, df):
        return [col for col in df.columns if col.startswith('f')]

    def minmax_scaler(self, df, new_max=1, new_min=0):
        df_scaled = df.copy()
        f_columns = self.get_f_columns(df)

        for col in f_columns:
            min_value = df_scaled[col].min()
            max_value = df_scaled[col].max()
            if max_value != min_value:
                df_scaled[col] = ((df_scaled[col] - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min
            else:
                df_scaled[col] = new_min

        return df_scaled

    def standardize_data(self, df):
        df_standardized = df.copy()
        f_columns = self.get_f_columns(df)

        for col in f_columns:
            mean_value = df_standardized[col].mean()
            std_value = df_standardized[col].std()
            if std_value != 0:
                df_standardized[col] = (df_standardized[col] - mean_value) / std_value
            else:
                df_standardized[col] = 0

        return df_standardized
