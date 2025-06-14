class CSV:
    @staticmethod
    def save(df, file_name):
        df.to_csv(file_name, index=False)
