from data_generation.csv_generator import CSV
from data_generation.filler import DataFiller
from data_generation.scaller import DataScaler


def data_generate(fill_na="mean", one_hot_encoding=True, scaling="minmax", info=False):
    cleaner = DataFiller("../ready_dataset_to_preprocessing.csv")
    data_scaler = DataScaler()
    print(f"Fill NA with {fill_na}, one hot encoding={one_hot_encoding}, {scaling} scaling")

    if fill_na == "mean":
        df_filled = cleaner.fill_na_with_mean(info=info)
    elif fill_na == "median":
        df_filled = cleaner.fill_na_with_median(info=info)
    else:
        df_filled = cleaner.fill_na_with_impute(info=info)
    CSV.save(df_filled, f"data/fill_na/{fill_na}.csv")

    if scaling == "minmax":
        df_scaled = data_scaler.minmax_scaler(df_filled)
    else:
        df_scaled = data_scaler.standardize_data(df_filled)

    CSV.save(df_scaled, f"data/{fill_na}_{scaling}.csv")


if __name__ == "__main__":
    data_generate(fill_na="mean", one_hot_encoding=True, scaling="minmax", info=True)
    data_generate(fill_na="median", one_hot_encoding=True, scaling="minmax", info=True)
    data_generate(fill_na="impute", one_hot_encoding=True, scaling="minmax", info=True)

    data_generate(fill_na="mean", one_hot_encoding=True, scaling="standard")
    data_generate(fill_na="median", one_hot_encoding=True, scaling="standard")
    data_generate(fill_na="impute", one_hot_encoding=True, scaling="standard")
