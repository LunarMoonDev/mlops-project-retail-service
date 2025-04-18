from pandas import DataFrame

def filter_outlier_data(df: DataFrame, feature: str, min: float, max: float):
    """
        Filter feature here is Purchase technically
        but to make it more configurable it's in config

        this method filters the dataframe according to the (min, max)
    """
    mask = (df[feature] >= min) & (df[feature] <= max)
    return df[mask]

