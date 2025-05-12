from pandas import DataFrame

def filter_outlier_data(df: DataFrame, feature: str, min: float, max: float):
    """
        Filter feature here is Purchase technically
        but to make it more configurable it's in config

        this method filters the dataframe according to the (min, max)
    """
    mask = (df[feature] >= min) & (df[feature] <= max)
    return df[mask]


def cast_columns_as_type(df: DataFrame, columns: list[str], dtype: type) -> None:
    """Cast given columns into given type

    Args:
        df (DataFrame): given input
        columns (list[str]): columns to cast
        dtype (type): type to cast with
    """
    for col in columns:
        df[col] = df[col].astype(dtype)


def cast_columns_as_date(df: DataFrame, columns: list[str]) -> None:
    """Cast given columns into date

    Args:
        df (DataFrame): given input
        columns (list[str]): columns to cast
    """
    for col in columns:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )
        df[col] = to_datetime(df[col], dayfirst=True)


def plot_duration_historgram(y_train, yp_train, y_valid, yp_valid) -> Figure:
    """creates hisogram plot for ff comparisons:
            - y_train (true) vs yp_train (pred)
            - y_valid (true) vs yp_valid (pred)

    Args:
        y_train (_type_): y training values
        yp_train (_type_): predicted y values from training set
        y_valid (_type_): y eval values
        yp_valid (_type_): predicted y values from eval set

    Returns:
        Figure: generated figure
    """
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    pred_config = {'label': 'pred', "color": 'C0', "stat": 'density', "kde": 'True'}
    true_config = {'label': 'true', "color": 'C1', "stat": 'density', "kde": 'True'}

    ax[0].set_title("train")
    sns.histplot(yp_train, ax=ax[0], **pred_config)
    sns.histplot(y_train, ax=ax[0], **true_config)

    ax[1].set_title("valid")
    sns.histplot(yp_valid, ax=ax[1], **pred_config)
    sns.histplot(y_valid, ax=ax[1], **true_config)

    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()

    return fig
