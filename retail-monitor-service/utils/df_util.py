from pandas import DataFrame, to_datetime


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
