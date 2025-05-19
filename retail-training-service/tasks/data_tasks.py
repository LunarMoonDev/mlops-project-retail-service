from datetime import datetime, timedelta

import awswrangler as wr
import pandas as pd
from prefect import get_run_logger, task
from sklearn.model_selection import train_test_split

from config import conf
from utils.df_util import (
    cast_columns_as_date,
    cast_columns_as_type,
    filter_outlier_data,
)


@task
def retrieve_data() -> pd.DataFrame:
    """Retrieves the batch data from the bucket
        grabbing the latest data in the bucket

    Returns:
        DataFrame: batch data
    """
    logger = get_run_logger()

    logger.info("Retrieving csv data from s3 bucket as dataframe ...")
    return wr.s3.read_csv(path=conf.BATCH_DATA_PATH)


@task
def clean_data(batch_data: pd.DataFrame) -> pd.DataFrame:
    """Cleans the batch data by renaming columns and
        fixing datatypes of features

    Args:
        batch_data (DataFrame): batch data

    Returns:
        DataFrame: cleaned data
    """
    logger = get_run_logger()

    logger.info("Mapping column names for proper renaming ...")
    # rename columns
    clean_df = batch_data.rename(columns=conf.FEATURES_MAPPING)

    logger.info("Casting columnar data with proper dtypes ...")
    cast_columns_as_type(clean_df, conf.CATEGORICALS, str)
    cast_columns_as_type(clean_df, conf.NUMERICALS, float)
    cast_columns_as_date(clean_df, conf.FEATURES_DATE)

    logger.info("Dropping N/A values in target data ...")
    # dropping na on target
    clean_df.dropna(subset=[conf.FEATURE_TARGET], inplace=True)
    return clean_df


@task
def split_data(batch_data: pd.DataFrame) -> tuple:
    """Splits the batch dataframe into train and valid datasets

    Args:
        batch_data (DataFrame): batch data

    Returns:
        tuple: tuple of x_train, x_valid, y_train, y_valid
    """
    logger = get_run_logger()

    # create x and y dataframes
    x_batch_data = batch_data[conf.FEATURES_LIST]
    y_batch_data = batch_data[conf.FEATURE_TARGET]

    logger.info("Splitting batch data into x train and valid, same for y ...")
    # splits the x and y data into train and valid
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_batch_data,
        y_batch_data,
        test_size=conf.BATCH_TEST_SIZE,
        random_state=conf.RANDOM_STATE,
    )
    return x_train, x_valid, y_train, y_valid


@task
def filter_features(batch_data: pd.DataFrame) -> pd.DataFrame:
    """Filters the batch data according to config
        toggled by FILTER_TOGGLE

    Args:
        batch_data (DataFrame): batch data

    Returns:
        DataFrame: dataframe filtered on features
    """
    logger = get_run_logger()

    if conf.FILTER_TOGGLE:
        logger.info("Filtering outlier values in %s data...", conf.FILTER_FEATURE)
        return filter_outlier_data(
            batch_data, conf.FILTER_FEATURE, conf.FILTER_MAX, conf.FILTER_MIN
        )
    return batch_data


@task
def drop_na_features(batch_data: pd.DataFrame) -> pd.DataFrame:
    """Drops na on the specified feature in the batch_data
        toggled by DROP_TOGGLE

    Args:
        batch_data (DataFrame): batch data

    Returns:
        DataFrame: dataframe without n/a features
    """
    logger = get_run_logger()

    if conf.DROP_TOGGLE:
        logger.info("Dropping N/A values in %s data...", conf.FILTER_FEATURE)
        return batch_data.dropna(subset=[conf.FILTER_FEATURE])

    return batch_data
