import awswrangler as wr
from pandas import DataFrame
from prefect import get_run_logger, task

from config import conf
from utils.df_util import cast_columns_as_date, cast_columns_as_type


@task
def retrieve_reference_data() -> DataFrame:
    """Retrieves training data csv

    Returns:
        DataFrame: Single DF of the training csv
    """
    logger = get_run_logger()
    logger.info('Retrieving csv files from s3 bucket into 1 dataframe...')

    return wr.s3.read_csv(path=conf.INPUT_REFERENCE_FILE)


@task
def clean_data(batch_data: DataFrame) -> DataFrame:
    """Cleans the batch data by renaming columns
        Fixes datatypes of features

    Args:
        batch_data (DataFrame): raw batch data

    Returns:
        DataFrame: cleaned batch data
    """
    logger = get_run_logger()

    logger.info('Mapping column names for proper processing...')
    mapping = conf.FEATURES_MAPPING.model_dump()
    # this is because it's reverse in conf
    clean_df = batch_data.rename(columns=dict(zip(mapping.values(), mapping.keys())))

    logger.info('Casting columnar data with proper dtypes')
    cast_columns_as_type(clean_df, conf.FEATURES_CAT, str)
    cast_columns_as_type(clean_df, conf.FEATURES_NUM, float)
    cast_columns_as_date(clean_df, conf.FEATURES_DATE)

    return clean_df


@task
def grab_features(batch_data: DataFrame, include_pred: bool = False) -> DataFrame:
    """Grabs features from batch data

    Args:
        batch_data (DataFrame): cleaned batch data

    Returns:
        DataFrame: df containing only the features
    """
    logger = get_run_logger()

    logger.info('Extracting features from batch data...')
    cols = conf.FEATURES_LIST
    if include_pred:
        cols += ['predictions']

    feature_df = batch_data[cols]
    return feature_df
