from datetime import datetime, timedelta

import awswrangler as wr
from pandas import DataFrame
from prefect import get_run_logger, task

from config import conf
from utils.df_util import cast_columns_as_date, cast_columns_as_type


@task
def retrieve_data() -> DataFrame:
    """Retrieves batch data of today's date plus some offset
        offset should be in days to move date back to desired date
        Grabs N csv files and combines them into a single Dataframe

    Returns:
        DataFrame: Single DF for all csv in the date's bucket
    """
    logger = get_run_logger()

    # grabs today date plus offset
    today = datetime.now() - timedelta(days=conf.INPUT_DAY_OFFSET)
    today_str = today.strftime('%Y-%m-%d')

    logger.info('Retrieving csv files from s3 bucket into 1 dataframe...')
    # path format will be s3://data/batch/{data_path}/####.csv
    INPUT_BATCH_PATH = conf.INPUT_BATCH_PATH.format(data_path=today_str)
    return wr.s3.read_csv(
        path=INPUT_BATCH_PATH,
        path_suffix='.csv',
        ignore_empty=True,
        use_threads=conf.INPUT_READ_THREADS,
    )


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
def grab_features(batch_data: DataFrame) -> DataFrame:
    """Grabs features from batch data

    Args:
        batch_data (DataFrame): cleaned batch data

    Returns:
        DataFrame: df containing only the features
    """
    logger = get_run_logger()

    logger.info('Extracting features from batch data...')
    feature_df = batch_data[conf.FEATURES_LIST]

    return feature_df

@task
def combine_data(orig_data: DataFrame, pred_data: DataFrame) -> DataFrame:
    """Combines the original data with customer id and etc with prediction df

    Args:
        orig_data (DataFrame): original data with customer id and other col
        pred_data (DataFrame): dataframe with predictions

    Returns:
        DataFrame: combined dataframe containing orig's cols + prediction
    """

    # change to inner join if we want to ignore records that got removed during pipeline
    combined_df = orig_data.join(pred_data, rsuffix='_dup')
    return combined_df