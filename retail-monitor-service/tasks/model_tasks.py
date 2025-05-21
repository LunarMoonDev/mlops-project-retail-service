from mlflow.pyfunc import PyFuncModel, load_model
from pandas import DataFrame
from prefect import get_run_logger, task

from config import conf


@task
def retrieve_model() -> PyFuncModel:
    """Retrieves the model from mlflow registry

    Returns:
        PyFuncModel: PyFuncModel class of the model
    """
    logger = get_run_logger()
    model_name = conf.REGISTRY_MODEL_NAME
    alias_name = conf.REGISTRY_ALIAS_NAME

    logger.info('Retrieving model from mlflow registry...')
    logger.info('Model name: %s; Alias name: %s', model_name, alias_name)
    champion_model = load_model(f'models:/{model_name}@{alias_name}')
    return champion_model


@task
def execute_model(feature_df: DataFrame, model: PyFuncModel) -> DataFrame:
    """Executes pyfunc model with predict method

    Args:
        feature_df (DataFrame): df containing features

    Returns:
        DataFrame: df of the batch data with prediction included
    """
    logger = get_run_logger()

    logger.info('Predicting with given feature dataframe...')
    feature_df['predictions'] = model.predict(feature_df)
    return feature_df
