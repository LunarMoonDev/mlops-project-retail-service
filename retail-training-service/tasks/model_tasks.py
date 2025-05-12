import pandas as pd
from prefect import get_run_logger, task
from sklearn.pipeline import Pipeline

from utils.feature_util import feature_pipeline


@task
def execute_feature_pipeline(
    pipeline: Pipeline, x_train: pd.DataFrame, x_valid: pd.DataFrame, pipe_tag: str
) -> tuple:
    """creates feature pipeline with given pipeline

    Args:
        pipeline (Pipeline): pipeline to execute for given data
        X_train (pd.DataFrame): X training set
        X_valid (pd.DataFrame): X eval set
        pipe_tag (str): label describing the feature pipe

    Returns:
        tuple: returns tuple of feature_pipe, training data, eval data
    """
    logger = get_run_logger()

    logger.info("Creating feature pipeline for [%s] ...", pipe_tag)
    feature_pipe = feature_pipeline(pipeline)

    logger.info("Fitting and Transforming for train and eval data ...")
    train_data = feature_pipe.fit_transform(x_train)
    valid_data = feature_pipe.transform(x_valid)

    return feature_pipe, train_data, valid_data


@task
def execute_model(
    trainer_class, model_data: dict, feature_pipe: Pipeline, pipe_tag: str
) -> Pipeline:
    """executes the given trainer

    Args:
        trainer_class (_type_): Trainer that trains their model
        model_data (dict): contains model required data
        feature_pipe (Pipeline): feature pipe to include in model pipe
        pipe_tag (str): label for feature pipe

    Returns:
        Pipeline: model pipeline
    """
    logger = get_run_logger()

    logger.info("Starting trainer[%s] process....", trainer_class.__name__)
    trainer = trainer_class(model_data, feature_pipe, pipe_tag)

    model_pipe = trainer.run(logger)
    return model_pipe
