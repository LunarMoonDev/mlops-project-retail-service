import mlflow
from prefect import get_run_logger, task

from config import conf


@task
def set_experiment_name() -> None:
    """sets the experiment name for mlflow with config"""
    logger = get_run_logger()

    logger.info("Setting experiment name [%s] for mlflow....", conf.EXPERIMENT_NAME)
    mlflow.set_tracking_uri(conf.TRACKING_URI)
    mlflow.set_experiment(experiment_name=conf.EXPERIMENT_NAME)

    return mlflow.get_experiment_by_name(conf.EXPERIMENT_NAME)
