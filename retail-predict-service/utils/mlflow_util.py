import mlflow

from config import conf


def configure_mlflow_setup() -> None:
    """configure mlflow settings like host setup"""

    # TODO: figure out how to log outside task
    mlflow.set_tracking_uri(conf.REGISTRY_TRACKING_URI)
