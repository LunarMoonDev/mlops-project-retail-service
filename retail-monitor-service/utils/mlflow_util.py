import mlflow

from config import conf


def configure_mlflow_setup() -> None:
    """configure mlflow settings like host setup"""

    mlflow.set_tracking_uri(conf.REGISTRY_TRACKING_URI)
