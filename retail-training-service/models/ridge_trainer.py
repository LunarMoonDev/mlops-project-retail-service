import mlflow
from pandas import DataFrame
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from utils.feature_util import make_model_pipeline
from utils.mlflow_util import mlflow_default_logging


class RidgeTrain:
    """Trainer for Ridge model"""

    def __init__(
        self, model_data: dict[str, DataFrame], feature_pipe: Pipeline, feature_tag: str
    ):
        self.model_data = model_data
        self.feature_pipe = feature_pipe
        self.feature_tag = feature_tag

    def run(self, logger) -> Pipeline:
        """trains a ridge model and creates a model pipeline
            logs the model pipeline in mlflow
                - including other meta=data

        Args:
            logger (_type_): logger from prefect

        Returns:
            Pipeline: model pipeline
        """

        with mlflow.start_run():
            trainer_label = "RidgeTrainer"

            model = Ridge(alpha=10)

            # pylint: disable=duplicate-code
            logger.info("Fitting model with training data...")
            model.fit(X=self.model_data["train_data"], y=self.model_data["y_train"])

            logger.info(
                "Creating model pipeline with feature pipe [%s]...", self.feature_tag
            )
            model_pipe = make_model_pipeline(self.feature_pipe, model)
            mlflow.sklearn.log_model(model_pipe, "model")

            logger.info("Logging model pipeline in mlflow...")
            mlflow_default_logging(
                model=model_pipe,
                model_tag=trainer_label,
                pipeline_tag=self.feature_tag,
                x_train=self.model_data["X_train"],
                x_valid=self.model_data["X_valid"],
                y_train=self.model_data["y_train"],
                y_valid=self.model_data["y_valid"],
            )
            # pylint: enable=duplicate-code

        return model_pipe
