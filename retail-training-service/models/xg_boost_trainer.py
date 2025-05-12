from functools import partial

import mlflow
import optuna
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from config import conf
from utils.feature_util import make_model_pipeline
from utils.mlflow_util import mlflow_default_logging


def objective(
    trial,
    model_data: dict[str, DataFrame],
    feature_pipe: Pipeline,
    feature_tag: str,
    logger,
) -> float:
    """objective function for optuna case study

    Args:
        trial (_type_): number of trials we are currently at
        model_data (dict[str, DataFrame]): data for model
        logger (_type_): logger from prefect

    Returns:
        float: scaler loss value from training model
    """

    params = {
        "max_depth": trial.suggest_int(
            "max_depth", conf.XG_MAX_DEPTH_MIN, conf.XG_MAX_DEPTH_MAX
        ),
        "n_estimators": trial.suggest_int(
            "n_estimators",
            conf.XG_N_ESTIMATORS_MIN,
            conf.XG_N_ESTIMATORS_MAX,
            step=conf.XG_N_ESTIMATORS_STEP,
        )
        * 100,
        "learning_rate": trial.suggest_float(
            "learning_rate",
            conf.XG_N_LEARNING_RATE_MIN,
            conf.XG_N_LEARNING_RATE_MAX,
            log=True,
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            conf.XG_CHILD_WEIGHT_MIN,
            conf.XG_CHILD_WEIGHT_MAX,
            log=True,
        ),
        "objective": "reg:squarederror",
        "seed": conf.RANDOM_STATE,
    }

    return execute_model(
        hyper_params=params,
        model_data=model_data,
        trainer_label=f"XGBoostTrainer_{trial.number}",
        feature_pipe=feature_pipe,
        feature_tag=feature_tag,
        logger=logger,
    )


def execute_model(
    *,
    hyper_params: dict,
    model_data: dict[str, DataFrame],
    trainer_label: str,
    feature_pipe: Pipeline,
    feature_tag: str,
    logger,
) -> float:
    """trains an xgboost model with given hyperparam

    Args:
        hyper_params (dict): hyper param from optuna
        model_data (dict[str, DataFrame]): data for model
        trainer_label (str): label of current trainer
        feature_pipe (Pipeline): feature pipeline
        feature_tag (str): tag of current feature pipe
        logger (_type_): logger from prefect

    Returns:
        float: rmse loss of valid data
    """
    with mlflow.start_run():
        logger.info("Current trainer: [%s] training model xgboost...", trainer_label)

        model = XGBRegressor(
            early_stopping_rounds=conf.XG_EARLY_STOPPING_ROUNDS,
            **hyper_params,
        )
        model.fit(
            model_data["train_data"],
            model_data["y_train"],
            eval_set=[(model_data["valid_data"], model_data["y_valid"])],
        )

        logger.info("Creating model pipeline for trainer [%s]...", trainer_label)
        model_pipe = make_model_pipeline(feature_pipe, model)

        logger.info("Saving into mlflow...")
        logs = mlflow_default_logging(
            model=model_pipe,
            model_tag=trainer_label,
            pipeline_tag=feature_tag,
            x_train=model_data["X_train"],
            x_valid=model_data["X_valid"],
            y_train=model_data["y_train"],
            y_valid=model_data["y_valid"],
        )

        mlflow.log_dict(hyper_params, "hyper_params.json")
        mlflow.sklearn.log_model(model_pipe, "model")

    return logs["rmse_valid"]


class XGBoostTrain:
    """Trainer for xgboost"""

    def __init__(
        self, model_data: dict[str, DataFrame], feature_pipe: Pipeline, feature_tag: str
    ):
        self.model_data = model_data
        self.feature_pipe = feature_pipe
        self.feature_tag = feature_tag

    def run(self, logger) -> None:
        """Trains multiple xgboost models under different hyperparameters

        Args:
            logger (_type_): logger from prefect
        """
        objective_partial = partial(
            objective,
            model_data=self.model_data,
            feature_pipe=self.feature_pipe,
            feature_tag=self.feature_tag,
            logger=logger,
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_partial, n_trials=conf.XG_N_TRIALS)
