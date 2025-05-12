import time

import mlflow
from sklearn.metrics import root_mean_squared_error

from utils.df_util import plot_duration_historgram


def mlflow_default_logging(
    *, model, model_tag, pipeline_tag, x_train, y_train, x_valid, y_valid
) -> dict:
    """logs given metrics, model, and data in mlflow server

    Args:
        model (_type_): model pipeline
        model_tag (_type_): label for model
        pipeline_tag (_type_): label for feature pipeline
        x_train (_type_): x training set
        y_train (_type_): y training set
        x_valid (_type_): x eval set
        y_valid (_type_): y eval set

    Returns:
        dict: rmse values for train and eval
    """

    start_time = time.time()
    yp_train = model.predict(x_train)
    yp_valid = model.predict(x_valid)
    elapsed = time.time() - start_time

    total_length = len(yp_train) + len(yp_valid)
    predict_time = elapsed / total_length

    # Metrics
    rmse_train = root_mean_squared_error(y_true=y_train, y_pred=yp_train)
    rmse_valid = root_mean_squared_error(y_true=y_valid, y_pred=yp_valid)

    fig = plot_duration_historgram(y_train, yp_train, y_valid, yp_valid)

    # logging
    mlflow.set_tag("model", model_tag)
    mlflow.set_tag("pipeline", pipeline_tag)
    # grab this from config
    # mlflow.log_param("train_data_path", data["train_data_path"])
    # mlflow.log_param("valid_data_path", data["valid_data_path"])

    mlflow.log_metric("rmse_train", rmse_train)
    mlflow.log_metric("rmse_valid", rmse_valid)
    mlflow.log_metric("predict_time", predict_time)

    mlflow.log_figure(fig, "plot.svg")

    return {"rmse_train": rmse_train, "rmse_valid": rmse_valid}
