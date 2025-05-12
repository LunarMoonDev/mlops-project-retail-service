from prefect import flow
from sklearn.preprocessing import PowerTransformer

from config import conf
from models import LassoTrain, LinearTrain, RidgeTrain, XGBoostTrain
from tasks.data_tasks import (
    clean_data,
    drop_na_features,
    filter_features,
    retrieve_data,
    split_data,
)
from tasks.mlflow_tasks import set_experiment_name
from tasks.model_tasks import execute_feature_pipeline, execute_model
from utils.lookup_util import ImputateTransformer
from utils.scaler_util import ColumnTransformerWrapper


# main flow
@flow()
def main():
    """main flow of this training orchestration"""
    # task 1 retrieve data from s3
    data = retrieve_data()

    # task 2 clean data of raw
    data = clean_data(data)

    # task 3 filter features of cleaned data if toggled
    data = filter_features(data)

    # task 4 drop na on feature of cleaned data if toggled
    data = drop_na_features(data)

    # task 5 split filtered data to train and valid
    x_train, x_valid, y_train, y_valid = split_data(data)

    # task 6 fit train and valid to feature pipelines
    imputater = ImputateTransformer(
        by=conf.DATA_IMPUTATE_CATEGORY,
        columns=conf.DATA_IMPUTATE_FEATURES,
        strats=conf.DATA_IMPUTATE_STRAT,
    )
    scaler = ColumnTransformerWrapper(
        transformer=PowerTransformer(method="yeo-johnson")
    )

    pipelines = [("imputate_scale", [imputater, scaler]), ("imputate", [imputater])]

    feature_pipelines = []
    for tag, pipeline in pipelines:
        feature_pipe, train_data, valid_data = execute_feature_pipeline(
            pipeline=pipeline, x_train=x_train, x_valid=x_valid, pipe_tag=tag
        )

        model_data = {
            "X_train": x_train,
            "X_valid": x_valid,
            "y_train": y_train,
            "y_valid": y_valid,
            "train_data": train_data,
            "valid_data": valid_data,
        }

        feature_pipelines.append((tag, feature_pipe, model_data))

    # task 6 setup mlflow experiment
    _ = set_experiment_name()

    # task 7 execute model trainer on each feature pipe with their data
    trainers = [LinearTrain, RidgeTrain, LassoTrain, XGBoostTrain]
    for tag, feature_pipe, model_data in feature_pipelines:
        for trainer in trainers:
            execute_model(trainer, model_data, feature_pipe, tag)


if __name__ == "__main__":
    main()
