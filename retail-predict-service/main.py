import logging

from prefect import flow

from tasks.data_tasks import clean_data, grab_features, retrieve_data, combine_data
from tasks.db_tasks import save_predictions
from tasks.model_tasks import execute_model, retrieve_model
from utils.mlflow_util import configure_mlflow_setup

logging.getLogger('mlflow').setLevel(logging.ERROR)


@flow
def main():
    """main flow of this prefect application"""
    # feature engineering tasks
    data = retrieve_data()
    data = clean_data(data)
    feature_df = grab_features(data)

    # model tasks
    model_pipe = retrieve_model()
    pred_df = execute_model(feature_df, model_pipe)
    combined_df = combine_data(data, pred_df)
    
    # output tasks
    save_predictions(combined_df)

if __name__ == "__main__":
    # configure mlflow connection
    configure_mlflow_setup()

    # run the main prefect application
    main.serve(
        name = "retail_predict_service_scheduled",
        cron = "0 18 * * *"
    )
