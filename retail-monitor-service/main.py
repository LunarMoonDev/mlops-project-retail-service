from prefect import flow

from tasks.data_tasks import clean_data, grab_features, retrieve_reference_data
from tasks.db_tasks import read_current_prediction, save_metrics
from tasks.model_tasks import execute_model, retrieve_model
from tasks.report_tasks import calculate_metrics
from utils.mlflow_util import configure_mlflow_setup


@flow
def main():
    """flow of monitoring service"""
    ref_data = retrieve_reference_data()
    ref_data = clean_data(ref_data)
    ref_data = grab_features(ref_data)

    model = retrieve_model()
    ref_data = execute_model(ref_data, model)

    curr_data = read_current_prediction()
    curr_data = grab_features(curr_data, include_pred=True)

    pred_drift, num_drifts, missing_values = calculate_metrics(ref_data, curr_data)
    save_metrics(pred_drift, num_drifts, missing_values)


if __name__ == '__main__':
    # configure mlflow
    configure_mlflow_setup()
    
    main.serve(
        name = "retail_monitor_service_scheduled",
        cron = "0 19 * * *"
    )
