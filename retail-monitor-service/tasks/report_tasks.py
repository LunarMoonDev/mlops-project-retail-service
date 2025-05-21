from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from pandas import DataFrame
from prefect import get_run_logger, task

column_mapping = ColumnMapping(
    prediction='predictions',
    numerical_features=['PurchaseAmount', 'predictions'],
    categorical_features=['ItemPurchased'],
    target=None,
)

report = Report(
    metrics=[
        ColumnDriftMetric(column_name='predictions'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ]
)


@task
def calculate_metrics(ref_data: DataFrame, curr_data: DataFrame) -> tuple:
    """Calculate the metrics for reference and curr_data

    Args:
        ref_dat (DataFrame): training data used for the model
        curr_data (DataFrame): curr data predicted today

    Returns:
        tuple: tuple containing prediction drift, num of drifted features, missing values
    """
    logger = get_run_logger()

    logger.info('Calculating report for drifts and missing values...')
    report.run(
        reference_data=ref_data, current_data=curr_data, column_mapping=column_mapping
    )
    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current'][
        'share_of_missing_values'
    ]

    return prediction_drift.item(), num_drifted_columns, share_missing_values
