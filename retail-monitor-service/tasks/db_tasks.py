from datetime import date

import psycopg2 as pos
from pandas import DataFrame
from prefect import get_run_logger, task

from config import conf

config = {
    'database': conf.DB_NAME,
    'host': conf.DB_HOST,
    'user': conf.DB_USER,
    'password': conf.DB_PASSWORD,
    'port': conf.DB_PORT,
}


@task
def read_current_prediction() -> DataFrame:
    """Reads the prediction from retail_reviews table for current date

    Returns:
        DataFrame: dataframe of predictions
    """
    logger = get_run_logger()

    logger.info('Connecting to Postgres DB [%s]...', conf.DB_NAME)
    query = """SELECT
            item_purchased, 
            purchase_amount, 
            date_purchase, 
            review_rating, 
            payment_method, 
            created_at, 
            modified_at 
        FROM retail_reviews
        WHERE created_at::date = %s"""

    with pos.connect(**config) as conn, conn.cursor() as curr:
        curr_date = date.today()
        curr.execute(query, (curr_date,))
        rows = curr.fetchall()

    cols = [
        'ItemPurchased',
        'PurchaseAmount',
        'DatePurchase',
        'predictions',
        'PaymentMethod',
        'CreatedAt',
        'ModifiedAt',
    ]
    return DataFrame(rows, columns=cols)


@task
def save_metrics(
    prediction_drift: float, num_drifted_columns: float, share_missing_values: float
) -> None:
    """Saves the given metrics in model_metric table

    Args:
        prediction_drift (float): drift in prediction column
        num_drifted_columns (float): num of drifted features
        share_missing_values (float): percentage of missing values among rows
    """
    logger = get_run_logger()

    logger.info('Connecting to Postgres DB [%s]...', conf.DB_NAME)
    query = """
        INSERT into
            model_metrics (
                prediction_drift,
                num_drifted_columns,
                share_missing_values
            )
        values
            (%s, %s, %s)
    """

    logger.info('Connecting to database for inserting metrics...')
    with pos.connect(**config) as conn, conn.cursor() as curr:
        curr.execute(
            query, (prediction_drift, num_drifted_columns, share_missing_values)
        )
