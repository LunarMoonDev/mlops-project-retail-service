import psycopg2 as pos

from psycopg2.extras import execute_batch

from pandas import DataFrame
from prefect import get_run_logger, task

from config import conf


@task
def save_predictions(pred_df: DataFrame) -> None:
    """inserts prediction in database

    Args:
        pred_df (DataFrame): dataframe with predictions
    """
    logger = get_run_logger()

    logger.info('Connecting to Postgres DB [%s]...', conf.DB_NAME)
    conn = pos.connect(
        database=conf.DB_NAME,
        host=conf.DB_HOST,
        user=conf.DB_USER,
        password=conf.DB_PASSWORD,
        port=conf.DB_PORT,
    )

    try:
        with conn.cursor() as cur:
            query = """INSERT INTO
                            retail_reviews (
                                order_id,
                                customer_id,
                                item_purchased,
                                purchase_amount,
                                date_purchase,
                                review_rating,
                                payment_method
                            )
                        VALUES
                            (%s, %s, %s, %s, %s, %s, %s)"""

            cols = [
                'OrderId',
                'CustomerId',
                'ItemPurchased',
                'PurchaseAmount',
                'DatePurchase',
                'predictions',
                'PaymentMethod',
            ]
            data = list(zip(*[pred_df[col] for col in cols]))

            logger.info('Inserting into table...')
            execute_batch(cur, query, data)
        conn.commit()
    finally:
        if conn:
            conn.close()
