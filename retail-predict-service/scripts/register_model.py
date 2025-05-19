import logging
import mlflow

from mlflow.tracking import MlflowClient

# variables
TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "retail_filter0_drop0_experiment"
MODEL_NAME = "retail_review_prediction"
ALIAS = "champion"

# config for logs
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

# config for mlflow
mlflow.set_tracking_uri(TRACKING_URI)

# grab mlflow client
client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

# search for runs
logging.info("Searching for runs in %s", EXPERIMENT_NAME)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=['metrics.rmse_valid DESC'],
    max_results=1
)

# grab run from filter, grab run_id
best_run = runs[0]
run_id = best_run.info.run_id
logging.info("Best runs found %s %f", run_id, best_run.data.metrics.get('rmse_valid'))

# register run_id in model_name (if existing register with new version)
logging.info("Registering %s", run_id)
MODEL_URI = f'runs:/{run_id}/model'
result = mlflow.register_model(MODEL_URI, MODEL_NAME)

# sets the alias for this version
logging.info("Tagging %s as %s", run_id, ALIAS)
client.set_registered_model_alias(name=MODEL_NAME, 
                                  alias=ALIAS, 
                                  version=result.version)