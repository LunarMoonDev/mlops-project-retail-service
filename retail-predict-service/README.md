### Description:
- this project focuses on running a batch service with the latest tagged model that runs everyday at 6:00 PM UTC
- This is attempt at using the trained model on production

### Requirements:
- make sure there are trained models in the model registry (run from retail-training-service)
- docker form parent directory is running (make sure postgres and minio are active)
- make sure you have minio aws credential and config
- make sure you have `.local` that contains the `Fashion_Retail_Sales.csv`
    - this can be similar to the training data but you may create a new one with random values
    - i also have aws credentials here which i reference in docker compose volumes

### How to run:
- run docker compose and it will execute containers for prefect and mlflow
    - after both prefect and mlflow are active, batch service then runs and waits for scheduled jobs
- while waiting for scheduled jobs, you may run `make setup_local` to push the batch data for today and tag a model as latest
- you may then perform a quick run on the deployed prefect app in prefect interface