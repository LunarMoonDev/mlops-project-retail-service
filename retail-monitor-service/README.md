### Description:
- this is my attempt at developing a monitoring service for the batch service that calculates and monitors data drift

### Requirements:
- this must execute after the batch service
    - this means batch service must predict the current batch service
    - predictions for current date must exist in the database
- training data must exist in s3 bucket
- there must be a latest tagged model
- minio profile must exist in AWS profile and credentials

### How to run:
- run `docker compose up` to run the prefect, mlflow, and grafana
    - this will also run a prefect application similar to retail-predict-service
    - there's no `setup_local` in `Makefile` so simply run the quick run to run the app