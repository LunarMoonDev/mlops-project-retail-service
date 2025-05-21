### Description
- this project focuses on training models with the objective of predictive review ratings from fashion retail sales dataset
- this is my attempt at tackling the final project task given by DataTalk' club from their MLOps zoomcamp
- models trained are linear, lasso, ridge regression and xgboost (you may look into models module for thier implementation)

### Requirements:
- make sure you have AWS_PROFILE for minio, if not create one in ~/.aws for credentials and config
- docker from parent directory is running (make sure postgres and minio are active)
- make sure you have `.local` directory containing the csv you wish to train the model upon
    - name this csv `Fashion_Retail_Sales.csv`
    - for details on how to acquire this csv, please visit [Retail Dataset](https://www.kaggle.com/datasets/atharvasoundankar/fashion-retail-sales/code)

### How to run:
- run `make setup_local` to create buckets and push data in s3 bucket running in minio
- run `make run_local` to run the prefect application and train the models
