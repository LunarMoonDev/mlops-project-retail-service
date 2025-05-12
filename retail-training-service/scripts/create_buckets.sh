#!/bin/bash

export AWS_PROFILE=minio

echo "Creating S3 Buckets..."
aws --endpoint-url=http://localhost:9000 s3 mb s3://mlflow
aws --endpoint-url=http://localhost:9000 s3 mb s3://data
