#!/bin/bash

# constants
SCRIPT_DIR=$(dirname "$(realpath "$0")")
LOCAL_CSV="$SCRIPT_DIR/../.local/Fashion_Retail_Sales.csv"
S3_BUCKET="s3://data/train/2025-18-05_batch_data.csv"

# specify the aws profile to use
export AWS_PROFILE=minio

echo "Uploading csv to aws s3..."
aws s3 cp $LOCAL_CSV $S3_BUCKET