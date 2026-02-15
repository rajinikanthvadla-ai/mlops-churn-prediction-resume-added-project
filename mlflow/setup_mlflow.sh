#!/bin/sh
# mlflow/setup_mlflow.sh

export MLFLOW_BACKEND_STORE_URI="mysql+pymysql://mlflow:mlflow_password@mysql:3306/mlflow"
export MLFLOW_DEFAULT_ARTIFACT_ROOT="s3://${S3_BUCKET}/mlflow/artifacts"

mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
  --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT}
