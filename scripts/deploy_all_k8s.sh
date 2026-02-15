#!/bin/bash
# scripts/deploy_all_k8s.sh

set -e

CLUSTER_NAME=${CLUSTER_NAME:-"mlops-churn-cluster"}
REGION=${AWS_REGION:-"us-east-1"}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "Configuring kubectl..."
aws eks update-kubeconfig --name ${CLUSTER_NAME} --region ${REGION}

echo "Deploying MySQL for MLflow..."
kubectl apply -f mlflow/k8s/mysql-deployment.yaml

echo "Waiting for MySQL to be ready..."
kubectl wait --for=condition=ready pod -l app=mysql --timeout=300s

echo "Creating IAM roles for EKS service accounts..."
chmod +x scripts/create_iam_roles_for_eks.sh
./scripts/create_iam_roles_for_eks.sh || echo "IAM roles may already exist"

echo "Deploying MLflow..."
export MLFLOW_IMAGE="${ECR_REGISTRY}/mlflow:latest"
export MLFLOW_IAM_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/MLflowEKSRole"
kubectl apply -f mlflow/k8s/configmap.yaml
envsubst < mlflow/k8s/deployment.yaml | kubectl apply -f -

echo "Waiting for MLflow to be ready..."
kubectl wait --for=condition=ready pod -l app=mlflow --timeout=300s

echo "Deploying FastAPI..."
export FASTAPI_IMAGE="${ECR_REGISTRY}/fastapi:latest"
export FASTAPI_IAM_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/FastAPIEKSRole"
envsubst < inference/k8s/deployment.yaml | kubectl apply -f -

echo "Deploying Streamlit..."
export STREAMLIT_IMAGE="${ECR_REGISTRY}/streamlit:latest"
envsubst < frontend/k8s/deployment.yaml | kubectl apply -f -

echo "Waiting for services..."
sleep 30

echo "Getting service URLs..."
kubectl get svc

echo "Deployment complete!"
