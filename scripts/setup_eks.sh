#!/bin/bash
# scripts/setup_eks.sh
# Author: Rajinikanth Vadla
# Description: Creates EKS cluster for MLOps deployment

set -e

CLUSTER_NAME=${CLUSTER_NAME:-"mlops-churn-cluster"}
REGION=${AWS_REGION:-"us-east-1"}
NODE_TYPE=${NODE_TYPE:-"t3.medium"}
NODE_COUNT=${NODE_COUNT:-2}

echo "Creating EKS cluster: ${CLUSTER_NAME}"

# Create EKS cluster
eksctl create cluster \
  --name ${CLUSTER_NAME} \
  --region ${REGION} \
  --node-type ${NODE_TYPE} \
  --nodes ${NODE_COUNT} \
  --nodes-min 2 \
  --nodes-max 3 \
  --managed \
  --with-oidc \
  --full-ecr-access \
  --alb-ingress-access

echo "Waiting for cluster to be ready..."
eksctl utils wait-cluster --name ${CLUSTER_NAME} --region ${REGION}

echo "Configuring kubectl..."
aws eks update-kubeconfig --name ${CLUSTER_NAME} --region ${REGION}

echo "Verifying cluster..."
kubectl get nodes

echo "EKS cluster ${CLUSTER_NAME} is ready!"
