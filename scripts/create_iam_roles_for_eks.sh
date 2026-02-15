#!/bin/bash
# scripts/create_iam_roles_for_eks.sh

set -e

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-"us-east-1"}
CLUSTER_NAME=${CLUSTER_NAME:-"mlops-churn-cluster"}
S3_BUCKET=${S3_BUCKET:-"mlops-churn-prediction-709055505508"}

echo "Creating IAM roles for EKS service accounts..."

# Create MLflow IAM Role
cat > /tmp/mlflow-trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/oidc.eks.${REGION}.amazonaws.com/id/$(aws eks describe-cluster --name ${CLUSTER_NAME} --query 'cluster.identity.oidc.issuer' --output text | cut -d '/' -f 5)"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.${REGION}.amazonaws.com/id/$(aws eks describe-cluster --name ${CLUSTER_NAME} --query 'cluster.identity.oidc.issuer' --output text | cut -d '/' -f 5):sub": "system:serviceaccount:default:mlflow-sa"
        }
      }
    }
  ]
}
EOF

aws iam create-role \
  --role-name MLflowEKSRole \
  --assume-role-policy-document file:///tmp/mlflow-trust-policy.json || echo "Role exists"

aws iam attach-role-policy \
  --role-name MLflowEKSRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess || echo "Policy attached"

# Create FastAPI IAM Role
cat > /tmp/fastapi-trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/oidc.eks.${REGION}.amazonaws.com/id/$(aws eks describe-cluster --name ${CLUSTER_NAME} --query 'cluster.identity.oidc.issuer' --output text | cut -d '/' -f 5)"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.${REGION}.amazonaws.com/id/$(aws eks describe-cluster --name ${CLUSTER_NAME} --query 'cluster.identity.oidc.issuer' --output text | cut -d '/' -f 5):sub": "system:serviceaccount:default:fastapi-sa"
        }
      }
    }
  ]
}
EOF

aws iam create-role \
  --role-name FastAPIEKSRole \
  --assume-role-policy-document file:///tmp/fastapi-trust-policy.json || echo "Role exists"

aws iam attach-role-policy \
  --role-name FastAPIEKSRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess || echo "Policy attached"

echo "IAM roles created successfully!"
