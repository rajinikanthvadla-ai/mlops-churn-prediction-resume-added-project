# End-to-End MLOps Pipeline — Customer Churn Prediction

## Overview

Production-grade MLOps pipeline using:

- **AWS SageMaker Pipelines** — Automated ML training and deployment
- **GitHub Actions CI/CD** — Automated pipeline triggers
- **EKS Cluster** — Kubernetes orchestration
- **MLflow** — Experiment tracking (MySQL backend + S3 artifacts)
- **FastAPI** — REST API for predictions
- **Streamlit** — Web UI for predictions
- **Evidently AI** — Drift detection (future)

## Architecture

```
GitHub Push
    ↓
GitHub Actions (OIDC Auth)
    ↓
SageMaker Pipeline:
  → Preprocess Data
  → Train XGBoost Model
  → Evaluate Model
  → Register Model
  → Deploy Endpoint
    ↓
EKS Cluster:
  → MLflow (MySQL + S3)
  → FastAPI (SageMaker Endpoint)
  → Streamlit UI
```

## Prerequisites

1. AWS Account with appropriate permissions
2. GitHub repository
3. AWS CLI configured
4. kubectl and eksctl installed (for local testing)

## Setup Instructions

### 1. AWS Prerequisites (One-time setup)

Follow Phase 1 from the original guide:
- IAM User with programmatic access
- SageMaker Execution Role
- GitHub Actions IAM Role (OIDC)
- S3 Bucket with data uploaded
- ECR Repositories

### 2. GitHub Secrets

Add these secrets in GitHub → Settings → Secrets → Actions:

- `AWS_ROLE_ARN` — GitHub Actions IAM role ARN
- `AWS_REGION` — us-east-1
- `S3_BUCKET` — Your S3 bucket name
- `SAGEMAKER_ROLE_ARN` — SageMaker execution role ARN

### 3. Workflow Execution

**Pipeline 1: Deploy SageMaker Pipeline**
- Triggers on: Push to main
- Creates SageMaker pipeline
- Trains and deploys model

**Pipeline 2: Deploy to EKS**
- Triggers after: SageMaker pipeline completes
- Creates EKS cluster
- Builds and pushes Docker images
- Deploys all services

## Services

### MLflow Tracking Server
- **Backend**: MySQL (in-cluster)
- **Artifacts**: S3
- **Access**: LoadBalancer service
- **URL**: Check `kubectl get svc mlflow`

### FastAPI Backend
- **Endpoint**: Calls SageMaker endpoint
- **Access**: LoadBalancer service
- **Health**: `/health`
- **Predict**: `POST /predict`

### Streamlit Frontend
- **Access**: LoadBalancer service
- **Features**: Customer churn prediction form

## Model Deployment Flow

1. **Training**: SageMaker Pipeline trains XGBoost model
2. **Evaluation**: Metrics logged to MLflow
3. **Registration**: Model registered in SageMaker Model Registry
4. **Deployment**: Model deployed to SageMaker endpoint
5. **Inference**: FastAPI calls endpoint, Streamlit provides UI

## Monitoring

- **MLflow UI**: View experiments, metrics, artifacts
- **SageMaker**: Model registry, endpoint metrics
- **CloudWatch**: EKS cluster logs

## Cost Estimation

| Resource | Estimated Cost/Hour |
|----------|-------------------|
| EKS Cluster (2x t3.medium) | $0.10 |
| SageMaker Endpoint (ml.m5.large) | $0.13 |
| MLflow (t3.medium) | $0.05 |
| **Total** | **~$0.28/hour** |

## Cleanup

Run cleanup script to remove all resources:

```bash
./scripts/cleanup.sh
```

## Author

MLOps Lab Project
