# Deployment Guide - EKS + MLflow + FastAPI + Streamlit

## Complete Setup Flow

### Step 1: Verify GitHub Secrets

Ensure these secrets exist in GitHub → Settings → Secrets → Actions:

```
AWS_ROLE_ARN=arn:aws:iam::709055505508:role/GitHubActionsMLOpsRole
AWS_REGION=us-east-1
S3_BUCKET=mlops-churn-prediction-709055505508
SAGEMAKER_ROLE_ARN=arn:aws:iam::709055505508:role/SageMakerMLOpsExecutionRole
```

### Step 2: Workflow Execution Order

**Workflow 1: Deploy SageMaker Pipeline** (`.github/workflows/deploy-pipeline.yml`)
- Triggers: Push to main branch
- Actions:
  1. Authenticates via OIDC
  2. Uploads pipeline scripts to S3
  3. Creates/updates SageMaker pipeline
  4. Triggers pipeline execution
  5. Pipeline trains model and deploys to endpoint

**Workflow 2: Deploy to EKS** (`.github/workflows/deploy-eks.yml`)
- Triggers: After SageMaker pipeline completes OR manual trigger
- Actions:
  1. Creates EKS cluster (if not exists)
  2. Builds Docker images:
     - MLflow (MySQL + S3 backend)
     - FastAPI (SageMaker endpoint client)
     - Streamlit (Web UI)
  3. Pushes images to ECR
  4. Deploys to EKS:
     - MySQL database for MLflow
     - MLflow tracking server
     - FastAPI backend
     - Streamlit frontend

### Step 3: Verify Deployment

After workflows complete:

```bash
# Configure kubectl
aws eks update-kubeconfig --name mlops-churn-cluster --region us-east-1

# Check all pods
kubectl get pods

# Check services (get LoadBalancer URLs)
kubectl get svc

# Check MLflow logs
kubectl logs -l app=mlflow

# Check FastAPI logs
kubectl logs -l app=fastapi
```

### Step 4: Access Services

1. **MLflow UI**: 
   - Get URL: `kubectl get svc mlflow -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'`
   - Access: `http://<loadbalancer-url>`

2. **FastAPI**:
   - Get URL: `kubectl get svc fastapi -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'`
   - Health: `http://<loadbalancer-url>/health`
   - Predict: `POST http://<loadbalancer-url>/predict`

3. **Streamlit**:
   - Get URL: `kubectl get svc streamlit -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'`
   - Access: `http://<loadbalancer-url>`

## MLflow Configuration

- **Backend Store**: MySQL database (in-cluster)
- **Artifact Store**: S3 bucket (`s3://mlops-churn-prediction-709055505508/mlflow/artifacts`)
- **Tracking URI**: `http://mlflow.default.svc.cluster.local:5000` (internal)
- **External Access**: Via LoadBalancer service

## Model Evaluation → MLflow

The evaluation script (`pipelines/evaluate.py`) automatically logs to MLflow:
- Metrics: accuracy, precision, recall, F1, AUC-ROC
- Artifacts: confusion matrix, evaluation report
- Experiment: `churn-prediction-evaluation`

## Troubleshooting

### EKS Cluster Creation Fails
- Check IAM permissions for eksctl
- Verify region is correct
- Check CloudFormation stack for errors

### MLflow Cannot Connect to MySQL
- Wait for MySQL pod to be ready: `kubectl wait --for=condition=ready pod -l app=mysql`
- Check MySQL logs: `kubectl logs -l app=mysql`

### FastAPI Cannot Call SageMaker
- Verify SageMaker endpoint exists: `aws sagemaker list-endpoints`
- Check IAM role for FastAPI service account
- Verify endpoint name in deployment: `churn-prediction-endpoint`

### Images Not Found
- Verify ECR repositories exist
- Check image tags match deployment YAML
- Verify ECR login in workflow

## Next Steps

1. Monitor MLflow experiments
2. Test predictions via Streamlit UI
3. Set up drift detection (Phase 7)
4. Configure auto-scaling for EKS
