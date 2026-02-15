# AWS SageMaker Quota Issue - Fix Guide
# Author: Rajinikanth Vadla

## Problem
Your AWS account doesn't have quota for `ml.m5.large` processing instances.

Error: "The account-level service limit 'ml.m5.large for processing job usage' is 0 Instances"

## Solution 1: Use Smaller Instance Types (Already Applied)
I've changed the default processing instance type from `ml.m5.large` to `ml.t3.medium` which is:
- More commonly available by default
- Lower cost
- Sufficient for preprocessing tasks

## Solution 2: Request Quota Increase (If Needed)
If you need `ml.m5.large` or larger instances:

1. Go to AWS Service Quotas Console:
   - https://console.aws.amazon.com/servicequotas/
   - Or: AWS Console → Service Quotas

2. Search for "SageMaker"
   - Find: "ml.m5.large for processing job usage"
   - Click "Request quota increase"

3. Request Details:
   - Desired quota value: 10 (or more)
   - Use case: "MLOps pipeline for customer churn prediction"
   - Submit request

4. Wait for approval (usually 24-48 hours)

## Solution 3: Use Different Instance Types
You can override instance types when running the pipeline:

In SageMaker Studio or via API:
- ProcessingInstanceType: `ml.t3.medium` (current default)
- TrainingInstanceType: `ml.m5.xlarge` or `ml.t3.medium`

## Current Configuration
- **Processing**: `ml.t3.medium` (changed from ml.m5.large)
- **Training**: `ml.m5.xlarge` (can be changed if quota issues)
- **Inference**: `ml.t3.medium`, `ml.m5.large`, `ml.m5.xlarge` (multiple options)

## Verify Quota
Check your current quotas:
```bash
aws service-quotas get-service-quota \
  --service-code sagemaker \
  --quota-code L-4EA4796A
```

## Note
The pipeline will now use `ml.t3.medium` for processing, which should work without quota increases.
