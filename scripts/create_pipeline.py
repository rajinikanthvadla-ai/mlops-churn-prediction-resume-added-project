# scripts/create_pipeline.py
# Author: Rajinikanth Vadla
# Description: Creates or updates SageMaker Pipeline for churn prediction

import os
import sys

# Add project root to Python path so we can import pipelines module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from pipelines.pipeline_definition import get_pipeline

def main():
    # Get values from environment or command line
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")
    bucket_name = os.environ.get("S3_BUCKET")
    region = os.environ.get("AWS_REGION", "us-east-1")
    
    if len(sys.argv) > 1:
        role_arn = sys.argv[1]
    if len(sys.argv) > 2:
        bucket_name = sys.argv[2]
    
    if not role_arn:
        print("Error: SAGEMAKER_ROLE_ARN not set")
        print("Usage: python scripts/create_pipeline.py <role_arn> <bucket_name>")
        sys.exit(1)
    
    if not bucket_name:
        print("Error: S3_BUCKET not set")
        print("Usage: python scripts/create_pipeline.py <role_arn> <bucket_name>")
        sys.exit(1)
    
    print(f"Creating pipeline with:")
    print(f"  Role ARN: {role_arn}")
    print(f"  Bucket: {bucket_name}")
    print(f"  Region: {region}")
    
    # Create pipeline
    pipeline = get_pipeline(
        role_arn=role_arn,
        bucket_name=bucket_name,
        region=region
    )
    
    # Upsert pipeline (create or update)
    pipeline.upsert(role_arn=role_arn)
    
    print(f"Pipeline 'churn-prediction-pipeline' created/updated successfully!")
    print(f"Pipeline ARN: {pipeline.arn}")

if __name__ == "__main__":
    main()
