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
    
    pipeline_name = pipeline.name
    print(f"Pipeline '{pipeline_name}' created/updated successfully!")
    
    # Get pipeline ARN after upsert
    # Author: Rajinikanth Vadla
    # Note: Pipeline object doesn't have .arn attribute, so we query it from AWS
    try:
        sagemaker_client = boto3.client("sagemaker", region_name=region)
        pipeline_response = sagemaker_client.describe_pipeline(
            PipelineName=pipeline_name
        )
        pipeline_arn = pipeline_response["PipelineArn"]
        print(f"Pipeline ARN: {pipeline_arn}")
    except Exception as e:
        print(f"Pipeline created/updated successfully!")
        print(f"Pipeline name: {pipeline_name}")
        print(f"Note: Could not retrieve ARN (this is non-critical): {e}")

if __name__ == "__main__":
    main()
