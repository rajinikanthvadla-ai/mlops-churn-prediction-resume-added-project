# scripts/upload_pipeline_scripts.py
# Author: Rajinikanth Vadla
# Description: Uploads pipeline scripts (preprocess, train, evaluate) to S3 for SageMaker processing

import os
import boto3
import sys

def upload_scripts(bucket_name, region="us-east-1"):
    """
    Upload pipeline scripts to S3 so SageMaker can access them.
    """
    s3_client = boto3.client("s3", region_name=region)
    
    scripts_to_upload = [
        ("pipelines/preprocess.py", "pipelines/preprocess.py"),
        ("pipelines/train.py", "pipelines/train.py"),
        ("pipelines/evaluate.py", "pipelines/evaluate.py")
    ]
    
    for local_path, s3_key in scripts_to_upload:
        if not os.path.exists(local_path):
            print(f"Warning: {local_path} not found, skipping...")
            continue
        
        print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
        s3_client.upload_file(local_path, bucket_name, s3_key)
        print(f"  ✓ Uploaded successfully")
    
    print("All scripts uploaded!")

if __name__ == "__main__":
    bucket_name = os.environ.get("S3_BUCKET")
    region = os.environ.get("AWS_REGION", "us-east-1")
    
    if len(sys.argv) > 1:
        bucket_name = sys.argv[1]
    
    if not bucket_name:
        print("Error: S3_BUCKET not set")
        print("Usage: python scripts/upload_pipeline_scripts.py <bucket_name>")
        sys.exit(1)
    
    upload_scripts(bucket_name, region)
