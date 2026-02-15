# scripts/trigger_pipeline.py
# Author: Rajinikanth Vadla
# Description: Triggers SageMaker Pipeline execution and waits for completion

import boto3
import os
import time

REGION = os.environ["AWS_REGION"]
PIPELINE_NAME = "churn-prediction-pipeline"

client = boto3.client("sagemaker", region_name=REGION)

def trigger_pipeline():
    response = client.start_pipeline_execution(
        PipelineName=PIPELINE_NAME
    )

    execution_arn = response["PipelineExecutionArn"]

    print(f"Pipeline started: {execution_arn}")

    return execution_arn

def wait_for_completion(execution_arn):

    while True:

        response = client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )

        status = response["PipelineExecutionStatus"]

        print(f"Current Status: {status}")

        if status in ["Succeeded", "Failed", "Stopped"]:
            break

        time.sleep(30)

    print(f"Final Status: {status}")

if __name__ == "__main__":

    execution_arn = trigger_pipeline()

    wait_for_completion(execution_arn)
