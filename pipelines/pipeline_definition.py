# pipelines/pipeline_definition.py
# Author: Rajinikanth Vadla
# Description: Defines complete SageMaker Pipeline with preprocessing, training, evaluation, and model registration

import os
import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.inputs import TrainingInput
import json

def get_pipeline(
    role_arn,
    bucket_name,
    region="us-east-1",
    pipeline_name="churn-prediction-pipeline"
):
    """
    Creates a complete SageMaker Pipeline for churn prediction.
    
    Args:
        role_arn: SageMaker execution role ARN
        bucket_name: S3 bucket name
        region: AWS region
        pipeline_name: Pipeline name
    """
    
    sagemaker_session = sagemaker.Session()
    
    # Pipeline parameters
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.m5.large"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval"
    )
    accuracy_threshold = ParameterFloat(
        name="AccuracyThreshold",
        default_value=0.75
    )
    
    # Step 1: Data Preprocessing
    sklearn_processor = ScriptProcessor(
        role=role_arn,
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/sklearn-processing:1.0-1.cpu",
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="churn-preprocess",
        sagemaker_session=sagemaker_session
    )
    
    preprocess_step = ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{bucket_name}/data/raw",
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket_name}/data/processed/train"
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/output/validation",
                destination=f"s3://{bucket_name}/data/processed/validation"
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{bucket_name}/data/processed/test"
            ),
            ProcessingOutput(
                output_name="model",
                source="/opt/ml/processing/output/model",
                destination=f"s3://{bucket_name}/data/processed/model"
            )
        ],
        code="s3://{}/pipelines/preprocess.py".format(bucket_name)
    )
    
    # Step 2: Model Training
    # Author: Rajinikanth Vadla
    # Use SageMaker's built-in XGBoost algorithm (no custom script needed)
    # Get XGBoost image URI for the region
    xgboost_image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.7-1",
        py_version="py3"
    )
    
    # Use standard Estimator with XGBoost image (built-in algorithm)
    estimator = Estimator(
        image_uri=xgboost_image_uri,
        role=role_arn,
        instance_type=training_instance_type,
        instance_count=1,
        hyperparameters={
            "max_depth": "6",
            "eta": "0.3",
            "min_child_weight": "1",
            "subsample": "0.8",
            "colsample_bytree": "0.8",
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "num_round": "100"
        },
        sagemaker_session=sagemaker_session
    )
    
    train_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )
    
    # Step 3: Model Evaluation
    evaluate_processor = ScriptProcessor(
        role=role_arn,
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/sklearn-processing:1.0-1.cpu",
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="churn-evaluate",
        sagemaker_session=sagemaker_session
    )
    
    evaluate_step = ProcessingStep(
        name="EvaluateModel",
        processor=evaluate_processor,
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket_name}/experiments/evaluation"
            )
        ],
        code="s3://{}/pipelines/evaluate.py".format(bucket_name),
        environment={
            "MLFLOW_TRACKING_URI": f"http://mlflow.default.svc.cluster.local:5000",
            "MLFLOW_EXPERIMENT_NAME": "churn-prediction-evaluation"
        }
    )
    
    # Step 4: Register Model (condition will be added later if needed)
    # Author: Rajinikanth Vadla
    # Create model for registration using the same XGBoost image
    from sagemaker.model import Model
    
    xgboost_image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.7-1",
        py_version="py3"
    )
    
    model = Model(
        image_uri=xgboost_image_uri,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role_arn,
        sagemaker_session=sagemaker_session
    )
    
    register_step = RegisterModel(
        name="RegisterModel",
        model=model,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="churn-prediction-models",
        approval_status=model_approval_status
    )
    
    # Create pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            training_instance_type,
            model_approval_status,
            accuracy_threshold
        ],
        steps=[preprocess_step, train_step, evaluate_step, register_step],
        sagemaker_session=sagemaker_session
    )
    
    return pipeline

if __name__ == "__main__":
    import sys
    
    role_arn = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SAGEMAKER_ROLE_ARN")
    bucket_name = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("S3_BUCKET")
    
    if not role_arn or not bucket_name:
        print("Usage: python pipeline_definition.py <role_arn> <bucket_name>")
        sys.exit(1)
    
    pipeline = get_pipeline(role_arn, bucket_name)
    pipeline.upsert(role_arn=role_arn)
    print(f"Pipeline '{pipeline.name}' created/updated successfully!")
