# inference/app/model_handler.py

import boto3
import json
import csv
import io
import os

sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))

def invoke_sagemaker_endpoint(endpoint_name: str, features: dict) -> dict:
    """
    Invoke SageMaker endpoint with customer features.
    Returns prediction probability.
    """
    # Convert features to CSV format (XGBoost expects CSV)
    # Note: This is simplified - you need to match exact feature order from training
    feature_values = [
        features.get("gender", 0),
        features.get("SeniorCitizen", 0),
        features.get("Partner", 0),
        features.get("Dependents", 0),
        features.get("tenure", 0),
        features.get("PhoneService", 0),
        features.get("MultipleLines", 0),
        features.get("InternetService", ""),
        features.get("OnlineSecurity", 0),
        features.get("OnlineBackup", 0),
        features.get("DeviceProtection", 0),
        features.get("TechSupport", 0),
        features.get("StreamingTV", 0),
        features.get("StreamingMovies", 0),
        features.get("Contract", ""),
        features.get("PaperlessBilling", 0),
        features.get("PaymentMethod", ""),
        features.get("MonthlyCharges", 0.0),
        features.get("TotalCharges", 0.0)
    ]
    
    # Create CSV string
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(feature_values)
    csv_payload = csv_buffer.getvalue()
    
    # Invoke endpoint
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/csv",
        Body=csv_payload.encode("utf-8")
    )
    
    # Parse response
    result = json.loads(response["Body"].read().decode("utf-8"))
    
    # XGBoost returns probabilities
    probability = float(result["predictions"][0]["score"])
    
    return {
        "probability": probability,
        "raw_response": result
    }
