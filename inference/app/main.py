# inference/app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
import json
import os
from typing import List, Optional
from .model_handler import invoke_sagemaker_endpoint

app = FastAPI(title="Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    churn_probability: float

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerFeatures):
    try:
        endpoint_name = os.getenv("SAGEMAKER_ENDPOINT_NAME", "churn-prediction-endpoint")
        
        result = invoke_sagemaker_endpoint(
            endpoint_name=endpoint_name,
            features=customer.dict()
        )
        
        churn_prob = result["probability"]
        prediction = "Churn" if churn_prob > 0.5 else "No Churn"
        
        return PredictionResponse(
            prediction=prediction,
            probability=churn_prob if prediction == "Churn" else 1 - churn_prob,
            churn_probability=churn_prob
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(customers: List[CustomerFeatures]):
    endpoint_name = os.getenv("SAGEMAKER_ENDPOINT_NAME", "churn-prediction-endpoint")
    results = []
    
    for customer in customers:
        try:
            result = invoke_sagemaker_endpoint(
                endpoint_name=endpoint_name,
                features=customer.dict()
            )
            results.append({
                "prediction": "Churn" if result["probability"] > 0.5 else "No Churn",
                "probability": result["probability"]
            })
        except Exception as e:
            results.append({"error": str(e)})
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
