# pipelines/preprocess.py

import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def preprocess_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-train", type=str, default="/opt/ml/processing/output/train")
    parser.add_argument("--output-validation", type=str, default="/opt/ml/processing/output/validation")
    parser.add_argument("--output-test", type=str, default="/opt/ml/processing/output/test")
    parser.add_argument("--output-model", type=str, default="/opt/ml/processing/output/model")
    args = parser.parse_args()
    
    print("Reading raw data...")
    input_files = os.listdir(args.input_data)
    csv_file = [f for f in input_files if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(args.input_data, csv_file))
    
    print(f"Original shape: {df.shape}")
    
    # Handle missing values in TotalCharges
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_total_charges = df["TotalCharges"].median()
    df["TotalCharges"].fillna(median_total_charges, inplace=True)
    
    # Drop customerID
    df = df.drop("customerID", axis=1)
    
    # Encode binary columns
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", 
                   "PaperlessBilling", "Churn"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})
    
    # One-hot encode multi-category columns
    categorical_cols = ["Contract", "PaymentMethod", "InternetService"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
    
    # Create engineered features
    df_encoded["AvgMonthlyCharges"] = df_encoded["TotalCharges"] / df_encoded["tenure"].replace(0, 1)
    
    contract_mapping = {"Month-to-month": 3, "One year": 2, "Two year": 1}
    if "Contract_Month-to-month" in df_encoded.columns:
        df_encoded["ContractRiskScore"] = (
            df_encoded["Contract_Month-to-month"] * 3 +
            df_encoded.get("Contract_One year", 0) * 2 +
            df_encoded.get("Contract_Two year", 0) * 1
        )
    else:
        df_encoded["ContractRiskScore"] = 2
    
    service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", 
                    "OnlineBackup", "DeviceProtection", "TechSupport", 
                    "StreamingTV", "StreamingMovies"]
    df_encoded["ServiceCount"] = df_encoded[service_cols].sum(axis=1)
    
    df_encoded["HasPremiumSupport"] = (
        (df_encoded.get("OnlineSecurity", 0) == 1) & 
        (df_encoded.get("TechSupport", 0) == 1)
    ).astype(int)
    
    # Separate target
    target = df_encoded["Churn"]
    features = df_encoded.drop("Churn", axis=1)
    
    # Scale numeric features
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    features[numeric_cols] = scaler.fit_transform(features[numeric_cols])
    
    # Split data: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, target, test_size=0.15, random_state=42, stratify=target
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
    
    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
    
    # Save splits
    os.makedirs(args.output_train, exist_ok=True)
    os.makedirs(args.output_validation, exist_ok=True)
    os.makedirs(args.output_test, exist_ok=True)
    os.makedirs(args.output_model, exist_ok=True)
    
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv(os.path.join(args.output_train, "train.csv"), index=False)
    val_data.to_csv(os.path.join(args.output_validation, "validation.csv"), index=False)
    test_data.to_csv(os.path.join(args.output_test, "test.csv"), index=False)
    
    # Save preprocessing artifacts
    joblib.dump(scaler, os.path.join(args.output_model, "scaler.joblib"))
    
    feature_columns = {"feature_columns": features.columns.tolist()}
    with open(os.path.join(args.output_model, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f)
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_data()
