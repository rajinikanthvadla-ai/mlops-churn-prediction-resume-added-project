# pipelines/train.py
# Author: Rajinikanth Vadla
# Description: XGBoost model training script with MLflow integration (reference implementation)

import argparse
import os
import json
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.xgboost
import joblib

def train_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--validation-data", type=str, default="/opt/ml/input/data/validation")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="churn-prediction-training")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--min-child-weight", type=int, default=1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--num-round", type=int, default=100)
    args = parser.parse_args()
    
    # Setup MLflow
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    
    print("Loading training data...")
    train_files = [f for f in os.listdir(args.train_data) if f.endswith(".csv")]
    train_df = pd.read_csv(os.path.join(args.train_data, train_files[0]))
    
    val_files = [f for f in os.listdir(args.validation_data) if f.endswith(".csv")]
    val_df = pd.read_csv(os.path.join(args.validation_data, val_files[0]))
    
    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]
    X_val = val_df.drop("Churn", axis=1)
    y_val = val_df["Churn"]
    
    print(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}")
    
    # Prepare DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Hyperparameters
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "random_state": 42
    }
    
    # Start MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(params)
        mlflow.log_param("num_round", args.num_round)
        
        # Train model
        evals = [(dtrain, "train"), (dval, "validation")]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=args.num_round,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=True
        )
        
        # Predictions
        y_train_pred = (model.predict(dtrain) > 0.5).astype(int)
        y_val_pred = (model.predict(dval) > 0.5).astype(int)
        y_val_pred_proba = model.predict(dval)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        
        metrics = {
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "validation_precision": val_precision,
            "validation_recall": val_recall,
            "validation_f1": val_f1,
            "validation_auc": val_auc
        }
        
        print("Metrics:", metrics)
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        # Save model locally for SageMaker
        os.makedirs(args.model_dir, exist_ok=True)
        model.save_model(os.path.join(args.model_dir, "xgboost-model.json"))
        
        # Save feature importance
        feature_importance = dict(zip(X_train.columns, model.get_score(importance_type="gain")))
        with open(os.path.join(args.model_dir, "feature_importance.json"), "w") as f:
            json.dump(feature_importance, f)
        
        print("Training complete!")
        print(f"Best validation AUC: {val_auc:.4f}")

if __name__ == "__main__":
    train_model()
