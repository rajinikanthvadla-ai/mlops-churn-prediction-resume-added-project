# pipelines/evaluate.py
# Author: Rajinikanth Vadla
# Description: Model evaluation script that logs metrics and artifacts to MLflow

import argparse
import os
import json
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import mlflow
import mlflow.xgboost
import numpy as np

def evaluate_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--test-data", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--evaluation-output", type=str, default="/opt/ml/processing/evaluation")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="churn-prediction-evaluation")
    args = parser.parse_args()
    
    # Setup MLflow
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    
    print("Loading model...")
    # Author: Rajinikanth Vadla
    # SageMaker XGBoost saves model as model.tar.gz containing xgboost-model file
    import tarfile
    import tempfile
    
    # Find model.tar.gz file
    model_tar = None
    for f in os.listdir(args.model_dir):
        if f.endswith(".tar.gz") or f == "model.tar.gz":
            model_tar = os.path.join(args.model_dir, f)
            break
    
    if not model_tar:
        # Try to find any .tar.gz file
        model_files = [f for f in os.listdir(args.model_dir) if f.endswith(".tar.gz")]
        if model_files:
            model_tar = os.path.join(args.model_dir, model_files[0])
        else:
            raise FileNotFoundError(f"No model.tar.gz found in {args.model_dir}")
    
    # Extract model.tar.gz to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(model_tar, "r:gz") as tar:
            tar.extractall(tmpdir)
        
        # Find xgboost-model file (could be xgboost-model or model.json)
        model_file = None
        for root, dirs, files in os.walk(tmpdir):
            for f in files:
                if f == "xgboost-model" or f.endswith(".json"):
                    model_file = os.path.join(root, f)
                    break
            if model_file:
                break
        
        if not model_file:
            raise FileNotFoundError(f"No xgboost-model file found in {model_tar}")
        
        model = xgb.Booster()
        model.load_model(model_file)
    
    print("Loading test data...")
    # Author: Rajinikanth Vadla
    # Test data has NO header and target is FIRST column (from preprocess.py)
    test_files = [f for f in os.listdir(args.test_data) if f.endswith(".csv")]
    test_df = pd.read_csv(os.path.join(args.test_data, test_files[0]), header=None)
    
    # First column is target, rest are features
    y_test = test_df.iloc[:, 0].astype(int)
    X_test = test_df.iloc[:, 1:]
    
    # Predictions
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "test_accuracy": float(accuracy),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_auc": float(auc)
    }
    
    print("Test Metrics:", metrics)
    print("Confusion Matrix:")
    print(cm)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save evaluation report
    os.makedirs(args.evaluation_output, exist_ok=True)
    evaluation_report = {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    
    with open(os.path.join(args.evaluation_output, "evaluation.json"), "w") as f:
        json.dump(evaluation_report, f, indent=2)
    
    # Log to MLflow
    mlflow_tracking_uri = args.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            mlflow.log_dict(evaluation_report, "evaluation_report.json")
            
            # Log confusion matrix as artifact
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.savefig(os.path.join(args.evaluation_output, "confusion_matrix.png"))
            mlflow.log_artifact(os.path.join(args.evaluation_output, "confusion_matrix.png"))
            print(f"Logged metrics to MLflow: {mlflow_tracking_uri}")
    
    print("Evaluation complete!")
    return metrics

if __name__ == "__main__":
    evaluate_model()
