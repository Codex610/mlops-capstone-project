import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import os
import mlflow.sklearn
import dagshub
from src.logger import logging

# -------------------------------------------------------------------------------------
# ✅ Tracking setup - this setup is for locally
# -------------------------------------------------------------------------------------
# TRACKING_URI = "https://dagshub.com/codex03080/mlops-capstone-project.mlflow"

# mlflow.set_tracking_uri(TRACKING_URI)
# dagshub.init(repo_owner='codex03080', repo_name='mlops-capstone-project', mlflow=True)

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "codex03080"
repo_name = "mlops-capstone-project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------
def load_model(path):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        logging.info("Model loaded from %s", path)
        return model
    except Exception as e:
        logging.error("Error loading model: %s", e)
        raise


def load_data(path):
    try:
        df = pd.read_csv(path)
        logging.info("Data loaded from %s", path)
        return df
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise


def evaluate_model(clf, X_test, y_test):
    try:
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob)
        }

        logging.info("Model evaluation completed")
        return metrics

    except Exception as e:
        logging.error("Error during evaluation: %s", e)
        raise


def save_json(data, path):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logging.info("Saved JSON to %s", path)
    except Exception as e:
        logging.error("Error saving JSON: %s", e)
        raise


# -------------------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------------------
def main():
    try:
        mlflow.set_experiment("my-dvc-pipeline")

        with mlflow.start_run() as run:

            # ✅ Load model + data
            clf = load_model('./models/model.pkl')
            test = load_data('./data/processed/test_bow.csv')

            X_test = test.iloc[:, :-1].values
            y_test = test.iloc[:, -1].values

            # ✅ Evaluate
            metrics = evaluate_model(clf, X_test, y_test)

            # ✅ Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # ✅ Log params
            if hasattr(clf, "get_params"):
                for k, v in clf.get_params().items():
                    mlflow.log_param(k, v)

            # 🔥 CRITICAL FIX (your required part)
            mlflow.sklearn.log_model(clf, artifact_path="model")

            # ✅ Save correct model info
            model_info = {
                "run_id": run.info.run_id,
                "model_path": "model"
            }

            save_json(model_info, "reports/experiment_info.json")
            save_json(metrics, "reports/metrics.json")

            # ✅ Log metrics file
            mlflow.log_artifact("reports/metrics.json")

            logging.info("Evaluation + logging completed successfully")

    except Exception as e:
        logging.error("Pipeline failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()