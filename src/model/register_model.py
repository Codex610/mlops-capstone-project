import json
import mlflow
import dagshub
from src.logger import logging

# ✅ Tracking setup
TRACKING_URI = "https://dagshub.com/codex03080/mlops-capstone-project.mlflow"

mlflow.set_tracking_uri(TRACKING_URI)
dagshub.init(repo_owner='codex03080', repo_name='mlops-capstone-project', mlflow=True)


def load_model_info(path):
    with open(path, "r") as f:
        return json.load(f)


def register_model(model_name, model_info):
    run_id = model_info["run_id"]
    model_path = model_info["model_path"]

    # 🔍 Debug: see what's actually inside run
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)

    logging.info("Available artifacts: %s", [a.path for a in artifacts])
    logging.info("Using model_path: %s", model_path)

    model_uri = f"runs:/{run_id}/{model_path}"

    logging.info("Registering model from: %s", model_uri)

    model_version = mlflow.register_model(model_uri, model_name)

    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"
    )

    logging.info("Model registered successfully!")


def main():
    model_info = load_model_info("reports/experiment_info.json")
    register_model("mymodel", model_info)


if __name__ == "__main__":
    main()