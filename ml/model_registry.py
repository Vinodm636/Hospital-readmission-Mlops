"""
ml/model_registry.py
---------------------
MLflow experiment tracking and model registry for the Hospital Readmission project.

What this module does:
    Integrates MLflow to:
    - Log model parameters, metrics, and artefacts for each training run
    - Register the best model in the MLflow Model Registry
    - Version models for deployment

Why MLflow:
    MLflow provides a standard API for tracking ML experiments, comparing runs,
    and promoting models through lifecycle stages (Staging → Production).
    It makes experiments reproducible and auditable.

Key concepts:
    Experiment: A named group of runs (e.g. 'hospital_readmission_prediction').
    Run: One execution of training + evaluation. Each run has an ID, parameters,
         metrics, and artefacts.
    Model Registry: A centralised store for versioned models with lifecycle stages.

Viva questions:
    Q: What is MLflow and why do we use it?
    A: MLflow is an open-source platform for managing the ML lifecycle.
       We use it to track experiment runs, compare model performance,
       and maintain a registry of production-ready models.

    Q: What is a model registry?
    A: A centralised system that stores model versions with metadata.
       Models can be tagged with stages: None → Staging → Production.
       This supports controlled deployment of ML models.

    Q: What artefacts do we log?
    A: Model file (.joblib), preprocessor, ROC curve PNG, calibration
       curve PNG, confusion matrix PNG, comparison table CSV, and feature
       names. This makes each experiment fully reproducible.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import yaml
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# MLflow setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_mlflow(config: dict) -> str:
    """
    Configure the MLflow tracking URI and create/set the experiment.

    Args:
        config (dict): Project configuration with MLflow settings.

    Returns:
        str: Active experiment name.
    """
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{PROJECT_ROOT / 'mlruns' / 'mlflow.db'}"
    )
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        config.get("ml", {}).get("experiment_name", "hospital_readmission_prediction"),
    )

    # Create experiment if it doesn't exist
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            tags={
                "project": "hospital_readmission",
                "course": "Data Engineering and MLOps",
                "dataset": "UCI Diabetes 130-US Hospitals",
            },
        )

    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking URI: %s", tracking_uri)
    logger.info("MLflow experiment: %s", experiment_name)
    return experiment_name


# ─────────────────────────────────────────────────────────────────────────────
# Log a single model run
# ─────────────────────────────────────────────────────────────────────────────

def log_model_run(
    model,
    model_name: str,
    model_key: str,
    metrics: dict,
    params: dict,
    preprocessing_result: dict,
    artefact_paths: dict,
    config: dict,
) -> str:
    """
    Log one model training run to MLflow.

    Logs:
        - Parameters: model hyperparameters + preprocessing config
        - Metrics: recall, precision, F1, ROC-AUC, Brier score, TP/FP/FN/TN
        - Model: scikit-learn model artefact
        - Artefacts: plots, comparison table, feature names, preprocessor

    Args:
        model: Fitted model.
        model_name (str): Human-readable model name.
        model_key (str): Key in models dict (e.g. 'logistic_regression').
        metrics (dict): Evaluation metrics.
        params (dict): Model hyperparameters.
        preprocessing_result (dict): Preprocessing metadata.
        artefact_paths (dict): Paths to files to log as artefacts.
        config (dict): Project configuration.

    Returns:
        str: MLflow run ID.
    """
    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: %s | run_id: %s", model_name, run_id)

        # ── Tags ─────────────────────────────────────────────────────────────
        mlflow.set_tags({
            "model_type": model_key,
            "dataset": "UCI Diabetes 130-US Hospitals",
            "target": "readmission_binary",
            "split": "stratified_random",
            "academic_prototype": "true",
        })

        # ── Parameters ───────────────────────────────────────────────────────
        mlflow.log_params(params)
        mlflow.log_params({
            "train_size": preprocessing_result.get("train_size"),
            "val_size": preprocessing_result.get("val_size"),
            "test_size": preprocessing_result.get("test_size"),
            "positive_rate": round(preprocessing_result.get("positive_rate", 0), 4),
            "n_features": len(preprocessing_result.get("feature_names", [])),
            "random_state": config.get("ml", {}).get("random_state", 42),
        })

        # ── Metrics ───────────────────────────────────────────────────────────
        metric_keys = [
            "recall", "precision", "f1_score", "roc_auc", "brier_score",
            "true_positives", "false_positives", "true_negatives", "false_negatives",
        ]
        for key in metric_keys:
            if key in metrics:
                mlflow.log_metric(key, metrics[key])

        # ── Log model ─────────────────────────────────────────────────────────
        if model_key == "xgboost":
            mlflow.xgboost.log_model(
                model, artifact_path="model",
                registered_model_name=None,  # Register separately after comparison
            )
        else:
            mlflow.sklearn.log_model(
                model, artifact_path="model",
                registered_model_name=None,
            )

        # ── Log artefacts ─────────────────────────────────────────────────────
        for name, path in artefact_paths.items():
            if path and Path(path).exists():
                mlflow.log_artifact(path, artifact_path=name)

        # ── Log feature names ─────────────────────────────────────────────────
        feature_names = preprocessing_result.get("feature_names", [])
        if feature_names:
            feature_file = Path("models") / "feature_names.txt"
            feature_file.parent.mkdir(exist_ok=True)
            feature_file.write_text("\n".join(feature_names))
            mlflow.log_artifact(str(feature_file), artifact_path="features")

        logger.info("MLflow run logged: %s", run_id)
        return run_id


# ─────────────────────────────────────────────────────────────────────────────
# Register the best model
# ─────────────────────────────────────────────────────────────────────────────

def register_best_model(
    best_model,
    best_model_key: str,
    best_metrics: dict,
    preprocessing_result: dict,
    run_id: str,
    config: dict,
) -> str:
    """
    Register the best model in the MLflow Model Registry.

    Lifecycle stages:
        None → Staging → Production
    For academic purposes we register directly to 'Staging'.

    Args:
        best_model: The best fitted model.
        best_model_key (str): Model type key.
        best_metrics (dict): Validation metrics for the best model.
        preprocessing_result (dict): Preprocessing metadata.
        run_id (str): MLflow run ID of the best model run.
        config (dict): Project configuration.

    Returns:
        str: Registered model name and version.
    """
    registered_model_name = "hospital_readmission_classifier"

    # Log the best model as a new run specifically for registration
    with mlflow.start_run(run_name=f"BEST_MODEL_{best_metrics['model_name']}") as run:
        if best_model_key == "xgboost":
            model_info = mlflow.xgboost.log_model(
                best_model,
                artifact_path="model",
                registered_model_name=registered_model_name,
            )
        else:
            model_info = mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
                registered_model_name=registered_model_name,
            )

        # Log best model metrics and tags
        mlflow.log_metrics({
            k: v for k, v in best_metrics.items()
            if isinstance(v, (int, float))
        })
        mlflow.set_tags({
            "model_type": best_model_key,
            "is_best_model": "true",
            "primary_metric": config.get("ml", {}).get("primary_metric", "recall"),
            "deployment_stage": "staging",
            "academic_prototype": "true",
            "clinical_disclaimer": "NOT FOR CLINICAL USE",
        })

        registered_run_id = run.info.run_id

    # Transition to Staging (not Production — academic prototype)
    client = mlflow.tracking.MlflowClient()
    try:
        latest = client.get_latest_versions(
            registered_model_name, stages=["None"]
        )
        if latest:
            version = latest[0].version
            client.transition_model_version_stage(
                name=registered_model_name,
                version=version,
                stage="Staging",
                archive_existing_versions=False,
            )
            logger.info(
                "Model registered: %s v%s → Staging", registered_model_name, version
            )
            return f"{registered_model_name}:v{version}"
    except Exception as exc:
        logger.warning("Could not transition model stage: %s", exc)

    return registered_model_name


# ─────────────────────────────────────────────────────────────────────────────
# Master MLflow runner
# ─────────────────────────────────────────────────────────────────────────────

def run_mlflow_tracking(
    models: dict,
    evaluation_result: dict,
    preprocessing_result: dict,
    config_path: str = "config/config.yaml",
    save_dir: str = "models",
) -> dict:
    """
    Log all model runs to MLflow and register the best model.

    Args:
        models (dict): Trained models dict from ml/train.py.
        evaluation_result (dict): Output of ml/evaluate.py.
        preprocessing_result (dict): Output of ml/preprocessing.py.
        config_path (str): Path to config.yaml.
        save_dir (str): Directory with model artefacts.

    Returns:
        dict: MLflow run IDs and registered model info.
    """
    config = load_config(config_path)
    experiment_name = setup_mlflow(config)

    logger.info("=" * 60)
    logger.info("MLFLOW TRACKING STARTED — experiment: %s", experiment_name)
    logger.info("=" * 60)

    val_metrics_by_name = {
        m["model_name"]: m for m in evaluation_result["val_metrics"]
    }

    run_ids = {}

    # Artefacts that apply to all models
    common_artefacts = {
        "plots": evaluation_result.get("roc_plot"),
        "calibration": evaluation_result.get("calibration_plot"),
        "comparison": evaluation_result.get("comparison_csv"),
    }

    # Log each model
    for key, info in models.items():
        model_name = info["name"]
        metrics = val_metrics_by_name.get(model_name, {})

        # Extract model params
        model_obj = info["model"]
        try:
            params = {k: str(v) for k, v in model_obj.get_params().items()}
        except Exception:
            params = {"model_type": key}

        # Model-specific artefact (confusion matrix)
        cm_path = str(
            Path(save_dir) / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        )
        artefacts = {**common_artefacts, "confusion_matrix": cm_path}

        run_id = log_model_run(
            model=model_obj,
            model_name=model_name,
            model_key=key,
            metrics=metrics,
            params=params,
            preprocessing_result=preprocessing_result,
            artefact_paths=artefacts,
            config=config,
        )
        run_ids[key] = run_id

    # Register the best model
    best_key = evaluation_result["best_model_key"]
    best_metrics = evaluation_result["best_val_metrics"]
    best_run_id = run_ids.get(best_key, "")

    registered_name = register_best_model(
        best_model=evaluation_result["best_model"],
        best_model_key=best_key,
        best_metrics=best_metrics,
        preprocessing_result=preprocessing_result,
        run_id=best_run_id,
        config=config,
    )

    logger.info("=" * 60)
    logger.info("MLFLOW TRACKING COMPLETE — registered: %s", registered_name)
    logger.info("=" * 60)

    return {
        "run_ids": run_ids,
        "registered_model": registered_name,
        "experiment_name": experiment_name,
        "best_model_key": best_key,
    }
