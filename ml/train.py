"""
ml/train.py
-----------
Model training pipeline for the Hospital Readmission project.

What this module does:
    Trains three models on the preprocessed dataset:
        1. Logistic Regression
        2. Random Forest
        3. XGBoost

    Each model is trained with class-imbalance handling (class_weight='balanced'
    or scale_pos_weight) because the readmission class is minority (~11%).

Why three models?
    Comparing a linear model (Logistic Regression) with ensemble methods
    (Random Forest, XGBoost) shows the benefit of non-linear models for
    healthcare prediction tasks. Project requirement Section 19.

Important clinical note:
    In healthcare readmission prediction, RECALL is the priority metric.
    Missing a high-risk patient (false negative) is more harmful than
    triggering an unnecessary follow-up (false positive).

Viva questions:
    Q: Why is recall more important than accuracy for this problem?
    A: If we predict "not readmitted" for a patient who will be readmitted,
       the hospital misses the opportunity to intervene. This is a false
       negative. High recall minimises false negatives.

    Q: What is class imbalance and how do you handle it?
    A: Only ~11% of encounters result in <30-day readmission (positive class).
       Without correction, models learn to always predict "not readmitted"
       and achieve 89% accuracy while being useless clinically. We handle
       it via class_weight='balanced' (Logistic Regression, Random Forest)
       and scale_pos_weight (XGBoost).
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import yaml
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_logistic_regression(config: dict) -> LogisticRegression:
    """
    Build a Logistic Regression classifier.

    Logistic Regression is the simplest interpretable model.
    It assigns a probability to each class using a sigmoid function applied
    to a linear combination of features.

    Parameters from config:
        C (float): Inverse regularisation strength. Smaller = more regularisation.
        max_iter (int): Maximum iterations for solver convergence.
        class_weight: 'balanced' corrects for class imbalance.
    """
    params = config.get("ml", {}).get("models", {}).get("logistic_regression", {})
    model = LogisticRegression(
        C=params.get("C", 1.0),
        max_iter=params.get("max_iter", 1000),
        class_weight=params.get("class_weight", "balanced"),
        solver=params.get("solver", "lbfgs"),
        random_state=config.get("ml", {}).get("random_state", 42),
        n_jobs=-1,
    )
    logger.info("Logistic Regression built: %s", model.get_params())
    return model


def build_random_forest(config: dict) -> RandomForestClassifier:
    """
    Build a Random Forest classifier.

    Random Forest trains many decision trees on bootstrapped subsets and
    averages their predictions (bagging). It handles non-linear relationships
    and provides feature importance.

    Parameters from config:
        n_estimators: Number of trees.
        max_depth: Maximum depth per tree (prevents overfitting).
        class_weight: 'balanced' adjusts sample weights for minority class.
    """
    params = config.get("ml", {}).get("models", {}).get("random_forest", {})
    model = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 10),
        min_samples_split=params.get("min_samples_split", 5),
        class_weight=params.get("class_weight", "balanced"),
        random_state=config.get("ml", {}).get("random_state", 42),
        n_jobs=-1,
    )
    logger.info("Random Forest built: %s", model.get_params())
    return model


def build_xgboost(config: dict, scale_pos_weight: float = 10.0) -> XGBClassifier:
    """
    Build an XGBoost classifier.

    XGBoost uses gradient boosting — sequentially building trees where
    each tree corrects errors of the previous one. It is often the
    best-performing model for tabular data.

    scale_pos_weight:
        The ratio of negative to positive samples. For ~11% positive rate,
        this is approximately (1 - 0.11) / 0.11 ≈ 8.  We use the actual
        computed value from the training set.

    Parameters from config:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage (eta).
    """
    params = config.get("ml", {}).get("models", {}).get("xgboost", {})
    model = XGBClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=config.get("ml", {}).get("random_state", 42),
        n_jobs=-1,
        verbosity=0,
    )
    logger.info("XGBoost built with scale_pos_weight=%.2f", scale_pos_weight)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
) -> Any:
    """
    Train a single model and log timing.

    Args:
        model: Unfitted scikit-learn compatible model.
        X_train (np.ndarray): Preprocessed training features.
        y_train (np.ndarray): Training labels.
        model_name (str): Human-readable model name for logging.

    Returns:
        Fitted model instance.
    """
    import time
    logger.info("Training %s on %d samples...", model_name, len(X_train))
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info("%s trained in %.1f seconds.", model_name, elapsed)
    return model


def save_model(model, model_name: str, save_dir: str = "models") -> str:
    """
    Save a fitted model to disk using joblib.

    Args:
        model: Fitted model.
        model_name (str): Name used for the filename.
        save_dir (str): Directory to save the model.

    Returns:
        str: Path to the saved model file.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model_file = save_path / f"{model_name}.joblib"
    joblib.dump(model, model_file)
    logger.info("Model saved: %s", model_file)
    return str(model_file)


# ─────────────────────────────────────────────────────────────────────────────
# Main training runner
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    preprocessing_result: dict,
    config_path: str = "config/config.yaml",
    save_dir: str = "models",
) -> Dict[str, Any]:
    """
    Train all three models and save them.

    Args:
        preprocessing_result (dict): Output of ml/preprocessing.py:run_ml_preprocessing().
        config_path (str): Path to config.yaml.
        save_dir (str): Directory for saved models.

    Returns:
        dict: Fitted models keyed by name.
    """
    config = load_config(config_path)

    X_train = preprocessing_result["X_train"]
    y_train = preprocessing_result["y_train"]

    # Compute scale_pos_weight for XGBoost from training labels
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)
    logger.info(
        "Class balance — neg: %d, pos: %d, scale_pos_weight: %.2f",
        neg_count, pos_count, scale_pos_weight,
    )

    logger.info("=" * 60)
    logger.info("MODEL TRAINING STARTED")
    logger.info("=" * 60)

    models = {}

    # ── Logistic Regression ──────────────────────────────────────────────────
    lr = build_logistic_regression(config)
    lr = train_model(lr, X_train, y_train, "LogisticRegression")
    save_model(lr, "logistic_regression", save_dir)
    models["logistic_regression"] = {"model": lr, "name": "Logistic Regression"}

    # ── Random Forest ────────────────────────────────────────────────────────
    rf = build_random_forest(config)
    rf = train_model(rf, X_train, y_train, "RandomForest")
    save_model(rf, "random_forest", save_dir)
    models["random_forest"] = {"model": rf, "name": "Random Forest"}

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb = build_xgboost(config, scale_pos_weight=scale_pos_weight)
    xgb = train_model(xgb, X_train, y_train, "XGBoost")
    save_model(xgb, "xgboost", save_dir)
    models["xgboost"] = {"model": xgb, "name": "XGBoost"}

    logger.info("=" * 60)
    logger.info("MODEL TRAINING COMPLETE — 3 models trained and saved")
    logger.info("=" * 60)

    return models


if __name__ == "__main__":
    from ml.preprocessing import run_ml_preprocessing
    print("Running ML preprocessing...")
    prep = run_ml_preprocessing(use_postgres=False)
    print("Training models...")
    trained = run_training(prep)
    print(f"\nTrained models: {list(trained.keys())}")
