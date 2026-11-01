"""
ml/evaluate.py
--------------
Model evaluation module for the Hospital Readmission project.

What this module does:
    Evaluates each trained model on the validation and test sets using:
        - Recall       (priority metric for healthcare)
        - Precision
        - F1-score
        - ROC-AUC
        - Calibration  (reliability of predicted probabilities)
        - Confusion matrix

Why recall is the primary metric:
    In hospital readmission prediction, a false negative (predicting "not
    readmitted" when the patient will be) means the hospital misses an
    opportunity to intervene and reduce the readmission. This has a direct
    patient safety impact. We therefore prioritise high recall, accepting
    lower precision if needed.

Calibration:
    A well-calibrated model means that when it predicts 70% probability
    of readmission, approximately 70% of such patients are actually
    readmitted. This is important for clinical decision support.

Output:
    - Evaluation metrics dict
    - Comparison table (all three models)
    - Saved plots (confusion matrix, ROC curve, calibration curve)

Viva questions:
    Q: What is the difference between precision and recall?
    A: Precision = of all predicted positive, how many are actually positive.
       Recall = of all actual positive, how many did we correctly predict.
       In healthcare we prefer high recall (catch all high-risk patients)
       even at the cost of some false alarms.

    Q: What is ROC-AUC?
    A: The Area Under the ROC Curve. It measures the model's ability to
       distinguish between positive and negative classes across all
       classification thresholds. AUC=0.5 is random, AUC=1.0 is perfect.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display required)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    recall_score,
    precision_score,
    f1_score,
    brier_score_loss,
)

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
# Single model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate a fitted model and return a comprehensive metrics dictionary.

    Args:
        model: Fitted scikit-learn model with predict_proba().
        X (np.ndarray): Preprocessed feature matrix.
        y (np.ndarray): True labels.
        model_name (str): Human-readable model name.
        threshold (float): Classification threshold (default 0.5).

    Returns:
        dict: All evaluation metrics.
    """
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Core metrics
    recall = recall_score(y, y_pred, zero_division=0)
    precision = precision_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, y_pred_proba)
    brier = brier_score_loss(y, y_pred_proba)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    metrics = {
        "model_name": model_name,
        "threshold": threshold,
        "recall": round(float(recall), 4),
        "precision": round(float(precision), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "brier_score": round(float(brier), 4),   # lower is better
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "positive_rate_predicted": round(float(y_pred.mean()), 4),
        "positive_rate_actual": round(float(y.mean()), 4),
    }

    logger.info(
        "[%s] Recall=%.3f | Precision=%.3f | F1=%.3f | AUC=%.3f | Brier=%.3f",
        model_name, recall, precision, f1, roc_auc, brier,
    )

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    model,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    save_dir: str = "models",
    threshold: float = 0.5,
) -> str:
    """Plot and save a confusion matrix heatmap."""
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Readmitted", "Readmitted <30"])
    ax.set_yticklabels(["Not Readmitted", "Readmitted <30"])
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)

    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    fig_file = save_path / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fig_file, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved: %s", fig_file)
    return str(fig_file)


def plot_roc_curves(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    save_dir: str = "models",
) -> str:
    """Plot ROC curves for all models on the same axes."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC=0.50)")

    colors = ["#E63946", "#457B9D", "#2A9D8F"]
    for (key, info), color in zip(models.items(), colors):
        model = info["model"]
        name = info["name"]
        y_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves — All Models (Validation Set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    save_path = Path(save_dir)
    fig_file = save_path / "roc_curves_all_models.png"
    plt.savefig(fig_file, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("ROC curves saved: %s", fig_file)
    return str(fig_file)


def plot_calibration_curves(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    save_dir: str = "models",
) -> str:
    """
    Plot calibration curves (reliability diagrams) for all models.

    A perfectly calibrated model has its curve on the diagonal.
    Curves above the diagonal are under-predicting (too conservative).
    Curves below are over-predicting (too confident).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    colors = ["#E63946", "#457B9D", "#2A9D8F"]
    for (key, info), color in zip(models.items(), colors):
        model = info["model"]
        name = info["name"]
        y_proba = model.predict_proba(X)[:, 1]
        prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10)
        brier = brier_score_loss(y, y_proba)
        ax.plot(prob_pred, prob_true, marker="o", color=color,
                label=f"{name} (Brier={brier:.3f})")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves — All Models (Validation Set)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    save_path = Path(save_dir)
    fig_file = save_path / "calibration_curves_all_models.png"
    plt.savefig(fig_file, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("Calibration curves saved: %s", fig_file)
    return str(fig_file)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table and model selection
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(all_metrics: list) -> pd.DataFrame:
    """
    Build a comparison table from a list of metric dicts.

    Args:
        all_metrics (list): List of metric dicts from evaluate_model().

    Returns:
        pd.DataFrame: Comparison table sorted by recall (primary metric).
    """
    df = pd.DataFrame(all_metrics)
    cols = ["model_name", "recall", "precision", "f1_score", "roc_auc",
            "brier_score", "true_positives", "false_negatives"]
    cols_present = [c for c in cols if c in df.columns]
    df = df[cols_present].sort_values("recall", ascending=False)
    logger.info("\nModel comparison table:\n%s", df.to_string(index=False))
    return df


def select_best_model(
    all_metrics: list,
    primary_metric: str = "recall",
) -> dict:
    """
    Select the best model based on the primary metric.

    Args:
        all_metrics (list): List of metric dicts.
        primary_metric (str): Metric name to optimise (default: 'recall').

    Returns:
        dict: Metrics dict of the best model.
    """
    best = max(all_metrics, key=lambda m: m.get(primary_metric, 0))
    logger.info(
        "Best model by %s: %s (%.4f)",
        primary_metric, best["model_name"], best.get(primary_metric, 0),
    )
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Master evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    models: dict,
    preprocessing_result: dict,
    config_path: str = "config/config.yaml",
    save_dir: str = "models",
) -> dict:
    """
    Evaluate all three models on the validation set and select the best.

    Args:
        models (dict): Output of ml/train.py:run_training().
        preprocessing_result (dict): Output of ml/preprocessing.py.
        config_path (str): Path to config.yaml.
        save_dir (str): Directory for saved plots and results.

    Returns:
        dict: Evaluation results, comparison table, best model info.
    """
    config = load_config(config_path)
    primary_metric = config.get("ml", {}).get("primary_metric", "recall")

    X_val = preprocessing_result["X_val"]
    y_val = preprocessing_result["y_val"]
    X_test = preprocessing_result["X_test"]
    y_test = preprocessing_result["y_test"]

    logger.info("=" * 60)
    logger.info("MODEL EVALUATION STARTED (validation set)")
    logger.info("=" * 60)

    # ── Evaluate each model on validation set ─────────────────────────────────
    val_metrics = []
    for key, info in models.items():
        metrics = evaluate_model(info["model"], X_val, y_val, info["name"])
        val_metrics.append(metrics)
        plot_confusion_matrix(info["model"], X_val, y_val, info["name"], save_dir)

    # ── ROC + Calibration plots ───────────────────────────────────────────────
    roc_plot = plot_roc_curves(models, X_val, y_val, save_dir)
    cal_plot = plot_calibration_curves(models, X_val, y_val, save_dir)

    # ── Comparison table ──────────────────────────────────────────────────────
    comparison_df = build_comparison_table(val_metrics)
    comparison_csv = Path(save_dir) / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    logger.info("Comparison table saved: %s", comparison_csv)

    # ── Select best model ─────────────────────────────────────────────────────
    best_val_metrics = select_best_model(val_metrics, primary_metric)
    best_model_key = next(
        k for k, v in models.items()
        if v["name"] == best_val_metrics["model_name"]
    )
    best_model = models[best_model_key]["model"]

    # ── Final evaluation on test set (only for the best model) ────────────────
    logger.info("=" * 60)
    logger.info("FINAL TEST SET EVALUATION — %s", best_val_metrics["model_name"])
    logger.info("=" * 60)
    test_metrics = evaluate_model(best_model, X_test, y_test,
                                  best_val_metrics["model_name"] + " [TEST]")

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)

    return {
        "val_metrics": val_metrics,
        "comparison_df": comparison_df,
        "best_model_key": best_model_key,
        "best_model": best_model,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "roc_plot": roc_plot,
        "calibration_plot": cal_plot,
        "comparison_csv": str(comparison_csv),
    }
