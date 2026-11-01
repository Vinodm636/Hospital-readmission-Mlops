"""
ml/explainability.py
--------------------
Model explainability for the Hospital Readmission project.

What this module does:
    1. Computes SHAP (SHapley Additive exPlanations) values for the best model
    2. Generates feature importance plots for tree-based models
    3. Creates SHAP summary plots showing global feature impact
    4. Generates SHAP waterfall plots for individual predictions
    5. Saves all explainability artefacts to disk and logs to MLflow

Why explainability matters:
    Healthcare AI models must be explainable. Clinicians and patients need to
    understand WHY the model predicts a high readmission risk. Without
    explainability, the model is a "black box" that cannot be trusted in a
    clinical setting.

SHAP explanation:
    SHAP assigns each feature a value representing its contribution to the
    model's prediction for a specific patient. Positive SHAP values push the
    prediction towards readmission; negative values push towards no readmission.

Academic disclaimer:
    This is a decision-support prototype only. SHAP explanations are for
    educational purposes. They should not be used for actual clinical decisions.

Viva questions:
    Q: What is SHAP?
    A: SHAP (SHapley Additive exPlanations) is a game-theory based method
       to explain the output of any ML model. It assigns each feature an
       importance value for a specific prediction.

    Q: What is a SHAP summary plot?
    A: It shows the distribution of SHAP values for each feature across
       all samples. Features at the top have the largest average impact.
       Colour shows whether high (red) or low (blue) feature values
       increase or decrease the predicted probability.

    Q: How do SHAP values help in healthcare?
    A: They answer the question: "Why did the model predict high readmission
       risk for this patient?" For example, SHAP might show that the patient's
       high prior_admission_count was the main driver of the high risk score.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance (tree-based models)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: list,
    model_name: str,
    top_n: int = 20,
    save_dir: str = "models",
) -> str:
    """
    Plot feature importance for tree-based models (Random Forest, XGBoost).

    Feature importance in tree models is the mean decrease in impurity
    (for Random Forest) or the total gain per feature (for XGBoost).

    Args:
        model: Fitted tree-based model with feature_importances_ attribute.
        feature_names (list): Names of input features after preprocessing.
        model_name (str): Model name for the plot title.
        top_n (int): Number of top features to display.
        save_dir (str): Directory to save the plot.

    Returns:
        str: Path to the saved plot.
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("%s does not have feature_importances_. Skipping.", model_name)
        return ""

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] if i < len(feature_names) else f"f{i}"
                    for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_features)), top_importances[::-1],
            color="#457B9D", edgecolor="white")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([f.replace("num__", "").replace("cat__", "") for f in top_features[::-1]])
    ax.set_xlabel("Feature Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    fig_file = save_path / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fig_file, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("Feature importance plot saved: %s", fig_file)
    return str(fig_file)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(
    model,
    X_sample: np.ndarray,
    model_key: str,
    max_samples: int = 1000,
):
    """
    Compute SHAP values for a sample of the dataset.

    For large datasets, computing SHAP for all rows is slow.
    We use a stratified sample of max_samples rows.

    Args:
        model: Fitted model.
        X_sample (np.ndarray): Feature matrix sample.
        model_key (str): 'logistic_regression', 'random_forest', or 'xgboost'.
        max_samples (int): Maximum samples to compute SHAP for.

    Returns:
        shap.Explanation or np.ndarray: SHAP values.
        shap.Explainer: The SHAP explainer object.
    """
    import shap

    # Use a subset to avoid long computation times
    if len(X_sample) > max_samples:
        idx = np.random.choice(len(X_sample), max_samples, replace=False)
        X_subset = X_sample[idx]
    else:
        X_subset = X_sample

    logger.info("Computing SHAP values for %d samples...", len(X_subset))

    if model_key == "xgboost":
        # TreeExplainer is fast and exact for XGBoost
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_subset)
    elif model_key == "random_forest":
        # TreeExplainer also works for Random Forest
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_subset)
        # For binary classification Random Forest, shap_values is a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 (readmitted)
    else:
        # Linear explainer for Logistic Regression
        background = shap.maskers.Independent(X_subset, max_samples=100)
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(X_subset)

    logger.info("SHAP values computed. Shape: %s", np.array(shap_values).shape)
    return shap_values, explainer, X_subset


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: list,
    model_name: str,
    save_dir: str = "models",
) -> str:
    """
    Generate a SHAP summary (beeswarm) plot.

    The summary plot shows:
        - Y-axis: Features ranked by mean absolute SHAP value
        - X-axis: SHAP value (positive = pushes toward readmission)
        - Colour: Feature value (red=high, blue=low)

    Args:
        shap_values: SHAP values array (samples × features).
        X_sample: The feature matrix used to compute SHAP.
        feature_names (list): Feature names.
        model_name (str): Model name for plot title.
        save_dir (str): Output directory.

    Returns:
        str: Path to saved plot.
    """
    import shap

    # Clean feature names for display
    clean_names = [
        n.replace("num__", "").replace("cat__", "").replace("_", " ")
        for n in feature_names[:X_sample.shape[1]]
    ]
    # Pad/trim if needed
    n_features = min(X_sample.shape[1], len(clean_names))
    X_display = X_sample[:, :n_features]
    sv_display = np.array(shap_values)[:, :n_features]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        sv_display, X_display,
        feature_names=clean_names[:n_features],
        show=False,
        max_display=15,
        plot_type="dot",
    )
    plt.title(f"SHAP Summary Plot — {model_name}\n"
              "(Academic prototype — not for clinical use)", fontsize=11)
    plt.tight_layout()

    save_path = Path(save_dir)
    fig_file = save_path / f"shap_summary_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fig_file, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved: %s", fig_file)
    return str(fig_file)


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: list,
    model_name: str,
    top_n: int = 15,
    save_dir: str = "models",
) -> str:
    """
    Plot mean absolute SHAP values as a bar chart.
    Shows global feature importance based on SHAP.

    Returns:
        str: Path to saved plot.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(mean_abs_shap)[::-1][:top_n]

    clean_names = [
        feature_names[i].replace("num__", "").replace("cat__", "").replace("_", " ")
        if i < len(feature_names) else f"f{i}"
        for i in indices
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(top_n), mean_abs_shap[indices][::-1], color="#E63946", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(clean_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} Features by SHAP Importance — {model_name}\n"
                 "(Academic prototype — not for clinical use)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    save_path = Path(save_dir)
    fig_file = save_path / f"shap_bar_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fig_file, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("SHAP bar plot saved: %s", fig_file)
    return str(fig_file)


# ─────────────────────────────────────────────────────────────────────────────
# Master explainability runner
# ─────────────────────────────────────────────────────────────────────────────

def run_explainability(
    models: dict,
    evaluation_result: dict,
    preprocessing_result: dict,
    save_dir: str = "models",
    max_shap_samples: int = 500,
) -> dict:
    """
    Run the full explainability pipeline for all models.

    Args:
        models (dict): Trained models dict.
        evaluation_result (dict): Evaluation result from ml/evaluate.py.
        preprocessing_result (dict): Preprocessing result.
        save_dir (str): Output directory.
        max_shap_samples (int): Max samples for SHAP computation.

    Returns:
        dict: Paths to all generated explainability artefacts.
    """
    logger.info("=" * 60)
    logger.info("EXPLAINABILITY STARTED")
    logger.info("=" * 60)

    feature_names = preprocessing_result.get("feature_names", [])
    X_val = preprocessing_result["X_val"]
    y_val = preprocessing_result["y_val"]
    artefacts = {}

    for key, info in models.items():
        model = info["model"]
        name = info["name"]
        logger.info("Generating explainability for: %s", name)

        # Feature importance (tree models)
        fi_path = plot_feature_importance(model, feature_names, name, save_dir=save_dir)
        if fi_path:
            artefacts[f"{key}_feature_importance"] = fi_path

        # SHAP analysis
        try:
            shap_values, explainer, X_subset = compute_shap_values(
                model, X_val, key, max_samples=max_shap_samples
            )

            summary_path = plot_shap_summary(
                shap_values, X_subset, feature_names, name, save_dir=save_dir
            )
            bar_path = plot_shap_bar(
                shap_values, feature_names, name, save_dir=save_dir
            )
            artefacts[f"{key}_shap_summary"] = summary_path
            artefacts[f"{key}_shap_bar"] = bar_path

            # Save SHAP values for the best model
            if key == evaluation_result.get("best_model_key"):
                shap_file = Path(save_dir) / "best_model_shap_values.npy"
                np.save(str(shap_file), shap_values)
                artefacts["best_shap_values"] = str(shap_file)
                logger.info("Best model SHAP values saved: %s", shap_file)

        except Exception as exc:
            logger.warning("SHAP failed for %s: %s", name, exc)

    logger.info("=" * 60)
    logger.info("EXPLAINABILITY COMPLETE — %d artefacts generated", len(artefacts))
    logger.info("=" * 60)

    return artefacts
