"""
ml/monitoring.py
----------------
Model monitoring and drift detection for the Hospital Readmission project.

What this module does:
    Monitors the production model for:
    1. Input data drift    — are incoming features drifting from training distribution?
    2. Prediction drift    — is the predicted positive rate changing?
    3. Model performance   — is AUC/recall degrading on labelled data?
    4. Subgroup performance— does performance differ by age/gender/diagnosis?
    5. Calibration         — is the predicted probability still well-calibrated?

Academic context:
    This is a monitoring DESIGN for an academic prototype. In production,
    this would run as a scheduled job (Airflow DAG) and send alerts to
    a clinical operations team.

Why monitoring matters:
    Hospital readmission patterns can change over time due to:
    - Changes in treatment protocols
    - Changes in patient population demographics
    - Seasonal variation
    - New diseases or coding changes
    If we do not monitor for these changes, the model may silently become
    inaccurate without anyone noticing.

Retraining criteria (documented here and in docs/):
    1. PSI > 0.2 for any critical feature (Population Stability Index)
    2. ROC-AUC drops > 5 percentage points from baseline
    3. Recall drops > 10 percentage points from baseline
    4. Calibration ECE (Expected Calibration Error) exceeds 0.05
    5. New labelled data of > 5,000 encounters available
    6. Significant change in feature distribution (KS test p < 0.05)

Viva questions:
    Q: What is data drift?
    A: Data drift means the statistical distribution of input features
       changes over time. For example, if the average patient age in new
       data is significantly different from the training data, the model's
       performance may degrade.

    Q: What is the Population Stability Index (PSI)?
    A: PSI measures how much a distribution has shifted between two samples.
       PSI < 0.1: No significant change
       PSI 0.1-0.2: Moderate change — monitor closely
       PSI > 0.2: Significant change — consider retraining

    Q: What does it mean for a model to be well-calibrated?
    A: It means the model's predicted probabilities match observed frequencies.
       If the model says 70% readmission risk, approximately 70% of such
       patients should actually be readmitted.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
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
# Population Stability Index (PSI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute the Population Stability Index (PSI) between two distributions.

    PSI = sum((actual% - expected%) * ln(actual% / expected%))

    Thresholds:
        PSI < 0.1  : No significant shift
        0.1–0.2   : Moderate shift
        PSI > 0.2  : Significant shift — consider retraining

    Args:
        reference (np.ndarray): Training set distribution (baseline).
        current (np.ndarray): New/production data distribution.
        n_bins (int): Number of bins for discretisation.

    Returns:
        float: PSI value.
    """
    # Discretise using training set bin edges
    _, bin_edges = np.histogram(reference, bins=n_bins)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    # Avoid division by zero or log(0)
    ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
    cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def interpret_psi(psi: float) -> str:
    """Return a human-readable interpretation of a PSI value."""
    if psi < 0.1:
        return "STABLE — No significant distribution shift"
    elif psi < 0.2:
        return "MODERATE — Monitor closely, potential drift"
    else:
        return "SIGNIFICANT — Distribution has shifted, consider retraining"


# ─────────────────────────────────────────────────────────────────────────────
# Feature drift report
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: list,
    psi_threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Compute PSI for all numeric features and flag those above the threshold.

    Args:
        reference_df (pd.DataFrame): Training data (reference distribution).
        current_df (pd.DataFrame): New production data.
        numeric_features (list): List of numeric feature column names.
        psi_threshold (float): PSI above which a feature is flagged.

    Returns:
        pd.DataFrame: Drift report with PSI values and flags.
    """
    results = []
    for col in numeric_features:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_vals = reference_df[col].dropna().values
        cur_vals = current_df[col].dropna().values

        if len(ref_vals) < 50 or len(cur_vals) < 50:
            continue

        psi = compute_psi(ref_vals, cur_vals)
        results.append({
            "feature": col,
            "psi": round(psi, 4),
            "interpretation": interpret_psi(psi),
            "flagged": psi > psi_threshold,
            "timestamp": datetime.now().isoformat(),
        })

    df = pd.DataFrame(results)
    flagged_count = df["flagged"].sum() if not df.empty else 0
    logger.info(
        "Feature drift report: %d features checked, %d flagged (PSI > %.1f)",
        len(df), flagged_count, psi_threshold,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Prediction drift
# ─────────────────────────────────────────────────────────────────────────────

def compute_prediction_drift(
    reference_predictions: np.ndarray,
    current_predictions: np.ndarray,
    threshold: float = 0.1,
) -> dict:
    """
    Detect drift in the model's predicted positive rate.

    If the model starts predicting many more (or fewer) high-risk patients
    than it did during training, this may indicate data drift or model degradation.

    Args:
        reference_predictions (np.ndarray): Training set predicted probabilities.
        current_predictions (np.ndarray): New data predicted probabilities.
        threshold (float): Absolute change in positive rate that triggers a flag.

    Returns:
        dict: Drift statistics.
    """
    ref_rate = float(np.mean(reference_predictions >= 0.5))
    cur_rate = float(np.mean(current_predictions >= 0.5))
    delta = abs(cur_rate - ref_rate)

    result = {
        "reference_positive_rate": round(ref_rate, 4),
        "current_positive_rate": round(cur_rate, 4),
        "delta": round(delta, 4),
        "flagged": delta > threshold,
        "interpretation": (
            f"Positive rate changed by {delta:.1%} "
            f"({'FLAGGED' if delta > threshold else 'OK'})"
        ),
        "timestamp": datetime.now().isoformat(),
    }
    logger.info("Prediction drift: delta=%.4f, flagged=%s", delta, result["flagged"])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Performance monitoring (requires labelled data)
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance_report(
    model,
    X_new: np.ndarray,
    y_new: np.ndarray,
    baseline_metrics: dict,
    performance_threshold: float = 0.05,
) -> dict:
    """
    Compare current model performance against baseline metrics.
    Flags degradation if performance drops beyond the threshold.

    Args:
        model: Current production model.
        X_new (np.ndarray): New labelled feature matrix.
        y_new (np.ndarray): True labels for new data.
        baseline_metrics (dict): Metrics from original evaluation (training time).
        performance_threshold (float): Degradation that triggers a flag.

    Returns:
        dict: Performance comparison report.
    """
    from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score

    y_proba = model.predict_proba(X_new)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    current_metrics = {
        "recall": float(recall_score(y_new, y_pred, zero_division=0)),
        "precision": float(precision_score(y_new, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_new, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_new, y_proba)),
    }

    flags = []
    for metric_name, current_val in current_metrics.items():
        baseline_val = baseline_metrics.get(metric_name, 0)
        degradation = baseline_val - current_val
        if degradation > performance_threshold:
            flags.append(
                f"{metric_name}: dropped {degradation:.3f} "
                f"(baseline={baseline_val:.3f} → current={current_val:.3f})"
            )

    report = {
        "baseline_metrics": baseline_metrics,
        "current_metrics": {k: round(v, 4) for k, v in current_metrics.items()},
        "degradation_flags": flags,
        "any_degradation": len(flags) > 0,
        "sample_size": len(y_new),
        "timestamp": datetime.now().isoformat(),
    }

    if flags:
        logger.warning("Performance degradation detected: %s", flags)
    else:
        logger.info("Performance monitoring: No significant degradation detected")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Subgroup performance
# ─────────────────────────────────────────────────────────────────────────────

def compute_subgroup_performance(
    model,
    X_new: np.ndarray,
    y_new: np.ndarray,
    subgroup_labels: pd.Series,
    subgroup_name: str = "age_group",
) -> pd.DataFrame:
    """
    Evaluate model performance separately for each demographic subgroup.

    This is important for detecting performance disparities across
    patient groups — e.g. the model may perform well on average but
    poorly for elderly patients or specific ethnic groups.

    Args:
        model: Fitted model.
        X_new: Feature matrix.
        y_new: True labels.
        subgroup_labels (pd.Series): Subgroup assignment for each row.
        subgroup_name (str): Name of the subgroup variable.

    Returns:
        pd.DataFrame: Per-subgroup metrics.
    """
    from sklearn.metrics import roc_auc_score, recall_score, f1_score

    y_proba = model.predict_proba(X_new)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    results = []
    for group in subgroup_labels.unique():
        mask = (subgroup_labels == group).values
        if mask.sum() < 30:  # Skip very small subgroups
            continue

        sub_y = y_new[mask]
        sub_pred = y_pred[mask]
        sub_proba = y_proba[mask]

        if sub_y.sum() < 5:  # Need positive examples for AUC
            continue

        results.append({
            "subgroup": group,
            "subgroup_name": subgroup_name,
            "n_samples": int(mask.sum()),
            "recall": round(float(recall_score(sub_y, sub_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(sub_y, sub_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(sub_y, sub_proba)), 4),
            "positive_rate": round(float(sub_y.mean()), 4),
        })

    df = pd.DataFrame(results)
    logger.info("Subgroup performance (%s): %d groups evaluated", subgroup_name, len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Retraining criteria
# ─────────────────────────────────────────────────────────────────────────────

RETRAINING_CRITERIA = """
# Hospital Readmission Model — Retraining Criteria
# =================================================
# Academic Prototype — Data Engineering and MLOps Course

## When to Retrain

### Automatic Triggers (any one of these should trigger retraining):

1. **Data Drift (PSI)**
   - PSI > 0.20 for any of the top-5 most important features
   - (Top features: prior_admission_count, length_of_stay, num_medications,
      age_midpoint, number_diagnoses)

2. **Performance Degradation**
   - ROC-AUC drops more than 5 percentage points from baseline
   - Recall drops more than 10 percentage points from baseline
   - Brier score increases by more than 0.05 from baseline

3. **Calibration Degradation**
   - Expected Calibration Error (ECE) exceeds 0.05
   - Reliability diagram shows systematic over- or under-prediction

4. **New Data Availability**
   - More than 5,000 new labelled encounters are available
   - Data from a new time period (e.g. post-COVID patterns)

### Manual Triggers (require clinical/domain expert review):

5. **Protocol Changes**
   - Significant change in hospital discharge or admission policies
   - New treatment guidelines that affect readmission patterns

6. **Population Shift**
   - Significant demographic change in patient population
   - New disease patterns or coding changes (e.g. new ICD codes)

7. **Subgroup Performance Failure**
   - Recall drops below 0.30 for any major demographic subgroup
   - Performance gap between demographic groups exceeds 15 percentage points

## Retraining Process

1. Collect new labelled data (minimum 5,000 encounters with outcomes)
2. Run full ETL pipeline on new data
3. Run validation checks (data_quality.py)
4. Re-run ml/preprocessing.py (fit new preprocessor on new training data)
5. Re-train all three models (ml/train.py)
6. Re-evaluate (ml/evaluate.py) — ensure recall ≥ baseline
7. Log new experiment to MLflow
8. Compare new vs old model metrics
9. If improved: register new version in MLflow Model Registry
10. Deploy new model to API (update models/ directory, restart API)
11. Document changes in docs/model_card.md

## Model Lifecycle

Data Collection
     ↓
ETL Pipeline (Airflow DAG)
     ↓
Feature Engineering
     ↓
Model Training (3 algorithms)
     ↓
Evaluation (Recall / F1 / AUC / Calibration)
     ↓
MLflow Experiment Tracking
     ↓
Model Registry (Staging)
     ↓
Deployment (FastAPI)
     ↓
Monitoring (PSI / AUC / Recall / Calibration)
     ↓
Retraining Triggered? → Back to Data Collection

## Academic Disclaimer

This retraining design is for ACADEMIC PURPOSES ONLY.
In a real clinical setting, model updates would require:
- Regulatory approval (e.g. FDA clearance for clinical decision support)
- Clinical validation by qualified medical professionals
- Ethics review for bias and fairness
- Legal review for patient privacy compliance
"""


def save_retraining_criteria(output_dir: str = "docs") -> str:
    """Save the retraining criteria documentation to a file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "retraining_criteria.md"
    output_file.write_text(RETRAINING_CRITERIA, encoding="utf-8")
    logger.info("Retraining criteria saved: %s", output_file)
    return str(output_file)


# ─────────────────────────────────────────────────────────────────────────────
# Save monitoring report
# ─────────────────────────────────────────────────────────────────────────────

def save_monitoring_report(
    drift_report: pd.DataFrame,
    prediction_drift: dict,
    performance_report: dict,
    output_dir: str = "data/logs",
) -> str:
    """Save a full monitoring report to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    report_file = output_path / f"monitoring_report_{today}.json"

    report = {
        "generated_at": datetime.now().isoformat(),
        "feature_drift_summary": drift_report.to_dict(orient="records") if not drift_report.empty else [],
        "prediction_drift": prediction_drift,
        "performance_report": performance_report,
        "academic_disclaimer": (
            "This monitoring report is generated by an academic prototype. "
            "Not for clinical use."
        ),
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Monitoring report saved: %s", report_file)
    return str(report_file)


if __name__ == "__main__":
    # Save retraining criteria doc
    path = save_retraining_criteria()
    print(f"Retraining criteria saved to: {path}")
    print(RETRAINING_CRITERIA[:500] + "...")
