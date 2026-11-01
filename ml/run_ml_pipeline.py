"""
run_ml_pipeline.py
------------------
Master script that runs the complete Part 2 ML pipeline end-to-end.

What this script does:
    Executes all ML phases in sequence:
        1. Preprocessing     (load data, split, fit preprocessor)
        2. Training          (Logistic Regression, Random Forest, XGBoost)
        3. Evaluation        (metrics, plots, comparison table)
        4. MLflow tracking   (log runs, register best model)
        5. Explainability    (SHAP values, feature importance plots)

Prerequisites:
    - Part 1 ETL pipeline must have been run (curated data must exist)
    - data/curated/diabetic_curated.csv OR PostgreSQL must be available
    - mlflow server must be running (or SQLite backend will be used)

Run:
    python ml/run_ml_pipeline.py

After running:
    - Models saved to models/
    - Plots saved to models/
    - MLflow experiments at http://localhost:5000 (or mlruns/)
    - API ready: uvicorn api.main:app --port 8000

Viva questions:
    Q: How do you run the full ML pipeline?
    A: python ml/run_ml_pipeline.py — this script calls all ML modules
       in sequence, from preprocessing through to model registration.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("HOSPITAL READMISSION ML PIPELINE — STARTING")
    logger.info("=" * 70)

    # ── Phase 1: Preprocessing ────────────────────────────────────────────────
    logger.info("Phase 1: ML Preprocessing...")
    from ml.preprocessing import run_ml_preprocessing
    prep_result = run_ml_preprocessing(use_postgres=False)
    logger.info(
        "Preprocessing complete — train=%d, val=%d, test=%d, features=%d",
        prep_result["train_size"],
        prep_result["val_size"],
        prep_result["test_size"],
        len(prep_result["feature_names"]),
    )

    # ── Phase 2: Model Training ───────────────────────────────────────────────
    logger.info("Phase 2: Model Training...")
    from ml.train import run_training
    models = run_training(prep_result)
    logger.info("Training complete — models: %s", list(models.keys()))

    # ── Phase 3: Model Evaluation ─────────────────────────────────────────────
    logger.info("Phase 3: Model Evaluation...")
    from ml.evaluate import run_evaluation
    eval_result = run_evaluation(models, prep_result)
    logger.info("Evaluation complete — best model: %s", eval_result["best_model_key"])

    # Print comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE (Validation Set)")
    print("=" * 70)
    print(eval_result["comparison_df"].to_string(index=False))
    print("\nBest model (by recall):", eval_result["best_val_metrics"]["model_name"])
    print("Test set results:", eval_result["test_metrics"])
    print("=" * 70 + "\n")

    # ── Phase 4: MLflow Tracking ──────────────────────────────────────────────
    logger.info("Phase 4: MLflow Experiment Tracking...")
    try:
        from ml.model_registry import run_mlflow_tracking
        mlflow_result = run_mlflow_tracking(models, eval_result, prep_result)
        logger.info("MLflow tracking complete — registered: %s",
                    mlflow_result["registered_model"])
    except Exception as exc:
        logger.warning("MLflow tracking failed (non-fatal): %s", exc)
        logger.info("To use MLflow: mlflow server --host 0.0.0.0 --port 5000")
        mlflow_result = {}

    # ── Phase 5: Explainability ───────────────────────────────────────────────
    logger.info("Phase 5: SHAP Explainability...")
    try:
        from ml.explainability import run_explainability
        explain_result = run_explainability(
            models, eval_result, prep_result,
            max_shap_samples=300,
        )
        logger.info("Explainability complete — %d artefacts", len(explain_result))
    except Exception as exc:
        logger.warning("Explainability failed (non-fatal): %s", exc)
        explain_result = {}

    # ── Phase 6: Save retraining criteria ─────────────────────────────────────
    from ml.monitoring import save_retraining_criteria
    save_retraining_criteria()

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("ML PIPELINE COMPLETE")
    logger.info("=" * 70)

    print("\n=== ML PIPELINE SUMMARY ===")
    print(f"  Models trained      : {list(models.keys())}")
    print(f"  Best model          : {eval_result['best_model_key']}")
    best = eval_result["best_val_metrics"]
    print(f"  Recall (val)        : {best.get('recall', 'N/A')}")
    print(f"  ROC-AUC (val)       : {best.get('roc_auc', 'N/A')}")
    print(f"  Models saved to     : models/")
    print(f"  Plots saved to      : models/")
    print(f"  MLflow experiment   : {mlflow_result.get('experiment_name', 'N/A')}")
    print("")
    print("  Next steps:")
    print("    1. Open MLflow: mlflow server --port 5000")
    print("    2. Start API:   uvicorn api.main:app --port 8000")
    print("    3. Open docs:   http://localhost:8000/docs")
    print("=" * 35 + "\n")


if __name__ == "__main__":
    main()
