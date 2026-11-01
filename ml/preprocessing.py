"""
ml/preprocessing.py
-------------------
Machine Learning preprocessing pipeline for the Hospital Readmission project.

What this module does:
    Loads the curated dataset (from PostgreSQL or CSV), selects features,
    handles missing values, encodes categoricals, scales numerics, and
    produces clean train/validation/test splits ready for model training.

Why separate ML preprocessing from ETL preprocessing?
    The ETL cleaning (src/preprocessing/) produces a general-purpose
    cleaned dataset.  This ML preprocessing applies model-specific
    transformations (one-hot encoding, StandardScaler) that would be
    inappropriate to apply to the analytical database.  Keeping them
    separate prevents data leakage and allows the ETL layer to serve
    dashboards independently.

Data leakage prevention:
    - The StandardScaler and imputer are fit ONLY on the training set
    - They are saved as artefacts and applied to validation/test sets
    - The test set is never touched until final evaluation

Train/Val/Test split:
    The UCI dataset has no timestamp column, so we use a random split
    stratified by the target variable (readmission_binary) to preserve
    class proportions.  If a temporal column were available, a chronological
    split would be preferred and is documented as such.

Viva questions:
    Q: What is data leakage and how do you prevent it?
    A: Data leakage is when information from the test set accidentally
       influences the model during training. We prevent it by fitting
       scalers and imputers only on training data, then applying them
       to validation and test sets.

    Q: Why stratify the train/test split?
    A: The readmission class is imbalanced (~11% positive). Without
       stratification, the test set might have a very different class
       ratio, making evaluation misleading.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_curated_data(
    config: dict,
    use_postgres: bool = True,
) -> pd.DataFrame:
    """
    Load the curated dataset from PostgreSQL or the curated CSV fallback.

    Args:
        config (dict): Project configuration.
        use_postgres (bool): Try PostgreSQL first. Falls back to CSV.

    Returns:
        pd.DataFrame: Curated dataset with all engineered features.
    """
    if use_postgres:
        try:
            from src.database.connection import get_engine
            from sqlalchemy import text
            engine = get_engine()
            sql = "SELECT * FROM curated.encounters LIMIT 200000"
            with engine.connect() as conn:
                df = pd.read_sql(text(sql), conn)
            logger.info("Loaded %d rows from PostgreSQL curated.encounters", len(df))
            return df
        except Exception as exc:
            logger.warning("PostgreSQL unavailable (%s). Falling back to CSV.", exc)

    # CSV fallback
    csv_path = Path(config["paths"]["data_curated"]) / "diabetic_curated.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Curated dataset not found: {csv_path}\n"
            "Run the ETL pipeline first: python src/ingestion/ingest.py"
        )
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("Loaded %d rows from CSV: %s", len(df), csv_path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature selection
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "diagnosis_count",
    "medication_count",
    "prior_admission_count",
    "age_midpoint",
    "total_service_utilisation",
]

CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "a1cresult",
    "insulin",
    "diabetesmed",
    "primary_diagnosis_category",
]

TARGET_COLUMN = "readmission_binary"

# Features to exclude from ML (identifiers, raw labels, redundant)
EXCLUDE_COLUMNS = [
    "curated_id", "pipeline_run_id", "created_at",
    "encounter_surrogate_id", "patient_surrogate_id",
    "encounter_id", "patient_nbr",
    "readmitted",           # Raw label — would be data leakage
    "readmission_binary",   # Target — handled separately
    "diag_1", "diag_2", "diag_3",  # Use primary_diagnosis_category instead
]


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select features (X) and target (y) from the curated DataFrame.

    Args:
        df (pd.DataFrame): Curated dataset.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X features, y target).
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            "Run feature engineering step first."
        )

    y = df[TARGET_COLUMN].astype(int)

    # Determine which feature columns actually exist in this dataset
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    feature_cols = num_cols + cat_cols

    X = df[feature_cols].copy()

    logger.info(
        "Feature selection — numeric: %d, categorical: %d, total: %d",
        len(num_cols), len(cat_cols), len(feature_cols),
    )
    logger.info(
        "Target distribution — positive: %d (%.1f%%), negative: %d",
        y.sum(), y.mean() * 100, (y == 0).sum(),
    )

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Train / Validation / Test split
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Split data into train / validation / test sets.

    Strategy:
        Stratified random split (no timestamp available in UCI dataset).
        If the dataset had a temporal column, we would sort by encounter
        date and split chronologically to simulate a real deployment scenario.

        Rationale: Stratify preserves the ~11% positive class ratio in
        all splits, preventing misleading evaluation metrics.

    Split sizes (default):
        Train: 70%  (~71,236 rows from 101,766)
        Val:   10%  (~10,177 rows)
        Test:  20%  (~20,353 rows)

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        test_size (float): Fraction for test set.
        val_size (float): Fraction of remaining data for validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # First split: separate test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Second split: separate validation from training
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction,
        stratify=y_trainval,
        random_state=random_state,
    )

    logger.info(
        "Data split — train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    logger.info(
        "Class balance — train: %.1f%% | val: %.1f%% | test: %.1f%%",
        y_train.mean() * 100, y_val.mean() * 100, y_test.mean() * 100,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline (scikit-learn ColumnTransformer)
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessor(
    numeric_features: list,
    categorical_features: list,
) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer for ML preprocessing.

    Numeric pipeline:
        1. SimpleImputer (median) — handles missing numeric values
        2. StandardScaler — normalises to zero mean, unit variance

    Categorical pipeline:
        1. SimpleImputer (constant 'Unknown') — handles missing categoricals
        2. OneHotEncoder — converts categories to binary dummy variables
           (handle_unknown='ignore' for categories unseen at training time)

    Args:
        numeric_features (list): Names of numeric columns.
        categorical_features (list): Names of categorical columns.

    Returns:
        ColumnTransformer: Unfitted preprocessor.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",   # Drop any columns not listed
        verbose_feature_names_out=True,
    )

    logger.info(
        "Preprocessor built — %d numeric features, %d categorical features",
        len(numeric_features), len(categorical_features),
    )
    return preprocessor


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    save_dir: str = "models",
) -> ColumnTransformer:
    """
    Fit the preprocessor on training data ONLY and save it.

    IMPORTANT: The preprocessor is fitted only on training data.
    Applying it to validation/test data prevents data leakage.

    Args:
        preprocessor (ColumnTransformer): Unfitted preprocessor.
        X_train (pd.DataFrame): Training features.
        save_dir (str): Directory to save the fitted preprocessor.

    Returns:
        ColumnTransformer: Fitted preprocessor.
    """
    logger.info("Fitting preprocessor on training data (%d rows)...", len(X_train))
    preprocessor.fit(X_train)

    # Save fitted preprocessor
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    preprocessor_file = save_path / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_file)
    logger.info("Preprocessor saved: %s", preprocessor_file)

    return preprocessor


def get_feature_names_after_preprocessing(
    preprocessor: ColumnTransformer,
) -> list:
    """
    Extract feature names after ColumnTransformer transformation.
    Useful for SHAP and feature importance.

    Returns:
        list: Feature names corresponding to the transformed columns.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"feature_{i}" for i in range(1000)]


# ─────────────────────────────────────────────────────────────────────────────
# Master preprocessing function
# ─────────────────────────────────────────────────────────────────────────────

def run_ml_preprocessing(
    config_path: str = "config/config.yaml",
    use_postgres: bool = True,
    save_dir: str = "models",
) -> dict:
    """
    Run the full ML preprocessing pipeline.

    Steps:
        1. Load curated data
        2. Select features and target
        3. Split into train/val/test
        4. Build and fit preprocessor
        5. Transform all splits
        6. Save preprocessor

    Args:
        config_path (str): Path to config.yaml.
        use_postgres (bool): Load from PostgreSQL (falls back to CSV).
        save_dir (str): Directory for saving models/preprocessor.

    Returns:
        dict: Contains all split arrays and metadata.
    """
    config = load_config(config_path)

    logger.info("=" * 60)
    logger.info("ML PREPROCESSING STARTED")
    logger.info("=" * 60)

    # Step 1: Load data
    df = load_curated_data(config, use_postgres=use_postgres)

    # Step 2: Select features
    X, y = select_features(df)

    # Determine which features actually exist
    num_features = [c for c in NUMERIC_FEATURES if c in X.columns]
    cat_features = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    # Step 3: Split
    ml_config = config.get("ml", {})
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        test_size=ml_config.get("test_size", 0.20),
        val_size=ml_config.get("val_size", 0.10),
        random_state=ml_config.get("random_state", 42),
    )

    # Step 4: Build and fit preprocessor
    preprocessor = build_preprocessor(num_features, cat_features)
    preprocessor = fit_preprocessor(preprocessor, X_train, save_dir=save_dir)

    # Step 5: Transform splits
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = get_feature_names_after_preprocessing(preprocessor)
    logger.info("Preprocessing complete. Output features: %d", len(feature_names))

    logger.info("=" * 60)
    logger.info("ML PREPROCESSING COMPLETE")
    logger.info("=" * 60)

    return {
        "X_train": X_train_proc,
        "X_val": X_val_proc,
        "X_test": X_test_proc,
        "y_train": y_train.values,
        "y_val": y_val.values,
        "y_test": y_test.values,
        "X_train_raw": X_train,
        "X_val_raw": X_val,
        "X_test_raw": X_test,
        "feature_names": feature_names,
        "num_features": num_features,
        "cat_features": cat_features,
        "preprocessor_path": str(Path(save_dir) / "preprocessor.joblib"),
        "class_weights": {0: 1, 1: int((y == 0).sum() / max(y.sum(), 1))},
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "positive_rate": float(y.mean()),
    }


if __name__ == "__main__":
    result = run_ml_preprocessing(use_postgres=False)
    print("\n=== ML PREPROCESSING RESULT ===")
    print(f"  Train size   : {result['train_size']}")
    print(f"  Val size     : {result['val_size']}")
    print(f"  Test size    : {result['test_size']}")
    print(f"  Feature count: {len(result['feature_names'])}")
    print(f"  Positive rate: {result['positive_rate']:.1%}")
    print("================================\n")
