"""
ml/predict.py
-------------
Inference helpers for the Hospital Readmission prediction model.

What this module does:
    Provides functions to:
    - Load a trained model and preprocessor from disk or MLflow
    - Accept a single patient feature dict and return a readmission risk score
    - Handle missing features gracefully

Used by:
    api/main.py — the FastAPI prediction endpoint

Disclaimer:
    This module is part of an academic prototype. Predictions must NOT be
    used for clinical decision-making.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from ml.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES

logger = get_logger(__name__)


def load_model(model_path: str = "models/xgboost.joblib"):
    """
    Load a saved model from disk.

    Args:
        model_path (str): Path to the .joblib model file.

    Returns:
        Fitted model object.

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Run ml/train.py first to train and save models."
        )
    model = joblib.load(path)
    logger.info("Model loaded: %s", model_path)
    return model


def load_preprocessor(preprocessor_path: str = "models/preprocessor.joblib"):
    """
    Load the saved preprocessing pipeline from disk.

    Args:
        preprocessor_path (str): Path to the .joblib preprocessor file.

    Returns:
        Fitted ColumnTransformer.
    """
    path = Path(preprocessor_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {preprocessor_path}\n"
            "Run ml/preprocessing.py first."
        )
    preprocessor = joblib.load(path)
    logger.info("Preprocessor loaded: %s", preprocessor_path)
    return preprocessor


def prepare_input(
    feature_dict: Dict[str, Any],
    num_features: list = None,
    cat_features: list = None,
) -> pd.DataFrame:
    """
    Convert a feature dictionary to a DataFrame for prediction.

    Handles:
    - Missing optional features (filled with NaN)
    - Column ordering consistent with training

    Args:
        feature_dict (dict): Input features from the API request.
        num_features (list): Expected numeric feature names.
        cat_features (list): Expected categorical feature names.

    Returns:
        pd.DataFrame: Single-row DataFrame ready for preprocessing.
    """
    if num_features is None:
        num_features = NUMERIC_FEATURES
    if cat_features is None:
        cat_features = CATEGORICAL_FEATURES

    all_features = num_features + cat_features
    row = {col: feature_dict.get(col, np.nan) for col in all_features}
    df = pd.DataFrame([row])
    return df


def predict_readmission(
    feature_dict: Dict[str, Any],
    model=None,
    preprocessor=None,
    model_path: str = "models/xgboost.joblib",
    preprocessor_path: str = "models/preprocessor.joblib",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Predict readmission risk for a single patient encounter.

    Args:
        feature_dict (dict): Patient features from the API.
        model: Pre-loaded model (avoids disk read on each call).
        preprocessor: Pre-loaded preprocessor.
        model_path (str): Fallback path to load model if not provided.
        preprocessor_path (str): Fallback path to load preprocessor.
        threshold (float): Classification threshold.

    Returns:
        dict: {
            'prediction': 0 or 1,
            'risk_score': float (probability of readmission),
            'risk_category': 'Low' | 'Medium' | 'High',
            'threshold': float,
            'model_version': str,
        }
    """
    if model is None:
        model = load_model(model_path)
    if preprocessor is None:
        preprocessor = load_preprocessor(preprocessor_path)

    # Prepare input
    df = prepare_input(feature_dict)

    # Preprocess
    X = preprocessor.transform(df)

    # Predict
    y_proba = model.predict_proba(X)[0][1]
    y_pred = int(y_proba >= threshold)

    # Risk category
    if y_proba < 0.20:
        risk_category = "Low"
    elif y_proba < 0.50:
        risk_category = "Medium"
    else:
        risk_category = "High"

    result = {
        "prediction": y_pred,
        "risk_score": round(float(y_proba), 4),
        "risk_category": risk_category,
        "threshold": threshold,
        "model_version": getattr(model, "__class__.__name__", "unknown"),
        "disclaimer": (
            "ACADEMIC PROTOTYPE — NOT FOR CLINICAL USE. "
            "Predictions must not be used for patient care or diagnosis."
        ),
    }

    logger.info(
        "Prediction: %d | risk_score: %.4f | category: %s",
        y_pred, y_proba, risk_category,
    )
    return result
