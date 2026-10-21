import os
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

from src.preprocessing import encode_single_customer, encode_batch_dataframe

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "churn_model_bundle.pkl")


def load_model_bundle():
    """
    Safely loads the trained Random Forest model bundle and its feature columns.
    Returns (model, feature_columns, active_path, is_loaded).
    """
    if os.path.exists(MODEL_PATH):
        try:
            bundle = joblib.load(MODEL_PATH)
            if isinstance(bundle, dict):
                model = bundle.get("model")
                feature_columns = bundle.get("feature_columns", [])
            else:
                model = bundle
                feature_columns = []
            return model, feature_columns, MODEL_PATH, True
        except Exception:
            pass
    return None, [], "", False


def get_risk_classification(prob: float):
    """
    Classifies customer churn probability into enterprise risk tiers based on thresholds:
    - Low Risk: prob < 0.30 (< 30%)
    - Medium Risk: 0.30 <= prob < 0.60 (30% to < 60%)
    - High Risk: prob >= 0.60 (>= 60%)
    """
    if prob > 1.0:
        prob = prob / 100.0

    if prob >= 0.60:
        return {
            "tier": "High Risk",
            "code": "high",
            "color": "#EF4444",
            "bg_color": "#FEF2F2",
            "border_color": "#FCA5A5",
            "badge_class": "badge-high",
            "summary": "Customer exhibits strong churn signals. Urgent proactive retention required."
        }
    elif prob >= 0.30:
        return {
            "tier": "Medium Risk",
            "code": "medium",
            "color": "#F59E0B",
            "bg_color": "#FFFBEB",
            "border_color": "#FCD34D",
            "badge_class": "badge-medium",
            "summary": "Customer displays moderate attrition risk. Targeted engagement recommended."
        }
    else:
        return {
            "tier": "Low Risk",
            "code": "low",
            "color": "#10B981",
            "bg_color": "#ECFDF5",
            "border_color": "#6EE7B7",
            "badge_class": "badge-low",
            "summary": "Customer demonstrates high loyalty. Nurture with advocacy and satisfaction perks."
        }


def predict_single_customer(customer_data: dict, model, feature_columns: list) -> tuple[float, dict]:
    """
    Generates real-time churn prediction and risk categorization for a single customer.
    Returns (churn_probability, risk_info).
    """
    if model is None or not feature_columns:
        raise ValueError("Model or feature columns not loaded.")

    X_encoded = encode_single_customer(customer_data, feature_columns)
    
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_encoded)[0][1])
    else:
        pred = model.predict(X_encoded)[0]
        prob = 1.0 if pred == 1 else 0.0

    risk_info = get_risk_classification(prob)
    return prob, risk_info


def score_customer_dataset(df_raw: pd.DataFrame, model, feature_columns: list) -> pd.DataFrame:
    """
    Scores an entire customer DataFrame with the trained model.
    Adds 'Predicted_Probability', 'Predicted_Churn_Rate_%', 'Risk_Tier', and 'Predicted_Class'.
    """
    df_scored = df_raw.copy()
    X_batch = encode_batch_dataframe(df_raw, feature_columns)
    
    probs = model.predict_proba(X_batch)[:, 1]
    
    df_scored["Predicted_Probability"] = np.round(probs, 4)
    df_scored["Predicted_Churn_Rate_%"] = np.round(probs * 100, 2)
    
    tiers = [get_risk_classification(p)["tier"] for p in probs]
    df_scored["Risk_Tier"] = tiers
    df_scored["Predicted_Class"] = (probs >= 0.50).astype(int)
    
    return df_scored


def get_actual_model_benchmarks():
    """
    Returns exact evaluation metrics computed in the Jupyter Notebook
    on the Telco test set (test_size=0.2, random_state=42, stratify=y, 1409 test samples).
    """
    return {
        "Logistic Regression": {
            "accuracy": 0.806955,
            "precision": 0.658385,
            "recall": 0.566845,
            "f1_score": 0.609195,
            "roc_auc": 0.841585,
            "confusion_matrix": [[925, 110], [162, 212]],
            "selected": False,
            "notes": "Fast linear baseline with highest ROC-AUC, but assumes linear log-odds across all customer service interactions."
        },
        "Random Forest": {
            "accuracy": 0.789922,
            "precision": 0.633562,
            "recall": 0.494652,
            "f1_score": 0.555556,
            "roc_auc": 0.826493,
            "confusion_matrix": [[928, 107], [189, 185]],
            "selected": True,
            "notes": "Selected production model. Robust ensemble of 200 decision trees handling complex non-linear feature interactions without requiring input feature scaling."
        },
        "Decision Tree": {
            "accuracy": 0.741661,
            "precision": 0.513889,
            "recall": 0.494652,
            "f1_score": 0.504087,
            "roc_auc": 0.662297,
            "confusion_matrix": [[860, 175], [189, 185]],
            "selected": False,
            "notes": "Single decision tree baseline. High variance and susceptible to overfitting on customer subsets."
        },
        "K-Nearest Neighbors": {
            "accuracy": 0.747339,
            "precision": 0.525281,
            "recall": 0.500000,
            "f1_score": 0.512329,
            "roc_auc": 0.771604,
            "confusion_matrix": [[866, 169], [187, 187]],
            "selected": False,
            "notes": "Instance-based learner (k=5). Suffers from the curse of dimensionality across 30 one-hot encoded customer attributes."
        }
    }
