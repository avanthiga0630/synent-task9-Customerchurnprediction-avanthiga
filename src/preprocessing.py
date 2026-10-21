import os
import pandas as pd
import numpy as np

# Default dataset path
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Telco-Customer-Churn.csv")


def load_and_clean_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Loads and cleans the Telco Customer Churn dataset matching the exact preprocessing
    performed in the training Jupyter notebook:
    1. Converts 'TotalCharges' to numeric with error coercion
    2. Imputes missing TotalCharges values with median
    3. Removes duplicates
    """
    if not os.path.exists(path):
        # Fallback to remote if local file is missing
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(path)

    # 1. Total charges numeric coercion
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # 2. Impute with median
    median_total = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total)
    
    # 3. Drop duplicate rows
    df.drop_duplicates(inplace=True)
    
    return df


def prepare_training_features(df: pd.DataFrame):
    """
    Separates features and target, performs one-hot encoding with drop_first=True,
    and returns X, y matching the exact notebook specifications.
    """
    df_clean = df.copy()
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])
        
    X = df_clean.drop(columns=["Churn"])
    y = df_clean["Churn"].map({"Yes": 1, "No": 0})
    
    # One-hot encoding on categorical columns
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Convert boolean columns to integer (0/1)
    bool_columns = X.select_dtypes(include=["bool"]).columns
    X[bool_columns] = X[bool_columns].astype(int)
    
    return X, y


def encode_single_customer(customer_data: dict, expected_columns: list) -> pd.DataFrame:
    """
    Transforms a single customer input dictionary into a 1-row DataFrame containing
    the exact 30 dummy-encoded columns required by the trained Random Forest model.
    """
    encoded = {col: 0 for col in expected_columns}
    
    # 1. Numeric attributes
    encoded["SeniorCitizen"] = int(customer_data.get("SeniorCitizen", 0))
    encoded["tenure"] = float(customer_data.get("tenure", 0))
    encoded["MonthlyCharges"] = float(customer_data.get("MonthlyCharges", 0.0))
    encoded["TotalCharges"] = float(customer_data.get("TotalCharges", 0.0))
    
    # 2. Demographics
    if customer_data.get("gender") == "Male":
        encoded["gender_Male"] = 1
    if customer_data.get("Partner") == "Yes":
        encoded["Partner_Yes"] = 1
    if customer_data.get("Dependents") == "Yes":
        encoded["Dependents_Yes"] = 1
        
    # 3. Phone services
    if customer_data.get("PhoneService") == "Yes":
        encoded["PhoneService_Yes"] = 1
    mult = customer_data.get("MultipleLines")
    if mult == "No phone service":
        encoded["MultipleLines_No phone service"] = 1
    elif mult == "Yes":
        encoded["MultipleLines_Yes"] = 1
        
    # 4. Internet services
    inet = customer_data.get("InternetService")
    if inet == "Fiber optic":
        encoded["InternetService_Fiber optic"] = 1
    elif inet == "No":
        encoded["InternetService_No"] = 1
        
    # Subscribed add-ons
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for addon in addons:
        val = customer_data.get(addon)
        if val == "No internet service":
            col_name = f"{addon}_No internet service"
            if col_name in encoded:
                encoded[col_name] = 1
        elif val == "Yes":
            col_name = f"{addon}_Yes"
            if col_name in encoded:
                encoded[col_name] = 1
                
    # 5. Contract terms
    contract = customer_data.get("Contract")
    if contract == "One year":
        encoded["Contract_One year"] = 1
    elif contract == "Two year":
        encoded["Contract_Two year"] = 1
        
    # 6. Billing & payment
    if customer_data.get("PaperlessBilling") == "Yes":
        encoded["PaperlessBilling_Yes"] = 1
        
    pm = customer_data.get("PaymentMethod")
    if pm == "Credit card (automatic)":
        encoded["PaymentMethod_Credit card (automatic)"] = 1
    elif pm == "Electronic check":
        encoded["PaymentMethod_Electronic check"] = 1
    elif pm == "Mailed check":
        encoded["PaymentMethod_Mailed check"] = 1
        
    return pd.DataFrame([encoded])[expected_columns]


def encode_batch_dataframe(df: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    """
    Transforms an entire DataFrame of customer records into the expected 30-feature matrix.
    Safely handles column name differences and preserves feature ordering.
    """
    df_temp = df.copy()
    if "customerID" in df_temp.columns:
        df_temp = df_temp.drop(columns=["customerID"])
    if "Churn" in df_temp.columns:
        df_temp = df_temp.drop(columns=["Churn"])
        
    # Total charges numeric coercion
    if "TotalCharges" in df_temp.columns:
        df_temp["TotalCharges"] = pd.to_numeric(df_temp["TotalCharges"], errors="coerce").fillna(0.0)
        
    # One-hot encode categorical features
    cat_cols = df_temp.select_dtypes(include=["object", "string"]).columns
    df_encoded = pd.get_dummies(df_temp, columns=cat_cols, drop_first=True)
    
    # Cast boolean to int
    bool_cols = df_encoded.select_dtypes(include=["bool"]).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    
    # Ensure all expected columns exist (add 0 if missing from batch sample)
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Return reindexed to exact column order
    return df_encoded[expected_columns]
