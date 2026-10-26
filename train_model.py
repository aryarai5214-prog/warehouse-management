"""
train_model.py - Industrial-grade training script for Warehouse Management System (WMS)
Trains:
1. Warehouse Demand Forecaster (HistGradientBoostingRegressor)
2. Stock Movement / Velocity Classifier (HistGradientBoostingClassifier)
Saves models, metrics, and metadata to `models/`.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
)

DATA_PATH = os.path.join("data", "Warehouse_and_Retail_Sales.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = "Warehouse_and_Retail_Sales.csv"

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_and_preprocess():
    print(f"Loading dataset from: {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows.")

    # 1. Clean missing values
    df["SUPPLIER"] = df["SUPPLIER"].fillna("UNKNOWN / DIRECT").astype(str).str.strip()
    df["ITEM TYPE"] = df["ITEM TYPE"].fillna("OTHER").astype(str).str.strip()
    df["RETAIL SALES"] = np.maximum(df["RETAIL SALES"].fillna(0.0), 0.0)
    df["RETAIL TRANSFERS"] = np.maximum(df["RETAIL TRANSFERS"].fillna(0.0), 0.0)
    df["WAREHOUSE SALES"] = np.maximum(df["WAREHOUSE SALES"].fillna(0.0), 0.0)

    # 2. Velocity Tier Classification Target
    def assign_tier(val):
        if val >= 20.0:
            return "Fast-Moving (A)"
        elif val >= 2.0:
            return "Moderate (B)"
        else:
            return "Slow/Low (C)"

    df["VELOCITY_TIER"] = df["WAREHOUSE SALES"].apply(assign_tier)

    # 3. Feature Engineering
    df["TOTAL_RETAIL_FLOW"] = df["RETAIL SALES"] + df["RETAIL TRANSFERS"]
    df["TRANSFER_RATIO"] = df["RETAIL TRANSFERS"] / (df["RETAIL SALES"] + 1.0)
    df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"] / 12)

    # Calculate Supplier & Category aggregated lookups
    supp_avg = df.groupby("SUPPLIER")["WAREHOUSE SALES"].mean().to_dict()
    supp_med = df.groupby("SUPPLIER")["WAREHOUSE SALES"].median().to_dict()
    supp_sku_count = df.groupby("SUPPLIER")["ITEM CODE"].nunique().to_dict()
    cat_avg = df.groupby("ITEM TYPE")["WAREHOUSE SALES"].mean().to_dict()

    global_supp_avg = float(df["WAREHOUSE SALES"].mean())
    global_cat_avg = float(df["WAREHOUSE SALES"].mean())

    df["SUPPLIER_AVG_SALES"] = df["SUPPLIER"].map(supp_avg).fillna(global_supp_avg)
    df["SUPPLIER_MED_SALES"] = df["SUPPLIER"].map(supp_med).fillna(0.0)
    df["ITEM_TYPE_AVG_SALES"] = df["ITEM TYPE"].map(cat_avg).fillna(global_cat_avg)

    # Log transforms
    df["LOG_RETAIL_SALES"] = np.log1p(df["RETAIL SALES"])
    df["LOG_RETAIL_TRANSFERS"] = np.log1p(df["RETAIL TRANSFERS"])
    df["LOG_TOTAL_RETAIL"] = np.log1p(df["TOTAL_RETAIL_FLOW"])

    feature_cols = [
        "MONTH", "MONTH_SIN", "MONTH_COS",
        "RETAIL SALES", "RETAIL TRANSFERS", "TOTAL_RETAIL_FLOW",
        "TRANSFER_RATIO", "LOG_RETAIL_SALES", "LOG_RETAIL_TRANSFERS", "LOG_TOTAL_RETAIL",
        "SUPPLIER_AVG_SALES", "SUPPLIER_MED_SALES", "ITEM_TYPE_AVG_SALES"
    ]

    metadata = {
        "feature_cols": feature_cols,
        "item_types": sorted(df["ITEM TYPE"].unique().tolist()),
        "top_suppliers": df["SUPPLIER"].value_counts().head(50).index.tolist(),
        "all_suppliers": sorted(df["SUPPLIER"].unique().tolist()),
        "supp_avg": supp_avg,
        "supp_med": supp_med,
        "cat_avg": cat_avg,
        "global_supp_avg": global_supp_avg,
        "global_cat_avg": global_cat_avg,
        "tier_labels": ["Fast-Moving (A)", "Moderate (B)", "Slow/Low (C)"]
    }

    return df, feature_cols, metadata

def train_and_evaluate():
    df, feature_cols, metadata = load_and_preprocess()

    X = df[feature_cols]
    y_reg = np.log1p(df["WAREHOUSE SALES"])
    y_clf = df["VELOCITY_TIER"]

    print("Splitting dataset into 80/20 train/test sets...")
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    print("Training Demand Forecasting Model (HistGradientBoostingRegressor)...")
    regressor = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.07,
        max_depth=9,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=42
    )
    regressor.fit(X_train, y_reg_train)

    y_reg_pred_log = regressor.predict(X_test)
    y_reg_pred = np.maximum(np.expm1(y_reg_pred_log), 0.0)
    y_reg_actual = np.expm1(y_reg_test)

    r2_log = float(r2_score(y_reg_test, y_reg_pred_log))
    mae_log = float(mean_absolute_error(y_reg_test, y_reg_pred_log))
    r2_orig = float(r2_score(y_reg_actual, y_reg_pred))
    mae_orig = float(mean_absolute_error(y_reg_actual, y_reg_pred))
    rmse_orig = float(np.sqrt(mean_squared_error(y_reg_actual, y_reg_pred)))
    med_ae_orig = float(np.median(np.abs(y_reg_actual - y_reg_pred)))

    print(f"Regressor Results:")
    print(f"  Log-scale R²: {r2_log:.4f}")
    print(f"  Log-scale MAE: {mae_log:.4f}")
    print(f"  Original-scale R²: {r2_orig:.4f}")
    print(f"  Original-scale MAE: {mae_orig:.2f} units")
    print(f"  Original-scale RMSE: {rmse_orig:.2f} units")
    print(f"  Original-scale Median AE: {med_ae_orig:.2f} units")

    print("\nTraining Inventory Velocity Classifier (HistGradientBoostingClassifier)...")
    classifier = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_depth=8,
        min_samples_leaf=25,
        random_state=42
    )
    classifier.fit(X_train, y_clf_train)

    y_clf_pred = classifier.predict(X_test)
    accuracy = float(accuracy_score(y_clf_test, y_clf_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_clf_test, y_clf_pred, average="weighted")
    cm = confusion_matrix(y_clf_test, y_clf_pred, labels=metadata["tier_labels"]).tolist()
    clf_report = classification_report(y_clf_test, y_clf_pred, output_dict=True)

    print(f"Classifier Results:")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  Weighted Precision: {p:.4f}")
    print(f"  Weighted Recall: {r:.4f}")
    print(f"  Weighted F1: {f1:.4f}")

    # Save artifacts
    print(f"\nSaving models and artifacts to `{MODELS_DIR}/`...")
    joblib.dump(regressor, os.path.join(MODELS_DIR, "demand_forecaster.pkl"))
    joblib.dump(classifier, os.path.join(MODELS_DIR, "velocity_classifier.pkl"))

    metrics = {
        "regressor": {
            "model_type": "HistGradientBoostingRegressor",
            "log_r2": round(r2_log, 4),
            "log_mae": round(mae_log, 4),
            "original_r2": round(r2_orig, 4),
            "original_mae": round(mae_orig, 2),
            "original_rmse": round(rmse_orig, 2),
            "original_median_ae": round(med_ae_orig, 2),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        },
        "classifier": {
            "model_type": "HistGradientBoostingClassifier",
            "accuracy": round(accuracy, 4),
            "weighted_precision": round(p, 4),
            "weighted_recall": round(r, 4),
            "weighted_f1": round(f1, 4),
            "confusion_matrix": cm,
            "classification_report": clf_report
        },
        "dataset_summary": {
            "total_records": len(df),
            "unique_items": int(df["ITEM CODE"].nunique()),
            "unique_suppliers": int(df["SUPPLIER"].nunique()),
            "categories": metadata["item_types"],
            "total_warehouse_sales_volume": float(df["WAREHOUSE SALES"].sum()),
            "total_retail_sales_volume": float(df["RETAIL SALES"].sum()),
            "total_retail_transfers_volume": float(df["RETAIL TRANSFERS"].sum())
        }
    }

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Create a small sample CSV for testing / download template in Streamlit
    sample_df = df.sample(n=25, random_state=42)[
        ["YEAR", "MONTH", "SUPPLIER", "ITEM CODE", "ITEM DESCRIPTION", "ITEM TYPE", "RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES"]
    ]
    sample_path = os.path.join("data", "sample_inventory.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"Generated sample template at {sample_path}")

    print("All models and statistics successfully saved!")

if __name__ == "__main__":
    train_and_evaluate()
