"""
Feature engineering and selection module for Predictive Pulse.

Creates derived features (pulse pressure, BMI categories, age groups)
and performs feature selection using SelectKBest and Random Forest importance.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def create_derived_features(df):
    """Create pulse pressure, BMI categories, and age group bins."""
    # Pulse pressure
    if "systolic_bp" in df.columns and "diastolic_bp" in df.columns:
        df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]

    # BMI category bins
    if "bmi" in df.columns:
        df["bmi_category"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25, 30, 100],
            labels=[0, 1, 2, 3],  # Underweight, Normal, Overweight, Obese
        ).astype(float)

    # Age group bins
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 30, 45, 60, 100],
            labels=[0, 1, 2, 3],  # Young, Middle, Senior, Elderly
        ).astype(float)

    return df


def select_features_kbest(X, y, k=10):
    """Select top k features using ANOVA F-value."""
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    selector.fit(X, y)
    scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
    selected = scores.head(k).index.tolist()
    print(f"  SelectKBest top {k} features:")
    for feat, score in scores.head(k).items():
        print(f"    {feat}: {score:.2f}")
    return selected, scores


def select_features_rf_importance(X, y, threshold=0.02):
    """Select features using Random Forest feature importance."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    selected = importance[importance >= threshold].index.tolist()
    print(f"\n  Random Forest feature importance (threshold={threshold}):")
    for feat, imp in importance.items():
        marker = "\u2713" if imp >= threshold else "\u2717"
        print(f"    {marker} {feat}: {imp:.4f}")
    return selected, importance


def feature_engineering_pipeline():
    """Run feature engineering and selection."""
    print("=" * 60)
    print("FEATURE ENGINEERING & SELECTION")
    print("=" * 60)

    # Load processed data
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))

    print(f"\nOriginal features ({X_train.shape[1]}): {list(X_train.columns)}")

    # Step 1: Create derived features
    print("\nStep 1: Creating derived features...")
    X_train = create_derived_features(X_train)
    X_test = create_derived_features(X_test)

    # Fill any NaN from binning
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_test.median())

    print(f"  Features after engineering ({X_train.shape[1]}): {list(X_train.columns)}")

    # Step 2: Feature selection
    print("\nStep 2: Feature selection with SelectKBest...")
    kbest_features, kbest_scores = select_features_kbest(X_train, y_train, k=12)

    print("\nStep 3: Feature selection with Random Forest...")
    rf_features, rf_importance = select_features_rf_importance(X_train, y_train)

    # Final feature set: union of both methods
    final_features = list(set(kbest_features) | set(rf_features))
    final_features = [f for f in X_train.columns if f in final_features]
    print(f"\nFinal selected features ({len(final_features)}): {final_features}")

    # Save with selected features
    X_train_final = X_train[final_features]
    X_test_final = X_test[final_features]

    X_train_final.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test_final.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)

    # Save feature list and importances
    joblib.dump(final_features, os.path.join(MODELS_DIR, "feature_names.pkl"))
    joblib.dump(rf_importance, os.path.join(MODELS_DIR, "feature_importance.pkl"))

    print("\n\u2705 Feature engineering complete!")
    return X_train_final, X_test_final, y_train


if __name__ == "__main__":
    feature_engineering_pipeline()
