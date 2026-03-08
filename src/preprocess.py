"""
Data generation and preprocessing module for Predictive Pulse.

Generates a synthetic hypertension dataset with realistic clinical parameters
and performs preprocessing: missing value handling, outlier detection,
encoding, scaling, and class balancing.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def generate_synthetic_data(n_samples=5000, random_state=42):
    """
    Generate a synthetic hypertension dataset with realistic distributions.
    """
    np.random.seed(random_state)

    age = np.random.normal(50, 15, n_samples).clip(18, 90).astype(int)
    gender = np.random.choice(["Male", "Female"], n_samples, p=[0.48, 0.52])
    bmi = np.random.normal(27, 5, n_samples).clip(15, 50).round(1)
    cholesterol = np.random.normal(200, 40, n_samples).clip(100, 350).round(0)
    glucose = np.random.normal(100, 25, n_samples).clip(60, 300).round(0)
    smoking = np.random.choice(
        ["Never", "Former", "Current"], n_samples, p=[0.55, 0.25, 0.20]
    )
    alcohol = np.random.choice(
        ["None", "Moderate", "Heavy"], n_samples, p=[0.40, 0.45, 0.15]
    )
    physical_activity = np.random.choice(
        ["Low", "Moderate", "High"], n_samples, p=[0.30, 0.45, 0.25]
    )
    diabetes = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    medication = np.random.choice([0, 1], n_samples, p=[0.70, 0.30])

    # Generate BP with correlations to risk factors
    systolic_base = 110 + 0.5 * (age - 30) + 2.0 * (bmi - 22)
    systolic_base += np.where(np.array(smoking) == "Current", 8, 0)
    systolic_base += np.where(np.array(alcohol) == "Heavy", 6, 0)
    systolic_base += np.where(np.array(physical_activity) == "Low", 5, 0)
    systolic_base += np.where(diabetes == 1, 10, 0)
    systolic_base += np.where(np.array(gender) == "Male", 3, 0)
    systolic = (systolic_base + np.random.normal(0, 12, n_samples)).clip(85, 210).round(0)

    diastolic_base = 0.6 * systolic + np.random.normal(-5, 8, n_samples)
    diastolic = diastolic_base.clip(50, 130).round(0)

    # Classify hypertension stages based on AHA guidelines
    stage = []
    for s, d in zip(systolic, diastolic):
        if s > 180 or d > 120:
            stage.append("Crisis")
        elif s >= 140 or d >= 90:
            stage.append("Stage 2")
        elif 130 <= s <= 139 or 80 <= d <= 89:
            stage.append("Stage 1")
        elif 120 <= s <= 129 and d < 80:
            stage.append("Elevated")
        else:
            stage.append("Normal")

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "systolic_bp": systolic.astype(int),
            "diastolic_bp": diastolic.astype(int),
            "cholesterol": cholesterol.astype(int),
            "glucose": glucose.astype(int),
            "smoking": smoking,
            "alcohol": alcohol,
            "physical_activity": physical_activity,
            "diabetes": diabetes,
            "medication": medication,
            "hypertension_stage": stage,
        }
    )

    # Introduce ~3% missing values randomly
    rng = np.random.default_rng(random_state)
    for col in ["bmi", "cholesterol", "glucose"]:
        mask = rng.random(n_samples) < 0.03
        df.loc[mask, col] = np.nan

    # Introduce a few duplicate rows
    dup_indices = rng.choice(n_samples, size=20, replace=False)
    df = pd.concat([df, df.iloc[dup_indices]], ignore_index=True)

    return df


def handle_missing_values(df):
    """Impute missing values: median for numerics."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    return df


def handle_duplicates(df):
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  Removed {before - len(df)} duplicate rows.")
    return df


def detect_and_cap_outliers(df, columns, factor=1.5):
    """Cap outliers using IQR method."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            df[col] = df[col].clip(lower, upper)
            print(f"  Capped {outliers} outliers in '{col}'.")
    return df


def encode_features(df):
    """Encode categorical features."""
    # Label encode ordinal features
    ordinal_maps = {
        "physical_activity": {"Low": 0, "Moderate": 1, "High": 2},
        "alcohol": {"None": 0, "Moderate": 1, "Heavy": 2},
        "smoking": {"Never": 0, "Former": 1, "Current": 2},
    }
    for col, mapping in ordinal_maps.items():
        df[col] = df[col].map(mapping)

    # One-hot encode nominal features
    df = pd.get_dummies(df, columns=["gender"], drop_first=True, dtype=int)

    # Encode target
    label_encoder = LabelEncoder()
    label_encoder.fit(["Normal", "Elevated", "Stage 1", "Stage 2", "Crisis"])
    df["hypertension_stage"] = label_encoder.transform(df["hypertension_stage"])

    return df, label_encoder


def scale_features(X_train, X_test):
    """Apply StandardScaler \u2013 fit on train only."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


def balance_classes(X_train, y_train):
    """Apply SMOTE to balance classes."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"  SMOTE: {len(X_train)} -> {len(X_resampled)} samples.")
    return X_resampled, y_resampled


def preprocess_pipeline():
    """Run the full preprocessing pipeline."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Step 1: Generate data
    print("Step 1: Generating synthetic dataset...")
    df = generate_synthetic_data(n_samples=5000)
    raw_path = os.path.join(RAW_DIR, "hypertension_data.csv")
    df.to_csv(raw_path, index=False)
    print(f"  Saved raw data: {raw_path} ({df.shape})")

    # Step 2: Handle missing values
    print("Step 2: Handling missing values...")
    print(f"  Missing before: {df.isnull().sum().sum()}")
    df = handle_missing_values(df)
    print(f"  Missing after: {df.isnull().sum().sum()}")

    # Step 3: Handle duplicates
    print("Step 3: Removing duplicates...")
    df = handle_duplicates(df)

    # Step 4: Cap outliers
    print("Step 4: Capping outliers...")
    outlier_cols = ["bmi", "cholesterol", "glucose", "systolic_bp", "diastolic_bp"]
    df = detect_and_cap_outliers(df, outlier_cols)

    # Step 5: Encode features
    print("Step 5: Encoding features...")
    df, label_encoder = encode_features(df)
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))

    # Step 6: Train/test split
    print("Step 6: Splitting data (80/20, stratified)...")
    X = df.drop("hypertension_stage", axis=1)
    y = df["hypertension_stage"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Step 7: Scale features
    print("Step 7: Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # Step 8: Balance classes
    print("Step 8: Balancing classes with SMOTE...")
    print(f"  Class distribution before SMOTE:\n{y_train.value_counts().to_dict()}")
    X_train_balanced, y_train_balanced = balance_classes(X_train_scaled, y_train)

    # Save processed data
    print("Step 9: Saving processed data...")
    X_train_balanced.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    y_train_balanced.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    # Also save the unscaled cleaned data for EDA
    df.to_csv(os.path.join(PROCESSED_DIR, "cleaned_data.csv"), index=False)

    feature_names = list(X.columns)
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

    print("\n\u2705 Preprocessing complete!")
    print(f"  Features: {feature_names}")
    print(f"  Classes: {list(label_encoder.classes_)}")

    return X_train_balanced, X_test_scaled, y_train_balanced, y_test


if __name__ == "__main__":
    preprocess_pipeline()
