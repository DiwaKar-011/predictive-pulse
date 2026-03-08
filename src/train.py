"""
Model training module for Predictive Pulse.

Trains multiple classifiers (Logistic Regression, Decision Tree, Random Forest,
XGBoost, LightGBM) with hyperparameter tuning via RandomizedSearchCV.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import joblib
import warnings
import time

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def get_models_and_params():
    """Define models and their hyperparameter search spaces."""
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs", "saga"],
            },
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                "criterion": ["gini", "entropy"],
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=1),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 15, 20, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        },
        "XGBoost": {
            "model": XGBClassifier(
                random_state=42,
                use_label_encoder=False,
                eval_metric="mlogloss",
                verbosity=0,
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
            },
        },
        "LightGBM": {
            "model": LGBMClassifier(random_state=42, verbose=-1, n_jobs=1),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7, -1],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "num_leaves": [15, 31, 63],
                "subsample": [0.7, 0.8, 1.0],
            },
        },
    }
    return models


def train_model(name, model, params, X_train, y_train, cv=5, n_iter=20):
    """Train a single model with RandomizedSearchCV."""
    print(f"\n  Training {name}...")
    start = time.time()

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        model,
        params,
        n_iter=min(n_iter, np.prod([len(v) for v in params.values()])),
        cv=cv_strategy,
        scoring="f1_weighted",
        n_jobs=1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)

    elapsed = time.time() - start
    print(f"    Best CV F1 (weighted): {search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")
    print(f"    Training time: {elapsed:.1f}s")

    return search.best_estimator_, search.best_score_, search.best_params_


def training_pipeline():
    """Run the full training pipeline."""
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load data
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Class distribution:\n{y_train.value_counts().sort_index().to_dict()}")

    models_config = get_models_and_params()
    results = {}

    for name, config in models_config.items():
        best_model, best_score, best_params = train_model(
            name, config["model"], config["params"], X_train, y_train
        )
        results[name] = {
            "model": best_model,
            "cv_f1_weighted": best_score,
            "best_params": best_params,
        }
        # Save each model
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(best_model, model_path)
        print(f"    Saved: {model_path}")

    # Select best model
    best_name = max(results, key=lambda k: results[k]["cv_f1_weighted"])
    best_model = results[best_name]["model"]

    print(f"\n{'='*60}")
    print("MODEL COMPARISON (CV F1-Weighted)")
    print(f"{'='*60}")
    for name, res in sorted(
        results.items(), key=lambda x: x[1]["cv_f1_weighted"], reverse=True
    ):
        marker = " \u2b50 BEST" if name == best_name else ""
        print(f"  {name}: {res['cv_f1_weighted']:.4f}{marker}")

    # Save best model
    best_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, best_path)
    joblib.dump(best_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))
    print(f"\n\u2705 Best model ({best_name}) saved to {best_path}")

    return results, best_name


if __name__ == "__main__":
    training_pipeline()
