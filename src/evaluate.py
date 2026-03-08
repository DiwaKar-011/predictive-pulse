"""
Model evaluation module for Predictive Pulse.

Generates predictions, computes metrics (accuracy, precision, recall, F1,
confusion matrix, ROC-AUC), compares models, and produces visualizations.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DOCS_DIR = os.path.join(BASE_DIR, "docs")


def load_test_data():
    """Load test data."""
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()
    return X_test, y_test


def evaluate_model(model, X_test, y_test, model_name, label_encoder):
    """Evaluate a single model and return metrics dict."""
    y_pred = model.predict(X_test)

    # Handle feature mismatch (test may not have engineered features)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # ROC-AUC (one-vs-rest)
    classes = np.unique(np.concatenate([y_test.values, y_pred]))
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            y_test_bin = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
            # Ensure shapes match
            if y_proba.shape[1] == y_test_bin.shape[1]:
                roc_auc = roc_auc_score(
                    y_test_bin, y_proba, multi_class="ovr", average="weighted"
                )
            else:
                roc_auc = None
        else:
            roc_auc = None
    except Exception:
        roc_auc = None

    metrics = {
        "model": model_name,
        "accuracy": acc,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "f1_weighted": f1,
        "roc_auc_weighted": roc_auc,
    }
    return metrics, y_pred


def plot_confusion_matrix(y_test, y_pred, label_encoder, model_name, save_dir):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    class_names = label_encoder.classes_

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix \u2014 {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_model_comparison(results_df, save_dir):
    """Bar chart comparing model metrics."""
    metrics_to_plot = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    available = [m for m in metrics_to_plot if m in results_df.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    results_df.set_index("model")[available].plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def evaluation_pipeline():
    """Run full evaluation pipeline."""
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    os.makedirs(DOCS_DIR, exist_ok=True)

    X_test, y_test = load_test_data()
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

    # Align test features
    for col in feature_names:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_names]

    model_files = [
        "LogisticRegression",
        "DecisionTree",
        "RandomForest",
        "XGBoost",
        "LightGBM",
    ]

    all_results = []
    for name in model_files:
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if not os.path.exists(model_path):
            print(f"  Skipping {name} (not found)")
            continue

        model = joblib.load(model_path)
        print(f"\nEvaluating {name}...")

        metrics, y_pred = evaluate_model(model, X_test, y_test, name, label_encoder)
        all_results.append(metrics)

        # Print classification report
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=label_encoder.classes_,
                zero_division=0,
            )
        )

        # Confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_pred, label_encoder, name, DOCS_DIR)
        print(f"  Confusion matrix saved: {cm_path}")

    # Results summary table
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("f1_weighted", ascending=False).reset_index(
        drop=True
    )

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    # Save comparison chart
    comp_path = plot_model_comparison(results_df, DOCS_DIR)
    print(f"\nModel comparison chart: {comp_path}")

    # Save results table
    results_df.to_csv(os.path.join(DOCS_DIR, "model_results.csv"), index=False)

    # Best model analysis
    best = results_df.iloc[0]
    print(f"\n\U0001f3c6 Best model: {best['model']}")
    print(f"   Accuracy: {best['accuracy']:.4f}")
    print(f"   F1 (weighted): {best['f1_weighted']:.4f}")
    if best["roc_auc_weighted"] is not None:
        print(f"   ROC-AUC (weighted): {best['roc_auc_weighted']:.4f}")

    # Misclassification analysis
    best_model = joblib.load(os.path.join(MODELS_DIR, f"{best['model']}.pkl"))
    y_pred_best = best_model.predict(X_test)
    misclassified = X_test[y_test != y_pred_best].copy()
    misclassified["true_label"] = y_test[y_test != y_pred_best]
    misclassified["predicted"] = y_pred_best[y_test != y_pred_best]

    print(f"\n  Misclassified samples: {len(misclassified)} / {len(y_test)}")
    if len(misclassified) > 0:
        print("  Most common misclassification pairs:")
        pairs = (
            misclassified.groupby(["true_label", "predicted"])
            .size()
            .sort_values(ascending=False)
            .head(5)
        )
        for (true, pred), count in pairs.items():
            true_name = label_encoder.classes_[true]
            pred_name = label_encoder.classes_[pred]
            print(f"    {true_name} \u2192 {pred_name}: {count}")

    print("\n\u2705 Evaluation complete!")
    return results_df


if __name__ == "__main__":
    evaluation_pipeline()
