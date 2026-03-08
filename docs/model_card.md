# Model Card: Predictive Pulse — Hypertension Classifier

## Model Details

- **Model Name:** Hypertension Stage Classifier
- **Version:** 1.0
- **Type:** Multi-class classification (5 classes)
- **Framework:** scikit-learn, XGBoost, LightGBM
- **Date:** March 2026

## Intended Use

- **Primary Use:** Clinical decision-support tool for hypertension risk assessment
- **Intended Users:** Healthcare professionals, researchers, students
- **Out-of-Scope:** This model is NOT intended for standalone medical diagnosis

## Training Data

- **Source:** Synthetic dataset generated with clinically realistic distributions
- **Size:** 5,000 samples (before SMOTE balancing)
- **Features:** Age, gender, BMI, systolic/diastolic BP, cholesterol, glucose, smoking status, alcohol intake, physical activity, diabetes, medication status
- **Target Classes:** Normal, Elevated, Stage 1 Hypertension, Stage 2 Hypertension, Hypertensive Crisis

## Classification Criteria (AHA Guidelines)

| Stage | Systolic (mmHg) | Diastolic (mmHg) |
|-------|-----------------|-------------------|
| Normal | < 120 | < 80 |
| Elevated | 120–129 | < 80 |
| Stage 1 | 130–139 | 80–89 |
| Stage 2 | ≥ 140 | ≥ 90 |
| Crisis | > 180 | > 120 |

## Models Evaluated

| Model | Cross-Val F1 (weighted) |
|-------|------------------------|
| Logistic Regression | Baseline |
| Decision Tree | Baseline |
| Random Forest | Advanced |
| XGBoost | Advanced |
| LightGBM | Advanced |

Best model selected based on highest weighted F1-score in 5-fold stratified cross-validation.

## Preprocessing Pipeline

1. Missing value imputation (median for numerics)
2. Duplicate removal
3. Outlier capping (IQR method)
4. Ordinal encoding (smoking, alcohol, physical activity)
5. One-hot encoding (gender)
6. Feature engineering (pulse pressure, BMI category, age group)
7. Standard scaling (fit on training data only)
8. SMOTE for class balancing

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score (per-class and weighted)
- Confusion Matrix
- ROC-AUC (one-vs-rest, weighted)

## Limitations

- **Synthetic Data:** Model trained on synthetic data — performance on real clinical data may differ significantly
- **Population Bias:** May not generalize to all demographic groups
- **Feature Set:** Limited to available clinical parameters; does not include family history, stress levels, sleep quality, etc.
- **Not Validated Clinically:** Has not undergone clinical validation or regulatory approval

## Ethical Considerations

- This is a **decision-support tool only** — not a medical diagnostic device
- Should **never** replace professional medical judgment
- Users must be informed about model limitations
- Predictions should be interpreted in conjunction with clinical expertise
- Patient data privacy must be maintained per HIPAA/GDPR guidelines

## How to Cite

```
Predictive Pulse: Harnessing Machine Learning for Blood Pressure Analysis
Version 1.0, March 2026
```
