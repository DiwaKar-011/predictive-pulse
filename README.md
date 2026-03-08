# 🩺 Predictive Pulse: Harnessing Machine Learning for Blood Pressure Analysis

> **Hypertension Prediction System** — A clinical decision-support tool powered by machine learning.

⚠️ **Disclaimer:** This system is intended as a clinical decision-support tool only. It is **not** a substitute for professional medical diagnosis or treatment. Always consult a qualified healthcare professional.

---

## 📋 Overview

Predictive Pulse is a machine learning-based system that predicts hypertension stages from patient clinical parameters. It classifies blood pressure into five categories:

| Stage | Systolic (mmHg) | Diastolic (mmHg) |
|-------|-----------------|-------------------|
| Normal | < 120 | < 80 |
| Elevated | 120–129 | < 80 |
| Stage 1 Hypertension | 130–139 | 80–89 |
| Stage 2 Hypertension | ≥ 140 | ≥ 90 |
| Hypertensive Crisis | > 180 | > 120 |

## 🔬 Clinical Parameters Used

- Systolic & Diastolic Blood Pressure
- Age, Gender, BMI
- Cholesterol, Glucose levels
- Smoking status, Alcohol intake
- Physical activity level
- Existing conditions (diabetes, etc.)
- Medication status

## 🏗️ Project Structure

```
predictive-pulse/
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned & encoded data
├── notebooks/
│   ├── EDA.ipynb
│   └── Model_Training.ipynb
├── src/
│   ├── preprocess.py         # Data preprocessing
│   ├── feature_engineering.py # Feature engineering & selection
│   ├── train.py              # Model training & tuning
│   ├── evaluate.py           # Model evaluation
│   ├── predict.py            # Prediction pipeline
│   └── recommendations.py   # Health recommendations
├── models/
│   └── best_model.pkl
├── app/
│   └── app.py                # Streamlit dashboard
├── tests/
│   └── test_pipeline.py
├── docs/
│   ├── model_card.md
│   └── user_guide.md
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/DiwaKar-011/predictive-pulse.git
cd predictive-pulse
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate synthetic data and train models
```bash
python src/preprocess.py
python src/feature_engineering.py
python src/train.py
python src/evaluate.py
```

### 5. Run the Streamlit app
```bash
streamlit run app/app.py
```

## 📊 Models Implemented

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable tree-based model |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosting framework |
| LightGBM | Fast gradient boosting |

## 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (per class)
- ROC-AUC (one-vs-rest for multi-class)
- Classification Report

## 🎯 Success Criteria

- Overall accuracy ≥ 85%
- Weighted F1-score ≥ 0.83
- No critical misclassification of Hypertensive Crisis as Normal

## ⚖️ Ethical Considerations

- This tool is for **educational and decision-support purposes only**
- It should **never** replace professional medical diagnosis
- Always consult a qualified healthcare professional for medical decisions
- The model was trained on synthetic/public data and may not generalize to all populations

## 📝 License

This project is for educational purposes. See [LICENSE](LICENSE) for details.

---

*Built with ❤️ for better health outcomes*
