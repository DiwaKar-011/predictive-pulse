# User Guide — Predictive Pulse

## Overview

Predictive Pulse is a machine learning-powered tool that predicts hypertension stages from patient clinical data and provides personalized health recommendations.

## Getting Started

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/DiwaKar-011/predictive-pulse.git
cd predictive-pulse

# Create virtual environment
python -m venv venv
venv\Scripts\activate         # Windows
source venv/bin/activate      # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

Before using the prediction system, you need to train the model:

```bash
python src/preprocess.py           # Generate & preprocess data
python src/feature_engineering.py  # Engineer features
python src/train.py                # Train models
python src/evaluate.py             # Evaluate models
```

### Running the App

```bash
streamlit run app/app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Using the Dashboard

### Step 1: Enter Patient Data

In the sidebar, fill in:
- **Demographics:** Age, Gender, BMI
- **Blood Pressure:** Systolic and Diastolic readings (mmHg)
- **Lab Values:** Cholesterol and Glucose levels (mg/dL)
- **Lifestyle:** Smoking, Alcohol, Physical Activity
- **Medical History:** Diabetes, Current BP Medication

### Step 2: Get Prediction

Click **Predict** to receive:
- **Predicted Stage:** The hypertension classification
- **Confidence Score:** How confident the model is
- **Risk Gauge:** Visual risk indicator
- **Probability Breakdown:** Likelihood for each stage

### Step 3: Review Recommendations

The system provides personalized recommendations across:
- 🥗 **Diet** — DASH diet guidelines, sodium limits
- 🏃 **Exercise** — Activity type and duration
- 💊 **Medication** — When to consult about medication
- 📊 **Monitoring** — How often to check BP
- 🎯 **Personalized** — Based on your risk factors

## Interpreting Results

### Hypertension Stages

| Stage | What It Means | Action |
|-------|--------------|--------|
| **Normal** | BP < 120/80 | Maintain healthy lifestyle |
| **Elevated** | 120-129/<80 | Lifestyle modifications |
| **Stage 1** | 130-139/80-89 | Lifestyle + possible medication |
| **Stage 2** | ≥140/≥90 | Medication + lifestyle changes |
| **Crisis** | >180/>120 | **Seek immediate medical help** |

### Confidence Scores

- **>90%:** High confidence — the model is very sure
- **70-90%:** Moderate confidence — reasonably reliable
- **<70%:** Lower confidence — consider additional clinical assessment

## Running Tests

```bash
python -m pytest tests/ -v
```

## Important Disclaimers

⚠️ **This system is for educational and decision-support purposes only.**

- It is NOT a medical diagnostic device
- It has NOT been clinically validated
- Always consult a qualified healthcare professional
- Do NOT make treatment decisions based solely on this tool
- The model was trained on synthetic data

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model not found" error | Run the training pipeline first |
| Streamlit won't start | Check `pip install streamlit` |
| Slow predictions | Normal for first load; subsequent predictions are faster |
| Import errors | Ensure virtual environment is activated |
