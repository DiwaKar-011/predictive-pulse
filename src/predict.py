"""
Prediction pipeline module for Predictive Pulse.

Loads the trained model and scaler, accepts raw patient input,
applies preprocessing, and returns predicted hypertension stage
with confidence/probability scores.
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class HypertensionPredictor:
    """Prediction pipeline for hypertension stage classification."""

    def __init__(self, models_dir=None):
        self.models_dir = models_dir or MODELS_DIR
        self._load_artifacts()

    def _load_artifacts(self):
        """Load all saved model artifacts."""
        self.model = joblib.load(os.path.join(self.models_dir, "best_model.pkl"))
        self.scaler = joblib.load(os.path.join(self.models_dir, "scaler.pkl"))
        self.label_encoder = joblib.load(
            os.path.join(self.models_dir, "label_encoder.pkl")
        )
        self.feature_names = joblib.load(
            os.path.join(self.models_dir, "feature_names.pkl")
        )
        self.model_name = joblib.load(
            os.path.join(self.models_dir, "best_model_name.pkl")
        )
        # The scaler was fit on the original features before feature engineering
        self.scaler_features = list(self.scaler.feature_names_in_)

    def preprocess_input(self, patient_data):
        """
        Preprocess raw patient input into model-ready format.

        Parameters
        ----------
        patient_data : dict
            Raw patient data with keys: age, gender, bmi, systolic_bp,
            diastolic_bp, cholesterol, glucose, smoking, alcohol,
            physical_activity, diabetes, medication

        Returns
        -------
        pd.DataFrame
            Scaled and feature-engineered dataframe ready for prediction.
        """
        df = pd.DataFrame([patient_data])

        # Encode ordinal features
        ordinal_maps = {
            "physical_activity": {"Low": 0, "Moderate": 1, "High": 2},
            "alcohol": {"None": 0, "Moderate": 1, "Heavy": 2},
            "smoking": {"Never": 0, "Former": 1, "Current": 2},
        }
        for col, mapping in ordinal_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        # One-hot encode gender
        if "gender" in df.columns:
            df["gender_Male"] = (df["gender"] == "Male").astype(int)
            df.drop("gender", axis=1, inplace=True)

        # Ensure all scaler features exist
        for col in self.scaler_features:
            if col not in df.columns:
                df[col] = 0

        df_for_scaling = df[self.scaler_features]

        # Scale using the original scaler
        df_scaled = pd.DataFrame(
            self.scaler.transform(df_for_scaling), columns=self.scaler_features
        )

        # Add derived features on the scaled data
        if "systolic_bp" in df_scaled.columns and "diastolic_bp" in df_scaled.columns:
            df_scaled["pulse_pressure"] = df_scaled["systolic_bp"] - df_scaled["diastolic_bp"]
        if "bmi" in df.columns:
            bmi_val = patient_data.get("bmi", 25.0)
            if bmi_val < 18.5:
                df_scaled["bmi_category"] = 0
            elif bmi_val < 25:
                df_scaled["bmi_category"] = 1
            elif bmi_val < 30:
                df_scaled["bmi_category"] = 2
            else:
                df_scaled["bmi_category"] = 3
        if "age" in df.columns:
            age_val = patient_data.get("age", 45)
            if age_val < 30:
                df_scaled["age_group"] = 0
            elif age_val < 45:
                df_scaled["age_group"] = 1
            elif age_val < 60:
                df_scaled["age_group"] = 2
            else:
                df_scaled["age_group"] = 3

        # Select only the model's expected features
        for col in self.feature_names:
            if col not in df_scaled.columns:
                df_scaled[col] = 0

        df_scaled = df_scaled[self.feature_names]

        return df_scaled

    def predict(self, patient_data):
        """
        Predict hypertension stage for a patient.

        Parameters
        ----------
        patient_data : dict
            Raw patient data dictionary.

        Returns
        -------
        dict
            Prediction result with stage, confidence, and probabilities.
        """
        X = self.preprocess_input(patient_data)

        # Predicted class
        pred_encoded = self.model.predict(X)[0]
        pred_stage = self.label_encoder.inverse_transform([pred_encoded])[0]

        # Probability scores
        probabilities = {}
        confidence = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            for i, cls in enumerate(self.label_encoder.classes_):
                if i < len(proba):
                    probabilities[cls] = round(float(proba[i]), 4)
            confidence = round(float(max(proba)), 4)

        # Risk level color mapping
        risk_colors = {
            "Normal": "green",
            "Elevated": "yellow",
            "Stage 1": "orange",
            "Stage 2": "red",
            "Crisis": "darkred",
        }

        return {
            "predicted_stage": pred_stage,
            "confidence": confidence,
            "probabilities": probabilities,
            "risk_color": risk_colors.get(pred_stage, "gray"),
            "model_used": self.model_name,
        }

    def predict_batch(self, patients_list):
        """Predict for multiple patients."""
        results = []
        for patient in patients_list:
            results.append(self.predict(patient))
        return results


def demo_predictions():
    """Demonstrate predictions with sample patients."""
    predictor = HypertensionPredictor()

    sample_patients = [
        {
            "name": "Normal BP Patient",
            "age": 28,
            "gender": "Female",
            "bmi": 22.0,
            "systolic_bp": 115,
            "diastolic_bp": 72,
            "cholesterol": 180,
            "glucose": 85,
            "smoking": "Never",
            "alcohol": "None",
            "physical_activity": "High",
            "diabetes": 0,
            "medication": 0,
        },
        {
            "name": "Stage 1 Patient",
            "age": 52,
            "gender": "Male",
            "bmi": 29.5,
            "systolic_bp": 135,
            "diastolic_bp": 85,
            "cholesterol": 230,
            "glucose": 110,
            "smoking": "Former",
            "alcohol": "Moderate",
            "physical_activity": "Low",
            "diabetes": 0,
            "medication": 0,
        },
        {
            "name": "Crisis Patient",
            "age": 68,
            "gender": "Male",
            "bmi": 34.0,
            "systolic_bp": 190,
            "diastolic_bp": 125,
            "cholesterol": 280,
            "glucose": 180,
            "smoking": "Current",
            "alcohol": "Heavy",
            "physical_activity": "Low",
            "diabetes": 1,
            "medication": 1,
        },
    ]

    print("=" * 60)
    print("PREDICTION DEMO")
    print("=" * 60)

    for patient in sample_patients:
        name = patient.pop("name")
        result = predictor.predict(patient)
        print(f"\n\U0001f3e5 Patient: {name}")
        print(f"   Predicted Stage: {result['predicted_stage']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Risk Level: {result['risk_color']}")
        print(f"   Probabilities: {result['probabilities']}")


if __name__ == "__main__":
    demo_predictions()
