"""
Unit tests for the Predictive Pulse pipeline.

Tests preprocessing, prediction pipeline, and recommendation module.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import (
    generate_synthetic_data,
    handle_missing_values,
    handle_duplicates,
    detect_and_cap_outliers,
    encode_features,
)
from src.recommendations import get_recommendations


class TestDataGeneration(unittest.TestCase):
    """Test synthetic data generation."""

    def setUp(self):
        self.df = generate_synthetic_data(n_samples=500, random_state=42)

    def test_shape(self):
        # 500 + ~20 duplicates
        self.assertGreater(len(self.df), 500)
        self.assertEqual(self.df.shape[1], 13)

    def test_columns_present(self):
        expected = [
            "age", "gender", "bmi", "systolic_bp", "diastolic_bp",
            "cholesterol", "glucose", "smoking", "alcohol",
            "physical_activity", "diabetes", "medication", "hypertension_stage",
        ]
        for col in expected:
            self.assertIn(col, self.df.columns)

    def test_target_classes(self):
        stages = set(self.df["hypertension_stage"].unique())
        valid = {"Normal", "Elevated", "Stage 1", "Stage 2", "Crisis"}
        self.assertTrue(stages.issubset(valid))

    def test_age_range(self):
        self.assertGreaterEqual(self.df["age"].min(), 18)
        self.assertLessEqual(self.df["age"].max(), 90)


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions."""

    def setUp(self):
        self.df = generate_synthetic_data(n_samples=200, random_state=123)

    def test_missing_values_handled(self):
        df = handle_missing_values(self.df.copy())
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_duplicates_removed(self):
        df = handle_duplicates(self.df.copy())
        self.assertEqual(df.duplicated().sum(), 0)

    def test_outlier_capping(self):
        df = handle_missing_values(self.df.copy())
        df = detect_and_cap_outliers(df, ["bmi", "systolic_bp"])
        # After capping, IQR check should pass
        for col in ["bmi", "systolic_bp"]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.assertGreaterEqual(df[col].min(), Q1 - 1.5 * IQR - 0.01)
            self.assertLessEqual(df[col].max(), Q3 + 1.5 * IQR + 0.01)

    def test_encoding(self):
        df = handle_missing_values(self.df.copy())
        df = handle_duplicates(df)
        df_encoded, le = encode_features(df)
        # Check no string columns remain (except possibly index)
        for col in df_encoded.columns:
            self.assertNotEqual(df_encoded[col].dtype, object)
        # Check label encoder classes
        self.assertEqual(len(le.classes_), 5)


class TestRecommendations(unittest.TestCase):
    """Test recommendation module."""

    def test_all_stages_produce_output(self):
        for stage in ["Normal", "Elevated", "Stage 1", "Stage 2", "Crisis"]:
            recs = get_recommendations(stage)
            self.assertIn("summary", recs)
            self.assertIn("diet", recs)
            self.assertIn("exercise", recs)
            self.assertIn("medication", recs)
            self.assertIn("monitoring", recs)
            self.assertIn("disclaimer", recs)

    def test_crisis_has_urgent_care(self):
        recs = get_recommendations("Crisis")
        self.assertTrue(len(recs["urgent_care"]) > 0)

    def test_normal_no_urgent_care(self):
        recs = get_recommendations("Normal")
        self.assertEqual(len(recs["urgent_care"]), 0)

    def test_smoker_gets_personalized(self):
        recs = get_recommendations("Stage 1", {"smoker": True})
        personalized_text = " ".join(recs.get("personalized", []))
        self.assertIn("Smoking", personalized_text)

    def test_diabetic_gets_personalized(self):
        recs = get_recommendations("Stage 2", {"diabetic": True})
        personalized_text = " ".join(recs.get("personalized", []))
        self.assertIn("Diabetes", personalized_text)

    def test_obese_gets_weight_advice(self):
        recs = get_recommendations("Elevated", {"bmi": 35.0})
        personalized_text = " ".join(recs.get("personalized", []))
        self.assertIn("Weight", personalized_text)


class TestPredictionPipeline(unittest.TestCase):
    """Test prediction pipeline (requires trained model)."""

    @classmethod
    def setUpClass(cls):
        """Try to load predictor — skip tests if model not trained."""
        try:
            from src.predict import HypertensionPredictor
            cls.predictor = HypertensionPredictor()
            cls.model_available = True
        except Exception:
            cls.model_available = False

    def setUp(self):
        if not self.model_available:
            self.skipTest("Model not trained yet")

    def _make_patient(self, systolic, diastolic, **kwargs):
        base = {
            "age": 45, "gender": "Male", "bmi": 25.0,
            "systolic_bp": systolic, "diastolic_bp": diastolic,
            "cholesterol": 200, "glucose": 100, "smoking": "Never",
            "alcohol": "None", "physical_activity": "Moderate",
            "diabetes": 0, "medication": 0,
        }
        base.update(kwargs)
        return base

    def test_normal_patient(self):
        result = self.predictor.predict(self._make_patient(115, 72))
        self.assertIn(result["predicted_stage"], ["Normal", "Elevated"])
        self.assertIsNotNone(result["confidence"])

    def test_stage1_patient(self):
        result = self.predictor.predict(
            self._make_patient(135, 85, age=55, bmi=29.0, smoking="Former")
        )
        self.assertIn(
            result["predicted_stage"], ["Stage 1", "Stage 2", "Elevated"]
        )

    def test_crisis_patient(self):
        result = self.predictor.predict(
            self._make_patient(
                195, 128, age=70, bmi=35.0, smoking="Current",
                alcohol="Heavy", diabetes=1
            )
        )
        self.assertIn(result["predicted_stage"], ["Crisis", "Stage 2"])

    def test_output_structure(self):
        result = self.predictor.predict(self._make_patient(120, 78))
        self.assertIn("predicted_stage", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIn("risk_color", result)
        self.assertIn("model_used", result)

    def test_probabilities_sum_to_one(self):
        result = self.predictor.predict(self._make_patient(130, 85))
        if result["probabilities"]:
            total = sum(result["probabilities"].values())
            self.assertAlmostEqual(total, 1.0, places=2)

    def test_batch_prediction(self):
        patients = [
            self._make_patient(110, 70),
            self._make_patient(150, 95),
            self._make_patient(185, 122),
        ]
        results = self.predictor.predict_batch(patients)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn("predicted_stage", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
