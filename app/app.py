"""
Streamlit dashboard for Predictive Pulse — Hypertension Prediction System.

Provides an interactive UI for entering patient data, viewing predictions,
confidence scores, risk visualizations, and personalized health recommendations.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import HypertensionPredictor
from src.recommendations import get_recommendations, format_recommendations_text

# Page config
st.set_page_config(
    page_title="Predictive Pulse — Hypertension Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_predictor():
    """Load the prediction pipeline (cached)."""
    try:
        return HypertensionPredictor()
    except Exception as e:
        return None


def create_risk_gauge(stage, confidence):
    """Create a risk gauge visualization."""
    stage_values = {
        "Normal": 20,
        "Elevated": 40,
        "Stage 1": 60,
        "Stage 2": 80,
        "Crisis": 100,
    }
    stage_colors = {
        "Normal": "#2ecc71",
        "Elevated": "#f1c40f",
        "Stage 1": "#e67e22",
        "Stage 2": "#e74c3c",
        "Crisis": "#8b0000",
    }

    value = stage_values.get(stage, 50)
    color = stage_colors.get(stage, "#95a5a6")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={"text": f"Risk Level: {stage}", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 20], "color": "#d5f5e3"},
                    {"range": [20, 40], "color": "#fdebd0"},
                    {"range": [40, 60], "color": "#fae5d3"},
                    {"range": [60, 80], "color": "#fadbd8"},
                    {"range": [80, 100], "color": "#f5b7b1"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(t=60, b=20, l=30, r=30))
    return fig


def create_probability_chart(probabilities):
    """Create a bar chart of class probabilities."""
    stages = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8b0000"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=stages,
                y=probs,
                marker_color=colors[: len(stages)],
                text=[f"{p:.1%}" for p in probs],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="Prediction Probability Distribution",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=350,
        margin=dict(t=50, b=30),
    )
    return fig


def main():
    # Header
    st.title("🩺 Predictive Pulse")
    st.subheader("Hypertension Prediction & Health Recommendation System")

    st.warning(
        "⚠️ **Disclaimer:** This tool is for educational and decision-support "
        "purposes only. It is NOT a substitute for professional medical diagnosis "
        "or treatment. Always consult a qualified healthcare professional."
    )

    # Load predictor
    predictor = load_predictor()

    if predictor is None:
        st.error(
            "❌ Model not found. Please run the training pipeline first:\n\n"
            "```\npython src/preprocess.py\npython src/feature_engineering.py\n"
            "python src/train.py\n```"
        )
        return

    st.markdown(f"**Model:** {predictor.model_name}")

    # Sidebar — Patient Input
    st.sidebar.header("📝 Patient Information")

    with st.sidebar.form("patient_form"):
        st.subheader("Demographics")
        age = st.slider("Age", 18, 90, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.number_input("BMI", 15.0, 50.0, 25.0, 0.5)

        st.subheader("Blood Pressure")
        systolic = st.number_input("Systolic BP (mmHg)", 80, 220, 120, 1)
        diastolic = st.number_input("Diastolic BP (mmHg)", 50, 140, 80, 1)

        st.subheader("Lab Values")
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 350, 200, 5)
        glucose = st.number_input("Glucose (mg/dL)", 60, 300, 100, 5)

        st.subheader("Lifestyle")
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol = st.selectbox("Alcohol Intake", ["None", "Moderate", "Heavy"])
        activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])

        st.subheader("Medical History")
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
        medication = st.selectbox(
            "BP Medication", [0, 1], format_func=lambda x: "Yes" if x else "No"
        )

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        patient_data = {
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "systolic_bp": systolic,
            "diastolic_bp": diastolic,
            "cholesterol": cholesterol,
            "glucose": glucose,
            "smoking": smoking,
            "alcohol": alcohol,
            "physical_activity": activity,
            "diabetes": diabetes,
            "medication": medication,
        }

        # Predict
        result = predictor.predict(patient_data)

        # Layout
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 🎯 Prediction Result")

            # Color-coded result
            color_map = {
                "green": "🟢",
                "yellow": "🟡",
                "orange": "🟠",
                "red": "🔴",
                "darkred": "⚫",
            }
            emoji = color_map.get(result["risk_color"], "⚪")

            st.markdown(
                f"## {emoji} {result['predicted_stage']}"
            )

            if result["confidence"]:
                st.metric("Confidence", f"{result['confidence']:.1%}")

            # Risk gauge
            gauge_fig = create_risk_gauge(
                result["predicted_stage"], result["confidence"]
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col2:
            st.markdown("### 📊 Probability Breakdown")
            if result["probabilities"]:
                prob_fig = create_probability_chart(result["probabilities"])
                st.plotly_chart(prob_fig, use_container_width=True)

                # Probability table
                prob_df = pd.DataFrame(
                    list(result["probabilities"].items()),
                    columns=["Stage", "Probability"],
                )
                prob_df["Probability"] = prob_df["Probability"].apply(
                    lambda x: f"{x:.2%}"
                )
                st.table(prob_df)

        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Health Recommendations")

        metadata = {
            "age": age,
            "smoker": smoking == "Current",
            "diabetic": diabetes == 1,
            "bmi": bmi,
        }
        recs = get_recommendations(result["predicted_stage"], metadata)

        # Summary
        st.info(recs["summary"])

        # Urgent care (show first if Crisis)
        if recs.get("urgent_care"):
            st.error("\n".join(recs["urgent_care"]))

        # Tabs for recommendation categories
        tabs = st.tabs(["🥗 Diet", "🏃 Exercise", "💊 Medication", "📊 Monitoring", "🎯 Personalized"])

        for tab, (key, items) in zip(
            tabs,
            [
                ("diet", recs.get("diet", [])),
                ("exercise", recs.get("exercise", [])),
                ("medication", recs.get("medication", [])),
                ("monitoring", recs.get("monitoring", [])),
                ("personalized", recs.get("personalized", [])),
            ],
        ):
            with tab:
                if items:
                    for item in items:
                        st.markdown(f"- {item}")
                else:
                    st.markdown("*No specific recommendations for this category.*")

        # Disclaimer at bottom
        st.markdown("---")
        st.caption(recs.get("disclaimer", ""))

        # Patient summary
        with st.expander("📋 Patient Input Summary"):
            input_df = pd.DataFrame([patient_data])
            st.dataframe(input_df)

    else:
        # Landing page content
        st.markdown("---")
        st.markdown(
            """
        ### How to Use
        1. Enter patient clinical parameters in the sidebar
        2. Click **Predict** to get the hypertension stage prediction
        3. Review the risk assessment and probability breakdown
        4. Follow the personalized health recommendations

        ### Hypertension Stages (AHA Guidelines)
        | Stage | Systolic (mmHg) | Diastolic (mmHg) |
        |-------|-----------------|-------------------|
        | Normal | < 120 | < 80 |
        | Elevated | 120\u2013129 | < 80 |
        | Stage 1 | 130\u2013139 | 80\u201389 |
        | Stage 2 | \u2265 140 | \u2265 90 |
        | Crisis | > 180 | > 120 |
        """
        )


if __name__ == "__main__":
    main()
