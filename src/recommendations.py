"""
Health recommendation module for Predictive Pulse.

Maps predicted hypertension stages to personalized lifestyle
recommendations covering diet, exercise, medication, monitoring,
and urgent care guidance.
"""


def get_recommendations(predicted_stage, patient_metadata=None):
    """
    Generate personalized health recommendations based on predicted
    hypertension stage and optional patient metadata.

    Parameters
    ----------
    predicted_stage : str
        One of: Normal, Elevated, Stage 1, Stage 2, Crisis
    patient_metadata : dict, optional
        Keys: age (int), smoker (bool), diabetic (bool), bmi (float)

    Returns
    -------
    dict
        Structured recommendations by category.
    """
    metadata = patient_metadata or {}
    age = metadata.get("age", None)
    is_smoker = metadata.get("smoker", False)
    is_diabetic = metadata.get("diabetic", False)
    bmi = metadata.get("bmi", None)

    base_recommendations = _get_stage_recommendations(predicted_stage)

    # Add personalized extras
    extras = []
    if is_smoker:
        extras.append(
            "\ud83d\udeac **Smoking Cessation:** Quitting smoking is one of the most impactful "
            "steps to reduce cardiovascular risk. Consult your doctor about cessation programs."
        )
    if is_diabetic:
        extras.append(
            "\ud83e\ude78 **Diabetes Management:** Monitor blood glucose regularly. Maintain HbA1c "
            "below 7%. Coordinate BP and diabetes treatment with your healthcare provider."
        )
    if bmi and bmi >= 30:
        extras.append(
            "\u2696\ufe0f **Weight Management:** A 5\u201310% reduction in body weight can significantly "
            "lower blood pressure. Consider consulting a dietitian for a personalized plan."
        )
    if age and age >= 65:
        extras.append(
            "\ud83d\udc74 **Age Considerations:** Older adults may be more sensitive to blood pressure "
            "medications. Regular monitoring and medication review are advised."
        )

    base_recommendations["personalized"] = extras
    base_recommendations["disclaimer"] = (
        "\u26a0\ufe0f These recommendations are for informational purposes only and are NOT a "
        "substitute for professional medical advice. Always consult a qualified "
        "healthcare professional before making health decisions."
    )

    return base_recommendations


def _get_stage_recommendations(stage):
    """Return stage-specific recommendations."""
    recommendations = {
        "Normal": {
            "summary": "Your blood pressure is within the normal range. Maintain your healthy lifestyle!",
            "risk_level": "Low",
            "diet": [
                "Continue a balanced diet rich in fruits, vegetables, and whole grains.",
                "Limit sodium intake to less than 2,300 mg/day.",
                "Include potassium-rich foods: bananas, spinach, sweet potatoes.",
            ],
            "exercise": [
                "Maintain at least 150 minutes of moderate aerobic activity per week.",
                "Include strength training 2\u20133 times per week.",
            ],
            "medication": [
                "No medication typically required.",
                "Continue routine health checkups.",
            ],
            "monitoring": [
                "Check blood pressure at least once a year.",
                "Monitor for any lifestyle changes that could affect BP.",
            ],
            "urgent_care": [],
        },
        "Elevated": {
            "summary": "Your blood pressure is elevated. Lifestyle modifications can help prevent progression.",
            "risk_level": "Moderate-Low",
            "diet": [
                "Adopt the DASH diet (Dietary Approaches to Stop Hypertension).",
                "Reduce sodium to less than 2,300 mg/day (ideally 1,500 mg).",
                "Limit processed foods, sugary beverages, and red meat.",
                "Increase intake of fruits, vegetables, lean proteins, and whole grains.",
            ],
            "exercise": [
                "Aim for 150+ minutes of moderate aerobic activity per week.",
                "Consider brisk walking, cycling, or swimming.",
                "Include flexibility and balance exercises.",
            ],
            "medication": [
                "Medication not typically prescribed at this stage.",
                "Focus on lifestyle modifications first.",
            ],
            "monitoring": [
                "Check blood pressure every 3\u20136 months.",
                "Track readings in a log to identify trends.",
            ],
            "urgent_care": [],
        },
        "Stage 1": {
            "summary": "You have Stage 1 Hypertension. Lifestyle changes and possible medication are recommended.",
            "risk_level": "Moderate",
            "diet": [
                "Follow the DASH diet strictly.",
                "Reduce sodium to less than 1,500 mg/day.",
                "Limit alcohol: \u22641 drink/day for women, \u22642 for men.",
                "Reduce caffeine intake and monitor its effect on BP.",
                "Increase omega-3 fatty acids (fish, walnuts, flaxseed).",
            ],
            "exercise": [
                "At least 150 minutes of moderate or 75 minutes of vigorous exercise weekly.",
                "Include aerobic, resistance, and flexibility training.",
                "Avoid heavy lifting without proper guidance.",
            ],
            "medication": [
                "Consult your doctor about starting antihypertensive medication.",
                "Common first-line: ACE inhibitors, ARBs, or calcium channel blockers.",
                "Medication may be recommended if lifestyle changes are insufficient after 3\u20136 months.",
            ],
            "monitoring": [
                "Monitor blood pressure at home weekly.",
                "Visit your healthcare provider every 1\u20133 months.",
                "Track medication adherence and side effects.",
            ],
            "urgent_care": [],
        },
        "Stage 2": {
            "summary": "You have Stage 2 Hypertension. Medical treatment and lifestyle changes are essential.",
            "risk_level": "High",
            "diet": [
                "Strict DASH diet compliance.",
                "Sodium intake below 1,500 mg/day.",
                "Eliminate processed foods, excess salt, and alcohol.",
                "Consider consulting a clinical dietitian.",
            ],
            "exercise": [
                "Regular physical activity as tolerated \u2014 consult doctor first.",
                "Moderate-intensity activities preferred (walking, swimming).",
                "Avoid high-intensity exercise until BP is better controlled.",
            ],
            "medication": [
                "Two or more antihypertensive medications may be needed.",
                "Take medications exactly as prescribed.",
                "Do NOT skip doses or stop without medical advice.",
                "Regular follow-up to adjust dosages.",
            ],
            "monitoring": [
                "Daily home blood pressure monitoring recommended.",
                "Medical follow-up every 1\u20132 months until stable.",
                "Monitor for organ damage: kidney function, eye exams, ECG.",
            ],
            "urgent_care": [
                "Seek medical attention if BP consistently exceeds 160/100 despite medication.",
            ],
        },
        "Crisis": {
            "summary": "\u26a0\ufe0f HYPERTENSIVE CRISIS DETECTED. Seek immediate medical attention!",
            "risk_level": "CRITICAL",
            "diet": [
                "Focus on medical stabilization first.",
                "Once stable, follow strict dietary guidelines per physician.",
            ],
            "exercise": [
                "NO exercise until medically cleared.",
                "Rest and avoid physical exertion.",
            ],
            "medication": [
                "IMMEDIATE medical treatment required.",
                "IV medications may be needed in emergency settings.",
                "Do not attempt to self-medicate.",
            ],
            "monitoring": [
                "Continuous monitoring in a medical facility.",
                "Monitor for signs of organ damage: chest pain, vision changes, severe headache, confusion.",
            ],
            "urgent_care": [
                "\ud83d\udea8 CALL EMERGENCY SERVICES (911) IMMEDIATELY if experiencing:",
                "  - Severe headache or confusion",
                "  - Chest pain or difficulty breathing",
                "  - Vision changes",
                "  - Numbness or weakness",
                "  - Blood pressure above 180/120 mmHg",
                "Go to the nearest emergency room NOW.",
            ],
        },
    }

    return recommendations.get(stage, recommendations["Normal"])


def format_recommendations_text(recs):
    """Format recommendations as readable text."""
    lines = []
    lines.append(f"\ud83d\udccb {recs['summary']}")
    lines.append(f"Risk Level: {recs['risk_level']}")
    lines.append("")

    sections = [
        ("\ud83e\udd57 Diet", "diet"),
        ("\ud83c\udfc3 Exercise", "exercise"),
        ("\ud83d\udc8a Medication", "medication"),
        ("\ud83d\udcca Monitoring", "monitoring"),
        ("\ud83d\udea8 Urgent Care", "urgent_care"),
    ]

    for title, key in sections:
        items = recs.get(key, [])
        if items:
            lines.append(f"{title}:")
            for item in items:
                lines.append(f"  \u2022 {item}")
            lines.append("")

    personalized = recs.get("personalized", [])
    if personalized:
        lines.append("\ud83c\udfaf Personalized Recommendations:")
        for item in personalized:
            lines.append(f"  \u2022 {item}")
        lines.append("")

    lines.append(recs.get("disclaimer", ""))
    return "\n".join(lines)


if __name__ == "__main__":
    # Demo
    for stage in ["Normal", "Elevated", "Stage 1", "Stage 2", "Crisis"]:
        print("=" * 60)
        recs = get_recommendations(
            stage, {"age": 55, "smoker": True, "diabetic": True, "bmi": 32.0}
        )
        print(format_recommendations_text(recs))
        print()
