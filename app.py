# Run with: python -m streamlit run app.py

import streamlit as st
import pandas as pd
import joblib

# ====== Page Config ======
st.set_page_config(page_title="Heart Stroke Predictor", page_icon="❤️", layout="centered")

# ====== Custom Accent CSS ======
st.markdown(
    """
    <style>
    /* Force red headings */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ff4b4b !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #ff1c1c;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ====== Load Model and Assets ======
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("heart_disease_scaler.pkl")
expected_columns = joblib.load("heart_disease_columns.pkl")

# ====== App Title ======
st.title("❤️ Heart Stroke Risk Prediction")
st.markdown(
    """
    Enter your details below to estimate your **risk of heart stroke**.  
    This tool uses a trained machine learning model to provide predictions.
    """
)

# ====== Inputs ======
st.subheader("🧍 Personal Information")
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["M", "F"])

st.subheader("🩺 Medical Information")
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TY", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholestrol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])
resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("Slope of ST Segment", ["Up", "Flat", "Down"])

# ====== Prediction ======
if st.button("🔍 Predict Risk"):
    try:
        # Prepare input data
        raw_input = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholestrol,
            "FastingBS": 1 if fasting_bs == "Yes" else 0,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            f"Sex_{sex}": 1,
            f"ChestPainType_{chest_pain}": 1,
            f"RestingECG_{resting_ecg}": 1,
            f"ExerciseAngina_{exercise_angina}": 1,
            f"ST_Slope_{st_slope}": 1
        }

        input_df = pd.DataFrame([raw_input])

        # Add missing columns
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Match column order
        input_df = input_df[expected_columns]

        # Scale features
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)[0]

        # Display result
        st.markdown("---")
        if prediction == 1:
            st.error("🚨 **High risk** of heart disease detected. Please consult a doctor immediately.")
        else:
            st.success("✅ **Low risk** of heart disease detected. Keep up your healthy lifestyle!")

    except Exception as e:
        st.error(f"An error occurred while processing your input: {e}")
