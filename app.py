import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained models and encoders
gpa_model = joblib.load('best_svr_gpa_model.joblib')
stress_model = joblib.load('best_svr_stress_model.joblib') # This now contains the SVC model
scaler = joblib.load('scaler.joblib')
stress_encoder = joblib.load('stress_ordinal_encoder.joblib')

# Page config
st.set_page_config(page_title="GPA & Stress Predictor", page_icon="🎓", layout="centered")

# Enhanced CSS styling
st.markdown("""
<style>
.big-font {
    font-size:22px !important;
}
.result-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    color: black;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Student Lifestyle Predictor")
st.subheader("📊 Predict your GPA & Stress Level from your daily habits")
st.write("Fill in your daily lifestyle habits and let the AI model analyze your academic and stress prediction!")

# --- Input Section ---
with st.form("prediction_form"):
    st.markdown("### ⏱️ Daily Habits")
    col1, col2 = st.columns(2)
    
    with col1:
        study = st.slider("📘 Study Hours per Day", 0.0, 12.0, 3.0, 0.5)
        sleep = st.slider("😴 Sleep Hours per Day", 0.0, 12.0, 7.0, 0.5)
        social = st.slider("💬 Social Hours per Day", 0.0, 12.0, 1.5, 0.5)
    
    with col2:
        extracurricular = st.slider("🎨 Extracurricular Hours per Day", 0.0, 12.0, 1.0, 0.5)
        physical = st.slider("🏃 Physical Activity Hours per Day", 0.0, 12.0, 1.0, 0.5)
    
    submitted = st.form_submit_button("🔍 Predict My Results")

# --- Prediction and Output ---
if submitted:
    
    total_hours = study + sleep + social + extracurricular + physical
    if total_hours > 24:
        st.error(f"🚨 Total hours ({total_hours:.1f}) cannot exceed 24. Please adjust the sliders.")
        st.stop()
    
    # The list of feature names MUST be in the exact same order as in your training script.
    feature_names = [
        'Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day',
        'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day',
        'Physical_Activity_Hours_Per_Day'
    ]

    # Create a pandas DataFrame from the user's input
    input_df = pd.DataFrame([{
        'Study_Hours_Per_Day': study,
        'Extracurricular_Hours_Per_Day': extracurricular,
        'Sleep_Hours_Per_Day': sleep,
        'Social_Hours_Per_Day': social,
        'Physical_Activity_Hours_Per_Day': physical
    }], columns=feature_names)

    # Scale the DataFrame
    scaled_input = scaler.transform(input_df)
    
    # Predict GPA using the SVR model
    gpa = gpa_model.predict(scaled_input)[0]
    
    # Predict Stress Level using the SVC model
    # The output is now a direct integer class (0, 1, or 2), no rounding is needed.
    stress_encoded = stress_model.predict(scaled_input)
    stress_level = stress_encoder.inverse_transform(stress_encoded.reshape(-1, 1))[0][0]
    
    # Recommendations
    def gpa_advice(gpa):
        if gpa >= 3.5:
            return "🌟 Great job! Maintain your study habits and balance."
        elif gpa >= 3.0:
            return "✅ You're doing well. Maybe increase focus time or reduce distractions."
        else:
            return "📚 Consider increasing your study hours and reducing stress to improve performance."
    
    def stress_advice(stress):
        if stress == "Low":
            return "😊 You're managing stress well. Keep a healthy routine!"
        elif stress == "Moderate":
            return "😐 Moderate stress detected. Consider more breaks and social time."
        else:
            return "⚠️ High stress detected! Prioritize sleep, exercise, and seek support if needed."
    
    # Display Results
    st.markdown(f"""
    <div class='result-box'>
        <h4>📈 Predicted GPA: <span class='big-font'>{gpa:.2f}</span></h4>
        <p>{gpa_advice(gpa)}</p>
        <h4>🧠 Predicted Stress Level: <span class='big-font'>{stress_level}</span></h4>
        <p>{stress_advice(stress_level)}</p>
    </div>
    """, unsafe_allow_html=True)