import streamlit as st
import numpy as np
import joblib

# Load trained models and encoders
gpa_model = joblib.load('best_svr_gpa_model.joblib')
stress_model = joblib.load('best_svr_stress_model.joblib')
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
    border-radius: 12px;`
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
    input_data = np.array([[study, extracurricular, sleep, social, physical]])
    scaled_input = scaler.transform(input_data)
    
    # Predict GPA and stress
    gpa = gpa_model.predict(scaled_input)[0]
    stress_prediction_raw = stress_model.predict(scaled_input)
    stress_encoded = np.round(stress_prediction_raw).astype(int)
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