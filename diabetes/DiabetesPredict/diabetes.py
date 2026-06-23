
# First, make sure joblib is installed
# In Google Colab, uncomment the next line:
# !pip install joblib streamlit pandas scikit-learn

import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
# Make sure your "diabetes_lr.pkl" file is in the same folder as this script
model, scaler, saved_cols = joblib.load("diabetes_lr.pkl")

st.title("🩺 Diabetes Prediction App")

# Collect user input
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prepare data
input_data = pd.DataFrame(
    [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
    columns=saved_cols
)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ The patient is likely to have Diabetes.")
    else:
        st.success("✅ The patient is unlikely to have Diabetes.")
