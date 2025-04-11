# app.py

import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Risk Predictor")

st.write("Enter patient data below to predict the risk of diabetes:")

# Input form
preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0.0, value=100.0)
bp = st.number_input("Blood Pressure", min_value=0.0, value=70.0)
skinthick = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
insulin = st.number_input("Insulin", min_value=0.0, value=80.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=1, value=33)

input_data = np.array([[preg, glucose, bp, skinthick, insulin, bmi, dpf, age]])

# Predict
if st.button("Predict Diabetes Risk"):
    scaled_input = scaler.transform(input_data)
    result = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    if result == 1:
        st.error(f"⚠️ High Risk of Diabetes ({prob*100:.2f}% probability)")
    else:
        st.success(f"✅ Low Risk of Diabetes ({prob*100:.2f}% probability)")
