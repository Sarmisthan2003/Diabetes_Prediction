import streamlit as st
import pickle
import numpy as np
import os

# Load the saved classifier and scaler
with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.markdown("<h1 style='text-align: center;'>Diabetes Prediction</h1>", unsafe_allow_html=True)

st.markdown("---")
st.write("### Enter the following details:")

# User inputs
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0.0)
bp = st.number_input("Blood Pressure", min_value=0.0)
skin = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    input_data = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
    input_array = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = classifier.predict(scaled_input)

    if prediction[0] == 0:
        st.success("✅ The person is **not diabetic**.")
    else:
        st.error("⚠️ The person **is diabetic**.")

