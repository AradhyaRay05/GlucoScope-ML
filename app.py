import streamlit as st
import numpy as np
import pandas as pd
import joblib

try:
    sc = joblib.load('scaler.pkl')
    classifier = joblib.load('model.pkl')
except FileNotFoundError:
    st.error("Model or scaler files not found. Please ensure 'scaler.pkl' and 'model.pkl' are in the same directory.")
    st.stop()

def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age):
    x = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]])
    x = sc.transform(x)

    # Make prediction using the trained classifier
    prediction = classifier.predict(x)
    return prediction[0]


st.title('Diabetes Prediction App')
st.write('Enter the patient details to predict if they have diabetes.')

# Input fields for patient data
pregnancies = st.slider('Pregnancies', 0, 17, 2)
glucose = st.slider('Glucose', 0, 200, 120)
blood_pressure = st.slider('Blood Pressure', 0, 122, 70)
skin_thickness = st.slider('Skin Thickness', 0, 99, 20)
insulin = st.slider('Insulin', 0, 846, 79)
bmi = st.slider('BMI', 0.0, 67.1, 32.0)
dpf = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.47)
age = st.slider('Age', 21, 81, 33)

if st.button('Predict'):
    # Make prediction
    result = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)

    if result == 1:
        st.error('Oops! The person is likely to have diabetes.')
    else:
        st.success('Great! The person is likely NOT to have diabetes.')
