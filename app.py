import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

model = joblib.load("best_model.pkl")

st.title("Student Exam Score Predictor")

study_hours = st.slider("Study Hours per Day",0.0,12.0,2.0)
attendence = st.slider("Attendence precentage",0.0,100.0,75.0)
mental_health = st.slider("Mental Health Rating (1-10)",1,10,5)
sleep_hours = st.slider("Sleep hours",0.0,12.0,7.0)
part_time_job = st.selectbox("Part-Time Job",["No","Yes"])

ptj_encoded = 1 if part_time_job == "Yes" else 0

if st.button("Predict Exam Score"):
    input_features = np.array([[study_hours, attendence, mental_health, sleep_hours, ptj_encoded]])
    prediciton = model.predict(input_features)[0]

    prediciton = max(0 ,  min(100, prediciton))

    st.success(f"Predicted Exam Score {prediciton:.2f}")