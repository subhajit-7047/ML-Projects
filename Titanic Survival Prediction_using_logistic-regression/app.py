import streamlit as st
import numpy as np
import pandas as pd
import joblib


    
# 🔹 Load the trained pipeline

pipeline = joblib.load("pipeline_1.pkl")
print("Pipeline loaded successfully.")
print(pipeline)


st.title("🚢 Titanic Survival Prediction (with Pipeline)")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Embarked", ['S', 'C', 'Q'])

# Collect input in dictionary
input_data = {
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
}

# Convert to DataFrame (important for pipeline)
input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict"):
    prediction = pipeline.predict(input_df)
    probability = pipeline.predict_proba(input_df)

    if prediction[0] == 1:
        st.success(f"🎉 The passenger is likely to SURVIVE (Probability: {probability[0][1]:.2f})")
    else:
        st.error(f"💀 The passenger is NOT likely to survive (Probability: {probability[0][0]:.2f})")
