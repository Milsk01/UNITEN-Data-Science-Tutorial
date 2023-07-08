import streamlit as st 
import json
import requests

# Title
st.header("Diabetes Prediction Demo")

# Input bar 1
Glucose = st.number_input("Enter Glucose", format = "%d" , value = 65)

# Input bar 2
BMI = st.number_input("Enter BMI",format = "%.2f", value = 21.0)

#  Input bar 3
Age = st.number_input("Enter Age", format = "%d", value = 21)

# If button is pressed
if st.button("Submit"):
    url = 'http://127.0.0.1:5000/diabetes/v1/predict'
    data = {"BMI":BMI, "Age":Age, "Glucose":Glucose}
    data_json = json.dumps(data)
    headers = {'Content-type':'application/json'}
    response = requests.post(url, data=data_json, headers=headers)
    predictions = json.loads(response.text)

    # Output prediction
    st.text("Diabetic" if predictions["prediction"] == 1 else "Not Diabetic")
    st.text("Confidence: " + predictions["confidence"] + "%")