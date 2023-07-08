import streamlit as st
import pandas as pd
import json
import requests

st.set_page_config(page_title="Results", page_icon="📊")

st.markdown("# Display results")

st.write("Displaying list of past predictions made by the model")

def get_data():
    url = "http://127.0.0.1:5000/diabetes/v1/results"
    headers = {'Content-type':'application/json'}

    response = requests.get(url, headers=headers)
    predictions = json.loads(response.text)

    ## Display error message if no predictions have been made
    print(len(predictions))
    if len(predictions) == 0:
        return "No predictions have been made yet!"

    df = pd.DataFrame(predictions)
    
    ## add columns to df 
    df.columns = ["ID","Glucose", "BMI", "Age", "Prediction", "Confidence", "Timestamp"]
    df["Prediction"] = df["Prediction"].apply(lambda x: "Diabetic" if x == 1 else "Not Diabetic")
    df.set_index("ID", inplace=True)
    
    return df


data = get_data()

if isinstance(data, str):
    st.write(data)
else:
    st.write("###", data.sort_index())


