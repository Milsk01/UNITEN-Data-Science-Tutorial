import pickle 
from flask import Flask, request, json, jsonify
import numpy as np
import requests
import sqlite3

app = Flask(__name__)

#---the filename of the saved model---
filename = '.\output\diabetes.sav'

#---load the saved model---
loaded_model = pickle.load(open(filename, 'rb'))


@app.route('/diabetes/v1/predict', methods=['POST'])
def predict():
    #---get the features to predict---
    features = request.json

    #---create the features list for prediction---
    features_list = [features["Glucose"],
                     features["BMI"],
                     features["Age"]]

    #---get the prediction class---
    prediction = loaded_model.predict([features_list])

    #---get the prediction probabilities---
    confidence = loaded_model.predict_proba([features_list])

    #---formulate the response to return to client---
    response = {}
    response['prediction'] = int(prediction[0])
    response['confidence'] = str(round(np.amax(confidence[0]) * 100 ,2))

    values = features_list + [response['prediction']]

    save(features["Glucose"], features["BMI"], features["Age"], response['prediction'],response['confidence'])

    return jsonify(response)

@app.route('/diabetes/v1/results', methods=['GET'])
def results():
    results = []
    try:
        conn = sqlite3.connect('test.db')
        cursor = conn.execute("SELECT * from RESULTS")
        results = cursor.fetchall()
        print(results)

    except sqlite3.Error as error:
        print("Failed to read results into sqlite table", error)
    finally:
        if conn:
            conn.close()
            print("The SQLite connection is closed")
    
    return jsonify(results)

def save(Glucose, BMI, Age, Prediction, Confidence):
    try:
        conn = sqlite3.connect('test.db')
        cursor = conn.cursor()
        print("Connected to SQLite")
    
        query = '''INSERT INTO RESULTS (GLUCOSE, BMI, AGE, PREDICTION,CONFIDENCE) \
                VALUES (?, ?, ?, ?,?)'''
    
        data = (Glucose, BMI, Age, Prediction, Confidence)

        conn.execute(query, data)
    
        conn.commit()
        print("Inserted Successfully")

        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert a new record into sqlite table", error)
    finally:
        if conn:
            conn.close()
            print("The SQLite connection is closed")
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)