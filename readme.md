# Diabetes Prediction using Machine Learning

## Introduction

This is a data science project from UNITEN that aims to predict whether a person has diabetes or not. The project is divided into 3 parts:

- Exploratory Data Analysis
- Model Training
- Deployment

## Getting Started

Final File structure

```
    ├───data
    │ └───diabetes.csv
    ├───client
    │ ├───home.py
    │ └───Pages
    │ ├───Predictor.py
    │ └───Results.py
    ├───output
    │ └───diabetes.sav
    ├───server
    │ └───app.py
    ├───dataset.py
    ├───database.py
    ├───read.py
    ├───readme.md
    ├───requirements.txt
    ├───test.db
    └───train.py
```

### Dependencies

- Python 3.11
- Flask
- Streamlit
- Scikit-learn
- Pandas
- Numpy

### Installation

Requires Python >= 3.11

```cmd
pip install -r requirements.txt
```

### Datasets

1.  Download the datasets from the following links:
    - [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
2.  Place the datasets in the `data` folder.
3.  Create `dataset.py` file to load the dataset and preprocess it.
    <details>
        <summary>Click to view example code</summary>

            import numpy as np
            import pandas as pd
            from sklearn import datasets

            def read_dataset(path):
                df = pd.read_csv(path)
                return df

            def preprocess(df):
                cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

                df[cols] = df[cols].replace(0,np.NaN)
                df.fillna(df.mean(), inplace = True)

                return df

            def check_nulls(df):
                return df.isnull().sum()

            def check_zeros(df):
                return df.eq(0).sum()

            def load_dataset():
                df = read_dataset('Data\diabetes.csv')
                df = preprocess(df)
                return df

            if __name__ == '__main__':
                df = load_dataset()
                print(df.head())

    </details>

## Training

1. Make sure the scikit-learn library is installed: pip install scikit-learn
2. Place the dataset.py file in the same directory as this script.
3. Run `train.py` script.The script will load the dataset, train a KNN classifier with 19 neighbors on the features

```cmd
python train.py
```

4. The model will be saved to the output directory.

## Deployment

Next, we will deploy the model using Flask and Streamlit. Flask is a web framework that allows us to deploy our machine learning model as a REST API. Streamlit is a library that allows us to create web applications for machine learning and data science. Before starting, make sure you have the following libraries installed:

- Flask
- Streamlit
- sqlite3

Note that we will be using sqlite3 as our database. It is recommended to use a more robust database such as MySQL or PostgreSQL for production.

### Creating Database

Database is created using sqlite3. The following code will create a database named test.db and a table named results. The table will have the following columns:

- ID : Primary key
- Glucose : Glucose level
- BMI : Body Mass Index
- Age : Age
- Prediction : Prediction
- Confidence : Confidence
- Created_at : Timestamp

```
import sqlite3

conn = sqlite3.connect('test.db')

print("Opened database successfully")

conn.execute('''CREATE TABLE RESULTS
        (ID INTEGER PRIMARY KEY AUTOINCREMENT,
         GLUCOSE INT NOT NULL,
         BMI INT NOT NULL,
         AGE INT NOT NULL,
         PREDICTION char(50) NOT NULL,
         CONFIDENCE real NOT NULL,
         CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL)
             ;''')

print("Table created successfully")

conn.close()
```

### Flask API

1. Import the required libraries

   ```
       import pickle
       from flask import Flask, request, json, jsonify
       import numpy as np
       import requests
       import sqlite3
   ```

2. Load the model

   ```
       #---the filename of the saved model---
       filename = '.\output\diabetes.sav'

       #---load the saved model---
       loaded_model = pickle.load(open(filename, 'rb'))
   ```

3. Create the Flask app

   ```
       #---create the flask app---
       app = Flask(__name__)
   ```

4. The following routes and functions will be created:

   - /diabetes/v1/predict

     - Method: POST
     - Content-Type: application/json
     - Input : JSON object containing the features
     - Function : Predicts whether a person has diabetes or not
     - Output : JSON object containing the prediction
     - Code

     ```
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
     ```

   - /diabetes/v1/results

     - Method: GET
     - Content-Type: application/json
     - Input : None
     - Function : Returns all the past predictions
     - Code

     ```
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
     ```

   - save(Glucose, Bmi, Age, Predicition, Confidence)

     - Input : Glucose, Bmi, Age, Predicition, Confidence
     - Function : Saves the prediction to the database
     - Output : None
     - Code

     ````
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
           ```
     ````

5. Run the Flask app
   ```
      if __name__ == '__main__':
           app.run(host='0.0.0.0', port=5000)
   ```

### Streamlit Client

1.  Create a new folder called client.
2.  Create a new file called home.py in the client folder.It is the home page of the web application.
    <details>
    <summary>Code</summary>

            import streamlit as st

            st.set_page_config(
            page_title="Diabetes Prediction App",
            page_icon=":hospital:",
            initial_sidebar_state="expanded",

            )

            st.write("# Diabetes Prediction App! 👋")

            st.markdown(
            """
            This app predicts the probability of a person having diabetes using the Pima Indians Diabetes Dataset!
            The algorithm used is **K-Nearest Neighbors** with **K = 19**. Given inputs of Glucose, BMI, and Age, the model predicts whether the person is diabetic or not.

                ### Documentation and references
                - Dataset source: [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
                - Strealit [documentation](https://docs.streamlit.io/en/stable/)
                - Flask [documentation](https://flask.palletsprojects.com/en/1.1.x/)
                - Ask a question in our [community

            """
            )

        </details>

3.  Create a subfolder called pages in the client folder.
4.  Create a new file called Predictor.py
    <details>
    <summary>Code</summary>

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

        </details>

5.  Create a new file called Results.py
    <details>
    <summary>Code</summary>

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

        </details>

## Run the application

    1. Run the server using the following command
    ```
    python server.app.py
    ```
    2. Open a new command prompt window and run the client using the following command
    ```
    streamlit run client/home.py
    ```
