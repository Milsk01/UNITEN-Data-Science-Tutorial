import pickle
import numpy as np

def load_model():

    model = pickle.load(open('diabetes.sav', 'rb'))
    return model

def main():
    Glucose = 65
    BMI = 70
    Age = 50

    loaded_model = load_model()

    prediction = loaded_model.predict([[Glucose, BMI, Age]])
    proba = loaded_model.predict_proba([[Glucose, BMI, Age]])

    result = "Non-diabetic" if prediction[0] else "Diabetic" 

    print(f"Glucose: {Glucose}, BMI: {BMI}, Age: {Age}")
    print(f"Prediction: {result}")

    print(f"Confidence: {str(round(np.amax(proba[0]) * 100 ,2))}%")

if __name__ == "__main__":
    main()

