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
