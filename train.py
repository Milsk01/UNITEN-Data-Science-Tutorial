from sklearn.neighbors import KNeighborsClassifier
from dataset import load_dataset
import pickle

def train():
    df = load_dataset()
    #---features---
    X = df[['Glucose','BMI','Age']]

    #---label---
    y = df.iloc[:,8]

    knn = KNeighborsClassifier(n_neighbors=19)
    knn.fit(X.values, y)

    return knn


def save (model, filename = 'output\diabetes.sav'):
    #---write to the file using write and binary mode---
    pickle.dump(model, open(filename, 'wb'))

if __name__ == "__main__":
    model = train()
    save(model)
    print("Model trained and saved successfully")