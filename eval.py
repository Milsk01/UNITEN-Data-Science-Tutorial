from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pandas as pd
from dataset import load_dataset
import time



optimal_k = 0

def logistic_regression(X, y, cv=10):
    log_regress = linear_model.LogisticRegression()
    log_regress_score = cross_val_score(log_regress, X, y, cv=cv, scoring='accuracy').mean()
    return log_regress_score


def k_nearest_neighbours(X,y,cv = 10):
    cv_scores = []

    folds = cv

    #---creating odd list of K for KNN---
    ks = list(range(1,int(len(X) * ((folds - 1)/folds)), 2))

    #---perform k-fold cross validation---
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X, y, cv=folds, scoring='accuracy').mean()
        cv_scores.append(score)

    #---get the maximum score---
    best_knn_score = max(cv_scores)

    #---find the optimal k that gives the highest score---
    global optimal_k 
    optimal_k = ks[cv_scores.index(best_knn_score)]

    return best_knn_score

def svm_linear(X,y,cv):
    linear_svm = svm.SVC(kernel='linear')
    linear_svm_score = cross_val_score(linear_svm, X, y,
                                    cv=cv, scoring='accuracy').mean()
    
    return linear_svm_score


def svm_rbf(X,y,cv):
    rbf_svm = svm.SVC(kernel='rbf')
    rbf_svm_score = cross_val_score(rbf_svm, X, y,
                                    cv=cv, scoring='accuracy').mean()
    
    return rbf_svm_score


def eval():
    df = load_dataset()

    #---algorithms---
    algorithms = {
    "Logistic Regression": logistic_regression,
    "K Nearest Neighbors": k_nearest_neighbours,
    "SVM Linear Kernel": svm_linear,
    "SVM RBF Kernel": svm_rbf
    }   

    #---features---
    X = df[['Glucose','BMI','Age']]

    #---label---
    y = df.iloc[:,8]

    #---cross validation folds---
    cv = 10

    result = []

    for algo in algorithms:
        start = time.time()

        cross_val_score = algorithms[algo](X,y,cv)

        end = time.time()

        elapsed_time = end - start

        result.append(cross_val_score)

        ## if algo is knn then print the optimal k value as well
        if algo == "K Nearest Neighbors":
            print("%s: %f (k: %d, time: %f)" % (algo, cross_val_score, optimal_k, elapsed_time))
        else:
            print("%s: %f (time: %f)" % (algo, cross_val_score, elapsed_time))

    cv_mean = pd.DataFrame(result,index = algorithms.keys())
    cv_mean.columns=["Accuracy"]
    cv_mean.sort_values(by="Accuracy",ascending=False)

    # print the best algorithm 
    # if the algorithm is knn, print the optimal k as well 

    print("The best algorithm is %s with accuracy %f" % (cv_mean.idxmax().values[0],cv_mean.max().values[0]))


if __name__ == '__main__':
    eval()