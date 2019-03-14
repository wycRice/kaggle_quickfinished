import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

def load_data():
    print("Loading data...")
    df = pd.read_csv('../Data/train.csv')
    y_train = df[df.columns[0]]
    X_train = df[df.columns[1:]]
    X_test = pd.read_csv('../Data/test.csv')

    return X_train, y_train, X_test

if __name__ == '__main__':
    X, y, X_submission = load_data()

    # model_params = {'colsample_bylevel': 0.47,
    #                 'gamma': 0.3,
    #                 'learning_rate': 0.1,
    #                 'max_depth': 3,
    #                 'min_child_weight': 1,
    #                 'subsample': 0.5}
    # print('Modeling...')
    # clf = XGBClassifier()
    # clf.fit(X, y)
    # predictions = clf.predict_proba(X_submission)[:,1]
    # #y_submission = pd.DataFrame(predictions).max(axis=1)
    #
    # print( "Saving Results..")
    # temp = pd.read_csv('../Data/svm_benchmark.csv')
    # temp.PredictedProbability = predictions
    # temp.to_csv('xgboost.csv', index=None)

    clfs = [
        LogisticRegression,
        RandomForestClassifier,
        DecisionTreeClassifier
    ]
    for clf in clfs:
        name =  str(clf).split('.')[-1][:-2]
        print('Modeling...,', name)
        clf = clf()
        clf.fit(X,y)
        predictions = clf.predict_proba(X_submission)[:, 1]

        print("Saving Results..")
        temp = pd.read_csv('../Data/svm_benchmark.csv')
        temp.PredictedProbability = predictions
        temp.to_csv(f'{name}.csv', index=None)

