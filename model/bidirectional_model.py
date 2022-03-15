import itertools

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import mains.bidirectional_modification as bd
import numpy as np
from sklearn import svm
import confusion_matrix as cm
import model_evaluation as e


def logistic_regression(df_train: pd.DataFrame, df_test: pd.DataFrame):
    x_train, x_test = df_train[['true', 'false']], df_test[['true', 'false']]
    y_train, y_test = df_train['label'], df_test['label']

    # Normalization
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_nor, x_test_nor = sc.transform(x_train), sc.transform(x_test)

    # Logistic Regression
    model = LogisticRegression()
    model.fit(x_train_nor, y_train)

    # print(lr.coef_)
    # print(lr.intercept_)
    y_predict = model.predict(x_test_nor)
    np.round(model.predict_proba(x_test_nor))

    cm.confusion_matrix(y_test, y_predict, "Logistic Regression Confusion Matrix")

    e.eva(y_test, y_predict, "Logistic Regression: ")


def support_vector_machine(df_train: pd.DataFrame, df_test: pd.DataFrame):
    x_train, x_test = df_train[['true', 'false']], df_test[['true', 'false']]
    y_train, y_test = df_train['label'], df_test['label']

    model = svm.SVC(kernel="linear", C=1, gamma="auto")
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    cm.confusion_matrix(y_test, y_predict, "SVM Confusion Matrix")

    e.eva(y_test, y_predict, "Support Vector Machine: ")


def kNN(df_train: pd.DataFrame, df_test: pd.DataFrame):
    x_train, x_test = df_train[['true', 'false']], df_test[['true', 'false']]
    y_train, y_test = df_train['label'], df_test['label']

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    cm.confusion_matrix(y_test, y_predict, "kNN Confusion Matrix")

    e.eva(y_test, y_predict, "K-Nearest Neighbor: ")


print("\n-------------- Scoring --------------\n")
logistic_regression(bd.training_dataset, bd.testing_dataset)
print("\n")
support_vector_machine(bd.training_dataset, bd.testing_dataset)
print("\n")
kNN(bd.training_dataset, bd.testing_dataset)
