import time

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from core.main.common import feature_extraction as f
from core.utils import config as c
from core.main.evaluation import confusion_matrix as cm, model_evaluation as e
from core.main.evaluation import time_complexity as t
from core.main.evaluation import cross_validation as cv

x_train_bi, x_test_bi, x_val_bi, y_train_bi, y_test_bi, y_val_bi = f.feature_extraction_smj(c.training_set, c.testing_set, c.val_set, True)
x_train_hax, x_test_hax, x_val_hex, y_train_hax, y_test_hax, y_val_hex = f.feature_extraction_smj(c.training_set, c.testing_set, c.val_set, False)


def decision_tree():
    t1 = time.time()
    model_bi, model_hex = DecisionTreeClassifier(criterion='entropy', max_depth=100, splitter='best', random_state=42), DecisionTreeClassifier(criterion='entropy', max_depth=100, splitter='best', random_state=42)

    model_bi.fit(x_train_bi, y_train_bi)
    model_hex.fit(x_train_hax, y_train_hax)

    y_predict_bi = model_bi.predict(x_test_bi)
    y_predict_hex = model_hex.predict(x_test_hax)

    # cm.confusion_matrix(y_test_bi, y_predict_bi, c.bidirectional_labels, "Decision Tree Confusion Matrix | 2-Ways")
    # cm.confusion_matrix(y_test_hax, y_predict_hex, c.labels, "Decision Tree Confusion Matrix | 6-Ways")

    t2 = time.time()
    t.time_complexity(t1, t2, "Decision Tree")

    e.eva_nnn(y_test_hax, y_predict_hex, y_test_bi, y_predict_bi, "Decision Tree Classifier: ")
    cv.validate(model_bi, model_hex, x_val_bi, y_val_bi, y_val_hex)


def logistic_regression():
    t1 = time.time()
    # Logistic Regression
    model_bi, model_hax = LogisticRegression(max_iter=30000), LogisticRegression(max_iter=30000)
    model_bi.fit(x_train_bi, y_train_bi)

    y_predict_bi = model_bi.predict(x_test_bi)
    y_predict_hax = model_bi.predict(x_test_hax)

    # cm.confusion_matrix(y_test_bi, y_predict_bi, c.bidirectional_labels, "Logistic Regression Confusion Matrix | 2-Ways (SMJ)")
    # cm.confusion_matrix(y_test_hax, y_predict_hax, c.labels, "Logistic Regression Confusion Matrix | 6-Ways (SMJ)")

    t2 = time.time()
    t.time_complexity(t1, t2, "Logistic Regression")

    e.eva_nnn(y_test_hax, y_predict_hax, y_test_bi, y_predict_bi, "Logistic Regression: ")
    cv.validate(model_bi, model_hax, x_val_bi, y_val_bi, y_val_hex)


def support_vector_machine():
    t1 = time.time()
    model_bi, model_hax = svm.SVC(kernel="linear", C=1, gamma="auto", degree=3), svm.SVC(kernel="linear", C=1, gamma="auto", degree=3)
    model_bi.fit(x_train_bi, y_train_bi)
    model_hax.fit(x_train_hax, y_train_hax)

    y_predict_bi = model_bi.predict(x_test_bi)
    y_predict_hax = model_hax.predict(x_test_hax)

    # cm.confusion_matrix(y_test_bi, y_predict_bi, c.bidirectional_labels, "SVM Confusion Matrix | 2-Ways (SMJ)")
    # cm.confusion_matrix(y_test_bi, y_predict_bi, c.labels, "SVM Confusion Matrix | 6-Ways (SMJ)")

    t2 = time.time()
    t.time_complexity(t1, t2, "Support Vector Machine")

    e.eva_nnn(y_test_hax, y_predict_hax, y_test_bi, y_predict_bi, "Support Vector Machine: ")
    cv.validate(model_bi, model_hax, x_val_bi, y_val_bi, y_val_hex)


def kNN(k_size=3):
    t1 = time.time()
    model_bi, model_hax = KNeighborsClassifier(n_neighbors=k_size), KNeighborsClassifier(n_neighbors=k_size)
    model_bi.fit(x_train_bi, y_train_bi)
    model_hax.fit(x_train_hax, y_train_hax)

    y_predict_bi = model_bi.predict(x_test_bi)
    y_predict_hax = model_hax.predict(x_test_hax)

    # cm.confusion_matrix(y_test_bi, y_predict_bi, c.bidirectional_labels, "kNN Confusion Matrix | 2-Ways (SMJ)")
    # cm.confusion_matrix(y_test_hax, y_test_hax, c.labels, "kNN Confusion Matrix | 6-Ways (SMJ)")

    t2 = time.time()
    t.time_complexity(t1, t2, f"K-Nearest Neighbor, k={k_size}")

    e.eva_nnn(y_test_hax, y_predict_hax, y_test_bi, y_predict_bi, "K-Nearest Neighbor: ")
    cv.validate(model_bi, model_hax, x_val_bi, y_val_bi, y_val_hex)
