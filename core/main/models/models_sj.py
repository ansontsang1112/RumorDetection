from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from core.main.common import feature_extraction as f
from core.utils import config as c
from core.main.evaluation import confusion_matrix as cm, model_evaluation as e

x_train_bi, x_test_bi, y_train_bi, y_test_bi = f.feature_extraction_sj(c.training_set, c.testing_set, True)
x_train_hax, x_test_hax, y_train_hax, y_test_hax = f.feature_extraction_sj(c.training_set, c.testing_set, False)


def decision_tree():
    model_bi, model_hex = DecisionTreeClassifier(criterion='entropy', max_depth=100, splitter='best', random_state=42), DecisionTreeClassifier(criterion='entropy', max_depth=100, splitter='best', random_state=42)

    model_bi.fit(x_train_bi, y_train_bi)
    model_hex.fit(x_train_hax, y_train_hax)

    y_predict_bi = model_bi.predict(x_test_bi)
    y_predict_hex = model_hex.predict(x_test_hax)

    cm.confusion_matrix(y_test_bi, y_predict_bi, c.bidirectional_labels, "Decision Tree Confusion Matrix | 2-Ways")
    cm.confusion_matrix(y_test_hax, y_predict_hex, c.labels, "Decision Tree Confusion Matrix | 6-Ways")

    e.eva(y_test_hax, y_predict_hex, y_test_bi, y_predict_bi, "Decision Tree Classifier: ")


def logistic_regression():
    # Logistic Regression
    model_bi, model_hax = LogisticRegression(), LogisticRegression()
    model_bi.fit(x_train_bi, y_train_bi)

    y_predict_bi = model_bi.predict(x_test_bi)
    y_predict_hax = model_bi.predict(x_test_hax)

    cm.confusion_matrix(y_test_bi, y_predict_bi, c.bidirectional_labels, "Logistic Regression Confusion Matrix | 2-Ways")
    cm.confusion_matrix(y_test_hax, y_predict_hax, c.labels, "Logistic Regression Confusion Matrix | 6-Ways")

    e.eva(y_test_hax, y_predict_hax, y_test_bi, y_predict_bi, "Logistic Regression: ")


def support_vector_machine():
    model_bi, model_hax = svm.SVC(kernel="linear", C=1, gamma="auto", degree=3), svm.SVC(kernel="linear", C=1, gamma="auto", degree=3)
    model_bi.fit(x_train_bi, y_train_bi)
    model_hax.fit(x_train_hax, y_train_hax)

    y_predict_bi = model_bi.predict(x_test_bi)
    y_predict_hax = model_hax.predict(x_test_hax)

    cm.confusion_matrix(y_test_bi, y_predict_bi, c.bidirectional_labels, "SVM Confusion Matrix | 2-Ways")
    cm.confusion_matrix(y_test_bi, y_predict_bi, c.labels, "SVM Confusion Matrix | 6-Ways")

    e.eva(y_test_hax, y_predict_hax, y_test_bi, y_predict_bi, "Support Vector Machine: ")


def kNN(k_size=3):
    model_bi, model_hax = KNeighborsClassifier(n_neighbors=k_size), KNeighborsClassifier(n_neighbors=k_size)
    model_bi.fit(x_train_bi, y_train_bi)
    model_hax.fit(x_train_hax, y_train_hax)

    y_predict_bi = model_bi.predict(x_test_bi)
    y_predict_hax = model_hax.predict(x_test_hax)

    cm.confusion_matrix(y_test_bi, y_predict_bi, c.bidirectional_labels, "kNN Confusion Matrix | 2-Ways")
    cm.confusion_matrix(y_test_hax, y_test_hax, c.labels, "kNN Confusion Matrix | 2-Ways")

    e.eva(y_test_hax, y_predict_hax, y_test_bi, y_predict_bi, "K-Nearest Neighbor: ")


print("\n-------------- Scoring --------------")
decision_tree()
print("\n")
logistic_regression()
print("\n")
support_vector_machine()
print("\n")
kNN(7)
