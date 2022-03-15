from sklearn.metrics import f1_score, accuracy_score


def eva(y_test, y_predict, title):
    print(f"{title}")
    print(f"F1 Score: {f1_score(y_test, y_predict, average='macro')}")
    print(f"Accuracy: {round(accuracy_score(y_test, y_predict) * 100, 2)}")
