from sklearn.metrics import f1_score, accuracy_score


def eva(y_test, y_predict, y_test_bi, y_predict_bi, title):
    print(f"{title}")
    print(f"F1 Score (2-Ways / 6-Ways): {round(f1_score(y_test_bi, y_predict_bi, average='macro'), 3)} / {round(f1_score(y_test, y_predict, average='macro'), 3)}")
    print(f"Accuracy (2-Ways / 6-Ways): {round(accuracy_score(y_test_bi, y_predict_bi) * 100, 2)}% / {round(accuracy_score(y_test, y_predict) * 100, 2)}%")
