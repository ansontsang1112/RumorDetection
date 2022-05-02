from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def eva_nnn(y_test, y_predict, y_test_bi, y_predict_bi, title):
    print(f"{title}")
    print(
        f"F1 Score (2-Ways / 6-Ways): {round(f1_score(y_test_bi, y_predict_bi, average='micro'), 3)} / {round(f1_score(y_test, y_predict, average='micro'), 3)}")
    print(
        f"Accuracy (2-Ways / 6-Ways): {round(accuracy_score(y_test_bi, y_predict_bi) * 100, 2)}% / {round(accuracy_score(y_test, y_predict) * 100, 2)}%")


def eva_nn(model_bi, model_hex, x_test, y_test_bi, y_test_hex, title):
    print(f"{title}")

    eva_bi = model_bi.evaluate(x_test, y_test_bi)
    eva_hex = model_hex.evaluate(x_test, y_test_hex)

    print(
        f"\nF1 Score (2-Ways / 6-Ways): {eva_bi[2]} / {eva_hex[2]}")
    print(
        f"Accuracy (2-Ways / 6-Ways): {round(eva_bi[1] * 100, 2)}% / {round(eva_hex[1] * 100, 2)}%")

