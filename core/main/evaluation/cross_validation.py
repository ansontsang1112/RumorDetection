from sklearn.model_selection import KFold, cross_val_score
from core.utils import config as c


def validate(model_bi, model_hex, x_val, y_val_bi, y_val_hex):
    cv = KFold(n_splits=c.k_fold, random_state=1, shuffle=True)

    scores_bi = cross_val_score(model_bi, x_val, y_val_bi, scoring="accuracy", cv=cv, n_jobs=-1)
    scores_hex = cross_val_score(model_hex, x_val, y_val_hex, scoring="accuracy", cv=cv, n_jobs=-1)

    print(f"Cross-validation (Accuracy) (2-Ways / 6-Ways): {scores_bi.mean()} / {scores_hex.mean()}")
