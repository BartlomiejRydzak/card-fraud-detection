import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

dataset = pd.read_csv("creditcard.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0,
    stratify=y
)

feature_counts = (
    10,
    15,
    X_train.shape[1]
)

for k in feature_counts:

    print(f"\n{'=' * 60}")
    print(f"SELECTKBEST - {k} FEATURES")
    print(f"{'=' * 60}")


    pipeline = Pipeline([
        ("minmax", MinMaxScaler()),

        ("selection", SelectKBest(
            score_func=chi2,
            k=k
        )),

        ("scaler", StandardScaler()),

        ("classifier", LogisticRegression(
            random_state=0,
            max_iter=1000
        ))
    ])


    pipeline.fit(
        X_train,
        y_train
    )

    selector = pipeline.named_steps["selection"]

    print(
        "Selected features:",
        selector.get_support(indices=True)
    )

    y_pred_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    threshold = 0.5

    y_pred = (
        y_pred_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nTN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_pred_probability
    )

    pr_auc = average_precision_score(
        y_test,
        y_pred_probability
    )

    cost_fp = 1

    cost_5 = fp * cost_fp + fn * 5
    cost_10 = fp * cost_fp + fn * 10
    cost_20 = fp * cost_fp + fn * 20


    print("\n===== METRYKI =====")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\n===== KOSZT =====")

    print(f"Koszt przy FN = 5:  {cost_5}")
    print(f"Koszt przy FN = 10: {cost_10}")
    print(f"Koszt przy FN = 20: {cost_20}")


for k in feature_counts:

    print(f"\n{'=' * 60}")
    print(f"SELECTFROMMODEL - {k} FEATURES")
    print(f"{'=' * 60}")


    pipeline = Pipeline([
        ("minmax", MinMaxScaler()),

        ("selection", SelectFromModel(
            estimator=LinearSVC(
                C=0.01,
                penalty="l1",
                dual=False,
                max_iter=5000
            ),
            threshold=-float("inf"),
            max_features=k
        )),

        ("scaler", StandardScaler()),

        ("classifier", LogisticRegression(
            random_state=0,
            max_iter=1000
        ))
    ])


    pipeline.fit(
        X_train,
        y_train
    )

    selector = pipeline.named_steps["selection"]

    print(
        "Selected features:",
        selector.get_support(indices=True)
    )

    y_pred_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    threshold = 0.5

    y_pred = (
        y_pred_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nTN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_pred_probability
    )

    pr_auc = average_precision_score(
        y_test,
        y_pred_probability
    )

    cost_fp = 1

    cost_5 = fp * cost_fp + fn * 5
    cost_10 = fp * cost_fp + fn * 10
    cost_20 = fp * cost_fp + fn * 20


    print("\n===== METRYKI =====")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\n===== KOSZT =====")

    print(f"Koszt przy FN = 5:  {cost_5}")
    print(f"Koszt przy FN = 10: {cost_10}")
    print(f"Koszt przy FN = 20: {cost_20}")
