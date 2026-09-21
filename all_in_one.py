import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

dataset = pd.read_csv("creditcard.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0,
    stratify=y,
)

feature_counts = (
    10,
    15,
    X_train.shape[1],
)

def make_select_kbest(k):
    return SelectKBest(score_func=chi2, k=k)


selection_methods = {
    "SelectKBest": make_select_kbest,
}

def make_models():
    return {
        "LogisticRegression": LogisticRegression(
            random_state=0,
            max_iter=1000,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=10,
            criterion="entropy",
            random_state=0,
        ),
        "XGBoost": XGBClassifier(
            random_state=0,
            eval_metric="logloss",
        ),
    }

def evaluate(y_test, y_pred, y_pred_probability):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_probability)
    pr_auc = average_precision_score(y_test, y_pred_probability)

    cost_fp = 1
    cost_5 = fp * cost_fp + fn * 5
    cost_10 = fp * cost_fp + fn * 10
    cost_20 = fp * cost_fp + fn * 20

    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "cost_5": cost_5,
        "cost_10": cost_10,
        "cost_20": cost_20,
    }


results = []

for selection_name, selection_factory in selection_methods.items():
    for k in feature_counts:
        for model_name, model in make_models().items():

            print(f"\n{'=' * 60}")
            print(f"{selection_name} - {k} FEATURES - {model_name}")
            print(f"{'=' * 60}")

            pipeline = Pipeline([
                ("minmax", MinMaxScaler()),
                ("selection", selection_factory(k)),
                ("scaler", StandardScaler()),
                ("classifier", model),
            ])

            pipeline.fit(X_train, y_train)

            selector = pipeline.named_steps["selection"]
            selected_features = selector.get_support(indices=True)
            print("Selected features:", selected_features)

            y_pred_probability = pipeline.predict_proba(X_test)[:, 1]

            threshold = 0.8
            y_pred = (y_pred_probability >= threshold).astype(int)

            metrics = evaluate(y_test, y_pred, y_pred_probability)

            print("\nConfusion Matrix:")
            print(f"TN: {metrics['TN']}  FP: {metrics['FP']}")
            print(f"FN: {metrics['FN']}  TP: {metrics['TP']}")

            print("\n===== METRYKI =====")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-Score:  {metrics['f1']:.4f}")
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"PR-AUC:    {metrics['pr_auc']:.4f}")

            print("\n===== KOSZT =====")
            print(f"Koszt przy FN = 5:  {metrics['cost_5']}")
            print(f"Koszt przy FN = 10: {metrics['cost_10']}")
            print(f"Koszt przy FN = 20: {metrics['cost_20']}")

            results.append({
                "selection_method": selection_name,
                "n_features": k,
                "model": model_name,
                "selected_features": ", ".join(map(str, selected_features)),
                **metrics,
            })


results_df = pd.DataFrame(results)

print(f"\n{'=' * 60}")
print("PODSUMOWANIE")
print(f"{'=' * 60}")
print(results_df.to_string(index=False))

results_df.to_csv("wyniki_modele_klasyczne.csv", index=False)