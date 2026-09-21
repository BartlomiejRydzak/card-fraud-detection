import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

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


class KerasAutoencoder:

    def __init__(
        self,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose

    def _build_model(self, n_features):

        model = tf.keras.Sequential([
            tf.keras.layers.Input(
                shape=(n_features,)
            ),

            # Encoder
            tf.keras.layers.Dense(
                64,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu",
            ),

            # Bottleneck
            tf.keras.layers.Dense(
                16,
                activation="relu",
            ),

            # Decoder
            tf.keras.layers.Dense(
                32,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                64,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                n_features,
                activation="linear",
            ),
        ])

        model.compile(
            optimizer="adam",
            loss="mse",
        )

        return model

    def fit(self, X):

        X = np.asarray(X)

        self.n_features_in_ = X.shape[1]

        self.model_ = self._build_model(
            self.n_features_in_
        )

        self.model_.fit(
            X,
            X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=self.verbose,
        )

        return self

    def reconstruction_error(self, X):

        X = np.asarray(X)

        X_reconstructed = self.model_.predict(
            X,
            verbose=0,
        )

        error = np.mean(
            np.square(
                X - X_reconstructed
            ),
            axis=1,
        )

        return error

    def reconstruct(self, X):

        X = np.asarray(X)

        return self.model_.predict(
            X,
            verbose=0,
        )


def evaluate(
    y_test,
    y_pred,
    y_pred_probability,
):

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
    ).ravel()

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_pred_probability,
    )

    pr_auc = average_precision_score(
        y_test,
        y_pred_probability,
    )

    cost_fp = 1

    cost_5 = (
        fp * cost_fp
        + fn * 5
    )

    cost_10 = (
        fp * cost_fp
        + fn * 10
    )

    cost_20 = (
        fp * cost_fp
        + fn * 20
    )

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
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

        print(f"\n{'=' * 60}")
        print(
            f"{selection_name} - "
            f"{k} FEATURES - AUTOENCODER"
        )
        print(f"{'=' * 60}")


        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        minmax = MinMaxScaler()

        selector = selection_factory(k)

        scaler = StandardScaler()


        X_train_scaled = minmax.fit_transform(
            X_train
        )

        X_test_scaled = minmax.transform(
            X_test
        )


        X_train_selected = selector.fit_transform(
            X_train_scaled,
            y_train,
        )

        X_test_selected = selector.transform(
            X_test_scaled
        )


        X_train_processed = scaler.fit_transform(
            X_train_selected
        )

        X_test_processed = scaler.transform(
            X_test_selected
        )


        selected_features = (
            selector.get_support(
                indices=True
            )
        )

        print(
            "Selected features:",
            selected_features,
        )


        # ----------------------------------------------------
        # AUTOENCODER
        # ----------------------------------------------------

        autoencoder = KerasAutoencoder(
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
        )


        autoencoder.fit(
            X_train_processed
        )


        # ----------------------------------------------------
        # BŁĄD REKONSTRUKCJI
        # ----------------------------------------------------

        train_error = (
            autoencoder.reconstruction_error(
                X_train_processed
            )
        )

        test_error = (
            autoencoder.reconstruction_error(
                X_test_processed
            )
        )


        # ----------------------------------------------------
        # PRÓG KLASYFIKACJI
        # ----------------------------------------------------

        threshold = np.percentile(
            train_error,
            95,
        )

        y_pred_probability = (
            test_error
        )

        y_pred = (
            test_error >= threshold
        ).astype(int)


        # ----------------------------------------------------
        # WYNIKI
        # ----------------------------------------------------

        metrics = evaluate(
            y_test,
            y_pred,
            y_pred_probability,
        )


        print("\nThreshold:")
        print(
            f"{threshold:.6f}"
        )


        print("\nConfusion Matrix:")

        print(
            f"TN: {metrics['TN']}  "
            f"FP: {metrics['FP']}"
        )

        print(
            f"FN: {metrics['FN']}  "
            f"TP: {metrics['TP']}"
        )


        print("\n===== METRYKI =====")

        print(
            f"Accuracy:  "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall:    "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1-Score:  "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC:   "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC:    "
            f"{metrics['pr_auc']:.4f}"
        )


        print("\n===== KOSZT =====")

        print(
            f"Koszt przy FN = 5:  "
            f"{metrics['cost_5']}"
        )

        print(
            f"Koszt przy FN = 10: "
            f"{metrics['cost_10']}"
        )

        print(
            f"Koszt przy FN = 20: "
            f"{metrics['cost_20']}"
        )


        results.append({
            "selection_method": selection_name,
            "n_features": k,
            "model": "Autoencoder",
            "selected_features": ", ".join(
                map(str, selected_features)
            ),
            "threshold": threshold,
            **metrics,
        })


results_df = pd.DataFrame(
    results
)


print(f"\n{'=' * 60}")
print("PODSUMOWANIE - AUTOENCODER")
print(f"{'=' * 60}")


print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    "wyniki_autoencoder.csv",
    index=False,
)
