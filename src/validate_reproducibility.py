import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_and_evaluate():
    """Train the same model twice and compare the results."""
    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "class_weight": "balanced",
    }

    results = []

    for run in range(2):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        }
        results.append(metrics)

        print(f"Run {run + 1}:")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-Score : {metrics['f1']:.4f}")

    reproducible = all(
        np.isclose(results[0][metric], results[1][metric])
        for metric in results[0]
    )

    print("\n--- Reproducibility Check ---")
    if reproducible:
        print("[SUCCESS] Both runs produced identical metrics.")
        print("[SUCCESS] Reproducibility validation passed!")
    else:
        print("[ERROR] The two runs produced different metrics.")
        raise SystemExit(1)


if __name__ == "__main__":
    train_and_evaluate()
