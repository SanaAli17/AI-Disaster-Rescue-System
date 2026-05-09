from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

# DATASET

# Features: [distance, risk, severity_score]
X = np.array([
    [5, 2, 3], [6, 3, 3], [7, 2, 3],
    [4, 1, 3], [8, 3, 3],

    [10, 5, 2], [12, 6, 2], [11, 4, 2],
    [9, 5, 2], [13, 6, 2],

    [15, 8, 1], [14, 7, 1], [13, 6, 1],
    [16, 9, 1], [12, 7, 1]
])

# Labels: 2 = High, 1 = Medium, 0 = Low
y = np.array([
    2, 2, 2, 2, 2,   # high priority (severity 3)
    1, 1, 1, 1, 1,   # medium
    0, 0, 0, 0, 0    # low
])


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# MODELS

knn = KNeighborsClassifier(n_neighbors=3)
nb = GaussianNB()
dt = DecisionTreeClassifier()

knn.fit(X_train, y_train)
nb.fit(X_train, y_train)
dt.fit(X_train, y_train)


# EVALUATION

def evaluate_models():
    print("\n--- ML MODEL EVALUATION ---")

    models = {
        "kNN": knn,
        "Naive Bayes": nb,
        "Decision Tree": dt
    }

    for name, model in models.items():
        preds = model.predict(X_test)
        print(f"\n{name} Accuracy:", accuracy_score(y_test, preds))

        cm = confusion_matrix(y_test, preds)

        print("\nConfusion Matrix:")
        print(cm)

        print(classification_report(y_test, preds))


def get_model_metrics():
    """Return per-model metrics for dashboard display."""
    models = {
        "knn": knn,
        "naive_bayes": nb,
        "decision_tree": dt,
    }

    metrics = {}
    for key, model in models.items():
        preds = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )
        metrics[key] = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 3),
            "precision": round(float(precision), 3),
            "recall": round(float(recall), 3),
            "f1": round(float(f1), 3),
        }

    return metrics


# PREDICTION FUNCTION

def predict_priority(distance, risk, severity):
    severity_map = {
        "critical": 3,
        "moderate": 2,
        "minor": 1
    }

    features = np.array([[distance, risk, severity_map[severity]]])

    knn_pred = knn.predict(features)[0]
    nb_pred = nb.predict(features)[0]
    dt_pred = dt.predict(features)[0]

    return knn_pred, nb_pred, dt_pred