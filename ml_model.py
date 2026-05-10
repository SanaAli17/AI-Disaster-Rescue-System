import pandas as pd
import numpy as np
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
 

# 1. LOAD REAL FEMA DATASET

df = pd.read_csv("DisasterDeclarationsSummaries.csv")
 
# Keep useful columns
df = df[["incidentType", "declarationType", "designatedArea"]].dropna()
 
 

# 2. FEATURE ENGINEERING  
 
# Risk score based on incident type 
risk_map = {
    "Fire":           3,
    "Hurricane":      3,
    "Flood":          3,
    "Earthquake":     3,
    "Tornado":        2,
    "Severe Storm":   2,
    "Winter Storm":   2,
    "Coastal Storm":  2,
    "Snowstorm":      2,
    "Other":          1,
}
df["risk"] = df["incidentType"].map(risk_map).fillna(1)
 
# Categorical encodings
df["incident_code"] = df["incidentType"].astype("category").cat.codes
df["area_code"]     = df["designatedArea"].astype("category").cat.codes
 

np.random.seed(42)
area_base          = (df["area_code"] % 15) + 1          # 1–15 km base
noise              = np.random.normal(0, 2.5, size=len(df))
df["distance"]     = (area_base + noise).clip(lower=0.5).round(2)
 
# Urgency score derived from risk + incident type
df["urgency"] = (df["risk"] * 1.5 + df["incident_code"] * 0.3).round(2)
 
 

# 3. TARGET LABEL  (FEMA declarationType)

# DR = Major Disaster Declaration  → HIGH   (2)
# EM = Emergency Declaration        → MEDIUM (1)
# FM = Fire Mgmt Assistance Decl.  → LOW    (0)
 
priority_map = {"DR": 2, "EM": 1, "FM": 0}
df["priority"] = df["declarationType"].map(priority_map)
df = df.dropna(subset=["priority"])
df["priority"] = df["priority"].astype(int)
 
 
# 4. FINAL FEATURE MATRIX

FEATURE_COLS = ["distance", "risk", "urgency", "incident_code", "area_code"]
 
X = df[FEATURE_COLS].values
y = df["priority"].values
 

# 5. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)
 
# Scale features — helps kNN and Naive Bayes significantly
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
 
# 6. MODELS

knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
nb  = GaussianNB()
dt  = DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42)
 
knn.fit(X_train, y_train)
nb.fit(X_train, y_train)
dt.fit(X_train, y_train)
 
# 7. EVALUATION

def evaluate_models():
    print("\n--- ML MODEL EVALUATION USING FEMA DATASET ---")
    print(f"Dataset Size : {len(df):,} records")
    print(f"Features Used: {', '.join(FEATURE_COLS)}")
    print("Labels       : 0 = LOW (FM)  |  1 = MEDIUM (EM)  |  2 = HIGH (DR)")
 
    models = {"kNN": knn, "Naive Bayes": nb, "Decision Tree": dt}
 
    for name, model in models.items():
        preds = model.predict(X_test)
        print(f"\n{'─'*45}")
        print(f"{name}  —  Accuracy: {accuracy_score(y_test, preds):.3f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, preds))
        print("\nClassification Report:")
        print(classification_report(y_test, preds, zero_division=0))
 
 
def get_model_metrics() -> dict:
    """Return per-model weighted metrics for dashboard display."""
    models = {"knn": knn, "naive_bayes": nb, "decision_tree": dt}
    metrics = {}
 
    for key, model in models.items():
        preds = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )
        metrics[key] = {
            "accuracy":  round(float(accuracy_score(y_test, preds)), 3),
            "precision": round(float(precision), 3),
            "recall":    round(float(recall), 3),
            "f1":        round(float(f1), 3),
        }
 
    return metrics
 
 
# 8. PREDICTION FUNCTION  (used by simulation)
─
# Severity → incident_code mapping aligned with training encoding
_SEVERITY_CODE = {"critical": 3, "moderate": 2, "minor": 1}
 
# Most common area_code (mode) — stable default for live simulation input
_DEFAULT_AREA_CODE = int(df["area_code"].mode()[0])
 
# Incident-type urgency look-up for simulation
_SEVERITY_URGENCY = {
    "critical": round(_SEVERITY_CODE["critical"] * 1.5 * 1.3, 2),   # high-risk boost
    "moderate": round(_SEVERITY_CODE["moderate"] * 1.5 * 1.0, 2),
    "minor":    round(_SEVERITY_CODE["minor"]    * 1.5 * 0.7, 2),
}
 
def predict_priority(
    distance: float,
    risk: int,
    severity: str,
    area_code: int | None = None,
) -> tuple[int, int, int]:
    """
    Predict rescue priority for a live simulation input.
 
    Parameters
    ----------
    distance  : float  — ambulance-to-victim distance in km (0.5–30)
    risk      : int    — hazard risk score (1 = low, 2 = medium, 3 = high)
    severity  : str    — victim severity: 'critical' | 'moderate' | 'minor'
    area_code : int    — optional area encoding; defaults to dataset mode
 
    Returns
    -------
    (knn_pred, nb_pred, dt_pred)  — each is 0 (LOW), 1 (MEDIUM), or 2 (HIGH)
    """
    if severity not in _SEVERITY_CODE:
        raise ValueError(f"severity must be one of {list(_SEVERITY_CODE)}")
 
    incident_code = _SEVERITY_CODE[severity]
    urgency       = _SEVERITY_URGENCY[severity]
    ac            = area_code if area_code is not None else _DEFAULT_AREA_CODE
 
    raw_features = np.array([[distance, risk, urgency, incident_code, ac]], dtype=float)
 
    # Apply the same scaler fitted on training data
    features = scaler.transform(raw_features)
 
    knn_pred = int(knn.predict(features)[0])
    nb_pred  = int(nb.predict(features)[0])
    dt_pred  = int(dt.predict(features)[0])
 
    return knn_pred, nb_pred, dt_pred
 