from pathlib import Path
import pandas as pd
import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 1. Load dataset from UCI
dataset = fetch_ucirepo(id=336)
X = dataset.data.features.copy()
y = dataset.data.targets.squeeze().copy()

# 🔥 Fix column names (IMPORTANT)
X.columns = X.columns.str.strip().str.lower()

# 2. Clean text values
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = (
            X[col]
            .astype(str)
            .str.strip()
            .replace({"?": pd.NA, "nan": pd.NA})
        )

y = (
    y.astype(str)
    .str.strip()
    .str.lower()
    .replace({"ckd\t": "ckd"})
)

# 3. Correct column names (FIXED HERE)
numeric_cols = [
    "age", "bp", "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wbcc", "rbcc"   # ✅ FIXED
]

categorical_cols = [col for col in X.columns if col not in numeric_cols]

# Convert numeric columns
for col in numeric_cols:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# 6. Model
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

# 7. Train
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Save model
joblib.dump(model, MODEL_DIR / "kidney_model.joblib")

# 10. Save sample input
sample_input = X.iloc[[0]].copy()
sample_input.to_csv(BASE_DIR / "sample_input.csv", index=False)

print("\nSaved model to models/kidney_model.joblib")
print("Saved sample input to sample_input.csv")