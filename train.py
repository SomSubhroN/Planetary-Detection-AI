# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from clean import load_and_preprocess  # your preprocessing module
from imblearn.over_sampling import SMOTE  # handle class imbalance

# -----------------------------
# 1️⃣ Load and preprocess dataset
# -----------------------------
df = load_and_preprocess("dataset.csv")

# -----------------------------
# 2️⃣ Binary target: 1 if any planet, 0 if none
# -----------------------------
df["has_planet"] = df["nconfp"].apply(lambda x: 1 if x > 0 else 0)

# -----------------------------
# 3️⃣ Features and target
# -----------------------------
feature_columns = df.drop(columns=["nconfp", "has_planet"]).columns.tolist()
X = df[feature_columns]
y = df["has_planet"]

# Save feature columns for FastAPI
joblib.dump(feature_columns, "feature_columns.pkl")
print("✅ feature_columns.pkl saved")

# -----------------------------
# 4️⃣ Train/test split (stratified)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5️⃣ Apply SMOTE oversampling
# -----------------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"✅ Training data balanced: {y_train_res.value_counts().to_dict()}")

# -----------------------------
# 6️⃣ Train RandomForest with balanced class weight
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train_res, y_train_res)

# -----------------------------
# 7️⃣ Evaluate
# -----------------------------
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# -----------------------------
# 8️⃣ Save model
# -----------------------------
joblib.dump(model, "exoplanet_model.pkl")
print("✅ Model saved as exoplanet_model.pkl")
