# train_xgb.py
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
import joblib
from clean import load_and_preprocess  # your existing preprocessing module

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

# Save feature columns for API
joblib.dump(feature_columns, "feature_columns.pkl")
print("✅ feature_columns.pkl saved")

# -----------------------------
# 4️⃣ Train/test split (stratified)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5️⃣ Train XGBoost classifier
# -----------------------------
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    scale_pos_weight=(len(y_train)-y_train.sum())/y_train.sum(),  # handle imbalance
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

# -----------------------------
# 6️⃣ Calibrate probabilities
# -----------------------------
calibrated_model = CalibratedClassifierCV(xgb_model, method='isotonic')
calibrated_model.fit(X_train, y_train)

# -----------------------------
# 7️⃣ Evaluate
# -----------------------------
y_pred = calibrated_model.predict(X_test)
print(classification_report(y_test, y_pred))

# -----------------------------
# 8️⃣ Save model
# -----------------------------
joblib.dump(calibrated_model, "exoplanet_model_xgb.pkl")
print("✅ Model saved as exoplanet_model_xgb.pkl")
