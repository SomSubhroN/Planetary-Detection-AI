# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from clean import load_and_preprocess

# Load preprocessed dataset
df = load_and_preprocess("dataset.csv")

# Features and target
X = df.drop("nconfp", axis=1)
y = df["nconfp"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save trained model
joblib.dump(model, "exoplanet_model.pkl")
print("Model saved as exoplanet_model.pkl ✅")
