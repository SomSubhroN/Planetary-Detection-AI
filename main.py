from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib

app = FastAPI()

# Load trained model and feature columns
model = joblib.load("exoplanet_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

@app.post("/predict/")
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    try:
        # Load CSV
        df = pd.read_csv(file.file)

        # Keep only the features used in training
        X = df[feature_columns]

        # Predict probabilities
        probs = model.predict_proba(X)[:, 1]

        # Apply threshold
        preds = (probs >= threshold).astype(int)

        # Return predictions with probability
        results = []
        for p, prob in zip(preds, probs):
            results.append({
                "prediction": int(p),
                "probability": float(prob),
                "label": "Planet detected" if p == 1 else "No planet"
            })

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}
