from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import io

# Load the trained model
model = joblib.load("exoplanet_model.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "🚀 Exoplanet Detection API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read uploaded CSV file
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    # Make predictions
    preds = model.predict(df)

    return {"predictions": preds.tolist()}
