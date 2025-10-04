from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib

app = FastAPI(docs_url=None, redoc_url=None)  # Remove FastAPI docs from top
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model and feature columns
model = joblib.load("exoplanet_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...), threshold: float = Form(...)):
    try:
        df = pd.read_csv(file.file)
        X = df[feature_columns]
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)

        results = [
            {
                "probability": round(float(prob), 3),
                "prediction": "🪐 Planet Detected" if p == 1 else "❌ No Planet",
            }
            for p, prob in zip(preds, probs)
        ]

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "results": results, "threshold": threshold},
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(e), "results": None},
        )
