Planetary Detection AI — A Machine Learning Approach to Exoplanet Prediction
Developed for NASA Space Apps Challenge 2025

Developer: Som Subhro Nath
Team: Raven Hart Vexmoor
Bachelor of Science (CS) — University of Calcutta
IBM Design Thinking Practitioner | Hackerrank Software Developer
Independent AI Researcher | FastAPI Developer | XGBoost Specialist

🚀 Project Overview

The Planetary Detection AI is an intelligent machine learning system designed to identify the presence of exoplanets around stars based on astrophysical datasets.
Our mission: to give the brain to space telescopes, transforming the vast raw data from the cosmos into meaningful, interpretable planetary insights.

The AI uses:

XGBoost — for precise and optimized binary classification.

SMOTE (Synthetic Minority Over-sampling Technique) — to reduce bias and ensure balanced learning.

FastAPI — to create a lightweight, interactive web interface.

Python — as the core language for preprocessing, denoising, and model training.

🧠 Technical Summary
Component	Description
Algorithm	XGBoost Classifier (Binary Classification)
Bias Reduction	SMOTE applied to resampled training data
Deployment Stack	FastAPI + Uvicorn + Pyngrok
Dataset Source	NASA & Kaggle (star datasets)
Training Data Size	~4 GB (sampled subset of larger astronomical datasets)
Goal	Identify presence (1) or absence (0) of planetary bodies
Output	Probability score of planet existence
🧩 Workflow Overview

Data Collection:
Publicly available NASA and Kaggle datasets were used for stellar characteristics.

Preprocessing:

Missing values handled

Noise removed

Features normalized

Data merged across multiple files for consistency

Bias Correction:
SMOTE applied to achieve parity between classes (planet_present, planet_absent).

Model Training:
XGBoost classifier was trained on cleansed, resampled data.

Deployment (FastAPI):

Built a local API endpoint for inference

Created user-facing interface via FastAPI routes

Enabled public deployment using NGROK (FaaS demonstration-ready)

🧰 Project Structure
/Planetary-Detection-AI
│
├── app.py                     # FastAPI app for user inference
├── model.pkl                  # Pre-trained XGBoost model
├── main.ipynb                 # Jupyter Notebook (data cleaning + training)
├── dataset/                   # Processed dataset files
├── requirements.txt            # All dependencies
├── README.md                   # Documentation (this file)
└── demo_video.mp4              # Demonstration video for NASA Space App

⚙️ Installation and Deployment Guide

Follow these simple steps to deploy the system locally or online — no premium hosting required.

🔹 Step 1: Clone the Repository
git clone https://github.com/somsubhronath/Planetary-Detection-AI.git
cd Planetary-Detection-AI

🔹 Step 2: Install Dependencies

Make sure Python 3.9+ is installed, then run:

pip install -r requirements.txt

🔹 Step 3: Run the FastAPI App (Local Deployment)
uvicorn app:app --host 0.0.0.0 --port 8000


Now open your browser and visit:

http://127.0.0.1:8000


You’ll see the interactive GUI and the Swagger API docs at:

http://127.0.0.1:8000/docs

🧪 Using the API

Endpoint: /predict
Method: POST
Input JSON Example:

{
  "temperature": 5778,
  "luminosity": 1.0,
  "radius": 1.0,
  "metallicity": 0.02,
  "rotation": 25
}


Output JSON:

{
  "planet_presence": 1,
  "confidence_score": 0.87
}

🧬 Deployment Philosophy

This project was designed to run even on low-end machines, yet scaleable to quantum computing frameworks through Function-as-a-Service (FaaS) or serverless deployments.

FastAPI was chosen over Flask or Django because:

It’s asynchronous, hence more efficient for data inference.

It supports OpenAPI documentation natively.

It ensures easy integration with AI pipelines.

🪐 Future Scope

Integrating real-time telescope data streams from NASA APIs

Adapting model for multi-class exoplanet classification

Hosting the model on serverless edge computing environments

Using quantum-inspired algorithms for hyperparameter optimization

🏁 Conclusion

This project represents a confluence of astronomical science, machine learning precision, and open-source accessibility.
It empowers future researchers to explore the stars with intelligence — one prediction at a time. 🌌

“In the silence of space, we taught the stars to speak.” — Som