"""
models.py
Baseline classical ML model: RandomForest pipeline for tabular features.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import numpy as np

FEATURE_COLUMNS = ["period", "duration", "depth", "snr", "shape_metric",
                   "odd_even_diff", "secondary_depth", "psd_peak", "psd_entropy"]

def build_baseline_model():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    pre = ColumnTransformer(transformers=[("num", numeric_transformer, FEATURE_COLUMNS)])
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe
