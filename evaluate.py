"""
evaluate.py
Evaluation helpers: classification report + saving metrics.
"""
import os, json
from sklearn.metrics import classification_report, roc_auc_score

def evaluate_and_save(y_true, y_pred_proba, outdir="artifacts"):
    os.makedirs(outdir, exist_ok=True)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        auc = None
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    payload = {"auc": auc, "report": report}
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)
    return payload
