"""
run_pipeline.py
End-to-end runner. Use --demo to run synthetic data that demonstrates planet vs non-planet signals.
Use --lightcurves to point to CSV files (time,flux).
"""
import argparse, os, glob, json
import numpy as np
import pandas as pd

from preprocess import prepare_lightcurve_from_df
from features import extract_features
from vetting import vet_flags
from models import build_baseline_model, FEATURE_COLUMNS
from evaluate import evaluate_and_save

def simulate_lightcurve(kind="planet", n_points=3000, period=5.0, duration_frac=0.03, depth=0.005, noise=0.001):
    rng = np.random.default_rng()
    time = np.cumsum(np.full(n_points, 0.02))  # uniform cadence ~0.02 day (~30 min)
    flux = rng.normal(1.0, noise, size=n_points)
    if kind == "planet":
        phase = (time % period) / period
        in_tr = (phase < duration_frac) | (phase > 1 - duration_frac)
        flux[in_tr] -= depth
    elif kind == "ebinary":
        phase = (time % period) / period
        in_tr = (phase < duration_frac) | (phase > 1 - duration_frac)
        # deeper, asymmetric
        flux[in_tr] -= depth * (1.5 + 0.5 * np.sin(2*np.pi*phase))
        # secondary
        phase2 = ((time + 0.5*period) % period) / period
        sec = (phase2 < duration_frac) | (phase2 > 1 - duration_frac)
        flux[sec] -= 0.6 * depth
    elif kind == "variable":
        flux += 0.01 * np.sin(2*np.pi*time/period) + 0.003 * np.sin(2*np.pi*time/(0.7*period))
    else:
        # meteor-like short spike
        i = rng.integers(low=100, high=n_points-100)
        flux[i:i+2] -= 0.1  # sharp dip
    return time, flux

def load_lightcurves_from_globs(patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    data = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            t, f = prepare_lightcurve_from_df(df)
            if len(t) > 0:
                data.append((fp, t, f))
        except Exception as e:
            print("Failed to load", fp, ":", e)
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="Run synthetic demo")
    ap.add_argument("--lightcurves", nargs="*", help="Glob patterns to CSV files (time,flux[,flux_err])")
    ap.add_argument("--outdir", default="artifacts", help="Save outputs here")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    X = []
    y = []
    meta = []

    if args.demo:
        kinds = ["planet","planet","ebinary","variable","planet","ebinary","variable","planet","meteor"]
        for k in kinds:
            t, f = simulate_lightcurve(kind=k)
            feat = extract_features(t, f-1.0)  # features expect relative flux near 0
            row = [feat.get(c, np.nan) for c in FEATURE_COLUMNS]
            X.append(row)
            y.append(1 if k == "planet" else 0)
            meta.append({"kind": k, "feat": feat, "vet": vet_flags(feat)})
    elif args.lightcurves:
        data = load_lightcurves_from_globs(args.lightcurves)
        for fp, t, f in data:
            feat = extract_features(t, f)
            row = [feat.get(c, np.nan) for c in FEATURE_COLUMNS]
            X.append(row)
            # placeholder label: unknown; team should label or use provided training labels
            y.append(0)
            meta.append({"file": fp, "feat": feat, "vet": vet_flags(feat)})
    else:
        ap.print_help()
        return

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    # Baseline model
    model = build_baseline_model()
    # Safe: if all labels equal, skip fit
    if len(np.unique(y)) > 1:
        model.fit(X, y)
        proba = model.predict_proba(X)[:,1]
    else:
        # trivial case for demo: use depth threshold
        proba = np.array([0.8 if (row[2] > 0.001) else 0.2 for row in X])  # depth index in FEATURE_COLUMNS
    metrics = evaluate_and_save(y, proba, outdir=args.outdir)

    # Save per-object predictions and flags
    rows = []
    for i, m in enumerate(meta):
        rows.append({
            "index": i,
            "pred_proba": float(proba[i]),
            "label": int(y[i]),
            "meta": m
        })
    with open(os.path.join(args.outdir, "predictions.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # Save feature table
    feat_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    feat_df["label"] = y
    feat_df["pred_proba"] = proba
    feat_df.to_csv(os.path.join(args.outdir, "features_predictions.csv"), index=False)

    print("Done. Outputs saved to:", args.outdir)
    print("Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
