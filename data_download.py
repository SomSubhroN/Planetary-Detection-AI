"""
data_download.py
Utilities to download Kepler / TESS light curves (uses lightkurve).
If you don't want to use it (no internet or lightkurve), skip this file and use the demo.
"""

import os
import pandas as pd

def save_lightcurve_as_csv(lc, outpath):
    # lc: a lightkurve LightCurve or similar object
    df = pd.DataFrame({
        "time": lc.time.value if hasattr(lc.time, "value") else lc.time,
        "flux": lc.flux.value if hasattr(lc.flux, "value") else lc.flux,
        "flux_err": getattr(lc, "flux_err", None)
    })
    df.to_csv(outpath, index=False)
    print(f"Saved {outpath}")

def download_kepler(target, outdir="data/kepler"):
    os.makedirs(outdir, exist_ok=True)
    try:
        import lightkurve as lk
    except ImportError:
        raise SystemExit("Install lightkurve to download Kepler/TESS data: pip install lightkurve")
    sr = lk.search_lightcurvefile(target, mission="Kepler")
    if len(sr) == 0:
        print("No Kepler data found for target:", target)
        return
    lcfs = sr.download_all()
    for i, lcf in enumerate(lcfs):
        lc = lcf.PDCSAP_FLUX.remove_nans()
        save_lightcurve_as_csv(lc, os.path.join(outdir, f"{target}_{i}.csv"))

def download_tess(target, outdir="data/tess"):
    os.makedirs(outdir, exist_ok=True)
    try:
        import lightkurve as lk
    except ImportError:
        raise SystemExit("Install lightkurve to download Kepler/TESS data: pip install lightkurve")
    sr = lk.search_lightcurve(target, mission="TESS")
    if len(sr) == 0:
        print("No TESS data found for target:", target)
        return
    lcs = sr.download_all()
    for i, lc in enumerate(lcs):
        lc = lc.remove_nans()
        save_lightcurve_as_csv(lc, os.path.join(outdir, f"{target}_{i}.csv"))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", choices=["kepler","tess"], required=True)
    ap.add_argument("--target", required=True, help="KIC/TIC/EPIC identifier or name")
    args = ap.parse_args()
    if args.mission == "kepler":
        download_kepler(args.target)
    else:
        download_tess(args.target)
