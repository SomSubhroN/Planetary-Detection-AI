"""
vetting.py
Use features to create human-interpretable vetting flags:
- odd-even mismatch -> eclipsing binaries
- secondary eclipse -> eclipsing binaries or self-luminous companion
- v-shape metric -> grazing/stellar eclipse
- variability domination -> star spots/ pulsations
- short single events -> moving objects (meteor/asteroid)
"""
import numpy as np

def vet_flags(feat):
    flags = {}
    depth = abs(feat.get("depth", 0.0) + 1e-9)
    # Eclipsing binary indicators
    flags["odd_even_flag"] = (feat.get("odd_even_diff", 0.0) > 0.1 * depth)
    flags["secondary_flag"] = (abs(feat.get("secondary_depth", 0.0)) > 0.3 * depth)
    flags["v_shape_flag"] = (feat.get("shape_metric", np.nan) > 0.6)
    # implausible duration fraction (very long duration relative to period)
    period = feat.get("period", np.nan)
    duration = feat.get("duration", np.nan)
    try:
        flags["duration_flag"] = (duration / period) > 0.15
    except Exception:
        flags["duration_flag"] = False
    # variability dominance
    flags["var_flag"] = (feat.get("psd_peak", 0.0) > 0.25) and (feat.get("psd_entropy", 999.0) < 6.0)
    # moving object heuristic (short, single, very asymmetric event) - placeholder: rely on data shape / cadence externally
    flags["moving_object_flag"] = False  # requires event-level analysis; default False here
    flags["any_flag"] = any(flags.get(k, False) for k in flags if k.endswith("_flag"))
    return flags
