"""
features.py
Compute transit-search features using Box Least Squares (BLS) and other heuristics.
Designed to produce tabular features for lightweight ML (RandomForest).
"""
import numpy as np
from astropy.timeseries import BoxLeastSquares
from scipy.signal import periodogram

def compute_bls(time, flux, min_period=0.3, max_period=30.0, n_periods=2000, q_min=0.005, q_max=0.1):
    # time: days, flux: relative (around 0)
    try:
        model = BoxLeastSquares(time, flux)
        periods = np.linspace(min_period, max_period, n_periods)
        durations = np.linspace(q_min, q_max, 10) * periods[:, None]
        power = model.power(periods, durations)
        idx = np.nanargmax(power.power)
        best_period = power.period[idx]
        # select corresponding best duration & t0 (closest index)
        best_duration = power.duration[idx]
        best_t0 = power.transit_time[idx]
        # depth approximate
        phase_mask = ((time - best_t0 + 0.5*best_period) % best_period) / best_period
        in_transit = (phase_mask > 0.5 - best_duration/(2*best_period)) & (phase_mask < 0.5 + best_duration/(2*best_period))
        depth = np.nanmedian(flux[in_transit]) - np.nanmedian(flux[~in_transit]) if np.sum(in_transit) > 0 else 0.0
        snr = power.power[idx] / (np.nanstd(flux) + 1e-9)
        return {"period": float(best_period), "duration": float(best_duration), "t0": float(best_t0),
                "depth": float(abs(depth)), "snr": float(snr)}, power
    except Exception:
        return {"period": np.nan, "duration": np.nan, "t0": np.nan, "depth": np.nan, "snr": np.nan}, None

def transit_shape_metric(time, flux, period, t0, duration):
    if np.isnan(period) or np.isnan(duration):
        return np.nan
    phase = ((time - t0 + 0.5*period) % period) / period
    in_transit = (phase > 0.5 - duration/(2*period)) & (phase < 0.5 + duration/(2*period))
    if np.sum(in_transit) < 5:
        return np.nan
    depth = np.nanmedian(flux[in_transit]) - np.nanmedian(flux[~in_transit])
    edges = np.gradient(flux.astype(float))
    slope = np.nanmedian(np.abs(edges[in_transit]))
    return np.abs(slope) / (abs(depth) + 1e-9)

def odd_even_depth_diff(time, flux, period, t0, duration):
    if np.isnan(period):
        return np.nan
    phase = ((time - t0) % period) / period
    in_transit = (phase < duration/period) | (phase > 1 - duration/period)
    epochs = np.floor((time - t0) / period + 0.5).astype(int)
    odd = in_transit & (epochs % 2 != 0)
    even = in_transit & (epochs % 2 == 0)
    try:
        d_odd = np.nanmedian(flux[odd]) - np.nanmedian(flux[~odd]) if np.sum(odd) > 0 else 0.0
        d_even = np.nanmedian(flux[even]) - np.nanmedian(flux[~even]) if np.sum(even) > 0 else 0.0
        return abs(d_odd - d_even)
    except Exception:
        return np.nan

def secondary_eclipse_signal(time, flux, period, t0, duration):
    if np.isnan(period):
        return np.nan
    t0_sec = t0 + 0.5 * period
    phase = ((time - t0_sec) % period) / period
    in_sec = (phase < duration/period) | (phase > 1 - duration/period)
    if np.sum(in_sec) < 3:
        return 0.0
    return np.nanmedian(flux[in_sec]) - np.nanmedian(flux[~in_sec])

def periodicity_features(time, flux):
    try:
        fs = 1.0 / np.median(np.diff(time))
        f, pxx = periodogram(flux, fs=fs)
        pxx = pxx / (np.sum(pxx) + 1e-12)
        peak_power = float(np.nanmax(pxx))
        entropy = -float(np.nansum(pxx * np.log(pxx + 1e-12)))
        return {"psd_peak": peak_power, "psd_entropy": entropy}
    except Exception:
        return {"psd_peak": np.nan, "psd_entropy": np.nan}

def extract_features(time, flux):
    feat = {}
    bls, power = compute_bls(time, flux)
    feat.update(bls)
    feat["shape_metric"] = transit_shape_metric(time, flux, feat["period"], feat["t0"], feat["duration"])
    feat["odd_even_diff"] = odd_even_depth_diff(time, flux, feat["period"], feat["t0"], feat["duration"])
    feat["secondary_depth"] = secondary_eclipse_signal(time, flux, feat["period"], feat["t0"], feat["duration"])
    feat.update(periodicity_features(time, flux))
    return feat
