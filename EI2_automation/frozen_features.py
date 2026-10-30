"""Unmodified relevant functions/definitions from src/extract_features.py; source SHA in SOURCE_HASHES.json."""
import math
import numpy as np
from scipy import signal

PSD_NPERSEG = 65_536

TARGET_TONES_HZ = (3265.0, 5554.0, 6897.0)

BANDS = (
    ("000000_000100", 0.0, 100.0),
    ("000100_001000", 100.0, 1_000.0),
    ("001000_002000", 1_000.0, 2_000.0),
    ("002000_004000", 2_000.0, 4_000.0),
    ("004000_006000", 4_000.0, 6_000.0),
    ("006000_008000", 6_000.0, 8_000.0),
    ("008000_010000", 8_000.0, 10_000.0),
    ("010000_020000", 10_000.0, 20_000.0),
    ("020000_040000", 20_000.0, 40_000.0),
    ("040000_070000", 40_000.0, 70_000.0),
    ("070000_100000", 70_000.0, 100_000.0),
    ("100000_200000", 100_000.0, 200_000.0),
    ("200000_400000", 200_000.0, 400_000.0),
    ("400000_700000", 400_000.0, 700_000.0),
    ("700000_nyquist", 700_000.0, math.inf),
)

def integrate_band(freq: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    high = min(high, float(freq[-1]))
    mask = (freq >= low) & (freq < high if high < freq[-1] else freq <= high)
    if not np.any(mask):
        return math.nan
    f = freq[mask]
    p = psd[mask]
    if len(f) == 1:
        df = float(freq[1] - freq[0]) if len(freq) > 1 else 1.0
        return float(p[0] * df)
    return float(np.trapezoid(p, f))

def periodogram_features(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    freq, psd = signal.welch(
        x, fs=fs, window="hann", nperseg=min(PSD_NPERSEG, len(x)),
        noverlap=0, detrend="constant", scaling="density", return_onesided=True,
    )
    psd = np.asarray(psd, dtype=np.float64)
    total = integrate_band(freq, psd, 0.0, float(freq[-1]))
    positive = np.maximum(psd, np.finfo(float).tiny)
    mass = positive / positive.sum()
    centroid = float(np.sum(freq * mass))
    bandwidth = float(np.sqrt(np.sum(((freq - centroid) ** 2) * mass)))
    cumulative = np.cumsum(mass)
    rolloff95 = float(freq[min(int(np.searchsorted(cumulative, 0.95)), len(freq) - 1)])
    entropy = float(-np.sum(mass * np.log(mass)) / np.log(len(mass)))
    flatness = float(np.exp(np.mean(np.log(positive))) / np.mean(positive))
    dom_mask = freq >= 10.0
    dominant = float(freq[dom_mask][int(np.argmax(psd[dom_mask]))])
    out: dict[str, float] = {
        "spectral_total_power": total,
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_rolloff95_hz": rolloff95,
        "spectral_entropy": entropy,
        "spectral_flatness": flatness,
        "dominant_frequency_hz": dominant,
    }
    for label, low, high in BANDS:
        power = integrate_band(freq, psd, low, high)
        out[f"band_power_{label}"] = power
        out[f"band_fraction_{label}"] = power / total if total > 0 and np.isfinite(power) else math.nan
    for center in TARGET_TONES_HZ:
        tag = str(int(round(center)))
        power = integrate_band(freq, psd, center - 100.0, center + 100.0)
        peak_mask = (freq >= center - 300.0) & (freq <= center + 300.0)
        local_mask = ((freq >= center - 1_000.0) & (freq <= center + 1_000.0)
                      & ~((freq >= center - 150.0) & (freq <= center + 150.0)))
        peak_freq = float(freq[peak_mask][int(np.argmax(psd[peak_mask]))])
        peak_psd = float(np.max(psd[peak_mask]))
        background = float(np.median(psd[local_mask]))
        prominence = 10.0 * math.log10(peak_psd / background) if peak_psd > 0 and background > 0 else math.nan
        out[f"tone_power_{tag}"] = power
        out[f"tone_peak_frequency_{tag}_hz"] = peak_freq
        out[f"tone_peak_psd_{tag}"] = peak_psd
        out[f"tone_prominence_{tag}_db"] = prominence
        out[f"tone_persistent_{tag}_6db"] = float(prominence >= 6.0) if np.isfinite(prominence) else math.nan
    return freq, psd, out
