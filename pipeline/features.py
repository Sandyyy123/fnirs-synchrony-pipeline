"""
Time-series feature extraction: cross-correlation, DTW, PLV, synchrony measures.
"""
import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt, correlate
from scipy.stats import zscore
import pandas as pd
from pathlib import Path


def bandpass(signal: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=-1)


def compute_plv(a: np.ndarray, b: np.ndarray, fs: float,
                band: tuple = (0.01, 0.1)) -> float:
    """Phase-Locking Value between two fNIRS channels."""
    a_f = bandpass(a, fs, *band)
    b_f = bandpass(b, fs, *band)
    phase_diff = np.angle(hilbert(a_f)) - np.angle(hilbert(b_f))
    return float(np.abs(np.mean(np.exp(1j * phase_diff))))


def compute_crosscorr(a: np.ndarray, b: np.ndarray, fs: float,
                      max_lag_s: float = 2.0) -> tuple:
    """
    Normalized cross-correlation. Returns (peak_r, lag_ms).
    Positive lag = b leads a.
    """
    max_lag = int(max_lag_s * fs)
    a_z, b_z = zscore(a), zscore(b)
    corr = correlate(a_z, b_z, mode="full") / len(a_z)
    lags = np.arange(-len(a_z) + 1, len(a_z))
    mid = len(lags) // 2
    window = corr[mid - max_lag: mid + max_lag + 1]
    lag_window = lags[mid - max_lag: mid + max_lag + 1]
    peak_idx = np.argmax(np.abs(window))
    return float(window[peak_idx]), float(lag_window[peak_idx] / fs * 1000)  # ms


def compute_dtw(a: np.ndarray, b: np.ndarray, window_frac: float = 0.1) -> float:
    """DTW distance with Sakoe-Chiba band constraint."""
    try:
        from tslearn.metrics import dtw as ts_dtw
        r = int(len(a) * window_frac)
        return float(ts_dtw(a.reshape(-1, 1), b.reshape(-1, 1),
                            global_constraint="sakoe_chiba", sakoe_chiba_radius=r))
    except ImportError:
        # Fallback: simple Euclidean distance
        return float(np.sqrt(np.mean((a - b) ** 2)))


FREQ_BANDS = {
    "delta": (0.01, 0.05),
    "low_freq": (0.05, 0.1),
    "cardiac": (0.5, 2.5),
}


def extract_synchrony_features(data: dict, out_dir: Path, n_channels: int = 8) -> pd.DataFrame:
    """
    Extract all synchrony features between channel pairs.
    For real data: pair channels across two participants.
    For demo: pair channels within the same recording.
    """
    hbo = data["hbo"][:n_channels]
    fs  = data["fs"]
    n   = hbo.shape[0]
    rows = []

    for i in range(n):
        for j in range(i + 1, n):
            row = {"ch_a": i, "ch_b": j}
            # Cross-correlation
            r, lag = compute_crosscorr(hbo[i], hbo[j], fs)
            row["xcorr_peak"] = r
            row["xcorr_lag_ms"] = lag
            # DTW
            row["dtw_dist"] = compute_dtw(hbo[i], hbo[j])
            # PLV per band
            for band_name, band_range in FREQ_BANDS.items():
                row[f"plv_{band_name}"] = compute_plv(hbo[i], hbo[j], fs, band_range)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "synchrony_features.csv", index=False)
    print(f"  Features: {len(df)} channel pairs, {len(df.columns)} features each")
    return df
