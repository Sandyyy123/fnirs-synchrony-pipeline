"""
fNIRS preprocessing: artifact removal, Beer-Lambert conversion, signal QA.
"""
import numpy as np
import mne
from pathlib import Path


def scalp_coupling_index(raw_intensity: np.ndarray, fs: float, window: float = 3.0) -> np.ndarray:
    """
    Compute Scalp Coupling Index (SCI) per channel.
    SCI = correlation between 850nm and 690nm at cardiac frequency (0.5-2.5 Hz).
    Returns array of SCI values in [0, 1].
    """
    from scipy.signal import butter, sosfiltfilt, welch
    sos = butter(4, [0.5, 2.5], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, raw_intensity, axis=-1)
    n_ch = filtered.shape[0] // 2
    sci = np.zeros(n_ch)
    for i in range(n_ch):
        a, b = filtered[i], filtered[i + n_ch]
        corr = np.corrcoef(a, b)[0, 1]
        sci[i] = max(0.0, corr)
    return sci


def apply_beer_lambert(raw_intensity, ppf: float = 6.0) -> np.ndarray:
    """Modified Beer-Lambert Law: convert intensity to HbO/HbR concentrations."""
    # Absorption coefficients at 690/850 nm (mm^-1 uM^-1)
    E = np.array([[0.15174, 0.67100],   # HbO2
                  [1.33537, 0.17526]])  # HHb
    OD = -np.log(raw_intensity / raw_intensity[:, :1])  # optical density change
    n_wl = OD.shape[0] // 2
    # Solve: E * c = OD / (ppf * dist)
    # Simplified: pseudoinverse across wavelength pairs
    E_inv = np.linalg.pinv(E)
    conc = ppf * E_inv @ OD[:n_wl * 2].reshape(2, n_wl, -1).mean(axis=1)
    return conc  # shape: (2, n_channels, n_times)


def motion_correct_tddr(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Temporal Derivative Distribution Repair (TDDR) motion correction.
    Reference: Fishburn et al. (2019) NeuroImage.
    """
    from scipy.stats import median_abs_deviation
    d = np.diff(signal, axis=-1)
    mad = median_abs_deviation(d, axis=-1, keepdims=True)
    robust_d = np.where(np.abs(d) > 4.685 * mad, 0.0, d)
    corrected = np.concatenate([signal[..., :1], signal[..., :1] + np.cumsum(robust_d, axis=-1)], axis=-1)
    return corrected


def preprocess_fnirs(data_path: str, out_dir: Path) -> dict:
    """
    Full preprocessing pipeline:
      1. Load .snirf / .nirs
      2. Compute SCI and flag bad channels
      3. Apply Beer-Lambert
      4. Bandpass filter (0.01-0.1 Hz)
      5. Motion correction (TDDR)
    Returns dict with HbO, HbR arrays and metadata.
    """
    path = str(data_path)
    if path.endswith(".snirf"):
        raw = mne.io.read_raw_snirf(path, preload=True, verbose=False)
    else:
        raw = mne.io.read_raw_nirx(path, preload=True, verbose=False)

    fs = raw.info["sfreq"]
    data = raw.get_data()

    sci = scalp_coupling_index(data, fs)
    bad_mask = sci < 0.5
    n_bad = bad_mask.sum()
    print(f"  SCI: {(~bad_mask).sum()} good / {n_bad} bad channels")

    hbo_hbr = apply_beer_lambert(data)
    hbo = motion_correct_tddr(hbo_hbr[0], fs)
    hbr = motion_correct_tddr(hbo_hbr[1], fs)

    # Bandpass filter
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [0.01, 0.1], btype="bandpass", fs=fs, output="sos")
    hbo = sosfiltfilt(sos, hbo, axis=-1)
    hbr = sosfiltfilt(sos, hbr, axis=-1)

    import numpy as np
    np.save(out_dir / "hbo.npy", hbo)
    np.save(out_dir / "hbr.npy", hbr)

    return {"hbo": hbo, "hbr": hbr, "fs": fs, "bad_channels": np.where(bad_mask)[0],
            "sci": sci, "n_channels": hbo.shape[0]}
