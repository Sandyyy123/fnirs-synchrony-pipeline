"""
Synthetic fNIRS demo data generator for testing the pipeline.
"""
import numpy as np


def generate_demo_data(n_channels: int = 16, duration_s: float = 300.0,
                       fs: float = 10.0) -> dict:
    """Generate synthetic hemodynamic signals with realistic properties."""
    rng = np.random.default_rng(seed=42)
    n_times = int(duration_s * fs)
    t = np.linspace(0, duration_s, n_times)

    # Simulate HbO: slow hemodynamic fluctuations + cardiac + noise
    hbo = np.zeros((n_channels, n_times))
    for i in range(n_channels):
        slow = 0.4 * np.sin(2 * np.pi * 0.03 * t + rng.uniform(0, 2 * np.pi))
        cardiac = 0.05 * np.sin(2 * np.pi * 1.1 * t)
        noise = 0.15 * rng.standard_normal(n_times)
        hbo[i] = slow + cardiac + noise

    # Add pairwise synchrony to first 4 channels (simulated cooperative condition)
    for i in range(4):
        hbo[i] += 0.3 * hbo[0] * rng.uniform(0.5, 1.0)

    hbr = -0.6 * hbo + 0.1 * rng.standard_normal((n_channels, n_times))

    return {"hbo": hbo, "hbr": hbr, "fs": fs, "bad_channels": np.array([]),
            "sci": np.ones(n_channels) * 0.85, "n_channels": n_channels}
