# fNIRS Synchrony Pipeline

**Functional near-infrared spectroscopy (fNIRS) analysis pipeline for social neuroscience research.**

Processes raw fNIRS recordings from dyadic social paradigms to extract neural synchrony features and train ML classifiers.

## What this does

- **Preprocessing**: Modified Beer-Lambert Law, artifact detection (SCI), motion correction (TDDR), bandpass filtering
- **Signal QA**: Per-channel scalp coupling index, bad channel flagging
- **Time-series analysis**: Cross-correlation (peak r + lag), Dynamic Time Warping (DTW with Sakoe-Chiba band), Phase-Locking Value (PLV) per frequency band
- **Machine learning**: Random Forest and SVM classifiers with LOOCV validation, SHAP feature importance
- **Visualization**: HTML summary reports, channel-pair feature matrices

## Architecture

```
fnirs-synchrony-pipeline/
├── main.py                    # Entry point
├── pipeline/
│   ├── preprocess.py          # Beer-Lambert, TDDR, SCI, bandpass
│   ├── features.py            # Cross-corr, DTW, PLV, synchrony features
│   ├── models.py              # RF, SVM, LOOCV, SHAP
│   ├── visualize.py           # Report generation
│   └── demo.py                # Synthetic data generator for testing
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Demo mode (no data needed):**
```bash
python main.py --demo --output results/
```

**Real fNIRS data:**
```bash
python main.py --data path/to/recording.snirf --output results/
```

## Key methods

| Analysis | Implementation | Output |
|----------|----------------|--------|
| Cross-correlation | scipy.signal.correlate | Peak r, lag (ms) |
| DTW | tslearn (Sakoe-Chiba band) | DTW distance |
| PLV | Hilbert transform per band | PLV in [0,1] |
| SCI artifact detection | Cardiac-frequency correlation | SCI per channel |
| Motion correction | TDDR (Fishburn et al. 2019) | Cleaned signal |

## Author

Dr. Sandeep Grover | PhD Data Science | 11 neuroscience publications (Neurology, Movement Disorders, Annals of Neurology)
