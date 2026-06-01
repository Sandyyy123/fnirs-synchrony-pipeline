#!/usr/bin/env python3
"""
fNIRS Neural Synchrony Analysis Pipeline
=========================================
Entry point — runs all modules in sequence.

Usage:
    python main.py --data path/to/data.snirf --output results/
"""
import argparse
from pathlib import Path
from pipeline.preprocess import preprocess_fnirs
from pipeline.features import extract_synchrony_features
from pipeline.models import train_classifiers
from pipeline.visualize import generate_report

def main():
    parser = argparse.ArgumentParser(description="fNIRS Synchrony Pipeline")
    parser.add_argument("--data", required=True, help="Path to .snirf or .nirs file")
    parser.add_argument("--output", default="results/", help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo data")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.demo:
        print("[DEMO] Generating synthetic fNIRS data...")
        from pipeline.demo import generate_demo_data
        raw = generate_demo_data()
    else:
        raw = preprocess_fnirs(args.data, out)

    features = extract_synchrony_features(raw, out)
    models   = train_classifiers(features, out)
    generate_report(features, models, out)

    print(f"\nPipeline complete. Results in: {out}")

if __name__ == "__main__":
    main()
