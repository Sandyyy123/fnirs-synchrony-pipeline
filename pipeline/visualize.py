"""Visualization and report generation for fNIRS synchrony analysis."""
import pandas as pd
import numpy as np
from pathlib import Path


def generate_report(features: pd.DataFrame, models: dict, out_dir: Path):
    """Generate HTML summary report."""
    lines = ["<html><body style='font-family:sans-serif;padding:20px'>"]
    lines.append("<h1>fNIRS Synchrony Analysis Report</h1>")
    lines.append(f"<p>Channel pairs analyzed: {len(features)}</p>")
    lines.append("<h2>Synchrony Feature Summary</h2>")
    lines.append(features.describe().to_html())
    lines.append("<h2>Model Performance</h2><ul>")
    for name, res in models.items():
        acc = res.get("cv_accuracy", "N/A")
        std = res.get("cv_std", 0)
        lines.append(f"<li>{name}: {acc:.3f} ± {std:.3f}</li>")
    lines.append("</ul></body></html>")
    report_path = out_dir / "report.html"
    report_path.write_text("\n".join(lines))
    print(f"  Report: {report_path}")
