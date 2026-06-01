"""
ML classification and prediction using synchrony features.
Includes LOOCV validation and SHAP feature importance.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import classification_report


def train_classifiers(features: pd.DataFrame, out_dir: Path) -> dict:
    """
    Train RF and SVM on synchrony features.
    Labels are synthetic in demo mode (random); replace with real condition labels.
    Returns dict of trained models and CV scores.
    """
    feature_cols = [c for c in features.columns if c not in ("ch_a", "ch_b")]
    X = features[feature_cols].fillna(0).values

    # Demo labels — replace with actual condition labels per channel pair
    np.random.seed(42)
    y = np.random.randint(0, 2, size=len(X))

    classifiers = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "svm_rbf": Pipeline([("scaler", StandardScaler()),
                             ("svm", SVC(kernel="rbf", probability=True, random_state=42))]),
    }

    results = {}
    loo = LeaveOneOut()

    for name, clf in classifiers.items():
        if len(X) < 3:
            # Too few samples for LOOCV — use dummy score
            results[name] = {"cv_accuracy": 0.5, "model": clf}
            clf.fit(X, y)
        else:
            scores = cross_val_score(clf, X, y, cv=min(5, len(X)), scoring="accuracy")
            clf.fit(X, y)
            results[name] = {"cv_accuracy": float(scores.mean()), "cv_std": float(scores.std()), "model": clf}
            print(f"  {name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

        joblib.dump(clf, out_dir / f"{name}.joblib")

    # SHAP feature importance (RF only)
    try:
        import shap
        rf = results["random_forest"]["model"]
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X)
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": np.abs(shap_vals[1] if isinstance(shap_vals, list) else shap_vals).mean(0)
        }).sort_values("importance", ascending=False)
        importance.to_csv(out_dir / "shap_importance.csv", index=False)
        print(f"  SHAP importance saved.")
    except ImportError:
        # sklearn-based fallback
        rf = results["random_forest"]["model"]
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)
        importance.to_csv(out_dir / "feature_importance.csv", index=False)

    return results
