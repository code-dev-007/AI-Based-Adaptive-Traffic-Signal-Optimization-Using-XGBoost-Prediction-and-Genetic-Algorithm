"""Linear Regression vs Random Forest vs XGBoost on the project's own pipeline.

Why this exists: Notebooks/Model_Comparison.ipynb splits by row order on a CSV that is sorted by
junction (so its test set is only the end of the last junction) and uses raw date parts without lag
features. Its numbers therefore are NOT comparable with the production XGBoost model. This module
re-runs the comparison with exactly the same features and chronological split as backend.model, so
all three models are scored on the same test rows.

Results are computed once, cached in memory and saved to models/model_comparison.json. The cache
is reused only while the dataset file and feature set are unchanged.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from backend import model as traffic_model
from backend import preprocessing as pp

CACHE_PATH = pp.PROJECT_ROOT / "models" / "model_comparison.json"
TEST_FRACTION = 0.2
SELECTION_METRIC = "MAE"

_lock = threading.Lock()
_memory: dict = {}


class ComparisonError(RuntimeError):
    """The comparison could not be produced (missing data, missing package, training failure)."""


def _fingerprint() -> dict:
    """Identifies the inputs: if any of these change, cached results are stale."""
    try:
        stat = pp.RAW_PATH.stat()
    except FileNotFoundError as error:
        raise ComparisonError(f"Dataset not found: {pp.RAW_PATH}. Put the CSV in data/raw/.") from error
    return {"dataset_size": stat.st_size, "dataset_mtime": int(stat.st_mtime), "features": pp.FEATURE_COLUMNS,
            "test_fraction": TEST_FRACTION}


def _make_models() -> dict:
    """Same feature columns for every model. Linear Regression one-hot encodes Junction (a label,
    not a quantity); the tree models use it as-is, as the production XGBoost model does."""
    linear = Pipeline([
        ("prep", ColumnTransformer([("junction", OneHotEncoder(handle_unknown="ignore"), ["Junction"])],
                                   remainder="passthrough")),
        ("model", LinearRegression()),
    ])
    return {
        "Linear Regression": linear,
        # Settings taken from the notebook.
        "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=1),
        # The production configuration, so this row is directly comparable with model_metadata.json.
        "XGBoost": traffic_model.make_regressor(),
    }


def compute() -> dict:
    """Train all three models on the earlier 80% of the timeline and score them on the later 20%."""
    fingerprint = _fingerprint()
    try:
        clean_df, _ = pp.clean(pp.load_raw())
        features = pp.build_features(clean_df)
        train, test = pp.split_chronological(features, TEST_FRACTION)
        if train.empty or test.empty:
            raise ValueError("Not enough data for a chronological train/test split.")
        models = _make_models()
    except ImportError as error:
        raise ComparisonError(str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise ComparisonError(str(error)) from error

    x_train, y_train = train[pp.FEATURE_COLUMNS], train[pp.TARGET]
    x_test, y_test = test[pp.FEATURE_COLUMNS], test[pp.TARGET]
    results = []
    for name, estimator in models.items():
        try:
            estimator.fit(x_train, y_train)
            predicted = np.clip(estimator.predict(x_test), 0, None)  # a vehicle count cannot be negative
        except Exception as error:  # noqa: BLE001
            raise ComparisonError(f"Training {name} failed: {error}") from error
        results.append({"name": name, **traffic_model.score(y_test, predicted)})

    best = min(results, key=lambda row: row["mae"])["name"]
    return {
        "models": results,
        "best_model": best,
        "selection_metric": SELECTION_METRIC,
        "evaluation": {
            "source": "reproducible pipeline (backend/model_comparison.py), not the notebook outputs",
            "target": pp.TARGET,
            "features": pp.FEATURE_COLUMNS,
            "split": "chronological, earlier 80% of timestamps train / later 20% test, all junctions",
            "train_rows": len(train),
            "test_rows": len(test),
            "train_range": [str(train["DateTime"].min()), str(train["DateTime"].max())],
            "test_range": [str(test["DateTime"].min()), str(test["DateTime"].max())],
            "computed_at": datetime.now().isoformat(timespec="seconds"),
        },
        "_fingerprint": fingerprint,
    }


def _read_saved(fingerprint: dict) -> dict | None:
    try:
        saved = json.loads(CACHE_PATH.read_text())
    except (OSError, ValueError):
        return None
    return saved if saved.get("_fingerprint") == fingerprint else None


def get_comparison(recompute: bool = False) -> dict:
    """Memory cache -> JSON file on disk -> train. Training happens once (not on every request)."""
    with _lock:
        fingerprint = _fingerprint()
        if not recompute:
            for source in (_memory.get("result"), _read_saved(fingerprint)):
                if source is not None and source.get("_fingerprint") == fingerprint:
                    _memory["result"] = source
                    return _public(source)
        result = compute()
        _memory["result"] = result
        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(json.dumps(result, indent=2))
        except OSError:
            pass  # an unwritable cache file must not break the endpoint
        return _public(result)


def _public(result: dict) -> dict:
    return {key: value for key, value in result.items() if not key.startswith("_")}


if __name__ == "__main__":
    print(json.dumps(get_comparison(recompute=True), indent=2))
