"""XGBoost traffic forecaster: train, evaluate against baselines, save, load, predict.

The target is the hourly vehicle COUNT of a junction (traffic volume), not individual vehicle movements.
Run `python -m backend.model` to train. Metrics are computed on the chronological
test set and written to models/model_metadata.json - nothing here is hard-coded.
"""
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend import preprocessing as pp

MODEL_PATH = pp.PROJECT_ROOT / "models" / "traffic_xgb_model.joblib"
META_PATH = pp.PROJECT_ROOT / "models" / "model_metadata.json"


class ModelNotFoundError(FileNotFoundError):
    """Raised when the model has not been trained yet."""


def make_regressor():
    """The XGBoost model (imported here so other modules work without xgboost installed)."""
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise ImportError("xgboost is not installed. Run: pip install -r requirements.txt") from error
    return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                        colsample_bytree=0.8, random_state=42, n_jobs=1)


def score(actual, predicted) -> dict:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def train_and_save(features: pd.DataFrame | None = None) -> dict:
    """Train on the earlier 80% of the timeline, evaluate on the later 20%, save model + metadata."""
    if features is None:
        clean_df, _ = pp.clean(pp.load_raw())
        features = pp.build_features(clean_df)
    train, test = pp.split_chronological(features, 0.2)
    model = make_regressor()
    model.fit(train[pp.FEATURE_COLUMNS], train[pp.TARGET])
    predicted = np.clip(model.predict(test[pp.FEATURE_COLUMNS]), 0, None)

    metrics = {
        "xgboost": score(test[pp.TARGET], predicted),
        "baseline_last_hour": score(test[pp.TARGET], test["lag_1"]),
        "baseline_same_hour_yesterday": score(test[pp.TARGET], test["lag_24"]),
        "baseline_same_hour_last_week": score(test[pp.TARGET], test["lag_168"]),
    }
    per_junction = {}
    for junction, rows in test.assign(predicted=predicted).groupby("Junction"):
        per_junction[str(int(junction))] = {"test_rows": len(rows), **score(rows[pp.TARGET], rows["predicted"])}
    best_baseline = min(("baseline_last_hour", "baseline_same_hour_yesterday", "baseline_same_hour_last_week"),
                        key=lambda k: metrics[k]["mae"])
    metadata = {
        "model": type(model).__name__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "features": pp.FEATURE_COLUMNS,
        "target": pp.TARGET,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_range": [str(train["DateTime"].min()), str(train["DateTime"].max())],
        "test_range": [str(test["DateTime"].min()), str(test["DateTime"].max())],
        "metrics": metrics,
        "per_junction_metrics": per_junction,
        "split": "chronological: earlier 80% of timestamps train, later 20% test (all junctions share the cut-off)",
        "note": "Target = hourly vehicle count per junction (traffic volume). One-hour-ahead predictions on held-out data.",
        "best_baseline": best_baseline,
        "mae_improvement_vs_best_baseline_pct": (
            100 * (1 - metrics["xgboost"]["mae"] / metrics[best_baseline]["mae"])
            if metrics[best_baseline]["mae"] > 0 else None
        ),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    META_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata


def load_model():
    """Return (model, metadata). Raises ModelNotFoundError if training has not been run."""
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise ModelNotFoundError("Model not trained yet. Run: python -m backend.model")
    metadata = json.loads(META_PATH.read_text())
    if metadata.get("features") != pp.FEATURE_COLUMNS:
        raise ModelNotFoundError("The saved model was trained with different features. Retrain with: python -m backend.model")
    return joblib.load(MODEL_PATH), metadata


def predict_rows(model, feature_rows: pd.DataFrame) -> np.ndarray:
    """One-hour-ahead predictions for rows that already have the feature columns."""
    return np.clip(model.predict(feature_rows[pp.FEATURE_COLUMNS]), 0, None)


def forecast(model, clean_df: pd.DataFrame, junction: int, horizon: int, as_of=None) -> pd.DataFrame:
    """Predict the `horizon` hours after the end of the data (or after `as_of`), feeding each prediction back in
    as history. With `as_of`, only records up to that hour are used, so the result can be checked against the
    later observed counts."""
    series = clean_df[clean_df["Junction"] == junction].sort_values("DateTime")
    if as_of is not None:
        series = series[series["DateTime"] <= as_of]
    if len(series) < max(pp.LAGS):
        raise ValueError(f"Junction {junction} needs at least {max(pp.LAGS)} hourly records before the forecast start.")
    values = series["Vehicles"].astype(float).tolist()
    last_time = series["DateTime"].iloc[-1]
    rows = []
    for step in range(1, horizon + 1):
        when = last_time + pd.Timedelta(hours=step)
        row = {"Junction": junction, "hour": when.hour, "day": when.day, "month": when.month,
               "weekday": when.dayofweek, "is_weekend": int(when.dayofweek >= 5)}
        for n in pp.LAGS:
            row[f"lag_{n}"] = values[-n]
        for n in pp.ROLLING_WINDOWS:
            row[f"roll_mean_{n}"] = float(np.mean(values[-n:]))
        predicted = float(predict_rows(model, pd.DataFrame([row]))[0])
        values.append(predicted)
        rows.append({"DateTime": when, "predicted": predicted})
    return pd.DataFrame(rows)


def evaluate_on_test_set(model, features: pd.DataFrame, junction: int) -> pd.DataFrame:
    """Actual vs predicted counts for one junction on the held-out (later) part of the timeline."""
    _, test = pp.split_chronological(features, 0.2)
    test = test[test["Junction"] == junction].sort_values("DateTime")
    return pd.DataFrame({"DateTime": test["DateTime"], "actual": test[pp.TARGET],
                         "predicted": predict_rows(model, test)})


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2))
