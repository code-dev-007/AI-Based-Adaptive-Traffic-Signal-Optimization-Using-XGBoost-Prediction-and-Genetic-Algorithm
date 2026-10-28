import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from backend import model as tm
from backend import preprocessing as pp


def _features(hours=600):
    times = pd.date_range("2024-01-01", periods=hours, freq="h")
    rows = [(i, t, j, 10 + 3 * j + (8 if t.hour in (7, 8, 9, 17, 18, 19) else 0)) for j in (1, 2) for i, t in enumerate(times)]
    clean_df, _ = pp.clean(pd.DataFrame(rows, columns=pp.REQUIRED_COLUMNS))
    return clean_df, pp.build_features(clean_df)


def _use_tmp_model(monkeypatch, tmp_path, regressor):
    monkeypatch.setattr(tm, "make_regressor", regressor)
    monkeypatch.setattr(tm, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(tm, "META_PATH", tmp_path / "m.json")


def test_missing_model_raises_clear_error(monkeypatch, tmp_path):
    _use_tmp_model(monkeypatch, tmp_path, None)
    with pytest.raises(tm.ModelNotFoundError):
        tm.load_model()


def test_training_pipeline_saves_metrics_and_predicts(monkeypatch, tmp_path):
    """Pipeline test with a scikit-learn stand-in regressor (XGBoost is covered by the next test)."""
    _use_tmp_model(monkeypatch, tmp_path, lambda: GradientBoostingRegressor(n_estimators=30, random_state=0))
    clean_df, features = _features()
    meta = tm.train_and_save(features)
    assert {"mae", "rmse", "r2"} <= set(meta["metrics"]["xgboost"])
    assert {"baseline_last_hour", "baseline_same_hour_yesterday", "baseline_same_hour_last_week"} <= set(meta["metrics"])
    assert set(meta["per_junction_metrics"]) == {"1", "2"}
    assert meta["train_range"][1] < meta["test_range"][0]          # chronological split
    model, loaded = tm.load_model()
    assert loaded["features"] == pp.FEATURE_COLUMNS
    result = tm.forecast(model, clean_df, junction=1, horizon=6)
    assert len(result) == 6 and (result["predicted"] >= 0).all()
    assert result["DateTime"].iloc[0] == clean_df["DateTime"].max() + pd.Timedelta(hours=1)
    evaluation = tm.evaluate_on_test_set(model, features, junction=2)
    assert list(evaluation.columns) == ["DateTime", "actual", "predicted"]


def test_real_xgboost_model_trains(monkeypatch, tmp_path):
    pytest.importorskip("xgboost")
    _use_tmp_model(monkeypatch, tmp_path, tm.make_regressor)
    _, features = _features()
    meta = tm.train_and_save(features)
    assert meta["model"] == "XGBRegressor"
