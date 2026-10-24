import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("sqlalchemy")
from fastapi.testclient import TestClient

from sqlalchemy import create_engine

from backend import db, main
from backend import preprocessing as pp

client = TestClient(main.app)       # signed in (see the temp_database fixture)
anonymous = TestClient(main.app)    # never signed in
START = "2024-04-15T07:00:00"


@pytest.fixture(autouse=True)
def temp_database(monkeypatch, tmp_path):
    """Every test writes to its own throw-away SQLite file, never to data/traffic_ai.db."""
    monkeypatch.setattr(db, "engine", create_engine(f"sqlite:///{tmp_path / 'api_test.db'}"))
    db.Base.metadata.create_all(db.engine)
    token = client.post("/api/v1/auth/register", json={"username": "tester", "password": "secret123"}).json()["token"]
    client.headers["Authorization"] = f"Bearer {token}"


@pytest.fixture()
def with_model(monkeypatch):
    """Put a small stand-in regressor in the API cache so model endpoints work without a trained file."""
    features = main.get_features()
    train, _ = pp.split_chronological(features)
    model = GradientBoostingRegressor(n_estimators=20, random_state=0).fit(train[pp.FEATURE_COLUMNS], train[pp.TARGET])
    monkeypatch.setitem(main._cache, "model", model)
    monkeypatch.setitem(main._cache, "meta", {"model": "stand-in", "metrics": {}})


def test_health_reports_status():
    body = client.get("/api/v1/health").json()
    assert body["dataset_available"] is True
    assert "model_available" in body


def test_history_and_summary():
    history = client.get("/api/v1/traffic/history", params={"junction": 1, "limit": 10}).json()
    assert len(history["items"]) == 10 and history["kind"] == "observed"
    assert client.get("/api/v1/traffic/history", params={"junction": 99}).status_code == 404
    assert client.get("/api/v1/traffic/summary").json()["junctions"] == [1, 2, 3, 4]


def test_predict_without_model_returns_503(monkeypatch):
    monkeypatch.delitem(main._cache, "model", raising=False)
    monkeypatch.setattr(main.traffic_model, "MODEL_PATH", main.pp.PROJECT_ROOT / "models" / "does_not_exist.joblib")
    assert client.post("/api/v1/predict", json={"junction": 1, "horizon": 3}).status_code == 503


def test_predict_with_model(with_model):
    body = client.post("/api/v1/predict", json={"junction": 2, "horizon": 4}).json()
    assert body["kind"] == "predicted" and len(body["predictions"]) == 4
    assert client.post("/api/v1/predict", json={"junction": 99, "horizon": 4}).status_code == 404


def test_optimize_respects_green_limits():
    greens = client.post("/api/v1/optimize", json={"demand_per_hour": [200, 300, 150, 400]}).json()["greens"]
    assert sum(greens) == 104 and all(10 <= g <= 60 for g in greens)
    assert client.post("/api/v1/optimize", json={}).status_code == 400


def test_simulation_lifecycle():
    assert client.post("/api/v1/simulation/start", json={"start": START, "controller": "fixed"}).status_code == 200
    assert client.post("/api/v1/simulation/step", json={"seconds": 30}).json()["metrics"]["time_s"] == 30
    assert client.post("/api/v1/simulation/stop").json()["running"] is False
    assert client.post("/api/v1/simulation/step", json={"seconds": 30}).json()["metrics"]["time_s"] == 30
    assert client.post("/api/v1/simulation/reset").json()["metrics"]["time_s"] == 0


def test_compare_returns_both_controllers(with_model):
    body = client.post("/api/v1/compare", json={"start": START, "hours": 1, "n_seeds": 2}).json()
    assert set(body["summary"]) == {"fixed", "adaptive"}
    assert client.post("/api/v1/compare", json={"start": "2030-01-01T00:00:00"}).status_code == 400


def test_runs_are_saved_listed_and_cleared():
    client.post("/api/v1/optimize", json={"demand_per_hour": [200, 300, 150, 400]})
    client.post("/api/v1/simulation/start", json={"start": START, "controller": "fixed"})
    client.post("/api/v1/simulation/step", json={"seconds": 20})
    client.post("/api/v1/simulation/stop")
    kinds = [run["kind"] for run in client.get("/api/v1/runs").json()["items"]]
    assert sorted(kinds) == ["optimization", "simulation"]
    assert client.delete("/api/v1/runs").json()["deleted"] == 2


def test_simulation_websocket_streams_state():
    client.post("/api/v1/simulation/start", json={"start": START, "controller": "fixed"})
    token = client.headers["Authorization"].removeprefix("Bearer ")
    with client.websocket_connect(f"/ws/simulation?token={token}") as socket:
        socket.send_json({"speed": 30})
        first, second = socket.receive_json(), socket.receive_json()
    assert second["metrics"]["time_s"] > first["metrics"]["time_s"] > 0
    assert len(second["signals"]) == 4


def test_register_login_and_me():
    assert client.get("/api/v1/auth/me").json() == {"username": "tester"}
    good = anonymous.post("/api/v1/auth/login", json={"username": "Tester", "password": "secret123"})   # username is case-insensitive
    assert good.status_code == 200 and good.json()["username"] == "tester"
    assert anonymous.post("/api/v1/auth/login", json={"username": "tester", "password": "nope-nope"}).status_code == 401
    assert anonymous.post("/api/v1/auth/login", json={"username": "ghost", "password": "secret123"}).status_code == 401


def test_register_validation():
    assert anonymous.post("/api/v1/auth/register", json={"username": "tester", "password": "secret123"}).status_code == 409
    assert anonymous.post("/api/v1/auth/register", json={"username": "ab", "password": "secret123"}).status_code == 400
    assert anonymous.post("/api/v1/auth/register", json={"username": "newuser", "password": "123"}).status_code == 400


def test_data_endpoints_need_login_but_health_does_not():
    assert anonymous.get("/api/v1/health").status_code == 200
    for url in ("/api/v1/traffic/summary", "/api/v1/model-info", "/api/v1/runs", "/api/v1/auth/me"):
        assert anonymous.get(url).status_code == 401
    assert anonymous.post("/api/v1/optimize", json={"demand_per_hour": [1, 2, 3, 4]}).status_code == 401
    assert anonymous.get("/api/v1/traffic/summary", headers={"Authorization": "Bearer forged.token"}).status_code == 401


def test_websocket_rejects_missing_token():
    with anonymous.websocket_connect("/ws/simulation") as socket:
        with pytest.raises(Exception):      # server closes with code 4401
            socket.receive_json()


def test_model_comparison_returns_real_metrics():
    body = client.get("/api/v1/model-comparison").json()
    names = [m["name"] for m in body["models"]]
    assert names == ["Linear Regression", "Random Forest", "XGBoost"]
    assert body["selection_metric"] == "MAE"
    assert body["best_model"] == min(body["models"], key=lambda m: m["mae"])["name"]
    assert all(m["mae"] > 0 and m["rmse"] >= m["mae"] for m in body["models"])
    assert body["evaluation"]["target"] == "Vehicles" and "ID" not in body["evaluation"]["features"]


def test_model_comparison_requires_login():
    assert anonymous.get("/api/v1/model-comparison").status_code == 401


def test_model_comparison_missing_dataset_returns_503(monkeypatch, tmp_path):
    from backend import model_comparison as mc
    monkeypatch.setattr(mc.pp, "RAW_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(mc, "_memory", {})
    assert client.get("/api/v1/model-comparison").status_code == 503
