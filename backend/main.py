"""FastAPI service for Traffic-AI. Start with:  uvicorn backend.main:app --reload

Every response says what kind of numbers it holds:
  observed  = rows from the CSV (hourly vehicle counts per junction)
  predicted = XGBoost model output
  simulated = simulator estimates (queues, waits, vehicles served, signal timing)
"""
from __future__ import annotations

import asyncio
import logging
import re

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import auth, db
from backend import genetic_algorithm as ga
from backend import model as traffic_model
from backend import model_comparison
from backend import preprocessing as pp
from backend.simulator import Simulation, compare_controllers

app = FastAPI(
    title="Traffic-AI API",
    description="XGBoost traffic-volume prediction + Genetic Algorithm signal optimisation on hourly junction vehicle counts.",
    version="0.2.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

log = logging.getLogger("traffic_ai")
db.init_db()
protected = APIRouter(dependencies=[Depends(auth.require_user)])  # every route on it needs a valid login token
_cache: dict = {}   # loaded dataset / features / model, kept in memory
_sim: dict = {"sim": None, "meta": {}}  # the single running simulation and the scenario it was started with


def _record(kind: str, scenario_start, result: dict):
    """Save a run summary to SQLite. A database problem must never break the API call."""
    try:
        db.save_run(kind, scenario_start, result)
    except Exception as error:  # noqa: BLE001
        log.warning("Could not save %s run: %s", kind, error)


def _save_sim_once(sim: Simulation):
    if sim.saved or sim.t == 0:
        return
    sim.saved = True
    m = sim.metrics()
    _record("simulation", _sim["meta"].get("start"), {
        **_sim["meta"], "controller": sim.controller, "finished": sim.t >= sim.duration, "simulated_seconds": sim.t,
        "avg_wait_s": m["avg_wait_s"], "total_arrived": m["total_arrived"], "total_served": m["total_served"],
        "max_queue": m["max_queue"], "greens": list(sim.greens)})


def _advance(sim: Simulation, seconds: int):
    if sim.running:
        sim.advance(seconds)
        sim.running = sim.t < sim.duration
        if not sim.running:
            _save_sim_once(sim)


# ---------------------------------------------------------------- request bodies
class Credentials(BaseModel):
    username: str
    password: str


class PredictRequest(BaseModel):
    junction: int
    horizon: int = Field(default=6, ge=1, le=48)
    as_of: str | None = None                     # optional: forecast the hours after this time instead of after the dataset end


class ScenarioRequest(BaseModel):
    start: str                                   # ISO datetime, e.g. "2024-04-15T07:00:00"
    hours: int = Field(default=1, ge=1, le=6)
    demand_scale: float = Field(default=3.0, gt=0, le=50)
    seed: int = 42
    controller: str = "adaptive"


class OptimizeRequest(BaseModel):
    start: str | None = None                     # use model predictions for this hour ...
    demand_per_hour: list[float] | None = None   # ... or supply the 4 demand values directly
    demand_scale: float = Field(default=3.0, gt=0, le=50)
    seed: int = 42
    # Signal-timing settings (seconds). Defaults come from backend/genetic_algorithm.py.
    cycle_length: int = Field(default=ga.CYCLE_LENGTH, ge=20, le=300)
    min_green: int = Field(default=ga.MIN_GREEN, ge=1, le=120)
    max_green: int = Field(default=ga.MAX_GREEN, ge=1, le=240)
    lost_time: int = Field(default=ga.LOST_TIME, ge=0, le=15)
    population_size: int = Field(default=40, ge=10, le=200)
    generations: int = Field(default=60, ge=5, le=300)


class StepRequest(BaseModel):
    seconds: int = Field(default=10, ge=1, le=600)


class CompareRequest(BaseModel):
    start: str
    hours: int = Field(default=1, ge=1, le=6)
    demand_scale: float = Field(default=3.0, gt=0, le=50)
    n_seeds: int = Field(default=5, ge=1, le=20)
    base_seed: int = 42


# ---------------------------------------------------------------- loading helpers
def get_clean() -> pd.DataFrame:
    if "clean" not in _cache:
        try:
            clean_df, report = pp.clean(pp.load_raw())
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error))
        _cache["clean"], _cache["report"] = clean_df, report
    return _cache["clean"]


def get_features() -> pd.DataFrame:
    if "features" not in _cache:
        _cache["features"] = pp.build_features(get_clean())
    return _cache["features"]


def get_model():
    if "model" not in _cache:
        try:
            _cache["model"], _cache["meta"] = traffic_model.load_model()
        except traffic_model.ModelNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error))
    return _cache["model"], _cache["meta"]


def _parse_time(text: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(text).floor("h")
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {text!r}")


def scenario_range(features: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Hours for which every junction has model features (Junction 4 only starts in 2017, and each junction
    needs a week of history for the lag features)."""
    per_junction = features.groupby("Junction")["DateTime"].agg(["min", "max"])
    return per_junction["min"].max(), per_junction["max"].min()


def build_scenario(start: str, hours: int, need_forecast: bool) -> dict:
    """Observed hourly counts per junction (drive the simulated arrivals) and, optionally,
    one-hour-ahead model predictions for the same hours (what the Adaptive controller plans with)."""
    features = get_features()
    times = pd.date_range(_parse_time(start), periods=hours, freq="h")
    window = features[features["DateTime"].isin(times)]
    if window["DateTime"].nunique() != len(times):
        first, last = scenario_range(features)
        raise HTTPException(status_code=400, detail=(
            f"Start time must leave {hours} hour(s) inside {first} to {last}."))
    observed = window.pivot(index="DateTime", columns="Junction", values=pp.TARGET).sort_index()
    all_junctions = sorted(int(j) for j in features["Junction"].unique())
    if list(observed.columns) != all_junctions or observed.isna().any().any():
        first, last = scenario_range(features)
        raise HTTPException(status_code=400, detail=(
            f"Not every junction has data for the whole window. A scenario needs all {len(all_junctions)} junctions with usable "
            f"history; choose a start between {first} and {last}."))
    scenario = {"observed": observed.values.tolist(), "forecast": None, "times": [str(t) for t in observed.index]}
    if need_forecast:
        model, _ = get_model()
        window = window.assign(predicted=traffic_model.predict_rows(model, window))
        scenario["forecast"] = window.pivot(index="DateTime", columns="Junction", values="predicted").sort_index().values.tolist()
    test_start = pp.split_chronological(features, 0.2)[1]["DateTime"].min()
    scenario["in_test_period"] = bool(times[0] >= test_start)
    return scenario


# ---------------------------------------------------------------- login
def _clean_username(raw: str) -> str:
    return raw.strip().lower()


@app.post("/api/v1/auth/register")
def register(credentials: Credentials):
    username = _clean_username(credentials.username)
    if not re.fullmatch(r"[a-z0-9_.-]{3,30}", username):
        raise HTTPException(status_code=400, detail="Username must be 3-30 characters: letters, numbers, dot, dash or underscore.")
    if len(credentials.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    if not db.create_user(username, auth.hash_password(credentials.password)):
        raise HTTPException(status_code=409, detail="That username is already taken.")
    return {"token": auth.create_token(username), "username": username}


@app.post("/api/v1/auth/login")
def login(credentials: Credentials):
    username = _clean_username(credentials.username)
    stored = db.get_password_hash(username)
    if stored is None or not auth.verify_password(credentials.password, stored):
        raise HTTPException(status_code=401, detail="Wrong username or password.")
    return {"token": auth.create_token(username), "username": username}


@app.get("/api/v1/auth/me")
def me(user: str = Depends(auth.require_user)):
    return {"username": user}


# ---------------------------------------------------------------- health / model / data
@app.get("/api/v1/health")
def health():
    status = {"status": "ok", "dataset_available": True, "model_available": True, "detail": []}
    try:
        status["rows"] = len(get_clean())
    except HTTPException as error:
        status.update(dataset_available=False, model_available=False)
        status["detail"].append(error.detail)
        return status
    try:
        get_model()
    except HTTPException as error:
        status["model_available"] = False
        status["detail"].append(error.detail)
    return status


@protected.get("/api/v1/model-info")
def model_info():
    _, meta = get_model()
    return meta


@protected.get("/api/v1/model-comparison")
def model_comparison_results(recompute: bool = False):
    """Linear Regression vs Random Forest vs XGBoost on the same chronological test split.
    Computed once and cached; pass ?recompute=true to retrain and refresh the cache."""
    try:
        result = model_comparison.get_comparison(recompute=recompute)
    except model_comparison.ComparisonError as error:
        raise HTTPException(status_code=503, detail=str(error))
    return {"kind": "evaluated", **result}


@protected.get("/api/v1/traffic/summary")
def traffic_summary():
    """Observed dataset statistics for the dashboard (everything is computed from the CSV)."""
    df = pp.add_time_features(get_clean())
    features = get_features()
    daily = df.assign(date=df["DateTime"].dt.strftime("%Y-%m-%d")).groupby(["date", "Junction"], as_index=False)["Vehicles"].mean()
    monthly = df.assign(month=df["DateTime"].dt.strftime("%Y-%m")).groupby(["month", "Junction"], as_index=False)["Vehicles"].mean()
    stats = df.groupby("Junction")["Vehicles"].agg(["count", "mean", "median", "std", "min", "max", "sum"]).round(2)
    stats["share_of_total_pct"] = (100 * stats["sum"] / stats["sum"].sum()).round(1)
    stats["first_record"] = df.groupby("Junction")["DateTime"].min().astype(str)
    stats["last_record"] = df.groupby("Junction")["DateTime"].max().astype(str)
    hourly_all = df.groupby("hour")["Vehicles"].mean()
    busiest = [int(h) for h in hourly_all.sort_values(ascending=False).index[:3]]
    test_start = pp.split_chronological(features, 0.2)[1]["DateTime"].min()
    first_ok, last_ok = scenario_range(features)
    # Suggested demo hour: the first occurrence, in the test period, of the busiest hour of the day.
    candidates = pd.date_range(max(test_start.ceil("h"), first_ok), last_ok, freq="h")
    default_start = next((t for t in candidates if t.hour == busiest[0]), candidates[0])
    return {
        "kind": "observed",
        "report": _cache["report"],
        "records": len(df),
        "junctions": sorted(int(j) for j in df["Junction"].unique()),
        "start": str(df["DateTime"].min()),
        "end": str(df["DateTime"].max()),
        "feature_start": str(features["DateTime"].min()),
        "test_start": str(test_start),
        "scenario_range": [str(first_ok), str(last_ok)],
        "default_scenario_start": str(default_start),
        "mean_vehicles": float(df["Vehicles"].mean()),
        "max_vehicles": int(df["Vehicles"].max()),
        "min_vehicles": int(df["Vehicles"].min()),
        "busiest_hours": busiest,
        "junction_stats": stats.reset_index().rename(columns={"sum": "total"}).to_dict("records"),
        "daily": daily.round(2).to_dict("records"),
        "monthly": monthly.round(2).to_dict("records"),
        "hourly": df.groupby(["hour", "Junction"], as_index=False)["Vehicles"].mean().round(2).to_dict("records"),
        "weekday": df.groupby(["weekday", "Junction"], as_index=False)["Vehicles"].mean().round(2).to_dict("records"),
    }


@protected.get("/api/v1/traffic/history")
def traffic_history(junction: int = Query(...), start: str | None = None, end: str | None = None,
                    limit: int = Query(500, ge=1, le=5000)):
    df = get_clean()
    df = df[df["Junction"] == junction]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Junction {junction} not found.")
    if start:
        df = df[df["DateTime"] >= _parse_time(start)]
    if end:
        df = df[df["DateTime"] <= _parse_time(end)]
    df = df.tail(limit)
    return {"kind": "observed", "junction": junction,
            "items": [{"DateTime": str(r.DateTime), "Vehicles": int(r.Vehicles)} for r in df.itertuples()]}


# ---------------------------------------------------------------- prediction
@protected.post("/api/v1/predict")
def predict(request: PredictRequest):
    clean_df = get_clean()
    if request.junction not in set(clean_df["Junction"]):
        raise HTTPException(status_code=404, detail=f"Junction {request.junction} not found.")
    model, _ = get_model()
    as_of = _parse_time(request.as_of) if request.as_of else None
    junction_data = clean_df[clean_df["Junction"] == request.junction]
    try:
        result = traffic_model.forecast(model, clean_df, request.junction, request.horizon, as_of)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    history = junction_data if as_of is None else junction_data[junction_data["DateTime"] <= as_of]
    recent = history.tail(72)
    # When forecasting from a past hour, the later observed counts are available to compare against.
    observed = junction_data.set_index("DateTime")["Vehicles"]
    return {
        "kind": "predicted",
        "junction": request.junction,
        "note": ("Predictions continue after the last observed hour; each prediction feeds the next (traffic volume per hour, "
                 "not individual vehicles)." if as_of is None else
                 f"Forecast made only from data up to {as_of}; observed counts for the same hours are shown for comparison."),
        "predictions": [{"DateTime": str(r.DateTime), "predicted": round(r.predicted, 2),
                         "observed": int(observed[r.DateTime]) if r.DateTime in observed.index else None} for r in result.itertuples()],
        "recent_observed": [{"DateTime": str(r.DateTime), "Vehicles": int(r.Vehicles)} for r in recent.itertuples()],
    }


@protected.get("/api/v1/predict/test")
def predict_on_test_set(junction: int = Query(...), limit: int = Query(168, ge=1, le=1000)):
    model, _ = get_model()
    result = traffic_model.evaluate_on_test_set(model, get_features(), junction).tail(limit)
    if result.empty:
        raise HTTPException(status_code=404, detail=f"No test rows for junction {junction}.")
    return {"kind": "observed vs predicted", "junction": junction,
            "items": [{"DateTime": str(r.DateTime), "actual": int(r.actual), "predicted": round(float(r.predicted), 2)}
                      for r in result.itertuples()]}


# ---------------------------------------------------------------- optimisation (GA)
@protected.post("/api/v1/optimize")
def optimize(request: OptimizeRequest):
    observed_counts = predicted_counts = None
    if request.demand_per_hour is not None:
        if len(request.demand_per_hour) != ga.N_APPROACHES or min(request.demand_per_hour) < 0:
            raise HTTPException(status_code=400, detail=f"Provide {ga.N_APPROACHES} non-negative demand values.")
        demand, source = [v * request.demand_scale for v in request.demand_per_hour], "manual input (simulated demand)"
    elif request.start:
        scenario = build_scenario(request.start, 1, need_forecast=True)
        observed_counts, predicted_counts = scenario["observed"][0], scenario["forecast"][0]
        demand = [v * request.demand_scale for v in predicted_counts]
        source = "XGBoost predictions for the selected hour (junction counts), scaled by the demand multiplier (simulation assumption)"
    else:
        raise HTTPException(status_code=400, detail="Provide either 'start' or 'demand_per_hour'.")
    try:
        result = ga.optimize(demand, population_size=request.population_size, generations=request.generations, seed=request.seed,
                             cycle_length=request.cycle_length, min_green=request.min_green, max_green=request.max_green,
                             lost_time=request.lost_time)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    result.update(demand_source=source, demand_scale=request.demand_scale, kind="simulated estimate",
                  lost_time_per_phase=request.lost_time, green_budget=result["constraints"]["green_budget"],
                  observed_vehicles=observed_counts, predicted_vehicles=predicted_counts,
                  note="Delays are Webster-formula ESTIMATES, not measured waiting times; approaches map to junctions 1-4 (assumption).")
    _record("optimization", request.start, {key: result[key] for key in (
        "greens", "fixed_greens", "fitness", "fixed_fitness", "demand_source", "demand_scale", "demand_per_hour")})
    return result


# ---------------------------------------------------------------- simulation control
def _current_sim() -> Simulation:
    if _sim["sim"] is None:
        raise HTTPException(status_code=409, detail="No simulation yet. Start one first.")
    return _sim["sim"]


@protected.post("/api/v1/simulation/start")
def simulation_start(request: ScenarioRequest):
    if request.controller not in ("fixed", "adaptive"):
        raise HTTPException(status_code=400, detail="controller must be 'fixed' or 'adaptive'.")
    scenario = build_scenario(request.start, request.hours, need_forecast=request.controller == "adaptive")
    sim = Simulation(scenario["observed"], scenario["forecast"], request.controller, request.seed, request.demand_scale)
    sim.running = True
    _sim["sim"] = sim
    _sim["meta"] = {"start": request.start, "hours": request.hours, "demand_scale": request.demand_scale, "seed": request.seed}
    return {**sim.snapshot(), "scenario_times": scenario["times"], "in_test_period": scenario["in_test_period"],
            "kind": "simulated"}


@protected.post("/api/v1/simulation/step")
def simulation_step(request: StepRequest):
    sim = _current_sim()
    _advance(sim, request.seconds)
    return {**sim.snapshot(), "kind": "simulated"}


@protected.post("/api/v1/simulation/stop")
def simulation_stop():
    sim = _current_sim()
    sim.running = False
    _save_sim_once(sim)
    return {**sim.snapshot(), "kind": "simulated"}


@protected.post("/api/v1/simulation/reset")
def simulation_reset():
    sim = _current_sim()
    sim.reset()
    return {**sim.snapshot(), "kind": "simulated"}


@protected.get("/api/v1/simulation/state")
def simulation_state():
    return {**_current_sim().snapshot(), "kind": "simulated"}


# ---------------------------------------------------------------- Fixed-Time vs Adaptive
@protected.post("/api/v1/compare")
def compare(request: CompareRequest):
    scenario = build_scenario(request.start, request.hours, need_forecast=True)
    runs = {"fixed": [], "adaptive": []}
    first = None
    for k in range(request.n_seeds):
        result = compare_controllers(scenario["observed"], scenario["forecast"], request.base_seed + k, request.demand_scale)
        first = first or result
        for name, snap in result.items():
            m = snap["metrics"]
            runs[name].append({"seed": request.base_seed + k, "avg_wait_s": m["avg_wait_s"], "total_served": m["total_served"],
                               "total_arrived": m["total_arrived"], "max_queue": m["max_queue"], "end_queue": sum(m["queues"])})
    summary = {}
    for name, rows in runs.items():
        summary[name] = {key: {"mean": float(np.mean([r[key] for r in rows])), "std": float(np.std([r[key] for r in rows]))}
                         for key in ("avg_wait_s", "total_served", "max_queue", "end_queue")}
    fixed_wait, adaptive_wait = summary["fixed"]["avg_wait_s"]["mean"], summary["adaptive"]["avg_wait_s"]["mean"]
    change = 100 * (adaptive_wait - fixed_wait) / fixed_wait if fixed_wait else None
    _record("comparison", request.start, {"hours": request.hours, "n_seeds": request.n_seeds, "demand_scale": request.demand_scale,
                                          "summary": summary, "wait_change_pct": change})
    return {
        "kind": "simulated",
        "scenario_times": scenario["times"],
        "in_test_period": scenario["in_test_period"],
        "demand_scale": request.demand_scale,
        "n_seeds": request.n_seeds,
        "summary": summary,
        "wait_change_pct": change,
        "runs": runs,
        "example_run": {name: {"greens": snap["greens"], "queues": snap["metrics"]["queues"],
                               "avg_wait_by_approach_s": snap["metrics"]["avg_wait_by_approach_s"],
                               "history": snap["history"]} for name, snap in first.items()},
    }


# ---------------------------------------------------------------- saved runs (SQLite)
@protected.get("/api/v1/runs")
def saved_runs(kind: str | None = None, limit: int = Query(20, ge=1, le=200)):
    return {"items": db.list_runs(kind, limit)}


@protected.delete("/api/v1/runs")
def delete_saved_runs():
    return {"deleted": db.clear_runs()}


# ---------------------------------------------------------------- live simulation stream (WebSocket)
@app.websocket("/ws/simulation")
async def simulation_stream(websocket: WebSocket, token: str | None = Query(default=None)):
    """Sends the simulation state once per second and advances a running simulation by `speed`
    simulated seconds per tick. The client changes the speed with {"speed": 60}; start/stop/reset use REST."""
    await websocket.accept()
    if auth.verify_token(token or "") is None:
        await websocket.close(code=4401)  # 4401 = not signed in; the frontend then shows the login page
        return
    speed = {"value": 60}

    async def listen():
        while True:
            message = await websocket.receive_json()
            speed["value"] = max(1, min(600, int(message.get("speed", speed["value"]))))

    listener = asyncio.create_task(listen())
    try:
        while not listener.done():
            sim = _sim["sim"]
            if sim is None:
                await websocket.send_json({"idle": True})
            else:
                await asyncio.to_thread(_advance, sim, speed["value"])
                await websocket.send_json({**sim.snapshot(), "kind": "simulated"})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    finally:
        listener.cancel()


app.include_router(protected)  # must come after every @protected route above
