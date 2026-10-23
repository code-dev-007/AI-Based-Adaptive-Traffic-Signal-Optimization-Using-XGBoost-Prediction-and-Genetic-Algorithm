# Traffic-AI: Adaptive Traffic Signal Optimization (XGBoost + Genetic Algorithm)

MCA mini-project. A React dashboard and FastAPI backend that (1) forecasts hourly vehicle counts with XGBoost,
(2) optimises green-light timings for a four-way intersection with a custom Genetic Algorithm and
(3) compares Fixed-Time and Adaptive control in a simple simulator that streams live over a WebSocket.

## Software requirements

| Item | Specification |
|---|---|
| Operating system | Windows 10/11 |
| Languages | Python 3.11+, JavaScript (Node.js 18+ for the frontend) |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | React.js, Vite, Tailwind CSS |
| Visualisation | Recharts, Framer Motion, Lucide React |
| Machine learning | XGBoost, scikit-learn, pandas, NumPy, joblib |
| Optimisation | Genetic Algorithm (custom implementation in Python) |
| Real-time communication | WebSockets (simulation stream), Axios (REST client) |
| Database | SQLite with SQLAlchemy ORM (users and saved run summaries) |
| Testing | pytest |
| Version control | Git, GitHub |

## Dataset (synthetic)

Synthetic Traffic Volume Data for ML Projects, Kaggle (author: sanathshukla):
https://www.kaggle.com/datasets/sanathshukla/synthetic-traffic-volume-data-for-ml-projects

12,000 **synthetically generated** hourly records, columns `ID, DateTime, Junction, Vehicles`
(4 junctions x 3,000 hours, 2024-01-01 to 2024-05-04). This is **not real-world traffic data**, so no result here should be
read as a statement about real roads. The CSV is `data/raw/traffic_ai.csv`; the pipeline never generates replacement data.

Every number in the app is labelled **Observed** (from the CSV), **Predicted** (XGBoost) or **Simulated** (queues, waits,
vehicles served, signal timings). The data has junction-level counts only: the simulator's four approaches take their arrival
rate from Junctions 1-4, a modelling assumption and **not** real North/South/East/West measurements.

## Run on Windows (two terminals)

One-time setup in the project folder:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m backend.preprocessing
python -m backend.model
cd frontend
npm install
cd ..
```

`backend.model` trains XGBoost, prints the real test-set metrics and saves `models/traffic_xgb_model.joblib` and
`models/model_metadata.json`. `backend.preprocessing` writes the cleaned CSVs to `data/processed/`.

**Terminal 1 - FastAPI backend** (project folder)
```
.venv\Scripts\activate
uvicorn backend.main:app --reload
```
API docs: http://127.0.0.1:8000/docs

**Terminal 2 - React frontend**
```
cd frontend
npm run dev
```
Open http://localhost:5173. You will see the login page: choose **Create an account** the first time, then sign in. Vite forwards `/api` (REST) and `/ws` (WebSocket) to the backend on port 8000.
The status badge (top right) is green when the API is connected, amber when the model/dataset needs setup, and red when the API is offline.

## Login

- Accounts live in the SQLite database (`users` table). There is no default account: use **Create an account** on the login page.
- Passwords are stored as PBKDF2-SHA256 hashes with a random salt, never as plain text (`backend/auth.py`).
- After login the backend returns a signed token (HMAC-SHA256, valid 8 hours). The frontend keeps it in the browser's local storage and
  sends it with every request; the WebSocket sends it in the URL. Every endpoint except `/health` and `/auth/*` needs a valid token.
  When it expires or is rejected, the app returns to the login page.
- The signing key is created once in `data/.secret_key`. To use your own key, set the `TRAFFIC_AI_SECRET` environment variable.
- Scope: simple login for a mini-project. All users share the same saved runs, and there is no password reset, e-mail verification,
  rate limiting or HTTPS, so it is not meant for a public deployment.

## How it works

1. **Preprocessing** (`backend/preprocessing.py`): validates columns, reports missing values / duplicates / hourly gaps, adds hour,
   weekday, weekend and peak-hour features (peak hours 07-09 and 17-19, read off the observed hourly averages) and lag features
   (1, 2, 3, 24, 168 hours) plus rolling means. Rolling means use `shift(1)`, so the target hour is never in its own features.
2. **Prediction** (`backend/model.py`): XGBoost one-hour-ahead regression, chronological 80/20 split, scored with MAE, RMSE and R²
   against two baselines (same as last hour; same hour yesterday). Multi-hour forecasts feed each prediction back in.
3. **Genetic Algorithm** (`backend/genetic_algorithm.py`): chromosome = 4 green times. Tournament selection, uniform crossover,
   transfer mutation, elitism. Greens are always 10-60 s and add up to the cycle budget (120 s - 4 x 4 s lost time = 104 s).
   Fitness = demand-weighted Webster uniform delay + oversaturation penalty.
4. **Simulator** (`backend/simulator.py`): one-second steps, Poisson arrivals, discharge at 1800 veh/h while green. Fixed-Time =
   equal split. Adaptive = GA plan using model-predicted demand, re-planned every simulated hour. Both controllers receive
   identical arrivals for the same seed.
5. **API** (`backend/main.py`): health, history, prediction, optimisation, simulation start/stop/reset, comparison, saved runs, and
   the `/ws/simulation` WebSocket (state once per second). **Database** (`backend/db.py`): optimisation, comparison and
   simulation summaries are saved to `data/traffic_ai.db` and listed on the Analytics page.
6. **Frontend** (`frontend/src`): Login page plus Dashboard, Traffic Prediction, Signal Simulation and Analytics pages.

## Simulation assumptions to explain in the viva

- Dataset counts (about 10-70 vehicles/hour) are tiny relative to one lane's capacity, so a **demand multiplier** (default 10,
  adjustable 1-20) scales them. At 1x the two controllers behave almost the same; results depend on this choice.
- Adaptive plans from **predicted** demand while arrivals follow **observed** counts. If predictions are off, Adaptive can do
  worse than Fixed-Time, so the comparison reports mean ± std over several seeds.
- Start simulations inside the test period (the default) so predictions come from data the model was not trained on.
- Simplifications: no turning movements, no start-up lost time, one lane per approach, a single fixed cycle length.

## Tests

```
pytest
```
Covers preprocessing, model training/prediction, GA timing constraints, the simulator, password hashing and tokens, the database and
API endpoints (login, protected routes, saved runs and the WebSocket stream). Tests that need `xgboost`, `fastapi`, `httpx` or `sqlalchemy` skip if it is missing.
