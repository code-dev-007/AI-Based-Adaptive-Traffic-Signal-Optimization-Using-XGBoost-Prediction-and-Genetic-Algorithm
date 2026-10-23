"""Load, validate and clean the traffic CSV, then build model features.

All values come from the supplied (synthetic) Kaggle CSV. Nothing is generated
or imputed here: problems are reported, not silently repaired.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "traffic_ai.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "traffic_clean.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "traffic_features.csv"

REQUIRED_COLUMNS = ["ID", "DateTime", "Junction", "Vehicles"]
# Peak hours chosen from the observed hourly averages (morning 07-09, evening 17-19).
PEAK_HOURS = (7, 8, 9, 17, 18, 19)
LAGS = (1, 2, 3, 24, 168)  # hours back; 168 = same hour one week earlier
ROLLING_WINDOWS = (3, 24)
FEATURE_COLUMNS = (
    ["Junction", "hour", "weekday", "is_weekend", "is_peak_hour"]
    + [f"lag_{n}" for n in LAGS]
    + [f"roll_mean_{n}" for n in ROLLING_WINDOWS]
)
TARGET = "Vehicles"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV. Raises FileNotFoundError / ValueError with a clear message."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Put the CSV in data/raw/.")
    df = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate and clean. Returns (clean_df, report) where report lists what was found."""
    df = df[REQUIRED_COLUMNS].copy()
    report = {"rows_in": len(df), "missing_values": int(df.isna().sum().sum())}
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    report["unparseable_datetimes"] = int(df["DateTime"].isna().sum())
    df = df.dropna()  # only rows that are unusable (missing / unparseable values)
    report["duplicates_dropped"] = int(df.duplicated(["DateTime", "Junction"]).sum())
    df = df.drop_duplicates(["DateTime", "Junction"])
    df["Junction"] = df["Junction"].astype(int)
    df["Vehicles"] = df["Vehicles"].astype(int)
    if (df["Vehicles"] < 0).any():
        raise ValueError("Negative vehicle counts found in the dataset.")
    df = df.sort_values(["Junction", "DateTime"]).reset_index(drop=True)
    gaps = df.groupby("Junction")["DateTime"].diff().dropna()
    report["hourly_gaps"] = int((gaps != pd.Timedelta(hours=1)).sum())
    report["rows_out"] = len(df)
    return df, report


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["DateTime"].dt.hour
    df["weekday"] = df["DateTime"].dt.dayofweek  # Monday = 0
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin(PEAK_HOURS).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag / rolling features per junction. Every feature uses only past hours:
    rolling means are computed on shift(1), so the current count is never included."""
    df = df.sort_values(["Junction", "DateTime"]).copy()
    by_junction = df.groupby("Junction")["Vehicles"]
    for n in LAGS:
        df[f"lag_{n}"] = by_junction.shift(n)
    df["_past"] = by_junction.shift(1)
    for n in ROLLING_WINDOWS:
        df[f"roll_mean_{n}"] = df.groupby("Junction")["_past"].transform(lambda s: s.rolling(n).mean())
    return df.drop(columns="_past")


def build_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Time + lag features; drops the first rows of each junction that lack history."""
    df = add_lag_features(add_time_features(clean_df))
    return df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def split_chronological(df: pd.DataFrame, test_fraction: float = 0.2):
    """Train on the earlier part of the timeline, test on the later part (same cut-off for all junctions)."""
    times = sorted(df["DateTime"].unique())
    cutoff = times[int(len(times) * (1 - test_fraction))]
    return df[df["DateTime"] < cutoff].copy(), df[df["DateTime"] >= cutoff].copy()


def run_preprocessing(raw_path: Path = RAW_PATH) -> dict:
    """Full stage-1 pipeline: load -> clean -> features -> save both CSVs."""
    clean_df, report = clean(load_raw(raw_path))
    features = build_features(clean_df)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_PATH, index=False)
    features.to_csv(FEATURES_PATH, index=False)
    report["feature_rows"] = len(features)
    return report


if __name__ == "__main__":
    print(run_preprocessing())
