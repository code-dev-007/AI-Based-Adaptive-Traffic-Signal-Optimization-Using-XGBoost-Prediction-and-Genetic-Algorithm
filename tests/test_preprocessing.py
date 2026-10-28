import pandas as pd
import pytest

from backend import preprocessing as pp


def _tiny_raw(hours=400):
    times = pd.date_range("2024-01-01", periods=hours, freq="h")
    rows = [(i, t, j, 10 + (t.hour % 5) + j) for j in (1, 2) for i, t in enumerate(times)]
    return pd.DataFrame(rows, columns=pp.REQUIRED_COLUMNS)


def test_missing_column_is_reported(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"ID": [1], "Vehicles": [3]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        pp.load_raw(path)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        pp.load_raw(tmp_path / "nope.csv")


def test_clean_reports_duplicates_and_missing():
    raw = _tiny_raw(50)
    raw = pd.concat([raw, raw.iloc[:3]])           # 3 duplicate rows
    raw.loc[raw.index[5], "Vehicles"] = None       # 1 missing value
    _, report = pp.clean(raw)
    assert report["duplicates_dropped"] == 3
    assert report["missing_values"] == 1


def test_time_features():
    df = pp.add_time_features(pd.DataFrame({"DateTime": pd.to_datetime(["2024-01-06 08:00", "2024-01-08 03:00"])}))
    assert df["is_weekend"].tolist() == [1, 0]      # Saturday, Monday
    assert df["month"].tolist() == [1, 1] and df["day"].tolist() == [6, 8] and df["hour"].tolist() == [8, 3]


def test_lag_features_use_only_the_past():
    clean_df, _ = pp.clean(_tiny_raw())
    feats = pp.build_features(clean_df)
    row = feats[feats.Junction == 1].iloc[10]
    series = clean_df[clean_df.Junction == 1].set_index("DateTime")["Vehicles"]
    t = row["DateTime"]
    assert row["lag_1"] == series[t - pd.Timedelta(hours=1)]
    assert row["lag_24"] == series[t - pd.Timedelta(hours=24)]
    past3 = series[t - pd.Timedelta(hours=3): t - pd.Timedelta(hours=1)]
    assert row["roll_mean_3"] == pytest.approx(past3.mean())  # current hour excluded


def test_chronological_split_has_no_overlap():
    clean_df, _ = pp.clean(_tiny_raw())
    train, test = pp.split_chronological(pp.build_features(clean_df), 0.2)
    assert train["DateTime"].max() < test["DateTime"].min()
    assert len(test) > 0


def test_real_csv_loads_and_is_not_modified():
    raw = pp.load_raw()
    clean_df, report = pp.clean(raw)
    assert len(clean_df) == len(raw) == report["rows_out"]            # nothing valid is dropped
    assert report["missing_values"] == 0 and report["duplicates_dropped"] == 0
    assert sorted(clean_df["Junction"].unique()) == [1, 2, 3, 4]
    assert clean_df["DateTime"].is_monotonic_increasing                # sorted chronologically
    merged = raw.merge(clean_df, on="ID", suffixes=("_raw", "_clean"))
    assert (merged["Vehicles_raw"] == merged["Vehicles_clean"]).all()  # original values preserved


def test_lag_features_are_per_junction_on_real_data():
    clean_df, _ = pp.clean(pp.load_raw())
    feats = pp.build_features(clean_df)
    row = feats[feats.Junction == 4].iloc[0]
    series = clean_df[clean_df.Junction == 4].set_index("DateTime")["Vehicles"]
    assert row["DateTime"] >= pd.Timestamp("2017-01-08")                # Junction 4 has no history before 2017
    assert row["lag_1"] == series[row["DateTime"] - pd.Timedelta(hours=1)]
    assert row["lag_168"] == series[row["DateTime"] - pd.Timedelta(hours=168)]
