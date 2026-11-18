# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "numpy>=1.26", "statsmodels>=0.14", "matplotlib>=3.8"]
# ///
"""Tandem group model v1: a linear regression for "will there be a dock?"

Question: a rider heading to a station at weekday evening rush (4-7 PM) wants
to know how many free docks it will have when they get there in 30 minutes.

Run from the repo root (one command, installs what it needs):

    uv run model/dock_regression.py

It prints the data summary, the fitted model, baseline vs v1 on the test rows,
a decision for one new input, and the value estimate. It also saves a chart
and the numbers to model/output/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO = Path(__file__).resolve().parents[1]
STATUS = REPO / "citibike" / "station_status"
STATIONS = REPO / "citibike" / "stations"
OUT = Path(__file__).resolve().parent / "output"

NEW_YORK = "America/New_York"
RUSH_START, RUSH_END = 16, 19  # 4 PM to 7 PM, New York time
AHEAD = 30  # minutes until the rider arrives
SEED = 42  # fixed so the random 70/30 split is the same every run

FEATURES = ["docks_now", "docks_change_15", "capacity", "minutes_after_4pm"]
OUTCOME = "docks_in_30"

# --- Decision rule and value assumptions (made up; see the README) ---
LOW_DOCKS = 3  # predict fewer than this -> send the rider to another station
MINUTES_LOST_FULL = 8  # rider arrives to a full station and has to find another
MINUTES_LOST_DETOUR = 3  # rider is sent elsewhere but the station had room
DOLLARS_PER_MINUTE = 0.50  # value of a rider's time
ARRIVALS_PER_WEEKDAY = 10_000  # evening-rush dock returns we would advise per day


# ---------------------------------------------------------------- 1. Load data
def rush_files() -> list[Path]:
    """Hourly files whose run started during weekday 4-7 PM New York time."""
    files = []
    for f in sorted(STATUS.glob("*/*.csv")):
        start = pd.Timestamp(f"{f.parent.name} {f.stem[:2]}:{f.stem[2:4]}", tz="UTC")
        local = start.tz_convert(NEW_YORK)
        # include the hour before 4 PM too, so 4 PM rows have a 15-minute history
        if local.day_of_week < 5 and RUSH_START - 1 <= local.hour < RUSH_END:
            files.append(f)
    return files


def file_to_slots(path: Path) -> pd.DataFrame:
    """Turn one hourly file into one row per station per 5-minute snapshot.

    Each file starts with every station, then only rows that changed, so we
    carry each station's last value forward - but only inside this file.
    """
    df = pd.read_csv(
        path,
        usecols=["fetched_at", "station_id", "docks_available", "is_installed", "is_returning"],
    )
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df = df.drop_duplicates(["station_id", "fetched_at"], keep="last")
    wide = df.pivot(index="fetched_at", columns="station_id")
    wide = wide.sort_index().ffill()
    return wide.stack(future_stack=True).dropna().reset_index()


def load_panel() -> pd.DataFrame:
    files = rush_files()
    print(f"Reading {len(files)} hourly files ...")
    panel = pd.concat([file_to_slots(f) for f in files], ignore_index=True)
    panel = panel.drop_duplicates(["station_id", "fetched_at"], keep="last")
    # keep stations that are open and accepting bikes
    panel = panel[(panel["is_installed"] == 1) & (panel["is_returning"] == 1)]
    return panel[["station_id", "fetched_at", "docks_available"]]


def station_capacity() -> pd.Series:
    """Most recent capacity listed for each station."""
    frames = [pd.read_csv(f, usecols=["station_id", "capacity"]) for f in sorted(STATIONS.glob("*.csv"))]
    return pd.concat(frames).drop_duplicates("station_id", keep="last").set_index("station_id")["capacity"]


def build_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per station per 5-minute moment during weekday 4-7 PM."""
    lookup = panel.set_index(["station_id", "fetched_at"])["docks_available"]

    def docks_at(offset_minutes: int) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays(
            [panel["station_id"], panel["fetched_at"] + pd.Timedelta(minutes=offset_minutes)]
        )
        return lookup.reindex(keys).to_numpy()

    rows = panel.rename(columns={"docks_available": "docks_now"}).copy()
    rows["docks_change_15"] = rows["docks_now"].to_numpy() - docks_at(-15)
    rows[OUTCOME] = docks_at(AHEAD)
    rows["capacity"] = rows["station_id"].map(station_capacity())

    local = rows["fetched_at"].dt.tz_convert(NEW_YORK)
    rows["date"] = local.dt.date
    rows["minutes_after_4pm"] = (local.dt.hour - RUSH_START) * 60 + local.dt.minute
    in_rush = (local.dt.hour >= RUSH_START) & (local.dt.hour < RUSH_END)

    rows = rows[in_rush & (rows["capacity"] > 0)].dropna(subset=FEATURES + [OUTCOME])
    return rows.reset_index(drop=True)


# ------------------------------------------------------------- 2. Fit and test
def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    return 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def decide(predicted_docks: float) -> str:
    """The product's rule: the one function the app would call."""
    if predicted_docks < LOW_DOCKS:
        return "Go to a different station - this one will likely be full."
    return "Head there - there should be a dock for you."


def minutes_lost(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Average rider minutes lost per arrival under the decision rule."""
    divert = predicted < LOW_DOCKS
    full = actual == 0
    lost = np.where(~divert & full, MINUTES_LOST_FULL, 0) + np.where(divert & ~full, MINUTES_LOST_DETOUR, 0)
    return float(lost.mean())


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = build_rows(load_panel())

    # 70/30 random split
    test_mask = np.random.default_rng(SEED).random(len(rows)) < 0.30
    train, test = rows[~test_mask], rows[test_mask]

    print("\n=== DATA ===")
    print(f"Rows (one per station per 5 minutes, weekday 4-7 PM): {len(rows):,}")
    print(f"Stations: {rows['station_id'].nunique():,}   Days: {rows['date'].nunique()} "
          f"({rows['date'].min()} to {rows['date'].max()})")
    print(f"Train: {len(train):,} rows (70%)   Test: {len(test):,} rows (30%)")

    # Fit v1
    model = sm.OLS(train[OUTCOME], sm.add_constant(train[FEATURES])).fit()
    print("\n=== V1 LINEAR REGRESSION (training rows) ===")
    print(model.summary().tables[1])
    print(f"Adjusted R-squared (training): {model.rsquared_adj:.3f}")

    # Baseline vs v1 on test rows
    actual = test[OUTCOME].to_numpy()
    baseline = np.full(len(test), train[OUTCOME].mean())
    v1 = model.predict(sm.add_constant(test[FEATURES], has_constant="add")).to_numpy()

    results = pd.DataFrame({
        "model": ["baseline (training average)", "v1 regression"],
        "test_R2": [r2(actual, baseline), r2(actual, v1)],
        "typical_error_docks (MAE)": [mae(actual, baseline), mae(actual, v1)],
        "minutes_lost_per_arrival": [minutes_lost(baseline, actual), minutes_lost(v1, actual)],
    })
    results["dollars_lost_per_weekday"] = (
        results["minutes_lost_per_arrival"] * ARRIVALS_PER_WEEKDAY * DOLLARS_PER_MINUTE
    )
    print("\n=== BASELINE VS V1 (test rows) ===")
    print(results.round(3).to_string(index=False))
    results.round(4).to_csv(OUT / "results.csv", index=False)

    # Decision for one realistic new input
    new = pd.DataFrame([{
        "docks_now": 3,  # 3 free docks right now
        "docks_change_15": -3,  # 3 docks filled in the last 15 minutes
        "capacity": 31,
        "minutes_after_4pm": 75,  # 5:15 PM
    }])
    pred = float(model.predict(sm.add_constant(new[FEATURES], has_constant="add"))[0])
    print("\n=== DECISION FOR A NEW INPUT ===")
    print(new.to_string(index=False))
    print(f"Predicted docks at arrival: {pred:.1f}  ->  {decide(pred)}")
    print(f"(Baseline predicts {train[OUTCOME].mean():.1f} for every station -> "
          f"{decide(train[OUTCOME].mean())})")

    saved = results.loc[0, "dollars_lost_per_weekday"] - results.loc[1, "dollars_lost_per_weekday"]
    print(f"\n=== VALUE ===\nv1 saves about ${saved:,.0f} of rider time per weekday evening "
          f"vs the baseline (assumptions at the top of this script).")

    # Chart for the slides
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(results["model"], results["typical_error_docks (MAE)"], color=["#999999", "#1f4e79"])
    for i, v in enumerate(results["typical_error_docks (MAE)"]):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    ax.set_ylabel("Typical error (docks)")
    ax.set_title("Docks 30 minutes ahead: test rows")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "baseline_vs_v1.png", dpi=200)
    print(f"\nSaved {OUT / 'results.csv'} and {OUT / 'baseline_vs_v1.png'}")


if __name__ == "__main__":
    main()
