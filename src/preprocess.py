"""Build a meal-level table from CGMacros.

One row per logged meal: pre-meal features + post-meal glucose targets.
Only information available AT OR BEFORE the meal start is used as a feature.

Usage:
    python src/preprocess.py --raw data/raw/CGMacros --out data/processed/meals.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = [30, 60, 120]          # minutes after meal start
WINDOW = 120                      # post-meal window for iAUC / exclusion
MAX_GAP_MIN = 20                  # interpolate CGM across gaps up to this length only
PRE_ACTIVITY_MIN = 60             # look-back for activity / HR features
GLUCOSE_COL = "Libre GL"          # primary sensor (Dexcom used as fallback)
FALLBACK_COL = "Dexcom GL"
PRE_STEPS, POST_STEPS = 12, 24    # 60 min before / 120 min after, 5-min steps
HISTORY_MIN_MEALS = 3             # personal history features need at least this many earlier meals


def add_history_features(meals):
    """Personal response history, computed ONLY from the same person's EARLIER meals.

    At prediction time these are known: past meals' glucose responses have already been observed.
    The current meal's own outcome is never used (shift(1) before the expanding mean).
    """
    meals = meals.sort_values(["participant", "meal_time"]).copy()
    g = meals.groupby("participant", group_keys=False)
    meals["hist_n_meals"] = g.cumcount()
    for col in ["delta_60", "iauc_120", "peak_delta"]:
        meals[f"hist_mean_{col}"] = g[col].transform(lambda s: s.shift(1).expanding().mean())
    # Personal carb sensitivity: past rise per 10 g carbs (carbs+10 avoids dividing by ~0)
    ratio = meals["delta_60"] / (meals["carbs"] + 10) * 10
    meals["hist_rise_per_10g_carb"] = ratio.groupby(meals["participant"]).transform(
        lambda s: s.shift(1).expanding().median())
    meals["hist_last_delta_60"] = g["delta_60"].shift(1)
    hist_cols = [c for c in meals if c.startswith("hist_") and c != "hist_n_meals"]
    meals.loc[meals["hist_n_meals"] < HISTORY_MIN_MEALS, hist_cols] = np.nan
    return meals


def a1c_group(a1c):
    """ADA HbA1c cut-offs (%). VERIFY against the group labels used in the CGMacros paper."""
    if pd.isna(a1c):
        return np.nan
    if a1c < 5.7:
        return "healthy"
    if a1c < 6.5:
        return "prediabetes"
    return "t2d"


def load_participant(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").drop_duplicates("Timestamp").set_index("Timestamp")
    return df


def glucose_series(df):
    """1-minute glucose series; gaps longer than MAX_GAP_MIN stay NaN."""
    col = GLUCOSE_COL if GLUCOSE_COL in df and df[GLUCOSE_COL].notna().mean() > 0.5 else FALLBACK_COL
    g = df[col].astype(float).resample("1min").mean()
    return g.interpolate(method="time", limit=MAX_GAP_MIN, limit_area="inside"), col


def meal_rows(df, g, pid):
    meals = df[df["Meal Type"].notna() & (df["Meal Type"].astype(str).str.strip() != "")]
    meal_times = meals.index.sort_values()
    rows = []
    for t0, m in meals.iterrows():
        rec = {"participant": pid, "meal_time": t0, "meal_type": str(m["Meal Type"]).strip().lower()}

        # Meal composition, scaled by fraction eaten when available
        frac = m.get("Amount Consumed", np.nan)
        frac = 1.0 if pd.isna(frac) else float(frac) / 100.0
        for k in ["Calories", "Carbs", "Protein", "Fat", "Fiber"]:
            v = m.get(k, np.nan)
            rec[k.lower()] = float(v) * frac if pd.notna(v) else np.nan

        # Timing
        hour = t0.hour + t0.minute / 60
        rec["hour_sin"], rec["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        prev = meal_times[meal_times < t0]
        rec["min_since_prev_meal"] = (t0 - prev[-1]).total_seconds() / 60 if len(prev) else np.nan

        # Exclude if another meal starts inside the post-meal window (response would be confounded)
        nxt = meal_times[meal_times > t0]
        rec["next_meal_in_window"] = bool(len(nxt) and (nxt[0] - t0).total_seconds() / 60 < WINDOW)

        # Pre-meal glucose state (known at t0)
        t0m = t0.floor("min")
        rec["glucose_t0"] = g.get(t0m, np.nan)
        g30 = g.get(t0m - pd.Timedelta(minutes=30), np.nan)
        rec["glucose_slope_30"] = (rec["glucose_t0"] - g30) / 30 if pd.notna(g30) else np.nan

        # Pre-meal activity (look-back only — no post-meal information)
        pre = df.loc[t0 - pd.Timedelta(minutes=PRE_ACTIVITY_MIN): t0 - pd.Timedelta(minutes=1)]
        rec["act_kcal_60"] = pre["Calories (Activity)"].sum(min_count=1) if "Calories (Activity)" in pre else np.nan
        rec["mets_mean_60"] = pre["METs"].mean() if "METs" in pre else (pre["Mets"].mean() if "Mets" in pre else np.nan)
        rec["hr_mean_60"] = pre["HR"].mean() if "HR" in pre else np.nan

        # Targets
        post = g.loc[t0m: t0m + pd.Timedelta(minutes=WINDOW)]
        for h in HORIZONS:
            gh = g.get(t0m + pd.Timedelta(minutes=h), np.nan)
            rec[f"delta_{h}"] = gh - rec["glucose_t0"]
        complete = len(post) == WINDOW + 1 and post.notna().all() and pd.notna(rec["glucose_t0"])
        if complete:
            inc = np.clip(post.values - rec["glucose_t0"], 0, None)
            rec["iauc_120"] = float(np.trapezoid(inc, dx=1.0))   # mg/dL·min
            rec["peak_delta"] = float(post.max() - rec["glucose_t0"])
        else:
            rec["iauc_120"] = rec["peak_delta"] = np.nan
        rec["window_complete"] = complete
        # Curves at 5-min resolution, relative to glucose at meal start (for the temporal model)
        for k in range(PRE_STEPS, 0, -1):          # pre-meal trace: -60 .. -5 min (known at t0)
            v = g.get(t0m - pd.Timedelta(minutes=5 * k), np.nan)
            rec[f"pre_{5 * k}"] = v - rec["glucose_t0"]
        for k in range(1, POST_STEPS + 1):         # post-meal curve: +5 .. +120 min (target)
            v = g.get(t0m + pd.Timedelta(minutes=5 * k), np.nan)
            rec[f"post_{5 * k}"] = v - rec["glucose_t0"]
        rows.append(rec)
    return rows


def load_bio(raw):
    bio = pd.read_csv(Path(raw) / "bio.csv")
    bio.columns = [c.strip() for c in bio.columns]
    id_col = next((c for c in bio.columns if c.lower() in {"subject", "participant", "id", "subject id"}), None)
    if id_col is None:
        raise ValueError(f"No participant id column found in bio.csv; columns = {list(bio.columns)}")
    bio["participant"] = bio[id_col].astype(str).str.extract(r"(\d+)")[0].astype(int)
    keep = {"Age": "age", "Gender": "sex", "BMI": "bmi", "A1c PDL (Lab)": "a1c",
            "Fasting GLU - PDL (Lab)": "fasting_glucose", "Insulin": "fasting_insulin",
            "Triglycerides": "triglycerides", "HDL": "hdl"}
    out = bio[["participant"] + [c for c in keep if c in bio]].rename(columns=keep)
    if "sex" in out:
        out["sex"] = out["sex"].astype(str).str.upper().str[0].map({"F": 0, "M": 1})
    out["group"] = out["a1c"].map(a1c_group)
    return out


def main(raw, out, overlap_rule=True, plausibility=True):
    raw = Path(raw)
    rows, log = [], []
    for csv in sorted(raw.glob("CGMacros-*/CGMacros-*.csv")):
        pid = int(re.search(r"(\d+)", csv.stem).group(1))
        df = load_participant(csv)
        g, col = glucose_series(df)
        r = meal_rows(df, g, pid)
        rows += r
        log.append({"participant": pid, "sensor": col, "meals_logged": len(r),
                    "cgm_coverage": round(g.notna().mean(), 3)})
    meals = pd.DataFrame(rows).merge(load_bio(raw), on="participant", how="left")

    meals["meal_type"] = meals["meal_type"].str.replace(r"^snack.*", "snack", regex=True)

    # Plausibility check on logged macros: energy from macros should roughly match logged calories
    macro_kcal = 4 * meals["carbs"] + 4 * meals["protein"].fillna(0) + 9 * meals["fat"].fillna(0)
    ratio = macro_kcal / meals["calories"].where(meals["calories"] > 0)
    implausible = (meals["calories"] > 3000) | (ratio < 0.5) | (ratio > 2.0)

    n0 = len(meals)
    steps = [("CGM window incomplete", ~meals["window_complete"]),
             ("another meal within 2 h", meals["next_meal_in_window"] & overlap_rule),
             ("carbs missing", meals["carbs"].isna()),
             ("implausible macros (>3000 kcal or macro/kcal mismatch)", implausible & plausibility)]
    keep = pd.Series(True, index=meals.index)
    for name, mask in steps:
        dropped = (keep & mask).sum()
        keep &= ~mask
        print(f"  excluded {dropped:4d}: {name}")
    meals = add_history_features(meals[keep])
    print(f"Meals logged: {n0} | kept after exclusions: {len(meals)} | participants: {meals.participant.nunique()}")
    print(meals.groupby("group").participant.agg(["nunique", "size"]).rename(columns={"nunique": "participants", "size": "meals"}))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    meals.drop(columns=["window_complete", "next_meal_in_window"]).to_csv(out, index=False)
    pd.DataFrame(log).to_csv(Path(out).with_name("participant_log.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/CGMacros")
    ap.add_argument("--out", default="data/processed/meals.csv")
    a = ap.parse_args()
    main(a.raw, a.out)
