"""Fake data in the documented CGMacros layout — ONLY for testing that the pipeline runs."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
root = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/synthetic/CGMacros")
bio = []
for pid in range(1, 46):
    a1c = rng.choice([rng.uniform(4.8, 5.6), rng.uniform(5.7, 6.4), rng.uniform(6.5, 8.5)])
    bio.append({"subject": pid, "Age": rng.integers(18, 70), "Gender": rng.choice(["F", "M"]),
                "BMI": rng.uniform(21, 45), "A1c PDL (Lab)": a1c, "Fasting GLU - PDL (Lab)": 70 + 20 * a1c,
                "Insulin": rng.uniform(3, 40), "Triglycerides": rng.uniform(50, 300), "HDL": rng.uniform(30, 90)})
    ts = pd.date_range("2020-01-01", periods=10 * 1440, freq="1min")
    base = 60 + 8 * a1c + 5 * np.sin(np.arange(len(ts)) / 300)
    g = base.copy()
    df = pd.DataFrame({"Timestamp": ts, "HR": rng.normal(75, 8, len(ts)),
                       "Calories (Activity)": rng.gamma(1, 1.2, len(ts)), "METs": rng.integers(10, 30, len(ts))})
    for col in ["Meal Type", "Calories", "Carbs", "Protein", "Fat", "Fiber", "Amount Consumed"]:
        df[col] = np.nan
    df["Meal Type"] = df["Meal Type"].astype(object)
    for day in range(10):
        for mt, hr in [("Breakfast", 8), ("Lunch", 13), ("Dinner", 19)]:
            i = day * 1440 + hr * 60 + rng.integers(-30, 30)
            carbs = rng.uniform(10, 120)
            df.loc[i, ["Meal Type", "Carbs", "Protein", "Fat", "Fiber", "Amount Consumed"]] = \
                [mt, carbs, rng.uniform(5, 50), rng.uniform(2, 40), rng.uniform(0, 12), 100]
            df.loc[i, "Calories"] = 4 * carbs + 30
            t = np.arange(180)
            g[i:i + 180] += carbs * (0.3 + 0.15 * (a1c - 5)) * (t / 45) * np.exp(1 - t / 45)
    df["Libre GL"] = np.where(np.arange(len(ts)) % 15 == 0, g + rng.normal(0, 4, len(ts)), np.nan)
    df["Dexcom GL"] = np.where(np.arange(len(ts)) % 5 == 0, g + rng.normal(0, 4, len(ts)), np.nan)
    out = root / f"CGMacros-{pid:03d}"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"CGMacros-{pid:03d}.csv", index=False)
pd.DataFrame(bio).to_csv(root / "bio.csv", index=False)
print("synthetic data written to", root)
