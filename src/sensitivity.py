"""Sensitivity analysis: does the main result survive other sensor / cleaning choices?

Usage: python src/sensitivity.py --raw data/raw/CGMacros --out results_v2
"""
import argparse, contextlib, io, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).parent))
import preprocess as pp
from train_v2 import STATIC, HIST, LGB_PARAMS, SEED, StratifiedGroupKFold, LinearRegression


def run(raw, name, sensor="Libre GL", **kw):
    pp.GLUCOSE_COL, pp.FALLBACK_COL = (sensor, "Dexcom GL" if sensor == "Libre GL" else "Libre GL")
    path = f"data/processed/sens_{name}.csv"
    with contextlib.redirect_stdout(io.StringIO()):
        pp.main(raw, path, **kw)
    df = pd.read_csv(path); df = df[df.group.notna()].reset_index(drop=True)
    fs = [f for f in STATIC if f in df]; fh = fs + HIST
    P = {k: np.zeros(len(df)) for k in ["B1", "M1", "M2"]}
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(df, df.group, df.participant):
        a, b = df.iloc[tr], df.iloc[te]
        P["B1"][te] = LinearRegression().fit(a[["carbs"]], a.delta_60).predict(b[["carbs"]])
        P["M1"][te] = lgb.LGBMRegressor(**LGB_PARAMS).fit(a[fs], a.delta_60).predict(b[fs])
        P["M2"][te] = lgb.LGBMRegressor(**LGB_PARAMS).fit(a[fh], a.delta_60).predict(b[fh])
    e = {k: np.abs(v - df.delta_60).mean() for k, v in P.items()}
    return dict(setting=name, meals=len(df), mae_carbs_only=e["B1"], mae_lgbm=e["M1"],
                mae_lgbm_history=e["M2"], pct_reduction_vs_carbs=100 * (1 - e["M2"] / e["B1"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/CGMacros"); ap.add_argument("--out", default="results_v2")
    a = ap.parse_args(); Path(a.out).mkdir(exist_ok=True)
    rows = [run(a.raw, "main"), run(a.raw, "dexcom_sensor", sensor="Dexcom GL"),
            run(a.raw, "no_overlap_exclusion", overlap_rule=False),
            run(a.raw, "no_plausibility_filter", plausibility=False)]
    r = pd.DataFrame(rows).round(2); r.to_csv(Path(a.out) / "results_v2_sensitivity.csv", index=False)
    print(r.to_string(index=False))
