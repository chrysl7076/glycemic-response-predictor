"""Grouped cross-validation: two baselines vs LightGBM, per target, overall and by subgroup.

Usage:
    python src/train.py --data data/processed/meals.csv --out results
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedGroupKFold

TARGETS = ["delta_30", "delta_60", "delta_120", "iauc_120"]
FEATURES = ["carbs", "protein", "fat", "fiber", "calories", "hour_sin", "hour_cos",
            "min_since_prev_meal", "glucose_t0", "glucose_slope_30",
            "act_kcal_60", "mets_mean_60", "hr_mean_60",
            "age", "sex", "bmi", "a1c", "fasting_glucose", "fasting_insulin", "triglycerides", "hdl"]
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=42, verbose=-1)
N_FOLDS, SEED, N_BOOT = 5, 42, 1000


def cv_predictions(df, target, feats):
    """Out-of-fold predictions for all three models. Folds split by participant, stratified by group."""
    y = df[target].values
    preds = {m: np.full(len(df), np.nan) for m in ["baseline_group_mean", "baseline_carbs_linear", "lightgbm"]}
    folds = np.full(len(df), -1)
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(cv.split(df, df["group"], df["participant"])):
        folds[te] = k
        trd, ted = df.iloc[tr], df.iloc[te]
        # B0: mean response of training participants in the same metabolic group
        gm = trd.groupby("group")[target].mean()
        preds["baseline_group_mean"][te] = ted["group"].map(gm).fillna(trd[target].mean()).values
        # B1: carbohydrate load only
        lr = LinearRegression().fit(trd[["carbs"]], y[tr])
        preds["baseline_carbs_linear"][te] = lr.predict(ted[["carbs"]])
        # Main model
        m = lgb.LGBMRegressor(**LGB_PARAMS).fit(trd[feats], y[tr])
        preds["lightgbm"][te] = m.predict(ted[feats])
    return preds, folds


def boot_ci_improvement(df, err_a, err_b, rng):
    """95% CI for % MAE reduction of b vs a, resampling PARTICIPANTS (not meals)."""
    pids = df["participant"].values
    uniq = np.unique(pids)
    idx = {p: np.where(pids == p)[0] for p in uniq}
    stats = []
    for _ in range(N_BOOT):
        s = np.concatenate([idx[p] for p in rng.choice(uniq, len(uniq), replace=True)])
        stats.append(100 * (1 - err_b[s].mean() / err_a[s].mean()))
    return np.percentile(stats, [2.5, 97.5])


def main(data, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data)
    feats = [f for f in FEATURES if f in df and df[f].notna().any()]
    rng = np.random.default_rng(SEED)
    overall, subgroup = [], []

    for target in TARGETS:
        d = df[df[target].notna() & df["group"].notna()].reset_index(drop=True)
        preds, folds = cv_predictions(d, target, feats)
        errs = {m: np.abs(p - d[target].values) for m, p in preds.items()}
        for m, e in errs.items():
            fold_mae = [e[folds == k].mean() for k in range(N_FOLDS)]
            overall.append({"target": target, "model": m, "mae": e.mean(),
                            "mae_fold_sd": np.std(fold_mae), "n_meals": len(d),
                            "n_participants": d.participant.nunique()})
        for ref in ["baseline_group_mean", "baseline_carbs_linear"]:
            lo, hi = boot_ci_improvement(d, errs[ref], errs["lightgbm"], rng)
            imp = 100 * (1 - errs["lightgbm"].mean() / errs[ref].mean())
            overall.append({"target": target, "model": f"lightgbm_vs_{ref}_pct_improvement",
                            "mae": imp, "ci95_low": lo, "ci95_high": hi})
        d["sex_label"] = d["sex"].map({0: "female", 1: "male"})
        d["age_band"] = pd.cut(d["age"], [0, 39, 120], labels=["<40", "40+"]).astype(str)
        for by in ["group", "sex_label", "age_band"]:
            for grp, gd in d.groupby(by):
                ix = gd.index.values
                row = {"target": target, "split_by": by.replace("_label", ""), "subgroup": grp,
                       "n_participants": gd.participant.nunique(), "n_meals": len(gd)}
                row.update({f"mae_{m}": errs[m][ix].mean() for m in errs})
                subgroup.append(row)

    res = pd.DataFrame(overall); res.to_csv(out / "results_overall.csv", index=False)
    sub = pd.DataFrame(subgroup); sub.to_csv(out / "results_subgroup.csv", index=False)
    pd.set_option("display.width", 160)
    print(res.round(2).to_string(index=False)); print(); print(sub.round(2).to_string(index=False))

    # SHAP on the 60-min model fitted to all data (for explanation, not evaluation)
    d = df[df["delta_60"].notna()]
    model = lgb.LGBMRegressor(**LGB_PARAMS).fit(d[feats], d["delta_60"])
    sv = shap.TreeExplainer(model).shap_values(d[feats])
    shap.summary_plot(sv, d[feats], show=False, max_display=15)
    plt.title("Drivers of predicted 60-min glucose change"); plt.tight_layout()
    plt.savefig(out / "shap_summary_delta60.png", dpi=150); plt.close()
    for f in ["carbs", "hour_sin", "a1c"]:
        if f in feats:
            shap.dependence_plot(f, sv, d[feats], show=False)
            plt.tight_layout(); plt.savefig(out / f"shap_dependence_{f}.png", dpi=150); plt.close()
    pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(0)}) \
        .sort_values("mean_abs_shap", ascending=False).to_csv(out / "shap_importance_delta60.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/meals.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    main(a.data, a.out)
