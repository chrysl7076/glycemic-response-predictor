"""v2 experiment: personalization + temporal deep model.

Models (all evaluated on the SAME held-out participants, folds grouped by participant):
  B0  group-mean baseline
  B1  carbs-only linear baseline
  M1  LightGBM, static features only (v1 model)
  M2  LightGBM + personal history features (person's own earlier meals)
  M3  Temporal CNN: pre-meal glucose trace + meal + person + history -> full 2-h curve
  M4  M3 fine-tuned on each test person's first half of meals (personalized)

Two evaluation views:
  "all meals"   : every meal of held-out participants (M4 not applicable)
  "later half"  : each held-out person's later 50% of meals (chronological), so M4 can use the
                  first 50% for fine-tuning. All models are scored on exactly these meals.

Usage: python src/train_v2.py --data data/processed/meals.csv --out results_v2
"""
import argparse, copy, json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SEED, N_FOLDS, N_BOOT = 42, 5, 1000
SPIKE_MGDL = 50                                  # "high spike" = peak rise >= 50 mg/dL within 2 h
HORIZONS = [30, 60, 120]
STATIC = ["carbs", "protein", "fat", "fiber", "calories", "hour_sin", "hour_cos", "min_since_prev_meal",
          "glucose_t0", "glucose_slope_30", "act_kcal_60", "mets_mean_60", "hr_mean_60",
          "age", "sex", "bmi", "a1c", "fasting_glucose", "fasting_insulin", "triglycerides", "hdl"]
HIST = ["hist_mean_delta_60", "hist_mean_iauc_120", "hist_mean_peak_delta",
        "hist_rise_per_10g_carb", "hist_last_delta_60"]
PRE = [f"pre_{5 * k}" for k in range(12, 0, -1)]
POST = [f"post_{5 * k}" for k in range(1, 25)]
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, verbose=-1)
torch.set_num_threads(2)


# ---------------------------------------------------------------- temporal CNN
class TemporalCNN(nn.Module):
    """Dilated 1-D convolutions over the pre-meal glucose trace, fused with static features,
    decoding the 24-step (5-min) post-meal curve."""
    def __init__(self, n_static, hidden=32, n_out=24):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden, 3, padding=1, dilation=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4), nn.GELU())
        self.static = nn.Sequential(nn.Linear(n_static, 64), nn.GELU(), nn.Dropout(0.1))
        self.head = nn.Sequential(nn.Linear(hidden * 2 + 64, 64), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(64, n_out))

    def forward(self, seq, stat):
        h = self.conv(seq.unsqueeze(1))                       # (B, hidden, 12)
        h = torch.cat([h.mean(-1), h[..., -1]], dim=1)        # average + most recent state
        return self.head(torch.cat([h, self.static(stat)], dim=1))


class Prep:
    """Standardise inputs using TRAINING data only; median-impute + missing indicators."""
    def fit(self, df, cols):
        self.cols = cols
        self.med = df[cols].median()
        x = df[cols].fillna(self.med)
        self.mu, self.sd = x.mean(), x.std().replace(0, 1)
        return self

    def transform(self, df):
        miss = df[HIST].isna().astype(float).values[:, :1]       # one flag: history available?
        x = ((df[self.cols].fillna(self.med) - self.mu) / self.sd).values
        stat = np.hstack([x, miss]).astype(np.float32)
        seq = (df[PRE].fillna(0).values / 20.0).astype(np.float32)
        return torch.tensor(seq), torch.tensor(stat)


def fit_cnn(model, seq, stat, y, epochs, lr, wd=1e-4, params=None, seed=SEED):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr, weight_decay=wd)
    n = len(y)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            b = perm[i:i + 64]
            loss = (model(seq[b], stat[b]) - y[b]).abs().mean()          # L1 = MAE on the curve
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def predict_cnn(model, seq, stat):
    model.eval()
    with torch.no_grad():
        return model(seq, stat).numpy() * 50.0


def train_cnn_ensemble(trd, prep, n_seeds=3, epochs=120):
    seq, stat = prep.transform(trd)
    y = torch.tensor(trd[POST].values / 50.0, dtype=torch.float32)
    models = []
    for s in range(n_seeds):
        torch.manual_seed(SEED + s)
        m = TemporalCNN(stat.shape[1])
        models.append(fit_cnn(m, seq, stat, y, epochs=epochs, lr=2e-3, seed=SEED + s))
    return models


def finetune(models, prep, person_early, epochs=40):
    """Personalise: adapt only the decoder head on the person's earlier meals (few samples)."""
    seq, stat = prep.transform(person_early)
    y = torch.tensor(person_early[POST].values / 50.0, dtype=torch.float32)
    out = []
    for m in models:
        mm = copy.deepcopy(m)
        fit_cnn(mm, seq, stat, y, epochs=epochs, lr=5e-4, wd=1e-3, params=mm.head.parameters())
        out.append(mm)
    return out


def curve_summary(curve):
    """From a (n, 24) curve of deltas: deltas at horizons, iAUC (trapezoid from 0), peak."""
    full = np.hstack([np.zeros((len(curve), 1)), curve])           # prepend t0 = 0
    out = {f"delta_{h}": full[:, h // 5] for h in HORIZONS}
    out["iauc_120"] = np.trapezoid(np.clip(full, 0, None), dx=5.0, axis=1)
    out["peak_delta"] = full.max(1)
    return out


# ---------------------------------------------------------------- evaluation helpers
def later_half_mask(df):
    rank = df.groupby("participant")["meal_time"].rank(method="first")
    n = df.groupby("participant")["meal_time"].transform("size")
    return (rank > n / 2).values


def boot_ci(pids, fn, rng):
    uniq = np.unique(pids); idx = {p: np.where(pids == p)[0] for p in uniq}
    vals = [fn(np.concatenate([idx[p] for p in rng.choice(uniq, len(uniq))])) for _ in range(N_BOOT)]
    return np.percentile(vals, [2.5, 97.5])


def main(data, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data, parse_dates=["meal_time"])
    df = df[df["group"].notna() & df[PRE].notna().all(axis=1)].reset_index(drop=True)
    df["spike"] = (df["peak_delta"] >= SPIKE_MGDL).astype(int)
    late = later_half_mask(df)
    targets = [f"delta_{h}" for h in HORIZONS] + ["iauc_120", "peak_delta"]
    feats_static = [f for f in STATIC if f in df]
    feats_hist = feats_static + HIST
    models = ["B0_group_mean", "B1_carbs_linear", "M1_lgbm_static", "M2_lgbm_history",
              "M2b_lgbm_personalized", "M3_temporal_cnn", "M4_cnn_personalized"]
    P = {m: {t: np.full(len(df), np.nan) for t in targets} for m in models}
    P_curve = {m: np.full((len(df), 24), np.nan) for m in ["M3_temporal_cnn", "M4_cnn_personalized"]}

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(cv.split(df, df["group"], df["participant"])):
        trd, ted = df.iloc[tr], df.iloc[te]
        print(f"fold {k + 1}/{N_FOLDS}: train {trd.participant.nunique()} people, test {ted.participant.nunique()}", flush=True)
        for t in targets:
            gm = trd.groupby("group")[t].mean()
            P["B0_group_mean"][t][te] = ted["group"].map(gm).values
            P["B1_carbs_linear"][t][te] = LinearRegression().fit(trd[["carbs"]], trd[t]).predict(ted[["carbs"]])
            P["M1_lgbm_static"][t][te] = lgb.LGBMRegressor(**LGB_PARAMS).fit(trd[feats_static], trd[t]).predict(ted[feats_static])
            P["M2_lgbm_history"][t][te] = lgb.LGBMRegressor(**LGB_PARAMS).fit(trd[feats_hist], trd[t]).predict(ted[feats_hist])

        # M2b: LightGBM personalised the simple way - add test people's EARLY meals to training,
        # score on their later meals (fair counterpart to M4 fine-tuning)
        early_te, late_te = te[~late[te]], te[late[te]]
        aug = pd.concat([trd, df.iloc[early_te]])
        for t in targets:
            P["M2b_lgbm_personalized"][t][late_te] = lgb.LGBMRegressor(**LGB_PARAMS).fit(
                aug[feats_hist], aug[t]).predict(df.iloc[late_te][feats_hist])

        prep = Prep().fit(trd, feats_hist)
        ens = train_cnn_ensemble(trd, prep)
        seq, stat = prep.transform(ted)
        curve = np.mean([predict_cnn(m, seq, stat) for m in ens], axis=0)
        P_curve["M3_temporal_cnn"][te] = curve
        for t, v in curve_summary(curve).items():
            P["M3_temporal_cnn"][t][te] = v

        # M4: per test person, fine-tune on their first half, predict their second half
        for pid in ted.participant.unique():
            ix = np.where((df.participant == pid).values)[0]
            early, later = ix[~late[ix]], ix[late[ix]]
            tuned = finetune(ens, prep, df.iloc[early])
            s2, st2 = prep.transform(df.iloc[later])
            c2 = np.mean([predict_cnn(m, s2, st2) for m in tuned], axis=0)
            P_curve["M4_cnn_personalized"][later] = c2
            for t, v in curve_summary(c2).items():
                P["M4_cnn_personalized"][t][later] = v

    # ------------------------------------------------------------ scoring
    rng = np.random.default_rng(SEED)
    pids = df["participant"].values
    rows, sub_rows, cls_rows = [], [], []
    for view, mask in [("all_meals", np.ones(len(df), bool)), ("later_half", late)]:
        for m in models:
            if m in ("M4_cnn_personalized", "M2b_lgbm_personalized") and view == "all_meals":
                continue
            for t in targets:
                e = np.abs(P[m][t] - df[t].values)
                rows.append({"view": view, "model": m, "target": t, "mae": e[mask].mean(), "n_meals": int(mask.sum())})
            if m in P_curve:
                ce = np.abs(P_curve[m] - df[POST].values).mean(1)
                rows.append({"view": view, "model": m, "target": "full_curve_5min", "mae": ce[mask].mean(), "n_meals": int(mask.sum())})
            # spike detection, using predicted peak as the score
            y, s = df["spike"].values[mask], P[m]["peak_delta"][mask]
            pred = s >= SPIKE_MGDL
            cls_rows.append({"view": view, "model": m, "auroc": roc_auc_score(y, s),
                             "sensitivity": (pred & (y == 1)).sum() / max((y == 1).sum(), 1),
                             "specificity": (~pred & (y == 0)).sum() / max((y == 0).sum(), 1),
                             "spike_rate": y.mean()})

    res = pd.DataFrame(rows)
    # CIs for the key comparisons (delta_60, participant bootstrap)
    ci_rows = []
    def pct_imp(a, b, mask):
        ea, eb = np.abs(P[a]["delta_60"] - df["delta_60"].values), np.abs(P[b]["delta_60"] - df["delta_60"].values)
        sel = np.where(mask)[0]
        f = lambda s: 100 * (1 - eb[sel][s].mean() / ea[sel][s].mean())
        lo, hi = boot_ci(pids[sel], f, rng)
        return f(np.arange(len(sel))), lo, hi
    for view, mask, comps in [
        ("all_meals", np.ones(len(df), bool), [("B1_carbs_linear", "M1_lgbm_static"), ("B1_carbs_linear", "M2_lgbm_history"),
                                              ("M1_lgbm_static", "M2_lgbm_history"), ("M2_lgbm_history", "M3_temporal_cnn")]),
        ("later_half", late, [("B1_carbs_linear", "M2b_lgbm_personalized"), ("M4_cnn_personalized", "M2b_lgbm_personalized"),
                              ("B1_carbs_linear", "M4_cnn_personalized"), ("M2_lgbm_history", "M4_cnn_personalized"),
                              ("M3_temporal_cnn", "M4_cnn_personalized")])]:
        for a, b in comps:
            imp, lo, hi = pct_imp(a, b, mask)
            ci_rows.append({"view": view, "reference": a, "model": b, "pct_mae_reduction_delta60": imp, "ci95_low": lo, "ci95_high": hi})

    # Subgroups (delta_60, all meals) with participant-bootstrap CIs on MAE
    for by in ["group", "sex", "age_band"]:
        d = df.copy()
        d["sex"] = d["sex"].map({0: "female", 1: "male"})
        d["age_band"] = pd.cut(d["age"], [0, 39, 120], labels=["<40", "40+"]).astype(str)
        for g, gd in d.groupby(by):
            ix = gd.index.values
            for m in ["B1_carbs_linear", "M1_lgbm_static", "M2_lgbm_history", "M3_temporal_cnn"]:
                e = np.abs(P[m]["delta_60"][ix] - df["delta_60"].values[ix])
                lo, hi = boot_ci(pids[ix], lambda s: e[s].mean(), rng)
                sub_rows.append({"split_by": by, "subgroup": g, "model": m, "n_participants": gd.participant.nunique(),
                                 "n_meals": len(ix), "mae_delta60": e.mean(), "ci95_low": lo, "ci95_high": hi})

    res.to_csv(out / "results_v2_mae.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(out / "results_v2_comparisons.csv", index=False)
    pd.DataFrame(sub_rows).to_csv(out / "results_v2_subgroups.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(out / "results_v2_spike_detection.csv", index=False)
    np.savez_compressed(out / "predictions_v2.npz", **{f"{m}__{t}": P[m][t] for m in models for t in targets},
                        **{f"{m}__curve": c for m, c in P_curve.items()}, late=late)
    df[["participant", "meal_time", "group"] + POST].to_csv(out / "meal_index.csv", index=False)

    pd.set_option("display.width", 200)
    print(res.pivot_table(index=["view", "model"], columns="target", values="mae").round(1).to_string())
    print(pd.DataFrame(ci_rows).round(1).to_string(index=False))
    print(pd.DataFrame(cls_rows).round(3).to_string(index=False))
    print(pd.DataFrame(sub_rows).round(1).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/meals.csv")
    ap.add_argument("--out", default="results_v2")
    a = ap.parse_args()
    main(a.data, a.out)
