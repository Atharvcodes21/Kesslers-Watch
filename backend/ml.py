"""AI risk model (build step 5): trained on ESA's Kelvins Collision Avoidance Challenge CDMs.

Run `python ml.py train` to (re)build backend/models/ from backend/data/kelvins/ (needs pandas, which
only training uses; the API server imports just numpy, scikit-learn and joblib).

Task (Kelvins): each event_id is a time series of Conjunction Data Messages (CDMs). The target is the
`risk` (log10 collision probability) in the event's LAST CDM. Inputs may only use CDMs with
time_to_tca >= 2 days (an avoidance manoeuvre is planned at least 2 days before TCA).
High-risk = final risk >= -6 (Kelvins scoring: r >= 1e-6).

Models
  baseline : final risk = latest available risk ("latest risk prediction", LRP)
  full     : HistGradientBoosting; isotonic-calibrated classifier -> P(final risk >= -6), and a regressor
             for the CHANGE from the latest risk (so zero change = the baseline). Uses the latest CDM
             (all columns) plus simple trend features from the earlier CDMs.
  live     : same families but ONLY on quantities our live screener also produces (see LIVE_FEATURE_MAP).
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger("kesslerwatch.ml")

BASE = Path(__file__).parent
DATA_DIR = BASE / "data" / "kelvins" / "kelvins_competition_data"
MODEL_DIR = BASE / "models"
METRICS_PATH = MODEL_DIR / "metrics.json"

SEED = 42
HIGH_RISK = -6.0  # log10; Kelvins: r >= 1e-6 is "high risk"
MIN_TCA_DAYS = 2.0  # inputs only from CDMs at least this far from TCA
MAX_TCA_DAYS = 7.0  # Kelvins CDMs are issued up to ~7 days before TCA
RISK_FLOOR = -30.0  # Kelvins clips risk at 1e-30
HOLDOUT_FRACTION = 0.2

# Kelvins `c_object_type` (the chaser / the other object) -> code; our catalog's OBJECT_TYPE maps the same way.
OBJECT_CODES = {"DEBRIS": 0, "PAYLOAD": 1, "ROCKET BODY": 2, "UNKNOWN": 3, "TBA": 3}

KELVINS_URL = "https://kelvins.esa.int/collision-avoidance-challenge/"
PAPER_URL = "https://arxiv.org/abs/2008.03069"
# Published results (Uriot et al., 2020, arXiv:2008.03069) on the official test set, for context only.
PUBLISHED = {
    "latest_risk_baseline": {"score": 0.694, "mse_hr": 0.513, "f2": 0.739},
    "winner_sesc": {"score": 0.556, "mse_hr": 0.407, "f2": 0.733},
    "source": PAPER_URL,
}

# Live model inputs, in order. Kelvins column (or derivation) -> what our /screen result provides.
LIVE_FEATURES = ["miss_distance_m", "relative_speed_ms", "time_to_tca_d", "log10_pc", "sigma_km", "miss_over_sigma", "object_type"]
LIVE_CATEGORICAL = ["object_type"]
LIVE_FEATURE_MAP = [
    {"feature": "miss_distance_m", "kelvins": "miss_distance [m]", "live": "miss_km × 1000"},
    {"feature": "relative_speed_ms", "kelvins": "relative_speed [m/s]", "live": "vrel_kms × 1000"},
    {"feature": "time_to_tca_d", "kelvins": "time_to_tca [days] of the latest CDM ≥ 2 d out", "live": "(TCA − now) in days; below 2 d the model is outside its training range"},
    {"feature": "log10_pc", "kelvins": "risk (ESA's log10 collision probability, full covariance)", "live": "log10(Pc) from our screener (isotropic assumed σ), floored at −30"},
    {"feature": "sigma_km", "kelvins": "√(((c_σr²+c_σt²+c_σn²)+(t_σr²+t_σt²+t_σn²))/3) / 1000", "live": "combined isotropic σ (km); identical formula when σ is isotropic"},
    {"feature": "miss_over_sigma", "kelvins": "miss_distance / (1000·sigma_km)", "live": "miss_km / sigma_km"},
    {"feature": "object_type", "kelvins": "c_object_type (TBA → UNKNOWN)", "live": "object_type of the other object (official Space-Track type)"},
]

# Columns whose history we summarise (value at first CDM, last change, slope per day toward TCA).
TREND_COLUMNS = [
    "risk", "max_risk_estimate", "miss_distance", "mahalanobis_distance",
    "c_position_covariance_det", "t_position_covariance_det", "c_sigma_r", "c_sigma_t", "c_sigma_n",
    "c_obs_used", "relative_speed",
]


# =========================================================================== feature construction (training)


def _isotropic_sigma_km(df) -> np.ndarray:
    """Combined isotropic-equivalent 1-sigma position error (km) from per-object R/T/N sigmas (m)."""
    cols = [f"{o}_sigma_{a}" for o in "ct" for a in "rtn"]
    sq = sum(df[c].to_numpy(float) ** 2 for c in cols)
    return np.sqrt(sq / 3.0) / 1000.0


def _slopes(d, col: str, t_col: str = "t_days"):
    """Per-event least-squares slope of `col` vs elapsed days, NaN if < 2 valid points."""
    sub = d[["event_id", t_col, col]].dropna()
    sub = sub.assign(tt=sub[t_col] ** 2, ty=sub[t_col] * sub[col])
    g = sub.groupby("event_id")
    n = g.size()
    sx, sy, sxy, sxx = g[t_col].sum(), g[col].sum(), g["ty"].sum(), g["tt"].sum()
    den = n * sxx - sx**2
    slope = (n * sxy - sx * sy) / den.where(den.abs() > 1e-9)
    return slope.where(n >= 2)


def build_event_table(df):
    """One row per event from CDMs with time_to_tca >= 2 d. Returns (feature DataFrame, latest-risk Series)."""
    import pandas as pd

    d = df[df["time_to_tca"] >= MIN_TCA_DAYS].sort_values(["event_id", "time_to_tca"], ascending=[True, False]).copy()
    g = d.groupby("event_id", sort=True)
    d["rev"] = g.cumcount(ascending=False)  # 0 = latest CDM at least 2 d out
    d["t_days"] = d.groupby("event_id")["time_to_tca"].transform("max") - d["time_to_tca"]  # elapsed since first CDM

    latest = d[d["rev"] == 0].set_index("event_id")
    prev = d[d["rev"] == 1].set_index("event_id").reindex(latest.index)
    first = d.groupby("event_id").head(1).set_index("event_id").reindex(latest.index)

    skip = {"event_id", "c_object_type", "rev", "t_days"}
    cols: dict = {c: latest[c].astype(float) for c in latest.columns if c not in skip}
    cols["object_type"] = latest["c_object_type"].map(OBJECT_CODES).astype(float)
    sigma = pd.Series(_isotropic_sigma_km(latest), index=latest.index)
    cols["sigma_km"] = sigma
    cols["miss_over_sigma"] = latest["miss_distance"] / (1000.0 * sigma.where(sigma > 0))
    cols["n_cdm"] = d.groupby("event_id").size().reindex(latest.index).astype(float)
    cols["span_days"] = latest["t_days"].astype(float)
    cols["risk_min"] = d.groupby("event_id")["risk"].min().reindex(latest.index)
    cols["risk_max"] = d.groupby("event_id")["risk"].max().reindex(latest.index)
    for c in TREND_COLUMNS:
        if c not in latest.columns:
            continue
        cols[f"{c}_first"] = first[c].astype(float)
        cols[f"{c}_delta"] = (latest[c] - prev[c]).astype(float)
        cols[f"{c}_slope"] = _slopes(d, c).reindex(latest.index)
    feats = pd.DataFrame(cols, index=latest.index).rename(columns={"time_to_tca": "time_to_tca_d", "risk": "latest_risk"})
    return feats, latest["risk"].astype(float)


def live_matrix(feats):
    """Live-model inputs from an event table (the Kelvins side of LIVE_FEATURE_MAP)."""
    out = feats[["miss_distance", "relative_speed", "time_to_tca_d", "latest_risk", "sigma_km", "miss_over_sigma", "object_type"]].copy()
    out.columns = LIVE_FEATURES
    return out


def final_risk_by_event(df):
    last = df.sort_values(["event_id", "time_to_tca"], ascending=[True, False]).groupby("event_id").tail(1).set_index("event_id")
    return last["risk"].astype(float), last["time_to_tca"].astype(float)


# =========================================================================== metrics


def _f_beta(p: float, r: float, beta: float) -> float:
    return (1 + beta**2) * p * r / (beta**2 * p + r) if (p + r) > 0 else 0.0


def score_set(y_true: np.ndarray, pred_risk: np.ndarray, p_high: np.ndarray, pred_high: np.ndarray, rank_score: np.ndarray) -> dict:
    """All requested metrics. y_true / pred_risk are log10 risk; p_high in [0,1]; pred_high is the 0/1 decision."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = (y_true >= HIGH_RISK).astype(int)
    pred = pred_high.astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f2 = _f_beta(prec, rec, 2.0)
    hr = y == 1
    # Official Kelvins MSE: high-risk events only, with predictions clipped below at -6 (everything under -6 is
    # "low risk"). This reproduces the published latest-risk baseline exactly (MSE 0.513, L 0.694 on the test set).
    mse_hr = float(np.mean((np.clip(pred_risk[hr], HIGH_RISK, None) - y_true[hr]) ** 2)) if hr.any() else float("nan")
    mse_hr_raw = float(np.mean((pred_risk[hr] - y_true[hr]) ** 2)) if hr.any() else float("nan")
    both = 0 < y.sum() < len(y)
    return {
        "n": int(len(y)),
        "n_high_risk": int(y.sum()),
        "brier": float(np.mean((p_high - y) ** 2)),
        "roc_auc": float(roc_auc_score(y, rank_score)) if both else float("nan"),
        "avg_precision": float(average_precision_score(y, rank_score)) if both else float("nan"),
        "precision": prec,
        "recall": rec,
        "f1": _f_beta(prec, rec, 1.0),
        "f2": f2,
        "false_alert_rate": fp / (fp + tn) if fp + tn else 0.0,  # false positives / all low-risk events
        "false_alerts_per_1000": 1000.0 * fp / len(y),
        "alerts": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "mse_log_risk": float(np.mean((pred_risk - y_true) ** 2)),
        "mse_high_risk": mse_hr,
        "mse_high_risk_unclipped": mse_hr_raw,
        # Official Kelvins loss L = MSE_HR / F2 (lower is better).
        "official_score": (mse_hr / f2) if f2 > 0 else float("inf"),
    }


def calibration_table(y_true: np.ndarray, p: np.ndarray) -> list[dict]:
    edges = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0000001]
    y = (y_true >= HIGH_RISK).astype(float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            rows.append({"bin": f"{lo:.2f}–{min(hi, 1.0):.2f}", "n": int(m.sum()), "mean_predicted": float(p[m].mean()), "observed_rate": float(y[m].mean()), "positives": int(y[m].sum())})
    return rows


def bootstrap_ci(y_true, preds: dict[str, dict], n_boot: int = 600, seed: int = SEED) -> dict:
    """95% bootstrap CIs (resampling events) for key metrics, and for each model's difference to the baseline."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ["brier", "roc_auc", "f2", "mse_high_risk", "official_score"]
    draws: dict[str, dict[str, list[float]]] = {m: {k: [] for k in keys} for m in preds}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if (y_true[idx] >= HIGH_RISK).sum() < 3:
            continue
        for m, p in preds.items():
            s = score_set(y_true[idx], p["risk"][idx], p["p"][idx], p["high"][idx], p["rank"][idx])
            for k in keys:
                draws[m][k].append(s[k])
    out: dict = {"n_boot": n_boot, "models": {}, "vs_baseline": {}}
    for m in preds:
        out["models"][m] = {k: _ci(draws[m][k]) for k in keys}
        if m != "baseline":
            out["vs_baseline"][m] = {k: _ci(np.array(draws[m][k]) - np.array(draws["baseline"][k])) for k in keys}
    return out


def _ci(v) -> dict:
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 20:
        return {"lo": None, "hi": None}
    return {"lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5))}


# =========================================================================== training


def _best_f2_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best, best_t = -1.0, 0.5
    for t in np.unique(np.round(np.concatenate([np.linspace(0.005, 0.95, 190), p]), 4)):
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        prec = tp / pred.sum() if pred.sum() else 0.0
        rec = tp / max(int((y == 1).sum()), 1)
        f2 = _f_beta(prec, rec, 2.0)
        if f2 > best:
            best, best_t = f2, float(t)
    return best_t


def _recall_matched_threshold(y: np.ndarray, p: np.ndarray, target_recall: float) -> float:
    """Highest probability threshold whose recall on these (out-of-fold) predictions is >= target_recall."""
    pos = np.sort(p[y == 1])[::-1]  # positives' probabilities, descending
    k = max(int(math.ceil(target_recall * len(pos))), 1)  # must catch at least this many positives
    return float(pos[min(k, len(pos)) - 1])


def _make_clf(depth: int, iters: int, cat_idx: list[int]):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=depth, max_iter=iters, min_samples_leaf=30, l2_regularization=1.0,
        categorical_features=cat_idx or None, random_state=SEED,
    )


def _make_reg(depth: int, iters: int, cat_idx: list[int]):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_depth=depth, max_iter=iters, min_samples_leaf=30, l2_regularization=1.0,
        categorical_features=cat_idx or None, random_state=SEED,
    )


def fit_family(X: np.ndarray, y_final: np.ndarray, latest: np.ndarray, names: list[str], cat_names: list[str]) -> dict:
    """Tune lightly by 5-fold CV on the training events only, then fit calibrated classifier + regressor."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    cat_idx = [names.index(c) for c in cat_names if c in names]
    y = (y_final >= HIGH_RISK).astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    best = None
    for depth in (2, 3, 4):
        for iters in (100, 250):
            oof = cross_val_predict(_make_clf(depth, iters, cat_idx), X, y, cv=cv, method="predict_proba")[:, 1]
            ll = log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))
            if best is None or ll < best[0]:
                best = (ll, depth, iters)
    _, depth, iters = best

    cal = CalibratedClassifierCV(_make_clf(depth, iters, cat_idx), method="isotonic", cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    oof_cal = cross_val_predict(CalibratedClassifierCV(_make_clf(depth, iters, cat_idx), method="isotonic", cv=StratifiedKFold(5, shuffle=True, random_state=SEED + 1)),
                                X, y, cv=cv, method="predict_proba")[:, 1]
    # Two operating points, both chosen on OUT-OF-FOLD TRAINING predictions only (never on test data):
    #  - threshold: catch at least as many high-risk events as the baseline does (the baseline's training recall),
    #    with as few false alerts as possible. This is the model's primary operating point.
    #  - threshold_f2: maximise F2 at the training prevalence (~1% high-risk); reported as an alternative.
    base_recall = float(np.mean(latest[y == 1] >= HIGH_RISK))
    threshold = _recall_matched_threshold(y, oof_cal, base_recall)
    threshold_f2 = _best_f2_threshold(y, oof_cal)
    cal.fit(X, y)

    # Regressor predicts the CHANGE from the latest risk; zero change reproduces the baseline.
    resid = y_final - latest
    best_r = None
    for rdepth in (2, 3, 4):
        for riters in (100, 250):
            oof_r = cross_val_predict(_make_reg(rdepth, riters, cat_idx), X, resid, cv=_kfold())
            mse = float(np.mean((latest + oof_r - y_final) ** 2))
            if best_r is None or mse < best_r[0]:
                best_r = (mse, rdepth, riters)
    _, rdepth, riters = best_r
    reg = _make_reg(rdepth, riters, cat_idx).fit(X, resid)

    # High-risk regressor: trained ONLY on events whose final risk is high, predicting the change from the
    # latest risk (floored at -6). It is used for events the classifier flags. The shrink weight alpha
    # (0 = plain baseline, 1 = full regressor) is chosen by cross-validation on those training events only.
    from sklearn.model_selection import KFold

    hr = y_final >= HIGH_RISK
    base_hr = np.clip(latest[hr], HIGH_RISK, None)
    resid_hr = y_final[hr] - base_hr
    Xh = X[hr]
    oof_h = np.zeros(hr.sum())
    for tr_i, va_i in KFold(5, shuffle=True, random_state=SEED).split(Xh):
        m = _make_reg(2, 60, cat_idx)
        m.set_params(min_samples_leaf=8)
        oof_h[va_i] = m.fit(Xh[tr_i], resid_hr[tr_i]).predict(Xh[va_i])
    alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
    cv_mse = {a: float(np.mean((base_hr + a * oof_h - y_final[hr]) ** 2)) for a in alphas}
    alpha = min(cv_mse, key=cv_mse.get)
    reg_hr = _make_reg(2, 60, cat_idx)
    reg_hr.set_params(min_samples_leaf=8)
    reg_hr.fit(Xh, resid_hr)
    return {
        "clf": cal, "reg": reg, "reg_hr": reg_hr, "alpha_hr": alpha, "threshold": threshold, "threshold_f2": threshold_f2,
        "baseline_train_recall": base_recall, "names": names, "cat_names": cat_names,
        "hyper": {"clf_depth": depth, "clf_iters": iters, "reg_depth": rdepth, "reg_iters": riters, "hr_alpha": alpha, "hr_cv_mse_by_alpha": cv_mse, "hr_train_events": int(hr.sum())},
    }


def _kfold():
    from sklearn.model_selection import KFold

    return KFold(5, shuffle=True, random_state=SEED)


def predict_family(m: dict, X: np.ndarray, latest: np.ndarray, threshold: float | None = None) -> dict:
    p = m["clf"].predict_proba(X)[:, 1]
    high = p >= (m["threshold"] if threshold is None else threshold)
    risk = np.clip(latest + m["reg"].predict(X), RISK_FLOOR, -1.0)  # all-event estimate
    # Events flagged high-risk get the high-risk regressor's estimate (never below the -6 threshold).
    risk_hr = np.clip(np.clip(latest, HIGH_RISK, None) + m["alpha_hr"] * m["reg_hr"].predict(X), HIGH_RISK, -1.0)
    risk = np.where(high, risk_hr, risk)
    return {"p": p, "risk": risk, "high": high, "rank": p}


def baseline_predictions(latest: np.ndarray) -> dict:
    high = latest >= HIGH_RISK
    return {"p": high.astype(float), "risk": latest, "high": high, "rank": latest}


def importance(m: dict, X: np.ndarray, y_final: np.ndarray, top: int = 15) -> list[dict]:
    """Permutation importance of the calibrated classifier: rise in Brier score when a feature is shuffled."""
    from sklearn.inspection import permutation_importance

    y = (y_final >= HIGH_RISK).astype(int)
    r = permutation_importance(m["clf"], X, y, scoring="neg_brier_score", n_repeats=8, random_state=SEED, n_jobs=1)
    order = np.argsort(-r.importances_mean)[:top]
    return [{"feature": m["names"][i], "importance": float(r.importances_mean[i]), "std": float(r.importances_std[i])} for i in order]


def train(data_dir: Path = DATA_DIR) -> dict:
    import joblib
    import pandas as pd
    import sklearn
    from sklearn.model_selection import train_test_split

    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("loading CSVs…", flush=True)
    tr = pd.read_csv(data_dir / "train_data" / "train_data.csv", low_memory=False)
    te = pd.read_csv(data_dir / "test_data.csv", low_memory=False)
    pv = pd.read_csv(data_dir / "test_data_private.csv", low_memory=False)

    # ---- training events: inputs from CDMs >= 2 d out; target = risk of the event's last CDM
    feats_all, latest_all = build_event_table(tr)
    final, last_tca = final_risk_by_event(tr)
    y_final_all = final.reindex(feats_all.index)
    eligible = last_tca.reindex(feats_all.index) < MIN_TCA_DAYS  # the final CDM itself must be within 2 days
    feats_all, latest_all, y_final_all = feats_all[eligible], latest_all[eligible], y_final_all[eligible]
    n_events_total = tr["event_id"].nunique()
    print(f"events: {n_events_total} total, {len(feats_all)} eligible, {int((y_final_all >= HIGH_RISK).sum())} high-risk", flush=True)

    # ---- fixed-seed split by event (each row here is a whole event, so no event can be in both sets)
    ids = feats_all.index.to_numpy()
    strat = (y_final_all >= HIGH_RISK).astype(int).to_numpy()
    tr_ids, ho_ids = train_test_split(ids, test_size=HOLDOUT_FRACTION, random_state=SEED, stratify=strat)
    assert not set(tr_ids) & set(ho_ids)

    # ---- official test set (inputs) and labels (true_risk of the final CDM, from test_data_private.csv)
    feats_te, latest_te = build_event_table(te)
    y_te = pv.set_index("event_id")["true_risk"].reindex(feats_te.index).to_numpy(float)
    assert not np.isnan(y_te).any() and te["time_to_tca"].min() >= MIN_TCA_DAYS

    full_names = list(feats_all.columns)
    full_cat = ["mission_id", "object_type"]
    live_all, live_te = live_matrix(feats_all), live_matrix(feats_te)

    def arrays(f, idx):
        return f.loc[idx].to_numpy(float)

    data = {
        "holdout": dict(full=arrays(feats_all, ho_ids), live=arrays(live_all, ho_ids), latest=latest_all.loc[ho_ids].to_numpy(float), y=y_final_all.loc[ho_ids].to_numpy(float)),
        "official_test": dict(full=feats_te.to_numpy(float), live=live_te.to_numpy(float), latest=latest_te.to_numpy(float), y=y_te),
    }
    Xtr_full, Xtr_live = arrays(feats_all, tr_ids), arrays(live_all, tr_ids)
    ltr, ytr = latest_all.loc[tr_ids].to_numpy(float), y_final_all.loc[tr_ids].to_numpy(float)

    print(f"fitting full model on {len(tr_ids)} events ({int((ytr >= HIGH_RISK).sum())} high-risk)…", flush=True)
    full = fit_family(Xtr_full, ytr, ltr, full_names, full_cat)
    print("  thresholds (recall-matched / F2-opt):", round(full["threshold"], 4), round(full["threshold_f2"], 4), "| base recall", round(full["baseline_train_recall"], 3), f"| {time.time()-t0:.0f}s", flush=True)
    print("fitting live model…", flush=True)
    live = fit_family(Xtr_live, ytr, ltr, list(LIVE_FEATURES), LIVE_CATEGORICAL)
    print("  thresholds (recall-matched / F2-opt):", round(live["threshold"], 4), round(live["threshold_f2"], 4), f"| {time.time()-t0:.0f}s", flush=True)

    results, calib, intervals = {}, {}, {}
    for name, d in data.items():
        preds = {
            "baseline": baseline_predictions(d["latest"]),
            "full": predict_family(full, d["full"], d["latest"]),
            "live": predict_family(live, d["live"], d["latest"]),
        }
        results[name] = {m: score_set(d["y"], p["risk"], p["p"], p["high"], p["rank"]) for m, p in preds.items()}
        # Alternative operating point: the F2-optimal threshold at the training prevalence.
        for m, fam in (("full", full), ("live", live)):
            alt = predict_family(fam, d[m], d["latest"], threshold=fam["threshold_f2"])
            a = score_set(d["y"], alt["risk"], alt["p"], alt["high"], alt["rank"])
            results[name][m]["alt_f2_threshold"] = {"threshold": fam["threshold_f2"], **{k: a[k] for k in ("precision", "recall", "f2", "false_alerts_per_1000", "mse_high_risk", "official_score")}}
        calib[name] = {m: calibration_table(d["y"], preds[m]["p"]) for m in ("full", "live")}
        intervals[name] = bootstrap_ci(d["y"], preds)

    imp = {
        "full": importance(full, data["holdout"]["full"], data["holdout"]["y"]),
        "live": importance(live, data["holdout"]["live"], data["holdout"]["y"], top=len(LIVE_FEATURES)),
    }

    sizes = {}
    for name, m in (("full", full), ("live", live)):
        joblib.dump(m, MODEL_DIR / f"{name}.joblib", compress=3)
        sizes[name] = (MODEL_DIR / f"{name}.joblib").stat().st_size

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "seed": SEED,
        "versions": {"scikit_learn": sklearn.__version__, "numpy": np.__version__},
        "dataset": {
            "name": "ESA Collision Avoidance Challenge (Kelvins)", "url": KELVINS_URL, "paper": PAPER_URL, "years": "2015–2019",
            "description": "Conjunction Data Messages (CDMs) received by ESA's Space Debris Office for its satellites, anonymised.",
            "train_events": int(n_events_total), "train_events_eligible": int(len(feats_all)),
            "train_high_risk_events": int((y_final_all >= HIGH_RISK).sum()),
            "official_test_events": int(len(feats_te)), "official_test_high_risk_events": int((y_te >= HIGH_RISK).sum()),
            "note": "The official test set was hand-picked to over-represent high-risk events (6.9% vs ~1% in the eligible training events), so Brier scores and calibration differ between the two evaluations.",
        },
        "task": {
            "target": "risk (log10) in the last CDM of the event", "inputs": "CDMs with time_to_tca >= 2 days only",
            "high_risk": "final risk >= -6 (collision probability >= 1e-6)",
            "eligibility": "training events need >= 1 CDM at >= 2 days and a final CDM within 2 days of TCA",
        },
        "split": {"train_events": int(len(tr_ids)), "holdout_events": int(len(ho_ids)), "holdout_high_risk": int((data["holdout"]["y"] >= HIGH_RISK).sum()), "seed": SEED},
        "models": {
            "baseline": "Latest risk prediction: final risk = risk of the latest CDM at least 2 days from TCA; high-risk if that risk >= -6.",
            "full": "HistGradientBoosting; isotonic-calibrated classifier (alert threshold set on out-of-fold training data to match the baseline's recall) + two regressors of the change from the latest risk: one over all events, one trained only on high-risk events and used for flagged events (shrunk toward the baseline by a cross-validated weight).",
            "live": "Same model families restricted to the inputs our live screener produces (see feature_mapping).",
        },
        "thresholds": {"full": full["threshold"], "live": live["threshold"], "full_f2": full["threshold_f2"], "live_f2": live["threshold_f2"],
                       "rule": "primary threshold = highest value that still catches at least as many high-risk events as the baseline does (out-of-fold training recall)"},
        "hyperparameters": {"full": full["hyper"], "live": live["hyper"]},
        "n_features": {"full": len(full_names), "live": len(LIVE_FEATURES)},
        "feature_mapping": LIVE_FEATURE_MAP,
        "results": results,
        "intervals": intervals,
        "calibration": calib,
        "importance": imp,
        "published": PUBLISHED,
        "model_files_bytes": sizes,
        "train_seconds": round(time.time() - t0, 1),
    }
    metrics["verdict"] = verdict(metrics)
    METRICS_PATH.write_text(json.dumps(_clean(metrics), indent=2), encoding="utf-8")
    print(f"saved models + metrics to {MODEL_DIR} ({sum(sizes.values())/1e6:.2f} MB of models, {time.time()-t0:.0f}s)", flush=True)
    return metrics


def _clean(o):
    """Recursively turn NaN/Infinity (invalid JSON) into None and numpy scalars into Python numbers."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def verdict(m: dict) -> dict:
    """Plain statements on whether each model beats the baseline, based on the bootstrap intervals."""
    out = {}
    for ds in ("holdout", "official_test"):
        rs, iv = m["results"][ds], m["intervals"][ds]["vs_baseline"]
        out[ds] = {}
        for model in ("full", "live"):
            lines = []
            for key, label, lower_better in (("official_score", "Kelvins score L", True), ("brier", "Brier score", True), ("f2", "F2", False), ("mse_high_risk", "MSE on high-risk events", True)):
                d = iv[model][key]
                a, b = rs[model][key], rs["baseline"][key]
                if d["lo"] is None:
                    lines.append(f"{label}: not enough data for a confidence interval.")
                    continue
                better = (d["hi"] < 0) if lower_better else (d["lo"] > 0)
                worse = (d["lo"] > 0) if lower_better else (d["hi"] < 0)
                status = "better than the baseline" if better else "worse than the baseline" if worse else "not distinguishable from the baseline"
                lines.append(f"{label}: {a:.3g} vs baseline {b:.3g}; {status} (95% CI of the difference {d['lo']:.3g} to {d['hi']:.3g}).")
            out[ds][model] = lines
    return out


# =========================================================================== runtime inference (no pandas)

_models: dict | None = None
_metrics: dict | None = None


def _load():
    global _models, _metrics
    if _models is not None:
        return
    import joblib

    _models = {}
    for name in ("full", "live"):
        p = MODEL_DIR / f"{name}.joblib"
        if p.exists():
            _models[name] = joblib.load(p)
    _metrics = _clean(json.loads(METRICS_PATH.read_text(encoding="utf-8"))) if METRICS_PATH.exists() else None


def available() -> bool:
    _load()
    return bool(_models and "live" in _models)


def model_metrics() -> dict | None:
    _load()
    return _metrics


def score_conjunctions(results: list[dict], now: datetime) -> None:
    """Add the live model's estimate to each /screen result (in place): ai_p_high, ai_log10_risk, ai_in_domain.

    ai_p_high is the calibrated probability that the FINAL collision risk will be >= 1e-6, as learned from ESA's
    CDM data; ai_log10_risk is the model's estimate of the final log10 risk. These are model estimates, not
    measurements. The model was trained on conjunctions 2-7 days before TCA; ai_in_domain is False outside that range.
    """
    _load()
    if not results or not _models or "live" not in _models:
        return
    m = _models["live"]
    X = np.empty((len(results), len(LIVE_FEATURES)))
    latest = np.empty(len(results))
    tte = np.empty(len(results))
    for i, c in enumerate(results):
        tca = datetime.fromisoformat(c["tca_utc"].replace("Z", "+00:00"))
        tte[i] = (tca - now).total_seconds() / 86400.0
        miss_m = c["miss_km"] * 1000.0
        sigma = max(c["sigma_km"], 1e-6)
        log_pc = max(math.log10(c["pc"]), RISK_FLOOR) if c["pc"] > 0 else RISK_FLOOR
        latest[i] = log_pc
        X[i] = [miss_m, c["vrel_kms"] * 1000.0, tte[i], log_pc, sigma, miss_m / (1000.0 * sigma), OBJECT_CODES.get(c.get("object_type", "UNKNOWN"), 3)]
    pred = predict_family(m, X, latest)
    for i, c in enumerate(results):
        c["ai_p_high"] = float(pred["p"][i])
        c["ai_log10_risk"] = float(pred["risk"][i])
        c["ai_in_domain"] = bool(MIN_TCA_DAYS <= tte[i] <= MAX_TCA_DAYS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        print("usage: python ml.py train")
