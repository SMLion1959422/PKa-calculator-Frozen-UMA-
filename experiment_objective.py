"""Are we optimising the wrong loss?

THE OVERSIGHT
Every LightGBM model in this repo is built with the default objective,
which is L2 (squared error). Every number we report and care about is
MAE. Squared error is dominated by the largest residuals, so it spends
capacity chasing a handful of hard molecules at the expense of the bulk
- exactly the wrong trade when the metric is the mean absolute error.

L1 ('regression_l1') optimises MAE directly. Huber and Fair are
robust compromises: quadratic near zero, linear in the tails.

This is orthogonal to everything tried so far (features, data, pooling,
architecture) - it changes what the trees are asked to minimise, not
what they see.

Screen with one seed, then run the full v20 blend for the winner and
score Novartis once.

    python experiment_objective.py --stage screen
    python experiment_objective.py --stage full --objective regression_l1
"""
import argparse

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from train_v20_ensemble import assemble

OUT = "models/model_core_v23_l1.pkl"
SEEDS = [42, 7, 2024]


def mk(seed, objective, alpha=None):
    kw = dict(n_estimators=800, num_leaves=31, learning_rate=0.05,
              min_child_samples=10, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, verbose=-1, random_state=seed)
    if objective:
        kw["objective"] = objective
    if alpha is not None:
        kw["alpha"] = alpha
    return lgb.LGBMRegressor(**kw)


def oof_single(X, y, kf, objective, alpha=None, seed=42):
    o = np.zeros(len(y))
    for tr, va in kf.split(X):
        m = mk(seed, objective, alpha)
        m.fit(np.ascontiguousarray(X[tr]), y[tr])
        o[va] = m.predict(np.ascontiguousarray(X[va]))
    cal = IsotonicRegression(out_of_bounds="clip").fit(o, y)
    return np.abs(cal.predict(o) - y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["screen", "full"], default="screen")
    ap.add_argument("--objective", default="regression_l1")
    a = ap.parse_args()

    X, y = assemble()
    print(f"{X.shape[0]} molecules x {X.shape[1]} features")
    kf = KFold(5, shuffle=True, random_state=42)

    if a.stage == "screen":
        print("\n=== objective screen (1 seed, identical folds) ===")
        cands = [(None, None, "L2 default (current)"),
                 ("regression_l1", None, "L1 / MAE"),
                 ("huber", 1.0, "Huber a=1.0"),
                 ("huber", 2.0, "Huber a=2.0"),
                 ("fair", 1.0, "Fair c=1.0")]
        base = None
        for obj, alpha, tag in cands:
            e = oof_single(X, y, kf, obj, alpha)
            if base is None:
                base = e
            d = base - e
            rng = np.random.default_rng(0)
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean()
                           for _ in range(2000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            flag = "" if tag.startswith("L2") else \
                   ("  REAL" if lo > 0 else ("  worse" if hi < 0 else "  noise"))
            print(f"  {tag:22s} OOF MAE {e.mean():.4f}   "
                  f"delta {d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}]{flag}")
        print("\n  -> rerun with --stage full --objective <winner>")
        return

    # ---- full v20 blend with the chosen objective ----
    obj = None if a.objective == "l2" else a.objective
    print(f"\nfull v20 blend, objective={a.objective}")
    preds = []
    for s in SEEDS:
        o = np.zeros(len(y))
        for tr, va in kf.split(X):
            m = mk(s, obj)
            m.fit(np.ascontiguousarray(X[tr]), y[tr])
            o[va] = m.predict(np.ascontiguousarray(X[va]))
        preds.append(o)
        print(f"  lgb {s}: {np.abs(o-y).mean():.4f}")
    r = np.zeros(len(y))
    for tr, va in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        r[va] = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(
            sc.transform(X[tr]), y[tr]).predict(sc.transform(X[va]))
    blend = 0.85 * np.mean(preds, axis=0) + 0.15 * r
    cal = IsotonicRegression(out_of_bounds="clip").fit(blend, y)
    print(f"  OOF calibrated {np.abs(cal.predict(blend)-y).mean():.4f}"
          f"   [v20 L2 was 0.4647]")

    fitted = [mk(s, obj).fit(np.ascontiguousarray(X), y) for s in SEEDS]
    sc = StandardScaler().fit(X)
    rr = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(X), y)
    bundle = {"kind": "v20_ensemble", "members": [], "n_lgb": len(fitted),
              "lgb": fitted, "hgb": [], "ridge": rr, "scaler": sc,
              "w_ridge": 0.15, "calibrator": cal,
              "config": f"lgb_bag3+ridge/{a.objective}",
              "feature_dim": int(X.shape[1])}
    joblib.dump(bundle, OUT)
    print(f"saved -> {OUT}")

    from umapka import electronic
    import pandas as pd
    ext = joblib.load("feat_external_learned.pkl")
    print("\n=== HELD-OUT (scored once) ===")
    for ds, rows in ext.items():
        Xe = np.vstack([np.asarray(rr_["feat"], dtype=np.float64).reshape(1, -1)
                        for rr_ in rows])
        ye = np.array([rr_["exp"] for rr_ in rows], dtype=float)
        na = np.array([rr_["n_atoms"] for rr_ in rows])
        err = np.abs(electronic.score_any_batch(bundle, Xe) - ye)
        print(f"  {ds:12s} n={len(ye):4d}  MAE {err.mean():.3f}")
        if ds == "novartis":
            d_ = pd.DataFrame({"err": err, "size": pd.cut(
                na, [0, 15, 22, 30, 1000], labels=["<15", "15-22", "22-30", ">30"])})
            print(d_.groupby("size", observed=True)["err"].agg(["mean", "count"])
                  .round(3).to_string())
    print("\n  v20 (L2) baseline: novartis 0.918 | avlilumove 0.411")


if __name__ == "__main__":
    main()
