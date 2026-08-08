"""Score any number of candidate heads against the cached external
features (feat_external_learned.pkl, built by cache_external_features.py
with the LEARNED site finder). No UMA passes - runs in seconds, so
model/ensemble/calibration variants can actually be compared instead of
each costing a 20-minute embedding pass.

Reported numbers are the real-world configuration: self-contained site
finding, no ChemAxon annotations at inference.

Novartis is the number that matters. check_test_novelty.py measured
median nearest-neighbour Tanimoto to training at 0.365 for Novartis
(0.7% of it above 0.9) versus 0.708 for AvLiLuMoVe (27.6% above 0.9) -
so Novartis is a genuine extrapolation test while AvLiLuMoVe is largely
interpolation. Do not chase the combined average; it flatters
whichever model memorized AvLiLuMoVe best.
"""
import sys

import joblib
import numpy as np
import pandas as pd

from umapka import electronic

CACHE = "feat_external_learned.pkl"


def load():
    cache = joblib.load(CACHE)
    out = {}
    for ds, rows in cache.items():
        X = np.vstack([np.asarray(r["feat"], dtype=np.float64).reshape(1, -1)
                       for r in rows])
        y = np.array([r["exp"] for r in rows], dtype=float)
        meta = pd.DataFrame([{k: r[k] for k in ("smiles", "kind", "kind_ok", "n_atoms")}
                             for r in rows])
        out[ds] = (X, y, meta)
    return out


def score_bundle(bundle, X):
    """Score a matrix with either bundle format: the v16/v17 hybrid
    (gbm+ridge blend) or the v20 multi-member ensemble."""
    if electronic.is_hybrid_bundle(bundle):
        return electronic.score_hybrid_batch(bundle, X)
    if bundle.get("kind") == "v20_ensemble":
        from train_v20_ensemble import predict_bundle
        return predict_bundle(bundle, X)
    raise ValueError(f"unrecognized bundle: keys={sorted(bundle)[:6]}")


def report(name, preds_by_ds, data):
    rows = []
    for ds, (X, y, meta) in data.items():
        err = np.abs(preds_by_ds[ds] - y)
        rows.append({"model": name, "dataset": ds, "n": len(y),
                     "MAE": round(float(err.mean()), 3),
                     "RMSE": round(float(np.sqrt((err ** 2).mean())), 3)})
    return rows


def main(paths):
    data = load()
    for ds, (X, y, meta) in data.items():
        print(f"{ds:12s} n={len(y):4d}  feat_dim={X.shape[1]}  "
              f"kind agreement vs Marvin: {meta.kind_ok.mean()*100:.1f}%")
    print()

    all_rows = []
    preds_cache = {}
    for path in paths:
        try:
            b = joblib.load(path)
        except Exception as exc:
            print(f"  skip {path}: {exc}")
            continue
        name = path.split("/")[-1].replace(".pkl", "")
        try:
            pb = {ds: score_bundle(b, X) for ds, (X, y, m) in data.items()}
        except Exception as exc:
            print(f"  skip {name}: {type(exc).__name__}: {exc}")
            continue
        preds_cache[name] = pb
        all_rows += report(name, pb, data)

    # equal-weight ensemble of everything that scored successfully
    if len(preds_cache) > 1:
        ens = {ds: np.mean([pb[ds] for pb in preds_cache.values()], axis=0)
               for ds in data}
        all_rows += report(f"ENSEMBLE({len(preds_cache)})", ens, data)

    df = pd.DataFrame(all_rows)
    piv = df.pivot(index="model", columns="dataset", values="MAE")
    print("=== MAE by dataset (learned sites, real-world config) ===")
    print(piv.to_string())
    print("\n  target: novartis <= 0.70   |   ChemAxon Marvin = 0.856   "
          "|  v16+oracle sites = 0.845")

    if "novartis" in piv.columns:
        best = piv["novartis"].idxmin()
        print(f"\nBEST on novartis: {best} at {piv.loc[best,'novartis']:.3f}")
        # the breakdown needs per-molecule metadata, which the synthetic
        # ENSEMBLE row has no single-model entry for - fall back to the
        # best real model so the diagnostic never silently disappears
        if best not in preds_cache:
            single = piv.drop(index=[i for i in piv.index if i.startswith("ENSEMBLE")],
                              errors="ignore")
            if len(single):
                best = single["novartis"].idxmin()
                print(f"  (breakdown below uses best single model: {best})")

    # size breakdown for the best model - where does the error still live?
    if "novartis" in piv.columns and best in preds_cache:
        X, y, meta = data["novartis"]
        e = np.abs(preds_cache[best]["novartis"] - y)
        m = meta.assign(err=e)
        m["size_bin"] = pd.cut(m.n_atoms, bins=[0, 15, 22, 30, 1000],
                               labels=["<15", "15-22", "22-30", ">30"])
        print(f"\n=== {best}: novartis error by molecule size ===")
        print(m.groupby("size_bin", observed=True)["err"].agg(["mean", "count"]).round(3))
        print(f"\n=== {best}: novartis error by kind agreement ===")
        print(m.groupby("kind_ok")["err"].agg(["mean", "count"]).round(3))


if __name__ == "__main__":
    args = sys.argv[1:] or [
        "models/model_core_v16_elec.pkl",
        "models/model_core_v17_best.pkl",
    ]
    main(args)
