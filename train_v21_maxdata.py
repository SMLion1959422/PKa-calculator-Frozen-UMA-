"""v21: the v20 ensemble retrained on the ~2.2x larger training pool
unlocked by build_electronic_extra.py.

WHY THIS AND NOT MORE MODEL TUNING
The Novartis shortfall is a GENERALIZATION gap, not a fitting gap. Two
separate OOF improvements failed to transfer:
    3-model average : OOF -6.0%  ->  Novartis -1.2%
    v20 (no bagging): OOF -3.6%  ->  Novartis WORSE (0.965 -> 0.989)
check_test_novelty.py explains it: training + AvLiLuMoVe are near
neighbours (median NN Tanimoto 0.708, 28% near-duplicates) while
Novartis is genuinely novel (0.365). Cross-validated fit on training
chemistry is therefore the wrong objective for Novartis, and the prior
hyperparameter search is already at its optimum. More DIVERSE data is
the only lever left with real headroom.

Data added: 6216 molecules from extra_pka_data.csv that already had UMA
embeddings, verified free of test contamination (0 exact overlap with
either held-out set), and filtered to those where the learned site
finder agrees with the SMARTS site the cached UMA features were built
from - see build_electronic_extra.py for why disagreements are dropped
rather than trusted.

Same honest protocol as v20: configurations are selected on 5-fold OOF
over training data only; the held-out sets are scored once, for the
winner.
"""
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from train_v20_ensemble import (mk_lgb, mk_hgb, oof_predict, predict_bundle,
                                 priority_atom, assemble)

RDLogger.DisableLog("rdApp.*")
EXT_CACHE = "feat_external_learned.pkl"
OUT = "models/model_core_v21_maxdata.pkl"


def assemble_max():
    """v16's original matrix plus the newly-unlocked extras."""
    X0, y0 = assemble()
    print(f"  base pool: {X0.shape[0]} molecules")

    U = joblib.load("feat_train_v6.pkl")
    E2 = joblib.load("feat_electronic_extra.pkl")
    extra_df = pd.read_csv("extra_pka_data.csv")
    labels = {}
    for r in extra_df.itertuples():
        try:
            v = float(r.pKa)
        except Exception:
            continue
        if 0 < v < 14:
            labels.setdefault(r.smiles, v)

    xs, ys = [], []
    for smi, e in E2.items():
        if smi not in U or smi not in labels:
            continue
        u = np.asarray(U[smi]).ravel()
        if u.shape != (2304,):
            continue
        xs.append(np.concatenate([u, np.asarray(e).ravel()]))
        ys.append(labels[smi])
    print(f"  extra pool: {len(xs)} molecules")

    X = np.vstack([X0] + ([np.vstack(xs)] if xs else []))
    y = np.concatenate([y0] + ([np.array(ys)] if ys else []))
    X = np.ascontiguousarray(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0),
                             dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    return X, y


def main():
    print("assembling enlarged training matrix...")
    X, y = assemble_max()
    print(f"  TOTAL: {X.shape[0]} molecules x {X.shape[1]} features")

    kf = KFold(5, shuffle=True, random_state=42)
    seeds = [42, 7, 2024]

    print("\nOOF per member...")
    lgbs, hgbs = [], []
    for s in seeds:
        p = oof_predict(lambda s=s: mk_lgb(s), X, y, kf)
        lgbs.append(p)
        print(f"  lgb {s}: {np.abs(p-y).mean():.4f}")
    for s in seeds[:2]:
        p = oof_predict(lambda s=s: mk_hgb(s), X, y, kf)
        hgbs.append(p)
        print(f"  hgb {s}: {np.abs(p-y).mean():.4f}")
    ridge = oof_predict(lambda: RidgeCV(alphas=np.logspace(-2, 4, 25)),
                        X, y, kf, scale=True)
    print(f"  ridge : {np.abs(ridge-y).mean():.4f}")

    cands = {
        "lgb_single":       lgbs[0],
        "lgb_bag3":         np.mean(lgbs, axis=0),
        "lgb_bag3+hgb2":    np.mean(lgbs + hgbs, axis=0),
        "lgb_bag3+ridge":   0.85 * np.mean(lgbs, axis=0) + 0.15 * ridge,
        "all_equal":        np.mean(lgbs + hgbs + [ridge], axis=0),
        "trees85_ridge15":  0.85 * np.mean(lgbs + hgbs, axis=0) + 0.15 * ridge,
    }
    print("\n=== OOF (selection on training data only) ===")
    scored = []
    for name, pr in cands.items():
        cal = IsotonicRegression(out_of_bounds="clip").fit(pr, y)
        mae = float(np.abs(cal.predict(pr) - y).mean())
        scored.append((mae, name))
        print(f"  {name:20s} raw {np.abs(pr-y).mean():.4f}   calibrated {mae:.4f}")
    scored.sort()
    best_mae, best = scored[0]
    print(f"\nOOF winner: {best} at {best_mae:.4f}")

    print(f"\nrefitting '{best}' on all {X.shape[0]} molecules...")
    fitted_lgb = [mk_lgb(s).fit(X, y) for s in seeds]
    fitted_hgb = [mk_hgb(s).fit(X, y) for s in seeds[:2]]
    sc = StandardScaler().fit(X)
    fitted_ridge = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(X), y)

    spec = {"lgb_single": ([], 0.0), "lgb_bag3": ([], 0.0),
            "lgb_bag3+hgb2": (["hgb"], 0.0), "lgb_bag3+ridge": ([], 0.15),
            "all_equal": (["hgb"], 1.0 / 6), "trees85_ridge15": (["hgb"], 0.15)}[best]
    use, w_ridge = spec
    n_lgb = 1 if best == "lgb_single" else len(fitted_lgb)

    bundle = {
        "kind": "v20_ensemble", "members": use, "n_lgb": n_lgb,
        "lgb": fitted_lgb[:n_lgb],
        "hgb": fitted_hgb if "hgb" in use else [],
        "ridge": fitted_ridge, "scaler": sc, "w_ridge": w_ridge,
        "calibrator": IsotonicRegression(out_of_bounds="clip").fit(cands[best], y),
        "oof_mae": best_mae, "config": best, "n_train": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
    }
    joblib.dump(bundle, OUT)
    print(f"saved -> {OUT}")

    ext = joblib.load(EXT_CACHE)
    print("\n=== HELD-OUT (scored once, OOF-selected winner) ===")
    for ds, rows in ext.items():
        Xe = np.vstack([np.asarray(r["feat"], dtype=np.float64).reshape(1, -1)
                        for r in rows])
        ye = np.array([r["exp"] for r in rows], dtype=float)
        pred = predict_bundle(bundle, Xe)
        print(f"  {ds:12s} n={len(ye):4d}  MAE {np.abs(pred-ye).mean():.3f}")
    print("\n  v20 (5184 mols): novartis 0.949 | avlilumove 0.411")
    print("  ChemAxon Marvin: novartis 0.856        target: 0.800")


if __name__ == "__main__":
    main()
