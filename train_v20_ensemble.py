"""v20: multi-model, multi-seed ensemble head on the v16 feature layout,
aiming to push Novartis below 0.8.

WHERE THE HEADROOM IS
eval_cached.py decomposed the Novartis error for v16 + learned sites:
    kind wrong (14 mols) MAE 3.901   -> fixed by models/kind_classifier.pkl
    kind right (261)     MAE 0.846   <- THIS is the remaining wall
0.846 is also the oracle-site number, so better site/kind selection
cannot get past it. Only a better regressor can.

WHY AN ENSEMBLE AND NOT A NEW ARCHITECTURE
dev/compare_regressor_heads.py measured, on identical folds:
    LightGBM              0.577
    HistGradientBoosting  0.591
    MLP                   0.596
    equal-weight average  0.543   (-6% vs the best single model)
Averaging decorrelated errors is the one gain that showed up reliably in
this small-data regime, where a bigger single model did not. ablation_xtb
separately found UMA + Gasteiger/EState beats UMA alone (0.574 -> 0.521),
which is the feature set used here. Multi-seed bagging is added on top
because tree ensembles at this data size carry real seed variance.

METHODOLOGY - READ THIS
Configurations are selected on 5-fold OOF CV over the TRAINING set only.
Novartis/AvLiLuMoVe are scored ONCE, for the OOF-selected winner. Picking
a config by its Novartis score would silently turn the held-out set into
a validation set and the reported number into a fiction. The per-config
Novartis column is printed for transparency but is NOT used to choose.

Features come entirely from existing caches - no UMA compute:
  feat_train_v6.pkl        2304-dim UMA (global + multiscale local)
  feat_electronic.pkl        81-dim Gasteiger/EState (both states + diff)
  feat_marvin_corrected.pkl  fallback features for molecules where the
                             SMARTS priority site disagreed with Marvin
Total 2385, matching models/model_core_v16_elec.pkl exactly.
"""
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

RDLogger.DisableLog("rdApp.*")

EXT_CACHE = "feat_external_learned.pkl"
OUT = "models/model_core_v20_ensemble.pkl"


def priority_atom(mol, acid_sites, base_sites):
    for _n, sm, ai in acid_sites:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m:
                return m[0][ai]
    for _n, sm, ai in base_sites:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m:
                return m[0][ai]
    return None


def assemble():
    """Rebuild v16's training matrix from caches. Mirrors ablation_xtb.py's
    assembly, including the marvin-corrected fallback, so the feature
    semantics match what model_core_v16_elec was trained on."""
    from umapka.predictor import ACID_SITES, BASE_SITES, neutralize
    U = joblib.load("feat_train_v6.pkl")
    E = joblib.load("feat_electronic.pkl")
    C = joblib.load("feat_marvin_corrected.pkl")
    valid = {s for s, v in U.items() if np.asarray(v).shape == (2304,)}

    rows = []
    for mol in Chem.ForwardSDMolSupplier(
            "mlpka/datasets/combined_training_datasets_unique.sdf"):
        if mol is None or not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")):
            continue
        try:
            exp = float(mol.GetProp("pKa"))
            ma = int(float(mol.GetProp("marvin_atom")))
            smi = Chem.MolToSmiles(mol)
            nm = neutralize(Chem.Mol(mol))
        except Exception:
            continue
        if not (0 < exp < 14) or ma >= nm.GetNumAtoms() or smi not in E:
            continue
        pidx = priority_atom(nm, ACID_SITES, BASE_SITES)
        vec = None
        if pidx is not None and pidx == ma and smi in valid:
            vec = U[smi]
        elif smi in C:
            vec = C[smi]["feat"]
            exp = C[smi]["pKa"]
        if vec is None:
            continue
        rows.append({"smiles": smi, "pKa": exp,
                     "x": np.concatenate([np.asarray(vec).ravel(),
                                          np.asarray(E[smi]).ravel()])})
    df = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
    X = np.ascontiguousarray(
        np.nan_to_num(np.vstack(df.x.values), nan=0.0, posinf=0.0, neginf=0.0),
        dtype=np.float64)
    y = np.ascontiguousarray(df.pKa.values, dtype=np.float64)
    return X, y


# --- candidate members -------------------------------------------------
def mk_lgb(seed, n=800, leaves=31, lr=0.05, mcs=10):
    """n_estimators/num_leaves/min_child_samples are the winners from
    hparam_search_results_fast.csv (800/31/10, reg_lambda 0), so there is
    no headroom left in those.

    subsample/colsample are NOT tuning - they are what makes seeding do
    anything at all. With the defaults (subsample=1.0,
    colsample_bytree=1.0) LightGBM is fully deterministic, so a
    "3-seed bag" produced three byte-identical models: seeds 42, 7 and
    2024 all scored OOF MAE 0.4961 and lgb_bag3 was numerically equal to
    lgb_single. Row and column subsampling restore the per-seed variation
    that averaging is supposed to exploit. subsample_freq must be >=1 or
    LightGBM ignores subsample entirely.
    """
    return lgb.LGBMRegressor(n_estimators=n, num_leaves=leaves, learning_rate=lr,
                              min_child_samples=mcs, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8,
                              verbose=-1, random_state=seed)


def mk_hgb(seed):
    # max_features<1 gives HistGB per-seed variation too; without it the
    # fit is deterministic and extra seeds add nothing (same trap as above)
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                          max_features=0.8, random_state=seed)


def oof_predict(make, X, y, kf, scale=False):
    pred = np.zeros(len(y))
    for tr, va in kf.split(X):
        Xtr, Xva = X[tr], X[va]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
        m = make()
        m.fit(np.ascontiguousarray(Xtr), y[tr])
        pred[va] = m.predict(np.ascontiguousarray(Xva))
    return pred


def main():
    print("assembling training matrix from caches...")
    X, y = assemble()
    print(f"  {X.shape[0]} molecules x {X.shape[1]} features")

    kf = KFold(5, shuffle=True, random_state=42)
    seeds = [42, 7, 2024]

    print("\nbuilding OOF predictions for each member...")
    members = {}
    for s in seeds:
        members[f"lgb{s}"] = oof_predict(lambda s=s: mk_lgb(s), X, y, kf)
        print(f"  lgb seed {s}: MAE {np.abs(members[f'lgb{s}']-y).mean():.4f}")
    for s in seeds[:2]:
        members[f"hgb{s}"] = oof_predict(lambda s=s: mk_hgb(s), X, y, kf)
        print(f"  hgb seed {s}: MAE {np.abs(members[f'hgb{s}']-y).mean():.4f}")
    members["ridge"] = oof_predict(
        lambda: RidgeCV(alphas=np.logspace(-2, 4, 25)), X, y, kf, scale=True)
    print(f"  ridge      : MAE {np.abs(members['ridge']-y).mean():.4f}")

    # --- candidate blends, all scored on OOF only ---
    lgbs = [members[f"lgb{s}"] for s in seeds]
    hgbs = [members[f"hgb{s}"] for s in seeds[:2]]
    cands = {
        "lgb_single":        members["lgb42"],
        "lgb_bag3":          np.mean(lgbs, axis=0),
        "lgb_bag3+hgb2":     np.mean(lgbs + hgbs, axis=0),
        "lgb_bag3+ridge":    0.85 * np.mean(lgbs, axis=0) + 0.15 * members["ridge"],
        "all_equal":         np.mean(lgbs + hgbs + [members["ridge"]], axis=0),
        "trees85_ridge15":   0.85 * np.mean(lgbs + hgbs, axis=0) + 0.15 * members["ridge"],
    }

    print("\n=== OOF (selection happens HERE, on training data only) ===")
    scored = []
    for name, pr in cands.items():
        cal = IsotonicRegression(out_of_bounds="clip").fit(pr, y)
        mae = float(np.abs(cal.predict(pr) - y).mean())
        raw = float(np.abs(pr - y).mean())
        scored.append((mae, raw, name))
        print(f"  {name:20s} raw {raw:.4f}   calibrated {mae:.4f}")
    scored.sort()
    best_mae, _raw, best = scored[0]
    print(f"\nOOF winner: {best} at {best_mae:.4f}")

    # --- refit winner on all data ---
    print(f"\nrefitting '{best}' on all data...")
    fitted_lgb = [mk_lgb(s).fit(X, y) for s in seeds]
    fitted_hgb = [mk_hgb(s).fit(X, y) for s in seeds[:2]]
    sc = StandardScaler().fit(X)
    fitted_ridge = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(X), y)

    spec = {
        "lgb_single":      (["lgb"], 0.0),
        "lgb_bag3":        (["lgb"], 0.0),
        "lgb_bag3+hgb2":   (["lgb", "hgb"], 0.0),
        "lgb_bag3+ridge":  (["lgb"], 0.15),
        "all_equal":       (["lgb", "hgb"], 1.0 / 6),
        "trees85_ridge15": (["lgb", "hgb"], 0.15),
    }[best]
    use, w_ridge = spec
    n_lgb = 1 if best == "lgb_single" else len(fitted_lgb)

    cal = IsotonicRegression(out_of_bounds="clip").fit(cands[best], y)
    bundle = {
        "kind": "v20_ensemble", "members": use, "n_lgb": n_lgb,
        "lgb": fitted_lgb[:n_lgb],
        "hgb": fitted_hgb if "hgb" in use else [],
        "ridge": fitted_ridge, "scaler": sc, "w_ridge": w_ridge,
        "calibrator": cal, "oof_mae": best_mae, "config": best,
        "feature_dim": X.shape[1],
    }
    joblib.dump(bundle, OUT)
    print(f"saved -> {OUT}")

    # --- score the winner on the external sets, ONCE ---
    try:
        ext = joblib.load(EXT_CACHE)
    except Exception as exc:
        print(f"\n{EXT_CACHE} unavailable ({exc}) - run cache_external_features.py")
        return
    print("\n=== HELD-OUT (scored once, for the OOF-selected winner) ===")
    for ds, rows in ext.items():
        Xe = np.ascontiguousarray(
            np.vstack([np.asarray(r["feat"], dtype=np.float64).reshape(1, -1)
                       for r in rows]), dtype=np.float64)
        ye = np.array([r["exp"] for r in rows], dtype=float)
        pred = predict_bundle(bundle, Xe)
        print(f"  {ds:12s} n={len(ye):4d}  MAE {np.abs(pred-ye).mean():.3f}")
    print("\n  reference: v16 = 1.001 novartis (learned sites, old kind rule)")
    print("             ChemAxon Marvin = 0.856   target = 0.800")


def predict_bundle(b, X):
    """Kept for backwards compatibility; the implementation now
    lives in umapka.electronic so inference never imports a
    training module."""
    from umapka import electronic
    return electronic.score_ensemble_batch(b, X)


if __name__ == "__main__":
    main()
