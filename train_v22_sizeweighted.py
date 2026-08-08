"""v22: v20's ensemble, retrained with inverse-propensity SIZE WEIGHTS.

THE MOTIVATION
The user's deployment target is Novartis-like chemistry: pharma-scale,
novel-scaffold, large molecules. But the training pool is badly
mismatched on exactly the axis where the model is worst:

    training pool  :  7.0% of molecules > 30 heavy atoms
    Novartis       : 22.5%
    v20 error, >30 : 1.065 MAE  (vs 0.820 for 15-22 atoms)

So the bucket that dominates real-world error is under-represented 3x in
training. Inverse-propensity weighting corrects that without adding a
single molecule: re-weight the existing examples so the EFFECTIVE
training distribution matches the target's size profile.

WHY THIS IS NOT TEST-SET PEEKING
The weights use only the *unlabeled* size histogram of the target domain
- a coarse, structural property available for any molecule you intend to
predict, with no pKa labels involved. That is standard covariate-shift
correction. The weighting scheme is fixed a priori here, and Novartis is
scored ONCE at the end. It is NOT tuned against Novartis; doing that on
275 molecules would be fitting noise.

WHY NOT JUST ADD MORE LARGE MOLECULES
Tried, and measured: v21 added 6216 molecules that were 3x richer in
large molecules (21.6% vs 7.0% >30), and every Novartis size bucket got
WORSE - the >30 bucket most of all (1.065 -> 1.371). Those molecules are
size-matched to Novartis but scaffold-matched to AvLiLuMoVe (median NN
Tanimoto 0.643 vs 0.330). Re-weighting what we already have avoids
importing that distribution shift.

Reuses cached features - no UMA compute.
"""
import joblib
import numpy as np
from rdkit import Chem, RDLogger
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from train_v20_ensemble import mk_lgb, assemble

RDLogger.DisableLog("rdApp.*")

SIZE_BINS = [0, 15, 22, 30, 10_000]
SIZE_LABELS = ["<15", "15-22", "22-30", ">30"]
OUT = "models/model_core_v22_sizeweighted.pkl"
EXT_CACHE = "feat_external_learned.pkl"


def size_bin(n):
    for i in range(len(SIZE_BINS) - 1):
        if SIZE_BINS[i] < n <= SIZE_BINS[i + 1]:
            return i
    return len(SIZE_LABELS) - 1


def target_profile():
    """Unlabeled size histogram of the deployment target (Novartis).
    Only molecular structure is read - no pKa values."""
    counts = np.zeros(len(SIZE_LABELS))
    for m in Chem.ForwardSDMolSupplier(
            "mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf"):
        if m is None:
            continue
        counts[size_bin(m.GetNumHeavyAtoms())] += 1
    return counts / counts.sum()


def train_sizes():
    """Heavy-atom counts aligned with assemble()'s row order."""
    from umapka.predictor import neutralize
    from train_v20_ensemble import priority_atom
    from umapka.predictor import ACID_SITES, BASE_SITES
    E = joblib.load("feat_electronic.pkl")
    U = joblib.load("feat_train_v6.pkl")
    C = joblib.load("feat_marvin_corrected.pkl")
    valid = {s for s, v in U.items() if np.asarray(v).shape == (2304,)}
    sizes, seen = [], set()
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
        ok = (pidx is not None and pidx == ma and smi in valid) or (smi in C)
        if not ok or smi in seen:
            continue
        seen.add(smi)
        sizes.append(mol.GetNumHeavyAtoms())
    return np.array(sizes)


def main():
    print("assembling (cached features, no UMA)...")
    X, y = assemble()
    sizes = train_sizes()
    assert len(sizes) == len(y), f"row misalignment: {len(sizes)} vs {len(y)}"
    print(f"  {X.shape[0]} molecules x {X.shape[1]} features")

    tgt = target_profile()
    bins = np.array([size_bin(n) for n in sizes])
    src = np.bincount(bins, minlength=len(SIZE_LABELS)) / len(bins)

    print(f"\n{'bin':8s} {'train':>8s} {'target':>8s} {'weight':>8s}")
    w_bin = np.zeros(len(SIZE_LABELS))
    for i, lab in enumerate(SIZE_LABELS):
        w_bin[i] = (tgt[i] / src[i]) if src[i] > 0 else 0.0
        print(f"{lab:8s} {src[i]*100:7.1f}% {tgt[i]*100:7.1f}% {w_bin[i]:8.2f}")
    w = w_bin[bins]
    w = w / w.mean()          # keep the effective sample size comparable

    kf = KFold(5, shuffle=True, random_state=42)
    seeds = [42, 7, 2024]

    def oof_lgb(seed):
        pred = np.zeros(len(y))
        for tr, va in kf.split(X):
            m = mk_lgb(seed)
            m.fit(np.ascontiguousarray(X[tr]), y[tr], sample_weight=w[tr])
            pred[va] = m.predict(np.ascontiguousarray(X[va]))
        return pred

    print("\nOOF (weighted)...")
    lgbs = []
    for s in seeds:
        pr = oof_lgb(s); lgbs.append(pr)
        print(f"  lgb {s}: {np.abs(pr-y).mean():.4f}")
    ridge_oof = np.zeros(len(y))
    for tr, va in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        r = RidgeCV(alphas=np.logspace(-2, 4, 25))
        r.fit(sc.transform(X[tr]), y[tr], sample_weight=w[tr])
        ridge_oof[va] = r.predict(sc.transform(X[va]))
    print(f"  ridge : {np.abs(ridge_oof-y).mean():.4f}")

    # v20's OOF-selected winning blend, kept fixed so the ONLY change is weighting
    blend = 0.85 * np.mean(lgbs, axis=0) + 0.15 * ridge_oof
    cal = IsotonicRegression(out_of_bounds="clip").fit(blend, y)
    print(f"\nOOF calibrated (weighted): {np.abs(cal.predict(blend)-y).mean():.4f}"
          f"   [v20 unweighted was 0.4647]")

    fitted = [mk_lgb(s).fit(X, y, sample_weight=w) for s in seeds]
    sc = StandardScaler().fit(X)
    fr = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(X), y, sample_weight=w)
    bundle = {"kind": "v20_ensemble", "members": [], "n_lgb": len(fitted),
              "lgb": fitted, "hgb": [], "ridge": fr, "scaler": sc,
              "w_ridge": 0.15, "calibrator": cal, "config": "lgb_bag3+ridge/sizeweighted",
              "n_train": int(X.shape[0]), "feature_dim": int(X.shape[1])}
    joblib.dump(bundle, OUT)
    print(f"saved -> {OUT}")

    from umapka import electronic
    import pandas as pd
    ext = joblib.load(EXT_CACHE)
    print("\n=== HELD-OUT (scored once) ===")
    for ds, rows in ext.items():
        Xe = np.vstack([np.asarray(r["feat"], dtype=np.float64).reshape(1, -1)
                        for r in rows])
        ye = np.array([r["exp"] for r in rows], dtype=float)
        na = np.array([r["n_atoms"] for r in rows])
        err = np.abs(electronic.score_any_batch(bundle, Xe) - ye)
        print(f"  {ds:12s} n={len(ye):4d}  MAE {err.mean():.3f}")
        if ds == "novartis":
            d = pd.DataFrame({"err": err,
                              "size": pd.cut(na, SIZE_BINS, labels=SIZE_LABELS)})
            print(d.groupby("size", observed=True)["err"]
                   .agg(["mean", "count"]).round(3).to_string())
    print("\n  v20 baseline: novartis 0.949 | avlilumove 0.411")
    print("  v20 by size : <15 0.854 | 15-22 0.820 | 22-30 0.999 | >30 1.065")


if __name__ == "__main__":
    main()
