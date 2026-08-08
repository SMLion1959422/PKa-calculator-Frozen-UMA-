"""Does explicit solvent-exposure (SASA) of the titratable atom add
information the frozen UMA embeddings lack?

THE PHYSICAL ARGUMENT
pKa depends on how well solvent stabilises the emerging charge. A lone
pair buried under bulky substituents cannot be solvated, which shifts
pKa substantially. UMA sees 3D coordinates, so this is not strictly
absent from its input - but nothing in the pipeline makes steric
shielding explicit, and vdW-radius-based accessible surface is not
obviously recoverable from a learned energy representation.

WHY THIS EXPERIMENT IS CHEAP
SASA comes from RDKit geometry, independent of UMA. So this needs ZERO
extra embedding passes: the cached 2385-dim features are reused and SASA
scalars appended. Contrast with the higher-L experiment, which required
re-embedding because it needed the backbone tensor.

WHAT IS NOT IMPLEMENTED, AND WHY
The proposal also suggested using SASA as POOLING WEIGHTS
(v_weighted = sum_i (SASA_i / sum SASA) * v_i). That is another pooling
variant, and pooling variants have been measured repeatedly here:
learned attention pooling scored 0.627 vs 0.543, and multiscale local
pooling is already in the production features. Re-weighting existing
information is not the same as adding new information. SASA as explicit
scalar features IS new information, so that is what gets tested.

Selection on OOF; Novartis scored once at the end.
"""
import argparse
import time

import joblib
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFreeSASA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
CACHE = "feat_sasa.pkl"
N_SASA = 8   # per state


def state_sasa(smi, site):
    """8 solvent-exposure descriptors for one protonation microstate."""
    from umapka.predictor import _smiles_to_atoms_with_site
    _atoms, s_idx, mol_h = _smiles_to_atoms_with_site(smi, site)
    radii = rdFreeSASA.classifyAtoms(mol_h)
    total = rdFreeSASA.CalcSASA(mol_h, radii)
    per = np.array([float(a.GetProp("SASA")) if a.HasProp("SASA") else 0.0
                    for a in mol_h.GetAtoms()])
    n = mol_h.GetNumAtoms()
    if s_idx >= n:
        raise ValueError("site out of range")
    dm = Chem.GetDistanceMatrix(mol_h)
    sh1 = np.where(dm[s_idx] <= 1)[0]
    sh2 = np.where(dm[s_idx] <= 2)[0]
    sh3 = np.where(dm[s_idx] <= 3)[0]
    return np.array([
        per[s_idx],                                  # absolute exposure of the site
        per[s_idx] / max(total, 1e-6),               # fraction of molecular surface
        per[sh1].mean(), per[sh2].mean(), per[sh3].mean(),   # shielding by neighbours
        total,                                        # molecular size proxy
        float(len(sh3)),                              # local crowding
        per[s_idx] / max(per[sh2].mean(), 1e-6),      # site exposure vs its shell
    ], dtype=float)


def sasa_pair(smi):
    from umapka.predictor import protonation_pair_site_tagged
    pr, pi_, dp, di_, _k = protonation_pair_site_tagged(smi, return_kind=True)
    a, b = state_sasa(pr, pi_), state_sasa(dp, di_)
    return np.concatenate([a, b, a - b])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "fit"], required=True)
    args = ap.parse_args()

    if args.stage == "build":
        from train_v20_ensemble import assemble
        import pandas as pd
        from rdkit.Chem import PandasTools
        from umapka.predictor import neutralize, ACID_SITES, BASE_SITES
        from train_v20_ensemble import priority_atom

        # reproduce assemble()'s row order so SASA rows line up with X
        E = joblib.load("feat_electronic.pkl")
        U = joblib.load("feat_train_v6.pkl")
        C = joblib.load("feat_marvin_corrected.pkl")
        valid = {s for s, v in U.items() if np.asarray(v).shape == (2304,)}
        smis, seen = [], set()
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
            seen.add(smi); smis.append(smi)

        print(f"computing SASA for {len(smis)} training molecules...")
        t0 = time.time(); rows = {}
        for i, s in enumerate(smis):
            try:
                rows[s] = sasa_pair(s)
            except Exception:
                rows[s] = np.full(N_SASA * 3, np.nan)
            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(smis)}] {(time.time()-t0)/60:.1f} min", flush=True)

        ext = joblib.load("feat_external_learned.pkl")
        ext_rows = {}
        for ds, rs in ext.items():
            print(f"computing SASA for {ds} ({len(rs)})...")
            for r in rs:
                try:
                    ext_rows[r["smiles"]] = sasa_pair(r["smiles"])
                except Exception:
                    ext_rows[r["smiles"]] = np.full(N_SASA * 3, np.nan)
        joblib.dump({"train_order": smis, "train": rows, "ext": ext_rows}, CACHE)
        print(f"saved -> {CACHE}   (now: --stage fit)")
        return

    # ---------------- fit ----------------
    from train_v20_ensemble import assemble, mk_lgb
    from umapka import electronic
    d = joblib.load(CACHE)
    X, y = assemble()
    S = np.array([d["train"][s] for s in d["train_order"]], dtype=float)
    assert len(S) == len(y), f"row misalignment {len(S)} vs {len(y)}"
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"{X.shape[0]} molecules | base {X.shape[1]}-dim | +SASA {X.shape[1]+S.shape[1]}-dim")

    kf = KFold(5, shuffle=True, random_state=42)
    seeds = [42, 7, 2024]

    def oof(Xa):
        Xa = np.ascontiguousarray(Xa)
        preds = []
        for s in seeds:
            o = np.zeros(len(y))
            for tr, va in kf.split(Xa):
                m = mk_lgb(s); m.fit(Xa[tr], y[tr]); o[va] = m.predict(Xa[va])
            preds.append(o)
        r = np.zeros(len(y))
        for tr, va in kf.split(Xa):
            sc = StandardScaler().fit(Xa[tr])
            rr = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(Xa[tr]), y[tr])
            r[va] = rr.predict(sc.transform(Xa[va]))
        blend = 0.85 * np.mean(preds, axis=0) + 0.15 * r      # v20's winning config
        cal = IsotonicRegression(out_of_bounds="clip").fit(blend, y)
        return np.abs(cal.predict(blend) - y)

    print("\n=== 5-fold OOF, identical folds ===")
    e_base = oof(X); e_aug = oof(np.hstack([X, S]))
    print(f"  v20 baseline        {e_base.mean():.4f}")
    print(f"  v20 + SASA          {e_aug.mean():.4f}   ({e_aug.mean()-e_base.mean():+.4f})")
    dd = e_base - e_aug
    rng = np.random.default_rng(0)
    bs = np.array([dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(3000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  paired delta {dd.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"  -> {'REAL' if lo > 0 else 'noise (CI spans 0)'}")
    if lo <= 0:
        print("\n  OOF gain not distinguishable from noise - not scoring Novartis.")
        print("  (Scoring the held-out set on a null OOF result would only invite")
        print("   reading noise as signal; four prior OOF gains failed to transfer.)")
        return

    print("\nrefitting on all data and scoring Novartis ONCE...")
    Xa = np.ascontiguousarray(np.hstack([X, S]))
    fitted = [mk_lgb(s).fit(Xa, y) for s in seeds]
    sc = StandardScaler().fit(Xa)
    rr = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(Xa), y)
    blend = 0.85 * np.mean([m.predict(Xa) for m in fitted], axis=0) + \
            0.15 * rr.predict(sc.transform(Xa))
    cal = IsotonicRegression(out_of_bounds="clip").fit(blend, y)
    ext = joblib.load("feat_external_learned.pkl")
    for ds, rs in ext.items():
        Xe = np.vstack([np.asarray(r["feat"], dtype=np.float64).reshape(1, -1) for r in rs])
        Se = np.nan_to_num(np.array([d["ext"][r["smiles"]] for r in rs], dtype=float))
        Xe = np.ascontiguousarray(np.hstack([Xe, Se]))
        ye = np.array([r["exp"] for r in rs])
        pb = 0.85 * np.mean([m.predict(Xe) for m in fitted], axis=0) + \
             0.15 * rr.predict(sc.transform(Xe))
        print(f"  {ds:12s} MAE {np.abs(cal.predict(pb)-ye).mean():.3f}")
    print("  v20 baseline: novartis 0.918 | avlilumove 0.411")


if __name__ == "__main__":
    main()
