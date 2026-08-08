"""Do UMA's discarded L=1/L=2 spherical-harmonic channels carry pKa signal?

THE OBSERVATION
UMA's backbone emits node_embedding of shape (n_atoms, 9, 128). For
lmax=2 those 9 channels are 1 (L=0) + 3 (L=1) + 5 (L=2). Every feature
this repo extracts uses the L=0 scalar block only; channels 1-8 - the
dipole and quadrupole components of the local electronic environment -
are computed on every forward pass and thrown away.

WHY NORMS AND NOT RAW COMPONENTS
Measured under a random 3D rotation of the same molecule:

    L1 raw components  max|diff| 1.194e+00     <- rotates, unusable
    L1 per-channel norm max|diff| 2.036e-04     <- invariant, usable
    L2 raw components  max|diff| 4.327e+00
    L2 per-channel norm max|diff| 2.420e-05

Raw L>0 components are equivariant, not invariant - feeding them to a
tree would make the prediction depend on molecular orientation. The
per-channel L2 norm of each irrep block is invariant, which is what gets
used here.

CONTROLLED DESIGN
Both arms are built from the SAME backbone forward pass, so the only
difference is the added channels:

    baseline  : L0 global mean/max pool + L0 at the site atom
    augmented : baseline + ||L1|| and ||L2|| at the site atom

Both use [prot ; deprot ; prot-deprot], matching the production
convention. The absolute MAE will NOT equal the production 0.577 -
these are backbone features, not the energy-head hook features
production uses - so compare the two arms to each other, not to 0.577.

    python experiment_higher_L.py --n 1000
"""
import argparse
import time

import joblib
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import PandasTools
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
import lightgbm as lgb

RDLogger.DisableLog("rdApp.*")
CACHE = "feat_higherL.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--shell", type=int, default=2,
                    help="bond radius for site-local pooling")
    a = ap.parse_args()

    from umapka import PkaPredictor
    from umapka.predictor import (protonation_pair_site_tagged,
                                   _smiles_to_atoms_with_site)
    from fairchem.core.datasets.atomic_data import (AtomicData,
                                                     atomicdata_list_to_batch)

    p = PkaPredictor("models/model_core_v3.pkl")
    INNER = p._calc.predictor.tracked_modules()["model"].module

    def node_emb(atoms):
        ad = AtomicData.from_ase(atoms, task_name="omol", molecule_cell_size=120.0,
                                 r_energy=False, r_forces=False, r_stress=False,
                                 r_data_keys=["charge", "spin"])
        with torch.no_grad():
            return INNER.backbone(
                atomicdata_list_to_batch([ad]))["node_embedding"].numpy()

    def state_feats(smi, site):
        """(L0 pooled+site, ||L1|| site, ||L2|| site) for one microstate."""
        atoms, s_idx, mol_h = _smiles_to_atoms_with_site(smi, site)
        ne = node_emb(atoms)                      # (n, 9, 128)
        n = min(ne.shape[0], len(atoms))
        L0 = ne[:n, 0, :]
        L1n = np.linalg.norm(ne[:n, 1:4, :], axis=1)   # rotation-invariant
        L2n = np.linalg.norm(ne[:n, 4:9, :], axis=1)
        if s_idx >= n:
            raise ValueError("site index out of range")
        # site-local shell by topological distance, same idea as pool_local
        dm = Chem.GetDistanceMatrix(mol_h)
        shell = np.where(dm[s_idx][:n] <= a.shell)[0]
        base = np.concatenate([L0.mean(0), L0.max(0), L0[s_idx],
                               L0[shell].mean(0)])
        extra = np.concatenate([L1n[s_idx], L2n[s_idx],
                                L1n[shell].mean(0), L2n[shell].mean(0)])
        return base, extra

    print("loading labels...")
    df = PandasTools.LoadSDF("mlpka/datasets/combined_training_datasets_unique.sdf")
    pk = next(c for c in df.columns if c.lower() in ("pka", "pka_value", "value"))
    pairs, seen = [], set()
    for _, r in df.iterrows():
        m = r.get("ROMol")
        if m is None:
            continue
        try:
            v = float(r[pk]); smi = Chem.MolToSmiles(m)
        except Exception:
            continue
        if 0 < v < 14 and smi not in seen:
            seen.add(smi); pairs.append((smi, v))
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(pairs))[:a.n]
    pairs = [pairs[i] for i in idx]
    print(f"  embedding {len(pairs)} molecules (2 backbone passes each)...")

    B, E, y = [], [], []
    t0 = time.time()
    for i, (smi, v) in enumerate(pairs):
        try:
            pr, pi_, dp, di_, _k = protonation_pair_site_tagged(smi, return_kind=True)
            bp, ep = state_feats(pr, pi_)
            bd, ed = state_feats(dp, di_)
            B.append(np.concatenate([bp, bd, bp - bd]))
            E.append(np.concatenate([ep, ed, ep - ed]))
            y.append(v)
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            el = (time.time() - t0) / 60
            print(f"    [{i+1}/{len(pairs)}] {el:.1f} min "
                  f"(~{el/(i+1)*(len(pairs)-i-1):.0f} min left)", flush=True)

    B = np.ascontiguousarray(np.nan_to_num(np.array(B)), dtype=np.float64)
    E = np.ascontiguousarray(np.nan_to_num(np.array(E)), dtype=np.float64)
    y = np.array(y, dtype=float)
    joblib.dump({"base": B, "extra": E, "y": y}, CACHE, compress=0)
    print(f"\nusable {len(y)} | baseline {B.shape[1]}-dim | "
          f"+higher-L {B.shape[1]+E.shape[1]}-dim -> {CACHE}")

    kf = KFold(5, shuffle=True, random_state=42)

    def cv(X, tag):
        oof = np.zeros(len(y))
        for tr, va in kf.split(X):
            m = lgb.LGBMRegressor(n_estimators=800, num_leaves=31,
                                   learning_rate=0.05, min_child_samples=10,
                                   subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, verbose=-1,
                                   random_state=42)
            m.fit(np.ascontiguousarray(X[tr]), y[tr])
            oof[va] = m.predict(np.ascontiguousarray(X[va]))
        cal = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
        mae = float(np.abs(cal.predict(oof) - y).mean())
        print(f"  {tag:34s} {X.shape[1]:5d}-dim   OOF MAE {mae:.4f}")
        return mae

    print("\n=== 5-fold OOF, identical molecules and folds ===")
    m_base = cv(B, "L=0 only (current practice)")
    m_aug = cv(np.hstack([B, E]), "+ ||L1||, ||L2|| (higher-L)")
    print(f"\n  delta {m_aug - m_base:+.4f}  "
          f"({'BETTER' if m_aug < m_base else 'no gain'})")
    print("\n  NOTE: these are backbone features; production uses the")
    print("  energy-head hook, so neither number is comparable to 0.577.")
    print("  Only the delta between these two arms is meaningful.")


if __name__ == "__main__":
    main()
