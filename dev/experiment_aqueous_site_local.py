"""
Real test: does site-local pooling improve the ACTUAL aqueous model's
scaffold-split MAE and Novartis external MAE - not just the anion
water subset from the last experiment?

Uses the exact public dataset and scaffold-split methodology already
in models/dev/verify_scaffold_real.py (your own reproducibility
check), plus the Novartis external set from the same mlpka clone
dev/characterize_external.py uses. Both feature variants are built
from fresh UMA embeddings computed in this run - nothing here reuses
feat_train.pkl, which only has the old global-pooled vectors.

Caveat: neither variant here includes the calibration step
model_core_v2.pkl ships with, so absolute numbers won't land exactly
on the documented 0.994 / 1.16. Compare the two arms to each other.

Run from the repo root, venv311 active. This re-embeds several
thousand molecules through UMA - expect it to take a while.

    python dev/experiment_aqueous_site_local.py
"""
import os, time
import numpy as np, pandas as pd, lightgbm as lgb
from rdkit import Chem, RDLogger
from rdkit.Chem import PandasTools
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error
RDLogger.DisableLog("rdApp.*")

from umapka import PkaPredictor
from umapka.site_features import (
    protonation_pair_with_site, smiles_to_atoms_with_site, pool_site, self_test,
)

SEED = 42

print("=== sanity check ===")
self_test()
print()

if not os.path.isdir("mlpka"):
    os.system("git clone --depth 1 https://github.com/czodrowskilab/Machine-learning-meets-pKa.git mlpka")

def to_table(fn, name):
    df = PandasTools.LoadSDF(f"mlpka/datasets/{fn}")
    pk = next(c for c in df.columns if c.lower() in ("pka", "pka_value", "value"))
    rec = []
    for _, r in df.iterrows():
        m = r.get("ROMol")
        if m is None:
            continue
        try:
            v = float(r[pk])
        except Exception:
            continue
        if not (0 < v < 14):
            continue
        try:
            rec.append({"smiles": Chem.MolToSmiles(m), "pKa": v})
        except Exception:
            pass
    t = pd.DataFrame(rec).drop_duplicates("smiles").reset_index(drop=True)
    print(f"  {name}: {len(t)}")
    return t

print("loading datasets...")
core = to_table("combined_training_datasets_unique.sdf", "core (train/test)")
novartis = to_table("novartis_cleaned_mono_unique_notraindata.sdf", "novartis (external)")

p = PkaPredictor("models/model_core.pkl")

def build_features(smiles_list, tag):
    X_global, X_both, kept_smiles = [], [], []
    t0 = time.time()
    n_skip = 0
    for i, smi in enumerate(smiles_list):
        try:
            prot, deprot = protonation_pair_with_site(smi)
            atoms_p, site_p = smiles_to_atoms_with_site(prot)
            atoms_d, site_d = smiles_to_atoms_with_site(deprot)
            if site_p is None or site_d is None:
                n_skip += 1
                continue
            emb_p = p.embeddings(atoms_p)
            emb_d = p.embeddings(atoms_d)
            h_p_g = p.pool(emb_p); h_d_g = p.pool(emb_d)
            h_p_l = pool_site(emb_p, atoms_p.get_positions(), site_p)
            h_d_l = pool_site(emb_d, atoms_d.get_positions(), site_d)
            feat_g = np.concatenate([h_p_g, h_d_g, h_p_g - h_d_g])
            h_p_b = np.concatenate([h_p_g, h_p_l]); h_d_b = np.concatenate([h_d_g, h_d_l])
            feat_b = np.concatenate([h_p_b, h_d_b, h_p_b - h_d_b])
        except Exception:
            n_skip += 1
            continue
        X_global.append(feat_g); X_both.append(feat_b)
        kept_smiles.append(smi)
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  [{tag}] [{i+1}/{len(smiles_list)}]  {el/60:.1f} min elapsed, skipped {n_skip}")
    print(f"  [{tag}] done: {len(kept_smiles)} usable, {n_skip} skipped")
    return np.array(X_global), np.array(X_both), kept_smiles

print("\nembedding core dataset (train/test)...")
core_map = dict(zip(core.smiles, core.pKa))
Xg_core, Xb_core, smiles_core = build_features(core.smiles.tolist(), "core")
y_core = np.array([core_map[s] for s in smiles_core])

print("\nembedding novartis (external)...")
nov_map = dict(zip(novartis.smiles, novartis.pKa))
Xg_nov, Xb_nov, smiles_nov = build_features(novartis.smiles.tolist(), "novartis")
y_nov = np.array([nov_map[s] for s in smiles_nov])

def scaffold(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
    except Exception:
        return ""

tab = pd.DataFrame({"smiles": smiles_core, "scaf": [scaffold(s) for s in smiles_core]})
groups = tab.groupby("scaf").indices
rng = np.random.default_rng(SEED)
scaf_keys = list(groups.keys())
rng.shuffle(scaf_keys)
test_idx, target = [], int(0.2 * len(tab))
for s in scaf_keys:
    if len(test_idx) >= target:
        break
    test_idx.extend(groups[s])
test_idx = np.array(sorted(test_idx))
train_idx = np.setdiff1d(np.arange(len(tab)), test_idx)
print(f"\nscaffold split: train={len(train_idx)}  test={len(test_idx)}")

def mk():
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                              num_leaves=31, verbose=-1, random_state=SEED)

def eval_variant(X):
    m = mk().fit(X[train_idx], y_core[train_idx])
    scaffold_mae = mean_absolute_error(y_core[test_idx], m.predict(X[test_idx]))
    m_full = mk().fit(X, y_core)   # refit on all core data, like the shipped model
    return m_full, scaffold_mae

print("\n=== scaffold-split MAE ===")
m_full_global, mae_scaf_global = eval_variant(Xg_core)
m_full_both, mae_scaf_both = eval_variant(Xb_core)
print(f"{'features':<32}{'scaffold MAE':>14}")
print(f"{'global pooling only (current)':<32}{mae_scaf_global:>14.3f}")
print(f"{'global + site-local pooling':<32}{mae_scaf_both:>14.3f}")
print("README claims (current shipped model): 0.994")

print("\n=== Novartis external MAE (model refit on ALL core data) ===")
mae_nov_global = mean_absolute_error(y_nov, m_full_global.predict(Xg_nov))
mae_nov_both = mean_absolute_error(y_nov, m_full_both.predict(Xb_nov))
print(f"{'features':<32}{'Novartis MAE':>14}")
print(f"{'global pooling only (current)':<32}{mae_nov_global:>14.3f}")
print(f"{'global + site-local pooling':<32}{mae_nov_both:>14.3f}")
print("RESULTS.md documented (model_core_v2, WITH calibration): 1.16")
print(f"\ndelta: {mae_nov_both - mae_nov_global:+.3f}  "
      f"({'better' if mae_nov_both < mae_nov_global else 'worse or no change'})")
