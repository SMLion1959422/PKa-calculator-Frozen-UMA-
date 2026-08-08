"""Adapted from your own models/dev/verify_scaffold_real.py, pointed at
feat_train_v3.pkl (1536-dim global+site-local) instead of feat_train.pkl
(768-dim global-only). Zero new UMA compute - just one LightGBM fit on
already-cached vectors, so this is the fastest possible sanity check.

What it checks: a scaffold-based train/test split WITHIN your own
training data (no molecule's Murcko scaffold appears in both train and
test). If a model can only do well on random splits but falls apart on
scaffold splits, that's a sign it's leaning on near-duplicate molecules
rather than genuinely generalizing - which would also explain part of
the OOF-vs-external gap we've been chasing. Original (v2, 768-dim)
claimed scaffold-split MAE = 0.994; this prints the v3 equivalent for
direct comparison.
"""
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from rdkit import Chem, RDLogger
from rdkit.Chem import PandasTools
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

RDLogger.DisableLog("rdApp.*")
SEED = 42

def to_table(fn, name):
    df = PandasTools.LoadSDF(f"mlpka/datasets/{fn}")
    pk = next(c for c in df.columns if c.lower() in ("pka", "pka_value", "value"))
    rec = []
    for _, r in df.iterrows():
        m = r.get("ROMol")
        if m is None: continue
        try: v = float(r[pk])
        except Exception: continue
        if not (0 < v < 14): continue
        try: rec.append({"smiles": Chem.MolToSmiles(m), "pKa": v})
        except Exception: pass
    t = pd.DataFrame(rec).drop_duplicates("smiles").reset_index(drop=True)
    print(f"  {name}: {len(t)}")
    return t

print("rebuilding 'core' table...")
core = to_table("combined_training_datasets_unique.sdf", "core")

print("loading v3 (1536-dim) cached features...")
f_train = joblib.load("feat_train_v3.pkl")
f_train = {k: v for k, v in f_train.items() if np.asarray(v).shape == (1536,)}

def mk(): return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                    num_leaves=31, verbose=-1, random_state=SEED)

tab = core[core.smiles.isin(f_train.keys())].reset_index(drop=True)
X = np.vstack([f_train[s] for s in tab.smiles])
y = tab.pKa.values
print(f"working set: {X.shape[0]} molecules\n")

def scaffold(smi):
    try: return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
    except Exception: return ""

tab["scaf"] = [scaffold(s) for s in tab.smiles]
groups = tab.groupby("scaf").indices
print(f"unique scaffolds: {len(groups)}")

rng = np.random.default_rng(SEED)
scafs = list(groups.keys()); rng.shuffle(scafs)
test_idx, target = [], int(0.2 * len(tab))
for s in scafs:
    if len(test_idx) >= target: break
    test_idx.extend(groups[s])
test_idx = np.array(sorted(test_idx))
train_idx = np.setdiff1d(np.arange(len(tab)), test_idx)
print(f"scaffold split: train={len(train_idx)}  test={len(test_idx)}")

tr_s = set(tab.scaf.iloc[train_idx]); te_s = set(tab.scaf.iloc[test_idx])
print(f"scaffold overlap between train and test: {len(tr_s & te_s)} (should be 0)\n")

m = mk().fit(X[train_idx], y[train_idx])
p = m.predict(X[test_idx])
mae = mean_absolute_error(y[test_idx], p)
r2 = r2_score(y[test_idx], p)
r, _ = pearsonr(y[test_idx], p)
print(f"v3 scaffold-split MAE = {mae:.3f}")
print(f"v3 scaffold-split R^2 = {r2:.3f}")
print(f"v3 scaffold-split r   = {r:.3f}")
print(f"\nv2 (768-dim, original) scaffold-split MAE was: 0.994")
print(f"if v3's number here is meaningfully lower, that's a THIRD piece of")
print(f"evidence (alongside the OOF and external numbers) that the fix is")
print(f"real generalization improvement, not just fitting the eval sets")
