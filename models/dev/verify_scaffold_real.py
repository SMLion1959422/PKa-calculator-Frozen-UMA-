import os, numpy as np, pandas as pd, joblib, lightgbm as lgb
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import PandasTools, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
RDLogger.DisableLog("rdApp.*")

SEED = 42

# --- 1. get the public dataset your README already cites ---
if not os.path.isdir("mlpka"):
    os.system("git clone --depth 1 https://github.com/czodrowskilab/Machine-learning-meets-pKa.git mlpka")

# --- 2. rebuild 'core' exactly as the notebook does ---
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

# --- 3. load your real precomputed UMA features ---
f_train = joblib.load("feat_train.pkl")

# --- 4. exact scaffold-split logic from the notebook (cell 55) ---
def mk(): return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                    num_leaves=31, verbose=-1, random_state=SEED)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

tab = core[core.smiles.isin(f_train.keys())].reset_index(drop=True)
feats = f_train
tab = tab[[feats.get(s) is not None for s in tab.smiles]].reset_index(drop=True)
X = np.vstack([feats[s] for s in tab.smiles])
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
print(f"REPRODUCED scaffold-split MAE = {mae:.3f}")
print(f"REPRODUCED scaffold-split R^2 = {r2:.3f}")
print(f"REPRODUCED scaffold-split r   = {r:.3f}")
print(f"\nREADME claims: MAE = 0.994")
