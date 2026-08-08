"""Train on ONLY the molecules whose cached embedding is already at the
Marvin ground-truth site (SMARTS agreed), then evaluate with Marvin
sites. Train and test then use the SAME site convention - fixing the
mismatch that v9 exposed, with ZERO re-embedding.

Trade-off: drops the ~7% of training molecules where SMARTS disagreed,
plus the extra_pka_data molecules (chembl26 etc.) which carry no
marvin_atom field to verify against. Fewer molecules, but a consistent
site convention throughout.
"""
import sys
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka.predictor import ACID_SITES, BASE_SITES, neutralize

def priority_atom(mol):
    for name, sm, ai in ACID_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m:
                return m[0][ai]
    for name, sm, ai in BASE_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m:
                return m[0][ai]
    return None

print("loading v6 features...")
f = joblib.load("feat_train_v6.pkl")
valid = {s for s, v in f.items() if np.asarray(v).shape == (2304,)}
print(f"  {len(valid)} valid entries")

print("scanning training SDF for marvin_atom agreement...")
rows = []
n_total = n_agree = 0
for mol in Chem.ForwardSDMolSupplier(
        "mlpka/datasets/combined_training_datasets_unique.sdf"):
    if mol is None or not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")):
        continue
    try:
        exp = float(mol.GetProp("pKa"))
        ma = int(float(mol.GetProp("marvin_atom")))
    except Exception:
        continue
    if not (0 < exp < 14):
        continue
    try:
        smi = Chem.MolToSmiles(mol)
        nm = neutralize(Chem.Mol(mol))
        pidx = priority_atom(nm)
    except Exception:
        continue
    n_total += 1
    if pidx is not None and pidx == ma and smi in valid:
        n_agree += 1
        rows.append({"smiles": smi, "pKa": exp})

core = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
print(f"  {n_total} molecules with marvin_atom")
print(f"  {n_agree} where SMARTS agreed AND features cached")
print(f"  {len(core)} unique -> training set")

X = np.vstack([f[s] for s in core.smiles])
y = core.pKa.values

print("\n5-fold OOF...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
for i, (tr, va) in enumerate(kf.split(X)):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                           num_leaves=31, verbose=-1, random_state=42)
    m.fit(X[tr], y[tr])
    oof[va] = m.predict(X[va])
    print(f"  fold {i+1}/5")
cal = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
print(f"\nOOF calibrated MAE: {np.mean(np.abs(cal.predict(oof) - y)):.3f}")

final = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                           num_leaves=31, verbose=-1, random_state=42)
final.fit(X, y)
joblib.dump({"regressor": final, "calibrator": cal},
            "models/model_core_v10_matched.pkl")
print("saved -> models/model_core_v10_matched.pkl")
