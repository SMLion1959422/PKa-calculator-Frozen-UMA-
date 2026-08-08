"""Retrain on matched-site molecules PLUS the newly Marvin-corrected
ones - the full site-consistent training set."""
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

f = joblib.load("feat_train_v6.pkl")
valid = {s for s, v in f.items() if np.asarray(v).shape == (2304,)}
corrected = joblib.load("feat_marvin_corrected.pkl")
print(f"cached v6: {len(valid)} | marvin-corrected: {len(corrected)}")

rows = []
for mol in Chem.ForwardSDMolSupplier(
        "mlpka/datasets/combined_training_datasets_unique.sdf"):
    if mol is None or not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")):
        continue
    try:
        exp = float(mol.GetProp("pKa")); ma = int(float(mol.GetProp("marvin_atom")))
        smi = Chem.MolToSmiles(mol); nm = neutralize(Chem.Mol(mol))
    except Exception:
        continue
    if not (0 < exp < 14) or ma >= nm.GetNumAtoms():
        continue
    pidx = priority_atom(nm)
    if pidx is not None and pidx == ma and smi in valid:
        rows.append({"smiles": smi, "pKa": exp, "vec": f[smi], "src": "agreed"})
    elif smi in corrected:
        rows.append({"smiles": smi, "pKa": corrected[smi]["pKa"],
                     "vec": corrected[smi]["feat"], "src": "corrected"})

core = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
print(f"training set: {len(core)}  ({(core.src=='agreed').sum()} agreed + "
      f"{(core.src=='corrected').sum()} corrected)")

X = np.vstack(core.vec.values); y = core.pKa.values
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
for i, (tr, va) in enumerate(kf.split(X)):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                           num_leaves=31, verbose=-1, random_state=42)
    m.fit(X[tr], y[tr]); oof[va] = m.predict(X[va])
    print(f"  fold {i+1}/5")
cal = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
print(f"\nOOF calibrated MAE: {np.mean(np.abs(cal.predict(oof)-y)):.3f}")

final = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                           num_leaves=31, verbose=-1, random_state=42)
final.fit(X, y)
joblib.dump({"regressor": final, "calibrator": cal},
            "models/model_core_v11.pkl")
print("saved -> models/model_core_v11.pkl")
