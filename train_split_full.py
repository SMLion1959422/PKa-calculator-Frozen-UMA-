"""Fair comparison: split acid/base models trained on the FULL dataset
(same 5,269 molecules v11 used), removing the 15% holdout confound from
train_split_acid_base.py."""
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
            if m: return m[0][ai]
    for name, sm, ai in BASE_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m: return m[0][ai]
    return None

f = joblib.load("feat_train_v6.pkl")
valid = {s for s, v in f.items() if np.asarray(v).shape == (2304,)}
corrected = joblib.load("feat_marvin_corrected.pkl")

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
    if not (0 < exp < 14) or ma >= nm.GetNumAtoms(): continue
    mt = mol.GetProp("marvin_pKa_type") if mol.HasProp("marvin_pKa_type") else ""
    kind = "acid" if mt.startswith("acid") else "base"
    pidx = priority_atom(nm)
    if pidx is not None and pidx == ma and smi in valid:
        rows.append({"smiles": smi, "pKa": exp, "vec": f[smi], "kind": kind})
    elif smi in corrected:
        rows.append({"smiles": smi, "pKa": corrected[smi]["pKa"],
                     "vec": corrected[smi]["feat"], "kind": kind})

core = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
print(f"FULL training set: {len(core)} (acid={(core.kind=='acid').sum()}, "
      f"base={(core.kind=='base').sum()})  <-- same as v11, no holdout")

def fit(df, tag):
    X = np.vstack(df.vec.values); y = df.pKa.values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, va in kf.split(X):
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                               num_leaves=31, verbose=-1, random_state=42)
        m.fit(X[tr], y[tr]); oof[va] = m.predict(X[va])
    cal = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    final = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                               num_leaves=31, verbose=-1, random_state=42)
    final.fit(X, y)
    print(f"  {tag}: n={len(y)}  OOF cal MAE={np.mean(np.abs(cal.predict(oof)-y)):.3f}")
    return final, cal

print("\ntraining on FULL data:")
acid_m, acid_cal = fit(core[core.kind == "acid"], "acid")
base_m, base_cal = fit(core[core.kind == "base"], "base")

joblib.dump({"acid": {"regressor": acid_m, "calibrator": acid_cal},
             "base": {"regressor": base_m, "calibrator": base_cal}},
            "models/model_core_v13_split_full.pkl")
print("saved -> models/model_core_v13_split_full.pkl")
