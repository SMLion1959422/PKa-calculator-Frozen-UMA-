"""v14 = v11 training data + hunt_et_al molecules."""
import sys, numpy as np, pandas as pd, joblib, lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka.predictor import ACID_SITES, BASE_SITES, neutralize

def priority_atom(mol):
    for n_, sm, ai in ACID_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m: return m[0][ai]
    for n_, sm, ai in BASE_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m: return m[0][ai]
    return None

f = joblib.load("feat_train_v6.pkl")
valid = {s for s, v in f.items() if np.asarray(v).shape == (2304,)}
corrected = joblib.load("feat_marvin_corrected.pkl")
hunt = joblib.load("feat_hunt.pkl")

rows = []
for mol in Chem.ForwardSDMolSupplier("mlpka/datasets/combined_training_datasets_unique.sdf"):
    if mol is None or not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")): continue
    try:
        exp = float(mol.GetProp("pKa")); ma = int(float(mol.GetProp("marvin_atom")))
        smi = Chem.MolToSmiles(mol); nm = neutralize(Chem.Mol(mol))
    except Exception: continue
    if not (0 < exp < 14) or ma >= nm.GetNumAtoms(): continue
    pidx = priority_atom(nm)
    if pidx is not None and pidx == ma and smi in valid:
        rows.append({"smiles": smi, "pKa": exp, "vec": f[smi], "src": "v11"})
    elif smi in corrected:
        rows.append({"smiles": smi, "pKa": corrected[smi]["pKa"],
                     "vec": corrected[smi]["feat"], "src": "v11"})
for k, v in hunt.items():
    if np.asarray(v["feat"]).shape == (2304,):
        rows.append({"smiles": k, "pKa": v["pKa"], "vec": v["feat"], "src": "hunt"})

core = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
print(f"training set: {len(core)}  (v11={( core.src=='v11').sum()}, hunt={(core.src=='hunt').sum()})")
print(f"  v11 alone was 5269 -> {(len(core)/5269-1)*100:+.0f}% more data")

X = np.vstack(core.vec.values); y = core.pKa.values
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
for i, (tr, va) in enumerate(kf.split(X)):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           verbose=-1, random_state=42)
    m.fit(X[tr], y[tr]); oof[va] = m.predict(X[va]); print(f"  fold {i+1}/5")
cal = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
print(f"\nOOF calibrated MAE: {np.mean(np.abs(cal.predict(oof)-y)):.3f}  (v11 was 0.544)")
final = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           verbose=-1, random_state=42)
final.fit(X, y)
joblib.dump({"regressor": final, "calibrator": cal}, "models/model_core_v14.pkl")
print("saved -> models/model_core_v14.pkl")
