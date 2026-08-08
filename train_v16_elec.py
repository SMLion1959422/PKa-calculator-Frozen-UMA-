"""Train the winning UMA+electronic hybrid, then evaluate externally."""
import sys, numpy as np, pandas as pd, joblib, lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
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
elec = joblib.load("feat_electronic.pkl")

rows = []
for mol in Chem.ForwardSDMolSupplier("mlpka/datasets/combined_training_datasets_unique.sdf"):
    if mol is None or not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")): continue
    try:
        exp = float(mol.GetProp("pKa")); ma = int(float(mol.GetProp("marvin_atom")))
        smi = Chem.MolToSmiles(mol); nm = neutralize(Chem.Mol(mol))
    except Exception: continue
    if not (0 < exp < 14) or ma >= nm.GetNumAtoms() or smi not in elec: continue
    pidx = priority_atom(nm)
    if pidx is not None and pidx == ma and smi in valid:
        rows.append({"smiles": smi, "pKa": exp, "vec": np.concatenate([f[smi], elec[smi]])})
    elif smi in corrected:
        rows.append({"smiles": smi, "pKa": corrected[smi]["pKa"],
                     "vec": np.concatenate([corrected[smi]["feat"], elec[smi]])})

core = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
X = np.nan_to_num(np.vstack(core.vec.values)); y = core.pKa.values
print(f"training set: {len(y)}  dim: {X.shape[1]}")

scaler = StandardScaler().fit(X); Xs = scaler.transform(X)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
og = np.zeros(len(y)); orr = np.zeros(len(y))
for i, (tr, va) in enumerate(kf.split(X)):
    g = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           verbose=-1, random_state=42).fit(X[tr], y[tr])
    og[va] = g.predict(X[va])
    r = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Xs[tr], y[tr])
    orr[va] = r.predict(Xs[va]); print(f"  fold {i+1}/5")
bw, bm = 0.0, 1e9
for w in np.arange(0, 1.01, 0.05):
    m = np.mean(np.abs((1-w)*og + w*orr - y))
    if m < bm: bm, bw = m, w
blend = (1-bw)*og + bw*orr
cal = IsotonicRegression(out_of_bounds="clip").fit(blend, y)
print(f"\nOOF calibrated: {np.mean(np.abs(cal.predict(blend)-y)):.3f}  (blend w={bw:.2f})")

gf = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                        verbose=-1, random_state=42).fit(X, y)
rf = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Xs, y)
joblib.dump({"gbm": gf, "ridge": rf, "scaler": scaler, "blend_w": bw,
             "calibrator": cal}, "models/model_core_v16_elec.pkl")
print("saved -> models/model_core_v16_elec.pkl")
