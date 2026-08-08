"""2) EXTRAPOLATION-CAPABLE MODEL.

LightGBM predicts leaf averages, so its output is HARD-BOUNDED by the
training label range - with training data spanning pKa 2-12 it cannot
emit 1.5 or 12.8 no matter what the features say. That is why the <4
and >10 buckets have been the worst all along and why no amount of tree
tuning fixed them.

Fix: blend the tree with a RIDGE model on the same features. Ridge is
linear, so it extrapolates smoothly outside the training range. The
blend weight is chosen by out-of-fold error, not by hand."""
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
try:
    hunt = joblib.load("feat_hunt.pkl"); print(f"including hunt: {len(hunt)}")
except FileNotFoundError:
    hunt = {}

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
        rows.append({"smiles": smi, "pKa": exp, "vec": f[smi]})
    elif smi in corrected:
        rows.append({"smiles": smi, "pKa": corrected[smi]["pKa"], "vec": corrected[smi]["feat"]})
for k, v in hunt.items():
    if np.asarray(v["feat"]).shape == (2304,):
        rows.append({"smiles": k, "pKa": v["pKa"], "vec": v["feat"]})

core = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
X = np.vstack(core.vec.values); y = core.pKa.values
print(f"training set: {len(y)}   label range: {y.min():.2f} - {y.max():.2f}")

scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_gbm = np.zeros(len(y)); oof_rdg = np.zeros(len(y))
for i, (tr, va) in enumerate(kf.split(X)):
    g = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           verbose=-1, random_state=42).fit(X[tr], y[tr])
    oof_gbm[va] = g.predict(X[va])
    r = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Xs[tr], y[tr])
    oof_rdg[va] = r.predict(Xs[va])
    print(f"  fold {i+1}/5")

print(f"\nOOF MAE  gbm={np.mean(np.abs(oof_gbm-y)):.3f}  ridge={np.mean(np.abs(oof_rdg-y)):.3f}")
best_w, best_mae = 0.0, 1e9
for w in np.arange(0, 1.01, 0.05):
    mae = np.mean(np.abs((1-w)*oof_gbm + w*oof_rdg - y))
    if mae < best_mae: best_mae, best_w = mae, w
print(f"best blend: {1-best_w:.2f}*gbm + {best_w:.2f}*ridge -> OOF MAE {best_mae:.3f}")

oof_blend = (1-best_w)*oof_gbm + best_w*oof_rdg
cal = IsotonicRegression(out_of_bounds="clip").fit(oof_blend, y)
print(f"calibrated OOF MAE: {np.mean(np.abs(cal.predict(oof_blend)-y)):.3f}")

print(f"\n--- EXTRAPOLATION CHECK ---")
print(f"gbm   output range: {oof_gbm.min():.2f} - {oof_gbm.max():.2f}")
print(f"ridge output range: {oof_rdg.min():.2f} - {oof_rdg.max():.2f}")
print(f"blend output range: {oof_blend.min():.2f} - {oof_blend.max():.2f}")
print("  (ridge/blend reaching beyond the tree's range = extrapolation works)")

gbm_f = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           verbose=-1, random_state=42).fit(X, y)
rdg_f = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Xs, y)
joblib.dump({"gbm": gbm_f, "ridge": rdg_f, "scaler": scaler,
             "blend_w": best_w, "calibrator": cal},
            "models/model_core_v15_hybrid.pkl")
print("\nsaved -> models/model_core_v15_hybrid.pkl")
