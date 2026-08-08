"""Separate acid and base regressors instead of one model for both.
v11 shows acid MAE 0.842 vs base 0.682 - one regressor is being asked
to learn two different physical relationships (neutral acid losing H+
vs protonated base losing H+). Nearly every published pKa method splits
these. Uses existing cached features - no re-embedding.

Also carves out an internal VALIDATION split so future decisions can be
made without looking at Novartis again (see chat: ~13 Novartis-guided
decisions have already biased that number).
"""
import sys
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
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
    mt = mol.GetProp("marvin_pKa_type") if mol.HasProp("marvin_pKa_type") else ""
    kind = "acid" if mt.startswith("acid") else "base"
    pidx = priority_atom(nm)
    if pidx is not None and pidx == ma and smi in valid:
        rows.append({"smiles": smi, "pKa": exp, "vec": f[smi], "kind": kind})
    elif smi in corrected:
        rows.append({"smiles": smi, "pKa": corrected[smi]["pKa"],
                     "vec": corrected[smi]["feat"], "kind": kind})

core = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
print(f"training set: {len(core)}  (acid={(core.kind=='acid').sum()}, "
      f"base={(core.kind=='base').sum()})")

# internal validation split - so we can stop tuning on Novartis
tr_df, val_df = train_test_split(core, test_size=0.15, random_state=7,
                                  stratify=core.kind)
print(f"internal train={len(tr_df)}  internal val={len(val_df)}")

def fit_eval(train_df, tag):
    X = np.vstack(train_df.vec.values); y = train_df.pKa.values
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

print("\n--- SINGLE model (current v11 approach) ---")
single, single_cal = fit_eval(tr_df, "combined")

print("\n--- SEPARATE acid / base models ---")
acid_m, acid_cal = fit_eval(tr_df[tr_df.kind == "acid"], "acid")
base_m, base_cal = fit_eval(tr_df[tr_df.kind == "base"], "base")

print("\n=== INTERNAL VALIDATION (never used for any decision) ===")
Xv = np.vstack(val_df.vec.values); yv = val_df.pKa.values
p_single = single_cal.predict(single.predict(Xv))
err_single = np.abs(p_single - yv)

p_split = np.zeros(len(yv))
for i, (v, k) in enumerate(zip(val_df.vec.values, val_df.kind.values)):
    v = v.reshape(1, -1)
    if k == "acid":
        p_split[i] = acid_cal.predict(acid_m.predict(v))[0]
    else:
        p_split[i] = base_cal.predict(base_m.predict(v))[0]
err_split = np.abs(p_split - yv)

print(f"single model  : MAE = {err_single.mean():.3f}")
print(f"split models  : MAE = {err_split.mean():.3f}")
print(f"change: {err_single.mean()-err_split.mean():+.3f} "
      f"({'SPLIT WINS' if err_split.mean() < err_single.mean() else 'no gain'})")

vd = val_df.copy(); vd["err_single"] = err_single; vd["err_split"] = err_split
print("\nby kind:")
print(vd.groupby("kind")[["err_single", "err_split"]].agg(["mean", "count"]).round(3))

joblib.dump({"acid": {"regressor": acid_m, "calibrator": acid_cal},
             "base": {"regressor": base_m, "calibrator": base_cal}},
            "models/model_core_v12_split.pkl")
print("\nsaved -> models/model_core_v12_split.pkl")
print("(only run the Novartis eval if the internal validation shows a gain)")
