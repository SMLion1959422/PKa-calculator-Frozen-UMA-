"""Same 5-fold OOF + isotonic calibration pipeline as train_core_v3.py,
but trained on the EXPANDED dataset: your original
combined_training_datasets_unique.sdf labels PLUS the net-new molecules
from extra_pka_data.csv, using feat_train_v3_expanded.pkl for features.
Same LightGBM hyperparameters as before, so any MAE change is
attributable to the added data, not a hyperparameter change.
"""
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from rdkit import Chem
from rdkit.Chem import PandasTools

print("loading expanded cached features...")
f_train = joblib.load("feat_train_v3_expanded.pkl")
valid_smiles = {smi for smi, vec in f_train.items() if np.asarray(vec).shape == (1536,)}
print(f"  {len(valid_smiles)} valid 1536-dim entries")

print("loading original pKa labels...")
df = PandasTools.LoadSDF("mlpka/datasets/combined_training_datasets_unique.sdf")
pk_col = next(c for c in df.columns if c.lower() in ("pka", "pka_value", "value"))
rows = []
for _, r in df.iterrows():
    m = r.get("ROMol")
    if m is None:
        continue
    try:
        v = float(r[pk_col])
    except Exception:
        continue
    if not (0 < v < 14):
        continue
    try:
        smi = Chem.MolToSmiles(m)
    except Exception:
        continue
    if smi in valid_smiles:
        rows.append({"smiles": smi, "pKa": v})
original_labels = pd.DataFrame(rows)
print(f"  {len(original_labels)} original labeled molecules")

print("loading extra pKa labels...")
extra = pd.read_csv("extra_pka_data.csv")
extra = extra[extra.smiles.isin(valid_smiles)][["smiles", "pKa"]]
print(f"  {len(extra)} extra labeled molecules with valid features")

core = pd.concat([original_labels, extra], ignore_index=True).drop_duplicates("smiles").reset_index(drop=True)
X = np.vstack([f_train[s] for s in core.smiles])
y = core.pKa.values
print(f"\ncombined training set: {len(y)} molecules "
      f"(was {len(original_labels)} before expansion)")

print("\n5-fold out-of-fold training (honest, no leakage)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(y))
for i, (tr, va) in enumerate(kf.split(X)):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           verbose=-1, random_state=42)
    m.fit(X[tr], y[tr])
    oof_pred[va] = m.predict(X[va])
    print(f"  fold {i+1}/5 done")

calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof_pred, y)
oof_cal = calibrator.predict(oof_pred)

raw_mae = np.mean(np.abs(oof_pred - y))
cal_mae = np.mean(np.abs(oof_cal - y))
print(f"\nout-of-fold raw MAE:        {raw_mae:.3f}")
print(f"out-of-fold calibrated MAE: {cal_mae:.3f}")
print(f"(compare: v3 (original data) OOF calibrated MAE was 0.575)")
print(f"\n(this is OOF, not external - run eval_core_v3.py pointed at "
      f"models/model_core_v3_expanded.pkl for the real Novartis/"
      f"AvLiLuMoVe comparison)")

print("\ntraining final production model on all data...")
final_model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                 verbose=-1, random_state=42)
final_model.fit(X, y)

joblib.dump({"regressor": final_model, "calibrator": calibrator},
            "models/model_core_v3_expanded.pkl")
print("\nsaved -> models/model_core_v3_expanded.pkl")
