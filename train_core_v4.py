"""Same pipeline as train_core_v3.py (5-fold OOF, isotonic calibration,
IDENTICAL LightGBM hyperparameters), pointed at the new 2304-dim
(global + multi-scale local, with multi-conformer base sites) features.
"""
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from rdkit import Chem
from rdkit.Chem import PandasTools

print("loading cached UMA features (v4: multi-scale + multi-conf, 2304-dim)...")
f_train = joblib.load("feat_train_v4.pkl")
valid_smiles = {smi for smi, vec in f_train.items() if np.asarray(vec).shape == (2304,)}
print(f"  {len(valid_smiles)} valid 2304-dim entries")

print("loading pKa labels...")
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

core = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
X = np.vstack([f_train[s] for s in core.smiles])
y = core.pKa.values
print(f"  {len(y)} labeled molecules with valid v4 features")

print("5-fold out-of-fold training (honest, no leakage)...")
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
print(f"(compare: v3's OOF calibrated MAE was 0.575/0.568 (untuned/tuned) - "
      f"{'improvement' if cal_mae < 0.568 else 'no improvement'})")
print(f"\n(this is OOF, not external - run eval_core_v4.py next for the "
      f"real Novartis/AvLiLuMoVe comparison)")

print("\ntraining final production model on all data...")
final_model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                 verbose=-1, random_state=42)
final_model.fit(X, y)

joblib.dump({"regressor": final_model, "calibrator": calibrator}, "models/model_core_v4.pkl")
print("\nsaved -> models/model_core_v4.pkl")
