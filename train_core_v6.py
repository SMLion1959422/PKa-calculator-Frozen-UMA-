"""Trains on feat_train_v6.pkl (multi-scale shell, expanded dataset).
Fits and saves BOTH isotonic and linear calibration, so the morning
evaluation can compare them directly without another multi-hour run -
directly targeting tonight's open question of whether isotonic
calibration instability in the sparse >10 pKa region contributed to
the expanded-data-with-v3-features run's degradation there.
"""
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from rdkit import Chem
from rdkit.Chem import PandasTools

print("loading v6 cached features (multi-scale shell, expanded data, 2304-dim)...")
f_train = joblib.load("feat_train_v6.pkl")
valid_smiles = {smi for smi, vec in f_train.items() if np.asarray(vec).shape == (2304,)}
print(f"  {len(valid_smiles)} valid entries")

print("loading combined labels...")
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
original = pd.DataFrame(rows).drop_duplicates("smiles")

extra = pd.read_csv("extra_pka_data.csv")
extra = extra[extra.smiles.isin(valid_smiles)][["smiles", "pKa"]]

core = pd.concat([original, extra], ignore_index=True).drop_duplicates("smiles").reset_index(drop=True)
X = np.vstack([f_train[s] for s in core.smiles])
y = core.pKa.values
print(f"  {len(y)} labeled molecules with valid v6 features")

print("\n5-fold out-of-fold training...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(y))
for i, (tr, va) in enumerate(kf.split(X)):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           verbose=-1, random_state=42)
    m.fit(X[tr], y[tr])
    oof_pred[va] = m.predict(X[va])
    print(f"  fold {i+1}/5 done")

calibrator_iso = IsotonicRegression(out_of_bounds="clip").fit(oof_pred, y)
calibrator_lin = LinearRegression().fit(oof_pred.reshape(-1, 1), y)

raw_mae = np.mean(np.abs(oof_pred - y))
iso_mae = np.mean(np.abs(calibrator_iso.predict(oof_pred) - y))
lin_mae = np.mean(np.abs(calibrator_lin.predict(oof_pred.reshape(-1, 1)) - y))

print(f"\nOOF raw MAE:              {raw_mae:.3f}")
print(f"OOF isotonic-calibrated:  {iso_mae:.3f}")
print(f"OOF linear-calibrated:    {lin_mae:.3f}")
print(f"(v3 original OOF: 0.575 | v3+expanded-data OOF: check your earlier log)")

# also report OOF by pKa bin, both calibrations - the key diagnostic
bins_ = pd.cut(y, bins=[0,4,7,10,14], labels=["<4","4-7","7-10",">10"])
diag = pd.DataFrame({
    "bin": bins_,
    "err_raw": np.abs(oof_pred - y),
    "err_iso": np.abs(calibrator_iso.predict(oof_pred) - y),
    "err_lin": np.abs(calibrator_lin.predict(oof_pred.reshape(-1,1)) - y),
})
print("\nOOF by pKa bin, raw vs both calibrations:")
print(diag.groupby("bin", observed=True)[["err_raw","err_iso","err_lin"]].agg(["mean","count"]).round(3))

print("\ntraining final production model on all data...")
final_model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                 verbose=-1, random_state=42)
final_model.fit(X, y)

joblib.dump({
    "regressor": final_model,
    "calibrator_isotonic": calibrator_iso,
    "calibrator_linear": calibrator_lin,
}, "models/model_core_v6.pkl")
print("\nsaved -> models/model_core_v6.pkl (both calibrators included)")
print("next: run eval_core_v6.py")
