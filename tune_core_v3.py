"""Hyperparameter search for the v3 (global+site-local) feature set.
Re-uses feat_train_v3.pkl - NO more UMA compute needed, this is pure
LightGBM/CPU and should run in a couple minutes.

Same honest 5-fold OOF methodology as train_core_v3.py (no leakage: a
fold's OOF prediction never comes from a model that saw that fold's
labels). Reports OOF MAE for each config, then retrains the best one on
all data and saves it separately as model_core_v3_tuned.pkl - your
existing model_core_v3.pkl is untouched, so this is a strict A/B, not
an overwrite you can't undo.
"""
import itertools
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from rdkit import Chem
from rdkit.Chem import PandasTools

print("loading cached v3 features + labels (same as train_core_v3.py)...")
f_train = joblib.load("feat_train_v3.pkl")
valid_smiles = {smi for smi, vec in f_train.items() if np.asarray(vec).shape == (1536,)}

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
print(f"  {len(y)} labeled molecules\n")

# A modest, sane grid - not exhaustive, just probing whether more
# capacity/regularization helps now that the input is 1536-dim instead
# of 768. Expand this yourself if you want a wider search.
GRID = {
    "n_estimators":    [400, 800, 1200],
    "num_leaves":      [31, 63, 127],
    "learning_rate":   [0.05, 0.03],
    "min_child_samples": [10, 20],
    "reg_lambda":      [0.0, 1.0],
}

keys = list(GRID.keys())
combos = list(itertools.product(*GRID.values()))
print(f"testing {len(combos)} configurations with 5-fold OOF each "
      f"(this will take a few minutes, all CPU, no UMA)...\n")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
best_so_far = float("inf")
pbar = tqdm(list(enumerate(combos)), desc="hparam search", unit="config")
for i, combo in pbar:
    params = dict(zip(keys, combo))
    oof_pred = np.zeros(len(y))
    for tr, va in kf.split(X):
        m = lgb.LGBMRegressor(verbose=-1, random_state=42, **params)
        m.fit(X[tr], y[tr])
        oof_pred[va] = m.predict(X[va])
    mae = np.mean(np.abs(oof_pred - y))
    results.append({**params, "oof_mae": mae})
    best_so_far = min(best_so_far, mae)
    pbar.set_postfix(mae=f"{mae:.4f}", best=f"{best_so_far:.4f}")
    tqdm.write(f"  [{i+1}/{len(combos)}] {params} -> OOF MAE {mae:.4f}")

results_df = pd.DataFrame(results).sort_values("oof_mae").reset_index(drop=True)
results_df.to_csv("hparam_search_results.csv", index=False)

best = results_df.iloc[0].to_dict()
best_params = {k: (int(v) if k in ("n_estimators", "num_leaves", "min_child_samples") else v)
               for k, v in best.items() if k != "oof_mae"}

print(f"\nbest config: {best_params}")
print(f"best OOF MAE: {best['oof_mae']:.4f}  "
      f"(train_core_v3.py's fixed-hyperparameter OOF was 0.575 - "
      f"{'improvement' if best['oof_mae'] < 0.575 else 'no improvement'})")

print("\ntraining final model with best config on all data...")
final_model = lgb.LGBMRegressor(verbose=-1, random_state=42, **best_params)
final_model.fit(X, y)

# recompute OOF predictions with the best config for calibration fitting
oof_pred = np.zeros(len(y))
for tr, va in kf.split(X):
    m = lgb.LGBMRegressor(verbose=-1, random_state=42, **best_params)
    m.fit(X[tr], y[tr])
    oof_pred[va] = m.predict(X[va])
calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof_pred, y)

joblib.dump({"regressor": final_model, "calibrator": calibrator},
            "models/model_core_v3_tuned.pkl")
print("saved -> models/model_core_v3_tuned.pkl")
print("\nnext: point eval_core_v3.py at models/model_core_v3_tuned.pkl "
      "instead of models/model_core_v3.pkl (one-line change) and re-run "
      "for the real, external, like-for-like comparison.")
