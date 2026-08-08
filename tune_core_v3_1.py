"""Faster hyperparameter search for the v3 (global+site-local) feature
set - trimmed grid + 3-fold search (not 5-fold) so this actually takes
minutes, not the ~90+ min the full 72-config x 5-fold grid in
tune_core_v3.py turned out to need. Once a winner is found, it gets a
proper full 5-fold OOF refit (matching train_core_v3.py's methodology)
so the reported MAE and the saved calibrator are trustworthy, not just
a quick 3-fold estimate.

Re-uses feat_train_v3.pkl - NO more UMA compute needed.
"""
import time
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

# Trimmed grid: 2x2x2x2 = 16 configs (vs 72 before), learning_rate fixed
# at 0.05 since the earlier run's partial results (yours or a re-run)
# can tell us later if it's worth adding back.
GRID = {
    "n_estimators":      [400, 800],
    "num_leaves":        [31, 63],
    "min_child_samples":  [10, 20],
    "reg_lambda":        [0.0, 1.0],
}
SEARCH_FOLDS = 3   # trimmed for the search phase
FINAL_FOLDS = 5    # full refit for the winner, matching train_core_v3.py

keys = list(GRID.keys())
combos = list(itertools.product(*GRID.values()))
print(f"testing {len(combos)} configs with {SEARCH_FOLDS}-fold OOF each...")

kf_search = KFold(n_splits=SEARCH_FOLDS, shuffle=True, random_state=42)
results = []
t0 = time.time()
pbar = tqdm(list(enumerate(combos)), desc="hparam search", unit="config")
for i, combo in pbar:
    params = dict(zip(keys, combo))
    oof_pred = np.zeros(len(y))
    for tr, va in kf_search.split(X):
        m = lgb.LGBMRegressor(learning_rate=0.05, verbose=-1, random_state=42, **params)
        m.fit(X[tr], y[tr])
        oof_pred[va] = m.predict(X[va])
    mae = np.mean(np.abs(oof_pred - y))
    results.append({**params, "oof_mae": mae})
    pbar.set_postfix(mae=f"{mae:.4f}", best=f"{min(r['oof_mae'] for r in results):.4f}")
    tqdm.write(f"  [{i+1}/{len(combos)}] {params} -> {SEARCH_FOLDS}-fold OOF MAE {mae:.4f}")
    if i == 0:
        per_config = time.time() - t0
        eta_min = per_config * (len(combos) - 1) / 60
        tqdm.write(f"  (first config took {per_config:.0f}s -> "
                   f"estimated ~{eta_min:.1f} more minutes for the rest)")

results_df = pd.DataFrame(results).sort_values("oof_mae").reset_index(drop=True)
results_df.to_csv("hparam_search_results_fast.csv", index=False)

best = results_df.iloc[0].to_dict()
best_params = {k: (int(v) if k in ("n_estimators", "num_leaves", "min_child_samples") else v)
               for k, v in best.items() if k != "oof_mae"}
best_params["learning_rate"] = 0.05

print(f"\nbest config ({SEARCH_FOLDS}-fold search): {best_params}")
print(f"{SEARCH_FOLDS}-fold OOF MAE: {best['oof_mae']:.4f}")

print(f"\nrefitting winner with full {FINAL_FOLDS}-fold OOF "
      f"(matching train_core_v3.py's methodology) for an honest final "
      f"number and a proper calibrator...")
kf_final = KFold(n_splits=FINAL_FOLDS, shuffle=True, random_state=42)
oof_pred = np.zeros(len(y))
for tr, va in kf_final.split(X):
    m = lgb.LGBMRegressor(verbose=-1, random_state=42, **best_params)
    m.fit(X[tr], y[tr])
    oof_pred[va] = m.predict(X[va])
calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof_pred, y)
final_mae = np.mean(np.abs(oof_pred - y))
cal_mae = np.mean(np.abs(calibrator.predict(oof_pred) - y))

print(f"\nfinal {FINAL_FOLDS}-fold OOF raw MAE:        {final_mae:.4f}")
print(f"final {FINAL_FOLDS}-fold OOF calibrated MAE: {cal_mae:.4f}")
print(f"(train_core_v3.py's fixed-hyperparameter OOF was 0.575 - "
      f"{'improvement' if cal_mae < 0.575 else 'no improvement'})")

print("\ntraining final model with best config on all data...")
final_model = lgb.LGBMRegressor(verbose=-1, random_state=42, **best_params)
final_model.fit(X, y)

joblib.dump({"regressor": final_model, "calibrator": calibrator},
            "models/model_core_v3_tuned.pkl")
print("saved -> models/model_core_v3_tuned.pkl")
print("\nnext: point eval_core_v3.py at models/model_core_v3_tuned.pkl "
      "instead of models/model_core_v3.pkl (one-line change) and re-run "
      "for the real, external, like-for-like comparison.")