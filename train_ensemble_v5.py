"""Ensemble UMA-embedding model (v4 features, already cached - zero new
UMA compute) with a fast ECFP4 fingerprint model (pure RDKit, seconds to
build). The two feature types capture genuinely different signal - 3D
pooled foundation-model embeddings vs. 2D substructure counts - so their
errors are less correlated than anything we've tried so far, which is
exactly the condition under which ensembling actually helps rather than
just averaging two similar mistakes together.

Uses OOF predictions (never a model's own training fold) to pick the
blend weight honestly, then trains final full models for deployment.
"""
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from rdkit import Chem
from rdkit.Chem import PandasTools, AllChem

print("loading cached v4 UMA features (2304-dim, no new UMA compute needed)...")
f_train = joblib.load("feat_train_v4.pkl")
valid_smiles = {smi for smi, vec in f_train.items() if np.asarray(vec).shape == (2304,)}

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
y = core.pKa.values
X_uma = np.vstack([f_train[s] for s in core.smiles])
print(f"  {len(y)} labeled molecules")

print("computing ECFP4 fingerprints (2048-bit, radius 2)...")
def ecfp4(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return np.array(fp, dtype=np.float32)

X_ecfp = np.vstack([ecfp4(s) for s in core.smiles])

print("5-fold OOF for both feature types (same folds, so predictions align)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_uma = np.zeros(len(y))
oof_ecfp = np.zeros(len(y))
for i, (tr, va) in enumerate(kf.split(X_uma)):
    m_uma = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               verbose=-1, random_state=42)
    m_uma.fit(X_uma[tr], y[tr])
    oof_uma[va] = m_uma.predict(X_uma[va])

    m_ecfp = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                verbose=-1, random_state=42)
    m_ecfp.fit(X_ecfp[tr], y[tr])
    oof_ecfp[va] = m_ecfp.predict(X_ecfp[va])
    print(f"  fold {i+1}/5 done")

mae_uma = np.mean(np.abs(oof_uma - y))
mae_ecfp = np.mean(np.abs(oof_ecfp - y))
print(f"\nOOF MAE, UMA-only:   {mae_uma:.4f}")
print(f"OOF MAE, ECFP4-only: {mae_ecfp:.4f}")

print("\nsearching blend weight alpha (pred = alpha*UMA + (1-alpha)*ECFP4)...")
best_alpha, best_mae = 1.0, mae_uma
for alpha in np.arange(0.0, 1.01, 0.05):
    blended = alpha * oof_uma + (1 - alpha) * oof_ecfp
    mae = np.mean(np.abs(blended - y))
    marker = ""
    if mae < best_mae:
        best_mae, best_alpha = mae, alpha
        marker = "  <-- best so far"
    print(f"  alpha={alpha:.2f}  OOF MAE={mae:.4f}{marker}")

print(f"\nbest blend: alpha={best_alpha:.2f}, OOF MAE={best_mae:.4f}")
print(f"(compare: UMA-only OOF was {mae_uma:.4f}; "
      f"{'ensembling helps' if best_mae < mae_uma else 'ensembling does NOT help here'})")

blended_oof = best_alpha * oof_uma + (1 - best_alpha) * oof_ecfp
calibrator = IsotonicRegression(out_of_bounds="clip").fit(blended_oof, y)
cal_mae = np.mean(np.abs(calibrator.predict(blended_oof) - y))
print(f"blended + calibrated OOF MAE: {cal_mae:.4f}")

print("\ntraining final full models for deployment...")
final_uma = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               verbose=-1, random_state=42).fit(X_uma, y)
final_ecfp = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                verbose=-1, random_state=42).fit(X_ecfp, y)

joblib.dump({
    "uma_regressor": final_uma,
    "ecfp_regressor": final_ecfp,
    "alpha": best_alpha,
    "calibrator": calibrator,
}, "models/model_ensemble_v5.pkl")
print("saved -> models/model_ensemble_v5.pkl")
print("\nnext: run eval_ensemble_v5.py for the real external comparison")
