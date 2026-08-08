"""Tests whether giving the model explicit chemistry features - instead
of asking it to infer everything from the embedding alone - helps.
Every weak spot we've found all session (base-only sites, large
molecules, high ring count) is something we can compute directly and
cheaply with RDKit; this just hands that information to the model
directly rather than hoping it's implicitly recoverable from geometry.

Uses your EXISTING feat_train_v3.pkl - ZERO new UMA compute. Safe to
run right now, in parallel with the v6 background embedding job (pure
CPU, LightGBM only, no UMA model loaded at all during training).

IMPORTANT: don't run the matching eval script until the v6 background
job finishes - eval needs to load UMA again for the external test set,
and running two UMA-loaded processes on the same CPU machine at once
will slow down (or risk memory issues for) the overnight job. Training
here is safe now; evaluating isn't, yet.
"""
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools
from umapka.predictor import protonation_pair_site_tagged, neutralize

def compute_aux_features(smiles):
    """6 cheap, RDKit-only scalar features - no UMA involved."""
    try:
        _, _, _, _, kind = protonation_pair_site_tagged(smiles, return_kind=True)
    except Exception:
        kind = "acid"  # fallback, shouldn't happen for molecules already
                        # successfully embedded
    mol = Chem.MolFromSmiles(smiles)
    mol = neutralize(mol)
    return np.array([
        1.0 if kind == "base" else 0.0,
        mol.GetNumHeavyAtoms(),
        Descriptors.RingCount(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumRotatableBonds(mol),
        float(Chem.GetFormalCharge(mol)),
    ])

print("loading cached v3 features...")
f_train = joblib.load("feat_train_v3.pkl")
valid_smiles = {smi for smi, vec in f_train.items() if np.asarray(vec).shape == (1536,)}

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
print(f"  {len(core)} labeled molecules")

print("computing auxiliary chemistry features (RDKit only, fast)...")
aux = np.vstack([compute_aux_features(s) for s in core.smiles])
print(f"  aux feature shape: {aux.shape}")

X_embed = np.vstack([f_train[s] for s in core.smiles])
X = np.concatenate([X_embed, aux], axis=1)   # 1536 + 6 = 1542 dims
y = core.pKa.values
print(f"  combined feature shape: {X.shape}")

print("\n5-fold out-of-fold training...")
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
cal_mae = np.mean(np.abs(oof_cal - y))
print(f"\nOOF calibrated MAE: {cal_mae:.3f}")
print(f"(compare: v3 original OOF was 0.575 - same molecules, same base")
print(f" features, only the 6 extra scalar features are new)")

# feature importance check - do the aux features actually get used?
final_model_check = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                       num_leaves=31, verbose=-1, random_state=42)
final_model_check.fit(X, y)
aux_names = ["is_base", "n_heavy_atoms", "n_rings", "n_aromatic_rings",
             "n_rotatable_bonds", "formal_charge"]
importances = final_model_check.feature_importances_
aux_importance = importances[1536:]
print(f"\naux feature importances (out of {importances.sum():.0f} total split-count):")
for name, imp in zip(aux_names, aux_importance):
    print(f"  {name}: {imp} ({imp/importances.sum()*100:.2f}%)")
print(f"(if these are near zero, the model isn't using them - if they're")
print(f" meaningfully nonzero, they're pulling real weight)")

joblib.dump({"regressor": final_model_check, "calibrator": calibrator},
            "models/model_core_v3_auxfeat.pkl")
print("\nsaved -> models/model_core_v3_auxfeat.pkl")
print("\nDO NOT run the eval script until the v6 background job finishes -")
print("see this script's docstring for why.")
