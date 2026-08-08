"""Directly addresses audit_data_quality.py's Check 3 finding: training
set is 7.0% large molecules (>30 atoms), test sets are 16.9% - a 2.4x
under-representation of exactly the bucket where error is worst.

This reweights training examples so the EFFECTIVE training distribution
(by molecule size) matches the test distribution, via inverse-propensity
weighting - a standard, principled technique for exactly this kind of
train/test covariate shift. Uses your EXISTING feat_train_v3.pkl - no
new embedding, no GPU, same LightGBM hyperparameters as before, so any
MAE change is attributable to the reweighting alone.
"""
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from rdkit import Chem
from rdkit.Chem import PandasTools

SIZE_BINS = [0, 15, 22, 30, 1000]
SIZE_LABELS = ["<15", "15-22", "22-30", ">30"]

def get_size_fractions(smiles_list):
    sizes = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            sizes.append(m.GetNumAtoms())
    sizes = np.array(sizes)
    binned = pd.Series(pd.cut(sizes, bins=SIZE_BINS, labels=SIZE_LABELS))
    return binned.value_counts(normalize=True), sizes

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

print("\ncomputing train vs. test size-bucket fractions...")
train_frac, train_sizes = get_size_fractions(core.smiles)

def load_smiles(path):
    df = PandasTools.LoadSDF(path)
    out = []
    for _, r in df.iterrows():
        m = r.get("ROMol")
        if m is not None:
            try:
                out.append(Chem.MolToSmiles(m))
            except Exception:
                pass
    return out

test_smiles = (load_smiles("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf") +
               load_smiles("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf"))
test_frac, _ = get_size_fractions(test_smiles)

print("\nsize-bucket fractions:")
comparison = pd.DataFrame({"train": train_frac, "test": test_frac}).fillna(0)
comparison["weight"] = (comparison["test"] / comparison["train"]).replace([np.inf, -np.inf], 1.0).fillna(1.0)
print(comparison)

# per-molecule weight = target(test) fraction / source(train) fraction
# for its own size bucket - upweights under-represented (large) molecules,
# downweights over-represented (small) ones, WITHOUT discarding any data
core["n_atoms"] = core.smiles.apply(lambda s: Chem.MolFromSmiles(s).GetNumAtoms())
core["size_bin"] = pd.cut(core["n_atoms"], bins=SIZE_BINS, labels=SIZE_LABELS)
# .map() on a Categorical column can return a Categorical result instead
# of float in some pandas versions - go through a plain str-keyed dict
# to sidestep that entirely, then force float.
weight_map = {str(k): v for k, v in comparison["weight"].items()}
core["weight"] = core["size_bin"].astype(str).map(weight_map).astype(float)
# normalize so average weight is 1.0 (keeps overall loss scale comparable
# to the unweighted run for a fair MAE comparison)
core["weight"] = core["weight"] / core["weight"].mean()

print(f"\nweight range: {core.weight.min():.2f} to {core.weight.max():.2f}")
print(f"(>30 atoms gets ~{comparison.loc['>30','weight']:.2f}x its natural "
      f"training frequency, matching how often it appears in your test sets)")

X = np.vstack([f_train[s] for s in core.smiles])
y = core.pKa.values
w = core.weight.values

print("\n5-fold out-of-fold training WITH size weighting...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(y))
for i, (tr, va) in enumerate(kf.split(X)):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           verbose=-1, random_state=42)
    m.fit(X[tr], y[tr], sample_weight=w[tr])
    oof_pred[va] = m.predict(X[va])
    print(f"  fold {i+1}/5 done")

calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof_pred, y)
oof_cal = calibrator.predict(oof_pred)

# report BOTH plain and size-weighted OOF MAE - the weighted one is what
# the model was optimized for, but the plain one is what's comparable to
# v3's reported 0.575
raw_mae = np.mean(np.abs(oof_pred - y))
cal_mae = np.mean(np.abs(oof_cal - y))
print(f"\nout-of-fold raw MAE (unweighted avg):        {raw_mae:.3f}")
print(f"out-of-fold calibrated MAE (unweighted avg): {cal_mae:.3f}")
print(f"(compare to v3's 0.575 - same metric, same molecules, only the")
print(f" TRAINING loss weighting changed)")

print("\nby size bucket (the number that actually matters here):")
core["oof_err"] = np.abs(oof_cal - y)
print(core.groupby("size_bin", observed=True)["oof_err"].agg(["mean", "count"]).round(3))
print("(compare to v3's un-reweighted by-size OOF - if >30 atoms improved")
print(" here without badly hurting the smaller buckets, this is a real,")
print(" free win worth keeping)")

print("\ntraining final production model on all data...")
final_model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                 verbose=-1, random_state=42)
final_model.fit(X, y, sample_weight=w)

joblib.dump({"regressor": final_model, "calibrator": calibrator},
            "models/model_core_v3_sizeweighted.pkl")
print("\nsaved -> models/model_core_v3_sizeweighted.pkl")
print("next: point eval_core_v3.py at this model instead of model_core_v3.pkl")
print("for the real external Novartis/AvLiLuMoVe comparison")
