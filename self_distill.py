"""Label the embedded distillation candidates using the trained model for self-distillation."""
import joblib
import numpy as np
import pandas as pd

OUT_FEAT = "feat_distill.pkl"
OUT_LABELED = "distill_pseudo_labeled.csv"

print("Loading features...")
feats = joblib.load(OUT_FEAT)
smiles = list(feats.keys())
X = np.array(list(feats.values()))
print(f"Loaded {len(smiles)} feature vectors.")

print("Loading model...")
model = joblib.load("models/model_core_v3.pkl")

print("Generating pseudo-labels...")
preds = model.predict(X)

df_out = pd.DataFrame({"smiles": smiles, "pka_pseudo": preds})
df_out.to_csv(OUT_LABELED, index=False)
print(f"Saved pseudo-labeled dataset to {OUT_LABELED} ({len(df_out)} rows)")
print("\nNEXT: Combine with noisy data for pretraining, then finetune.")
