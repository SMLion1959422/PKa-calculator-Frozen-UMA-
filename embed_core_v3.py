"""Recompute UMA embeddings for the aqueous training set, adding the new
site-local block (see umapka/predictor.py: pool_local/features_local)
alongside the existing global one. Reuses the SAME SMILES list as your
existing feat_train.pkl - the pKa label file isn't needed for this step,
only for train_core_v3.py afterward.

Run from the project root, in your UMA-authenticated venv. Same UMA
forward passes as your original embedding run, just keeping more of
each one's output - expect similar wall-clock time to whatever
feat_train.pkl originally took you.

Checkpoints every 200 molecules to feat_train_v3.pkl.partial so a crash
partway through doesn't lose progress; delete that file to start over
from scratch.
"""
import joblib
import numpy as np
from tqdm import tqdm
from umapka import PkaPredictor
from umapka.predictor import protonation_pair_site_tagged

CHECKPOINT_EVERY = 200
OUT_PATH = "feat_train_v3.pkl"
PARTIAL_PATH = "feat_train_v3.pkl.partial"

print("loading existing feat_train.pkl for its SMILES list...")
old = joblib.load("feat_train.pkl")
smiles_list = list(old.keys())
n_old_valid = sum(1 for v in old.values() if np.asarray(v).shape == (768,))
print(f"  {len(smiles_list)} molecules total ({n_old_valid} were valid "
      f"768-dim in the original file)")

try:
    out = joblib.load(PARTIAL_PATH)
    print(f"resuming from checkpoint: {len(out)} already done")
except FileNotFoundError:
    out = {}

print("loading UMA model (uma-s-1p1)...")
# any valid model path works here - we only use the embedding stack,
# not this file's regressor
p = PkaPredictor("models/model_core_v2.pkl")

n_fail = 0
todo = [s for s in smiles_list if s not in out]
for smi in tqdm(todo):
    try:
        prot, prot_idx, deprot, deprot_idx = protonation_pair_site_tagged(smi)
        # both blocks from the SAME 2 forward passes (prot once, deprot
        # once) - _state_features returns (global, local) together
        hg_p, hl_p = p._state_features(prot, prot_idx, need_local=True)
        hg_d, hl_d = p._state_features(deprot, deprot_idx, need_local=True)
        global_feat = np.concatenate([hg_p, hg_d, hg_p - hg_d])
        local_feat = np.concatenate([hl_p, hl_d, hl_p - hl_d])
        out[smi] = np.concatenate([global_feat, local_feat])
    except Exception:
        n_fail += 1
        out[smi] = np.array(())  # same "failed" marker convention as the
                                   # original feat_train.pkl
    if len(out) % CHECKPOINT_EVERY == 0:
        joblib.dump(out, PARTIAL_PATH)

joblib.dump(out, OUT_PATH)
valid = sum(1 for v in out.values() if np.asarray(v).shape == (1536,))
print(f"\ndone: {valid} valid 1536-dim entries, {n_fail} failed this run, "
      f"saved -> {OUT_PATH}")
print(f"(original feat_train.pkl had {n_old_valid} valid entries - if "
      f"{valid} is meaningfully higher, the new SMARTS coverage from "
      f"earlier recovered some previously-unparseable molecules for free)")
