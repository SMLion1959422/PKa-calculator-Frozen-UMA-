"""Embeds ONLY the net-new molecules from extra_pka_data.csv (produced
by fetch_extra_pka_data.py), using the SAME v3-style global+site-local
feature pipeline as your existing feat_train_v3.pkl, then merges the
result into a new feat_train_v3_expanded.pkl - so you pay UMA compute
only for the genuinely new molecules, not the ~5.5k you already have
cached.

Uses v3 (not v4) deliberately: v4's multi-conformer/multi-scale changes
showed flat-to-negative results in our ablations, so there's no reason
to pay that extra compute here. If you want v4-style features on the
new data instead, swap state_features_v4 in for the plain
global+pool_local calls below - but given what we found, v3 is the
better base to expand data on.
"""
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from umapka import PkaPredictor
from umapka.predictor import protonation_pair_site_tagged

CHECKPOINT_EVERY = 200
OUT_PATH = "feat_train_v3_expanded.pkl"
PARTIAL_PATH = "feat_train_v3_expanded.pkl.partial"

print("loading extra_pka_data.csv...")
extra = pd.read_csv("extra_pka_data.csv")
print(f"  {len(extra)} candidate molecules to embed")

print("loading existing feat_train_v3.pkl (will be merged, not redone)...")
existing = joblib.load("feat_train_v3.pkl")
print(f"  {len(existing)} already-cached entries")

try:
    out = joblib.load(PARTIAL_PATH)
    print(f"resuming from checkpoint: {len(out)} new entries already done")
except FileNotFoundError:
    out = {}

print("loading UMA...")
p = PkaPredictor("models/model_core_v3.pkl")

n_fail = 0
todo = [s for s in extra.smiles if s not in out and s not in existing]
print(f"{len(todo)} molecules actually need embedding "
      f"(some may already be in feat_train_v3.pkl by coincidence)\n")

for smi in tqdm(todo):
    try:
        prot, prot_idx, deprot, deprot_idx = protonation_pair_site_tagged(smi)
        from umapka.predictor import _smiles_to_atoms_with_site
        atoms_p, idx_p, mol_p = _smiles_to_atoms_with_site(prot, prot_idx)
        atoms_d, idx_d, mol_d = _smiles_to_atoms_with_site(deprot, deprot_idx)
        emb_p = p.embeddings(atoms_p)
        emb_d = p.embeddings(atoms_d)
        hg_p, hl_p = p.pool(emb_p), p.pool_local(emb_p, idx_p, mol_p)
        hg_d, hl_d = p.pool(emb_d), p.pool_local(emb_d, idx_d, mol_d)
        global_feat = np.concatenate([hg_p, hg_d, hg_p - hg_d])
        local_feat = np.concatenate([hl_p, hl_d, hl_p - hl_d])
        out[smi] = np.concatenate([global_feat, local_feat])
    except Exception:
        n_fail += 1
        out[smi] = np.array(())
    if len(out) % CHECKPOINT_EVERY == 0:
        joblib.dump(out, PARTIAL_PATH)

merged = {**existing, **out}
joblib.dump(merged, OUT_PATH)
valid_new = sum(1 for v in out.values() if np.asarray(v).shape == (1536,))
valid_total = sum(1 for v in merged.values() if np.asarray(v).shape == (1536,))
print(f"\ndone: {valid_new} new valid entries ({n_fail} failed), "
      f"{valid_total} total valid entries in {OUT_PATH}")
print("next: run train_core_v3_expanded.py")
