"""v4 re-embedding: multi-scale shell pooling (1/2/3-bond radii instead
of a single fixed radius) for everyone, PLUS multi-conformer averaging
specifically for base sites (amines/anilines/pyridines) - see
umapka/predictor.py: pool_local_multiscale / state_features_v4 for the
reasoning. Targets the two specific weak spots v3's eval turned up:
3+-ring systems (didn't improve much - hypothesis: needs a wider view
than 2 bonds) and base-only sites (worse than acid-only - hypothesis:
a single MMFF conformer misrepresents a basic nitrogen's geometry).

Reuses the SAME SMILES list as feat_train.pkl/feat_train_v3.pkl.

HONEST COST WARNING: base-kind molecules now need ~3x the UMA forward
passes (3 conformers x 2 states, vs 1 conformer x 2 states before).
Acid-kind molecules are unchanged (still 1 conformer x 2 states). If
most of your training set is base sites (the eval breakdown suggested
it is), expect this run to take meaningfully longer than
embed_core_v3.py did - plausibly 2-2.5x, not the same ballpark. Reduce
N_CONFS_BASE below if that's too slow to be worth it.

Checkpoints every 200 molecules, same as embed_core_v3.py.
"""
import joblib
import numpy as np
from tqdm import tqdm
from umapka import PkaPredictor
from umapka.predictor import protonation_pair_site_tagged

N_CONFS_BASE = 3   # conformers averaged for base sites only; lower this
                    # (e.g. to 2) if the cost warning above is too much
CHECKPOINT_EVERY = 200
OUT_PATH = "feat_train_v4.pkl"
PARTIAL_PATH = "feat_train_v4.pkl.partial"

print("loading existing feat_train_v3.pkl for its SMILES list...")
old = joblib.load("feat_train_v3.pkl")
smiles_list = list(old.keys())
n_old_valid = sum(1 for v in old.values() if np.asarray(v).shape == (1536,))
print(f"  {len(smiles_list)} molecules total ({n_old_valid} were valid "
      f"1536-dim in feat_train_v3.pkl)")

try:
    out = joblib.load(PARTIAL_PATH)
    print(f"resuming from checkpoint: {len(out)} already done")
except FileNotFoundError:
    out = {}

print("loading UMA model (uma-s-1p1)...")
p = PkaPredictor("models/model_core_v2.pkl")  # any valid model path works;
                                                # only the embedding stack
                                                # is used here

n_fail = 0
n_acid, n_base = 0, 0
todo = [s for s in smiles_list if s not in out]
for smi in tqdm(todo):
    try:
        prot, prot_idx, deprot, deprot_idx, kind = protonation_pair_site_tagged(
            smi, return_kind=True)
        if kind == "acid": n_acid += 1
        else: n_base += 1
        hg_p, hl_p = p.state_features_v4(prot, prot_idx, kind, N_CONFS_BASE)
        hg_d, hl_d = p.state_features_v4(deprot, deprot_idx, kind, N_CONFS_BASE)
        global_feat = np.concatenate([hg_p, hg_d, hg_p - hg_d])       # 768
        local_feat = np.concatenate([hl_p, hl_d, hl_p - hl_d])        # 1536
        out[smi] = np.concatenate([global_feat, local_feat])           # 2304
    except Exception:
        n_fail += 1
        out[smi] = np.array(())
    if len(out) % CHECKPOINT_EVERY == 0:
        joblib.dump(out, PARTIAL_PATH)
        tqdm.write(f"  checkpoint: {len(out)} done "
                   f"(acid={n_acid}, base={n_base}, fail={n_fail} this run)")

joblib.dump(out, OUT_PATH)
valid = sum(1 for v in out.values() if np.asarray(v).shape == (2304,))
print(f"\ndone: {valid} valid 2304-dim entries, {n_fail} failed this run, "
      f"saved -> {OUT_PATH}")
print(f"acid sites: {n_acid}, base sites: {n_base} (base got "
      f"{N_CONFS_BASE}x conformer averaging, acid did not)")
