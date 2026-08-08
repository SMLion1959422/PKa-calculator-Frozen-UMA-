"""v6 overnight run: multi-scale shell pooling (v4's validated-by-
evidence piece) WITHOUT multi-conformer (v4's expensive, unproven-by-
evidence piece), applied to the FULL EXPANDED dataset (original +
extra_pka_data.csv, ~13,600+ molecules) instead of the original ~5,489.

Reuses state_features_v4(..., n_confs_base=1) - with n_confs_base=1,
the "base site" branch generates exactly ONE conformer (same cost as
the single-conformer path), so this gets multi-scale shell pooling for
EVERYONE at v3-comparable UMA cost (2 forward passes/molecule), NOT
v4's original 3x-for-base-sites multiconf cost. No predictor.py changes
needed - this is a pure reuse of already-tested code.

Estimated cost: ~1.9s/molecule (matching tonight's extra-data embed
rate, since multi-scale shell adds no extra UMA passes over v3) x
~13,600 molecules =~ 7-8 hours. Checkpoints every 200 molecules.
"""
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import PandasTools
from umapka import PkaPredictor
from umapka.predictor import protonation_pair_site_tagged

CHECKPOINT_EVERY = 200
OUT_PATH = "feat_train_v6.pkl"
PARTIAL_PATH = "feat_train_v6.pkl.partial"

def load_labels():
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
        rows.append({"smiles": smi, "pKa": v})
    original = pd.DataFrame(rows).drop_duplicates("smiles")

    extra = pd.read_csv("extra_pka_data.csv")[["smiles", "pKa"]]
    combined = pd.concat([original, extra], ignore_index=True).drop_duplicates("smiles")
    return combined

print("loading combined label set (original + expanded)...")
labels = load_labels()
smiles_list = labels.smiles.tolist()
print(f"  {len(smiles_list)} unique molecules to embed")

try:
    out = joblib.load(PARTIAL_PATH)
    print(f"resuming from checkpoint: {len(out)} already done")
except FileNotFoundError:
    out = {}

print("loading UMA...")
p = PkaPredictor("models/model_core_v2.pkl")  # any valid model path works;
                                                # only embedding stack used

n_fail, n_acid, n_base = 0, 0, 0
todo = [s for s in smiles_list if s not in out]
print(f"{len(todo)} molecules remaining\n")

for smi in tqdm(todo):
    try:
        prot, prot_idx, deprot, deprot_idx, kind = protonation_pair_site_tagged(
            smi, return_kind=True)
        if kind == "acid": n_acid += 1
        else: n_base += 1
        # n_confs_base=1 -> single conformer for EVERYONE (acid and base
        # alike), multi-scale shell pooling only, no multiconf cost
        hg_p, hl_p = p.state_features_v4(prot, prot_idx, kind, n_confs_base=1)
        hg_d, hl_d = p.state_features_v4(deprot, deprot_idx, kind, n_confs_base=1)
        global_feat = np.concatenate([hg_p, hg_d, hg_p - hg_d])
        local_feat = np.concatenate([hl_p, hl_d, hl_p - hl_d])
        out[smi] = np.concatenate([global_feat, local_feat])
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
print(f"acid: {n_acid}, base: {n_base}")
print("next: run train_core_v6.py")
