"""PILOT ONLY - runs UMA-relaxed geometry on a small random subsample of
the external test set (default 40 molecules) to get (a) a timing
estimate and (b) an early accuracy signal, BEFORE committing to any
full training-set re-embed. Learn from v4: that took 12h38m for 403
molecules with no advance timing estimate. Don't repeat that blind.

Trains nothing - reuses the ALREADY-TRAINED v3 model's regressor
(1536-dim: global + local-2-bond-shell) and just swaps in relaxed
geometry for the FEATURE EXTRACTION at inference time on this small
sample, to see if the geometry change alone moves predictions in the
right direction before spending real compute finding out properly.
This is a rough signal, not a real evaluation (v3's regressor was
trained on non-relaxed features, so this is testing a mismatched
combination) - if it looks promising, the next real step is training a
model with relaxed-geometry features to match, on the full training
set.
"""
import time
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from tqdm import tqdm
from umapka import PkaPredictor
from umapka.predictor import protonation_pair_site_tagged

N_SAMPLE = 40
MAX_STEPS = 60   # relaxation steps per molecule state - lower this if
                  # the timing estimate below looks too expensive

def load_set(path, name):
    df = PandasTools.LoadSDF(path)
    pk_col = next(c for c in df.columns if c.lower() in ("pka","pka_value","value"))
    out = []
    for _, r in df.iterrows():
        m = r.get("ROMol")
        if m is None: continue
        try: v = float(r[pk_col])
        except: continue
        if not (0 < v < 14): continue
        try: out.append({"dataset": name, "smiles": Chem.MolToSmiles(m), "exp": v})
        except: pass
    return pd.DataFrame(out).drop_duplicates("smiles")

print("loading external test sets...")
sets = pd.concat([
    load_set("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf", "novartis"),
    load_set("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf", "avlilumove"),
]).reset_index(drop=True)

random.seed(42)
sample = sets.sample(n=min(N_SAMPLE, len(sets)), random_state=42).reset_index(drop=True)
print(f"piloting on {len(sample)} randomly sampled molecules "
      f"(max_steps={MAX_STEPS} per relaxation)\n")

print("loading UMA + v3 model (regressor unused for feature extraction, "
      "only for the rough side-by-side prediction)...")
p = PkaPredictor("models/model_core_v3.pkl")
import joblib
bundle = joblib.load("models/model_core_v3.pkl")
regressor, calibrator = bundle["regressor"], bundle["calibrator"]

rows = []
t_start = time.time()
for i, r in enumerate(tqdm(sample.itertuples(), total=len(sample))):
    t0 = time.time()
    try:
        prot, prot_idx, deprot, deprot_idx = protonation_pair_site_tagged(r.smiles)

        # baseline (non-relaxed, matches what v3 was actually trained on)
        feat_base = p.features(prot, deprot, prot_idx, deprot_idx)
        pred_base = float(calibrator.predict([regressor.predict(feat_base)[0]])[0])

        # relaxed-geometry variant (mismatched vs v3's training - rough
        # signal only, see module docstring)
        hg_p, hl_p = p.state_features_relaxed(prot, prot_idx, max_steps=MAX_STEPS)
        hg_d, hl_d = p.state_features_relaxed(deprot, deprot_idx, max_steps=MAX_STEPS)
        global_feat = np.concatenate([hg_p, hg_d, hg_p - hg_d])
        local_feat = np.concatenate([hl_p, hl_d, hl_p - hl_d])
        feat_relaxed = np.concatenate([global_feat, local_feat]).reshape(1, -1)
        pred_relaxed = float(calibrator.predict([regressor.predict(feat_relaxed)[0]])[0])

        elapsed = time.time() - t0
        rows.append({
            "smiles": r.smiles, "exp": r.exp,
            "pred_base": pred_base, "err_base": abs(pred_base - r.exp),
            "pred_relaxed": pred_relaxed, "err_relaxed": abs(pred_relaxed - r.exp),
            "seconds": elapsed,
        })
        if i == 4:
            per_mol = np.mean([x["seconds"] for x in rows])
            full_train_est_hr = per_mol * 5489 / 3600
            full_ext_est_min = per_mol * 393 / 60
            tqdm.write(f"\n  timing after 5 molecules: {per_mol:.1f}s/molecule")
            tqdm.write(f"  -> full external eval (393 mol) would take ~{full_ext_est_min:.0f} min")
            tqdm.write(f"  -> full training re-embed (~5489 mol) would take ~{full_train_est_hr:.1f} hours\n")
    except Exception as e:
        tqdm.write(f"  failed on {r.smiles}: {e}")

d = pd.DataFrame(rows)
total_min = (time.time() - t_start) / 60
print(f"\n=== PILOT RESULTS ({len(d)}/{len(sample)} succeeded, {total_min:.1f} min total) ===")
print(f"baseline (non-relaxed) MAE:  {d.err_base.mean():.3f}")
print(f"relaxed-geometry MAE:        {d.err_relaxed.mean():.3f}")
print(f"(remember: relaxed features are MISMATCHED with v3's training - ")
print(f" this only tells us the DIRECTION, not the real achievable MAE.")
print(f" A real answer needs a full re-embed + retrain with matched features.)")
print(f"\nmean time per molecule: {d.seconds.mean():.1f}s")
print(f"projected full training re-embed (~5489 mol, x2 for embed+train "
      f"needing BOTH states per molecule already counted): "
      f"~{d.seconds.mean()*5489/3600:.1f} hours")
