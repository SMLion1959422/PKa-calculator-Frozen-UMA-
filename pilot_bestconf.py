"""PILOT for best-of-N MMFF conformer selection - the classical, UMA-free
alternative to relax_with_uma (which pilot_relaxed_geometry.py already
showed is both too expensive - ~134h projected full re-embed - AND
directionally negative, 0.950->0.990 MAE). This should be MUCH cheaper:
the extra work here is pure classical MMFF conformer search, not UMA,
so UMA cost per molecule stays identical to v3 (one embedding pass per
state). Same 40-molecule pilot-first discipline as before - verify
timing AND direction before committing to a full re-embed.
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
N_CONFS = 10

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
      f"(n_confs={N_CONFS} per molecule state)\n")

print("loading UMA + v3 model...")
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

        # baseline (single-shot, matches what v3 was actually trained on)
        feat_base = p.features(prot, deprot, prot_idx, deprot_idx)
        pred_base = float(calibrator.predict([regressor.predict(feat_base)[0]])[0])

        # best-of-N variant (mismatched vs v3's training - rough signal only)
        hg_p, hl_p = p.state_features_bestconf(prot, prot_idx, n_confs=N_CONFS)
        hg_d, hl_d = p.state_features_bestconf(deprot, deprot_idx, n_confs=N_CONFS)
        global_feat = np.concatenate([hg_p, hg_d, hg_p - hg_d])
        local_feat = np.concatenate([hl_p, hl_d, hl_p - hl_d])
        feat_bestconf = np.concatenate([global_feat, local_feat]).reshape(1, -1)
        pred_bestconf = float(calibrator.predict([regressor.predict(feat_bestconf)[0]])[0])

        elapsed = time.time() - t0
        rows.append({
            "smiles": r.smiles, "exp": r.exp,
            "pred_base": pred_base, "err_base": abs(pred_base - r.exp),
            "pred_bestconf": pred_bestconf, "err_bestconf": abs(pred_bestconf - r.exp),
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
print(f"baseline (single-shot) MAE:  {d.err_base.mean():.3f}")
print(f"best-of-{N_CONFS} MAE:              {d.err_bestconf.mean():.3f}")
print(f"(remember: bestconf features are MISMATCHED with v3's training - ")
print(f" this only tells us the DIRECTION, not the real achievable MAE)")
print(f"\nmean time per molecule: {d.seconds.mean():.1f}s")
print(f"projected full training re-embed (~5489 mol): "
      f"~{d.seconds.mean()*5489/3600:.1f} hours")
print(f"\n(compare to relaxation pilot's 161.4s/molecule, 134h projected -")
print(f" this should be dramatically cheaper since there's no UMA-force")
print(f" optimization involved, just classical MMFF conformer search)")
