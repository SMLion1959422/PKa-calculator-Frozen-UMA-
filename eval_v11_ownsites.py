"""Rescores the v7 configuration (own SMARTS site-finding) on EXACTLY
the 280 molecules v9/v11/v13 used, so the paper's site-finding-error
decomposition (0.998 vs 0.875) compares identical molecule sets.

Only difference from eval_core_v11: uses the SMARTS priority atom
instead of marvin_atom, but keeps v11's model and the same molecule
filter (requires marvin_atom to exist, so the set matches)."""
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, RDLogger
from tqdm import tqdm
from umapka import PkaPredictor
from umapka.predictor import (neutralize, _tag_and_reparse,
                               _shift_hydrogen_tagged, ACID_SITES, BASE_SITES)
RDLogger.DisableLog("rdApp.*")

b = joblib.load("models/model_core_v11.pkl")
regressor, calibrator = b["regressor"], b["calibrator"]
print("loading UMA...")
p = PkaPredictor("models/model_core_v3.pkl")

def priority_pick(mol):
    for name, sm, ai in ACID_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m: return m[0][ai], "acid"
    for name, sm, ai in BASE_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m: return m[0][ai], "base"
    return None, None

rows = []
for path, ds in [
    ("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf", "novartis"),
    ("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf", "avlilumove"),
]:
    mols = [m for m in Chem.ForwardSDMolSupplier(path) if m is not None]
    print(f"\n{ds}: {len(mols)}")
    for mol in tqdm(mols):
        # SAME filter as v9/v11/v13 so the molecule set matches exactly
        if not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")):
            continue
        try:
            exp = float(mol.GetProp("pKa")); ma = int(float(mol.GetProp("marvin_atom")))
        except Exception:
            continue
        if not (0 < exp < 14): continue
        try:
            nm = neutralize(Chem.Mol(mol))
            if ma >= nm.GetNumAtoms(): continue
            pidx, pkind = priority_pick(nm)          # <-- OWN site-finding
            if pidx is None: continue
            if pkind == "acid":
                prot, pi_ = _tag_and_reparse(nm, pidx)
                dep, di_ = _shift_hydrogen_tagged(nm, pidx, -1, -1)
            else:
                dep, di_ = _tag_and_reparse(nm, pidx)
                prot, pi_ = _shift_hydrogen_tagged(nm, pidx, +1, +1)
            if prot is None or dep is None: continue
            hg_p, hl_p = p.state_features_v4(prot, pi_, pkind, n_confs_base=1)
            hg_d, hl_d = p.state_features_v4(dep, di_, pkind, n_confs_base=1)
            g = np.concatenate([hg_p, hg_d, hg_p - hg_d])
            l = np.concatenate([hl_p, hl_d, hl_p - hl_d])
            feat = np.concatenate([g, l]).reshape(1, -1)
            pred = float(calibrator.predict(regressor.predict(feat))[0])
        except Exception:
            continue
        rows.append({"dataset": ds, "smiles": Chem.MolToSmiles(mol), "exp": exp,
                     "pred": pred, "err": abs(pred-exp),
                     "site_correct": (pidx == ma), "n_atoms": nm.GetNumAtoms()})

d = pd.DataFrame(rows)
d.to_csv("characterization_external_v11_ownsites.csv", index=False)

print(f"\n=== v11 model + OWN SMARTS site-finding (n={len(d)}) ===")
print(f"MAE = {d.err.mean():.3f}")
print("\n=== BY DATASET ===")
print(d.groupby("dataset")["err"].agg(["mean","count"]).round(3))
print("""
  PAPER TABLE (now on IDENTICAL molecule sets):
    v11 + Marvin sites      -> novartis 0.875
    v11 + own site-finding  -> novartis (above)
    ChemAxon Marvin         -> novartis 0.856
  The gap between rows 1 and 2 is YOUR site-finding error.""")
print("\n=== site-finding accuracy on this set ===")
print(d.groupby("dataset")["site_correct"].agg(["mean","count"]).round(3))
