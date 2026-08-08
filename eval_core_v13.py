"""Eval the split acid/base models. Uses Marvin's marvin_pKa_type to
route each molecule to the right regressor."""
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, RDLogger
from tqdm import tqdm
from umapka import PkaPredictor
from umapka.predictor import (neutralize, _tag_and_reparse,
                               _shift_hydrogen_tagged, ACID_SITES, BASE_SITES)
RDLogger.DisableLog("rdApp.*")

b = joblib.load("models/model_core_v13_split_full.pkl")
acid_m, acid_cal = b["acid"]["regressor"], b["acid"]["calibrator"]
base_m, base_cal = b["base"]["regressor"], b["base"]["calibrator"]
print("loading UMA...")
p = PkaPredictor("models/model_core_v3.pkl")

def priority_atom(mol):
    for name, sm, ai in ACID_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m: return m[0][ai]
    for name, sm, ai in BASE_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m: return m[0][ai]
    return None

rows = []
for path, ds in [
    ("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf", "novartis"),
    ("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf", "avlilumove"),
]:
    mols = [m for m in Chem.ForwardSDMolSupplier(path) if m is not None]
    print(f"\n{ds}: {len(mols)}")
    for mol in tqdm(mols):
        if not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")):
            continue
        try:
            exp = float(mol.GetProp("pKa")); ma = int(float(mol.GetProp("marvin_atom")))
        except Exception:
            continue
        if not (0 < exp < 14): continue
        mt = mol.GetProp("marvin_pKa_type") if mol.HasProp("marvin_pKa_type") else ""
        kind = "acid" if mt.startswith("acid") else "base"
        try:
            nm = neutralize(Chem.Mol(mol))
            if ma >= nm.GetNumAtoms(): continue
            pidx = priority_atom(nm)
            if kind == "acid":
                prot, pi_ = _tag_and_reparse(nm, ma)
                dep, di_ = _shift_hydrogen_tagged(nm, ma, -1, -1)
            else:
                dep, di_ = _tag_and_reparse(nm, ma)
                prot, pi_ = _shift_hydrogen_tagged(nm, ma, +1, +1)
            if prot is None or dep is None: continue
            hg_p, hl_p = p.state_features_v4(prot, pi_, kind, n_confs_base=1)
            hg_d, hl_d = p.state_features_v4(dep, di_, kind, n_confs_base=1)
            g = np.concatenate([hg_p, hg_d, hg_p - hg_d])
            l = np.concatenate([hl_p, hl_d, hl_p - hl_d])
            feat = np.concatenate([g, l]).reshape(1, -1)
            if kind == "acid":
                pred = float(acid_cal.predict(acid_m.predict(feat))[0])
            else:
                pred = float(base_cal.predict(base_m.predict(feat))[0])
        except Exception:
            continue
        rows.append({"dataset": ds, "exp": exp, "pred": pred,
                     "err": abs(pred-exp), "kind": kind,
                     "smarts_agreed": (pidx == ma) if pidx is not None else None,
                     "n_atoms": nm.GetNumAtoms()})

d = pd.DataFrame(rows)
d.to_csv("characterization_external_v13.csv", index=False)
print(f"\n=== v13: SPLIT trained on FULL data (n={len(d)}) ===")
print(f"MAE = {d.err.mean():.3f}     (v11 was 0.737 combined)")
print("\n=== BY DATASET ===")
print(d.groupby("dataset")["err"].agg(["mean","count"]).round(3))
print("  v11: novartis 0.875 | avlilumove 0.420")
print("  Marvin (commercial): novartis 0.856 | avlilumove 0.566")
print("\n=== BY KIND ===")
print(d.groupby("kind")["err"].agg(["mean","count"]).round(3))
print("  v11: acid 0.842 | base 0.682")
print("\n=== BY SITE AGREEMENT ===")
print(d.groupby(["dataset","smarts_agreed"])["err"].agg(["mean","count"]).round(3))
d["size_bin"] = pd.cut(d.n_atoms, bins=[0,15,22,30,1000],
                        labels=["<15","15-22","22-30",">30"])
print("\n=== BY SIZE ===")
print(d.groupby("size_bin", observed=True)["err"].agg(["mean","count"]).round(3))
