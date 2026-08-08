"""Eval using Marvin's ground-truth protonation atom instead of SMARTS
first-match. Inference-only test: reuses model_core_v7_clean.pkl, no
re-embedding. Reports error split by whether SMARTS agreed with Marvin,
which directly shows whether wrong-atom selection is the error source.
"""
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from tqdm import tqdm
from umapka import PkaPredictor
from umapka.predictor import (ACID_SITES, BASE_SITES, neutralize,
                               _tag_and_reparse, _shift_hydrogen_tagged)

RDLogger.DisableLog("rdApp.*")
bundle = joblib.load("models/model_core_v7_clean.pkl")
regressor = bundle["regressor"]
calibrator = bundle.get("calibrator_isotonic") or bundle.get("calibrator")
print("loading UMA...")
p = PkaPredictor("models/model_core_v3.pkl")


def priority_atom(mol):
    for name, sm, ai in ACID_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m:
                return m[0][ai], "acid"
    for name, sm, ai in BASE_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m:
                return m[0][ai], "base"
    return None, None


def pair_at_atom(mol, atom_idx, kind):
    if kind == "acid":
        prot, pi_ = _tag_and_reparse(mol, atom_idx)
        dep, di_ = _shift_hydrogen_tagged(mol, atom_idx, -1, -1)
    else:
        dep, di_ = _tag_and_reparse(mol, atom_idx)
        prot, pi_ = _shift_hydrogen_tagged(mol, atom_idx, +1, +1)
    if prot is None or dep is None:
        raise RuntimeError("pair build failed")
    return prot, pi_, dep, di_


def features(prot, pi_, dep, di_, kind):
    hg_p, hl_p = p.state_features_v4(prot, pi_, kind, n_confs_base=1)
    hg_d, hl_d = p.state_features_v4(dep, di_, kind, n_confs_base=1)
    g = np.concatenate([hg_p, hg_d, hg_p - hg_d])
    l = np.concatenate([hl_p, hl_d, hl_p - hl_d])
    return np.concatenate([g, l]).reshape(1, -1)


rows = []
for path, dsname in [
    ("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf", "novartis"),
    ("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf", "avlilumove"),
]:
    mols = [m for m in Chem.ForwardSDMolSupplier(path) if m is not None]
    print(f"\n{dsname}: {len(mols)} molecules")
    for mol in tqdm(mols):
        if not mol.HasProp("marvin_atom") or not mol.HasProp("pKa"):
            continue
        try:
            exp = float(mol.GetProp("pKa"))
            ma = int(float(mol.GetProp("marvin_atom")))
        except Exception:
            continue
        if not (0 < exp < 14):
            continue
        mtype = mol.GetProp("marvin_pKa_type") if mol.HasProp("marvin_pKa_type") else ""
        kind = "acid" if mtype.startswith("acid") else "base"
        try:
            nm = neutralize(Chem.Mol(mol))
            if ma >= nm.GetNumAtoms():
                continue
            pidx, pkind = priority_atom(nm)
            prot, pi_, dep, di_ = pair_at_atom(nm, ma, kind)
            feat = features(prot, pi_, dep, di_, kind)
            raw = float(regressor.predict(feat)[0])
            pred = float(calibrator.predict([raw])[0])
        except Exception:
            continue
        rows.append({
            "dataset": dsname, "exp": exp, "pred": pred,
            "err": abs(pred - exp), "marvin_kind": kind,
            "smarts_agreed": (pidx == ma),
            "n_atoms": nm.GetNumAtoms(),
        })

d = pd.DataFrame(rows)
d.to_csv("characterization_external_v9_marvin.csv", index=False)

print(f"\n=== v9: MARVIN ground-truth sites (n={len(d)}) ===")
print(f"MAE = {d.err.mean():.3f}")
print("  v7-clean (SMARTS sites): 0.809 combined / 0.998 Novartis / 0.385 AvLiLuMoVe")
print("  v4 (best Novartis):      0.803 combined / 0.963 Novartis")

print("\n=== BY DATASET ===")
print(d.groupby("dataset")["err"].agg(["mean", "count"]).round(3))

print("\n=== THE KEY SPLIT: did SMARTS already agree with Marvin? ===")
print(d.groupby(["dataset", "smarts_agreed"])["err"].agg(["mean", "count"]).round(3))
print("  smarts_agreed=False = molecules SMARTS was featurizing at the")
print("  WRONG atom. Their error here is what correct site selection buys.")

print("\n=== BY MARVIN KIND ===")
print(d.groupby("marvin_kind")["err"].agg(["mean", "count"]).round(3))

d["size_bin"] = pd.cut(d.n_atoms, bins=[0, 15, 22, 30, 1000],
                        labels=["<15", "15-22", "22-30", ">30"])
print("\n=== BY SIZE ===")
print(d.groupby("size_bin", observed=True)["err"].agg(["mean", "count"]).round(3))
print("\nsaved -> characterization_external_v9_marvin.csv")
