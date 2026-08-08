"""Embed the ~7% of training molecules where SMARTS disagreed with
Marvin - AT THEIR MARVIN ATOM - then retrain on everything. This adds
back exactly the site-chemistry v10's filter removed, which is why its
62 hardest Novartis molecules stayed at 1.478.

Small job: only the disagreeing molecules need embedding (~400), not
the whole set. Checkpoints every 50.
"""
import sys, os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka import PkaPredictor
from umapka.predictor import (ACID_SITES, BASE_SITES, neutralize,
                               _tag_and_reparse, _shift_hydrogen_tagged)

OUT = "feat_marvin_corrected.pkl"
PARTIAL = OUT + ".partial"

def priority_atom(mol):
    for name, sm, ai in ACID_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m:
                return m[0][ai]
    for name, sm, ai in BASE_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m:
                return m[0][ai]
    return None

print("finding training molecules where SMARTS != Marvin...")
todo = []
for mol in Chem.ForwardSDMolSupplier(
        "mlpka/datasets/combined_training_datasets_unique.sdf"):
    if mol is None or not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")):
        continue
    try:
        exp = float(mol.GetProp("pKa"))
        ma = int(float(mol.GetProp("marvin_atom")))
    except Exception:
        continue
    if not (0 < exp < 14):
        continue
    try:
        smi = Chem.MolToSmiles(mol)
        nm = neutralize(Chem.Mol(mol))
    except Exception:
        continue
    if ma >= nm.GetNumAtoms():
        continue
    pidx = priority_atom(nm)
    if pidx is not None and pidx != ma:
        mt = mol.GetProp("marvin_pKa_type") if mol.HasProp("marvin_pKa_type") else ""
        kind = "acid" if mt.startswith("acid") else "base"
        todo.append((smi, ma, kind, exp))

print(f"  {len(todo)} molecules to re-embed at Marvin sites")

try:
    out = joblib.load(PARTIAL)
    print(f"resuming: {len(out)} done")
except FileNotFoundError:
    out = {}

print("loading UMA...")
p = PkaPredictor("models/model_core_v3.pkl")

n_fail = 0
remaining = [t for t in todo if t[0] not in out]
print(f"{len(remaining)} remaining\n")
for smi, ma, kind, exp in tqdm(remaining):
    try:
        nm = neutralize(Chem.MolFromSmiles(smi))
        if kind == "acid":
            prot, pi_ = _tag_and_reparse(nm, ma)
            dep, di_ = _shift_hydrogen_tagged(nm, ma, -1, -1)
        else:
            dep, di_ = _tag_and_reparse(nm, ma)
            prot, pi_ = _shift_hydrogen_tagged(nm, ma, +1, +1)
        if prot is None or dep is None:
            raise RuntimeError("pair build failed")
        hg_p, hl_p = p.state_features_v4(prot, pi_, kind, n_confs_base=1)
        hg_d, hl_d = p.state_features_v4(dep, di_, kind, n_confs_base=1)
        g = np.concatenate([hg_p, hg_d, hg_p - hg_d])
        l = np.concatenate([hl_p, hl_d, hl_p - hl_d])
        out[smi] = {"feat": np.concatenate([g, l]), "pKa": exp}
    except Exception:
        n_fail += 1
    if len(out) % 50 == 0:
        joblib.dump(out, PARTIAL)

joblib.dump(out, OUT)
print(f"\ndone: {len(out)} embedded, {n_fail} failed -> {OUT}")
