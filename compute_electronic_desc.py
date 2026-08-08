"""Fallback electronic descriptors using RDKit only - no compiler, no
conda. Weaker than GFN2-xTB but captures the same KIND of information
UMA embeddings lack: local charge distribution and electronic
environment at the ionizable atom.

Gasteiger partial charges + EState indices + local topology, computed
at the protonation site and its neighbourhood, for both charge states.
~40 extra features appended to the 2304-dim UMA vector."""
import sys, numpy as np, pandas as pd, joblib
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.EState import EStateIndices
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka.predictor import ACID_SITES, BASE_SITES, neutralize, _tag_and_reparse, _shift_hydrogen_tagged

def elec_desc(smi, site_idx):
    """Electronic descriptors at the site + 1-2 bond neighbourhood."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None or site_idx is None or site_idx >= mol.GetNumAtoms():
        return None
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        return None
    q = np.array([float(a.GetPropsAsDict().get("_GasteigerCharge", 0.0))
                  for a in mol.GetAtoms()])
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        est = np.array(EStateIndices(mol))
    except Exception:
        est = np.zeros(mol.GetNumAtoms())
    dm = Chem.GetDistanceMatrix(mol)
    s1 = np.where(dm[site_idx] <= 1)[0]
    s2 = np.where(dm[site_idx] <= 2)[0]
    s3 = np.where(dm[site_idx] <= 3)[0]
    a = mol.GetAtomWithIdx(site_idx)
    return np.array([
        q[site_idx], est[site_idx],
        q[s1].mean(), q[s1].min(), q[s1].max(), est[s1].mean(),
        q[s2].mean(), q[s2].min(), q[s2].max(), est[s2].mean(),
        q[s3].mean(), q[s3].min(), q[s3].max(), est[s3].mean(),
        q.mean(), q.min(), q.max(), q.std(),
        float(a.GetDegree()), float(a.GetTotalNumHs()),
        float(a.GetFormalCharge()), float(a.GetIsAromatic()),
        float(a.IsInRing()), float(a.GetAtomicNum()),
        Descriptors.TPSA(mol), Crippen.MolLogP(mol),
        float(Chem.GetFormalCharge(mol)),
    ], dtype=float)

def priority_atom(mol):
    for n_, sm, ai in ACID_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m: return m[0][ai]
    for n_, sm, ai in BASE_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m: return m[0][ai]
    return None

out = {}
n_fail = 0
print("computing electronic descriptors for training molecules...")
for mol in tqdm(list(Chem.ForwardSDMolSupplier(
        "mlpka/datasets/combined_training_datasets_unique.sdf"))):
    if mol is None or not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")):
        continue
    try:
        exp = float(mol.GetProp("pKa")); ma = int(float(mol.GetProp("marvin_atom")))
        smi = Chem.MolToSmiles(mol); nm = neutralize(Chem.Mol(mol))
        if not (0 < exp < 14) or ma >= nm.GetNumAtoms(): continue
        mt = mol.GetProp("marvin_pKa_type") if mol.HasProp("marvin_pKa_type") else ""
        kind = "acid" if mt.startswith("acid") else "base"
        if kind == "acid":
            prot, pi_ = _tag_and_reparse(nm, ma); dep, di_ = _shift_hydrogen_tagged(nm, ma, -1, -1)
        else:
            dep, di_ = _tag_and_reparse(nm, ma); prot, pi_ = _shift_hydrogen_tagged(nm, ma, +1, +1)
        dp = elec_desc(prot, pi_); dd = elec_desc(dep, di_)
        if dp is None or dd is None: n_fail += 1; continue
        out[smi] = np.concatenate([dp, dd, dp - dd])
    except Exception:
        n_fail += 1

joblib.dump(out, "feat_electronic.pkl")
print(f"\ndone: {len(out)} molecules, {n_fail} failed")
print(f"descriptor dim: {len(next(iter(out.values())))}")
print("saved -> feat_electronic.pkl")
print("\nNEXT: train_hybrid_plus_electronic.py concatenates these onto")
print("the 2304-dim UMA features and reports whether they help.")
