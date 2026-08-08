"""Real GFN2-xTB electronic descriptors at the protonation site.

RUN THIS IN THE 'xtb' CONDA ENV (prompt should show "(xtb)").
It writes feat_xtb.pkl, which your normal venv311 then reads. The two
environments never need to talk to each other.

Replaces the Gasteiger/EState fallback, which already gave 0.537 ->
0.485 OOF and pushed Novartis past ChemAxon Marvin (0.845 vs 0.856).
xTB gives actual quantum-mechanical charges plus frontier orbital
energies - the same class of descriptor QupKake uses to reach 0.55.

NOTE: we use xTB CHARGES and ORBITAL ENERGIES, not total energies.
The project's own earlier work showed xTB TOTAL energies fail for pKa
by 2.5-3.5x the signal - but per-atom charges and frontier orbitals are
a different, far better-conditioned quantity.

Covers training AND both test sets in one pass (~6,400 molecules),
checkpointing every 100 so it can be interrupted and resumed.
"""
import sys
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka.predictor import (ACID_SITES, BASE_SITES, neutralize,
                               _tag_and_reparse, _shift_hydrogen_tagged)
from xtb.interface import Calculator
from xtb.utils import get_method

BOHR = 1.8897259886


def xtb_desc(smi, site_idx):
    """GFN2-xTB descriptors at site_idx and its 1/2/3-bond shells."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None or site_idx is None or site_idx >= mol.GetNumAtoms():
        return None
    charge = Chem.GetFormalCharge(mol)
    molh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(molh, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(molh)
    except Exception:
        pass
    nums = np.array([a.GetAtomicNum() for a in molh.GetAtoms()])
    pos = molh.GetConformer().GetPositions() * BOHR
    try:
        calc = Calculator(get_method("GFN2-xTB"), nums, pos, charge=float(charge))
        calc.set_verbosity(0)
        res = calc.singlepoint()
        q = np.array(res.get_charges())
        try:
            orb = np.array(res.get_orbital_eigenvalues())
            occ = np.array(res.get_orbital_occupations())
            homo = float(orb[occ > 0.5].max()) if (occ > 0.5).any() else 0.0
            lumo = float(orb[occ <= 0.5].min()) if (occ <= 0.5).any() else 0.0
        except Exception:
            homo = lumo = 0.0
        try:
            wbo = np.array(res.get_bond_orders())
        except Exception:
            wbo = None
    except Exception:
        return None
    dm = Chem.GetDistanceMatrix(molh)
    s1 = np.where(dm[site_idx] <= 1)[0]
    s2 = np.where(dm[site_idx] <= 2)[0]
    s3 = np.where(dm[site_idx] <= 3)[0]
    bo = float(wbo[site_idx].sum()) if (wbo is not None and wbo.ndim == 2) else 0.0
    return np.array([
        q[site_idx],
        q[s1].mean(), q[s1].min(), q[s1].max(),
        q[s2].mean(), q[s2].min(), q[s2].max(),
        q[s3].mean(), q[s3].min(), q[s3].max(),
        q.mean(), q.min(), q.max(), q.std(),
        homo, lumo, lumo - homo, bo,
        float(charge), float(molh.GetNumAtoms()),
    ], dtype=float)


def priority_atom(mol):
    for _n, sm, ai in ACID_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m:
                return m[0][ai]
    for _n, sm, ai in BASE_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m:
                return m[0][ai]
    return None


OUT = "feat_xtb.pkl"
PARTIAL = "feat_xtb.pkl.partial"

try:
    out = joblib.load(PARTIAL)
    print(f"resuming from checkpoint: {len(out)} already done")
except FileNotFoundError:
    out = {}

print("collecting molecules (training + both test sets)...")
targets = []
for path in ["mlpka/datasets/combined_training_datasets_unique.sdf",
             "mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf",
             "mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf"]:
    for mol in Chem.ForwardSDMolSupplier(path):
        if mol is None or not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")):
            continue
        try:
            exp = float(mol.GetProp("pKa"))
            ma = int(float(mol.GetProp("marvin_atom")))
            smi = Chem.MolToSmiles(mol)
            nm = neutralize(Chem.Mol(mol))
        except Exception:
            continue
        if not (0 < exp < 14) or ma >= nm.GetNumAtoms():
            continue
        mt = mol.GetProp("marvin_pKa_type") if mol.HasProp("marvin_pKa_type") else ""
        targets.append((smi, ma, "acid" if mt.startswith("acid") else "base"))

seen, uniq = set(), []
for t in targets:
    if t[0] not in seen:
        seen.add(t[0])
        uniq.append(t)
todo = [t for t in uniq if t[0] not in out]
print(f"{len(uniq)} unique molecules, {len(todo)} remaining\n")

n_fail = 0
for smi, ma, kind in tqdm(todo):
    try:
        nm = neutralize(Chem.MolFromSmiles(smi))
        if kind == "acid":
            prot, pi_ = _tag_and_reparse(nm, ma)
            dep, di_ = _shift_hydrogen_tagged(nm, ma, -1, -1)
        else:
            dep, di_ = _tag_and_reparse(nm, ma)
            prot, pi_ = _shift_hydrogen_tagged(nm, ma, +1, +1)
        if prot is None or dep is None:
            n_fail += 1
            continue
        dp = xtb_desc(prot, pi_)
        dd = xtb_desc(dep, di_)
        if dp is None or dd is None:
            n_fail += 1
            continue
        out[smi] = np.concatenate([dp, dd, dp - dd])
    except Exception:
        n_fail += 1
    if len(out) % 100 == 0:
        joblib.dump(out, PARTIAL)

joblib.dump(out, OUT)
print(f"\ndone: {len(out)} molecules embedded, {n_fail} failed")
if out:
    print(f"descriptor dim: {len(next(iter(out.values())))}")
print(f"saved -> {OUT}")
print("\nNEXT (back in venv311): run the xTB ablation to compare")
print("UMA+Gasteiger vs UMA+xTB vs UMA+both.")
