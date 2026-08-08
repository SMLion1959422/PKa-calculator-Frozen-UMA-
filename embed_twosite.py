"""STEP 1 of the polyprotic training pipeline: embed all 4 microstates
for each 2-site training molecule.

WHY: the polyprotic benchmark sits at MAE 2.006, and the diagnosed cause
is that only 0.27% of training transitions involve a background of
|charge| >= 2. We have 1,303 two-site molecules with a real label. Each
gives 4 microstates and 4 transitions, of which only ONE is labelled -
but a thermodynamic cycle constrains all four, so the label can be
propagated to the unlabelled (often charged-background) transitions.

RUNTIME: ~1,303 molecules x 4 microstates = ~5,200 UMA calls. At the
~2s/call observed in eval runs, expect 3-4 hours. Checkpoints every 25
molecules to feat_twosite.pkl.partial - safe to interrupt and resume.

OUTPUT: feat_twosite.pkl mapping smiles -> {
    "states":     {(s0,s1): feature_vector_2385d},   per microstate/site
    "transitions": [(from_state, to_state, site_idx, is_labelled, label)],
    "label":      experimental pKa,
    "label_site": which site index carries it }
"""
import sys
if "venv311" not in sys.prefix:
    sys.exit("WRONG PYTHON: " + sys.prefix + "\n  activate venv311 first")

import re
import itertools
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.EState import EStateIndices

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka import PkaPredictor
from umapka.predictor import (ACID_SITES, BASE_SITES, neutralize,
                               _smiles_to_atoms_with_site)

TAG0 = 101
OUT = "feat_twosite.pkl"
PARTIAL = OUT + ".partial"
CHECKPOINT = 25


def all_sites(mol):
    out, seen = [], set()
    for n_, sm, ai in ACID_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is None:
            continue
        for m in mol.GetSubstructMatches(pt):
            if m[ai] not in seen:
                seen.add(m[ai])
                out.append((n_, "acid", m[ai]))
    for n_, sm, ai in BASE_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is None:
            continue
        for m in mol.GetSubstructMatches(pt):
            if m[ai] not in seen:
                seen.add(m[ai])
                out.append((n_, "base", m[ai]))
    return out


def strip_tags(smi):
    return re.sub(r":\d+\]", "]", smi)


def shift(smiles, tag, d_h, d_q):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    idx = next((a.GetIdx() for a in mol.GetAtoms()
                if a.GetAtomMapNum() == tag), None)
    if idx is None:
        return None
    rw = Chem.RWMol(mol)
    a = rw.GetAtomWithIdx(idx)
    nh = a.GetTotalNumHs() + d_h
    if nh < 0:
        return None
    a.SetNumExplicitHs(nh)
    a.SetNoImplicit(True)
    a.SetFormalCharge(a.GetFormalCharge() + d_q)
    try:
        o = rw.GetMol()
        Chem.SanitizeMol(o)
        return Chem.MolToSmiles(o)
    except Exception:
        return None


def elec_desc(smi, idx):
    mol = Chem.MolFromSmiles(smi)
    if mol is None or idx is None or idx >= mol.GetNumAtoms():
        return None
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        return None
    q = np.nan_to_num(np.array([
        (float(a.GetDoubleProp("_GasteigerCharge"))
         if a.HasProp("_GasteigerCharge") else 0.0)
        for a in mol.GetAtoms()]), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        est = np.array(EStateIndices(mol))
    except Exception:
        est = np.zeros(mol.GetNumAtoms())
    dm = Chem.GetDistanceMatrix(mol)
    s1 = np.where(dm[idx] <= 1)[0]
    s2 = np.where(dm[idx] <= 2)[0]
    s3 = np.where(dm[idx] <= 3)[0]
    a = mol.GetAtomWithIdx(idx)
    return np.array([q[idx], est[idx],
        q[s1].mean(), q[s1].min(), q[s1].max(), est[s1].mean(),
        q[s2].mean(), q[s2].min(), q[s2].max(), est[s2].mean(),
        q[s3].mean(), q[s3].min(), q[s3].max(), est[s3].mean(),
        q.mean(), q.min(), q.max(), q.std(),
        float(a.GetDegree()), float(a.GetTotalNumHs()),
        float(a.GetFormalCharge()), float(a.GetIsAromatic()),
        float(a.IsInRing()), float(a.GetAtomicNum()),
        Descriptors.TPSA(mol), Crippen.MolLogP(mol),
        float(Chem.GetFormalCharge(mol))], dtype=float)


print("loading two-site molecules...")
df = pd.read_csv("twosite_train.csv")
print(f"  {len(df)} molecules")

try:
    out = joblib.load(PARTIAL)
    print(f"resuming: {len(out)} already done")
except FileNotFoundError:
    out = {}

print("loading UMA...")
p = PkaPredictor("models/model_core_v3.pkl")

todo = df[~df.smiles.isin(out.keys())]
print(f"{len(todo)} remaining\n")

n_fail = 0
for r in tqdm(todo.itertuples(), total=len(todo)):
    try:
        nm = neutralize(Chem.MolFromSmiles(r.smiles))
        sites = all_sites(nm)
        if len(sites) != 2:
            n_fail += 1
            continue
        atoms_list = [s[2] for s in sites]
        if r.marvin_atom not in atoms_list:
            n_fail += 1
            continue
        label_site = atoms_list.index(r.marvin_atom)

        rw = Chem.RWMol(nm)
        for i, (_, _, idx) in enumerate(sites):
            rw.GetAtomWithIdx(idx).SetAtomMapNum(TAG0 + i)
        base = Chem.MolToSmiles(rw.GetMol())

        # build and embed all 4 microstates
        cache = {}
        for st in itertools.product([0, 1], repeat=2):
            smi = base
            for i, (_, kind, _) in enumerate(sites):
                t = TAG0 + i
                if kind == "base" and st[i] == 1:
                    smi = shift(smi, t, +1, +1)
                elif kind == "acid" and st[i] == 0:
                    smi = shift(smi, t, -1, -1)
                if smi is None:
                    break
            if smi is None:
                continue
            tm = Chem.MolFromSmiles(smi)
            if tm is None:
                continue
            tag_idx = {a.GetAtomMapNum(): a.GetIdx() for a in tm.GetAtoms()
                       if a.GetAtomMapNum() >= TAG0}
            clean = strip_tags(smi)
            cm = Chem.MolFromSmiles(clean)
            if cm is None:
                continue
            if not all(cm.GetAtomWithIdx(v).GetSymbol()
                       == tm.GetAtomWithIdx(v).GetSymbol()
                       for v in tag_idx.values()):
                continue
            atoms, _, mol_h = _smiles_to_atoms_with_site(
                clean, next(iter(tag_idx.values())))
            cache[st] = {"clean": clean, "emb": p.embeddings(atoms),
                         "mol_h": mol_h, "tag_idx": tag_idx}

        if len(cache) < 4:
            n_fail += 1
            continue

        # build feature vector for every transition
        feats, trans = {}, []
        for st in cache:
            for i in range(2):
                if st[i] != 1:
                    continue
                s2 = list(st); s2[i] = 0; s2 = tuple(s2)
                if s2 not in cache:
                    continue
                cp, cd = cache[st], cache[s2]
                t = TAG0 + i
                ip, idd = cp["tag_idx"][t], cd["tag_idx"][t]
                hgp = p.pool(cp["emb"])
                hlp = p.pool_local_multiscale(cp["emb"], ip, cp["mol_h"])
                hgd = p.pool(cd["emb"])
                hld = p.pool_local_multiscale(cd["emb"], idd, cd["mol_h"])
                g_ = np.concatenate([hgp, hgd, hgp - hgd])
                l_ = np.concatenate([hlp, hld, hlp - hld])
                dp = elec_desc(cp["clean"], ip)
                dd = elec_desc(cd["clean"], idd)
                if dp is None or dd is None:
                    continue
                feats[(st, s2)] = np.nan_to_num(
                    np.concatenate([g_, l_, dp, dd, dp - dd])).astype(np.float32)
                # the labelled transition: the labelled site ionizing from
                # the state where the OTHER site is in its neutral form
                other = 1 - i
                other_neutral = 1 if sites[other][1] == "acid" else 0
                is_lab = (i == label_site and st[other] == other_neutral)
                trans.append((st, s2, i, bool(is_lab)))

        if len(feats) < 4:
            n_fail += 1
            continue

        out[r.smiles] = {"feats": feats, "trans": trans,
                         "label": float(r.pKa), "label_site": label_site,
                         "site_kinds": [s[1] for s in sites]}
    except Exception:
        n_fail += 1

    if len(out) % CHECKPOINT == 0:
        joblib.dump(out, PARTIAL)

joblib.dump(out, OUT)
n_lab = sum(1 for v in out.values() if any(t[3] for t in v["trans"]))
print(f"\ndone: {len(out)} molecules, {n_fail} failed")
print(f"  with an identifiable labelled transition: {n_lab}")
print(f"  total transitions: {sum(len(v['feats']) for v in out.values())}")
print(f"saved -> {OUT}")
print("\nNEXT: python train_polyprotic_mlp.py")
