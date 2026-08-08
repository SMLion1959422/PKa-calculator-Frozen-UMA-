"""Aniline is IN TRAINING with label 4.60 but predict_ladder says 9.88.
A GBM should fit its own training data. So the ladder must be building
DIFFERENT features than training/eval did.

Builds aniline's feature vector BOTH ways and compares numerically."""
import sys
import numpy as np
import joblib
sys.path.insert(0, ".")
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.EState import EStateIndices
RDLogger.DisableLog("rdApp.*")
from umapka import PkaPredictor
from umapka.predictor import (neutralize, _tag_and_reparse,
                               _shift_hydrogen_tagged, ACID_SITES, BASE_SITES)

def elec_desc(smi, idx):
    mol = Chem.MolFromSmiles(smi)
    if mol is None or idx is None or idx >= mol.GetNumAtoms(): return None
    try: AllChem.ComputeGasteigerCharges(mol)
    except Exception: return None
    q = np.nan_to_num(np.array([
        (float(a.GetDoubleProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge") else 0.0)
        for a in mol.GetAtoms()]), nan=0.0, posinf=0.0, neginf=0.0)
    try: est = np.array(EStateIndices(mol))
    except Exception: est = np.zeros(mol.GetNumAtoms())
    dm = Chem.GetDistanceMatrix(mol)
    s1 = np.where(dm[idx] <= 1)[0]; s2 = np.where(dm[idx] <= 2)[0]
    s3 = np.where(dm[idx] <= 3)[0]; a = mol.GetAtomWithIdx(idx)
    return np.array([q[idx], est[idx],
        q[s1].mean(), q[s1].min(), q[s1].max(), est[s1].mean(),
        q[s2].mean(), q[s2].min(), q[s2].max(), est[s2].mean(),
        q[s3].mean(), q[s3].min(), q[s3].max(), est[s3].mean(),
        q.mean(), q.min(), q.max(), q.std(),
        float(a.GetDegree()), float(a.GetTotalNumHs()), float(a.GetFormalCharge()),
        float(a.GetIsAromatic()), float(a.IsInRing()), float(a.GetAtomicNum()),
        Descriptors.TPSA(mol), Crippen.MolLogP(mol), float(Chem.GetFormalCharge(mol))],
        dtype=float)

b = joblib.load("models/model_core_v16_elec.pkl")
gbm, ridge, sc, bw, cal = b["gbm"], b["ridge"], b["scaler"], b["blend_w"], b["calibrator"]
print("loading UMA...")
p = PkaPredictor("models/model_core_v3.pkl")

def build_and_predict(prot, pi_, dep, di_, kind, label):
    hg_p, hl_p = p.state_features_v4(prot, pi_, kind, n_confs_base=1)
    hg_d, hl_d = p.state_features_v4(dep, di_, kind, n_confs_base=1)
    g_ = np.concatenate([hg_p, hg_d, hg_p - hg_d])
    l_ = np.concatenate([hl_p, hl_d, hl_p - hl_d])
    dp = elec_desc(prot, pi_); dd = elec_desc(dep, di_)
    feat = np.nan_to_num(np.concatenate([g_, l_, dp, dd, dp - dd])).reshape(1, -1)
    raw = (1-bw)*gbm.predict(feat)[0] + bw*ridge.predict(sc.transform(feat))[0]
    pk = float(cal.predict([raw])[0])
    print(f"  {label}")
    print(f"    prot = {prot!r}  idx={pi_}")
    print(f"    dep  = {dep!r}  idx={di_}")
    print(f"    -> pKa {pk:.2f}")
    return feat, pk

nm = neutralize(Chem.MolFromSmiles("Nc1ccccc1"))
n_idx = next(a.GetIdx() for a in nm.GetAtoms() if a.GetSymbol() == "N")
print(f"\naniline, N at index {n_idx}. TRAINING LABEL = 4.60\n")

print("=" * 62)
print("PATH A: training/eval convention (_tag_and_reparse + _shift_hydrogen_tagged)")
print("=" * 62)
dep_a, di_a = _tag_and_reparse(nm, n_idx)
prot_a, pi_a = _shift_hydrogen_tagged(nm, n_idx, +1, +1)
fa, pka_a = build_and_predict(prot_a, pi_a, dep_a, di_a, "base", "eval convention")

print("\n" + "=" * 62)
print("PATH B: predict_ladder convention (shift_h with NoImplicit + render_site)")
print("=" * 62)
TAG = 101
rw = Chem.RWMol(nm); rw.GetAtomWithIdx(n_idx).SetAtomMapNum(TAG)
work = Chem.MolToSmiles(rw.GetMol())

def find_tag(mol, tag):
    for a in mol.GetAtoms():
        if a.GetAtomMapNum() == tag: return a.GetIdx()
    return None

def shift_h(smiles, tag, d_h, d_q):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    idx = find_tag(mol, tag)
    if idx is None: return None
    rw = Chem.RWMol(mol); a = rw.GetAtomWithIdx(idx)
    n_h = a.GetTotalNumHs() + d_h
    if n_h < 0: return None
    a.SetNumExplicitHs(n_h); a.SetNoImplicit(True)
    a.SetFormalCharge(a.GetFormalCharge() + d_q)
    try:
        out = rw.GetMol(); Chem.SanitizeMol(out); return Chem.MolToSmiles(out)
    except Exception: return None

def render_site(smiles, tag):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None, None
    idx = find_tag(mol, tag)
    if idx is None: return None, None
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms(): a.SetAtomMapNum(0)
    rw.GetAtomWithIdx(idx).SetAtomMapNum(99)
    m2 = rw.GetMol(); Chem.SanitizeMol(m2)
    rt = Chem.MolFromSmiles(Chem.MolToSmiles(m2))
    if rt is None: return None, None
    ni = None
    for a in rt.GetAtoms():
        if a.GetAtomMapNum() == 99: ni = a.GetIdx(); a.SetAtomMapNum(0)
    return Chem.MolToSmiles(rt), ni

prot_work = shift_h(work, TAG, +1, +1)
prot_b, pi_b = render_site(prot_work, TAG)
dep_work = shift_h(prot_work, TAG, -1, -1)
dep_b, di_b = render_site(dep_work, TAG)
fb, pka_b = build_and_predict(prot_b, pi_b, dep_b, di_b, "base", "ladder convention")

print("\n" + "=" * 62)
print("COMPARISON")
print("=" * 62)
print(f"  eval convention  : {pka_a:.2f}")
print(f"  ladder convention: {pka_b:.2f}")
print(f"  training label   : 4.60")
print(f"\n  SMILES identical? prot: {prot_a == prot_b}   dep: {dep_a == dep_b}")
print(f"  canonical match?  prot: {Chem.MolToSmiles(Chem.MolFromSmiles(prot_a))==Chem.MolToSmiles(Chem.MolFromSmiles(prot_b))}"
      f"   dep: {Chem.MolToSmiles(Chem.MolFromSmiles(dep_a))==Chem.MolToSmiles(Chem.MolFromSmiles(dep_b))}")
d = np.abs(fa - fb).ravel()
print(f"\n  feature vectors: max|diff|={d.max():.4f}  mean|diff|={d.mean():.5f}")
print(f"  n features differing by >1e-6: {(d > 1e-6).sum()} / {len(d)}")
if d.max() > 1e-6:
    idxs = np.argsort(-d)[:10]
    print("  biggest differing features (index, A, B):")
    for i in idxs:
        seg = "global" if i < 768 else ("local" if i < 2304 else "electronic")
        print(f"    [{i}] {seg:10s} A={fa.ravel()[i]:+.4f}  B={fb.ravel()[i]:+.4f}")
print("""
VERDICT
  If PATH A gives ~4.6 and PATH B gives ~9.9, predict_ladder's SMILES
  construction is the bug and the v16 model is fine.
  If BOTH give ~9.9, the model genuinely fails on a molecule it was
  trained on, which would point at the training pipeline instead.""")
