"""External eval for v16 (UMA + electronic descriptors, hybrid head)."""
import sys, numpy as np, pandas as pd, joblib
from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka import PkaPredictor
from umapka.predictor import (neutralize, _tag_and_reparse,
                               _shift_hydrogen_tagged, ACID_SITES, BASE_SITES)
exec(open("compute_electronic_desc.py", encoding="utf-8").read().split("out = {}")[0].split("import sys")[1].replace("sys.path.insert(0, \".\")", ""), globals()) if False else None
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.EState import EStateIndices

def elec_desc(smi, site_idx):
    mol = Chem.MolFromSmiles(smi)
    if mol is None or site_idx is None or site_idx >= mol.GetNumAtoms(): return None
    try: AllChem.ComputeGasteigerCharges(mol)
    except Exception: return None
    q = np.nan_to_num(np.array([float(a.GetPropsAsDict().get("_GasteigerCharge", 0.0))
                                 for a in mol.GetAtoms()]), nan=0.0, posinf=0.0, neginf=0.0)
    try: est = np.array(EStateIndices(mol))
    except Exception: est = np.zeros(mol.GetNumAtoms())
    dm = Chem.GetDistanceMatrix(mol)
    s1 = np.where(dm[site_idx] <= 1)[0]; s2 = np.where(dm[site_idx] <= 2)[0]
    s3 = np.where(dm[site_idx] <= 3)[0]; a = mol.GetAtomWithIdx(site_idx)
    return np.array([q[site_idx], est[site_idx],
        q[s1].mean(), q[s1].min(), q[s1].max(), est[s1].mean(),
        q[s2].mean(), q[s2].min(), q[s2].max(), est[s2].mean(),
        q[s3].mean(), q[s3].min(), q[s3].max(), est[s3].mean(),
        q.mean(), q.min(), q.max(), q.std(),
        float(a.GetDegree()), float(a.GetTotalNumHs()), float(a.GetFormalCharge()),
        float(a.GetIsAromatic()), float(a.IsInRing()), float(a.GetAtomicNum()),
        Descriptors.TPSA(mol), Crippen.MolLogP(mol), float(Chem.GetFormalCharge(mol))], dtype=float)

import torch, torch.nn as nn
class Head(nn.Module):
    def __init__(s,d,h=512):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.2),
                            nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h//2,1))
    def forward(s,x): return s.net(x).squeeze(-1)
b = joblib.load("models/model_core_v19_pretrained.pkl")
_m = Head(b["dim"], b["hidden"])
_m.load_state_dict({k: torch.tensor(v) for k,v in b["state_dict"].items()})
_m.eval()
scaler, cal = b["scaler"], b["calibrator"]
print("loading UMA...")
p = PkaPredictor("models/model_core_v3.pkl")

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

rows = []
for path, ds in [
    ("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf", "novartis"),
    ("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf", "avlilumove"),
]:
    mols = [m for m in Chem.ForwardSDMolSupplier(path) if m is not None]
    print(f"\n{ds}: {len(mols)}")
    for mol in tqdm(mols):
        if not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")): continue
        try:
            exp = float(mol.GetProp("pKa")); ma = int(float(mol.GetProp("marvin_atom")))
            if not (0 < exp < 14): continue
            mt = mol.GetProp("marvin_pKa_type") if mol.HasProp("marvin_pKa_type") else ""
            kind = "acid" if mt.startswith("acid") else "base"
            nm = neutralize(Chem.Mol(mol))
            if ma >= nm.GetNumAtoms(): continue
            pidx = priority_atom(nm)
            if kind == "acid":
                prot, pi_ = _tag_and_reparse(nm, ma); dep, di_ = _shift_hydrogen_tagged(nm, ma, -1, -1)
            else:
                dep, di_ = _tag_and_reparse(nm, ma); prot, pi_ = _shift_hydrogen_tagged(nm, ma, +1, +1)
            if prot is None or dep is None: continue
            hg_p, hl_p = p.state_features_v4(prot, pi_, kind, n_confs_base=1)
            hg_d, hl_d = p.state_features_v4(dep, di_, kind, n_confs_base=1)
            g_ = np.concatenate([hg_p, hg_d, hg_p - hg_d])
            l_ = np.concatenate([hl_p, hl_d, hl_p - hl_d])
            dp = elec_desc(prot, pi_); dd = elec_desc(dep, di_)
            if dp is None or dd is None: continue
            feat = np.nan_to_num(np.concatenate([g_, l_, dp, dd, dp - dd])).reshape(1, -1)
            with torch.no_grad():
                raw = float(_m(torch.tensor(scaler.transform(feat), dtype=torch.float32)).item())
            pred = float(cal.predict([raw])[0])
        except Exception:
            continue
        rows.append({"dataset": ds, "exp": exp, "pred": pred, "err": abs(pred-exp),
                     "site_ok": (pidx == ma), "n_atoms": nm.GetNumAtoms(),
                     "pka_bin": pd.cut([exp], bins=[0,4,7,10,14],
                                        labels=["<4","4-7","7-10",">10"])[0]})

d = pd.DataFrame(rows)
d.to_csv("characterization_external_v19.csv", index=False)
print(f"\n=== v19: + noisy-site pretraining (n={len(d)}) ===")
print(f"MAE = {d.err.mean():.3f}     (v11 was 0.737 combined)")
print("\n=== BY DATASET ===")
print(d.groupby("dataset")["err"].agg(["mean","count"]).round(3))
print("  v11: novartis 0.875 | avlilumove 0.420")
print("  Marvin (commercial): novartis 0.856 | avlilumove 0.566")
print("\n=== BY pKa RANGE ===")
print(d.groupby("pka_bin", observed=True)["err"].agg(["mean","count"]).round(3))
print("\n=== BY SITE AGREEMENT ===")
print(d.groupby(["dataset","site_ok"])["err"].agg(["mean","count"]).round(3))
d["size_bin"] = pd.cut(d.n_atoms, bins=[0,15,22,30,1000], labels=["<15","15-22","22-30",">30"])
print("\n=== BY SIZE ===")
print(d.groupby("size_bin", observed=True)["err"].agg(["mean","count"]).round(3))
