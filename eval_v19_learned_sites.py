"""v19: v16's model with LEARNED site-finding at inference - no Marvin
annotations used. This is the fully self-sufficient configuration.

Compare against:
  v16 + marvin_atom sites   : 0.845 novartis  (needs ChemAxon)
  v16 + SMARTS priority     : 0.998 novartis  (free, but 56.5% sites)
  v19 + learned site-finder : this run        (free, 97.4% sites)"""
import sys, numpy as np, pandas as pd, joblib
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.EState import EStateIndices
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka import PkaPredictor
from umapka.predictor import (ACID_SITES, BASE_SITES, neutralize,
                               _tag_and_reparse, _shift_hydrogen_tagged)

ALL_PATTS = ([(n,s,ai,"acid") for n,s,ai in ACID_SITES] +
             [(n,s,ai,"base") for n,s,ai in BASE_SITES])
PATT_NAMES = [p[0] for p in ALL_PATTS]
PATT_IDX = {n:i for i,n in enumerate(PATT_NAMES)}
HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
       Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D]

sf = joblib.load("models/site_finder_v2.pkl")["model"]
b = joblib.load("models/model_core_v16_elec.pkl")
gbm, ridge, scaler, bw, cal = b["gbm"], b["ridge"], b["scaler"], b["blend_w"], b["calibrator"]
print("loading UMA...")
p = PkaPredictor("models/model_core_v3.pkl")

def candidate_atoms(mol):
    hits = {}
    for name, smarts, ai, kind in ALL_PATTS:
        pt = Chem.MolFromSmarts(smarts)
        if pt is None: continue
        for m in mol.GetSubstructMatches(pt):
            idx = m[ai]
            hits.setdefault(idx, {"patterns": set(), "kinds": set()})
            hits[idx]["patterns"].add(name); hits[idx]["kinds"].add(kind)
    return hits

def atom_features(mol, idx, info, gast, est, dm, ctx, n_c, rank):
    a = mol.GetAtomWithIdx(idx)
    pv = [0.0]*len(PATT_NAMES)
    for q in info["patterns"]:
        if q in PATT_IDX: pv[PATT_IDX[q]] = 1.0
    s1 = np.where(dm[idx] <= 1)[0]; s2 = np.where(dm[idx] <= 2)[0]
    hyb = [1.0 if a.GetHybridization()==h else 0.0 for h in HYB]
    return pv + hyb + ctx + [
        float(a.GetAtomicNum()), float(a.GetFormalCharge()),
        float(a.GetTotalNumHs()), float(a.GetDegree()),
        float(a.GetIsAromatic()), float(a.IsInRing()), float(a.GetTotalValence()),
        gast[idx], est[idx],
        gast[s1].mean(), gast[s1].min(), gast[s1].max(),
        gast[s2].mean(), gast[s2].min(), gast[s2].max(),
        est[s1].mean(), est[s2].mean(),
        float(len(s1)-1), float(len(s2)-1),
        1.0 if "acid" in info["kinds"] else 0.0,
        1.0 if "base" in info["kinds"] else 0.0,
        float(len(info["patterns"])), float(n_c), float(rank)]

def find_site(nm):
    hits = candidate_atoms(nm)
    if not hits: return None, None
    AllChem.ComputeGasteigerCharges(nm)
    gast = np.nan_to_num(np.array([float(a.GetPropsAsDict().get("_GasteigerCharge",0.0))
                                    for a in nm.GetAtoms()]), nan=0.0, posinf=0.0, neginf=0.0)
    est = np.array(EStateIndices(nm)); dm = Chem.GetDistanceMatrix(nm)
    ctx = [Descriptors.MolWt(nm), Crippen.MolLogP(nm), Descriptors.TPSA(nm),
           float(Descriptors.RingCount(nm)), float(Descriptors.NumAromaticRings(nm)),
           float(nm.GetNumAtoms()), float(Chem.GetFormalCharge(nm))]
    ordered = sorted(hits.keys())
    F = np.array([atom_features(nm, i, hits[i], gast, est, dm, ctx, len(ordered), r)
                  for r, i in enumerate(ordered)], dtype=float)
    best = ordered[int(np.argmax(sf.predict(F)))]
    kinds = hits[best]["kinds"]
    if "acid" in kinds and "base" not in kinds:
        kind = "acid"
    elif "base" in kinds and "acid" not in kinds:
        kind = "base"
    else:
        # AMBIGUOUS: atom matched both acid and base patterns.
        # Decide by whether it actually HAS a proton to give up - an
        # N-H flagged as acidic (amide, sulfonamide, aromatic N-H) is
        # an acid site; a nitrogen with a lone pair and no acidic H is
        # a base site. The old element-symbol rule scored 0/12 here.
        atom = nm.GetAtomWithIdx(best)
        kind = "acid" if atom.GetTotalNumHs() > 0 else "base"
    return best, kind

def elec_desc(smi, si):
    mol = Chem.MolFromSmiles(smi)
    if mol is None or si is None or si >= mol.GetNumAtoms(): return None
    try: AllChem.ComputeGasteigerCharges(mol)
    except Exception: return None
    q = np.nan_to_num(np.array([float(a.GetPropsAsDict().get("_GasteigerCharge",0.0))
                                 for a in mol.GetAtoms()]), nan=0.0, posinf=0.0, neginf=0.0)
    try: est = np.array(EStateIndices(mol))
    except Exception: est = np.zeros(mol.GetNumAtoms())
    dm = Chem.GetDistanceMatrix(mol)
    s1 = np.where(dm[si]<=1)[0]; s2 = np.where(dm[si]<=2)[0]; s3 = np.where(dm[si]<=3)[0]
    a = mol.GetAtomWithIdx(si)
    return np.array([q[si], est[si],
        q[s1].mean(), q[s1].min(), q[s1].max(), est[s1].mean(),
        q[s2].mean(), q[s2].min(), q[s2].max(), est[s2].mean(),
        q[s3].mean(), q[s3].min(), q[s3].max(), est[s3].mean(),
        q.mean(), q.min(), q.max(), q.std(),
        float(a.GetDegree()), float(a.GetTotalNumHs()), float(a.GetFormalCharge()),
        float(a.GetIsAromatic()), float(a.IsInRing()), float(a.GetAtomicNum()),
        Descriptors.TPSA(mol), Crippen.MolLogP(mol), float(Chem.GetFormalCharge(mol))], dtype=float)

rows = []
for path, ds in [
    ("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf","novartis"),
    ("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf","avlilumove"),
]:
    mols = [m for m in Chem.ForwardSDMolSupplier(path) if m is not None]
    print(f"\n{ds}: {len(mols)}")
    for mol in tqdm(mols):
        if not mol.HasProp("pKa"): continue
        try:
            exp = float(mol.GetProp("pKa"))
            if not (0 < exp < 14): continue
            ma = int(float(mol.GetProp("marvin_atom"))) if mol.HasProp("marvin_atom") else -1
            nm = neutralize(Chem.Mol(mol))
            site, kind = find_site(nm)
            if site is None: continue
            if kind == "acid":
                prot, pi_ = _tag_and_reparse(nm, site); dep, di_ = _shift_hydrogen_tagged(nm, site, -1, -1)
            else:
                dep, di_ = _tag_and_reparse(nm, site); prot, pi_ = _shift_hydrogen_tagged(nm, site, +1, +1)
            if prot is None or dep is None: continue
            hg_p, hl_p = p.state_features_v4(prot, pi_, kind, n_confs_base=1)
            hg_d, hl_d = p.state_features_v4(dep, di_, kind, n_confs_base=1)
            g_ = np.concatenate([hg_p, hg_d, hg_p-hg_d]); l_ = np.concatenate([hl_p, hl_d, hl_p-hl_d])
            dp = elec_desc(prot, pi_); dd = elec_desc(dep, di_)
            if dp is None or dd is None: continue
            feat = np.nan_to_num(np.concatenate([g_, l_, dp, dd, dp-dd])).reshape(1,-1)
            raw = (1-bw)*gbm.predict(feat)[0] + bw*ridge.predict(scaler.transform(feat))[0]
            pred = float(cal.predict([raw])[0])
        except Exception:
            continue
        rows.append({"dataset": ds, "exp": exp, "pred": pred, "err": abs(pred-exp),
                     "site_correct": (site == ma) if ma >= 0 else None,
                     "n_atoms": nm.GetNumAtoms(),
                     "pka_bin": pd.cut([exp], bins=[0,4,7,10,14],
                                        labels=["<4","4-7","7-10",">10"])[0]})

d = pd.DataFrame(rows)
d.to_csv("characterization_external_v19.csv", index=False)
print(f"\n=== v19: LEARNED SITE-FINDING, NO MARVIN AT INFERENCE (n={len(d)}) ===")
print(f"MAE = {d.err.mean():.3f}")
print("\n=== BY DATASET ===")
print(d.groupby("dataset")["err"].agg(["mean","count"]).round(3))
print("  v16 + marvin sites : novartis 0.845 | avlilumove 0.441")
print("  v16 + SMARTS sites : novartis 0.998")
print("  ChemAxon Marvin    : novartis 0.856 | avlilumove 0.566")
print("\n=== SITE ACCURACY ACHIEVED ===")
print(d.groupby("dataset")["site_correct"].agg(["mean","count"]).round(3))
print("\n=== BY pKa RANGE ===")
print(d.groupby("pka_bin", observed=True)["err"].agg(["mean","count"]).round(3))
d["size_bin"] = pd.cut(d.n_atoms, bins=[0,15,22,30,1000], labels=["<15","15-22","22-30",">30"])
print("\n=== BY SIZE ===")
print(d.groupby("size_bin", observed=True)["err"].agg(["mean","count"]).round(3))
