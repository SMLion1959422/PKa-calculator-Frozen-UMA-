"""Re-check kind accuracy with the H-count rule BEFORE spending 15 min
on a full eval."""
import sys, numpy as np, pandas as pd, joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.EState import EStateIndices
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka.predictor import ACID_SITES, BASE_SITES, neutralize

ALL_PATTS = ([(n,s,ai,"acid") for n,s,ai in ACID_SITES] +
             [(n,s,ai,"base") for n,s,ai in BASE_SITES])
PATT_NAMES = [p[0] for p in ALL_PATTS]; PATT_IDX = {n:i for i,n in enumerate(PATT_NAMES)}
HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
       Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D]
sf = joblib.load("models/site_finder_v2.pkl")["model"]

def cand(mol):
    hits = {}
    for name, sm, ai, kind in ALL_PATTS:
        pt = Chem.MolFromSmarts(sm)
        if pt is None: continue
        for m in mol.GetSubstructMatches(pt):
            i = m[ai]; hits.setdefault(i, {"patterns":set(),"kinds":set()})
            hits[i]["patterns"].add(name); hits[i]["kinds"].add(kind)
    return hits

def feats(mol, idx, info, gast, est, dm, ctx, n_c, rank):
    a = mol.GetAtomWithIdx(idx)
    pv = [0.0]*len(PATT_NAMES)
    for q in info["patterns"]:
        if q in PATT_IDX: pv[PATT_IDX[q]] = 1.0
    s1 = np.where(dm[idx]<=1)[0]; s2 = np.where(dm[idx]<=2)[0]
    hyb = [1.0 if a.GetHybridization()==h else 0.0 for h in HYB]
    return pv + hyb + ctx + [float(a.GetAtomicNum()), float(a.GetFormalCharge()),
        float(a.GetTotalNumHs()), float(a.GetDegree()), float(a.GetIsAromatic()),
        float(a.IsInRing()), float(a.GetTotalValence()), gast[idx], est[idx],
        gast[s1].mean(), gast[s1].min(), gast[s1].max(),
        gast[s2].mean(), gast[s2].min(), gast[s2].max(),
        est[s1].mean(), est[s2].mean(), float(len(s1)-1), float(len(s2)-1),
        1.0 if "acid" in info["kinds"] else 0.0, 1.0 if "base" in info["kinds"] else 0.0,
        float(len(info["patterns"])), float(n_c), float(rank)]

rows = []
for path, ds in [("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf","novartis"),
                 ("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf","avlilumove")]:
    for mol in Chem.ForwardSDMolSupplier(path):
        if mol is None or not (mol.HasProp("marvin_atom") and mol.HasProp("marvin_pKa_type")): continue
        try:
            ma = int(float(mol.GetProp("marvin_atom")))
            tk = "acid" if mol.GetProp("marvin_pKa_type").startswith("acid") else "base"
            nm = neutralize(Chem.Mol(mol))
            if ma >= nm.GetNumAtoms(): continue
            hits = cand(nm)
            if not hits: continue
            AllChem.ComputeGasteigerCharges(nm)
            gast = np.nan_to_num(np.array([float(a.GetPropsAsDict().get("_GasteigerCharge",0.0))
                                            for a in nm.GetAtoms()]), nan=0.0,posinf=0.0,neginf=0.0)
            est = np.array(EStateIndices(nm)); dm = Chem.GetDistanceMatrix(nm)
            ctx = [Descriptors.MolWt(nm), Crippen.MolLogP(nm), Descriptors.TPSA(nm),
                   float(Descriptors.RingCount(nm)), float(Descriptors.NumAromaticRings(nm)),
                   float(nm.GetNumAtoms()), float(Chem.GetFormalCharge(nm))]
            ordered = sorted(hits.keys())
            F = np.array([feats(nm,i,hits[i],gast,est,dm,ctx,len(ordered),r)
                          for r,i in enumerate(ordered)], dtype=float)
            best = ordered[int(np.argmax(sf.predict(F)))]
            kinds = hits[best]["kinds"]
            if "acid" in kinds and "base" not in kinds: gk = "acid"
            elif "base" in kinds and "acid" not in kinds: gk = "base"
            else: gk = "acid" if nm.GetAtomWithIdx(best).GetTotalNumHs() > 0 else "base"
        except Exception: continue
        rows.append({"dataset":ds,"atom_ok":best==ma,"kind_ok":gk==tk,
                     "ambiguous":("acid" in kinds and "base" in kinds)})

d = pd.DataFrame(rows)
print(f"=== WITH H-COUNT RULE (n={len(d)}) ===\n")
print(d.groupby("dataset")[["atom_ok","kind_ok"]].mean().round(3))
amb = d[d.ambiguous]
if len(amb):
    print(f"\nambiguous subset (was 0/12): {amb.kind_ok.mean()*100:.0f}% correct, n={len(amb)}")
d["both"] = d.atom_ok & d.kind_ok
print("\nboth atom AND kind correct:")
print(d.groupby("dataset")["both"].agg(["mean","count"]).round(3))
print("\n  v19 old rule: novartis both_ok = 0.878")
print("  If this is meaningfully higher, re-run eval_v19_learned_sites.py.")
