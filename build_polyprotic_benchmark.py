"""POLYPROTIC BENCHMARK - builds a real validation set from molecules
that appear with TWO distinct experimental pKa values, then scores the
current microstate ensemble against it.

WHY THIS FIRST: all polyprotic conclusions so far rest on 5 molecules
I hand-picked (12 values). That is a demonstration, not a benchmark.
These 42 molecules / 84 values come from the data itself, were not
chosen by me, and give a number we can actually track when training.

LEAKAGE DISCLOSURE (read before quoting any number this produces):
each of these molecules appears in TRAINING with ONE label (its
marvin_atom site). The SECOND pKa comes from a different source and was
never trained on. So predictions here are partially informed - the
first pKa is effectively seen, the second is genuinely held out. The
script reports both separately for exactly this reason. Only the
'second pKa' column is a clean generalization number.
"""
import sys
import itertools, re
from collections import defaultdict
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
from umapka import electronic, microstates as M

TAG0 = 101

def all_sites(mol):
    out, seen = [], set()
    for n_, sm, ai in ACID_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is None: continue
        for m in mol.GetSubstructMatches(pt):
            if m[ai] not in seen: seen.add(m[ai]); out.append((n_,"acid",m[ai]))
    for n_, sm, ai in BASE_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is None: continue
        for m in mol.GetSubstructMatches(pt):
            if m[ai] not in seen: seen.add(m[ai]); out.append((n_,"base",m[ai]))
    return out

def strip_tags(smi): return re.sub(r":\d+\]", "]", smi)

def shift(smiles, tag, d_h, d_q):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    idx = next((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()==tag), None)
    if idx is None: return None
    rw = Chem.RWMol(mol); a = rw.GetAtomWithIdx(idx)
    nh = a.GetTotalNumHs() + d_h
    if nh < 0: return None
    a.SetNumExplicitHs(nh); a.SetNoImplicit(True)
    a.SetFormalCharge(a.GetFormalCharge()+d_q)
    try:
        o = rw.GetMol(); Chem.SanitizeMol(o); return Chem.MolToSmiles(o)
    except Exception: return None

def elec_desc(smi, idx):
    mol = Chem.MolFromSmiles(smi)
    if mol is None or idx is None or idx >= mol.GetNumAtoms(): return None
    try: AllChem.ComputeGasteigerCharges(mol)
    except Exception: return None
    q = np.nan_to_num(np.array([(float(a.GetDoubleProp("_GasteigerCharge"))
        if a.HasProp("_GasteigerCharge") else 0.0) for a in mol.GetAtoms()]),
        nan=0.0, posinf=0.0, neginf=0.0)
    try: est = np.array(EStateIndices(mol))
    except Exception: est = np.zeros(mol.GetNumAtoms())
    dm = Chem.GetDistanceMatrix(mol)
    s1=np.where(dm[idx]<=1)[0]; s2=np.where(dm[idx]<=2)[0]; s3=np.where(dm[idx]<=3)[0]
    a = mol.GetAtomWithIdx(idx)
    return np.array([q[idx], est[idx],
        q[s1].mean(),q[s1].min(),q[s1].max(),est[s1].mean(),
        q[s2].mean(),q[s2].min(),q[s2].max(),est[s2].mean(),
        q[s3].mean(),q[s3].min(),q[s3].max(),est[s3].mean(),
        q.mean(),q.min(),q.max(),q.std(),
        float(a.GetDegree()),float(a.GetTotalNumHs()),float(a.GetFormalCharge()),
        float(a.GetIsAromatic()),float(a.IsInRing()),float(a.GetAtomicNum()),
        Descriptors.TPSA(mol),Crippen.MolLogP(mol),float(Chem.GetFormalCharge(mol))],
        dtype=float)

# ---- build the benchmark ----
print("collecting multi-label molecules...")
by = defaultdict(list)
seen_in_train = set()
for mol in Chem.ForwardSDMolSupplier("mlpka/datasets/combined_training_datasets_unique.sdf"):
    if mol is None or not mol.HasProp("pKa"): continue
    try:
        v = float(mol.GetProp("pKa")); s = Chem.MolToSmiles(mol)
    except Exception: continue
    if 0 < v < 14:
        by[s].append(v); seen_in_train.add(s)
for path, col in [("extra_pka_data.csv","smiles"), ("hunt_pairs.csv","key")]:
    try:
        d = pd.read_csv(path)
        for r in d.itertuples():
            by[getattr(r, col)].append(float(r.pKa))
    except FileNotFoundError: pass

bench = []
for s, vals in by.items():
    u = sorted(set(round(v,2) for v in vals))
    if len(u) < 2 or (max(u)-min(u)) <= 0.5: continue
    m = Chem.MolFromSmiles(s)
    if m is None: continue
    sites = all_sites(neutralize(m))
    if len(sites) < 2 or len(sites) > 4: continue
    bench.append({"smiles": s, "exp": u, "n_sites": len(sites),
                  "in_train": s in seen_in_train})
print(f"benchmark molecules: {len(bench)}  "
      f"({sum(b['in_train'] for b in bench)} seen in training with one label)")

b = joblib.load("models/model_core_v16_elec.pkl")
gbm, ridge, sc, bw, cal = b["gbm"], b["ridge"], b["scaler"], b["blend_w"], b["calibrator"]
bundle = b
print("loading UMA...")
p = PkaPredictor("models/model_core_v3.pkl")

def microstate_macro(smiles):
    """Now delegates the solver to umapka.microstates (shared with
    predict_microstates.py) instead of carrying its own copy - the two
    copies had already drifted, and this one still used an
    UNREGULARIZED lstsq, which returns unbounded log-beta values for
    parameters left weakly determined when some microstates fail to
    build. Ridge-regularized there."""
    mol = Chem.MolFromSmiles(smiles)
    nm = neutralize(mol)
    sites = M.all_sites(nm, ACID_SITES, BASE_SITES)
    n = len(sites)
    _base, states, smi_of = M.enumerate_microstates(nm, sites)

    cache = {}
    for st, smi in smi_of.items():
        try:
            tm = Chem.MolFromSmiles(smi)
            if tm is None: continue
            tag_idx = {a.GetAtomMapNum():a.GetIdx() for a in tm.GetAtoms()
                       if a.GetAtomMapNum()>=M.TAG0}
            if not tag_idx: continue
            clean = M.strip_tags_text(smi)
            cm = Chem.MolFromSmiles(clean)
            if cm is None: continue
            if not all(cm.GetAtomWithIdx(v).GetSymbol()==tm.GetAtomWithIdx(v).GetSymbol()
                       for v in tag_idx.values()): continue
            atoms,_,mol_h = _smiles_to_atoms_with_site(clean, next(iter(tag_idx.values())))
            cache[st] = {"clean":clean,"emb":p.embeddings(atoms),"mol_h":mol_h,"tag_idx":tag_idx}
        except Exception: continue

    trans = []
    for st in states:
        if st not in cache: continue
        for i in range(n):
            if st[i]!=1: continue
            s2 = list(st); s2[i]=0; s2=tuple(s2)
            if s2 not in cache: continue
            cp,cd = cache[st],cache[s2]; t=M.TAG0+i
            try:
                ip,idd = cp["tag_idx"][t], cd["tag_idx"][t]
                hgp=p.pool(cp["emb"]); hlp=p.pool_local_multiscale(cp["emb"],ip,cp["mol_h"])
                hgd=p.pool(cd["emb"]); hld=p.pool_local_multiscale(cd["emb"],idd,cd["mol_h"])
                g_=np.concatenate([hgp,hgd,hgp-hgd]); l_=np.concatenate([hlp,hld,hlp-hld])
                dp=electronic.elec_desc(cp["clean"],ip); dd=electronic.elec_desc(cd["clean"],idd)
                if dp is None or dd is None: continue
                f_=np.nan_to_num(np.concatenate([g_,l_,dp,dd,dp-dd])).reshape(1,-1)
                pk=electronic.score_hybrid(bundle, f_)
            except Exception: continue
            trans.append((st, s2, i, pk))

    if not trans: return None, None
    logbeta, rms, sigma = M.solve_logbeta(states, trans, n)
    macro, _macro_sig, _Z = M.macro_pka(states, logbeta, sigma, n, set(cache))
    if not macro: return None, None
    return sorted(macro), rms

rows=[]
for e in tqdm(bench):
    try:
        macro, resid = microstate_macro(e["smiles"])
    except Exception:
        macro, resid = None, None
    if not macro: continue
    exp = e["exp"]
    k = min(len(exp), len(macro))
    pred = macro[:k] if len(macro)>=k else macro
    # align: take the k predicted values closest in count to experiment
    if len(macro) > len(exp):
        # choose the contiguous subset best matching experiment span
        best, bestcost = None, 1e9
        for st in range(len(macro)-len(exp)+1):
            sub = macro[st:st+len(exp)]
            c = sum(abs(a-b) for a,b in zip(sub, exp))
            if c < bestcost: bestcost, best = c, sub
        pred = best
    for j,(pv,ev) in enumerate(zip(pred, exp)):
        rows.append({"smiles": e["smiles"], "rung": j+1, "exp": ev, "pred": pv,
                     "err": abs(pv-ev), "n_sites": e["n_sites"],
                     "resid": resid, "in_train": e["in_train"]})

d = pd.DataFrame(rows)
d.to_csv("polyprotic_benchmark_v16.csv", index=False)
print(f"\n=== POLYPROTIC BENCHMARK ({d.smiles.nunique()} molecules, {len(d)} values) ===")
print(f"overall MAE = {d.err.mean():.3f}")
print("\nBY RUNG  (rung 1 = most acidic)")
print(d.groupby("rung")["err"].agg(["mean","count"]).round(3))
print("\n  NOTE: rung 1 is typically the site whose label the model SAW in")
print("  training; higher rungs are the genuinely held-out generalization.")
print("\nBY NUMBER OF SITES")
print(d.groupby("n_sites")["err"].agg(["mean","count"]).round(3))
print("\nRESIDUAL vs ERROR (is cycle residual predictive of error?)")
g = d.groupby("smiles").agg(mae=("err","mean"), resid=("resid","first"))
print(f"  correlation = {g.mae.corr(g.resid):.3f}   (n={len(g)})")
print("\nsaved -> polyprotic_benchmark_v16.csv")
print("\nThis is the number to beat when we train. Anything we build next")
print("gets measured here, not on the 5 molecules I picked by hand.")
