"""Embed the 6,320 distillation candidates to expand the pretraining set
from 8,111 to ~14,400. Pretraining just improved Novartis 0.845->0.830
and AvLiLuMoVe 0.441->0.318, so growing that set is the best-justified
remaining compute. Checkpoints every 100 - safe to interrupt."""
import sys
if "venv311" not in sys.prefix: sys.exit("activate venv311")
import numpy as np, pandas as pd, joblib
from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0,".")
from umapka import PkaPredictor
from umapka.predictor import (ACID_SITES, BASE_SITES, neutralize,
                               _tag_and_reparse, _shift_hydrogen_tagged)

def named_site(mol):
    for n_,sm,ai in ACID_SITES:
        pt=Chem.MolFromSmarts(sm)
        if pt is not None:
            m=mol.GetSubstructMatches(pt)
            if m: return n_,"acid",m[0][ai]
    for n_,sm,ai in BASE_SITES:
        pt=Chem.MolFromSmarts(sm)
        if pt is not None:
            m=mol.GetSubstructMatches(pt)
            if m: return n_,"base",m[0][ai]
    return None,None,None

OUT="feat_distill.pkl"; PARTIAL=OUT+".partial"
df=pd.read_csv("distill_candidates.csv")
print(f"{len(df)} candidates")
try:
    out=joblib.load(PARTIAL); print(f"resuming: {len(out)}")
except FileNotFoundError: out={}

print("loading UMA...")
p=PkaPredictor("models/model_core_v3.pkl")
todo=df[~df.smiles.isin(out.keys())]
print(f"{len(todo)} remaining\n")
n_fail=0
for r in tqdm(todo.itertuples(), total=len(todo)):
    try:
        nm=neutralize(Chem.MolFromSmiles(r.smiles))
        _,kind,idx=named_site(nm)
        if idx is None: n_fail+=1; continue
        if kind=="acid":
            prot,pi_=_tag_and_reparse(nm,idx); dep,di_=_shift_hydrogen_tagged(nm,idx,-1,-1)
        else:
            dep,di_=_tag_and_reparse(nm,idx); prot,pi_=_shift_hydrogen_tagged(nm,idx,+1,+1)
        if prot is None or dep is None: n_fail+=1; continue
        hg_p,hl_p=p.state_features_v4(prot,pi_,kind,n_confs_base=1)
        hg_d,hl_d=p.state_features_v4(dep,di_,kind,n_confs_base=1)
        g=np.concatenate([hg_p,hg_d,hg_p-hg_d]); l=np.concatenate([hl_p,hl_d,hl_p-hl_d])
        out[r.smiles]=np.concatenate([g,l]).astype(np.float32)
    except Exception:
        n_fail+=1
    if len(out)%100==0: joblib.dump(out,PARTIAL)
joblib.dump(out,OUT)
print(f"\ndone: {len(out)} embedded, {n_fail} failed -> {OUT}")
print("\nNEXT: these have NO experimental labels, so label them with v19")
print("(self-distillation), then pretrain on noisy+distilled and finetune.")
