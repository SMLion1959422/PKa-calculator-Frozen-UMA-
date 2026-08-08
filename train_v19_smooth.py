"""v19 = v16 recipe + noisy-site pretraining (arm C, best at 0.486 OOF).

Pretrain on 8,111 experimental labels whose sites come from SMARTS
matching (~79-93% correct), then fine-tune on the 5,184 with Marvin
ground-truth sites. Both sets were ALREADY embedded - zero new UMA cost.

Uses an MLP because pretrain->finetune requires warm-starting, which
LightGBM cannot do. That is itself a change from v16, so this script
also trains an MLP-scratch control: if v19 beats v16 externally we need
to know whether it was the pretraining or just the architecture swap."""
import sys
if "venv311" not in sys.prefix: sys.exit("activate venv311")
import numpy as np, pandas as pd, joblib, torch, torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.EState import EStateIndices
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka.predictor import (ACID_SITES, BASE_SITES, neutralize,
                               _tag_and_reparse, _shift_hydrogen_tagged)

def named_site(mol):
    for n_, sm, ai in ACID_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m: return n_, "acid", m[0][ai]
    for n_, sm, ai in BASE_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m: return n_, "base", m[0][ai]
    return None, None, None

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
    return np.array([q[idx],est[idx],
        q[s1].mean(),q[s1].min(),q[s1].max(),est[s1].mean(),
        q[s2].mean(),q[s2].min(),q[s2].max(),est[s2].mean(),
        q[s3].mean(),q[s3].min(),q[s3].max(),est[s3].mean(),
        q.mean(),q.min(),q.max(),q.std(),
        float(a.GetDegree()),float(a.GetTotalNumHs()),float(a.GetFormalCharge()),
        float(a.GetIsAromatic()),float(a.IsInRing()),float(a.GetAtomicNum()),
        Descriptors.TPSA(mol),Crippen.MolLogP(mol),float(Chem.GetFormalCharge(mol))],
        dtype=float)

f6 = joblib.load("feat_train_v6.pkl")
elec = joblib.load("feat_electronic.pkl")
corrected = joblib.load("feat_marvin_corrected.pkl")
valid6 = {s for s,v in f6.items() if np.asarray(v).shape==(2304,)}

Xc, yc = [], []
for mol in Chem.ForwardSDMolSupplier("mlpka/datasets/combined_training_datasets_unique.sdf"):
    if mol is None or not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")): continue
    try:
        exp=float(mol.GetProp("pKa")); ma=int(float(mol.GetProp("marvin_atom")))
        smi=Chem.MolToSmiles(mol); nm=neutralize(Chem.Mol(mol))
    except Exception: continue
    if not (0<exp<14) or ma>=nm.GetNumAtoms() or smi not in elec: continue
    _,_,pidx = named_site(nm); vec=None
    if pidx is not None and pidx==ma and smi in valid6: vec=f6[smi]
    elif smi in corrected: vec=corrected[smi]["feat"]; exp=corrected[smi]["pKa"]
    if vec is None: continue
    Xc.append(np.concatenate([vec, elec[smi]])); yc.append(exp)
Xc=np.nan_to_num(np.vstack(Xc)).astype(np.float32); yc=np.array(yc,dtype=np.float32)
print(f"clean: {len(yc)}")

Xn, yn = [], []
extra = pd.read_csv("extra_pka_data.csv")
for r in tqdm(extra.itertuples(), total=len(extra), desc="noisy set"):
    smi=r.smiles
    if smi not in valid6: continue
    mol=Chem.MolFromSmiles(smi)
    if mol is None: continue
    nm=neutralize(mol); _,kind,idx = named_site(nm)
    if idx is None: continue
    try:
        if kind=="acid":
            prot,pi_=_tag_and_reparse(nm,idx); dep,di_=_shift_hydrogen_tagged(nm,idx,-1,-1)
        else:
            dep,di_=_tag_and_reparse(nm,idx); prot,pi_=_shift_hydrogen_tagged(nm,idx,+1,+1)
        if prot is None or dep is None: continue
        dp=elec_desc(prot,pi_); dd=elec_desc(dep,di_)
        if dp is None or dd is None: continue
    except Exception: continue
    Xn.append(np.concatenate([f6[smi],dp,dd,dp-dd])); yn.append(float(r.pKa))
Xn=np.nan_to_num(np.vstack(Xn)).astype(np.float32); yn=np.array(yn,dtype=np.float32)
print(f"noisy: {len(yn)}")

DIM=Xc.shape[1]
sc=StandardScaler().fit(np.vstack([Xc,Xn]))
Xct=torch.tensor(sc.transform(Xc),dtype=torch.float32)
Xnt=torch.tensor(sc.transform(Xn),dtype=torch.float32)

class Head(nn.Module):
    def __init__(s,d,h=512):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.2),
                            nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h//2,1))
    def forward(s,x): return s.net(x).squeeze(-1)

def fit(model,X,y,epochs,lr):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); l1=nn.MSELoss()
    yt=torch.tensor(y)
    for _ in range(epochs):
        model.train(); idx=torch.randperm(len(yt))
        for i in range(0,len(yt),256):
            b=idx[i:i+256]; opt.zero_grad(); l1(model(X[b]),yt[b]).backward(); opt.step()
    return model

print("\nOOF for calibration + the MLP-scratch control...")
kf=KFold(5,shuffle=True,random_state=42)
oof=np.zeros(len(yc)); oof_ctl=np.zeros(len(yc))
for tr,va in kf.split(Xct):
    torch.manual_seed(42)
    m=fit(fit(Head(DIM),Xnt,yn,80,1e-3),Xct[tr],yc[tr],120,3e-4); m.eval()
    with torch.no_grad(): oof[va]=m(Xct[va]).numpy()
    torch.manual_seed(42)
    c=fit(Head(DIM),Xct[tr],yc[tr],120,1e-3); c.eval()
    with torch.no_grad(): oof_ctl[va]=c(Xct[va]).numpy()
cal=Ridge(alpha=1.0).fit(oof.reshape(-1, 1), yc)
print(f"  v19 pretrained OOF calibrated : {np.mean(np.abs(cal.predict(oof)-yc)):.3f}")
print(f"  MLP-scratch control OOF       : {np.mean(np.abs(oof_ctl-yc)):.3f}")
print(f"  (v16 LightGBM OOF was 0.485 - control tells us if any external")
print(f"   change is from pretraining or just the MLP swap)")

print("\ntraining final...")
torch.manual_seed(42)
final=fit(fit(Head(DIM),Xnt,yn,80,1e-3),Xct,yc,120,3e-4); final.eval()
joblib.dump({"state_dict":{k:v.numpy() for k,v in final.state_dict().items()},
             "scaler":sc,"dim":DIM,"hidden":512,"calibrator":cal},
            "models/model_core_v19_pretrained.pkl")
print("saved -> models/model_core_v19_pretrained.pkl")
