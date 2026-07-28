import numpy as np, pandas as pd, joblib
from sklearn.model_selection import KFold
import lightgbm as lgb

SOLVENTS = {
    "O":(1.00,1.0,"Water"),"CS(C)=O":(0.60,0.0,"DMSO"),"CC#N":(0.48,0.0,"Acetonitrile"),
    "CN(C)C=O":(0.47,0.0,"DMF"),"CO":(0.42,0.5,"Methanol"),"CCO":(0.31,0.5,"Ethanol"),
    "CN1CCCC1=O":(0.42,0.0,"NMP"),"C(CO)O":(0.51,0.5,"EthyleneGlycol"),
}
feat_cache = joblib.load("solvent_feat_cache.pkl")
df = pd.read_csv(r"anion_data\data\D2A-pKa.csv")

def split_rxn(r):
    try: a,b=r.split(">>"); return a.strip(),b.strip()
    except: return None,None

# build arrays directly in plain python lists - NO pandas iterrows on array-cells
feats, svecs, pkas, solvents = [], [], [], []
for _, r in df.iterrows():
    sol=r["solvent_smiles"]
    if sol not in SOLVENTS: continue
    pka=r["pKa_avg"]
    if not (-15<pka<45): continue
    a,b=split_rxn(r["reaction_smiles"])
    if not a or not b: continue
    fa,fb=feat_cache.get(a),feat_cache.get(b)
    if fa is None or fb is None: continue
    eps,protic,name=SOLVENTS[sol]
    feats.append(np.concatenate([fa,fb,fa-fb]))
    svecs.append([eps,protic]); pkas.append(pka); solvents.append(name)

X = np.hstack([np.array(feats), np.array(svecs)])   # fast: vectorized, no iterrows
y = np.array(pkas)
solvents = np.array(solvents)
print(f"usable: {len(y)}   feature dim: {X.shape[1]}")

def mk(): return lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05, verbose=-1, random_state=42)

print("\n=== RANDOM 5-FOLD ===")
kf=KFold(5,shuffle=True,random_state=42)
pred=np.zeros(len(y))
for tr,te in kf.split(X):
    m=mk(); m.fit(X[tr],y[tr]); pred[te]=m.predict(X[te])
print(f"{'solvent':<16}{'MAE':>8}{'n':>6}")
for s in sorted(set(solvents)):
    mask=solvents==s
    print(f"{s:<16}{np.abs(pred[mask]-y[mask]).mean():>8.2f}{mask.sum():>6}")
print(f"OVERALL: {np.abs(pred-y).mean():.2f}")

print("\n=== LEAVE-ONE-SOLVENT-OUT ===")
print(f"{'held-out':<16}{'MAE':>8}{'n':>6}")
for held in sorted(set(solvents)):
    tr=solvents!=held; te=solvents==held
    if tr.sum()<50 or te.sum()<10: continue
    m=mk(); m.fit(X[tr],y[tr]); ph=m.predict(X[te])
    print(f"{held:<16}{np.abs(ph-y[te]).mean():>8.2f}{te.sum():>6}")

final=mk(); final.fit(X,y)
joblib.dump({"model":final,"solvents":SOLVENTS},"models/multisolvent_model.pkl")
print("\nsaved -> models/multisolvent_model.pkl")
