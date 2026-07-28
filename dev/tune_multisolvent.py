import numpy as np, pandas as pd, joblib, itertools, time
from sklearn.model_selection import KFold
import lightgbm as lgb
from tqdm import tqdm

SOLVENTS = {
    "O":(1.00,1.0,"Water"),"CS(C)=O":(0.60,0.0,"DMSO"),"CC#N":(0.48,0.0,"Acetonitrile"),
    "CN(C)C=O":(0.47,0.0,"DMF"),"CO":(0.42,0.5,"Methanol"),"CCO":(0.31,0.5,"Ethanol"),
    "CN1CCCC1=O":(0.42,0.0,"NMP"),"C(CO)O":(0.51,0.5,"EthyleneGlycol"),
}
feat_cache = joblib.load("solvent_feat_cache.pkl")

def split_rxn(r):
    try: a,b=r.split(">>"); return a.strip(),b.strip()
    except: return None,None

def build(csv):
    df=pd.read_csv(csv)
    feats,svecs,pkas,sols=[],[],[],[]
    for _,r in df.iterrows():
        sol=r["solvent_smiles"]
        if sol not in SOLVENTS: continue
        pka=r["pKa_avg"]
        if not (-15<pka<45): continue
        a,b=split_rxn(r["reaction_smiles"])
        if not a or not b: continue
        fa,fb=feat_cache.get(a),feat_cache.get(b)
        if fa is None or fb is None: continue
        eps,protic,name=SOLVENTS[sol]
        feats.append(np.concatenate([fa,fb,fa-fb])); svecs.append([eps,protic])
        pkas.append(pka); sols.append(name)
    return np.hstack([np.array(feats),np.array(svecs)]), np.array(pkas), np.array(sols)

print("building feature matrices...")
X, y, sols = build(r"anion_data\data\D2A-pKa.csv")
Xtr, ytr, _ = build(r"anion_data\data\data_splits\D2A-pKa-train.csv")
Xte, yte, ste = build(r"anion_data\data\data_splits\D2A-pKa-test.csv")
print(f"full={len(y)}  their_train={len(ytr)}  their_test={len(yte)}\n")

grid = {"n_estimators":[300,600,1000],"num_leaves":[31,63,127],
        "learning_rate":[0.03,0.05],"min_child_samples":[20,50]}
combos=list(itertools.product(*grid.values()))
kf=KFold(5,shuffle=True,random_state=42)
results=[]

# progress bar over (config x fold) = total fits, with live time-remaining
total_fits=len(combos)*5
with tqdm(total=total_fits, desc="grid search", unit="fit") as pbar:
    for n_est,leaves,lr,mcs in combos:
        oof=np.zeros(len(y))
        for tr,va in kf.split(X):
            m=lgb.LGBMRegressor(n_estimators=n_est,num_leaves=leaves,learning_rate=lr,
                                min_child_samples=mcs,verbose=-1,random_state=42)
            m.fit(X[tr],y[tr]); oof[va]=m.predict(X[va])
            pbar.update(1)
        results.append({"n_estimators":n_est,"num_leaves":leaves,"learning_rate":lr,
                        "min_child_samples":mcs,"cv_mae":np.abs(oof-y).mean()})

res=pd.DataFrame(results).sort_values("cv_mae")
print("\nTOP 5 CONFIGS:"); print(res.head(5).to_string(index=False))
best=res.iloc[0]

bm=lgb.LGBMRegressor(n_estimators=int(best.n_estimators),num_leaves=int(best.num_leaves),
                     learning_rate=best.learning_rate,min_child_samples=int(best.min_child_samples),
                     verbose=-1,random_state=42)
bm.fit(Xtr,ytr); pred=bm.predict(Xte)
print(f"\n=== HELD-OUT TEST (their split, best config) overall MAE: {np.abs(pred-yte).mean():.3f} ===")
print(f"{'solvent':<16}{'MAE':>8}{'n':>6}")
for s in sorted(set(ste)):
    mask=ste==s
    if mask.sum()==0: continue
    print(f"{s:<16}{np.abs(pred[mask]-yte[mask]).mean():>8.2f}{mask.sum():>6}")

base=lgb.LGBMRegressor(n_estimators=300,num_leaves=31,learning_rate=0.05,verbose=-1,random_state=42)
base.fit(Xtr,ytr)
print(f"\nuntuned baseline test MAE: {np.abs(base.predict(Xte)-yte).mean():.3f}")
print(f"tuned test MAE:            {np.abs(pred-yte).mean():.3f}")

joblib.dump({"model":bm,"solvents":SOLVENTS,"config":dict(best)},"models/multisolvent_tuned.pkl")
print("saved -> models/multisolvent_tuned.pkl")
