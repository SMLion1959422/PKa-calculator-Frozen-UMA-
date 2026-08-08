"""Three-way ablation: does real GFN2-xTB beat the Gasteiger fallback,
and do they stack?

  1. UMA only                     - 0.537 OOF baseline
  2. UMA + Gasteiger/EState       - 0.485 (current best, beat Marvin)
  3. UMA + xTB                    - real quantum descriptors
  4. UMA + Gasteiger + xTB        - do they complement?
  5. xTB only                     - how much does UMA still contribute?

Same molecules, same folds, same hybrid gbm+ridge head throughout, so
differences are attributable to features alone."""
import sys, numpy as np, pandas as pd, joblib, lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka.predictor import ACID_SITES, BASE_SITES, neutralize

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

print("loading caches...")
f = joblib.load("feat_train_v6.pkl")
valid = {s for s, v in f.items() if np.asarray(v).shape == (2304,)}
corrected = joblib.load("feat_marvin_corrected.pkl")
elec = joblib.load("feat_electronic.pkl")
xtb = joblib.load("feat_xtb.pkl")
print(f"  UMA={len(valid)}  gasteiger={len(elec)}  xtb={len(xtb)}")

rows = []
for mol in Chem.ForwardSDMolSupplier("mlpka/datasets/combined_training_datasets_unique.sdf"):
    if mol is None or not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")): continue
    try:
        exp = float(mol.GetProp("pKa")); ma = int(float(mol.GetProp("marvin_atom")))
        smi = Chem.MolToSmiles(mol); nm = neutralize(Chem.Mol(mol))
    except Exception: continue
    if not (0 < exp < 14) or ma >= nm.GetNumAtoms(): continue
    if smi not in elec or smi not in xtb: continue
    pidx = priority_atom(nm)
    vec = None
    if pidx is not None and pidx == ma and smi in valid:
        vec = f[smi]
    elif smi in corrected:
        vec = corrected[smi]["feat"]; exp = corrected[smi]["pKa"]
    if vec is None: continue
    rows.append({"smiles": smi, "pKa": exp, "uma": vec,
                 "elec": elec[smi], "xtb": xtb[smi]})

core = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
print(f"\nmolecules with ALL feature types: {len(core)}")
y = core.pKa.values
U = np.vstack(core.uma.values)
E = np.vstack(core.elec.values)
X = np.vstack(core.xtb.values)
print(f"  UMA {U.shape[1]}d | gasteiger {E.shape[1]}d | xtb {X.shape[1]}d")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
def ev(M, tag):
    M = np.nan_to_num(M)
    Ms = StandardScaler().fit_transform(M)
    og = np.zeros(len(y)); orr = np.zeros(len(y))
    for tr, va in kf.split(M):
        g = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               verbose=-1, random_state=42).fit(M[tr], y[tr])
        og[va] = g.predict(M[va])
        r = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Ms[tr], y[tr])
        orr[va] = r.predict(Ms[va])
    bw, bm = 0.0, 1e9
    for w in np.arange(0, 1.01, 0.05):
        m = np.mean(np.abs((1-w)*og + w*orr - y))
        if m < bm: bm, bw = m, w
    bl = (1-bw)*og + bw*orr
    cal = IsotonicRegression(out_of_bounds="clip").fit(bl, y)
    cm = np.mean(np.abs(cal.predict(bl) - y))
    print(f"  {tag:28s} blend={bm:.3f} (w={bw:.2f})  calibrated={cm:.3f}")
    return cm

print("\n" + "=" * 66)
print("ABLATION - identical molecules and folds")
print("=" * 66)
r1 = ev(U,                        "1. UMA only")
r2 = ev(np.hstack([U, E]),        "2. UMA + gasteiger")
r3 = ev(np.hstack([U, X]),        "3. UMA + xTB")
r4 = ev(np.hstack([U, E, X]),     "4. UMA + gasteiger + xTB")
r5 = ev(X,                        "5. xTB only (no UMA)")

print("\n" + "=" * 66)
best = min([(r1,"UMA"),(r2,"UMA+gast"),(r3,"UMA+xtb"),(r4,"UMA+both"),(r5,"xtb only")])
print(f"xTB vs gasteiger:       {r3:.3f} vs {r2:.3f} ({r3-r2:+.3f})")
print(f"stacking both:          {r4:.3f} ({r4-min(r2,r3):+.3f} vs better single)")
print(f"UMA's contribution:     {r5:.3f} (xtb alone) -> {r3:.3f} (with UMA) = {r5-r3:+.3f}")
print(f"\nBEST: {best[1]} at {best[0]:.3f}")

# save the winner for external eval
combos = {"UMA": U, "UMA+gast": np.hstack([U,E]), "UMA+xtb": np.hstack([U,X]),
          "UMA+both": np.hstack([U,E,X]), "xtb only": X}
M = np.nan_to_num(combos[best[1]])
sc = StandardScaler().fit(M); Ms = sc.transform(M)
og = np.zeros(len(y)); orr = np.zeros(len(y))
for tr, va in kf.split(M):
    og[va] = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                verbose=-1, random_state=42).fit(M[tr], y[tr]).predict(M[va])
    orr[va] = RidgeCV(alphas=np.logspace(-2,4,25)).fit(Ms[tr], y[tr]).predict(Ms[va])
bw, bm = 0.0, 1e9
for w in np.arange(0, 1.01, 0.05):
    m = np.mean(np.abs((1-w)*og + w*orr - y))
    if m < bm: bm, bw = m, w
cal = IsotonicRegression(out_of_bounds="clip").fit((1-bw)*og + bw*orr, y)
gf = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                        verbose=-1, random_state=42).fit(M, y)
rf = RidgeCV(alphas=np.logspace(-2,4,25)).fit(Ms, y)
joblib.dump({"gbm": gf, "ridge": rf, "scaler": sc, "blend_w": bw,
             "calibrator": cal, "combo": best[1]},
            "models/model_core_v17_best.pkl")
print(f"saved best combo -> models/model_core_v17_best.pkl (combo={best[1]})")
