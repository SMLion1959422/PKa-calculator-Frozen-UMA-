"""Tests the electronic descriptors AND runs the UMA ablation in one go.

Four models on identical molecules and folds:
  1. UMA embeddings only            (2304-dim) - current approach
  2. Electronic descriptors only    (  81-dim) - cheap RDKit, no UMA
  3. UMA + electronic               (2385-dim) - do they complement?
  4. ECFP4 fingerprint only         (2048-dim) - classical baseline

Model 2 vs 1 is the key scientific result: if 81 cheap RDKit numbers
come close to 2304 UMA dims, the foundation model is adding little for
this task - a genuinely publishable finding about foundation-model
transfer. Model 4 is the standard control.

All use the same hybrid gbm+ridge setup so extrapolation capability is
held constant across arms."""
import sys, numpy as np, pandas as pd, joblib, lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
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
try:
    hunt = joblib.load("feat_hunt.pkl"); print(f"  hunt: {len(hunt)}")
except FileNotFoundError:
    hunt = {}; print("  hunt: not yet built")

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

rows = []
for mol in Chem.ForwardSDMolSupplier("mlpka/datasets/combined_training_datasets_unique.sdf"):
    if mol is None or not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")): continue
    try:
        exp = float(mol.GetProp("pKa")); ma = int(float(mol.GetProp("marvin_atom")))
        smi = Chem.MolToSmiles(mol); nm = neutralize(Chem.Mol(mol))
    except Exception: continue
    if not (0 < exp < 14) or ma >= nm.GetNumAtoms(): continue
    if smi not in elec: continue
    pidx = priority_atom(nm)
    vec = None
    if pidx is not None and pidx == ma and smi in valid:
        vec = f[smi]
    elif smi in corrected:
        vec = corrected[smi]["feat"]; exp = corrected[smi]["pKa"]
    if vec is None: continue
    try:
        fp = np.array(gen.GetFingerprint(Chem.MolFromSmiles(smi)), dtype=float)
    except Exception:
        continue
    rows.append({"smiles": smi, "pKa": exp, "uma": vec, "elec": elec[smi], "fp": fp})

core = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
print(f"\nmolecules in ablation: {len(core)}")

y = core.pKa.values
X_uma  = np.vstack(core.uma.values)
X_elec = np.vstack(core.elec.values)
X_both = np.hstack([X_uma, X_elec])
X_fp   = np.vstack(core.fp.values)
print(f"  UMA {X_uma.shape} | elec {X_elec.shape} | both {X_both.shape} | ECFP4 {X_fp.shape}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(X, tag):
    Xs = StandardScaler().fit_transform(np.nan_to_num(X))
    og = np.zeros(len(y)); orr = np.zeros(len(y))
    for tr, va in kf.split(X):
        g = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               verbose=-1, random_state=42).fit(X[tr], y[tr])
        og[va] = g.predict(X[va])
        r = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Xs[tr], y[tr])
        orr[va] = r.predict(Xs[va])
    bw, bm = 0.0, 1e9
    for w in np.arange(0, 1.01, 0.05):
        m = np.mean(np.abs((1-w)*og + w*orr - y))
        if m < bm: bm, bw = m, w
    blend = (1-bw)*og + bw*orr
    cal = IsotonicRegression(out_of_bounds="clip").fit(blend, y)
    cm = np.mean(np.abs(cal.predict(blend) - y))
    print(f"  {tag:24s} gbm={np.mean(np.abs(og-y)):.3f}  "
          f"blend={bm:.3f} (w={bw:.2f})  calibrated={cm:.3f}")
    return cm

print("\n" + "=" * 68)
print("ABLATION - 5-fold OOF, identical molecules and folds")
print("=" * 68)
r_uma  = evaluate(X_uma,  "1. UMA only (2304)")
r_elec = evaluate(X_elec, "2. electronic only (81)")
r_both = evaluate(X_both, "3. UMA + electronic")
r_fp   = evaluate(X_fp,   "4. ECFP4 only (2048)")

print("\n" + "=" * 68)
print("WHAT THIS MEANS")
print("=" * 68)
print(f"UMA vs 81 cheap RDKit numbers : {r_uma:.3f} vs {r_elec:.3f} "
      f"({(r_elec-r_uma):+.3f})")
print(f"UMA vs classical ECFP4        : {r_uma:.3f} vs {r_fp:.3f} "
      f"({(r_fp-r_uma):+.3f})")
print(f"adding electronic to UMA      : {r_uma:.3f} -> {r_both:.3f} "
      f"({(r_both-r_uma):+.3f})")
print()
if r_elec < r_uma + 0.05:
    print("  >>> 81 cheap descriptors ~MATCH 2304 UMA dims. The foundation")
    print("      model is adding little for this task. That is a real,")
    print("      publishable finding - and it says put effort into")
    print("      descriptors/data, not into UMA.")
else:
    print("  >>> UMA embeddings clearly beat cheap descriptors - the")
    print("      foundation model IS contributing real signal.")
if r_both < r_uma - 0.01:
    print("  >>> electronic descriptors COMPLEMENT UMA. Keep them.")
else:
    print("  >>> electronic descriptors add nothing on top of UMA -")
    print("      the information is already in the embeddings.")

best = min([(r_uma, "uma"), (r_elec, "elec"), (r_both, "both"), (r_fp, "fp")])
print(f"\nbest configuration: {best[1]} at {best[0]:.3f}")
