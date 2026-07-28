import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools
from tqdm import tqdm
from umapka import PkaPredictor, protonation_pair
from umapka.predictor import ACID_SITES, BASE_SITES, neutralize

p = PkaPredictor("models/model_core.pkl")
bundle = joblib.load("models/model_core_v2.pkl")
regressor, calibrator = bundle["regressor"], bundle["calibrator"]

def site_type(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return "unparseable"
    mol = neutralize(mol)
    acid = any(mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for _,s,_ in ACID_SITES if Chem.MolFromSmarts(s))
    base = any(mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for _,s,_ in BASE_SITES if Chem.MolFromSmarts(s))
    if acid and base: return "both"
    if acid: return "acid-only"
    if base: return "base-only"
    return "neither"

def load_set(path, name):
    df = PandasTools.LoadSDF(path)
    pk_col = next(c for c in df.columns if c.lower() in ("pka","pka_value","value"))
    out = []
    for _, r in df.iterrows():
        m = r.get("ROMol")
        if m is None: continue
        try: v = float(r[pk_col])
        except: continue
        if not (0 < v < 14): continue
        try: out.append({"dataset": name, "smiles": Chem.MolToSmiles(m), "exp": v})
        except: pass
    return pd.DataFrame(out).drop_duplicates("smiles")

sets = pd.concat([
    load_set("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf", "novartis"),
    load_set("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf", "avlilumove"),
]).reset_index(drop=True)

rows = []
for r in tqdm(sets.itertuples(), total=len(sets)):
    try:
        prot, deprot = protonation_pair(r.smiles)
        feat = p.features(prot, deprot)
        raw = float(regressor.predict(feat)[0])
        pred = float(calibrator.predict([raw])[0])
    except Exception:
        continue
    mol = Chem.MolFromSmiles(r.smiles)
    rows.append({"dataset": r.dataset, "exp": r.exp, "pred": pred, "err": abs(pred-r.exp),
                 "n_atoms": mol.GetNumAtoms(), "rings": Descriptors.RingCount(mol),
                 "site": site_type(r.smiles),
                 "pka_bin": pd.cut([r.exp], bins=[0,4,7,10,14], labels=["<4","4-7","7-10",">10"])[0]})

d = pd.DataFrame(rows)
d.to_csv("characterization_external.csv", index=False)

print(f"\n=== EXTERNAL OVERALL (n={len(d)}) MAE={d.err.mean():.3f} ===\n")
print("=== BY pKa RANGE ==="); print(d.groupby("pka_bin", observed=True)["err"].agg(["mean","count"]).round(3), "\n")
print("=== BY SITE TYPE ==="); print(d.groupby("site")["err"].agg(["mean","count"]).round(3), "\n")
d["size_bin"] = pd.cut(d["n_atoms"], bins=[0,15,22,30,100], labels=["<15","15-22","22-30",">30"])
print("=== BY SIZE ==="); print(d.groupby("size_bin", observed=True)["err"].agg(["mean","count"]).round(3), "\n")
d["ring_bin"] = pd.cut(d["rings"], bins=[-1,0,1,2,10], labels=["0","1","2","3+"])
print("=== BY RINGS ==="); print(d.groupby("ring_bin", observed=True)["err"].agg(["mean","count"]).round(3))
