"""Mirrors your existing characterize_external.py exactly (same two SDF
files, same MAE/by-size/by-rings/by-site breakdown), but scored with
model_core_v3.pkl instead of model_core_v2.pkl - the direct, like-for-
like comparison against the 1.16-1.17 Novartis MAE reported in
RESULTS.md/README. The by-size breakdown is the important one: if
site-local pooling is doing what it's supposed to, the >30-atom bucket
should improve the most (that's where global-pooling dilution was
worst).
"""
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools
from tqdm import tqdm
from umapka import PkaPredictor
from umapka.predictor import ACID_SITES, BASE_SITES, neutralize, protonation_pair_site_tagged

p = PkaPredictor("models/model_core_v3.pkl")  # auto-activates site-local
                                                # features: this regressor
                                                # is 1536-dim
bundle = joblib.load("models/model_core_v3.pkl")
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
        prot, prot_idx, deprot, deprot_idx = protonation_pair_site_tagged(r.smiles)
        feat = p.features(prot, deprot, prot_idx, deprot_idx)
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
d.to_csv("characterization_external_v3.csv", index=False)

print(f"\n=== EXTERNAL OVERALL v3 (n={len(d)}) MAE={d.err.mean():.3f} ===")
print(f"    (compare: v2/global-only was 1.16-1.17 per RESULTS.md/README)\n")
print("=== BY pKa RANGE ==="); print(d.groupby("pka_bin", observed=True)["err"].agg(["mean","count"]).round(3), "\n")
print("=== BY SITE TYPE ==="); print(d.groupby("site")["err"].agg(["mean","count"]).round(3), "\n")
d["size_bin"] = pd.cut(d["n_atoms"], bins=[0,15,22,30,100], labels=["<15","15-22","22-30",">30"])
print("=== BY SIZE (the key diagnostic - see docstring) ===")
print(d.groupby("size_bin", observed=True)["err"].agg(["mean","count"]).round(3), "\n")
d["ring_bin"] = pd.cut(d["rings"], bins=[-1,0,1,2,10], labels=["0","1","2","3+"])
print("=== BY RINGS ==="); print(d.groupby("ring_bin", observed=True)["err"].agg(["mean","count"]).round(3))
