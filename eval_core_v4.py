"""Mirrors eval_core_v3.py exactly (same two SDF files, same breakdown
tables), but scored with model_core_v4.pkl (multi-scale shell + multi-
conformer base sites) using state_features_v4 directly, since v4 isn't
wired into PkaPredictor.features()'s auto-dimension dispatch (kept that
way on purpose so v2/v3 inference stays completely unaffected).

Watch specifically: does "3+ rings" improve from 0.930 (v3), and does
"base-only" improve from 0.856 (v3)? Those are the two buckets this
was built to fix - if neither moves, the hypothesis was wrong and we
should look elsewhere rather than keep iterating on this approach.
"""
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools
from tqdm import tqdm
from umapka import PkaPredictor
from umapka.predictor import (ACID_SITES, BASE_SITES, neutralize,
                               protonation_pair_site_tagged)

p = PkaPredictor("models/model_core_v2.pkl")  # only using the embedding
                                                # stack + the v4 methods,
                                                # not this file's regressor
bundle = joblib.load("models/model_core_v4.pkl")
regressor, calibrator = bundle["regressor"], bundle["calibrator"]

def build_features_v4(smiles):
    prot, prot_idx, deprot, deprot_idx, kind = protonation_pair_site_tagged(
        smiles, return_kind=True)
    hg_p, hl_p = p.state_features_v4(prot, prot_idx, kind)
    hg_d, hl_d = p.state_features_v4(deprot, deprot_idx, kind)
    global_feat = np.concatenate([hg_p, hg_d, hg_p - hg_d])
    local_feat = np.concatenate([hl_p, hl_d, hl_p - hl_d])
    return np.concatenate([global_feat, local_feat]).reshape(1, -1)

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
        feat = build_features_v4(r.smiles)
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
d.to_csv("characterization_external_v4.csv", index=False)

print(f"\n=== EXTERNAL OVERALL v4 (n={len(d)}) MAE={d.err.mean():.3f} ===")
print(f"    (v3 was 0.832 combined; v2 was 1.16-1.17 Novartis-only per RESULTS.md)\n")
print("=== BY DATASET (the real comparison) ===")
print(d.groupby("dataset")["err"].agg(["mean","count"]).round(3))
print("    (v3: novartis=1.018, avlilumove=0.421 - watch for movement here)\n")
print("=== BY pKa RANGE ==="); print(d.groupby("pka_bin", observed=True)["err"].agg(["mean","count"]).round(3), "\n")
print("=== BY SITE TYPE (watch base-only vs v3's 0.856) ===")
print(d.groupby("site")["err"].agg(["mean","count"]).round(3), "\n")
d["size_bin"] = pd.cut(d["n_atoms"], bins=[0,15,22,30,100], labels=["<15","15-22","22-30",">30"])
print("=== BY SIZE ==="); print(d.groupby("size_bin", observed=True)["err"].agg(["mean","count"]).round(3), "\n")
d["ring_bin"] = pd.cut(d["rings"], bins=[-1,0,1,2,10], labels=["0","1","2","3+"])
print("=== BY RINGS (watch 3+ vs v3's 0.930) ===")
print(d.groupby("ring_bin", observed=True)["err"].agg(["mean","count"]).round(3))
