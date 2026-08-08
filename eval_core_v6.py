"""External evaluation for v6 (multi-scale shell, expanded dataset,
single conformer). Reports BOTH isotonic and linear calibration
side-by-side, plus the standard by-dataset/by-size/by-rings/by-pKa-range
breakdowns, so the calibration question is answered in the same run
that gives the headline number.
"""
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools
from tqdm import tqdm
from umapka import PkaPredictor
from umapka.predictor import ACID_SITES, BASE_SITES, neutralize, protonation_pair_site_tagged

MODEL_PATH = "models/model_core_v6.pkl"
p = PkaPredictor("models/model_core_v2.pkl")  # embedding stack only
bundle = joblib.load(MODEL_PATH)
regressor = bundle["regressor"]
cal_iso = bundle["calibrator_isotonic"]
cal_lin = bundle["calibrator_linear"]

def build_features_v6(smiles):
    prot, prot_idx, deprot, deprot_idx, kind = protonation_pair_site_tagged(
        smiles, return_kind=True)
    hg_p, hl_p = p.state_features_v4(prot, prot_idx, kind, n_confs_base=1)
    hg_d, hl_d = p.state_features_v4(deprot, deprot_idx, kind, n_confs_base=1)
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
        feat = build_features_v6(r.smiles)
        raw = float(regressor.predict(feat)[0])
        pred_iso = float(cal_iso.predict([raw])[0])
        pred_lin = float(cal_lin.predict([[raw]])[0])
    except Exception:
        continue
    mol = Chem.MolFromSmiles(r.smiles)
    rows.append({"dataset": r.dataset, "exp": r.exp,
                 "pred_iso": pred_iso, "err_iso": abs(pred_iso-r.exp),
                 "pred_lin": pred_lin, "err_lin": abs(pred_lin-r.exp),
                 "n_atoms": mol.GetNumAtoms(), "rings": Descriptors.RingCount(mol),
                 "site": site_type(r.smiles),
                 "pka_bin": pd.cut([r.exp], bins=[0,4,7,10,14], labels=["<4","4-7","7-10",">10"])[0]})

d = pd.DataFrame(rows)
d.to_csv("characterization_external_v6.csv", index=False)

print(f"\n=== EXTERNAL OVERALL v6 (n={len(d)}) ===")
print(f"isotonic calibration: MAE={d.err_iso.mean():.3f}")
print(f"linear calibration:   MAE={d.err_lin.mean():.3f}")
print(f"(compare: v4 was 0.803 combined / 0.963 Novartis-only - the best result so far)")
print(f"          v3 was 0.832 combined / 1.018 Novartis-only")

print(f"\n=== BY DATASET (use whichever calibration won above) ===")
print(d.groupby("dataset")[["err_iso","err_lin"]].agg(["mean","count"]).round(3))

print(f"\n=== BY pKa RANGE - THE KEY CALIBRATION DIAGNOSTIC ===")
print(d.groupby("pka_bin", observed=True)[["err_iso","err_lin"]].agg(["mean","count"]).round(3))
print("(if err_lin is meaningfully better than err_iso specifically in the")
print(" >10 bucket, that confirms isotonic instability was part of tonight's")
print(" earlier problem, and linear calibration should become the default)")

print(f"\n=== BY SITE TYPE ===")
print(d.groupby("site")[["err_iso","err_lin"]].agg(["mean","count"]).round(3))

d["size_bin"] = pd.cut(d["n_atoms"], bins=[0,15,22,30,100], labels=["<15","15-22","22-30",">30"])
print(f"\n=== BY SIZE ===")
print(d.groupby("size_bin", observed=True)[["err_iso","err_lin"]].agg(["mean","count"]).round(3))

d["ring_bin"] = pd.cut(d["rings"], bins=[-1,0,1,2,10], labels=["0","1","2","3+"])
print(f"\n=== BY RINGS ===")
print(d.groupby("ring_bin", observed=True)[["err_iso","err_lin"]].agg(["mean","count"]).round(3))
