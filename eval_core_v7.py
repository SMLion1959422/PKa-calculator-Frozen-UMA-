"""External eval for model_core_v7_clean.pkl. Path hardcoded."""
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, PandasTools
from tqdm import tqdm
from umapka import PkaPredictor
from umapka.predictor import (ACID_SITES, BASE_SITES, neutralize,
                               protonation_pair_site_tagged)

RDLogger.DisableLog("rdApp.*")
MODEL_PATH = "models/model_core_v7_clean.pkl"
OUT_CSV = "characterization_external_v7.csv"

print(f"loading {MODEL_PATH} ...")
bundle = joblib.load(MODEL_PATH)
regressor = bundle["regressor"]
cal_iso = bundle.get("calibrator_isotonic") or bundle.get("calibrator")
cal_lin = bundle.get("calibrator_linear") or cal_iso

print("loading UMA...")
p = PkaPredictor("models/model_core_v3.pkl")


def build_features(smiles):
    prot, prot_idx, deprot, deprot_idx, kind = protonation_pair_site_tagged(
        smiles, return_kind=True)
    hg_p, hl_p = p.state_features_v4(prot, prot_idx, kind, n_confs_base=1)
    hg_d, hl_d = p.state_features_v4(deprot, deprot_idx, kind, n_confs_base=1)
    g = np.concatenate([hg_p, hg_d, hg_p - hg_d])
    l = np.concatenate([hl_p, hl_d, hl_p - hl_d])
    return np.concatenate([g, l]).reshape(1, -1)


def site_type(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "unparseable"
    mol = neutralize(mol)
    a = any(mol.HasSubstructMatch(Chem.MolFromSmarts(s))
            for _, s, _ in ACID_SITES if Chem.MolFromSmarts(s))
    b = any(mol.HasSubstructMatch(Chem.MolFromSmarts(s))
            for _, s, _ in BASE_SITES if Chem.MolFromSmarts(s))
    if a and b:
        return "both"
    if a:
        return "acid-only"
    if b:
        return "base-only"
    return "neither"


def load_set(path, name):
    df = PandasTools.LoadSDF(path)
    pk = next(c for c in df.columns if c.lower() in ("pka", "pka_value", "value"))
    out = []
    for _, r in df.iterrows():
        m = r.get("ROMol")
        if m is None:
            continue
        try:
            v = float(r[pk])
        except Exception:
            continue
        if not (0 < v < 14):
            continue
        try:
            out.append({"dataset": name, "smiles": Chem.MolToSmiles(m), "exp": v})
        except Exception:
            pass
    return pd.DataFrame(out).drop_duplicates("smiles")


sets = pd.concat([
    load_set("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf", "novartis"),
    load_set("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf", "avlilumove"),
]).reset_index(drop=True)

rows = []
for r in tqdm(sets.itertuples(), total=len(sets)):
    try:
        feat = build_features(r.smiles)
        raw = float(regressor.predict(feat)[0])
        pi = float(cal_iso.predict([raw])[0])
        pl = float(cal_lin.predict([raw])[0])
    except Exception:
        continue
    mol = Chem.MolFromSmiles(r.smiles)
    rows.append({
        "dataset": r.dataset, "smiles": r.smiles, "exp": r.exp,
        "pred_iso": pi, "err_iso": abs(pi - r.exp),
        "pred_lin": pl, "err_lin": abs(pl - r.exp),
        "n_atoms": mol.GetNumAtoms(), "rings": Descriptors.RingCount(mol),
        "site": site_type(r.smiles),
        "pka_bin": pd.cut([r.exp], bins=[0, 4, 7, 10, 14],
                           labels=["<4", "4-7", "7-10", ">10"])[0],
    })

d = pd.DataFrame(rows)
d.to_csv(OUT_CSV, index=False)

print(f"\n=== EXTERNAL OVERALL v7-clean (n={len(d)}) ===")
print(f"isotonic: MAE={d.err_iso.mean():.3f}   linear: MAE={d.err_lin.mean():.3f}")
print("  v6 was 0.848 combined / 1.077 Novartis / 0.334 AvLiLuMoVe")
print("  v4 was 0.803 combined / 0.963 Novartis  <-- best Novartis so far")

print("\n=== BY DATASET (the number that matters) ===")
print(d.groupby("dataset")[["err_iso", "err_lin"]].agg(["mean", "count"]).round(3))

print("\n=== BY pKa RANGE ===")
print(d.groupby("pka_bin", observed=True)[["err_iso", "err_lin"]].mean().round(3))

print("\n=== BY SITE TYPE ===")
print(d.groupby("site")[["err_iso", "err_lin"]].agg(["mean", "count"]).round(3))

d["size_bin"] = pd.cut(d.n_atoms, bins=[0, 15, 22, 30, 1000],
                        labels=["<15", "15-22", "22-30", ">30"])
print("\n=== BY SIZE ===")
print(d.groupby("size_bin", observed=True)[["err_iso", "err_lin"]].agg(["mean", "count"]).round(3))

d["ring_bin"] = pd.cut(d.rings, bins=[-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"])
print("\n=== BY RINGS ===")
print(d.groupby("ring_bin", observed=True)[["err_iso", "err_lin"]].mean().round(3))
print(f"\nsaved -> {OUT_CSV}")
