"""External eval using the TRAINED SITE SELECTOR at inference time
instead of the fixed SMARTS priority order.
"""
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, PandasTools
from tqdm import tqdm
from umapka import PkaPredictor
from umapka.predictor import (ACID_SITES, BASE_SITES, neutralize,
                               _tag_and_reparse, _shift_hydrogen_tagged,
                               protonation_pair_site_tagged)

RDLogger.DisableLog("rdApp.*")
MODEL_PATH = "models/model_core_v7_clean.pkl"
SELECTOR_PATH = "models/site_selector.pkl"
OUT_CSV = "characterization_external_v8.csv"

print(f"loading {MODEL_PATH} ...")
bundle = joblib.load(MODEL_PATH)
regressor = bundle["regressor"]
calibrator = bundle.get("calibrator_isotonic") or bundle.get("calibrator")

print(f"loading {SELECTOR_PATH} ...")
sel_bundle = joblib.load(SELECTOR_PATH)
selector = sel_bundle["model"]
ALL_SITE_NAMES = sel_bundle["site_names"]
SITE_RANGES = sel_bundle["site_ranges"]
SITE_INDEX = {n: i for i, n in enumerate(ALL_SITE_NAMES)}

print("loading UMA...")
p = PkaPredictor("models/model_core_v3.pkl")


def candidates(mol):
    out, rank = [], 0
    for name, smarts, ai in ACID_SITES:
        pt = Chem.MolFromSmarts(smarts)
        if pt is not None and mol.GetSubstructMatches(pt):
            out.append((name, "acid", rank, ai, smarts))
        rank += 1
    for name, smarts, ai in BASE_SITES:
        pt = Chem.MolFromSmarts(smarts)
        if pt is not None and mol.GetSubstructMatches(pt):
            out.append((name, "base", rank, ai, smarts))
        rank += 1
    return out


def mol_descriptors(mol):
    return [
        Descriptors.MolWt(mol), Crippen.MolLogP(mol),
        Descriptors.TPSA(mol), Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol), Descriptors.RingCount(mol),
        Descriptors.NumAromaticRings(mol), Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol), float(mol.GetNumAtoms()),
        float(Chem.GetFormalCharge(mol)),
    ]


def featurize(site_name, kind, prio_rank, n_cands, mdesc):
    onehot = [0.0] * len(ALL_SITE_NAMES)
    idx = SITE_INDEX.get(site_name)
    if idx is not None:
        onehot[idx] = 1.0
    rng = SITE_RANGES.get(site_name, (0.0, 14.0))
    return onehot + mdesc + [
        1.0 if kind == "acid" else 0.0,
        float(prio_rank), float(n_cands),
        rng[0], rng[1], (rng[0] + rng[1]) / 2.0,
    ]


def choose_site(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = neutralize(mol)
    cands = candidates(mol)
    if not cands:
        return None
    if len(cands) == 1:
        name, kind, rank, ai, smarts = cands[0]
        m = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return name, kind, m[0][ai], mol
    mdesc = mol_descriptors(mol)
    feats = np.array([featurize(n, k, r, len(cands), mdesc)
                      for n, k, r, ai, s in cands], dtype=float)
    scores = selector.predict_proba(feats)[:, 1]
    best = int(np.argmax(scores))
    name, kind, rank, ai, smarts = cands[best]
    m = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
    return name, kind, m[0][ai], mol


def pair_for_chosen(smiles):
    got = choose_site(smiles)
    if got is None:
        raise RuntimeError("no candidate site")
    name, kind, atom_idx, mol = got
    if kind == "acid":
        prot_smi, prot_idx = _tag_and_reparse(mol, atom_idx)
        dep_smi, dep_idx = _shift_hydrogen_tagged(mol, atom_idx, -1, -1)
    else:
        dep_smi, dep_idx = _tag_and_reparse(mol, atom_idx)
        prot_smi, prot_idx = _shift_hydrogen_tagged(mol, atom_idx, +1, +1)
    if prot_smi is None or dep_smi is None:
        raise RuntimeError("could not build pair at chosen site")
    return prot_smi, prot_idx, dep_smi, dep_idx, kind, name


def build_features(prot, prot_idx, deprot, deprot_idx, kind):
    hg_p, hl_p = p.state_features_v4(prot, prot_idx, kind, n_confs_base=1)
    hg_d, hl_d = p.state_features_v4(deprot, deprot_idx, kind, n_confs_base=1)
    g = np.concatenate([hg_p, hg_d, hg_p - hg_d])
    l = np.concatenate([hl_p, hl_d, hl_p - hl_d])
    return np.concatenate([g, l]).reshape(1, -1)


def site_type(mol):
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
n_changed = 0
for r in tqdm(sets.itertuples(), total=len(sets)):
    try:
        prot, pi_, deprot, di_, kind, chosen = pair_for_chosen(r.smiles)
        feat = build_features(prot, pi_, deprot, di_, kind)
        raw = float(regressor.predict(feat)[0])
        pred = float(calibrator.predict([raw])[0])
    except Exception:
        continue
    try:
        _, _, _, _, old_kind = protonation_pair_site_tagged(r.smiles, return_kind=True)
    except Exception:
        old_kind = None
    mol = Chem.MolFromSmiles(r.smiles)
    changed = (old_kind is not None and old_kind != kind)
    if changed:
        n_changed += 1
    rows.append({
        "dataset": r.dataset, "smiles": r.smiles, "exp": r.exp,
        "pred": pred, "err": abs(pred - r.exp),
        "chosen_site": chosen, "chosen_kind": kind,
        "kind_changed_vs_priority": changed,
        "n_atoms": mol.GetNumAtoms(), "rings": Descriptors.RingCount(mol),
        "site": site_type(neutralize(mol)),
        "pka_bin": pd.cut([r.exp], bins=[0, 4, 7, 10, 14],
                           labels=["<4", "4-7", "7-10", ">10"])[0],
    })

d = pd.DataFrame(rows)
d.to_csv(OUT_CSV, index=False)

print(f"\n=== EXTERNAL v8 = clean training + SELECTOR-chosen sites (n={len(d)}) ===")
print(f"MAE = {d.err.mean():.3f}")
print("  v7-clean (priority sites): 0.809 combined / 0.998 Novartis / 0.385 AvLiLuMoVe")
print("  v4 (best Novartis so far): 0.803 combined / 0.963 Novartis")
print(f"\nacid/base decision CHANGED vs priority on {n_changed} molecules "
      f"({n_changed/max(len(d),1)*100:.1f}%)")

print("\n=== BY DATASET (the number that matters) ===")
print(d.groupby("dataset")["err"].agg(["mean", "count"]).round(3))

print("\n=== DID IT HELP THE MOLECULES IT CHANGED? ===")
print(d.groupby("kind_changed_vs_priority")["err"].agg(["mean", "count"]).round(3))

print("\n=== BY SITE TYPE (watch 'both' - was 0.928 in v7) ===")
print(d.groupby("site")["err"].agg(["mean", "count"]).round(3))

print("\n=== BY pKa RANGE ===")
print(d.groupby("pka_bin", observed=True)["err"].agg(["mean", "count"]).round(3))

d["size_bin"] = pd.cut(d.n_atoms, bins=[0, 15, 22, 30, 1000],
                        labels=["<15", "15-22", "22-30", ">30"])
print("\n=== BY SIZE ===")
print(d.groupby("size_bin", observed=True)["err"].agg(["mean", "count"]).round(3))

print("\n=== MOST-CHOSEN SITES ===")
print(d.chosen_site.value_counts().head(10))
print(f"\nsaved -> {OUT_CSV}")
