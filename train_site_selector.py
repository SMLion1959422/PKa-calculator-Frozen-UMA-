"""Trains a PROTONATION-SITE SELECTOR to replace the hand-ordered
SMARTS priority list at inference time. RDKit + LightGBM only - no UMA,
no GPU, runs in ~2 minutes.
"""
import sys
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, PandasTools

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka.predictor import ACID_SITES, BASE_SITES, neutralize

SITE_RANGES = {
    "carboxylic_acid": (1.5, 6.5), "sulfonic_acid": (-2.0, 3.0),
    "phosphoric_acid": (0.5, 4.0), "tetrazole": (3.0, 7.0),
    "tetrazole_2": (3.0, 7.0), "sulfonamide_2": (6.5, 12.5),
    "sulfonamide_1": (8.0, 13.0), "thiol": (6.5, 12.0),
    "hydroxamic_acid": (7.0, 11.5), "phenol": (6.5, 12.5),
    "imide": (6.0, 12.0), "aromatic_lactam": (7.5, 13.0),
    "malonate_dicarbonyl": (3.0, 13.0), "cyanoacetate": (7.0, 13.0),
    "malononitrile": (8.0, 13.0), "oxime": (8.5, 13.0),
    "guanidine": (10.0, 14.0), "amidine": (8.5, 13.0),
    "prim_amine": (7.5, 12.0), "sec_amine": (7.5, 12.0),
    "tert_amine": (6.5, 11.5), "pyridine_N": (2.0, 8.0),
    "aniline": (1.5, 6.5), "aniline_sec": (1.5, 7.0),
    "aniline_tert": (2.0, 7.5),
}

ALL_SITE_NAMES = [n for n, _, _ in ACID_SITES] + [n for n, _, _ in BASE_SITES]
SITE_INDEX = {n: i for i, n in enumerate(ALL_SITE_NAMES)}


def candidates(mol):
    out, rank = [], 0
    for name, smarts, _ in ACID_SITES:
        p = Chem.MolFromSmarts(smarts)
        if p is not None and mol.GetSubstructMatches(p):
            out.append((name, "acid", rank))
        rank += 1
    for name, smarts, _ in BASE_SITES:
        p = Chem.MolFromSmarts(smarts)
        if p is not None and mol.GetSubstructMatches(p):
            out.append((name, "base", rank))
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


def featurize(mol, site_name, kind, prio_rank, n_cands, mdesc):
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


def load_labels():
    frames = []
    df = PandasTools.LoadSDF("mlpka/datasets/combined_training_datasets_unique.sdf")
    pk = next(c for c in df.columns if c.lower() in ("pka", "pka_value", "value"))
    rows = []
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
            rows.append({"smiles": Chem.MolToSmiles(m), "pKa": v})
        except Exception:
            pass
    frames.append(pd.DataFrame(rows))
    try:
        frames.append(pd.read_csv("extra_pka_data.csv")[["smiles", "pKa"]])
    except FileNotFoundError:
        pass
    return pd.concat(frames, ignore_index=True).drop_duplicates("smiles").reset_index(drop=True)


print("loading labels...")
data = load_labels()
print(f"  {len(data)} molecules")

print("building site-selection training set...")
X, y, groups, meta = [], [], [], []
n_single, n_ambig, n_dropped = 0, 0, 0

for gi, row in enumerate(data.itertuples()):
    mol = Chem.MolFromSmiles(row.smiles)
    if mol is None:
        continue
    mol = neutralize(mol)
    cands = candidates(mol)
    if not cands:
        continue
    if len(cands) == 1:
        n_single += 1
        correct = cands[0][0]
    else:
        n_ambig += 1
        fits = [n for n, k, r in cands
                if n in SITE_RANGES
                and SITE_RANGES[n][0] <= row.pKa <= SITE_RANGES[n][1]]
        if len(fits) != 1:
            n_dropped += 1
            continue
        correct = fits[0]

    mdesc = mol_descriptors(mol)
    for name, kind, rank in cands:
        X.append(featurize(mol, name, kind, rank, len(cands), mdesc))
        y.append(1 if name == correct else 0)
        groups.append(gi)
        meta.append({"smiles": row.smiles, "site": name,
                      "correct": name == correct, "n_cands": len(cands)})

X = np.array(X, dtype=float)
y = np.array(y)
groups = np.array(groups)
meta = pd.DataFrame(meta)
print(f"  single-site molecules:      {n_single}")
print(f"  multi-site molecules used:  {n_ambig - n_dropped}")
print(f"  dropped (unresolvable):     {n_dropped}")
print(f"  training rows (site cands): {len(X)}  positives: {y.sum()}")

print("\ntraining site selector (grouped 5-fold)...")
gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(y))
for i, (tr, va) in enumerate(gkf.split(X, y, groups)):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                            num_leaves=31, verbose=-1, random_state=42)
    m.fit(X[tr], y[tr])
    oof[va] = m.predict_proba(X[va])[:, 1]
    print(f"  fold {i+1}/5 done")

meta["score"] = oof
print("\n" + "=" * 60)
print("SITE-SELECTION ACCURACY (out-of-fold)")
print("=" * 60)
picked = meta.loc[meta.groupby("smiles")["score"].idxmax()]
print(f"top-1 accuracy, all molecules:        {picked['correct'].mean()*100:.1f}%")

multi = meta[meta.n_cands > 1]
picked_multi = multi.loc[multi.groupby("smiles")["score"].idxmax()]
print(f"top-1 accuracy, MULTI-site only:      {picked_multi['correct'].mean()*100:.1f}%")
print(f"  (n={len(picked_multi)} molecules where priority must choose)")

prio_idx = multi.groupby("smiles", sort=False).head(1).index
prio = multi.loc[prio_idx]
print(f"\ncurrent SMARTS-priority rule, same molecules: {prio['correct'].mean()*100:.1f}%")
print(f"  (>>> gap between these two = headroom at inference)")

print("\ntraining final selector on all data...")
final = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                            num_leaves=31, verbose=-1, random_state=42)
final.fit(X, y)
joblib.dump({"model": final, "site_names": ALL_SITE_NAMES,
             "site_ranges": SITE_RANGES}, "models/site_selector.pkl")
print("saved -> models/site_selector.pkl")
