"""LEARNED ACID/BASE KIND CLASSIFIER - the highest-value fix found by
error decomposition.

THE EVIDENCE
eval_cached.py split Novartis error by whether the predicted acid/base
KIND matched ChemAxon's annotation:

    kind wrong (14 mols, 5.1%) : MAE 3.901
    kind right (261 mols)      : MAE 0.846
    -------------------------------------
    overall                    : MAE 1.001

Getting the kind wrong is not a small error - it computes the OPPOSITE
proton transfer (deprotonate when the true equilibrium is protonation),
so the prediction is meaningless, not merely imprecise. Those 14
molecules contribute 0.199 of the 1.001 total. Eliminating them lands
Novartis at ~0.846, which is past ChemAxon Marvin (0.856) and matches
the oracle-site ceiling (0.845).

WHY A CLASSIFIER AND NOT ANOTHER HEURISTIC
Kind is currently decided by an H-count rule in site_finder._kind():
acid if the atom still carries a hydrogen, else base. That is right
94.9% of the time on Novartis, and it is a genuine improvement over the
"acid always wins" rule it replaced (which scored 0/12 on ambiguous
atoms). But it is a one-feature rule applied to a question that has
plenty of signal available - element, charge, hybridisation, aromaticity,
neighbourhood Gasteiger charge and EState, which SMARTS patterns matched.
This trains on exactly those features, with real annotations as targets.

Ground truth is marvin_pKa_type ("acidic"/"basic"), available for all
5994 training molecules. It is independent of the pKa VALUE, so there is
no circularity with the regression target.

Held-out check is on the same Novartis/AvLiLuMoVe files used everywhere
else, which are excluded from training by construction (the
*_notraindata.sdf naming).

Output: models/kind_classifier.pkl
"""
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.EState import EStateIndices
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from tqdm import tqdm

from umapka.predictor import ACID_SITES, BASE_SITES, neutralize
from umapka import site_finder as SF

RDLogger.DisableLog("rdApp.*")

TRAIN = "mlpka/datasets/combined_training_datasets_unique.sdf"
TESTS = [
    ("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf", "novartis"),
    ("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf", "avlilumove"),
]


def featurize_atom(mol, idx, patt_names, patt_idx):
    """Per-atom features built EXACTLY as site_finder.predict_kind builds
    them at inference time.

    This used to `return None` when the atom matched no SMARTS pattern,
    which silently dropped the 358 training molecules (6.0%) whose true
    site is not SMARTS-matched - precisely the atom types the expanded
    ballot now surfaces (primary sulfonamides, N-oxides, thiols,
    phosphonic acids). The classifier therefore had never seen them, and
    guessed on them at inference.

    Two things must mirror predict_kind or the training distribution
    drifts from the serving one:
      - an unmatched atom gets an EMPTY pattern/kind info dict, so its 60
        pattern one-hots are zero (same as at inference)
      - n_cands / prio_rank are computed over hits UNION {idx}, not over
        the raw SMARTS hits, because that is the set predict_kind ranks
        within once it injects the atom
    """
    hits = SF._candidate_atoms(mol, ACID_SITES, BASE_SITES)
    if idx not in hits:
        if mol.GetAtomWithIdx(idx).GetSymbol() not in ("N", "O", "S", "P"):
            return None
        hits = dict(hits)
        hits[idx] = {"patterns": set(), "kinds": set()}
    try:
        AllChem.ComputeGasteigerCharges(mol)
        gast = np.nan_to_num(np.array(
            [float(a.GetPropsAsDict().get("_GasteigerCharge", 0.0))
             for a in mol.GetAtoms()]), nan=0.0, posinf=0.0, neginf=0.0)
        est = np.array(EStateIndices(mol))
        dm = Chem.GetDistanceMatrix(mol)
        mol_ctx = [Descriptors.MolWt(mol), Crippen.MolLogP(mol),
                   Descriptors.TPSA(mol), float(Descriptors.RingCount(mol)),
                   float(Descriptors.NumAromaticRings(mol)),
                   float(mol.GetNumAtoms()), float(Chem.GetFormalCharge(mol))]
        ordered = sorted(hits.keys())
        rank = ordered.index(idx)
        return SF._atom_features(mol, idx, hits[idx], gast, est, dm, mol_ctx,
                                 len(ordered), rank, patt_names, patt_idx)
    except Exception:
        return None


def build(path, patt_names, patt_idx, tag):
    X, y, smis = [], [], []
    mols = [m for m in Chem.ForwardSDMolSupplier(path) if m is not None]
    for mol in tqdm(mols, desc=tag[:26]):
        if not (mol.HasProp("marvin_atom") and mol.HasProp("marvin_pKa_type")):
            continue
        try:
            ma = int(float(mol.GetProp("marvin_atom")))
            kind = 1 if mol.GetProp("marvin_pKa_type").startswith("acid") else 0
            nm = neutralize(Chem.Mol(mol))
            if ma < 0 or ma >= nm.GetNumAtoms():
                continue
            f = featurize_atom(nm, ma, patt_names, patt_idx)
        except Exception:
            continue
        if f is None:
            continue
        X.append(f); y.append(kind); smis.append(Chem.MolToSmiles(mol))
    return np.array(X, dtype=float), np.array(y), smis


def hcount_rule(smi, ma):
    """The rule currently in site_finder._kind(), for a like-for-like
    baseline on the identical molecules."""
    try:
        nm = neutralize(Chem.MolFromSmiles(smi))
        hits = SF._candidate_atoms(nm, ACID_SITES, BASE_SITES)
        if ma not in hits:
            return None
        kinds = hits[ma]["kinds"]
        if "acid" in kinds and "base" not in kinds:
            return 1
        if "base" in kinds and "acid" not in kinds:
            return 0
        return 1 if nm.GetAtomWithIdx(ma).GetTotalNumHs() > 0 else 0
    except Exception:
        return None


def stage_build():
    """Featurize train + test sets and cache them, then STOP.

    Kept in a separate process from training on purpose. Doing the heavy
    RDKit work (Gasteiger charges + EState over ~6000 molecules) and then
    calling LightGBM's .fit() in the SAME process reliably dies with
    'access violation reading 0x0' inside LGBM_DatasetSetField on this
    platform - reproduced with contiguous float64 features, float32
    labels and all-finite values, so it is C-library/OpenMP state
    corruption, not the data. Splitting the phases sidesteps it entirely
    and makes the features reusable, which is how the rest of this repo
    already works (embed -> cache -> train).
    """
    _d = joblib.load("models/site_finder_v2.pkl")
    patt_names = _d["patt_names"]
    patt_idx = {n: i for i, n in enumerate(patt_names)}

    out = {"patt_names": patt_names}
    print("building training set (target = marvin_pKa_type)...")
    X, y, sm = build(TRAIN, patt_names, patt_idx, "train")
    out["train"] = (X, y, sm)
    print(f"  {len(X)} molecules, {X.shape[1]} features, {y.mean()*100:.1f}% acidic")

    for path, name in TESTS:
        X, y, sm = build(path, patt_names, patt_idx, name)
        out[name] = (X, y, sm)
        print(f"  {name}: {len(X)} molecules")

    # H-count baseline computed here too, while RDKit is already warm
    base = {}
    for path, name in TESTS:
        ok = []
        for mol in Chem.ForwardSDMolSupplier(path):
            if mol is None or not (mol.HasProp("marvin_atom")
                                   and mol.HasProp("marvin_pKa_type")):
                continue
            try:
                ma = int(float(mol.GetProp("marvin_atom")))
                truth = 1 if mol.GetProp("marvin_pKa_type").startswith("acid") else 0
                r = hcount_rule(Chem.MolToSmiles(mol), ma)
            except Exception:
                continue
            if r is not None:
                ok.append(r == truth)
        base[name] = (float(np.mean(ok)) if ok else float("nan"), len(ok))
    out["hcount_baseline"] = base

    joblib.dump(out, "kind_features.pkl", compress=0)
    print("\nsaved -> kind_features.pkl   (now run: --stage fit)")


def stage_fit():
    """Train + evaluate from the cached features, in a fresh process."""
    d = joblib.load("kind_features.pkl")
    patt_names = d["patt_names"]
    Xtr, ytr, _ = d["train"]
    Xtr = np.ascontiguousarray(np.nan_to_num(Xtr, nan=0.0, posinf=0.0,
                                             neginf=0.0), dtype=np.float64)
    ytr = np.ascontiguousarray(ytr, dtype=np.int32)
    print(f"train: {Xtr.shape}  {ytr.mean()*100:.1f}% acidic")

    print("\n5-fold CV...")
    kf = GroupKFold(n_splits=5)
    oof = np.zeros(len(ytr))
    for i, (tr, va) in enumerate(kf.split(Xtr, ytr, np.arange(len(ytr)))):
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                num_leaves=31, verbose=-1, random_state=42)
        m.fit(Xtr[tr], ytr[tr])
        oof[va] = m.predict_proba(Xtr[va])[:, 1]
        print(f"  fold {i+1}/5")
    print(f"\nOOF accuracy: {((oof>0.5).astype(int)==ytr).mean()*100:.2f}%")

    final = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                num_leaves=31, verbose=-1, random_state=42)
    final.fit(Xtr, ytr)
    joblib.dump({"model": final, "patt_names": patt_names},
                "models/kind_classifier.pkl")
    print("saved -> models/kind_classifier.pkl")

    print("\n" + "=" * 64)
    print("HELD-OUT: learned classifier vs the current H-count rule")
    print("=" * 64)
    base = d.get("hcount_baseline", {})
    for name in ("novartis", "avlilumove"):
        if name not in d:
            continue
        Xte, yte, _ = d[name]
        if not len(Xte):
            continue
        Xte = np.ascontiguousarray(np.nan_to_num(Xte, nan=0.0, posinf=0.0,
                                                 neginf=0.0), dtype=np.float64)
        pred = final.predict(Xte)
        acc = (pred == yte).mean()
        b_acc, b_n = base.get(name, (float("nan"), 0))
        print(f"\n{name} (n={len(yte)})")
        print(f"  learned classifier : {acc*100:.1f}%")
        print(f"  H-count rule       : {b_acc*100:.1f}%   (n={b_n})")
        print(f"  errors remaining   : {int((pred != yte).sum())}"
              f"  (was ~{int(round((1-b_acc)*len(yte)))})")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "fit"], required=True)
    a = ap.parse_args()
    stage_build() if a.stage == "build" else stage_fit()


if __name__ == "__main__":
    main()
