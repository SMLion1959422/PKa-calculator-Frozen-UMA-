"""Hard-negative SITE DETECTOR: score every heteroatom, not just the ones
a hand-written SMARTS list happens to match.

THE GAP THIS CLOSES (measured, not assumed)

    true titratable site NOT in the SMARTS candidate set
        TRAIN       358 / 5994   (6.0%)
        NOVARTIS     11 /  280   (3.9%)
        AvLiLuMoVe    0 /  123   (0.0%)

On those molecules the site finder cannot possibly be right - the
correct atom is not even on the ballot, so the ranker picks some other
candidate and the prediction is for the wrong equilibrium. Wrong-site
predictions cost roughly 4 pKa units (cf. the acid/base kind analysis in
RESULTS.md section 6), so ~4% of Novartis is being scored at ~4 MAE.

THE APPROACH
Harvest EVERY N/O/S/P atom, label the experimentally annotated one
positive and all the rest hard negatives, and train a classifier on
physicochemical features. Amides, esters and backbone linkers become
explicit negatives, which is what stops an unconstrained heteroatom scan
from flagging them.

  Task 1  classification : titratable or not, trained on ALL atoms
  Task 2  regression     : pKa, trained ONLY on positives (masked)

The mask matters: a neutral amide has no pKa, so including it in the
regression target would teach the model to map junk values into the
latent space.

TWO FEATURE VARIANTS, AND WHY
`_atom_features` is 95-dim, of which 60 are one-hot SMARTS-pattern
indicators. For an atom no pattern matches, all 60 are zero - so a model
given those features can score well by simply learning "pattern matched
=> positive", which is exactly the SMARTS dependence we are trying to
remove. So both are trained:

  full  (95-dim) - includes the pattern one-hots
  phys  (35-dim) - pattern one-hots REMOVED; purely physicochemical

and both are reported separately on the SMARTS-INVISIBLE subset, which
is the only measurement that answers the actual question. A model that
looks great overall but cannot find pattern-unmatched sites has not
solved anything.

WHY TWO STAGES
Doing heavy RDKit work and LightGBM .fit() in one process reliably dies
on this platform with "access violation" inside LGBM_DatasetSetField.
Splitting them sidesteps it and makes the features reusable.

    python train_hard_negatives.py --stage build
    python train_hard_negatives.py --stage fit

Output: models/site_detector.pkl
(Named for what it does - detecting whether an atom is titratable at
all. models/kind_classifier.pkl remains a different model answering a
different question: acid vs base for an atom already known titratable.)
"""
import argparse

import joblib
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors
from rdkit.Chem.EState import EStateIndices
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from tqdm import tqdm

from umapka import site_finder as SF
from umapka.predictor import ACID_SITES, BASE_SITES, neutralize

RDLogger.DisableLog("rdApp.*")

TRAIN = "mlpka/datasets/combined_training_datasets_unique.sdf"
TESTS = [
    ("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf", "novartis"),
    ("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf", "avlilumove"),
]
CACHE = "hard_negative_features.pkl"
OUT = "models/site_detector.pkl"

# S and P included: 1.8% of training sites are on neither N nor O
# (thiols, phosphonic/phosphoric acids). An N/O-only scan would make
# those unreachable by construction - the same failure being fixed here.
HETERO = {"N", "O", "S", "P"}


def harvest(mol):
    """Every candidate heteroatom index in the neutralized molecule."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in HETERO]


def atom_rows(mol, patt_names, patt_idx):
    """95-dim feature row for every harvested heteroatom.

    Atoms matched by no SMARTS get an empty pattern/kind set, so their 60
    pattern one-hots are all zero - which is precisely why the `phys`
    variant below drops those columns."""
    hits = SF._candidate_atoms(mol, ACID_SITES, BASE_SITES)
    idxs = harvest(mol)
    if not idxs:
        return {}
    try:
        AllChem.ComputeGasteigerCharges(mol)
        gast = np.nan_to_num(np.array(
            [float(a.GetPropsAsDict().get("_GasteigerCharge", 0.0))
             for a in mol.GetAtoms()]), nan=0.0, posinf=0.0, neginf=0.0)
        est = np.array(EStateIndices(mol))
        dm = Chem.GetDistanceMatrix(mol)
        ctx = [Descriptors.MolWt(mol), Crippen.MolLogP(mol), Descriptors.TPSA(mol),
               float(Descriptors.RingCount(mol)),
               float(Descriptors.NumAromaticRings(mol)),
               float(mol.GetNumAtoms()), float(Chem.GetFormalCharge(mol))]
    except Exception:
        return {}
    ordered = sorted(idxs)
    out = {}
    for rank, i in enumerate(ordered):
        info = hits.get(i, {"patterns": set(), "kinds": set()})
        try:
            out[i] = (SF._atom_features(mol, i, info, gast, est, dm, ctx,
                                        len(ordered), rank, patt_names, patt_idx),
                      i in hits)
        except Exception:
            continue
    return out


def build_split(path, patt_names, patt_idx, tag):
    X, y, pka, grp, smarts_seen = [], [], [], [], []
    mols = [m for m in Chem.ForwardSDMolSupplier(path) if m is not None]
    g = 0
    for mol in tqdm(mols, desc=tag[:24]):
        if not (mol.HasProp("marvin_atom") and mol.HasProp("pKa")):
            continue
        try:
            ma = int(float(mol.GetProp("marvin_atom")))
            v = float(mol.GetProp("pKa"))
            nm = neutralize(Chem.Mol(mol))
            if ma < 0 or ma >= nm.GetNumAtoms() or not (0 < v < 14):
                continue
        except Exception:
            continue
        rows = atom_rows(nm, patt_names, patt_idx)
        if not rows or ma not in rows:
            # the annotated site is not even a harvested heteroatom
            # (e.g. a carbon acid) - nothing to learn from here
            continue
        for i, (feat, in_smarts) in rows.items():
            X.append(feat)
            y.append(1 if i == ma else 0)
            pka.append(v if i == ma else np.nan)
            grp.append(g)
            smarts_seen.append(in_smarts)
        g += 1
    return (np.array(X, dtype=float), np.array(y), np.array(pka, dtype=float),
            np.array(grp), np.array(smarts_seen, dtype=bool))


def stage_build():
    d = joblib.load("models/site_finder_v2.pkl")
    patt_names = d["patt_names"]
    patt_idx = {n: i for i, n in enumerate(patt_names)}
    out = {"patt_names": patt_names, "n_patt": len(patt_names)}

    print("harvesting every N/O/S/P atom...")
    out["train"] = build_split(TRAIN, patt_names, patt_idx, "train")
    X, y, _pk, grp, ss = out["train"]
    print(f"  train: {X.shape[0]} atoms from {len(set(grp))} molecules "
          f"({y.mean()*100:.1f}% positive)")
    print(f"         positives invisible to SMARTS: "
          f"{int((~ss & (y == 1)).sum())}")
    for path, name in TESTS:
        out[name] = build_split(path, patt_names, patt_idx, name)
        Xe, ye, _p, ge, se = out[name]
        print(f"  {name}: {Xe.shape[0]} atoms from {len(set(ge))} molecules, "
              f"{int((~se & (ye == 1)).sum())} SMARTS-invisible positives")
    joblib.dump(out, CACHE, compress=0)
    print(f"\nsaved -> {CACHE}   (now: --stage fit)")


def _clean(X):
    return np.ascontiguousarray(
        np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)


def stage_fit():
    d = joblib.load(CACHE)
    n_patt = d["n_patt"]
    X, y, pka, grp, ss = d["train"]
    X = _clean(X)
    # phys variant: drop the SMARTS one-hots so the model cannot lean on them
    phys_cols = np.arange(n_patt, X.shape[1])
    print(f"train atoms {X.shape[0]} | full {X.shape[1]}-dim | "
          f"phys {len(phys_cols)}-dim | positives {y.mean()*100:.1f}%")

    pos_w = float((y == 0).sum()) / max(1, (y == 1).sum())
    print(f"scale_pos_weight {pos_w:.2f}\n")

    results = {}
    for tag, cols in (("full", np.arange(X.shape[1])), ("phys", phys_cols)):
        Xv = np.ascontiguousarray(X[:, cols])
        # grouped by MOLECULE - atoms of one molecule must never straddle folds
        gkf = GroupKFold(n_splits=5)
        oof = np.zeros(len(y))
        for tr, va in gkf.split(Xv, y, grp):
            m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05,
                                    num_leaves=31, scale_pos_weight=pos_w,
                                    verbose=-1, random_state=42)
            m.fit(Xv[tr], y[tr])
            oof[va] = m.predict_proba(Xv[va])[:, 1]
        pred = (oof > 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        # top-1 per molecule: does the highest-scoring atom = the true site?
        top1 = np.mean([y[grp == g][np.argmax(oof[grp == g])] == 1
                        for g in np.unique(grp)])
        inv = (~ss) & (y == 1)
        rec_inv = float((pred[inv] == 1).mean()) if inv.sum() else float("nan")
        print(f"[{tag}] OOF precision {prec:.3f} recall {rec:.3f} | "
              f"top-1-per-molecule {top1*100:.1f}%")
        print(f"       recall on SMARTS-INVISIBLE positives "
              f"({int(inv.sum())}): {rec_inv*100:.1f}%   <-- the point of this model")
        clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05,
                                  num_leaves=31, scale_pos_weight=pos_w,
                                  verbose=-1, random_state=42).fit(Xv, y)
        results[tag] = {"model": clf, "cols": cols, "top1": top1,
                        "rec_inv": rec_inv, "precision": prec, "recall": rec}

    # --- masked regression: positives only ---
    mask = y == 1
    print(f"\nmasked pKa regression on {int(mask.sum())} positives "
          f"(negatives excluded, not zero-filled)")
    gkf = GroupKFold(n_splits=5)
    Xp, yp, gp = np.ascontiguousarray(X[mask]), pka[mask], grp[mask]
    oofr = np.zeros(len(yp))
    for tr, va in gkf.split(Xp, yp, gp):
        r = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                               verbose=-1, random_state=42)
        r.fit(Xp[tr], yp[tr])
        oofr[va] = r.predict(Xp[va])
    print(f"  OOF MAE {np.abs(oofr-yp).mean():.3f}  (RDKit-only features; the UMA"
          f" pipeline is 0.949 on Novartis - this head is a coarse prior, not a"
          f" replacement)")
    reg = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                            verbose=-1, random_state=42).fit(Xp, yp)

    print("\n=== HELD-OUT ===")
    for name in ("novartis", "avlilumove"):
        if name not in d:
            continue
        Xe, ye, _pe, ge, se = d[name]
        if not len(Xe):
            continue
        Xe = _clean(Xe)
        for tag in ("full", "phys"):
            r = results[tag]
            sc = r["model"].predict_proba(np.ascontiguousarray(Xe[:, r["cols"]]))[:, 1]
            top1 = np.mean([ye[ge == g][np.argmax(sc[ge == g])] == 1
                            for g in np.unique(ge)])
            inv = (~se) & (ye == 1)
            hit = (np.mean([ye[ge == g][np.argmax(sc[ge == g])] == 1
                            for g in np.unique(ge[inv])]) if inv.sum() else float("nan"))
            print(f"  {name:11s} [{tag}] top-1 {top1*100:5.1f}%  | "
                  f"top-1 on the {int(inv.sum())} SMARTS-invisible molecules: "
                  f"{hit*100:.1f}%")

    joblib.dump({"classifier_full": results["full"]["model"],
                 "classifier_phys": results["phys"]["model"],
                 "phys_cols": results["phys"]["cols"],
                 "regressor": reg, "patt_names": d["patt_names"],
                 "n_patt": n_patt, "hetero": sorted(HETERO),
                 "metrics": {k: {m: v[m] for m in
                                 ("top1", "rec_inv", "precision", "recall")}
                             for k, v in results.items()}}, OUT)
    print(f"\nsaved -> {OUT}")
    print("Compare top-1 against the current SMARTS+ranker: 97.4% novartis.")
    print("This model only earns its place if it matches that AND recovers")
    print("the SMARTS-invisible sites the current pipeline cannot reach.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "fit"], required=True)
    a = ap.parse_args()
    stage_build() if a.stage == "build" else stage_fit()
