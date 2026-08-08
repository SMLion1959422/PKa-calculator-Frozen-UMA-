"""Does expanding the candidate ballot with the hard-negative site
detector, while the production ranker still makes the pick, beat the
SMARTS-only baseline?

THE MEASUREMENT PROBLEM THIS FIXES
Two of the ranker's 95 features are `prio_rank` and `n_cands` - the
atom's position within the candidate list and the list's length. A
previous quick test scored the ranker on features whose rank/n_cands had
been computed over ALL heteroatoms, a distribution the ranker never saw
in training (it was trained on SMARTS candidate lists). That made the
comparison meaningless: "ranker over all heteroatoms" collapsed to 76.8%
purely from feature mismatch.

Here BOTH arms recompute those features over their own candidate set:

    baseline : features over the SMARTS candidates      (production)
    union    : features over SMARTS candidates + detector-flagged atoms

so the only thing differing is which atoms are on the ballot.

Evaluated on identical molecules, and reported separately for the
subset whose true site SMARTS cannot see - the molecules this whole
exercise exists to recover.
"""
import argparse

import joblib
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors
from rdkit.Chem.EState import EStateIndices

from umapka import site_finder as SF
from umapka.predictor import ACID_SITES, BASE_SITES, neutralize

RDLogger.DisableLog("rdApp.*")
HETERO = {"N", "O", "S", "P"}
TESTS = [
    ("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf", "novartis"),
    ("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf", "avlilumove"),
]


def _context(mol):
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
    return gast, est, dm, ctx


def features_over(mol, idxs, hits, patt_names, patt_idx, cache):
    """Feature rows for `idxs`, with prio_rank/n_cands computed over
    EXACTLY that set - the whole point of this script."""
    gast, est, dm, ctx = cache
    ordered = sorted(idxs)
    rows = []
    for rank, i in enumerate(ordered):
        info = hits.get(i, {"patterns": set(), "kinds": set()})
        rows.append(SF._atom_features(mol, i, info, gast, est, dm, ctx,
                                      len(ordered), rank, patt_names, patt_idx))
    X = np.ascontiguousarray(np.nan_to_num(np.array(rows, dtype=float),
                                            nan=0.0, posinf=0.0, neginf=0.0),
                              dtype=np.float64)
    return ordered, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="detector probability above which an atom joins the ballot")
    args = ap.parse_args()

    rk = joblib.load("models/site_finder_v2.pkl")
    ranker, patt_names = rk["model"], rk["patt_names"]
    patt_idx = {n: i for i, n in enumerate(patt_names)}
    det = joblib.load("models/site_detector.pkl")["classifier_full"]

    for path, tag in TESTS:
        res = {"base": [], "union": [], "inv_base": [], "inv_union": [],
               "added": 0, "n": 0}
        for mol in Chem.ForwardSDMolSupplier(path):
            if mol is None or not mol.HasProp("marvin_atom"):
                continue
            try:
                ma = int(float(mol.GetProp("marvin_atom")))
                nm = neutralize(Chem.Mol(mol))
                if ma < 0 or ma >= nm.GetNumAtoms():
                    continue
                cache = _context(nm)
            except Exception:
                continue

            hits = SF._candidate_atoms(nm, ACID_SITES, BASE_SITES)
            smarts = set(hits)
            hetero = [a.GetIdx() for a in nm.GetAtoms() if a.GetSymbol() in HETERO]
            if not smarts or not hetero:
                continue

            # --- detector scores: features over ALL heteroatoms, matching
            #     how the detector was TRAINED (its own convention) ---
            try:
                h_ord, Xh = features_over(nm, hetero, hits, patt_names, patt_idx, cache)
                prob = det.predict_proba(Xh)[:, 1]
                flagged = {h_ord[k] for k in range(len(h_ord))
                           if prob[k] > args.threshold}
            except Exception:
                flagged = set()

            union = smarts | flagged
            res["added"] += len(union - smarts)
            res["n"] += 1
            invisible = ma not in smarts

            for name, cand in (("base", smarts), ("union", union)):
                try:
                    order, X = features_over(nm, cand, hits, patt_names,
                                             patt_idx, cache)
                    pick = order[int(np.argmax(ranker.predict(X)))]
                    ok = (pick == ma)
                except Exception:
                    ok = False
                res[name].append(ok)
                if invisible:
                    res[f"inv_{name}"].append(ok)

        n = len(res["base"])
        if not n:
            continue
        b, u = np.mean(res["base"]) * 100, np.mean(res["union"]) * 100
        print(f"\n=== {tag}  (n={n} molecules, identical rows both arms) ===")
        print(f"  baseline  SMARTS candidates only : {b:5.1f}%")
        print(f"  UNION     + detector p>{args.threshold:<4}       : {u:5.1f}%   "
              f"({u-b:+.1f})")
        print(f"  extra candidates added: {res['added']} "
              f"({res['added']/res['n']:.2f} per molecule)")
        if res["inv_base"]:
            ib = np.mean(res["inv_base"]) * 100
            iu = np.mean(res["inv_union"]) * 100
            print(f"  on the {len(res['inv_base'])} SMARTS-INVISIBLE molecules: "
                  f"baseline {ib:.1f}% -> union {iu:.1f}%")


if __name__ == "__main__":
    main()
