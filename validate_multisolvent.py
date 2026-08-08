"""Validate the MULTISOLVENT path against experimental pKa on the
published held-out test split (Nevolianis et al., Zenodo 15604045,
CC BY 4.0) - anion_data/data/data_splits/D2A-pKa-test.csv, 603 rows.

WHY THIS SCRIPT EXISTS
RESULTS.md already reports per-solvent MAE on this split (overall 0.711),
but those numbers were produced by feeding the dataset's EXACT reaction
pairs (AH>>A-) straight in as features. That is not what a user gets:
predict(smiles, solvent=...) has to FIND the site itself. Nobody had
measured the production path, so "is multisolvent accurate?" was
genuinely unanswered for real usage.

Two numbers are reported per solvent:
  exact-pair : dataset's own AH/A- pair -> comparable to RESULTS.md,
               isolates the regressor's error
  production : predict(protonated_smiles, solvent=...) -> what a user
               actually gets, includes site-finding error
The gap between them is the cost of self-contained site finding in
non-aqueous solvent, which is otherwise invisible.

Embeddings are memoized per (smiles, site) so the two paths share work.
"""
import time

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from umapka import PkaPredictor
from umapka import solvents as sv
from umapka.predictor import protonation_pair_site_tagged

RDLogger.DisableLog("rdApp.*")

TEST = "anion_data/data/data_splits/D2A-pKa-test.csv"
SMILES_TO_NAME = {i.smiles: k for k, i in sv.SOLVENTS.items()}


def split_rxn(r):
    try:
        a, b = r.split(">>")
        return a.strip(), b.strip()
    except Exception:
        return None, None


def main():
    df = pd.read_csv(TEST)
    print(f"held-out test rows: {len(df)}")
    df["solvent"] = df.solvent_smiles.map(SMILES_TO_NAME)
    unknown = df[df.solvent.isna()].solvent_smiles.unique()
    if len(unknown):
        print(f"  skipping unsupported solvent SMILES: {list(unknown)}")
    df = df[df.solvent.notna()].reset_index(drop=True)
    print(f"  usable: {len(df)}\n")

    print("loading UMA...")
    p = PkaPredictor("models/model_core_v3.pkl")

    memo = {}

    def pair_feat(prot, dep, pi=None, di=None):
        key = (prot, dep)
        if key not in memo:
            memo[key] = p.features(prot, dep, pi, di)
        return memo[key]

    rows = []
    t0 = time.time()
    for i, r in enumerate(df.itertuples()):
        prot_x, dep_x = split_rxn(r.reaction_smiles)
        if not prot_x or not dep_x:
            continue
        exp, solv = float(r.pKa_avg), r.solvent

        # --- exact-pair path (RESULTS.md-comparable) ---
        pred_exact = None
        try:
            f = pair_feat(prot_x, dep_x)
            pred_exact = float(p._base_pka(f, solv))
        except Exception:
            pass

        # --- production path (site found by the model itself) ---
        pred_prod, site_same = None, None
        try:
            pr, pi, dp_, di, _kind = protonation_pair_site_tagged(
                prot_x, return_kind=True)
            f2 = pair_feat(pr, dp_, pi, di)
            pred_prod = float(p._base_pka(f2, solv))
            cx = Chem.CanonSmiles(dep_x)
            site_same = (Chem.CanonSmiles(dp_) == cx)
        except Exception:
            pass

        rows.append({"solvent": solv, "exp": exp,
                     "pred_exact": pred_exact, "pred_prod": pred_prod,
                     "site_same": site_same})
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(df)}]  {(time.time()-t0)/60:.1f} min",
                  flush=True)

    d = pd.DataFrame(rows)
    d["err_exact"] = (d.pred_exact - d.exp).abs()
    d["err_prod"] = (d.pred_prod - d.exp).abs()
    d.to_csv("validation_multisolvent.csv", index=False)

    print("\n" + "=" * 72)
    print("MULTISOLVENT VALIDATION vs EXPERIMENT (published held-out split)")
    print("=" * 72)
    g = d.groupby("solvent").agg(
        n=("exp", "size"),
        MAE_exact=("err_exact", "mean"),
        MAE_prod=("err_prod", "mean"),
        site_agree=("site_same", "mean"),
    ).round(3).sort_values("n", ascending=False)
    print(g.to_string())
    print(f"\noverall  exact-pair MAE = {d.err_exact.mean():.3f}   "
          f"(RESULTS.md reference: 0.711)")
    print(f"overall  production MAE = {d.err_prod.mean():.3f}   "
          f"<- what a user actually gets")
    print(f"site agreement with dataset pair: {d.site_same.mean()*100:.1f}%")

    thin = g[g.n < 15]
    if len(thin):
        print(f"\n[!] too few held-out points to trust: "
              f"{', '.join(f'{k} (n={int(v)})' for k, v in thin.n.items())}")
    print("\nsaved -> validation_multisolvent.csv")


if __name__ == "__main__":
    main()
