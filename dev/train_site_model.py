"""Train the site-local UMA model (feature mode "site_v3").

Changes vs model_core*.pkl, both motivated by dev/site_ablation.py:

  1. The training pair is built at the ANNOTATED ionizable atom
     (``marvin_atom`` in the Baltruschat & Czodrowski SDFs), not at the
     first SMARTS match. The SMARTS priority rule disagrees with the
     annotation for 16% of the training set and 24% of Novartis
     (wrong site, uncovered site, or no site at all), so those rows are
     trained/evaluated on the wrong proton.
  2. Embeddings are pooled around the site (``umapka.predictor.pool_site``)
     instead of over the whole molecule, and UMA's deprotonation energy
     is appended as a feature.

In the RDKit-only ablation these two changes take Novartis MAE from
1.48 to 0.99 (annotated site at test time) / 1.09 (fully automatic).
Whether UMA embeddings add to that is exactly what this script tells
you: it prints the same table for UMA features.

Needs a GPU machine with access to facebook/UMA (see SETUP.md) and the
dataset repo:
    git clone https://github.com/czodrowskilab/Machine-learning-meets-pKa mlpka
    python dev/train_site_model.py mlpka/datasets

Embeddings are cached to site_v3_cache.pkl (resumable).
Output: models/model_site_v3.pkl, loadable with PkaPredictor(...).
"""
import os
import sys
import joblib
import numpy as np
import lightgbm as lgb
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold

from umapka import PkaPredictor
from umapka.sites import Site, find_sites, neutralize

RDLogger.DisableLog("rdApp.*")
D = sys.argv[1] if len(sys.argv) > 1 else "mlpka/datasets"
CACHE = "site_v3_cache.pkl"
OUT = "models/model_site_v3.pkl"


def annotated_site(mol, atom, pka_type):
    """Site object for the annotated atom; reuse the SMARTS site (for its
    charge-sharing atoms) when one exists at that atom."""
    for s in find_sites(mol):
        if s.atom == atom:
            return s
    kind = "acid" if pka_type == "acidic" else "base"
    return Site(index=-1, atom=atom, group="annotated_only", kind=kind,
                element=mol.GetAtomWithIdx(atom).GetSymbol(),
                charge_atoms=[(atom, 1.0)])


def rows(name):
    out = []
    for m in Chem.SDMolSupplier(f"{D}/{name}.sdf"):
        if m is None:
            continue
        p = m.GetPropsAsDict()
        n = neutralize(m)
        out.append(dict(mol=n, smiles=Chem.MolToSmiles(n), y=float(p["pKa"]),
                        site=annotated_site(n, int(p["marvin_atom"]), p["marvin_pKa_type"]),
                        smarts_sites=find_sites(n)))
    return out


def featurize(pred, cache, key, mol, site):
    if key not in cache:
        try:
            cache[key] = pred.features_site(mol, site)[0]
        except Exception as e:
            cache[key] = None
            print(f"  skip {key}: {e}")
    return cache[key]


def matrix(pred, cache, data, choose="annotated"):
    X, y, keep = [], [], []
    for i, r in enumerate(data):
        if choose == "annotated":
            site = r["site"]
        elif r["smarts_sites"]:
            site = r["smarts_sites"][0]        # what PkaPredictor.predict() does
        else:
            continue
        f = featurize(pred, cache, f"{r['smiles']}|{site.atom}|{site.kind}", r["mol"], site)
        if f is not None:
            X.append(f); y.append(r["y"]); keep.append(i)
    return np.array(X), np.array(y), keep


def make():
    return lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.02, num_leaves=63,
                             min_child_samples=10, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.4, reg_lambda=1.0,
                             verbose=-1, random_state=0)


def main():
    pred = PkaPredictor("models/model_core_v2.pkl", multisolvent_model_path=None)
    cache = joblib.load(CACHE) if os.path.exists(CACHE) else {}
    train = rows("combined_training_datasets_unique")
    tests = {"novartis": rows("novartis_cleaned_mono_unique_notraindata"),
             "AvLiLuMoVe": rows("AvLiLuMoVe_cleaned_mono_unique_notraindata")}

    X, y, keep = matrix(pred, cache, train)
    joblib.dump(cache, CACHE)
    print(f"train rows with features: {len(y)}/{len(train)}")

    # scaffold-grouped CV (no Bemis-Murcko core shared across folds)
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=train[i]["mol"]) for i in keep]
    groups = np.unique(scaf, return_inverse=True)[1]
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, y, groups):
        oof[va] = make().fit(X[tr], y[tr]).predict(X[va])
    print(f"scaffold 5-fold CV MAE: {np.abs(oof - y).mean():.3f}")

    model = make().fit(X, y)
    metrics = {"scaffold_cv_mae": float(np.abs(oof - y).mean())}
    for name, data in tests.items():
        for choose in ("annotated", "smarts_priority"):
            Xt, yt, _ = matrix(pred, cache, data, choose)
            mae = float(np.abs(model.predict(Xt) - yt).mean())
            metrics[f"{name}_{choose}"] = mae
            print(f"{name:11s} site={choose:16s} MAE {mae:.3f}  (n={len(yt)}/{len(data)})")
    joblib.dump(cache, CACHE)
    joblib.dump({"regressor": model, "calibrator": None, "feature": "site_v3",
                 "metrics": metrics}, OUT)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
