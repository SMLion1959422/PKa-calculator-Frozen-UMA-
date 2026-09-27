"""Automatic site choice: which titratable site does a measured pKa belong to?

How the reference site was made: the Baltruschat & Czodrowski datasets
keep only molecules for which ChemAxon Marvin predicts exactly ONE pKa
inside 2-12, and store that atom as ``marvin_atom``
(Machine-learning-meets-pKa/scripts/gen_clean_mono_dataset.py). So the
task is "which site has its pKa in the measurable window", and rules
based on that are compared with today's SMARTS priority:

  priority      first SMARTS site (PkaPredictor.predict / protonation_pair)
  window        the site whose predicted pKa lies in [2, 12]; if several,
                SMARTS priority among them; if none, the site closest to
                the window
  window_micro  as "window", but on apparent site pKas from the coupled
                microstate model (umapka.microstates), so a site pushed
                out of the window by a neighbouring charge is not chosen
  oracle        the annotated site (upper bound)

Per-site pKas come from any intrinsic model; by default the GPU-free
umapka.rdkit_site model trained on the training SDF. With --uma MODEL the
UMA site_v3 model scores every site instead (needs a GPU; uses the same
--cache as dev/train_site_model.py so already-featurized sites are free).

    python dev/site_choice.py path/to/mlpka/datasets
    python dev/site_choice.py path/to/mlpka/datasets --uma MODEL --cache CACHE
"""
import argparse
import os
import numpy as np
from rdkit import Chem, RDLogger

from umapka import microstates as ms
from umapka.sites import find_sites, neutralize
from umapka.rdkit_site import RDKitSitePredictor

RDLogger.DisableLog("rdApp.*")
LO, HI = 2.0, 12.0


def dist_to_window(p):
    return 0.0 if LO <= p <= HI else min(abs(p - LO), abs(p - HI))


def pick(sites, pk, rule):
    if rule == "priority":
        return 0
    in_win = [i for i, p in enumerate(pk) if LO <= p <= HI]
    if in_win:
        return in_win[0]            # sites are in SMARTS priority order
    return int(np.argmin([dist_to_window(p) for p in pk]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets")
    ap.add_argument("--uma")
    ap.add_argument("--cache", default="site_v3_cache.pkl")
    args = ap.parse_args()

    if args.uma:
        import joblib
        from umapka import PkaPredictor
        pred = PkaPredictor(args.uma, multisolvent_model_path=None)
        cache = joblib.load(args.cache) if os.path.exists(args.cache) else {}
        reg = pred.regressor["regressor"]

        def score(mol, s):
            key = f"{Chem.MolToSmiles(mol)}|{s.atom}|{s.kind}"
            if cache.get(key) is None:
                cache[key] = pred.features_site(mol, s)[0]
            return float(reg.predict(np.ascontiguousarray(cache[key]).reshape(1, -1))[0])
        source = f"UMA {args.uma}"
    else:
        rs = RDKitSitePredictor().fit_sdf(f"{args.datasets}/combined_training_datasets_unique.sdf")
        score = rs.predict_site
        source = "rdkit_site"

    print(f"per-site pKas: {source}")
    rules = ("priority", "window", "window_micro", "oracle")
    for name in ("novartis", "AvLiLuMoVe"):
        path = f"{args.datasets}/{name}_cleaned_mono_unique_notraindata.sdf"
        err = {r: [] for r in rules}
        hit = {r: 0 for r in rules}
        n_multi = n = 0
        for m in Chem.SDMolSupplier(path):
            if m is None:
                continue
            p = m.GetPropsAsDict()
            mol = neutralize(m)
            sites = find_sites(mol)
            if not sites:
                continue
            try:
                pk = [score(mol, s) for s in sites]
            except Exception:
                continue
            n += 1
            n_multi += len(sites) > 1
            truth = [i for i, s in enumerate(sites) if s.atom == p["marvin_atom"]]
            model = ms.build_model(Chem.MolToSmiles(mol), dict(enumerate(pk))) if len(sites) <= 10 else None
            app = model.apparent_site_pkas() if model else pk
            app = [a if a is not None else (30.0 if s.kind == "acid" else -30.0)
                   for a, s in zip(app, sites)]
            for r in rules:
                if r == "oracle":
                    i = truth[0] if truth else 0
                elif r == "window_micro":
                    i = pick(sites, app, "window")
                else:
                    i = pick(sites, pk, r)
                hit[r] += bool(truth) and i == truth[0]
                err[r].append(abs(pk[i] - p["pKa"]))
        print(f"\n{name}: {n} molecules with >=1 SMARTS site ({n_multi} with several)")
        for r in rules:
            print(f"  {r:13s} MAE {np.mean(err[r]):.3f}   picks annotated site {hit[r]}/{n}")
    if args.uma:
        joblib.dump(cache, args.cache)


if __name__ == "__main__":
    main()
