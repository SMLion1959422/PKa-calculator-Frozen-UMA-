"""Unlock the ~8k extra training molecules that already have UMA
embeddings but no electronic descriptors.

THE OPPORTUNITY
feat_train_v6.pkl holds 13654 molecules with valid 2304-dim UMA features
(the expensive part, already paid for). feat_electronic.pkl covers only
5994, so training currently uses 5184 molecules. The missing 8165 have
pKa labels in extra_pka_data.csv and need only Gasteiger/EState
descriptors - pure RDKit, no UMA. Since the Novartis gap is a
GENERALIZATION gap (OOF gains repeatedly failed to transfer), more
diverse training data is the one lever with real headroom.

CONTAMINATION - CHECKED BEFORE USING
extra_pka_data.csv vs the two held-out sets:
    novartis   : 0 exact overlap, 1/280 at >=0.95 Tanimoto, median NN 0.330
    avlilumove : 0 exact overlap, 1/123 at >=0.95 Tanimoto, median NN 0.643
Clean. (v18 "maxdata" got WORSE on Novartis with AvLiLuMoVe at a
suspicious 0.238, so this was verified rather than assumed.)

THE SITE-CONSISTENCY TRAP
The cached UMA block for these molecules was built by embed_core_v6.py
via protonation_pair_site_tagged() as it behaved THEN - a pure
SMARTS-priority walk. That function now consults the learned ranker
first, so calling it today would return a different atom for a large
fraction of molecules, and the electronic block would describe a
different site than the UMA block sitting next to it in the same row.

So the old behaviour is reproduced exactly, by disabling the learned
path (best_site_atom -> None forces the SMARTS fallback), and the
electronic descriptors are computed on THAT site.

A molecule is only kept if the learned site finder AGREES with that old
SMARTS choice (same atom AND same acid/base kind). Where they disagree,
the SMARTS site is probably not the site the experimental pKa was
measured at, so the cached UMA features are describing the wrong
equilibrium - and training on features for site A against a label
measured at site B is label poisoning. That is a plausible reason v18's
extra data hurt Novartis, so those rows are dropped rather than trusted.

Output: feat_electronic_extra.pkl  {smiles: 81-dim vector}
"""
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

from umapka import electronic, site_finder as SF
from umapka.predictor import (ACID_SITES, BASE_SITES, neutralize,
                               protonation_pair_site_tagged)

RDLogger.DisableLog("rdApp.*")
OUT = "feat_electronic_extra.pkl"


def pair_via_smarts(smi):
    """protonation_pair_site_tagged() with the learned ranker disabled,
    i.e. exactly the SMARTS-priority behaviour embed_core_v6.py saw."""
    real = SF.best_site_atom
    SF.best_site_atom = lambda *a, **k: None
    try:
        return protonation_pair_site_tagged(smi, return_kind=True)
    finally:
        SF.best_site_atom = real


def learned_choice(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return SF.best_site_atom(neutralize(mol), ACID_SITES, BASE_SITES)


def smarts_choice_atom(smi):
    """The atom the SMARTS walk selects, in the neutralized frame, so it
    is comparable to learned_choice()."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    nm = neutralize(mol)
    for table, kind in ((ACID_SITES, "acid"), (BASE_SITES, "base")):
        for _n, sm, ai in table:
            pt = Chem.MolFromSmarts(sm)
            if pt is None:
                continue
            m = nm.GetSubstructMatches(pt)
            if m:
                return m[0][ai], kind
    return None


def main():
    U = joblib.load("feat_train_v6.pkl")
    valid = {s for s, v in U.items() if np.asarray(v).shape == (2304,)}
    have = set(joblib.load("feat_electronic.pkl"))
    extra_df = pd.read_csv("extra_pka_data.csv")

    labels = {}
    for r in extra_df.itertuples():
        try:
            v = float(r.pKa)
        except Exception:
            continue
        if 0 < v < 14:
            labels.setdefault(r.smiles, v)

    todo = [s for s in labels if s in valid and s not in have]
    print(f"UMA-featurized: {len(valid)}   already have electronic: {len(have)}")
    print(f"candidates (labelled, UMA-ready, no electronic yet): {len(todo)}")

    out, n_disagree, n_fail = {}, 0, 0
    for smi in tqdm(todo, desc="electronic"):
        try:
            old = smarts_choice_atom(smi)
            new = learned_choice(smi)
            if old is None or new is None or old != new:
                n_disagree += 1
                continue
            prot, pi_, dep, di_, _kind = pair_via_smarts(smi)
            dp = electronic.elec_desc(prot, pi_)
            dd = electronic.elec_desc(dep, di_)
            if dp is None or dd is None:
                n_fail += 1
                continue
            out[smi] = np.concatenate([dp, dd, dp - dd])
        except Exception:
            n_fail += 1

    joblib.dump(out, OUT)
    print(f"\nkept {len(out)}   dropped (site disagreement) {n_disagree}   "
          f"failed {n_fail}")
    print(f"saved -> {OUT}")
    if out:
        print(f"dim: {len(next(iter(out.values())))}  (must be 81)")
    print(f"\ntraining pool becomes ~{5184 + len(out)} molecules "
          f"(was 5184)")


if __name__ == "__main__":
    main()
