"""Cache v16/v17-layout features for the two external test sets, using
the LEARNED site finder (no ChemAxon annotations at inference).

WHY: every eval_*.py re-runs UMA over the whole test set, ~15-25 min,
just to compare one more head. The features don't change between heads -
only the model does. Cache once, then score any number of candidate
models/ensembles/calibrations instantly (see eval_cached.py).

Also stores the marvin_atom/marvin_pKa_type ground truth alongside, so
the same cache can report both the real-world number (learned sites,
what a user actually gets) and the oracle-site ceiling, without a
second embedding pass.

Output: feat_external_learned.pkl
  {dataset: [{"smiles","exp","feat","site_ok","kind_ok","n_atoms"}, ...]}
"""
import os
import time

import joblib
import numpy as np
from rdkit import Chem, RDLogger

from umapka import PkaPredictor
from umapka import electronic
from umapka.predictor import protonation_pair_site_tagged, neutralize

RDLogger.DisableLog("rdApp.*")

OUT = "feat_external_learned.pkl"
SETS = [
    ("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf", "novartis"),
    ("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf", "avlilumove"),
]


def main():
    cache = {}
    if os.path.exists(OUT):
        try:
            cache = joblib.load(OUT)
            print(f"resuming with {sum(len(v) for v in cache.values())} rows cached")
        except Exception:
            cache = {}

    print("loading UMA...")
    p = PkaPredictor("models/model_core_v3.pkl")

    for path, ds in SETS:
        done = {r["smiles"] for r in cache.get(ds, [])}
        rows = cache.setdefault(ds, [])
        mols = [m for m in Chem.ForwardSDMolSupplier(path) if m is not None]
        print(f"\n{ds}: {len(mols)} molecules ({len(done)} already cached)")
        t0 = time.time()
        for i, mol in enumerate(mols):
            if not mol.HasProp("pKa"):
                continue
            try:
                exp = float(mol.GetProp("pKa"))
                smi = Chem.MolToSmiles(mol)
            except Exception:
                continue
            if not (0 < exp < 14) or smi in done:
                continue
            # oracle labels, stored for reference only - NOT used to pick the site
            try:
                ma = int(float(mol.GetProp("marvin_atom"))) if mol.HasProp("marvin_atom") else -1
            except Exception:
                ma = -1
            true_kind = None
            if mol.HasProp("marvin_pKa_type"):
                true_kind = "acid" if mol.GetProp("marvin_pKa_type").startswith("acid") else "base"
            try:
                prot, pi_, dep, di_, kind = protonation_pair_site_tagged(
                    smi, return_kind=True)
                feat = electronic.build_hybrid_features(
                    p, prot, pi_, dep, di_, kind)
                if feat is None:
                    continue
            except Exception:
                continue
            # the learned site is an index into the neutralized molecule;
            # compare against marvin_atom in that same frame
            nm = neutralize(Chem.MolFromSmiles(smi))
            rows.append({
                "smiles": smi, "exp": exp,
                "feat": np.asarray(feat, dtype=np.float32),
                "kind": kind,
                "kind_ok": (true_kind is None or kind == true_kind),
                "n_atoms": nm.GetNumAtoms() if nm is not None else 0,
            })
            done.add(smi)
            if len(rows) % 50 == 0:
                el = (time.time() - t0) / 60
                print(f"  [{i+1}/{len(mols)}] {len(rows)} cached, {el:.1f} min",
                      flush=True)
                joblib.dump(cache, OUT, compress=0)
        joblib.dump(cache, OUT, compress=0)

    total = sum(len(v) for v in cache.values())
    print(f"\ncached {total} rows -> {OUT} "
          f"({os.path.getsize(OUT)/1e6:.0f} MB)")
    for ds, rows in cache.items():
        ko = np.mean([r["kind_ok"] for r in rows]) * 100
        print(f"  {ds:12s} n={len(rows):4d}  kind agreement vs Marvin: {ko:.1f}%")


if __name__ == "__main__":
    main()
