"""
Run this in your local environment (PowerShell, where rdkit/lightgbm
are already installed) from the folder containing feat_train.pkl,
feat_nov.pkl, feat_avli.pkl, pka_model_extended.pkl, and
pka_model_augmented.pkl (i.e. wherever you unzipped the notebook
export).

Goal: figure out which model file is "the" single-site model your
README's headline numbers (0.673 CV MAE etc.) refer to, by checking
which one's predictions actually match known experimental pKa values
for molecules already sitting in your own feature caches.

    python identify_model.py
"""
import joblib
from rdkit import Chem

# name -> (SMILES as you'd type it, known experimental pKa)
KNOWN = {
    "acetic acid":   ("CC(=O)O", 4.76),
    "phenol":        ("Oc1ccccc1", 9.95),
    "benzoic acid":  ("OC(=O)c1ccccc1", 4.20),
    "ethylamine":    ("CCN", 10.70),
    "pyridine":      ("c1ccncc1", 5.23),
    "ibuprofen":     ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 4.90),
}


def canon(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None


def try_model(model_path: str, feat_caches: dict):
    print(f"\n{'='*70}\n{model_path}\n{'='*70}")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"  FAILED TO LOAD: {e}")
        return

    n_checked = 0
    errs = []
    for name, (smi, exp) in KNOWN.items():
        c = canon(smi)
        feat = None
        source = None
        for cache_name, cache in feat_caches.items():
            if c in cache:
                feat = cache[c]
                source = cache_name
                break
        if feat is None:
            print(f"  {name:<14} not found in any feature cache (canonical: {c})")
            continue
        try:
            pred = float(model.predict(feat.reshape(1, -1))[0])
        except Exception as e:
            print(f"  {name:<14} predict() FAILED: {e}")
            continue
        err = abs(pred - exp)
        errs.append(err)
        n_checked += 1
        print(f"  {name:<14} pred {pred:6.2f}   exp {exp:6.2f}   "
              f"err {err:5.2f}   (from {source})")

    if errs:
        print(f"\n  checked {n_checked} molecules, mean abs error = "
              f"{sum(errs)/len(errs):.3f}")
        print("  (compare this to your README's reported MAE ~0.67-1.2 "
              "depending on split - a wildly larger number here, e.g. "
              ">3, suggests this is NOT the matching single-site model)")
    else:
        print("  no known molecules found in any cache for this model - "
              "inconclusive")


if __name__ == "__main__":
    print("Loading feature caches (SMILES -> 768-dim vector)...")
    feat_caches = {}
    for f in ["feat_train.pkl", "feat_nov.pkl", "feat_avli.pkl", "feat_ext.pkl"]:
        try:
            feat_caches[f] = joblib.load(f)
            print(f"  loaded {f}: {len(feat_caches[f])} entries")
        except FileNotFoundError:
            print(f"  {f} not found, skipping")

    for model_path in ["pka_model_extended.pkl", "pka_model_augmented.pkl"]:
        try_model(model_path, feat_caches)

    print("\n" + "=" * 70)
    print("Whichever model has the LOWER mean abs error above, and whose "
          "error magnitude is in the ballpark of your README's reported "
          "MAE, is almost certainly the one to rename to models/model_core.pkl")
