"""Ingest an external pKa dataset, QC it, and keep only the part that is
actually useful for Novartis-like chemistry.

WHY FILTERING MATTERS MORE THAN VOLUME
This repo has already measured that adding data can make things WORSE:

  v21: +6216 clean, contamination-free molecules
       -> AvLiLuMoVe 0.411 -> 0.284  (better)
       -> Novartis   0.949 -> 1.075  (worse, every size bucket)

The added molecules were size-matched to Novartis but scaffold-matched
to AvLiLuMoVe (median NN Tanimoto 0.643 vs 0.330). So "more data" is not
the goal - data that resembles the DEPLOYMENT distribution is. This
script enforces that instead of assuming it.

WHAT IT CHECKS
  1. parse + normalise SMILES, keep 0 < pKa < 14, deduplicate
  2. contamination against BOTH held-out sets (exact SMILES + >=0.95
     Tanimoto near-duplicates) - anything hitting a test molecule is
     dropped, not reported and kept
  3. novelty vs the CURRENT training pool (adding near-duplicates of what
     you already have buys nothing)
  4. size profile vs the deployment target
  5. optional --min-heavy filter for drug-like size

`--min-heavy` filters on molecular SIZE, a property of your deployment
target that you stated up front. It does NOT filter by similarity to the
test set - that would turn Novartis into a validation set and make any
later number meaningless.

USAGE
    python ingest_external_pka.py data.csv  --smiles-col smiles --pka-col pKa
    python ingest_external_pka.py data.sdf  --pka-prop pKa --min-heavy 20

Output: ingested_pka.csv  (smiles,pKa) ready for embed_core_v6.py
"""
import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

TESTS = [
    ("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf", "novartis"),
    ("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf", "avlilumove"),
]
TRAIN = "mlpka/datasets/combined_training_datasets_unique.sdf"


def load_sdf(path, pka_prop="pKa"):
    out = {}
    for m in Chem.ForwardSDMolSupplier(path):
        if m is None:
            continue
        try:
            smi = Chem.MolToSmiles(m)
            v = float(m.GetProp(pka_prop)) if m.HasProp(pka_prop) else None
        except Exception:
            continue
        if v is not None and 0 < v < 14:
            out.setdefault(smi, v)
    return out


def read_input(path, args):
    if path.lower().endswith((".sdf", ".sd")):
        return load_sdf(path, args.pka_prop)
    df = pd.read_csv(path)
    sc = args.smiles_col or next(
        (c for c in df.columns if c.lower() in ("smiles", "smi", "canonical_smiles")), None)
    pc = args.pka_col or next(
        (c for c in df.columns if "pka" in c.lower()), None)
    if sc is None or pc is None:
        raise SystemExit(f"could not find smiles/pKa columns in {list(df.columns)[:12]}"
                         f" - pass --smiles-col / --pka-col")
    print(f"  using columns: smiles='{sc}'  pKa='{pc}'")
    out = {}
    for r in df.itertuples():
        try:
            v = float(getattr(r, pc))
            m = Chem.MolFromSmiles(str(getattr(r, sc)))
        except Exception:
            continue
        if m is None or not (0 < v < 14):
            continue
        out.setdefault(Chem.MolToSmiles(m), v)
    return out


def fps(smis):
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out.append(GEN.GetFingerprint(m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--smiles-col"); ap.add_argument("--pka-col")
    ap.add_argument("--pka-prop", default="pKa")
    ap.add_argument("--min-heavy", type=int, default=0,
                    help="keep only molecules with >= this many heavy atoms "
                         "(20+ approximates drug-like / Novartis-scale)")
    ap.add_argument("--out", default="ingested_pka.csv")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        raise SystemExit(f"not found: {args.path}")

    print(f"reading {args.path} ...")
    data = read_input(args.path, args)
    print(f"  parsed, in-range, deduplicated: {len(data)}")
    if not data:
        raise SystemExit("nothing usable")

    train = load_sdf(TRAIN)
    tests = {name: load_sdf(p) for p, name in TESTS}

    # --- 2. contamination: drop, do not merely report ---
    drop = set()
    for name, t in tests.items():
        exact = set(data) & set(t)
        drop |= exact
        print(f"  exact overlap with {name}: {len(exact)}")
    tfps = {n: fps(list(t)) for n, t in tests.items()}
    keys = [k for k in data if k not in drop]
    near = 0
    for k in keys:
        m = Chem.MolFromSmiles(k)
        if m is None:
            drop.add(k); continue
        fp = GEN.GetFingerprint(m)
        if any(max(DataStructs.BulkTanimotoSimilarity(fp, f)) >= 0.95
               for f in tfps.values() if f):
            drop.add(k); near += 1
    print(f"  near-duplicates (>=0.95) of a test molecule: {near}")
    data = {k: v for k, v in data.items() if k not in drop}
    print(f"  after contamination removal: {len(data)}")

    # --- 3. novelty vs current training pool ---
    already = set(data) & set(train)
    data = {k: v for k, v in data.items() if k not in already}
    print(f"  already in current training pool: {len(already)}  -> {len(data)} new")
    if not data:
        raise SystemExit("nothing new after filtering")

    # --- 4. size profile ---
    sizes = {}
    for k in list(data):
        m = Chem.MolFromSmiles(k)
        if m is None:
            data.pop(k); continue
        sizes[k] = m.GetNumHeavyAtoms()
    sz = np.array(list(sizes.values()))
    print(f"\n  size: median {np.median(sz):.0f} heavy atoms | "
          f">30 atoms {(sz > 30).mean()*100:.1f}%")
    print(f"  reference -> novartis: median 25, 22.5% >30 | "
          f"current training: median 15, 7.0% >30")

    # --- 5. optional drug-like size filter ---
    if args.min_heavy > 0:
        data = {k: v for k, v in data.items() if sizes[k] >= args.min_heavy}
        sz = np.array([sizes[k] for k in data])
        print(f"\n  --min-heavy {args.min_heavy}: kept {len(data)}"
              + (f" | median {np.median(sz):.0f} | >30 {(sz>30).mean()*100:.1f}%"
                 if len(data) else ""))
    if not data:
        raise SystemExit("nothing left after --min-heavy")

    # --- report similarity, for information only (NOT used to filter) ---
    nfps = fps(list(data))
    print("\n  nearest-neighbour similarity of each TEST SET to this new data:")
    for name, t in tests.items():
        s = np.array([max(DataStructs.BulkTanimotoSimilarity(GEN.GetFingerprint(
            Chem.MolFromSmiles(k)), nfps)) for k in list(t)[:400]
            if Chem.MolFromSmiles(k) is not None])
        print(f"     {name:12s} median {np.median(s):.3f} | >=0.7 {(s>=0.7).mean()*100:.1f}%")
    print("  (higher for novartis than avlilumove = useful for your target;")
    print("   the reverse is the v21 failure mode - see RESULTS.md section 6)")

    pd.DataFrame({"smiles": list(data), "pKa": list(data.values())}).to_csv(
        args.out, index=False)
    print(f"\nwrote {len(data)} molecules -> {args.out}")
    print("next: add to embed_core_v6.py's SMILES list, embed, then retrain "
          "with train_v21_maxdata.py's assemble_max() pointed at this file.")


if __name__ == "__main__":
    main()
