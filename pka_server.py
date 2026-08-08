"""Fast repeated pKa prediction: load the model ONCE, answer many queries.

WHY THIS EXISTS
Timing a single `python predict_pka.py <smiles>` on CPU:

    import torch                24.10 s   one-time
    import umapka                6.52 s   one-time
    load UMA checkpoint         83.50 s   one-time   <-- the real cost
    load regressor head          1.49 s   one-time
    features (2 UMA passes)      1.16 s   PER MOLECULE
    score head                   0.007s   PER MOLECULE
    ----------------------------------------------
    warm, per molecule           0.68 s

The prediction is already sub-second. What is slow is re-reading a
1.17 GB UMA checkpoint for every invocation. Nothing about the model,
the features, or the accuracy needs to change - the process just has to
stay alive. Same weights, same features, byte-identical predictions.

MODES
    # interactive: one SMILES per line on stdin, blank line or Ctrl-D to exit
    python pka_server.py

    # batch: a file with one SMILES per line (or a CSV with a smiles column)
    python pka_server.py --batch molecules.txt --out results.csv

    # any pure solvent / salt supported by predict_pka.py
    python pka_server.py --batch mols.txt --solvent dmso

A persistent SMILES->pKa cache (--cache) makes repeat queries instant,
which matters when the same scaffolds recur across a screening run.
"""
import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser(
        description="Load the pKa model once, then predict many molecules fast.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--batch", help="file of SMILES (one per line, or CSV with a smiles column)")
    ap.add_argument("--out", help="write results here as CSV (batch mode)")
    ap.add_argument("--solvent", default="water")
    ap.add_argument("--salt"); ap.add_argument("--molarity", type=float)
    ap.add_argument("--model-path", default="models/model_core_v3.pkl")
    ap.add_argument("--hybrid-model-path", default="models/model_core_v20_ensemble.pkl")
    ap.add_argument("--cache", default=".pka_cache.json",
                    help="persistent SMILES->pKa cache; '' disables")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import logging, warnings
    logging.getLogger().setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    t0 = time.time()
    import joblib
    from umapka import PkaPredictor, electronic, solvents as sv
    from umapka.predictor import protonation_pair_site_tagged
    p = PkaPredictor(args.model_path)
    try:
        bundle = joblib.load(args.hybrid_model_path)
        if not (electronic.is_hybrid_bundle(bundle)
                or electronic.is_ensemble_bundle(bundle)):
            bundle = None
    except Exception:
        bundle = None
    load_s = time.time() - t0
    is_water = sv.resolve_solvent(args.solvent).name == "Water"
    if not args.quiet:
        print(f"model loaded in {load_s:.1f}s "
              f"({'v20 ensemble' if bundle is not None else 'base regressor'}, "
              f"{args.solvent})", file=sys.stderr)
        if not is_water or args.salt:
            print("  note: non-water/salt requests use the base regressor path",
                  file=sys.stderr)

    cache = {}
    if args.cache and os.path.exists(args.cache):
        try:
            cache = json.load(open(args.cache))
        except Exception:
            cache = {}

    def predict(smi):
        key = f"{smi}|{args.solvent}|{args.salt}|{args.molarity}"
        if key in cache:
            return cache[key], True
        # the hybrid ensemble is the accurate path, and it is water/no-salt
        # only - exactly the routing predict_pka.py uses, so numbers match
        if bundle is not None and is_water and not args.salt:
            prot, pi_, dep, di_, kind = protonation_pair_site_tagged(
                smi, return_kind=True)
            feat = electronic.build_hybrid_features(p, prot, pi_, dep, di_, kind)
            if feat is None:
                raise RuntimeError("feature extraction failed")
            val = electronic.score_any(bundle, feat)
        else:
            val = p.predict(smi, solvent=args.solvent, salt=args.salt,
                            salt_concentration=args.molarity)
        cache[key] = float(val)
        return float(val), False

    def save_cache():
        if args.cache:
            try:
                json.dump(cache, open(args.cache, "w"))
            except Exception:
                pass

    # ---------------- batch ----------------
    if args.batch:
        smis = []
        if args.batch.lower().endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(args.batch)
            col = next((c for c in df.columns if c.lower() in
                        ("smiles", "smi", "canonical_smiles")), df.columns[0])
            smis = [str(x) for x in df[col].dropna()]
        else:
            smis = [ln.strip() for ln in open(args.batch) if ln.strip()
                    and not ln.startswith("#")]
        print(f"predicting {len(smis)} molecules...", file=sys.stderr)

        rows, t0, n_hit, n_fail = [], time.time(), 0, 0
        for i, smi in enumerate(smis):
            try:
                val, hit = predict(smi)
                n_hit += hit
                rows.append({"smiles": smi, "pKa": round(val, 2), "error": ""})
            except Exception as exc:
                n_fail += 1
                rows.append({"smiles": smi, "pKa": "", "error": type(exc).__name__})
            if not args.quiet and (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(smis)}  {el:.1f}s  "
                      f"({el/(i+1):.2f}s each)", file=sys.stderr, flush=True)
        el = time.time() - t0
        save_cache()

        if args.out:
            import pandas as pd
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"wrote {args.out}", file=sys.stderr)
        else:
            for r in rows:
                print(f"{r['smiles']}\t{r['pKa']}\t{r['error']}")
        ok = len(smis) - n_fail
        print(f"\n{ok} predicted, {n_fail} failed, {n_hit} from cache",
              file=sys.stderr)
        if ok:
            print(f"{el:.1f}s total = {el/max(ok,1):.2f}s per molecule "
                  f"(vs ~{load_s:.0f}s startup if each were its own process)",
                  file=sys.stderr)
        return

    # ---------------- interactive ----------------
    print("Enter a SMILES per line (blank line or Ctrl-D to quit).", file=sys.stderr)
    while True:
        try:
            line = input("smiles> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        t = time.time()
        try:
            val, hit = predict(line)
            print(f"  pKa = {val:.2f}   ({time.time()-t:.2f}s"
                  f"{', cached' if hit else ''})")
        except Exception as exc:
            print(f"  ERROR {type(exc).__name__}: {exc}")
    save_cache()
    print("bye", file=sys.stderr)


if __name__ == "__main__":
    main()
