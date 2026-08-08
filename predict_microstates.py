"""FULL MICROSTATE ENSEMBLE for polyprotic molecules - the Uni-pKa
formulation applied post-hoc to independently predicted transitions.

WHY THIS BEATS INDEPENDENT-SITE PREDICTION
Predicting every site against the same neutral background gets aspartic
acid's amine pKa badly wrong (7.46 vs a real ~9.9), because by the time
the amine ionizes both carboxylates already carry negative charge, which
stabilizes the ammonium and RAISES its pKa. Independent prediction
cannot see that. Here every microstate is built explicitly, so each
transition is predicted on the charge state that actually exists.

THE MATH
  - state s = binary vector, s[i]=1 if site i is protonated
  - every transition s -> s\\i gives log10(beta_s) - log10(beta_s\\i) = pKa_i
  - that is an overdetermined linear system in log10(beta); solving it
    gives a thermodynamically consistent, path-independent solution
  - macro pKa for the m-th proton = log10(Z_m / Z_(m-1)) from the
    partition function, NOT from one greedy path

The solver now lives in umapka/microstates.py (shared with
predict_ladder.py and build_polyprotic_benchmark.py, which previously
each carried their own drifting copy). It is ridge-regularized so a
microstate that fails to build cannot throw the whole ladder, and it
propagates a PER-RUNG standard error from the cycle residual - the
residual correlates ~0.51 with real error, so it is genuine signal.

HONEST LIMITATION - READ BEFORE TRUSTING A NUMBER
The underlying transition model is trained on MONO-ionizable molecules
(mostly neutral <-> +/-1). Predicting on a dianion background is
extrapolation. On the repo's polyprotic benchmark the overall MAE is
~1.8, and that benchmark itself has label-assignment noise (it can treat
two measurements of ONE site as two different sites). Treat these
values as directional, and read the reported sigma and residual.

USAGE
  python predict_microstates.py "NCC(=O)O"
  python predict_microstates.py "OC(=O)CC(N)C(=O)O" --ph 7.4
"""
import argparse
import sys

import joblib
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")

from umapka import PkaPredictor
from umapka import electronic, microstates as M
from umapka.predictor import (ACID_SITES, BASE_SITES, neutralize,
                               _smiles_to_atoms_with_site)

TAG0 = M.TAG0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("smiles")
    ap.add_argument("--ph", type=float, default=7.4)
    ap.add_argument("--model", default="models/model_core_v16_elec.pkl")
    ap.add_argument("--max-sites", type=int, default=5,
                    help="cap on titratable sites (2**n microstates); "
                         "lowest-ranked sites beyond this are dropped")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    if not electronic.is_hybrid_bundle(bundle):
        sys.exit(f"{args.model} is not a hybrid (gbm+ridge) bundle")

    print("loading UMA...")
    p = PkaPredictor("models/model_core_v3.pkl")

    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        sys.exit(f"could not parse SMILES: {args.smiles}")
    nm = neutralize(mol)
    found = M.all_sites(nm, ACID_SITES, BASE_SITES)
    if not found:
        sys.exit("no ionizable sites found")
    sites, dropped = M.filter_sites(nm, found, ACID_SITES, BASE_SITES,
                                    max_sites=args.max_sites)
    n = len(sites)

    print(f"\nMolecule : {args.smiles}")
    print(f"Sites    : {n}   microstates: {2**n}")
    for i, (name, kind, idx) in enumerate(sites):
        print(f"  [{TAG0+i}] {name:22s} atom {idx:3d}  {kind}")
    if dropped:
        print(f"  (dropped for tractability, --max-sites={args.max_sites}: "
              f"{', '.join(d[0] for d in dropped)})")

    neutral_tagged, states, smi_of = M.enumerate_microstates(nm, sites)
    print(f"\nbuilt {len(smi_of)}/{len(states)} microstates")

    # ---- embed each microstate ONCE, pool per site from the same pass ----
    print("embedding microstates (one UMA call each)...")
    cache = {}
    for s, tagged in smi_of.items():
        try:
            clean = M.strip_tags_text(tagged)
            tagged_mol = Chem.MolFromSmiles(tagged)
            clean_mol = Chem.MolFromSmiles(clean)
            if tagged_mol is None or clean_mol is None:
                continue
            tag_idx = {a.GetAtomMapNum(): a.GetIdx() for a in tagged_mol.GetAtoms()
                       if a.GetAtomMapNum() >= TAG0}
            if not tag_idx:
                continue
            # tag stripping must not have reordered atoms
            if not all(clean_mol.GetAtomWithIdx(v).GetSymbol()
                       == tagged_mol.GetAtomWithIdx(v).GetSymbol()
                       for v in tag_idx.values()):
                continue
            atoms, _, mol_h = _smiles_to_atoms_with_site(
                clean, next(iter(tag_idx.values())))
            cache[s] = {"clean": clean, "emb": p.embeddings(atoms),
                        "mol_h": mol_h, "tag_idx": tag_idx}
            print(f"  {''.join(map(str,s))}  "
                  f"q={Chem.GetFormalCharge(clean_mol):+d}  {clean[:44]}")
        except Exception as exc:
            print(f"  {''.join(map(str,s))}  FAILED: {type(exc).__name__}")

    # ---- predict every single-proton transition ----
    print("\npredicting transitions...")
    trans = []
    for s in states:
        if s not in cache:
            continue
        for i in range(n):
            if s[i] != 1:
                continue
            s2 = list(s); s2[i] = 0; s2 = tuple(s2)
            if s2 not in cache:
                continue
            cp, cd = cache[s], cache[s2]
            tag = TAG0 + i
            try:
                ip, id_ = cp["tag_idx"][tag], cd["tag_idx"][tag]
                hg_p = p.pool(cp["emb"])
                hl_p = p.pool_local_multiscale(cp["emb"], ip, cp["mol_h"])
                hg_d = p.pool(cd["emb"])
                hl_d = p.pool_local_multiscale(cd["emb"], id_, cd["mol_h"])
                g_ = np.concatenate([hg_p, hg_d, hg_p - hg_d])
                l_ = np.concatenate([hl_p, hl_d, hl_p - hl_d])
                dp = electronic.elec_desc(cp["clean"], ip)
                dd = electronic.elec_desc(cd["clean"], id_)
                if dp is None or dd is None:
                    continue
                feat = np.nan_to_num(
                    np.concatenate([g_, l_, dp, dd, dp - dd])).reshape(1, -1)
                pk = electronic.score_hybrid(bundle, feat)
            except Exception:
                continue
            trans.append((s, s2, i, pk))
            print(f"  {''.join(map(str,s))} -> {''.join(map(str,s2))}  "
                  f"{sites[i][0]:20s} micro-pKa {pk:6.2f}")

    if not trans:
        sys.exit("\nno transitions could be predicted")

    # ---- thermodynamically consistent solve (ridge-regularized) ----
    logbeta, rms, sigma = M.solve_logbeta(states, trans, n)
    macro, macro_sig, Z = M.macro_pka(states, logbeta, sigma, n, set(cache))

    print("\n--- THERMODYNAMIC CONSISTENCY ---")
    print(f"  transitions: {len(trans)}   free parameters: {len(states)-1}")
    print(f"  RMS residual: {rms:.3f} pKa units")
    print("  (how inconsistent the raw predictions were across different")
    print("   paths - correlates ~0.51 with real error, so large = distrust)")

    print("\n--- MACRO pKa (partition function) ---")
    if not macro:
        print("  none resolvable - too few microstates built")
    for i, (pk, sg) in enumerate(zip(macro, macro_sig), 1):
        flag = "   [!] high uncertainty" if sg > 1.0 else ""
        print(f"  pKa{i} = {pk:6.2f}  +/- {sg:.2f}{flag}")
    if macro and macro != sorted(macro):
        print("\n  [!] rungs are not monotonically increasing, which is")
        print("      thermodynamically impossible - treat as unreliable.")

    if not Z:
        return
    print(f"\n--- SPECIES DISTRIBUTION AT pH {args.ph} ---")
    d = M.population(Z, args.ph)
    for m in sorted(d, reverse=True):
        print(f"  {m} proton(s)  {d[m]*100:6.2f}%  {'#'*int(round(d[m]*40))}")

    print("\n--- pH PROFILE (by proton count) ---")
    ms = sorted(Z.keys(), reverse=True)
    print(f"{'pH':>5s}  " + "  ".join(f"{m}H".rjust(6) for m in ms))
    for ph in [1, 2, 3, 4, 5, 6, 7, 7.4, 8, 9, 10, 11, 12, 13]:
        dd_ = M.population(Z, ph)
        print(f"{ph:5.1f}  " + "  ".join(f"{dd_.get(m,0)*100:5.1f}%" for m in ms))

    print("""
METHOD: full 2^n microstate ensemble, ridge-regularized least-squares
thermodynamic consistency, macro pKa from the partition function. The
transition model is trained on mono-ionizable molecules, so multiply-
charged backgrounds are extrapolation - read the sigma and residual.""")


if __name__ == "__main__":
    main()
