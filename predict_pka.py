#!/usr/bin/env python
"""predict_pka.py -- predict pKa for any molecule, in one pure solvent,
in a binary solvent mixture, and/or with a salt at a given molarity.

Examples
--------
  # water, no salt
  python predict_pka.py "CC(=O)O"

  # pure non-aqueous solvent
  python predict_pka.py "CC(=O)O" --solvent dmso

  # add ionic strength (physics-based Debye-Hueckel/Davies correction,
  # NOT trained on data - see umapka/solvation.py for exactly what tier
  # fires and why)
  python predict_pka.py "CC(=O)O" --salt NaCl --molarity 0.15

  # binary solvent mixture (endpoint-anchored Yasuda-Shedlovsky
  # interpolation - see umapka/mixtures.py for the method and caveats)
  python predict_pka.py "CC(=O)O" --mix water:acetonitrile --fraction 0.3

  # all sites together: macro pKas, pI, charge and species at a pH,
  # at 37 C (umapka/microstates.py)
  python predict_pka.py "NCC(=O)O" --macro --pH 7.4 --temperature 37
  #   (add --salt NaCl --molarity 0.15 for physiological ionic strength)

  # list what's supported
  python predict_pka.py --list-solvents
"""
import argparse, sys

import logging
import warnings
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(
        description="Predict pKa in any supported solvent, solvent mixture, "
                    "and/or ionic strength.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("smiles", nargs="?", help="neutral molecule SMILES")
    ap.add_argument("--prot", help="protonated SMILES (explicit)")
    ap.add_argument("--deprot", help="deprotonated SMILES (explicit)")
    ap.add_argument("--site", type=int, default=None,
                    help="predict this specific site index (see --sites); "
                        "default: first site by SMARTS priority")
    ap.add_argument("--sites", action="store_true",
                    help="list titratable sites found in --smiles and exit "
                        "(does not predict)")
    ap.add_argument("--solvent", default="water",
                    help="pure solvent name or SMILES (default: water). "
                        "Ignored if --mix is given.")
    ap.add_argument("--mix", metavar="SOLVENT_A:SOLVENT_B",
                    help="predict in a binary mixture of two pure solvents, "
                        "e.g. water:acetonitrile. Requires --fraction. See "
                        "umapka/mixtures.py for method and confidence caveats.")
    ap.add_argument("--fraction", type=float,
                    help="volume fraction of SOLVENT_B in --mix, 0-1")
    ap.add_argument("--salt", help="salt formula, e.g. NaCl (see --list-salts)")
    ap.add_argument("--molarity", type=float, dest="salt_concentration",
                    help="salt concentration in mol/L (requires --salt)")
    ap.add_argument("--model-path", default="models/model_core_v2.pkl",
                    help="aqueous regressor to use (default: model_core_v2.pkl, "
                        "the leakage-fixed model - see RESULTS.md; "
                        "model_core.pkl is the older, leaky one)")
    ap.add_argument("--macro", action="store_true",
                    help="thermodynamic multi-site prediction: couple every "
                        "site, report macroscopic pKas, isoelectric point, and "
                        "charge/species at --pH (use for zwitterions and "
                        "polyprotic molecules)")
    ap.add_argument("--pH", type=float, default=7.4,
                    help="pH for --macro species/charge output (default 7.4)")
    ap.add_argument("--temperature", type=float, default=25.0,
                    help="temperature in C for --macro (default 25)")
    ap.add_argument("--titration", metavar="CSV",
                    help="with --macro: write net charge vs pH (0-14) to CSV")
    ap.add_argument("--list-solvents", action="store_true")
    ap.add_argument("--list-salts", action="store_true")
    args = ap.parse_args()

    from umapka import solvents as sv
    from umapka import solvation as sol

    if args.list_solvents:
        print("Supported pure solvents (name: test MAE, ionic-strength support):")
        for key in sorted(sv.SOLVENTS):
            info = sv.SOLVENTS[key]
            mae = f"{info.test_mae:.2f}" if info.test_mae is not None else "insufficient test data"
            ion = "yes" if info.solvation_key else "no"
            print(f"  {info.name:<16} MAE={mae:<22} ionic-strength-correction={ion}")
        print("\nFor mixtures: --mix A:B --fraction X, where A and B are any of the")
        print("names above. Best-supported for water/organic-cosolvent pairs (see")
        print("umapka/mixtures.py MIXTURE_DIELECTRIC_DATA for pairs with a real")
        print("measured dielectric curve on file; others fall back to a cruder estimate).")
        return

    if args.list_salts:
        print("Supported salts (curated registry, umapka/solvation.py SALTS):")
        for s in sorted(sol.SALTS):
            print(f"  {s}")
        return

    if not args.smiles and not (args.prot and args.deprot):
        print("ERROR: provide SMILES, or both --prot and --deprot. "
              "Use --list-solvents / --list-salts for reference info.")
        sys.exit(1)

    if args.salt and args.salt_concentration is None:
        print("ERROR: --salt given without --molarity."); sys.exit(1)
    if args.salt_concentration is not None and not args.salt:
        print("ERROR: --molarity given without --salt."); sys.exit(1)
    if args.mix and not args.fraction and args.fraction != 0.0:
        print("ERROR: --mix given without --fraction."); sys.exit(1)

    from umapka import PkaPredictor
    p = PkaPredictor(args.model_path)

    if args.sites:
        if not args.smiles:
            print("ERROR: --sites requires a SMILES argument."); sys.exit(1)
        for s in p.sites(args.smiles):
            print(f"  [{s['index']}] {s['group']:<16} atom {s['atom']:<3} ({s['kind']})")
        return

    print(f"\nMolecule : {args.smiles or args.prot}")

    if args.macro:
        if not args.smiles:
            print("ERROR: --macro requires a SMILES argument."); sys.exit(1)
        out = p.predict_macro(args.smiles, pH=args.pH, solvent=args.solvent,
                              T_K=args.temperature + 273.15, salt=args.salt,
                              salt_concentration=args.salt_concentration,
                              return_model=bool(args.titration))
        c = out["conditions"]
        print(f"Conditions : {out['solvent']}, {args.temperature:g} C, "
              f"I = {c['ionic_strength']:.3g} M")
        print("Sites (intrinsic = others neutral; apparent = half-protonation pH):")
        for st in out["sites"]:
            ap_ = st["pKa_apparent"]
            print(f"  [{st['index']}] {st['group']:<16} {st['kind']:<5} "
                  f"intrinsic {st['pKa_intrinsic']:6.2f}   apparent "
                  f"{'   n/a' if ap_ is None else f'{ap_:6.2f}'}")
        print("Macroscopic pKas : " + ", ".join(f"{v:.2f}" for v in out["macro_pKas"]))
        pi = out["isoelectric_point"]
        print(f"Isoelectric point: {'none (charge never changes sign)' if pi is None else f'{pi:.2f}'}")
        print(f"At pH {args.pH:g}: net charge {out['net_charge']:+.2f}, "
              f"uncharged fraction {out['fraction_uncharged']:.3g}")
        for sp in out["dominant_species"]:
            print(f"  {sp['fraction']:6.1%}  {sp['smiles']}  (charge {sp['net_charge']:+d})")
        if out.get("unscored_sites"):
            print(f"WARNING: sites not scored: {out['unscored_sites']}")
        if args.titration:
            import csv
            tc = out["model"].titration_curve()
            with open(args.titration, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["pH", "net_charge", "mean_protons", "fraction_uncharged"])
                w.writerows(zip(tc["pH"], tc["net_charge"], tc["mean_protons"],
                                tc["fraction_uncharged"]))
            print(f"titration curve -> {args.titration}")
        return

    if args.mix:
        try:
            solvent_a, solvent_b = args.mix.split(":")
        except ValueError:
            print("ERROR: --mix must be SOLVENT_A:SOLVENT_B, e.g. water:acetonitrile")
            sys.exit(1)
        from umapka.mixtures import predict_mixed_solvent_pka
        result = predict_mixed_solvent_pka(
            p, args.smiles, solvent_a, solvent_b, args.fraction,
            site_index=args.site,
        )
        print(f"Mixture  : {solvent_a} / {solvent_b}  ({args.fraction:.0%} {solvent_b} by volume)")
        print(f"Predicted pKa : {result['pKa']:.2f}   "
              f"[pure {solvent_a}: {result['endpoint_a']:.2f}, "
              f"pure {solvent_b}: {result['endpoint_b']:.2f}]")
        print(f"Confidence    : {result['confidence']}  "
              f"(epsilon_mix={result['epsilon_mix']:.1f}, method={result['epsilon_method']})")
        if result["warning"]:
            print(f"WARNING: {result['warning']}")
        return

    kwargs = dict(solvent=args.solvent, salt=args.salt,
                 salt_concentration=args.salt_concentration)
    if args.site is not None:
        pka = p.predict_site(args.smiles, args.site, **kwargs)
        detail = {"pKa": pka}
    else:
        if args.prot and args.deprot:
            # explicit pair bypasses solvent/salt plumbing in predict();
            # fall back to the raw feature + base regressor for this case.
            pair_feat = p.features(args.prot, args.deprot)
            pka = p._base_pka(pair_feat, args.solvent)
            detail = {"pKa": pka, "base_pKa": pka,
                     "correction": {"shift": 0.0, "tier": "none",
                                    "note": "explicit --prot/--deprot bypasses salt correction"}}
        else:
            detail = p.predict_detailed(args.smiles, **kwargs)

    info = sv.resolve_solvent(args.solvent)
    print(f"Solvent  : {info.name}")
    print(f"Predicted pKa : {detail['pKa']:.2f}")
    if detail.get("correction", {}).get("tier", "none") != "none":
        c = detail["correction"]
        print(f"  base pKa (no salt) : {detail['base_pKa']:.2f}")
        print(f"  salt correction    : {detail['pKa'] - detail['base_pKa']:+.2f} "
              f"(tier: {c['tier']})")
        print(f"  note: {c['note']}")
    if info.test_mae is not None:
        print(f"(typical error: +/- {info.test_mae:.2f} pKa units)")
    else:
        print(f"(WARNING: {info.name} had too few test points -- treat as rough)")


if __name__ == "__main__":
    main()
