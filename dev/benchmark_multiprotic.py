"""Benchmark: does the microstate layer beat independent sites on
molecules with measured multiple pKas?

Both methods get the SAME intrinsic per-site pKas, so the comparison
isolates the thermodynamic layer:

  independent   what umapka did before v3: every site scored with the
                others neutral, pKas reported as the sorted site values
                (= predict_all_sites); charge and pI from independent
                Henderson-Hasselbalch sites
  microstate    umapka.microstates: coupled sites, macro pKas from the
                binding polynomial, conditions applied (SAMPL sets were
                measured in 0.15 M KCl, so I = 0.15 there)

Intrinsic pKas come from the GPU-free ``umapka.rdkit_site`` model,
retrained here with every benchmark molecule removed from the training
set (InChIKey skeleton match), or from a UMA model with --uma.

Matching of predicted to experimental pKas uses the Hungarian
algorithm on |error|, as in the SAMPL6 analysis (Isik et al. 2021).
Experimental pKas left unmatched are "missing"; predicted pKas inside
the 2-12 measurement window left unmatched are "extra".

Data: data/benchmark_multiprotic.csv, data/sampl6_pka.csv,
data/sampl7_pka.csv (sources in data/README.md).

Run from the repo root:
    python dev/benchmark_multiprotic.py path/to/mlpka/datasets
    python dev/benchmark_multiprotic.py path/to/mlpka/datasets --uma models/model_site_v3.pkl
Writes results/benchmark_multiprotic.{txt,csv}.
"""
import argparse
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
from rdkit import Chem, RDLogger

from umapka import microstates as ms
from umapka.sites import find_sites, neutralize
from umapka.rdkit_site import RDKitSitePredictor

RDLogger.DisableLog("rdApp.*")
WINDOW = (2.0, 12.0)


def load_sets():
    sets = {}
    rows = list(csv.DictReader(open("data/benchmark_multiprotic.csv")))
    sets["amino/polyprotic (I=0)"] = [dict(
        id=r["name"], smiles=r["smiles"], pkas=[float(x) for x in r["pKas"].split(";")],
        pI=float(r["pI"]) if r["pI"] else None, charge=int(r["charge_7_4"]),
        I=0.0) for r in rows]
    for name, path in (("SAMPL6 (I=0.15)", "data/sampl6_pka.csv"),
                       ("SAMPL7 (I=0.15)", "data/sampl7_pka.csv")):
        sets[name] = [dict(id=r["id"], smiles=r["smiles"],
                           pkas=[float(x) for x in r["pKas"].split(";")],
                           pI=None, charge=None, I=0.15)
                      for r in csv.DictReader(open(path)) if r["pKas"]]
    return sets


def skeleton_key(smiles):
    return Chem.MolToInchiKey(neutralize(Chem.MolFromSmiles(smiles)))[:14]


def match(pred, exp):
    """Hungarian matching; returns (signed errors pred-exp of matched pairs,
    n_missing, n_extra_in_window)."""
    if not pred:
        return [], len(exp), 0
    signed = np.subtract.outer(np.array(pred), np.array(exp)).T   # pred - exp
    cost = np.abs(signed)
    r, c = linear_sum_assignment(cost)
    errs = signed[r, c].tolist()
    unmatched_pred = [p for j, p in enumerate(pred) if j not in set(c)]
    extra = sum(WINDOW[0] <= p <= WINDOW[1] for p in unmatched_pred)
    return errs, len(exp) - len(r), extra


def independent_model(m):
    return ms.MicrostateModel(m.smiles, m.sites, m.pka_intrinsic,
                              np.zeros_like(m.W), m.ionic_strength,
                              m.eps_solvent, m.T_K, _mol=m._mol)


def evaluate(entry, intrinsic_fn):
    mol = neutralize(Chem.MolFromSmiles(entry["smiles"]))
    sites = find_sites(mol)
    if not sites:
        return None
    intr = intrinsic_fn(entry["smiles"], mol, sites)
    if not intr:
        return None
    coupled = ms.build_model(entry["smiles"], intr, ionic_strength=entry["I"])
    indep = independent_model(ms.build_model(entry["smiles"], intr))
    out = {"id": entry["id"], "n_sites": len(sites), "exp": entry["pkas"]}
    models = {"independent": indep, "microstate": coupled}
    preds = {"independent": sorted(intr.values()),
             "microstate": sorted(coupled.macro_pkas())}
    if entry["I"] > 0:   # control: same coupling, no ionic-strength terms
        models["microstate_I0"] = ms.build_model(entry["smiles"], intr)
        preds["microstate_I0"] = sorted(models["microstate_I0"].macro_pkas())
    for name, model in models.items():
        errs, miss, extra = match(preds[name], entry["pkas"])
        out[name] = dict(pred=preds[name], errs=errs, missing=miss, extra=extra,
                         pI=model.isoelectric_point(),
                         charge=model.net_charge(7.4))
    return out


def summarize(results, has_pi):
    lines = []
    for name in [m for m in ("independent", "microstate", "microstate_I0") if m in results[0]]:
        signed = [e for r in results for e in r[name]["errs"]]
        errs = np.abs(signed)
        miss = sum(r[name]["missing"] for r in results)
        extra = sum(r[name]["extra"] for r in results)
        line = (f"  {name:14s} MAE {np.mean(errs):.2f}  RMSE {np.sqrt(np.mean(np.square(errs))):.2f}"
                f"  bias {np.mean(signed):+.2f}"
                f"  matched {len(errs)}  missing {miss}  extra-in-window {extra}")
        if has_pi:
            pis = [(r[name]["pI"], r["pI"]) for r in results if r["pI"] is not None]
            pi_err = [abs(p - e) if p is not None else np.nan for p, e in pis]
            ch = [round(r[name]["charge"]) == r["charge"] for r in results if r["charge"] is not None]
            line += (f"\n  {'':14s} pI MAE {np.nanmean(pi_err):.2f} (n={len(pis)})"
                     f"  charge@7.4 correct {sum(ch)}/{len(ch)}")
        lines.append(line)
    # paired per-molecule comparison on molecules both methods scored
    a, b = [], []
    for r in results:
        if r["independent"]["errs"] and r["microstate"]["errs"]:
            a.append(np.mean(np.abs(r["independent"]["errs"])))
            b.append(np.mean(np.abs(r["microstate"]["errs"])))
    a, b = np.array(a), np.array(b)
    d = b - a
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    changed = np.abs(d) > 1e-9
    p = wilcoxon(d[changed]).pvalue if changed.sum() >= 5 else float("nan")
    lines.append(f"  paired per-molecule MAE change (microstate - independent): "
                 f"{d.mean():+.2f} [95% CI {lo:+.2f}, {hi:+.2f}], "
                 f"better {int((d < -1e-9).sum())} / worse {int((d > 1e-9).sum())} / "
                 f"same {int((~changed).sum())}, Wilcoxon p={p:.3g}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", help="Machine-learning-meets-pKa/datasets directory")
    ap.add_argument("--uma", help="use this PkaPredictor model for intrinsic pKas")
    args = ap.parse_args()

    sets = load_sets()
    keys = {skeleton_key(e["smiles"]) for s in sets.values() for e in s}
    if args.uma:
        from umapka import PkaPredictor
        pred = PkaPredictor(args.uma)
        source = f"UMA model {args.uma}"

        def intrinsic_fn(smiles, mol, sites):
            out = {}
            for s in sites:
                try:
                    out[s.index] = pred.predict_site(smiles, s.index)
                except Exception:
                    pass
            return out
    else:
        rs = RDKitSitePredictor().fit_sdf(
            f"{args.datasets}/combined_training_datasets_unique.sdf", exclude_inchikeys=keys)
        source = (f"rdkit_site model, {rs.n_train} training molecules, "
                  f"{rs.n_excluded} removed as benchmark overlaps")

        def intrinsic_fn(smiles, mol, sites):
            return {s.index: rs.predict_site(mol, s) for s in sites}

    report = [f"intrinsic pKas: {source}", ""]
    table = []
    for set_name, entries in sets.items():
        results, unscored = [], []
        for e in entries:
            r = evaluate(e, intrinsic_fn)
            if r is None or not any(r[m]["errs"] for m in ("independent", "microstate")):
                unscored.append(e["id"])
                continue
            r["pI"], r["charge"] = e["pI"], e["charge"]
            results.append(r)
            for m in [k for k in ("independent", "microstate", "microstate_I0") if k in r]:
                table.append({"set": set_name, "id": r["id"], "method": m,
                              "exp": ";".join(f"{x:.2f}" for x in r["exp"]),
                              "pred": ";".join(f"{x:.2f}" for x in r[m]["pred"]),
                              "mean_abs_err": f"{np.mean(np.abs(r[m]['errs'])):.3f}" if r[m]["errs"] else "",
                              "pI": "" if r[m]["pI"] is None else f"{r[m]['pI']:.2f}",
                              "charge_7_4": f"{r[m]['charge']:.2f}"})
        n_exp = sum(len(e["pkas"]) for e in entries)
        report.append(f"{set_name}: {len(results)}/{len(entries)} molecules scored, "
                      f"{n_exp} experimental pKas"
                      + (f"; no SMARTS site: {', '.join(unscored)}" if unscored else ""))
        report += summarize(results, any(e["pI"] for e in entries))
        report.append("")

    text = "\n".join(report)
    print(text)
    open("results/benchmark_multiprotic.txt", "w").write(text + "\n")
    with open("results/benchmark_multiprotic.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


if __name__ == "__main__":
    main()
