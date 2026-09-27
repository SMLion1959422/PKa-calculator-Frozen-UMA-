"""Two checks on the per-solvent offset, before anything is built on it.

CHECK A -- is the acid-class split runnable on this dataset?
The offset's leftover (after the proton term) was argued to be mean ANION
desolvation, tracking the solvent's H-bond DONOR ability (alpha). The
sharpest available test of that is a charge-type flip: a cationic acid
(BH+ -> B + H+) puts the charge on the REACTANT, as a cation, which is an
H-bond DONOR and is therefore stabilized by H-bond ACCEPTORS. So for
cationic acids the leftover should track beta rather than alpha, with a
structurally different sign. No curve fit produces that by accident.

But D2A-pKa is an ANION solvation dataset: its reactions are written
HA >> A-. If it contains no (or few) cationic acids, the test is not
runnable here and needs a different source (iBonD; the Leito basicity
scales in MeCN/THF). This check counts them rather than assuming.

CHECK B -- is the per-solvent offset confounded by composition?
The offset is a median over whatever chemistry each solvent happens to
contain, and the dataset is 60% DMSO. If solvents differ in their mix of
families, pooled offsets differ for reasons that are not about the
solvent. That is the shape of "three aprotics span 2.5 pK units with no
alpha variation". Recomputed here on a MATCHED set: only reactions
measured in water and in at least `--min-solvents` non-water solvents, so
every solvent's offset is taken over the same molecules. If the aprotic
spread shrinks, the anomaly was compositional.

    python dev/offset_confounds.py zenodo
"""
import argparse
import importlib.util
import os
import pathlib

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sse", HERE / "solvent_shift_experiment.py")
sse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sse)

RDLogger.DisableLog("rdApp.*")
WATER = "Water"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--min-solvents", type=int, default=2)
    ap.add_argument("--out", default="results/offset_confounds")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data, _ = sse.resolve_data(args.data, None)
    d, info = sse.load(data)
    L = []

    # ---------------- CHECK A ----------------
    L.append("== CHECK A: charge type of the acid in each reaction ==")
    ct = d.drop_duplicates("rxn").charge_type.value_counts().to_dict()
    L.append(f"unique reactions by charge type: {ct}")
    fam = d[d.solvent != WATER].family.value_counts().to_dict()
    L.append(f"non-water rows by family: {fam}")
    n_cat = d[d.charge_type == "cationic"].rxn.nunique()
    L.append(f"\ncationic acids (BH+ -> B + H+): {n_cat} reactions")
    if n_cat:
        sub = d[d.charge_type == "cationic"]
        L.append(f"  rows per solvent: {sub.solvent.value_counts().to_dict()}")
        w = set(sub[sub.solvent == WATER].rxn)
        pair = sub[(sub.solvent != WATER) & sub.rxn.isin(w)]
        L.append(f"  of these, paired with a water measurement: {len(pair)} rows, "
                 f"{pair.solvent.value_counts().to_dict()}")
        L.append("  -> if a solvent has >= ~20 paired cationic rows the beta/alpha")
        L.append("     flip test is runnable; otherwise it needs another dataset.")
    else:
        L.append("  -> ZERO. D2A-pKa is built from HA >> A- reactions, so the")
        L.append("     charge-type flip test CANNOT be run on this dataset.")
        L.append("     It requires cationic-acid pKa data in the same solvents:")
        L.append("     iBonD, or the Leito basicity scales (MeCN, THF).")
    L.append("")

    # ---------------- CHECK B ----------------
    s = d[d.solvent != WATER]
    w = d[d.solvent == WATER].groupby("rxn").pka.mean()
    s = s[s.rxn.isin(w.index)].copy()
    s["shift"] = s.pka - s.rxn.map(w)
    cnt = s.groupby("rxn").solvent.nunique()
    matched = set(cnt[cnt >= args.min_solvents].index)
    L.append(f"== CHECK B: offsets on all paired rows vs a MATCHED set ==")
    L.append(f"reactions measured in water and >= {args.min_solvents} non-water solvents: "
             f"{len(matched)}")
    L.append(f"{'solvent':15s}{'n_all':>7}{'off_all':>9}{'n_match':>9}{'off_match':>11}{'delta':>8}")
    rows = []
    for sol, g in s.groupby("solvent"):
        gm = g[g.rxn.isin(matched)]
        o_all = float(np.median(g["shift"]))
        o_m = float(np.median(gm["shift"])) if len(gm) >= 5 else np.nan
        rows.append(dict(solvent=sol, n_all=len(g), off_all=o_all,
                         n_match=len(gm), off_match=o_m))
        L.append(f"{sol:15s}{len(g):7d}{o_all:9.2f}{len(gm):9d}"
                 f"{'n/a' if np.isnan(o_m) else f'{o_m:.2f}':>11}"
                 f"{'' if np.isnan(o_m) else f'{o_m - o_all:+.2f}':>8}")
    B = pd.DataFrame(rows)
    B.to_csv(f"{args.out}/offsets_matched.csv", index=False)

    aprotic = ["DMSO", "DMF", "NMP", "Acetonitrile"]
    for col, lab in (("off_all", "all paired rows"), ("off_match", "matched set")):
        v = B[B.solvent.isin(aprotic)][col].dropna()
        if len(v) >= 2:
            L.append(f"  aprotic spread, {lab}: {v.max() - v.min():.2f} pK "
                     f"(min {v.min():.2f}, max {v.max():.2f}, n={len(v)})")
    L.append("  -> a smaller spread on the matched set means the anomaly was")
    L.append("     compositional, not a property of the solvents.")
    L.append("")

    # per-family offsets, to see the composition effect directly
    L.append("== per-solvent, per-family offsets (median shift) ==")
    piv = s.pivot_table(index="solvent", columns="family", values="shift", aggfunc="median")
    cnts = s.pivot_table(index="solvent", columns="family", values="shift", aggfunc="size")
    L.append(piv.round(2).to_string())
    L.append("\ncounts:")
    L.append(cnts.fillna(0).astype(int).to_string())
    L.append("\n  Family composition differing across solvents is the confound; the")
    L.append("  spread WITHIN a solvent across families bounds how much a single")
    L.append("  pooled offset can ever mean.")
    piv.to_csv(f"{args.out}/offsets_by_family.csv")

    text = "\n".join(L)
    print(text)
    open(f"{args.out}/summary.txt", "w").write(text + "\n")


if __name__ == "__main__":
    main()
