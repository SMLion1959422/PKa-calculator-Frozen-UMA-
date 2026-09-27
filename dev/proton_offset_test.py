"""Is the per-solvent calibration offset the proton transfer term?

Background. dev/solvent_shift_experiment.py found that no model predicts
pKa in an unseen solvent (LOSO MAE ~3), but ONE measured pKa in that
solvent removes 40-60% of the error. The thermodynamic cycle says why it
could: relative to water,

    pKa(S) - pKa(W) = [dG_tr(A-) - dG_tr(HA)] / (RT ln10)  +  dG_tr(H+) / (RT ln10)
                      \_________ solute-dependent _______/     \___ solvent constant ___/

If the fitted offset IS dG_tr(H+), the empirical calibration curve
becomes a mechanistic statement: the whole unseen-solvent gap is one
physical constant, and one measurement recovers it.

Two independent tests, and the first needs no external data:

  TEST 1 (internal, decisive). Is the offset a property of the SOLVENT
  alone? Fit it on one chemical family (carboxylic acids, say) and use
  it on a different family (phenols, N-H acids, ...) in the same
  solvent. A solvent constant transfers; a fitted fudge factor does not.
  Scored against two baselines: the offset fitted on the target family
  itself (a cheating lower bound) and the offset pooled over all
  families.

  TEST 2 (external). Regress the fitted offsets on literature
  dG_tr(H+) (umapka.proton_transfer). Slope ~1 and intercept ~0 support
  the identification. Reported with and without acetonitrile, and over
  the verified-value subset only, since single-ion transfer energies
  carry real uncertainty and only some entries are cross-checked.

A caveat worth stating in any write-up: the offset absorbs everything
constant per solvent, including the mean solute transfer term over the
dataset's chemistry and any standard-state convention. Agreement is
evidence, not proof, and with <10 solvents the correlation is small-n.

    python dev/proton_offset_test.py zenodo
    python dev/proton_offset_test.py path/to/D2A-pKa.csv
"""
import argparse
import importlib.util
import os
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sse", HERE / "solvent_shift_experiment.py")
sse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sse)

from umapka.proton_transfer import DG_TR_PROTON, pk_units  # noqa: E402

WATER = "Water"
MIN_PAIRS = 8          # per solvent (and per family) to report a number
N_BOOT = 2000


def paired_shifts(d):
    """One row per (reaction, non-water solvent) that also has water:
    shift = pKa(S) - pKa(W)."""
    w = d[d.solvent == WATER].groupby("rxn").pka.mean()
    s = d[d.solvent != WATER].copy()
    s = s[s.rxn.isin(w.index)]
    s["pka_w"] = s.rxn.map(w)
    s["shift"] = s.pka - s.pka_w
    return s


def boot_ci(x, fn=np.median, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    b = [fn(rng.choice(x, len(x))) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help='"zenodo", a directory, or D2A-pKa.csv')
    ap.add_argument("--out", default="results/proton_offset")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data, _ = sse.resolve_data(args.data, None)
    d, _ = sse.load(data)
    s = paired_shifts(d)
    L = [f"paired rows (same reaction measured in water and in S): {len(s)}",
         f"solvents: {s.solvent.value_counts().to_dict()}", ""]

    # ---------------- offsets ----------------
    L.append("== fitted per-solvent offset (median shift), vs literature dG_tr(H+) ==")
    L.append(f"{'solvent':14s}{'n':>6}{'offset':>9}{'95% CI':>18}{'spread':>9}"
             f"{'lit dG_tr(H+)':>15}{'lit pK':>8}{'diff':>8}  note")
    rows = []
    for sol, g in s.groupby("solvent"):
        if len(g) < MIN_PAIRS:
            L.append(f"{sol:14s}{len(g):6d}   (too few paired measurements)")
            continue
        off = float(np.median(g["shift"]))
        lo, hi = boot_ci(g["shift"].values)
        iqr = float(np.subtract(*np.percentile(g["shift"], [75, 25])))
        e = DG_TR_PROTON.get(sol)
        lit = pk_units(sol)
        rows.append(dict(solvent=sol, n=len(g), offset=off, lo=lo, hi=hi, iqr=iqr,
                         lit_kcal=None if e is None else e.kcal_per_mol,
                         lit_pk=lit, verified=None if e is None else e.verified))
        L.append(f"{sol:14s}{len(g):6d}{off:9.2f}{f'[{lo:.2f}, {hi:.2f}]':>18}{iqr:9.2f}"
                 f"{'n/a' if e is None else f'{e.kcal_per_mol:+.1f} kcal':>15}"
                 f"{'n/a' if lit is None else f'{lit:+.2f}':>8}"
                 f"{'n/a' if lit is None else f'{off - lit:+.2f}':>8}  "
                 f"{'' if e is None else ('verified' if e.verified else 'UNVERIFIED lit value')}")
    O = pd.DataFrame(rows)
    O.to_csv(f"{args.out}/offsets.csv", index=False)
    L.append("  'spread' is the interquartile range of the per-molecule shift: the")
    L.append("  solute-dependent part the offset cannot capture.")
    L.append("")

    # ---------------- TEST 1 ----------------
    L.append("== TEST 1 (internal): does the offset transfer across chemical families? ==")
    L.append("MAE of predicted shift for a held-out family, using an offset fitted on")
    L.append("the OTHER families, vs cheating (fitted on the family itself) and vs")
    L.append("predicting no shift at all.")
    L.append(f"{'solvent':14s}{'family':20s}{'n':>6}{'other-fam':>11}{'own-fam':>9}{'no shift':>10}")
    t1 = []
    for sol, g in s.groupby("solvent"):
        if len(g) < MIN_PAIRS:
            continue
        for fam, gf in g.groupby("family"):
            other = g[g.family != fam]
            if len(gf) < MIN_PAIRS or len(other) < MIN_PAIRS or other.family.nunique() < 2:
                continue
            y = gf["shift"].values
            a = float(np.mean(np.abs(y - np.median(other["shift"]))))
            b = float(np.mean(np.abs(y - np.median(y))))
            c = float(np.mean(np.abs(y)))
            t1.append(dict(solvent=sol, family=fam, n=len(gf), other_family=a,
                           own_family=b, no_shift=c))
            L.append(f"{sol:14s}{fam:20s}{len(gf):6d}{a:11.2f}{b:9.2f}{c:10.2f}")
    T1 = pd.DataFrame(t1)
    if len(T1):
        T1.to_csv(f"{args.out}/family_transfer.csv", index=False)
        w = T1.n.values
        L.append(f"{'WEIGHTED MEAN':34s}{w.sum():6d}"
                 f"{np.average(T1.other_family, weights=w):11.2f}"
                 f"{np.average(T1.own_family, weights=w):9.2f}"
                 f"{np.average(T1.no_shift, weights=w):10.2f}")
        L.append("")
        L.append("  Reading: if 'other-fam' is close to 'own-fam', the offset is a property")
        L.append("  of the solvent, not of the chemistry -> consistent with the proton term.")
        L.append("  If 'other-fam' is much worse, the offset is absorbing solute effects too.")
    L.append("")

    # ---------------- TEST 2 ----------------
    L.append("== TEST 2 (external): offset vs literature dG_tr(H+) ==")
    have = O.dropna(subset=["lit_pk"])
    for label, sub in (("all solvents with a literature value", have),
                       ("excluding acetonitrile", have[have.solvent != "Acetonitrile"]),
                       ("verified literature values only", have[have.verified == True])):  # noqa: E712
        if len(sub) < 3:
            L.append(f"  {label}: n={len(sub)}, too few to regress")
            continue
        x, y = sub.lit_pk.values.astype(float), sub.offset.values.astype(float)
        a, b = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        L.append(f"  {label}: n={len(sub)}  slope {a:.2f}  intercept {b:+.2f}  "
                 f"r {r:.3f}  r^2 {r * r:.3f}")
        L.append("     " + ", ".join(f"{s_}: fitted {o:+.2f} vs lit {l:+.2f}"
                                     for s_, o, l in zip(sub.solvent, sub.offset, sub.lit_pk)))
    L.append("")
    L.append("  slope ~1, intercept ~0 would support 'the offset IS the proton term'.")
    L.append("  A slope far from 1 means the offset also carries a systematic solute")
    L.append("  transfer term averaged over this dataset's chemistry.")
    L.append("  Small n (<=7 solvents) and unverified literature values both limit this test.")

    text = "\n".join(L)
    print(text)
    open(f"{args.out}/summary.txt", "w").write(text + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(have) >= 3:
            fig, ax = plt.subplots(figsize=(5.5, 5))
            for _, r_ in have.iterrows():
                ax.errorbar(r_.lit_pk, r_.offset,
                            yerr=[[r_.offset - r_.lo], [r_.hi - r_.offset]],
                            fmt="o", color="C0" if r_.verified else "C1")
                ax.annotate(r_.solvent, (r_.lit_pk, r_.offset), fontsize=8,
                            xytext=(4, 4), textcoords="offset points")
            lim = [min(have.lit_pk.min(), have.offset.min()) - 1,
                   max(have.lit_pk.max(), have.offset.max()) + 1]
            ax.plot(lim, lim, "k--", lw=1, label="y = x (offset = proton term)")
            ax.set_xlabel(r"literature $\Delta G_{tr}(H^+)$ (pK units)")
            ax.set_ylabel("fitted per-solvent offset (pK units)")
            ax.set_title("Is the calibration offset the proton transfer term?")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(f"{args.out}/offset_vs_proton.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
