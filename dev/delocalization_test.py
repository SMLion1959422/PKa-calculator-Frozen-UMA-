"""Does a continuous delocalization descriptor predict a molecule's
sensitivity to solvent H-bond donation?

Background. The solvent shift of an acid decomposes (relative to water) as

    pKa(i,S) - pKa(i,W) = dG_tr(H+)(S)/RTln10  +  [dG_tr(A-) - dG_tr(HA)](i,S)/RTln10

The second term is anion desolvation, and it scales with the solvent's
H-bond DONOR ability alpha. How strongly should depend on how LOCALIZED
the anion's charge is: a carboxylate concentrates charge on two oxygens
and needs H-bond donors; an enolate spreads it over a pi system and needs
them less. Discrete families already show this ordering (carboxylic -6.92,
N-H -6.41, phenol -6.03, C-H -2.84) but with overlapping CIs - six
categories are too coarse to establish it.

This script replaces the category with a CONTINUOUS per-molecule
descriptor and asks whether it predicts the alpha sensitivity.

THE UMA DESCRIPTOR. Deprotonation changes UMA's per-atom embeddings.
Where it changes them measures where the electronic structure responded,
i.e. where the charge went. With heavy-atom order preserved between the
two species (umapka.predictor._ionize_keep_order), define per heavy atom

    w_a = || h(A-)_a - h(HA)_a ||          (embedding response)

and summarize its spatial extent by the bond-distance from the site:

    R_deloc = sum_a d(a, site) * w_a / sum_a w_a      (mean response radius)
    F_far   = fraction of sum(w) more than 2 bonds from the site
    IPR     = 1 / sum_a (w_a/sum w)^2      (participation ratio: how many
                                            atoms share the response)

Localized anion -> small R_deloc, small F_far, small IPR. These are
UMA-derived, physically interpretable, and need no pKa data to compute.

BASELINES it must beat, or UMA has not earned its place here:
  family      the 6 discrete categories (what we already have)
  gasteiger   IPR of the Gasteiger charge difference between A- and HA -
              the same idea from a 1980 empirical charge model, free
  topo        count of heavy atoms conjugated to the site (sp2/aromatic)

THE TEST. Using every paired row (molecule measured in water and S),
with y = shift - dG_tr(H+)(S)/RTln10 the anion transfer term:

    M0   y = a*d_alpha + b                       no molecule dependence
    M1   y = (a + c*D)*d_alpha + b + e*D         D modulates the alpha slope

and the question is whether the INTERACTION c*D*d_alpha improves the fit.
Reported as a nested F-test plus leave-one-solvent-out and
leave-one-family-out MAE, so a descriptor cannot win by memorizing either.
Molecules are kept whole across folds.

    python dev/delocalization_test.py zenodo --uma      # UMA alone; no trained model
    python dev/delocalization_test.py zenodo            # baselines only, no GPU
"""
import argparse
import importlib.util
import json
import os
import pathlib

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sse", HERE / "solvent_shift_experiment.py")
sse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sse)

from umapka.solvent_descriptors import DESCRIPTORS          # noqa: E402
from umapka.proton_transfer import pk_units                 # noqa: E402

RDLogger.DisableLog("rdApp.*")
WATER = "Water"
ALPHA_W = DESCRIPTORS["Water"][1]


# ---------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------
def _summarize(w, dist):
    """(R_deloc, F_far, IPR) from a per-atom response w and bond distances."""
    tot = float(w.sum())
    if tot <= 1e-12:
        return None
    p = w / tot
    return (float((dist * p).sum()),
            float(p[dist > 2].sum()),
            float(1.0 / np.sum(p ** 2)))


def gasteiger_descriptor(prot, deprot, site):
    """Same construction from Gasteiger partial charges: the free baseline."""
    try:
        a, b = Chem.Mol(prot), Chem.Mol(deprot)
        AllChem.ComputeGasteigerCharges(a)
        AllChem.ComputeGasteigerCharges(b)
        qa = np.array([at.GetDoubleProp("_GasteigerCharge") for at in a.GetAtoms()])
        qb = np.array([at.GetDoubleProp("_GasteigerCharge") for at in b.GetAtoms()])
        if len(qa) != len(qb) or not np.all(np.isfinite(qa)) or not np.all(np.isfinite(qb)):
            return None
        d = Chem.GetDistanceMatrix(prot)[site]
        return _summarize(np.abs(qb - qa), d)
    except Exception:
        return None


def topo_descriptor(prot, site):
    """Heavy atoms conjugated to the site: a crude delocalization count."""
    seen, stack = {site}, [site]
    while stack:
        i = stack.pop()
        for nb in prot.GetAtomWithIdx(i).GetNeighbors():
            j = nb.GetIdx()
            if j in seen:
                continue
            bond = prot.GetBondBetweenAtoms(i, j)
            conj = bond.GetIsConjugated() or bond.GetIsAromatic() or \
                nb.GetIsAromatic() or nb.GetHybridization() == Chem.HybridizationType.SP2
            if conj:
                seen.add(j)
                stack.append(j)
    return float(len(seen))


def uma_descriptor(pred, prot, deprot, site):
    """UMA embedding response: where does deprotonation change the
    per-atom representation? Heavy-atom order is preserved by
    _ionize_keep_order and _mol_to_atoms, so atom i corresponds."""
    from umapka.predictor import _mol_to_atoms
    n = prot.GetNumHeavyAtoms()
    hp = pred.embeddings(_mol_to_atoms(prot))[:n]
    hd = pred.embeddings(_mol_to_atoms(deprot))[:n]
    w = np.linalg.norm(hd - hp, axis=1)
    return _summarize(w, Chem.GetDistanceMatrix(prot)[site][:n])


# ---------------------------------------------------------------------
def build(args):
    data, _ = sse.resolve_data(args.data, None)
    d, info = sse.load(data)
    w = d[d.solvent == WATER].groupby("rxn").pka.mean()
    s = d[(d.solvent != WATER) & d.rxn.isin(w.index)].copy()
    s = s[s.solvent.map(lambda x: pk_units(x) is not None)]
    s["y"] = s.pka - s.rxn.map(w) - s.solvent.map(pk_units)     # anion transfer term
    s["d_alpha"] = s.solvent.map(lambda x: DESCRIPTORS[x][1]) - ALPHA_W
    print(f"paired rows with a proton-term value: {len(s)}; "
          f"reactions {s.rxn.nunique()}; solvents {sorted(s.solvent.unique())}")

    cache = {}
    if args.cache and os.path.exists(args.cache):
        for line in open(args.cache):
            k, v = json.loads(line)
            cache[k] = v
    pred = None
    if args.uma is not None:
        from umapka import PkaPredictor
        # the descriptors use embeddings() only, so no pKa head is required
        pred = PkaPredictor(args.uma or None, multisolvent_model_path=None)

    rows, new = [], 0
    fh = open(args.cache, "a") if args.cache else None
    for rxn in s.rxn.unique():
        if rxn in cache:
            rows.append({"rxn": rxn, **cache[rxn]})
            continue
        prot, site = info[rxn]
        deprot = None
        try:
            from umapka.predictor import _ionize_keep_order
            deprot = _ionize_keep_order(prot, site, "acid")
        except Exception:
            pass
        rec = {}
        g = gasteiger_descriptor(prot, deprot, site) if deprot is not None else None
        if g:
            rec.update(gast_R=g[0], gast_F=g[1], gast_IPR=g[2])
        rec["topo_conj"] = topo_descriptor(prot, site)
        if pred is not None and deprot is not None:
            try:
                u = uma_descriptor(pred, prot, deprot, site)
                if u:
                    rec.update(uma_R=u[0], uma_F=u[1], uma_IPR=u[2])
            except Exception as e:
                print(f"  skip {rxn}: {e}")
        cache[rxn] = rec
        rows.append({"rxn": rxn, **rec})
        new += 1
        if fh:
            fh.write(json.dumps([rxn, rec]) + "\n")
            if new % 200 == 0:
                fh.flush()
                print(f"  {new} reactions descriptorized", flush=True)
    if fh:
        fh.close()
    return s.merge(pd.DataFrame(rows), on="rxn", how="left")


# ---------------------------------------------------------------------
def design(df, D):
    """M1 design: [d_alpha, D*d_alpha, D, 1]. M0 drops columns 1 and 2."""
    a = df.d_alpha.values
    if D is None:
        return np.c_[a, np.ones(len(df))], None
    v = df[D].values.astype(float)
    v = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    return np.c_[a, v * a, v, np.ones(len(df))], v


def interaction_F(df, D, y):
    """F for adding the D*d_alpha interaction (and D) to the d_alpha-only model."""
    X0, _ = design(df, None)
    X1, _ = design(df, D)
    c0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    c1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r0 = float(np.sum((X0 @ c0 - y) ** 2))
    r1 = float(np.sum((X1 @ c1 - y) ** 2))
    dfn, dfd = X1.shape[1] - X0.shape[1], len(y) - X1.shape[1]
    if dfd <= 0 or r1 <= 0:
        return np.nan, np.nan
    return ((r0 - r1) / dfn) / (r1 / dfd), c1[1]


def permutation_p(df, D, y, n_perm=2000, seed=0):
    """Molecule-level permutation p-value.

    Each molecule contributes one row per solvent, so rows are NOT
    independent and the nominal F-test p-value is anticonservative
    (roughly by the cluster size, ~5). Permuting the descriptor ACROSS
    MOLECULES - keeping each molecule's value attached to all of its rows,
    and leaving the solvent structure untouched - gives a valid null for
    "this molecular property carries no information about the alpha slope".
    """
    obs, _ = interaction_F(df, D, y)
    if not np.isfinite(obs):
        return np.nan, np.nan
    per_mol = df.drop_duplicates("rxn").set_index("rxn")[D]
    rxns = per_mol.index.to_numpy()
    vals = per_mol.to_numpy()
    rng = np.random.default_rng(seed)
    work = df.copy()
    ge = 0
    for _ in range(n_perm):
        mapping = dict(zip(rxns, rng.permutation(vals)))
        work[D] = work.rxn.map(mapping)
        f, _ = interaction_F(work, D, y)
        if np.isfinite(f) and f >= obs:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


def fit_mae(Xtr, ytr, Xte, yte):
    c, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return float(np.mean(np.abs(Xte @ c - yte)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--uma", nargs="?", const="", default=None,
                    help="compute the UMA descriptors. Bare --uma loads UMA alone "
                         "(no trained model needed - only embeddings are used). "
                         "Optionally pass a model path. Omit entirely for baselines only.")
    ap.add_argument("--cache", default="deloc_cache.jsonl")
    ap.add_argument("--out", default="results/delocalization")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--no-perm", action="store_true",
                    help="skip the permutation test (it is the slow part)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = build(args)
    cands = [c for c in ("uma_R", "uma_F", "uma_IPR", "gast_R", "gast_F",
                         "gast_IPR", "topo_conj") if c in df.columns]
    df = df.dropna(subset=["y", "d_alpha"])
    L = [f"rows {len(df)}, reactions {df.rxn.nunique()}",
         f"descriptors available: {cands}", ""]
    df.to_csv(f"{args.out}/rows.csv", index=False)

    # baseline M0 on the complete-case rows, so every model sees the same data
    complete = df.dropna(subset=cands) if cands else df
    L.append(f"complete-case rows (all descriptors present): {len(complete)}, "
             f"reactions {complete.rxn.nunique()}")
    y = complete.y.values
    X0, _ = design(complete, None)
    c0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    rss0 = float(np.sum((X0 @ c0 - y) ** 2))
    L.append(f"M0 (no molecule dependence): slope on d_alpha {c0[0]:+.2f}, "
             f"RSS {rss0:.0f}, MAE {np.mean(np.abs(X0 @ c0 - y)):.3f}")
    L.append("")

    # family dummies, as the discrete reference point
    fam = pd.get_dummies(complete.family).values.astype(float)
    a = complete.d_alpha.values[:, None]
    Xf = np.hstack([fam * a, fam, np.ones((len(complete), 1))])
    cf, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    rssf = float(np.sum((Xf @ cf - y) ** 2))
    L.append(f"family dummies (6 slopes): RSS {rssf:.0f}, "
             f"MAE {np.mean(np.abs(Xf @ cf - y)):.3f}  [discrete reference]")
    L.append("")

    L.append("== interaction test: does D modulate the alpha slope? ==")
    L.append(f"{'descriptor':12s}{'interaction':>13}{'F':>9}{'p':>11}"
             f"{'MAE':>8}{'LOSO':>8}{'LOFO':>8}")
    res = []
    for D in cands:
        X1, _ = design(complete, D)
        c1, *_ = np.linalg.lstsq(X1, y, rcond=None)
        rss1 = float(np.sum((X1 @ c1 - y) ** 2))
        dfn, dfd = X1.shape[1] - X0.shape[1], len(y) - X1.shape[1]
        F = ((rss0 - rss1) / dfn) / (rss1 / dfd)
        p = 1 - stats.f.cdf(F, dfn, dfd)
        loso = np.mean([fit_mae(X1[complete.solvent != s], y[complete.solvent.values != s],
                                X1[complete.solvent == s], y[complete.solvent.values == s])
                        for s in complete.solvent.unique()])
        lofo = np.mean([fit_mae(X1[complete.family != f], y[complete.family.values != f],
                                X1[complete.family == f], y[complete.family.values == f])
                        for f in complete.family.unique()])
        res.append(dict(descriptor=D, interaction=c1[1], F=F, p=p,
                        mae=float(np.mean(np.abs(X1 @ c1 - y))), loso=loso, lofo=lofo))
        L.append(f"{D:12s}{c1[1]:13.3f}{F:9.1f}{p:11.2e}"
                 f"{res[-1]['mae']:8.3f}{loso:8.3f}{lofo:8.3f}")
    # M0 reference for the held-out columns
    loso0 = np.mean([fit_mae(X0[complete.solvent != s], y[complete.solvent.values != s],
                             X0[complete.solvent == s], y[complete.solvent.values == s])
                     for s in complete.solvent.unique()])
    lofo0 = np.mean([fit_mae(X0[complete.family != f], y[complete.family.values != f],
                             X0[complete.family == f], y[complete.family.values == f])
                     for f in complete.family.unique()])
    L.append(f"{'(none, M0)':12s}{'-':>13}{'-':>9}{'-':>11}"
             f"{np.mean(np.abs(X0 @ c0 - y)):8.3f}{loso0:8.3f}{lofo0:8.3f}")
    pd.DataFrame(res).to_csv(f"{args.out}/interaction.csv", index=False)
    L.append("")
    L.append("  The interaction coefficient should be POSITIVE if a larger")
    L.append("  descriptor means more delocalized: the alpha slope is negative,")
    L.append("  so a delocalized anion has a SHALLOWER (less negative) slope.")
    L.append("  LOFO is the honest column: a descriptor that only reproduces the")
    L.append("  family labels cannot help on a family it never saw.")

    # --- the decisive test: does D work WITHIN a family? ---
    L.append("")
    L.append("== WITHIN-FAMILY test (family label is constant, so useless here) ==")
    L.append("If a descriptor only encodes the family, it can do nothing here.")
    L.append(f"{'family':16s}{'rows':>6}{'descriptor':>12}{'interaction':>13}{'F':>8}{'p':>10}")
    wf = []
    for fam_name, g in complete.groupby("family"):
        if len(g) < 40 or g.rxn.nunique() < 15 or g.solvent.nunique() < 3:
            continue
        yg = g.y.values
        Xg0, _ = design(g, None)
        cg0, *_ = np.linalg.lstsq(Xg0, yg, rcond=None)
        rg0 = float(np.sum((Xg0 @ cg0 - yg) ** 2))
        for D in cands:
            if g[D].nunique() < 5:
                continue
            Xg1, _ = design(g, D)
            cg1, *_ = np.linalg.lstsq(Xg1, yg, rcond=None)
            rg1 = float(np.sum((Xg1 @ cg1 - yg) ** 2))
            dfn, dfd = Xg1.shape[1] - Xg0.shape[1], len(yg) - Xg1.shape[1]
            if dfd <= 0 or rg1 <= 0:
                continue
            F = ((rg0 - rg1) / dfn) / (rg1 / dfd)
            pv = 1 - stats.f.cdf(F, dfn, dfd)
            wf.append(dict(family=fam_name, descriptor=D, interaction=cg1[1], F=F, p=pv,
                           rows=len(g)))
            L.append(f"{fam_name:16s}{len(g):6d}{D:>12}{cg1[1]:13.3f}{F:8.1f}{pv:10.2e}")
    # --- molecule-level permutation p-values for the within-family effects ---
    if wf and not args.no_perm:
        L.append("")
        L.append("== the same, with MOLECULE-LEVEL PERMUTATION p-values ==")
        L.append("Rows are clustered by molecule (~5 solvents each), so the F-test")
        L.append("p-values above are anticonservative. Permuting the descriptor across")
        L.append("molecules gives a valid null. This is the number to quote.")
        L.append(f"{'family':16s}{'descriptor':>12}{'F':>8}{'p_perm':>10}")
        perm_rows = []
        for fam_name, g in complete.groupby("family"):
            if len(g) < 40 or g.rxn.nunique() < 15 or g.solvent.nunique() < 3:
                continue
            for D in cands:
                if g[D].nunique() < 5:
                    continue
                f, pp = permutation_p(g, D, g.y.values, n_perm=args.n_perm)
                if np.isfinite(f):
                    perm_rows.append(dict(family=fam_name, descriptor=D, F=f, p_perm=pp,
                                          n_mol=g.rxn.nunique()))
                    L.append(f"{fam_name:16s}{D:>12}{f:8.1f}{pp:10.4f}")
        if perm_rows:
            P = pd.DataFrame(perm_rows)
            P.to_csv(f"{args.out}/within_family_permutation.csv", index=False)
            L.append("")
            L.append("  families where the effect survives permutation (p<0.01):")
            for D, g in P.groupby("descriptor"):
                L.append(f"    {D:12s} {(g.p_perm < 0.01).sum()}/{len(g)}")

    if wf:
        W = pd.DataFrame(wf)
        W.to_csv(f"{args.out}/within_family.csv", index=False)
        L.append("")
        L.append("  consistency across families (a real effect should keep its sign):")
        for D, g in W.groupby("descriptor"):
            sg = np.sign(g.interaction.values)
            L.append(f"    {D:12s} sign {'+' if sg.mean() > 0 else '-'} in "
                     f"{int(max((sg > 0).sum(), (sg < 0).sum()))}/{len(sg)} families, "
                     f"significant (p<0.01) in {(g.p < 0.01).sum()}/{len(g)}")
    else:
        L.append("  (no family had enough rows/solvents for a within-family test)")

    if "uma_R" in cands and "gast_R" in cands:
        L.append("")
        L.append("== does UMA add anything over the free Gasteiger version? ==")
        Xg, _ = design(complete, "gast_R")
        cg, *_ = np.linalg.lstsq(Xg, y, rcond=None)
        rssg = float(np.sum((Xg @ cg - y) ** 2))
        vg = (complete.gast_R.values - complete.gast_R.mean()) / (complete.gast_R.std() + 1e-9)
        vu = (complete.uma_R.values - complete.uma_R.mean()) / (complete.uma_R.std() + 1e-9)
        Xb = np.c_[complete.d_alpha.values, vg * complete.d_alpha.values,
                   vu * complete.d_alpha.values, vg, vu, np.ones(len(complete))]
        cb, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        rssb = float(np.sum((Xb @ cb - y) ** 2))
        dfn, dfd = Xb.shape[1] - Xg.shape[1], len(y) - Xb.shape[1]
        F = ((rssg - rssb) / dfn) / (rssb / dfd)
        L.append(f"  adding UMA on top of Gasteiger: F {F:.1f}, "
                 f"p {1 - stats.f.cdf(F, dfn, dfd):.2e}")
        L.append(f"  correlation between uma_R and gast_R: "
                 f"{np.corrcoef(vu, vg)[0, 1]:+.3f}")

    text = "\n".join(L)
    print(text)
    open(f"{args.out}/summary.txt", "w").write(text + "\n")


if __name__ == "__main__":
    main()
