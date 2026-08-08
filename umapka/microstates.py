"""Shared microstate-ensemble solver for polyprotic molecules (the
Uni-pKa formulation, applied post-hoc to independently predicted
single-proton transitions).

Extracted here because predict_microstates.py and
build_polyprotic_benchmark.py each had their own copy of this solver -
so any fix had to be made twice, and the two copies had already drifted.

Three defects in those copies are fixed here:

1. SPURIOUS SITES. The SMARTS tables over-detect. A nitro group's
   nitrogen matched "aniline_tert" until that pattern was given a +0
   charge constraint, adding a fake ionizable site to every nitroarene -
   which was the direct cause of the worst errors in
   polyprotic_benchmark_v16.csv. Every spurious site doubles the
   microstate count and injects transitions the model was never trained
   for. filter_sites() uses the learned site-finder's own ranking to
   drop candidates it scores far below the best one, so enumeration
   starts from sites the ranker actually believes in.

2. UNREGULARIZED LEAST SQUARES. np.linalg.lstsq on the transition
   matrix is ill-conditioned whenever some microstates fail to build
   (common - strained or unsanitizable protonation states), because the
   affected log-beta parameters become weakly determined or entirely
   unconstrained, and lstsq happily returns huge values for them. A
   small ridge term keeps those bounded instead of letting one
   unbuildable microstate throw the whole ladder.

3. NO UNCERTAINTY. The cycle residual was reported as a single global
   number, though it correlates 0.51 with per-molecule error - i.e. it
   carries real signal. macro_pka() now propagates it into a per-rung
   standard error, so a caller can tell which specific rung is shaky
   rather than only that the molecule overall is.
"""
from __future__ import annotations

import itertools

import numpy as np
from rdkit import Chem

TAG0 = 101


def all_sites(mol, acid_sites, base_sites):
    """Every atom matched by any ionizable SMARTS, as
    (pattern_name, kind, atom_idx). First match per atom wins."""
    out, seen = [], set()
    for name, sm, ai in acid_sites:
        pt = Chem.MolFromSmarts(sm)
        if pt is None:
            continue
        for m in mol.GetSubstructMatches(pt):
            if m[ai] not in seen:
                seen.add(m[ai])
                out.append((name, "acid", m[ai]))
    for name, sm, ai in base_sites:
        pt = Chem.MolFromSmarts(sm)
        if pt is None:
            continue
        for m in mol.GetSubstructMatches(pt):
            if m[ai] not in seen:
                seen.add(m[ai])
                out.append((name, "base", m[ai]))
    return out


def filter_sites(mol, sites, acid_sites, base_sites, max_sites=5):
    """Cap the site count so 2**n microstates stays tractable, keeping
    the sites the learned ranker scores highest.

    DELIBERATELY DOES NOT DROP SITES BY SCORE MARGIN. That was tried and
    is wrong in principle: site_finder's model is a LambdaRank ranker
    trained to put the ONE ChemAxon-annotated site on top of
    mono-ionizable molecules, so it is trained to push every OTHER
    genuine site down. Its scores rank candidates; they carry no
    calibrated notion of "this is not a real site." Measured directly:
    a margin filter dropped BOTH carboxylic acids from aspartic acid
    (leaving 1 of 3 real sites) and the carboxyl from lysine - i.e. it
    destroyed exactly the polyprotic molecules it was meant to help.

    The right fix for spurious sites is at the SMARTS level, where the
    false positive actually originates - e.g. adding the +0 charge
    constraint to the aniline patterns so a nitro nitrogen stops
    matching as a basic site. That is a real, checkable chemistry
    statement; a score threshold is not.

    Ranker order is still used for WHICH sites survive the max_sites
    truncation, since if some must be dropped for tractability, dropping
    the lowest-ranked is better than dropping arbitrary ones.

    Returns (kept_sites, dropped_sites); dropped is non-empty only when
    len(sites) > max_sites.
    """
    if len(sites) <= max_sites:
        return sites, []
    from . import site_finder
    ranked = site_finder.rank_candidates(mol, acid_sites, base_sites)
    if ranked:
        score = {idx: sc for idx, _kind, sc in ranked}
        ordered = sorted(sites, key=lambda s: -score.get(s[2], -1e9))
    else:
        ordered = list(sites)
    return ordered[:max_sites], ordered[max_sites:]


def strip_tags_text(smi: str) -> str:
    """Remove ':NNN' atom-map annotations by TEXT substitution.

    Never re-canonicalize to strip tags: MolToSmiles reorders atoms, so
    an index found while tags were present is invalid for a freshly
    canonicalized string. That exact bug made aniline predict 9.88
    instead of 4.45 (see predictor._strip_map_tag for the long version).
    """
    import re
    return re.sub(r":\d+\]", "]", smi)


def shift(smiles, tag, d_h, d_q):
    """Add/remove one H at the atom carrying `tag`, preserving all tags."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    idx = next((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == tag), None)
    if idx is None:
        return None
    rw = Chem.RWMol(mol)
    a = rw.GetAtomWithIdx(idx)
    n_h = a.GetTotalNumHs() + d_h
    if n_h < 0:
        return None
    a.SetNumExplicitHs(n_h)
    a.SetNoImplicit(True)
    a.SetFormalCharge(a.GetFormalCharge() + d_q)
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return Chem.MolToSmiles(out)
    except Exception:
        return None


def build_state(neutral_tagged, sites, state):
    """Apply a protonation pattern: base protonated -> +1H/+1q,
    acid deprotonated -> -1H/-1q."""
    smi = neutral_tagged
    for i, (_name, kind, _idx) in enumerate(sites):
        tag = TAG0 + i
        if kind == "base" and state[i] == 1:
            smi = shift(smi, tag, +1, +1)
        elif kind == "acid" and state[i] == 0:
            smi = shift(smi, tag, -1, -1)
        if smi is None:
            return None
    return smi


def solve_logbeta(states, transitions, n_sites, ridge=1e-3):
    """Least-squares solve for thermodynamically consistent log-beta.

    `transitions` is [(state_from, state_to, site_i, pKa), ...]. Each
    contributes log10(beta_from) - log10(beta_to) = pKa. The fully
    deprotonated state is the reference (log-beta 0).

    Ridge-regularized (see module docstring, defect 2): with some
    microstates unbuildable, plain lstsq can return unbounded values for
    weakly-determined parameters. Returns (logbeta, rms_residual,
    per_param_sigma).
    """
    idx_of = {s: i for i, s in enumerate(states)}
    ref = idx_of[tuple([0] * n_sites)]
    keep = [j for j in range(len(states)) if j != ref]
    col = {j: k for k, j in enumerate(keep)}

    A = np.zeros((len(transitions), len(keep)))
    b = np.zeros(len(transitions))
    for r, (s_from, s_to, _i, pk) in enumerate(transitions):
        if idx_of[s_from] in col:
            A[r, col[idx_of[s_from]]] = 1.0
        if idx_of[s_to] in col:
            A[r, col[idx_of[s_to]]] = -1.0
        b[r] = pk

    # ridge: append sqrt(lambda) * I rows, targeting 0
    lam = float(ridge)
    A_aug = np.vstack([A, np.sqrt(lam) * np.eye(len(keep))])
    b_aug = np.concatenate([b, np.zeros(len(keep))])
    sol, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)

    logbeta = np.zeros(len(states))
    logbeta[keep] = sol
    resid = A @ sol - b
    dof = max(1, len(transitions) - len(keep))
    rms = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else 0.0
    s2 = float(resid @ resid) / dof
    try:
        cov = np.linalg.pinv(A.T @ A + lam * np.eye(len(keep))) * s2
        sigma = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        sigma = np.full(len(keep), np.nan)
    sig_full = np.zeros(len(states))
    sig_full[keep] = sigma
    return logbeta, rms, sig_full


def macro_pka(states, logbeta, sigma, n_sites, built):
    """Macro pKa values from the partition function, with a per-rung
    standard error propagated from the least-squares fit.

    Returns (macro_pkas, macro_sigmas, Z) where Z[m] is log10 of the
    partition function over states with m protons.
    """
    idx_of = {s: i for i, s in enumerate(states)}
    Z, Zs = {}, {}
    for m in range(n_sites + 1):
        vals, sg = [], []
        for s in states:
            if sum(s) == m and s in built:
                vals.append(logbeta[idx_of[s]])
                sg.append(sigma[idx_of[s]])
        if vals:
            mx = max(vals)
            w = [10.0 ** (v - mx) for v in vals]
            tot = sum(w)
            Z[m] = mx + np.log10(tot)
            # weighted-average sigma over the states that dominate Z_m
            Zs[m] = float(np.sqrt(sum((wi / tot) ** 2 * si ** 2
                                      for wi, si in zip(w, sg))))
    macro, macro_sig = [], []
    for m in range(n_sites, 0, -1):
        if m in Z and (m - 1) in Z:
            macro.append(Z[m] - Z[m - 1])
            macro_sig.append(float(np.hypot(Zs.get(m, 0.0), Zs.get(m - 1, 0.0))))
    return macro, macro_sig, Z


def population(Z, ph):
    """Fraction of molecules with m protons at a given pH."""
    ex = {m: Z[m] - m * ph for m in Z}
    mx = max(ex.values())
    w = {m: 10.0 ** (ex[m] - mx) for m in ex}
    t = sum(w.values())
    return {m: w[m] / t for m in w}


def enumerate_microstates(mol, sites):
    """(neutral_tagged_smiles, [states], {state: tagged_smiles})."""
    rw = Chem.RWMol(mol)
    for i, (_n, _k, idx) in enumerate(sites):
        rw.GetAtomWithIdx(idx).SetAtomMapNum(TAG0 + i)
    neutral_tagged = Chem.MolToSmiles(rw.GetMol())
    states = list(itertools.product([0, 1], repeat=len(sites)))
    smi_of = {}
    for s in states:
        built = build_state(neutral_tagged, sites, s)
        if built is not None:
            smi_of[s] = built
    return neutral_tagged, states, smi_of
