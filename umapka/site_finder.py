"""Learned site-finder: ranks candidate ionizable ATOMS with a LightGBM
LambdaRank model trained on real ChemAxon (marvin_atom) site labels,
instead of always taking the first SMARTS match by fixed priority order.

Validated out-of-fold accuracy (see train_site_finder_v2.py):
  Novartis (n=269):     learned 97.4% vs SMARTS-priority 56.5%
  AvLiLuMoVe (n=123):   learned 99.2% vs SMARTS-priority 78.0%
The gap is largest on multi-candidate molecules (several atoms of the
same ionizable type) - exactly the case fixed-priority order can't get
right even in principle, since it always picks the same one.

Feature schema here MUST match train_site_finder_v2.py's atom_features()
exactly (same order, same columns) - this module intentionally mirrors
that code rather than importing it, since the training script has
top-level side effects (it runs a full training pass on import).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.EState import EStateIndices

_MODEL_PATH = "models/site_finder_v2.pkl"
_KIND_MODEL_PATH = "models/kind_classifier.pkl"
_HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D]


_warned = set()


def _warn_once(key: str, msg: str):
    """Emit `msg` once per process. The learned ranker degrades to
    SMARTS-priority order on failure, which is a ~40-point drop in site
    accuracy on Novartis - it must never be silent. A previously
    unnoticed instance of exactly this: site_finder_v2.pkl pickled by
    lightgbm 3.3.5 raises TypeError inside .predict() under 4.x
    (_n_classes is None), so every call fell back while appearing to
    work normally."""
    if key not in _warned:
        _warned.add(key)
        import warnings
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


@lru_cache(maxsize=1)
def _load():
    """Cached load of the trained ranker. Returns None (not an
    exception) if the model file is missing or unloadable, so callers
    can fall back to SMARTS-priority behaviour without crashing - but
    warns, since that fallback is much less accurate."""
    import joblib
    import os
    if not os.path.exists(_MODEL_PATH):
        _warn_once("missing",
                   f"{_MODEL_PATH} not found - falling back to SMARTS-priority "
                   f"site selection (~56% vs ~97% atom accuracy on Novartis). "
                   f"Run train_site_finder_v2.py to build it.")
        return None
    try:
        d = joblib.load(_MODEL_PATH)
        model = d["model"]
        # NOTE: deliberately NO load-time .predict() probe here. One was
        # tried, to fail fast on a cross-lightgbm-version pickle that
        # loads cleanly but raises on first predict. It backfired: on
        # Windows, LightGBM and torch can ship conflicting OpenMP
        # runtimes, so predict() intermittently dies with an access
        # violation. Because this loader is lru_cached, a single
        # transient probe failure permanently downgraded a working model
        # to SMARTS-priority fallback for the rest of the process - much
        # worse than the silent failure it was meant to catch. The
        # per-call warning in rank_candidates() surfaces a genuinely
        # broken model without that failure mode.
        return model, d["patt_names"], {n: i for i, n in enumerate(d["patt_names"])}
    except Exception as exc:
        _warn_once("broken",
                   f"{_MODEL_PATH} could not be used ({type(exc).__name__}: {exc}) - "
                   f"falling back to SMARTS-priority site selection (~56% vs ~97% "
                   f"atom accuracy on Novartis). If this is a lightgbm version "
                   f"mismatch, re-run train_site_finder_v2.py to re-pickle it.")
        return None


@lru_cache(maxsize=1)
def _load_kind():
    """Cached load of the learned acid/base KIND classifier, or None.

    Why this model exists: error decomposition on Novartis showed the 14
    molecules (5.1%) whose acid/base kind was wrong had MAE 3.901, vs
    0.846 for the 261 that were right - because a wrong kind computes the
    OPPOSITE proton transfer, so the answer is meaningless rather than
    merely imprecise. Held out, this classifier is 99.3% correct on
    Novartis and 100% on AvLiLuMoVe, against 94.9% for the H-count rule
    it supersedes (2 remaining errors instead of 14).
    """
    import joblib
    import os
    if not os.path.exists(_KIND_MODEL_PATH):
        _warn_once("kind_missing",
                   f"{_KIND_MODEL_PATH} not found - falling back to the "
                   f"H-count kind rule (94.9% vs 99.3% on Novartis; a wrong "
                   f"acid/base kind costs ~3 pKa units). Run "
                   f"train_kind_classifier.py --stage build then --stage fit.")
        return None
    try:
        d = joblib.load(_KIND_MODEL_PATH)
        return d["model"], d["patt_names"], {n: i for i, n in enumerate(d["patt_names"])}
    except Exception as exc:
        _warn_once("kind_broken",
                   f"{_KIND_MODEL_PATH} unusable ({type(exc).__name__}: {exc}) - "
                   f"falling back to the H-count kind rule.")
        return None


def predict_kind(mol, idx, acid_sites, base_sites):
    """Learned acid/base kind for atom `idx`: "acid", "base", or None if
    the classifier is unavailable or scoring fails (caller should fall
    back to the pattern/H-count rule)."""
    loaded = _load_kind()
    if loaded is None:
        return None
    model, patt_names, patt_idx = loaded
    hits = _candidate_atoms(mol, acid_sites, base_sites)
    if idx not in hits:
        # Atom added by _expand_ballot (no SMARTS matched it). Score it
        # anyway with empty pattern features - refusing here would send
        # exactly the atoms ballot expansion exists to recover to the
        # weaker H-count fallback.
        if mol.GetAtomWithIdx(idx).GetSymbol() not in _HETERO:
            return None
        hits = dict(hits)
        hits[idx] = {"patterns": set(), "kinds": set()}
    try:
        AllChem.ComputeGasteigerCharges(mol)
        gast = np.nan_to_num(np.array(
            [float(a.GetPropsAsDict().get("_GasteigerCharge", 0.0))
             for a in mol.GetAtoms()]), nan=0.0, posinf=0.0, neginf=0.0)
        est = np.array(EStateIndices(mol))
        dm = Chem.GetDistanceMatrix(mol)
        mol_ctx = [Descriptors.MolWt(mol), Crippen.MolLogP(mol),
                   Descriptors.TPSA(mol), float(Descriptors.RingCount(mol)),
                   float(Descriptors.NumAromaticRings(mol)),
                   float(mol.GetNumAtoms()), float(Chem.GetFormalCharge(mol))]
        ordered = sorted(hits.keys())
        x = _atom_features(mol, idx, hits[idx], gast, est, dm, mol_ctx,
                           len(ordered), ordered.index(idx), patt_names, patt_idx)
        X = np.ascontiguousarray(
            np.nan_to_num(np.array([x], dtype=float), nan=0.0,
                          posinf=0.0, neginf=0.0), dtype=np.float64)
        return "acid" if int(model.predict(X)[0]) == 1 else "base"
    except Exception as exc:
        _warn_once(f"kind_score:{type(exc).__name__}",
                   f"learned kind scoring failed ({type(exc).__name__}: {exc}) - "
                   f"falling back to the H-count rule.")
        return None


def _candidate_atoms(mol, acid_sites, base_sites):
    """Every atom matched by any ACID_SITES/BASE_SITES SMARTS, with
    which named patterns and kinds ("acid"/"base") hit it."""
    hits = {}
    for name, smarts, ai in acid_sites:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for m in mol.GetSubstructMatches(patt):
            idx = m[ai]
            hits.setdefault(idx, {"patterns": set(), "kinds": set()})
            hits[idx]["patterns"].add(name)
            hits[idx]["kinds"].add("acid")
    for name, smarts, ai in base_sites:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for m in mol.GetSubstructMatches(patt):
            idx = m[ai]
            hits.setdefault(idx, {"patterns": set(), "kinds": set()})
            hits[idx]["patterns"].add(name)
            hits[idx]["kinds"].add("base")
    return hits


def _atom_features(mol, idx, info, gast, est, dm, mol_ctx, n_cands, prio_rank,
                    patt_names, patt_idx):
    a = mol.GetAtomWithIdx(idx)
    patt_vec = [0.0] * len(patt_names)
    for p in info["patterns"]:
        if p in patt_idx:
            patt_vec[patt_idx[p]] = 1.0
    s1 = np.where(dm[idx] <= 1)[0]
    s2 = np.where(dm[idx] <= 2)[0]
    hyb = [1.0 if a.GetHybridization() == h else 0.0 for h in _HYB]
    return patt_vec + hyb + mol_ctx + [
        float(a.GetAtomicNum()), float(a.GetFormalCharge()),
        float(a.GetTotalNumHs()), float(a.GetDegree()),
        float(a.GetIsAromatic()), float(a.IsInRing()),
        float(a.GetTotalValence()),
        gast[idx], est[idx],
        gast[s1].mean(), gast[s1].min(), gast[s1].max(),
        gast[s2].mean(), gast[s2].min(), gast[s2].max(),
        est[s1].mean(), est[s2].mean(),
        float(len(s1) - 1), float(len(s2) - 1),
        1.0 if "acid" in info["kinds"] else 0.0,
        1.0 if "base" in info["kinds"] else 0.0,
        float(len(info["patterns"])), float(n_cands), float(prio_rank),
    ]


_DETECTOR_PATH = "models/site_detector.pkl"
_HETERO = {"N", "O", "S", "P"}
_DETECTOR_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def _load_detector():
    """Hard-negative site detector, or None. Absence is not an error -
    the pipeline simply keeps the SMARTS-only ballot it always used."""
    import joblib
    import os
    if not os.path.exists(_DETECTOR_PATH):
        return None
    try:
        return joblib.load(_DETECTOR_PATH)["classifier_full"]
    except Exception as exc:
        _warn_once("detector_broken",
                   f"{_DETECTOR_PATH} unusable ({type(exc).__name__}: {exc}) - "
                   f"continuing with SMARTS-only candidates.")
        return None


def _expand_ballot(mol, hits, patt_names, patt_idx):
    """Add confidently-titratable heteroatoms that no SMARTS matched.

    The detector's features are built with prio_rank/n_cands over ALL
    harvested heteroatoms, matching how it was trained. The ranker then
    re-derives its own features over the final ballot, so each model
    sees the feature distribution it expects - getting that wrong made an
    earlier version of this test read 76.8% instead of 96.8%.

    Returns `hits` unchanged on any failure: expansion is strictly
    additive and must never cost a molecule that already worked.
    """
    det = _load_detector()
    if det is None:
        return hits
    extra = [a.GetIdx() for a in mol.GetAtoms()
             if a.GetSymbol() in _HETERO and a.GetIdx() not in hits]
    if not extra:
        return hits
    try:
        AllChem.ComputeGasteigerCharges(mol)
        gast = np.nan_to_num(np.array(
            [float(a.GetPropsAsDict().get("_GasteigerCharge", 0.0))
             for a in mol.GetAtoms()]), nan=0.0, posinf=0.0, neginf=0.0)
        est = np.array(EStateIndices(mol))
        dm = Chem.GetDistanceMatrix(mol)
        ctx = [Descriptors.MolWt(mol), Crippen.MolLogP(mol), Descriptors.TPSA(mol),
               float(Descriptors.RingCount(mol)),
               float(Descriptors.NumAromaticRings(mol)),
               float(mol.GetNumAtoms()), float(Chem.GetFormalCharge(mol))]
        every = sorted(set(hits) | set(extra))
        rows = [_atom_features(mol, i, hits.get(i, {"patterns": set(), "kinds": set()}),
                               gast, est, dm, ctx, len(every), r, patt_names, patt_idx)
                for r, i in enumerate(every)]
        X = np.ascontiguousarray(np.nan_to_num(np.array(rows, dtype=float),
                                                nan=0.0, posinf=0.0, neginf=0.0),
                                  dtype=np.float64)
        prob = det.predict_proba(X)[:, 1]
    except Exception as exc:
        _warn_once(f"expand:{type(exc).__name__}",
                   f"ballot expansion failed ({type(exc).__name__}: {exc}) - "
                   f"using SMARTS-only candidates.")
        return hits

    out = dict(hits)
    for i, idx in enumerate(every):
        if idx not in out and prob[i] > _DETECTOR_THRESHOLD:
            # no pattern matched it, so kind must come from the learned
            # kind classifier downstream, not from pattern membership
            out[idx] = {"patterns": set(), "kinds": set()}
    return out


def rank_candidates(mol, acid_sites, base_sites):
    """Rank every SMARTS-matched candidate atom with the learned
    ranker. Returns a list of (atom_idx, kind, score) sorted best-first
    (kind is "acid" if any acid pattern matched and no base pattern
    did, "base" if the reverse, or the majority-vote kind - by pattern
    count - when an atom matched both, which is rare), or None if the
    model file isn't present or scoring fails for any reason (caller
    should fall back to SMARTS-priority order in that case).
    """
    loaded = _load()
    if loaded is None:
        return None
    model, patt_names, patt_idx = loaded

    hits = _candidate_atoms(mol, acid_sites, base_sites)
    if not hits:
        return None

    # BALLOT EXPANSION: SMARTS enumerates candidates, so a titratable atom
    # no pattern matches is unreachable no matter how good the ranker is -
    # 3.9% of Novartis (6.0% of training). site_detector.pkl scores every
    # N/O/S/P atom and adds the confident ones to the ballot; the ranker
    # still makes the pick.
    #
    # Measured on Novartis with rank/n_cands recomputed over each arm's own
    # candidate set (eval_union_hybrid.py):
    #     SMARTS only : 94.2%
    #     union       : 96.8%    7 molecules recovered, 0 broken
    #     on the 9 SMARTS-invisible molecules: 0% -> 77.8%
    # AvLiLuMoVe was unchanged (99.2%, no atoms added). Only 8 atoms were
    # added across 278 molecules, so this is surgical, not a wider net.
    hits = _expand_ballot(mol, hits, patt_names, patt_idx)

    try:
        AllChem.ComputeGasteigerCharges(mol)
        gast = np.nan_to_num(np.array(
            [float(a.GetPropsAsDict().get("_GasteigerCharge", 0.0))
             for a in mol.GetAtoms()]), nan=0.0, posinf=0.0, neginf=0.0)
        est = np.array(EStateIndices(mol))
        dm = Chem.GetDistanceMatrix(mol)
        mol_ctx = [Descriptors.MolWt(mol), Crippen.MolLogP(mol),
                   Descriptors.TPSA(mol), float(Descriptors.RingCount(mol)),
                   float(Descriptors.NumAromaticRings(mol)),
                   float(mol.GetNumAtoms()), float(Chem.GetFormalCharge(mol))]

        ordered = sorted(hits.keys())
        X = np.array([
            _atom_features(mol, idx, hits[idx], gast, est, dm, mol_ctx,
                            len(ordered), rank, patt_names, patt_idx)
            for rank, idx in enumerate(ordered)
        ], dtype=float)
        scores = model.predict(X)
    except Exception as exc:
        _warn_once(f"score:{type(exc).__name__}",
                   f"learned site scoring failed ({type(exc).__name__}: {exc}) - "
                   f"falling back to SMARTS-priority site selection for this and "
                   f"any similarly-failing molecule.")
        return None

    def _kind(idx):
        # Learned classifier first - it is 99.3% correct on Novartis vs
        # 94.9% for the pattern/H-count logic below, and it is consulted
        # for EVERY atom, not just the acid/base-ambiguous ones, because
        # the 5% of kind errors were not confined to ambiguous atoms.
        learned = predict_kind(mol, idx, acid_sites, base_sites)
        if learned is not None:
            return learned
        kinds = hits[idx]["kinds"]
        if "acid" in kinds and "base" not in kinds:
            return "acid"
        if "base" in kinds and "acid" not in kinds:
            return "base"
        # Atom matched BOTH acid- and base-type patterns. Deciding this
        # by "acid always wins" scored 0/12 on the ambiguous subset of
        # Novartis/AvLiLuMoVe (see recheck_kind.py). The H-count rule
        # below is the validated replacement: an atom that still carries
        # a hydrogen has one to LOSE (acid); one that doesn't can only
        # GAIN a proton (base). Getting this wrong is not a small error -
        # it computes the opposite transition and the predicted pKa is
        # wildly wrong, which is why kind accuracy matters as much as
        # atom accuracy here.
        return "acid" if mol.GetAtomWithIdx(idx).GetTotalNumHs() > 0 else "base"

    order = np.argsort(-scores)
    return [(ordered[i], _kind(ordered[i]), float(scores[i])) for i in order]


def best_site_atom(mol, acid_sites, base_sites):
    """Top-ranked (atom_idx, kind) from the learned ranker, or None."""
    ranked = rank_candidates(mol, acid_sites, base_sites)
    if not ranked:
        return None
    idx, kind, _ = ranked[0]
    return idx, kind
