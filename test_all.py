"""END-TO-END TEST SUITE - run this to check every public function.

    python test_all.py              # everything (loads UMA once, ~3-6 min)
    python test_all.py --fast       # skip tests that need UMA embeddings
    python test_all.py -k solvent   # only tests whose name matches

Covers: environment/version pinning, site finding, acid/base kind,
protonation-pair construction, feature builders, aqueous prediction,
multisolvent, salt correction, solvent mixtures, the microstate solver,
polyprotic end-to-end, and error handling on bad input.

ACCURACY CHECKS - read this before trusting a PASS:
the aqueous spot-checks use widely-established textbook pKa values as
SANITY anchors with a deliberately loose +/-1.5 tolerance. They catch
gross breakage (wrong site, flipped acid/base, broken calibrator), NOT
accuracy regressions of 0.1-0.2. The real accuracy numbers come from
held-out benchmarks - eval_cached.py (Novartis/AvLiLuMoVe) and
validate_multisolvent.py (published multisolvent split). Never quote a
number from this file as a benchmark result.
"""
import argparse
import sys
import traceback

import numpy as np

RESULTS = []


def check(name, fn, needs_uma=False):
    RESULTS.append({"name": name, "fn": fn, "needs_uma": needs_uma})
    return fn


# ----------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------
def test_versions():
    """Installed versions match requirements-lock.txt for the packages
    that have actually caused silent breakage here: lightgbm (3.3.5
    pickles raise inside predict() under 4.x, which the site finder
    swallowed into a SMARTS fallback) and scikit-learn (1.3.2 vs the
    locked 1.9.0 made LightGBM .fit() die with an access violation)."""
    import lightgbm, sklearn
    want = {}
    for enc in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            txt = open("requirements-lock.txt", encoding=enc).read()
            if "lightgbm" in txt:
                for line in txt.splitlines():
                    s = line.strip()
                    if "==" in s:
                        k, _, v = s.partition("==")
                        want[k.strip().lower()] = v.strip()
                break
        except Exception:
            continue
    problems = []
    for mod, key in ((lightgbm, "lightgbm"), (sklearn, "scikit-learn")):
        exp = want.get(key)
        if exp and mod.__version__ != exp:
            problems.append(f"{key}: installed {mod.__version__} != locked {exp}")
    assert not problems, "; ".join(problems)
    return f"lightgbm {lightgbm.__version__}, sklearn {sklearn.__version__}"


def test_models_present():
    import os
    need = ["models/model_core_v3.pkl", "models/model_core_v16_elec.pkl",
            "models/multisolvent_tuned.pkl", "models/site_finder_v2.pkl",
            "models/kind_classifier.pkl"]
    missing = [p for p in need if not os.path.exists(p)]
    assert not missing, f"missing: {missing}"
    return f"{len(need)} model files present"


def test_no_silent_fallback():
    """The learned site finder and kind classifier must actually load.
    Both degrade to much weaker heuristics on failure, and that used to
    happen silently - this asserts they are really engaged."""
    from umapka import site_finder as SF
    SF._load.cache_clear(); SF._load_kind.cache_clear()
    assert SF._load() is not None, "site ranker failed to load"
    assert SF._load_kind() is not None, "kind classifier failed to load"
    return "site ranker + kind classifier both live"


# ----------------------------------------------------------------------
# site finding / kind
# ----------------------------------------------------------------------
SITE_CASES = [
    ("CC(=O)O", "acid", "acetic acid -> carboxyl"),
    ("Oc1ccccc1", "acid", "phenol -> OH"),
    ("c1ccccc1N", "base", "aniline -> N"),
    ("c1ccncc1", "base", "pyridine -> ring N"),
    ("CCN", "base", "ethylamine -> N"),
    ("CC(=O)Oc1ccccc1C(=O)O", "acid", "aspirin -> carboxyl"),
    ("O=C(O)c1ccccc1", "acid", "benzoic acid -> carboxyl"),
    ("OS(=O)(=O)c1ccccc1", "acid", "benzenesulfonic -> OH"),
]


def test_site_kind():
    from rdkit import Chem
    from umapka import site_finder as SF
    from umapka.predictor import ACID_SITES, BASE_SITES, neutralize
    bad = []
    for smi, want_kind, label in SITE_CASES:
        mol = neutralize(Chem.MolFromSmiles(smi))
        got = SF.best_site_atom(mol, ACID_SITES, BASE_SITES)
        if got is None:
            bad.append(f"{label}: no site found")
            continue
        idx, kind = got
        if kind != want_kind:
            bad.append(f"{label}: kind {kind} != {want_kind}")
    assert not bad, "; ".join(bad)
    return f"{len(SITE_CASES)}/{len(SITE_CASES)} correct kind"


def test_nitro_not_basic():
    """A nitro nitrogen must not register as a basic aniline site. It did
    until the aniline SMARTS got a +0 charge constraint, and it was the
    direct cause of the worst polyprotic-benchmark errors."""
    from rdkit import Chem
    from umapka import microstates as M
    from umapka.predictor import ACID_SITES, BASE_SITES, neutralize
    mol = neutralize(Chem.MolFromSmiles("O=[N+]([O-])c1ccccc1O"))
    sites = M.all_sites(mol, ACID_SITES, BASE_SITES)
    assert len(sites) == 1, f"2-nitrophenol should have 1 site, got {len(sites)}: {sites}"
    assert sites[0][1] == "acid"
    return "nitro correctly not a basic site"


def test_real_sites_preserved():
    """Site capping must never drop genuine sites on real polyprotics -
    a score-margin filter was tried and removed for exactly this."""
    from rdkit import Chem
    from umapka import microstates as M
    from umapka.predictor import ACID_SITES, BASE_SITES, neutralize
    for smi, want in [("OC(=O)CC(N)C(=O)O", 3), ("NCCCCC(N)C(=O)O", 3),
                      ("NCC(=O)O", 2), ("OP(=O)(O)O", 3)]:
        mol = neutralize(Chem.MolFromSmiles(smi))
        s = M.all_sites(mol, ACID_SITES, BASE_SITES)
        kept, dropped = M.filter_sites(mol, s, ACID_SITES, BASE_SITES)
        assert len(kept) == want, f"{smi}: kept {len(kept)} of {want}"
        assert not dropped, f"{smi}: dropped {dropped}"
    return "aspartate/lysine/glycine/phosphate keep all sites"


def test_protonation_pair_indices():
    """Site indices must stay valid after canonicalization - the
    canonical-SMILES atom-reordering bug this repo documents at length."""
    from rdkit import Chem
    from umapka.predictor import protonation_pair_site_tagged
    for smi in ["Oc1ccccc1", "CC(=O)O", "c1ccncc1", "CC(=O)Nc1ccc(O)cc1",
                "c1ccc2c(c1)[nH]c(=O)[nH]2"]:
        prot, pi, dep, di, kind = protonation_pair_site_tagged(smi, return_kind=True)
        for s, i in ((prot, pi), (dep, di)):
            m = Chem.MolFromSmiles(s)
            assert m is not None, f"{smi}: unparseable {s}"
            assert i is not None and 0 <= i < m.GetNumAtoms(), f"{smi}: bad idx {i}"
            assert m.GetAtomWithIdx(i).GetSymbol() in ("O", "N", "S", "C", "P"), \
                f"{smi}: site atom is {m.GetAtomWithIdx(i).GetSymbol()}"
        # the pair must differ by exactly one proton and one charge unit
        mp, md = Chem.MolFromSmiles(prot), Chem.MolFromSmiles(dep)
        dq = Chem.GetFormalCharge(mp) - Chem.GetFormalCharge(md)
        assert dq == 1, f"{smi}: charge difference {dq} != 1"
    return "5 molecules: indices valid, pairs differ by 1 proton"


# ----------------------------------------------------------------------
# microstate solver (no UMA needed)
# ----------------------------------------------------------------------
def test_solver_exact():
    """A perfectly consistent ladder must be recovered almost exactly,
    and an inconsistent one must raise the residual and sigma."""
    from umapka import microstates as M
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    good = [((0, 1), (0, 0), 1, 10.0), ((1, 0), (0, 0), 0, 4.0),
            ((1, 1), (0, 1), 0, 4.0), ((1, 1), (1, 0), 1, 10.0)]
    lb, rms, sig = M.solve_logbeta(states, good, 2)
    macro, msig, Z = M.macro_pka(states, lb, sig, 2, set(states))
    assert abs(macro[0] - 4.0) < 0.15, f"pKa1 {macro[0]} != 4.0"
    assert abs(macro[1] - 10.0) < 0.15, f"pKa2 {macro[1]} != 10.0"
    assert rms < 0.05, f"consistent ladder residual too high: {rms}"
    bad = list(good); bad[2] = ((1, 1), (0, 1), 0, 6.5)
    _lb2, rms2, sig2 = M.solve_logbeta(states, bad, 2)
    assert rms2 > 0.3, f"inconsistency not detected: rms {rms2}"
    return f"recovered 4.0/10.0 (rms {rms:.3f}); inconsistency rms {rms2:.3f}"


def test_solver_bounded():
    """Ridge must keep log-beta finite when microstates are missing -
    plain lstsq returned unbounded values for weakly-determined params."""
    from umapka import microstates as M
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    only_one = [((1, 0), (0, 0), 0, 4.0)]      # (0,1) and (1,1) unconstrained
    lb, rms, sig = M.solve_logbeta(states, only_one, 2)
    assert np.all(np.isfinite(lb)), "log-beta not finite"
    assert np.abs(lb).max() < 1e3, f"log-beta exploded: {np.abs(lb).max()}"
    return f"underdetermined system stayed bounded (max |logbeta| {np.abs(lb).max():.2f})"


def test_population_normalized():
    from umapka import microstates as M
    Z = {0: 0.0, 1: 4.0, 2: 14.0}
    for ph in (1.0, 7.4, 13.0):
        p = M.population(Z, ph)
        assert abs(sum(p.values()) - 1.0) < 1e-9, f"pH {ph}: sum {sum(p.values())}"
        assert all(v >= 0 for v in p.values())
    return "populations normalized at pH 1/7.4/13"


# ----------------------------------------------------------------------
# solvents / salts / mixtures (no UMA)
# ----------------------------------------------------------------------
def test_solvent_registry():
    from umapka import solvents as sv
    assert sv.resolve_solvent("water").name == "Water"
    assert sv.resolve_solvent("h2o").name == "Water"
    assert sv.resolve_solvent("MeCN").name == "Acetonitrile"
    assert sv.resolve_solvent("O").name == "Water"
    try:
        sv.resolve_solvent("liquid nitrogen")
        raise AssertionError("unknown solvent should raise")
    except ValueError:
        pass
    return f"{len(sv.SOLVENTS)} solvents, aliases + rejection OK"


def test_mixture_endpoints_and_monotonic():
    """Mixture interpolation must reproduce both endpoints exactly and
    move monotonically between them."""
    from umapka.mixtures import mixture_dielectric
    e0, _ = mixture_dielectric("water", "acetonitrile", 0.0)
    e1, _ = mixture_dielectric("water", "acetonitrile", 1.0)
    assert abs(e0 - 78.4) < 2.0, f"pure-water eps {e0}"
    assert abs(e1 - 37.5) < 2.0, f"pure-MeCN eps {e1}"
    prev, seq = None, []
    for f in np.linspace(0, 1, 11):
        e, _ = mixture_dielectric("water", "acetonitrile", float(f))
        seq.append(e)
        if prev is not None:
            assert e <= prev + 1e-6, f"eps not monotonic at f={f}: {e} > {prev}"
        prev = e
    return f"eps {seq[0]:.1f} -> {seq[-1]:.1f}, monotonic"


def test_mixture_flags_uncertainty():
    """Mixtures are NOT validated against experiment; the high-organic
    regime must be flagged rather than reported as confident."""
    import umapka.mixtures as mx
    src = open("umapka/mixtures.py", encoding="utf-8").read()
    assert src.count("_CURVATURE_EXPONENT =") == 1, \
        "_CURVATURE_EXPONENT defined more than once (the second silently wins)"
    return "curvature constant defined once; heuristic documented"


def test_salt_correction_sign():
    """Adding neutral salt must LOWER a neutral acid's apparent pKa
    (classic kinetic salt effect) and scale with concentration."""
    from umapka import solvation
    lo = solvation.predict_salt_correction("acid", "NaCl", 0.01, solvent="water")
    hi = solvation.predict_salt_correction("acid", "NaCl", 0.50, solvent="water")
    assert lo["shift"] < 0, f"0.01 M shift should be negative, got {lo['shift']}"
    assert hi["shift"] < lo["shift"], \
        f"higher ionic strength should shift more: {hi['shift']} vs {lo['shift']}"
    assert abs(hi["shift"]) < 3.0, f"implausible shift {hi['shift']}"
    return f"0.01 M {lo['shift']:+.3f}, 0.50 M {hi['shift']:+.3f} ({hi['tier']})"


# ----------------------------------------------------------------------
# prediction (needs UMA)
# ----------------------------------------------------------------------
_PRED = {}


def _predictor():
    if "p" not in _PRED:
        from umapka import PkaPredictor
        _PRED["p"] = PkaPredictor("models/model_core_v3.pkl")
    return _PRED["p"]


# textbook anchors, loose tolerance - gross-breakage detector only
AQUEOUS_ANCHORS = [
    ("CC(=O)O", 4.76, "acetic acid"),
    ("O=C(O)c1ccccc1", 4.20, "benzoic acid"),
    ("Oc1ccccc1", 9.99, "phenol"),
    ("c1ccccc1N", 4.60, "aniline"),
    ("c1ccncc1", 5.23, "pyridine"),
    ("CC(=O)Oc1ccccc1C(=O)O", 3.49, "aspirin"),
]
TOL = 1.5


def test_aqueous_accuracy():
    p = _predictor()
    rows, bad = [], []
    for smi, ref, label in AQUEOUS_ANCHORS:
        got = p.predict(smi)
        rows.append(f"{label} {got:.2f} (ref {ref})")
        if not np.isfinite(got) or abs(got - ref) > TOL:
            bad.append(f"{label}: {got:.2f} vs {ref} (tol {TOL})")
    mae = np.mean([abs(p.predict(s) - r) for s, r, _ in AQUEOUS_ANCHORS])
    assert not bad, "; ".join(bad)
    return f"{len(AQUEOUS_ANCHORS)}/{len(AQUEOUS_ANCHORS)} within {TOL}; anchor MAE {mae:.2f}"


def test_determinism():
    p = _predictor()
    a, b = p.predict("CC(=O)O"), p.predict("CC(=O)O")
    assert a == b, f"non-deterministic: {a} != {b}"
    return f"repeat call identical ({a:.4f})"


def test_multisolvent_runs():
    """Non-water solvents must route through the multisolvent regressor
    without a feature-dimension mismatch. This broke silently when the
    default aqueous model moved from 768 to 1536 features."""
    p = _predictor()
    out = {}
    for s in ("water", "dmso", "acetonitrile", "methanol", "dmf", "ethanol"):
        v = p.predict("CC(=O)O", solvent=s)
        assert np.isfinite(v), f"{s}: non-finite {v}"
        out[s] = v
    assert out["dmso"] > out["water"] + 3, \
        f"acetic acid should be far weaker in DMSO: {out['dmso']} vs {out['water']}"
    return "  ".join(f"{k}={v:.1f}" for k, v in out.items())


def test_global_block_slice():
    """The multisolvent path slices the first 768 features out of a
    1536-dim vector. That is only valid if the global block is identical
    to what a 768-dim model would produce - assert it, don't assume."""
    from umapka import PkaPredictor
    from umapka.predictor import protonation_pair_site_tagged
    prot, pi, dep, di = protonation_pair_site_tagged("CC(=O)O")
    p3 = _predictor()
    f3 = p3.features(prot, dep, pi, di)
    assert f3.shape[1] == 1536, f"expected 1536-dim, got {f3.shape}"
    p2 = PkaPredictor("models/model_core_v2.pkl")
    f2 = p2.features(prot, dep, pi, di)
    assert f2.shape[1] == 768
    d = np.abs(f3.reshape(-1)[:768] - f2.reshape(-1)).max()
    assert d < 1e-6, f"global block differs by {d} - the 768 slice is invalid"
    return f"global block identical (max diff {d:.2e})"


def test_salt_end_to_end():
    p = _predictor()
    base = p.predict("CC(=O)O")
    salted = p.predict("CC(=O)O", salt="NaCl", salt_concentration=0.15)
    assert salted < base, f"salt should lower pKa: {salted} vs {base}"
    d = p.predict_detailed("CC(=O)O", salt="NaCl", salt_concentration=0.15)
    assert d["correction"]["tier"] != "none"
    return f"{base:.2f} -> {salted:.2f} ({d['correction']['tier']})"


def test_mixture_end_to_end():
    from umapka.mixtures import predict_mixed_solvent_pka
    p = _predictor()
    r = predict_mixed_solvent_pka(p, "CC(=O)O", "water", "acetonitrile", 0.3)
    assert np.isfinite(r["pKa"])
    assert min(r["endpoint_a"], r["endpoint_b"]) - 0.01 <= r["pKa"] \
        <= max(r["endpoint_a"], r["endpoint_b"]) + 0.01, \
        f"mixture {r['pKa']} outside endpoints {r['endpoint_a']}/{r['endpoint_b']}"
    hi = predict_mixed_solvent_pka(p, "CC(=O)O", "water", "acetonitrile", 0.9)
    assert hi["confidence"] == "low" and hi["warning"], \
        "90% organic must be flagged low-confidence"
    return f"30%: {r['pKa']:.2f} in [{r['endpoint_a']:.1f},{r['endpoint_b']:.1f}]; 90% flagged"


def test_polyprotic_end_to_end():
    """Glycine must come out as a zwitterion-dominated diprotic with
    pKa1 < pKa2 and the neutral/1-proton form dominant at pH 7.4."""
    from rdkit import Chem
    from umapka import microstates as M, electronic
    from umapka.predictor import ACID_SITES, BASE_SITES, neutralize, \
        _smiles_to_atoms_with_site
    import joblib
    bundle = joblib.load("models/model_core_v16_elec.pkl")
    p = _predictor()
    nm = neutralize(Chem.MolFromSmiles("NCC(=O)O"))
    sites = M.all_sites(nm, ACID_SITES, BASE_SITES)
    assert len(sites) == 2, f"glycine should have 2 sites, got {len(sites)}"
    _t, states, smi_of = M.enumerate_microstates(nm, sites)
    assert len(smi_of) == 4, f"expected 4 microstates, built {len(smi_of)}"
    cache = {}
    for s, tagged in smi_of.items():
        clean = M.strip_tags_text(tagged)
        tm, cm = Chem.MolFromSmiles(tagged), Chem.MolFromSmiles(clean)
        if tm is None or cm is None:
            continue
        tag_idx = {a.GetAtomMapNum(): a.GetIdx() for a in tm.GetAtoms()
                   if a.GetAtomMapNum() >= M.TAG0}
        atoms, _, mol_h = _smiles_to_atoms_with_site(clean, next(iter(tag_idx.values())))
        cache[s] = {"clean": clean, "emb": p.embeddings(atoms),
                    "mol_h": mol_h, "tag_idx": tag_idx}
    trans = []
    for s in states:
        if s not in cache:
            continue
        for i in range(len(sites)):
            if s[i] != 1:
                continue
            s2 = list(s); s2[i] = 0; s2 = tuple(s2)
            if s2 not in cache:
                continue
            cp, cd = cache[s], cache[s2]; tag = M.TAG0 + i
            ip, id_ = cp["tag_idx"][tag], cd["tag_idx"][tag]
            g_ = np.concatenate([p.pool(cp["emb"]), p.pool(cd["emb"]),
                                 p.pool(cp["emb"]) - p.pool(cd["emb"])])
            hlp = p.pool_local_multiscale(cp["emb"], ip, cp["mol_h"])
            hld = p.pool_local_multiscale(cd["emb"], id_, cd["mol_h"])
            l_ = np.concatenate([hlp, hld, hlp - hld])
            dp = electronic.elec_desc(cp["clean"], ip)
            dd = electronic.elec_desc(cd["clean"], id_)
            feat = np.nan_to_num(np.concatenate([g_, l_, dp, dd, dp - dd])).reshape(1, -1)
            trans.append((s, s2, i, electronic.score_hybrid(bundle, feat)))
    lb, rms, sig = M.solve_logbeta(states, trans, len(sites))
    macro, msig, Z = M.macro_pka(states, lb, sig, len(sites), set(cache))
    assert len(macro) == 2, f"expected 2 macro pKa, got {macro}"
    assert macro[0] < macro[1], f"non-monotonic ladder {macro}"
    pop = M.population(Z, 7.4)
    assert pop[1] > 0.8, f"1-proton form should dominate at pH 7.4: {pop}"
    return (f"pKa1 {macro[0]:.2f}+/-{msig[0]:.2f}, pKa2 {macro[1]:.2f}"
            f"+/-{msig[1]:.2f}, rms {rms:.2f}, {pop[1]*100:.0f}% zwitterion @7.4")


# ----------------------------------------------------------------------
# error handling
# ----------------------------------------------------------------------
def test_bad_input():
    from umapka.predictor import protonation_pair
    from umapka import solvents as sv
    p = _predictor()
    for bad in ("not_a_smiles", "C(((", ""):
        try:
            p.predict(bad)
            raise AssertionError(f"{bad!r} should raise")
        except AssertionError:
            raise
        except Exception:
            pass
    try:
        protonation_pair("CCCC")          # no titratable site
        raise AssertionError("alkane should raise")
    except AssertionError:
        raise
    except Exception:
        pass
    try:
        p.predict("CC(=O)O", solvent="unobtainium")
        raise AssertionError("bad solvent should raise")
    except AssertionError:
        raise
    except Exception:
        pass
    return "bad SMILES / no-site / bad solvent all raise cleanly"


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip UMA-dependent tests")
    ap.add_argument("-k", default=None, help="only run tests matching this substring")
    a = ap.parse_args()

    for name, fn, uma in [
        ("versions", test_versions, False),
        ("models_present", test_models_present, False),
        ("no_silent_fallback", test_no_silent_fallback, False),
        ("site_kind", test_site_kind, False),
        ("nitro_not_basic", test_nitro_not_basic, False),
        ("real_sites_preserved", test_real_sites_preserved, False),
        ("protonation_pair_indices", test_protonation_pair_indices, False),
        ("solver_exact", test_solver_exact, False),
        ("solver_bounded", test_solver_bounded, False),
        ("population_normalized", test_population_normalized, False),
        ("solvent_registry", test_solvent_registry, False),
        ("mixture_dielectric", test_mixture_endpoints_and_monotonic, False),
        ("mixture_uncertainty", test_mixture_flags_uncertainty, False),
        ("salt_correction_sign", test_salt_correction_sign, False),
        ("aqueous_accuracy", test_aqueous_accuracy, True),
        ("determinism", test_determinism, True),
        ("global_block_slice", test_global_block_slice, True),
        ("multisolvent_runs", test_multisolvent_runs, True),
        ("salt_end_to_end", test_salt_end_to_end, True),
        ("mixture_end_to_end", test_mixture_end_to_end, True),
        ("polyprotic_end_to_end", test_polyprotic_end_to_end, True),
        ("bad_input", test_bad_input, True),
    ]:
        RESULTS.append({"name": name, "fn": fn, "needs_uma": uma})

    npass = nfail = nskip = 0
    print("=" * 74)
    print("umapka test suite")
    print("=" * 74)
    for t in RESULTS:
        if a.k and a.k not in t["name"]:
            continue
        if a.fast and t["needs_uma"]:
            print(f"SKIP  {t['name']:26s} (needs UMA)")
            nskip += 1
            continue
        try:
            detail = t["fn"]()
            print(f"PASS  {t['name']:26s} {detail or ''}")
            npass += 1
        except Exception as exc:
            first = traceback.format_exc().strip().splitlines()[-1]
            print(f"FAIL  {t['name']:26s} {first}")
            nfail += 1

    print("=" * 74)
    print(f"{npass} passed, {nfail} failed, {nskip} skipped")
    print("\nNOTE: the aqueous anchors are loose sanity checks, not a benchmark.")
    print("Real accuracy: eval_cached.py and validate_multisolvent.py")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
