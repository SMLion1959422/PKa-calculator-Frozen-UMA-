"""Delocalization descriptors: does the embedding-response summary behave?"""
import importlib.util
import pathlib
import numpy as np
import pytest

pytest.importorskip("scipy")
from rdkit import Chem
from umapka.sites import find_sites, neutralize

spec = importlib.util.spec_from_file_location(
    "dl", pathlib.Path(__file__).parents[1] / "dev" / "delocalization_test.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


def test_summarize_localized_vs_delocalized():
    dist = np.array([0., 1., 2., 3., 4., 5.])
    local = dl._summarize(np.array([1., .1, .01, 0., 0., 0.]), dist)
    spread = dl._summarize(np.array([1., 1., 1., 1., 1., 1.]), dist)
    # localized: small radius, little weight beyond 2 bonds, few atoms sharing
    assert local[0] < spread[0]
    assert local[1] < spread[1]
    assert local[2] < spread[2]
    assert spread[2] == pytest.approx(6.0)      # IPR = n when weight is uniform


def test_summarize_handles_zero_response():
    assert dl._summarize(np.zeros(4), np.arange(4.)) is None


def test_topo_descriptor_counts_conjugation():
    # acetate: site O, conjugated only to C(=O)O ; benzoate: extends into the ring
    for smi, lo, hi in [("CC(=O)O", 2, 4), ("OC(=O)c1ccccc1", 7, 11)]:
        mol = neutralize(Chem.MolFromSmiles(smi))
        site = find_sites(mol)[0].atom
        n = dl.topo_descriptor(mol, site)
        assert lo <= n <= hi, (smi, n)
    a = dl.topo_descriptor(*(lambda m: (m, find_sites(m)[0].atom))(
        neutralize(Chem.MolFromSmiles("CC(=O)O"))))
    b = dl.topo_descriptor(*(lambda m: (m, find_sites(m)[0].atom))(
        neutralize(Chem.MolFromSmiles("OC(=O)c1ccccc1"))))
    assert b > a          # aromatic carboxylate is more conjugated than acetate


def test_gasteiger_descriptor_runs_and_is_localized_for_acetate():
    from umapka.predictor import _ionize_keep_order
    mol = neutralize(Chem.MolFromSmiles("CC(=O)O"))
    site = find_sites(mol)[0].atom
    out = dl.gasteiger_descriptor(mol, _ionize_keep_order(mol, site, "acid"), site)
    assert out is not None
    r, f_far, ipr = out
    assert 0 <= r < 4 and 0 <= f_far <= 1 and ipr >= 1


def test_uma_descriptor_atom_alignment_with_stub():
    """The UMA path must compare atom i to atom i across the two species."""
    pytest.importorskip("ase")
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from test_predictor_plumbing import _stub
    from umapka.predictor import _ionize_keep_order
    p = _stub()
    mol = neutralize(Chem.MolFromSmiles("OC(=O)c1ccccc1"))
    site = find_sites(mol)[0].atom
    out = dl.uma_descriptor(p, mol, _ionize_keep_order(mol, site, "acid"), site)
    assert out is not None
    r, f_far, ipr = out
    assert np.isfinite(r) and 0 <= f_far <= 1 and ipr >= 1


def _toy(n_mol=80, n_solv=5, informative=True, seed=0):
    """Molecules x solvents, where the alpha slope depends on `truth`.
    `noise` is an uninformative descriptor; `partial` is a noisy copy of truth."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=n_mol)
    noise = rng.normal(size=n_mol)
    partial = truth + rng.normal(scale=1.2, size=n_mol)
    alphas = np.linspace(-1.2, -0.2, n_solv)
    rows = []
    for i in range(n_mol):
        for s, a in enumerate(alphas):
            slope = -6.0 + (2.0 * truth[i] if informative else 0.0)
            rows.append(dict(rxn=f"m{i}", solvent=f"s{s}", d_alpha=a,
                             y=slope * a + rng.normal(scale=0.2),
                             truth=truth[i], noise=noise[i], partial=partial[i]))
    return pd.DataFrame(rows)


def test_nested_permutation_detects_real_added_information():
    df = _toy(informative=True)
    F, p = dl.nested_permutation_p(df, ["noise"], ["truth"], df.y.values, n_perm=300)
    assert F > 10 and p < 0.01, (F, p)


def test_nested_permutation_rejects_noise():
    """An uninformative descriptor must NOT come out significant."""
    df = _toy(informative=True)
    F, p = dl.nested_permutation_p(df, ["truth"], ["noise"], df.y.values, n_perm=300)
    assert p > 0.05, (F, p)


def test_nested_permutation_asymmetry_when_one_subsumes_the_other():
    """truth subsumes a noisy copy of itself: truth adds over partial,
    partial adds little over truth."""
    df = _toy(informative=True)
    y = df.y.values
    _, p_truth_adds = dl.nested_permutation_p(df, ["partial"], ["truth"], y, n_perm=300)
    _, p_partial_adds = dl.nested_permutation_p(df, ["truth"], ["partial"], y, n_perm=300)
    assert p_truth_adds < 0.01
    assert p_partial_adds > p_truth_adds


def test_permutation_respects_molecule_clustering():
    """With no real effect, p must be roughly uniform - not driven to 0 by
    treating the ~5 rows per molecule as independent."""
    df = _toy(informative=False)
    ps = [dl.nested_permutation_p(df, ["noise"], ["truth"], df.y.values,
                                  n_perm=200, seed=s)[1] for s in range(5)]
    assert min(ps) > 0.01, ps
