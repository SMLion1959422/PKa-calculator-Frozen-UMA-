"""Tests for the thermodynamic microstate layer (no UMA needed)."""
import math
import numpy as np
import pytest

from umapka import microstates as ms
from umapka.sites import find_sites, neutralize
from rdkit import Chem


def test_single_site_reproduces_intrinsic():
    m = ms.build_model("CC(=O)O", {0: 4.76})
    assert m.macro_pkas() == pytest.approx([4.76], abs=1e-9)
    assert m.fraction_uncharged(4.76) == pytest.approx(0.5, abs=1e-9)
    assert m.isoelectric_point() is None


def test_uncoupled_identical_sites_give_statistical_factor():
    m = ms.build_model("OC(=O)CCCCCCCCCCCCCCC(=O)O", {0: 4.5, 1: 4.5}, D=1e9,
                       eps_solvent=1e12)
    p1, p2 = m.macro_pkas()
    assert p2 - p1 == pytest.approx(math.log10(4), abs=1e-6)


def test_macro_pka_sum_is_path_independent():
    m = ms.build_model("NCC(=O)O", {0: 4.38, 1: 7.75})
    full = m.states.sum(1) == len(m.sites)
    none = m.states.sum(1) == 0
    assert sum(m.macro_pkas()) == pytest.approx(
        m.log_w0[full][0] - m.log_w0[none][0], abs=1e-9)


def test_glycine_zwitterion():
    m = ms.build_model("NCC(=O)O", {0: 4.38, 1: 7.75})
    pk1, pk2 = m.macro_pkas()
    # coupling must move both macro pKas outward, towards exp 2.35 / 9.78
    assert pk1 < 3.0 and pk2 > 9.0
    assert 5.5 < m.isoelectric_point() < 6.6
    top = m.species(7.0)[0]
    assert top["smiles"] == "[NH3+]CC(=O)[O-]" and top["fraction"] > 0.95


def test_net_charge_monotonic():
    m = ms.build_model("NCCCC[C@H](N)C(=O)O", {0: 4.2, 1: 10.5, 2: 9.5})
    q = m.titration_curve()["net_charge"]
    assert all(a >= b - 1e-12 for a, b in zip(q, q[1:]))
    assert q[0] > 1.5 and q[-1] < -0.5


def test_ionic_strength_screens_coupling():
    g0 = ms.build_model("OC(=O)CCC(=O)O", {0: 4.5, 1: 4.5})
    g1 = ms.build_model("OC(=O)CCC(=O)O", {0: 4.5, 1: 4.5}, ionic_strength=0.15)
    gap = lambda m: m.macro_pkas()[1] - m.macro_pkas()[0]
    assert gap(g1) < gap(g0)
    # Davies: acids appear more acidic at finite I
    assert g1.macro_pkas()[0] < g0.macro_pkas()[0]


def test_temperature_shifts_amines_not_carboxylic_acids():
    a25 = ms.build_model("CCN", {0: 10.6}).macro_pkas()[0]
    a37 = ms.build_model("CCN", {0: 10.6}, T_K=310.15).macro_pkas()[0]
    assert a25 - a37 == pytest.approx(12 * (10.6 - 0.9) / 298.15, abs=1e-6)
    c37 = ms.build_model("CC(=O)O", {0: 4.76}, T_K=310.15).macro_pkas()[0]
    assert c37 == pytest.approx(4.76, abs=1e-9)


def test_water_dielectric_and_debye_length():
    assert ms.water_dielectric(298.15) == pytest.approx(78.4, abs=0.1)
    assert 1 / ms.debye_kappa(0.1, 78.4) == pytest.approx(3.04 / math.sqrt(0.1), rel=0.01)


def test_carboxylate_charge_is_shared():
    s = find_sites(neutralize(Chem.MolFromSmiles("CC(=O)O")))[0]
    assert len(s.charge_atoms) == 2
    assert sum(w for _, w in s.charge_atoms) == pytest.approx(1.0)


def test_mapping_restricts_sites():
    m = ms.build_model("NCC(=O)O", {1: 9.6})
    assert [s.kind for s in m.sites] == ["base"]


def test_tetrazole_site_is_the_nh():
    for smi in ("c1ccccc1-c1nn[nH]n1", "c1ccccc1-c1nnn[nH]1"):
        mol = neutralize(Chem.MolFromSmiles(smi))
        s = [x for x in find_sites(mol) if x.group.startswith("tetrazole")][0]
        at = mol.GetAtomWithIdx(s.atom)
        assert at.GetSymbol() == "N" and at.GetTotalNumHs() == 1


def test_rdkit_site_predictor_roundtrip(tmp_path):
    pytest.importorskip("lightgbm")
    from umapka.rdkit_site import RDKitSitePredictor, pair_features
    rng = np.random.default_rng(0)
    smis = ["CC(=O)O", "CCN", "Oc1ccccc1", "CCCN", "OC(=O)c1ccccc1", "CNC"] * 5
    X, y = [], []
    for smi in smis:
        mol = neutralize(Chem.MolFromSmiles(smi))
        s = find_sites(mol)[0]
        X.append(pair_features(mol, s.atom, s.kind))
        y.append((4.5 if s.kind == "acid" else 10.5) + rng.normal(0, 0.1))
    p = RDKitSitePredictor(n_estimators=20, min_child_samples=2).fit(np.array(X), np.array(y))
    path = tmp_path / "m.pkl"
    p.save(str(path))
    q = RDKitSitePredictor.load(str(path))
    intr = q.intrinsic_pkas("NCC(=O)O")
    assert sorted(intr) == [0, 1]
    m = ms.build_model("NCC(=O)O", intr)
    assert len(m.macro_pkas()) == 2
