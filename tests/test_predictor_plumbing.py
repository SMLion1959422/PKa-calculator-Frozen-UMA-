"""Predictor plumbing with UMA replaced by a deterministic stub, so the
site-local features and predict_macro wiring are tested without model
weights. Skipped if ase is not installed."""
import numpy as np
import pytest

pytest.importorskip("ase")
from rdkit import Chem
from umapka import predictor as P
from umapka.sites import find_sites, neutralize


class _Reg:
    """Stub regressor: pKa from the last two features (dE, n_heavy)."""
    def predict(self, X):
        return np.array([4.0 + 0.1 * X[0, -2]])


def _stub(feature="site_v3", p=None):
    p = p if p is not None else P.PkaPredictor.__new__(P.PkaPredictor)
    p.regressor = {"regressor": _Reg(), "calibrator": None, "feature": feature}
    p.feature_mode = feature
    p._multisolvent_model_path = None
    p._multisolvent_bundle = None
    p._free_energy_model = None

    def embeddings(atoms):
        z = atoms.get_atomic_numbers().astype(float)
        p._last_energy = float(-z.sum() - 0.5 * atoms.info["charge"])
        rng = np.random.default_rng(0)
        basis = rng.normal(size=(100, 128))
        return basis[z.astype(int)] + 0.1 * atoms.info["charge"]
    p.embeddings = embeddings
    return p


def test_ionize_keeps_atom_order():
    mol = neutralize(Chem.MolFromSmiles("NCC(=O)O"))
    for s in find_sites(mol):
        ion = P._ionize_keep_order(mol, s.atom, s.kind)
        assert [a.GetSymbol() for a in ion.GetAtoms()] == [a.GetSymbol() for a in mol.GetAtoms()]
        assert ion.GetAtomWithIdx(s.atom).GetFormalCharge() == (-1 if s.kind == "acid" else 1)


def test_mol_to_atoms_heavy_order():
    mol = Chem.MolFromSmiles("OC(=O)c1ccncc1")
    atoms = P._mol_to_atoms(mol)
    assert list(atoms.get_chemical_symbols()[:mol.GetNumAtoms()]) == \
        [a.GetSymbol() for a in mol.GetAtoms()]


def test_pool_site_is_local():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(30, 128))
    pos = np.c_[np.arange(30) * 1.5, np.zeros(30), np.zeros(30)]
    a = P.pool_site(emb, pos, [0])
    emb2 = emb.copy()
    emb2[-1] += 50          # perturb an atom 43 A away
    b = P.pool_site(emb2, pos, [0])
    assert a.shape == (512,)
    np.testing.assert_allclose(a[:384], b[:384], atol=1e-6)   # local blocks unchanged
    assert not np.allclose(a[384:], b[384:])                    # global block changes


def test_features_site_shape_and_predict_macro():
    p = _stub()
    mol = neutralize(Chem.MolFromSmiles("NCC(=O)O"))
    f = p.features_site(mol, find_sites(mol)[0])
    assert f.shape == (1, 1538)
    out = p.predict_macro("NCC(=O)O", pH=7.0)
    assert len(out["macro_pKas"]) == 2
    assert len(out["sites"]) == 2
    assert out["solvent"] == "Water"
    assert abs(sum(s["fraction"] for s in out["dominant_species"])) <= 1.0 + 1e-9


def test_predictor_without_model_gives_embeddings_but_not_predictions():
    """model_path=None is the UMA-only mode used by the delocalization
    descriptors: embeddings work, pKa prediction refuses clearly."""
    import umapka.predictor as P
    p = _stub()
    p.regressor = None
    p.feature_mode = "pair_v1"
    mol = Chem.MolFromSmiles("CC(=O)O")
    atoms = P._mol_to_atoms(mol)
    assert p.embeddings(atoms).shape[0] == atoms.get_global_number_of_atoms()
    with pytest.raises(RuntimeError, match="UMA only"):
        p._base_pka(np.zeros((1, 768)), "water")
