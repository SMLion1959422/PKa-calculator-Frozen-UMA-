"""Reaction-site location and family labels for the solvent-shift experiment."""
import importlib.util
import pathlib
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")
spec = importlib.util.spec_from_file_location(
    "sse", pathlib.Path(__file__).parents[1] / "dev" / "solvent_shift_experiment.py")
sse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sse)


@pytest.mark.parametrize("rxn,element,fam", [
    ("CC(=O)O>>CC(=O)[O-]", "O", "carboxylic acid"),
    ("Oc1ccccc1>>[O-]c1ccccc1", "O", "phenol"),
    ("CC(=O)CC(C)=O>>CC(=O)[CH-]C(C)=O", "C", "C-H acid"),
    ("O=S(=O)(NC)c1ccccc1>>O=S(=O)([N-]C)c1ccccc1", "N", "N-H acid"),
    ("C[NH3+]>>CN", "N", "cationic acid (BH+)"),
    ("CCO>>CC[O-]", "O", "other O-H acid"),
])
def test_locate_site_and_family(rxn, element, fam):
    prot_s, dep_s = rxn.split(">>")
    prot, idx = sse.locate_site(prot_s, dep_s)
    assert idx is not None
    assert prot.GetAtomWithIdx(idx).GetSymbol() == element
    assert sse.family(prot, idx) == fam


def test_locate_site_rejects_non_proton_transfer():
    assert sse.locate_site("CCO", "CC[O-]C")[1] is None
