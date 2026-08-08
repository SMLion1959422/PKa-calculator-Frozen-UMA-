"""
Site-local pooling experiment - additive, opt-in, does NOT modify
predictor.py or any existing pickled model's expected input shape.

Hypothesis (from RESULTS.md): global mean/max pooling over every atom
dilutes the pKa signal on large molecules (error scales 0.68 -> 1.44
with size). This module adds a pooling variant centered on the actual
titratable atom, using per-atom embeddings PkaPredictor already
computes but currently discards after pooling.

RUN self_test() FIRST. It verifies the site-atom tracking survives
SMILES round-trips and 3D embedding correctly before you trust any
downstream MAE numbers from this.
"""
from __future__ import annotations
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms

from umapka.predictor import ACID_SITES, BASE_SITES, neutralize, _shift_hydrogen

_SITE_ATOM_MAP_NUM = 99


def protonation_pair_with_site(smiles: str) -> tuple[str, str]:
    """Same site-finding logic as umapka.predictor.protonation_pair,
    but tags the titratable atom with an RDKit atom-map number so it
    can be relocated after SMILES round-trips and 3D embedding.
    Returns (protonated_smiles, deprotonated_smiles); both carry the
    tag on the same physical atom.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles}")
    mol = neutralize(mol)

    for _, smarts, ai in ACID_SITES:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            idx = matches[0][ai]
            mol.GetAtomWithIdx(idx).SetAtomMapNum(_SITE_ATOM_MAP_NUM)
            dep = _shift_hydrogen(mol, idx, -1, -1)
            if dep is not None:
                return Chem.MolToSmiles(mol), Chem.MolToSmiles(dep)
            mol.GetAtomWithIdx(idx).SetAtomMapNum(0)

    for _, smarts, ai in BASE_SITES:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            idx = matches[0][ai]
            mol.GetAtomWithIdx(idx).SetAtomMapNum(_SITE_ATOM_MAP_NUM)
            pro = _shift_hydrogen(mol, idx, +1, +1)
            if pro is not None:
                return Chem.MolToSmiles(pro), Chem.MolToSmiles(mol)
            mol.GetAtomWithIdx(idx).SetAtomMapNum(0)

    raise RuntimeError(f"no titratable site found in {smiles}")


def smiles_to_atoms_with_site(smiles: str, seed: int = 42):
    """Like umapka.predictor._smiles_to_atoms, but also returns the
    0-based index (same ordering PkaPredictor.embeddings() uses) of
    the atom tagged with the site atom-map number, or None if the
    SMILES has no such tag.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles}")

    site_idx = None
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == _SITE_ATOM_MAP_NUM:
            site_idx = atom.GetIdx()
            atom.SetAtomMapNum(0)
            break

    charge = Chem.GetFormalCharge(mol)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"3D embedding failed for {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    atoms = Atoms(
        symbols=[a.GetSymbol() for a in mol.GetAtoms()],
        positions=mol.GetConformer().GetPositions(),
    )
    atoms.info = {"charge": int(charge), "spin": 1}
    return atoms, site_idx


def pool_site(emb: np.ndarray, positions: np.ndarray, site_idx: int,
              radius: float = 3.0) -> np.ndarray:
    """Site-local counterpart to PkaPredictor.pool(): instead of
    pooling over every atom, pool over the titratable atom itself
    plus whatever falls within `radius` angstrom of it in the
    3D-embedded conformer. Targets the 'global pooling dilutes local
    pKa signal on large molecules' failure mode from RESULTS.md.
    Returns a 256-dim vector (128 site-atom + 128 local mean), the
    same width as PkaPredictor.pool()'s mean+max output.
    """
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    d = np.linalg.norm(positions - positions[site_idx], axis=1)
    local_mask = d <= radius
    local_mask[site_idx] = True
    return np.concatenate([norm[site_idx], norm[local_mask].mean(0)])


def features_with_site(predictor, protonated: str, deprotonated: str,
                        radius: float = 3.0) -> np.ndarray:
    """Like PkaPredictor.features(), but appends site-local pooled
    features alongside the existing global mean+max pooling, for both
    protonation states and their difference. `protonated`/
    `deprotonated` must come from protonation_pair_with_site (plain
    SMILES with no atom-map tag raises ValueError). Returns a
    (1, 1536) array instead of features()'s (1, 768).
    """
    atoms_p, site_p = smiles_to_atoms_with_site(protonated)
    atoms_d, site_d = smiles_to_atoms_with_site(deprotonated)
    if site_p is None or site_d is None:
        raise ValueError(
            "no tagged site atom found - pass SMILES from "
            "protonation_pair_with_site, not plain protonation_pair"
        )

    emb_p = predictor.embeddings(atoms_p)
    emb_d = predictor.embeddings(atoms_d)

    h_p_global = predictor.pool(emb_p)
    h_d_global = predictor.pool(emb_d)
    h_p_local = pool_site(emb_p, atoms_p.get_positions(), site_p, radius)
    h_d_local = pool_site(emb_d, atoms_d.get_positions(), site_d, radius)

    h_p = np.concatenate([h_p_global, h_p_local])
    h_d = np.concatenate([h_d_global, h_d_local])
    return np.concatenate([h_p, h_d, h_p - h_d]).reshape(1, -1)


def self_test():
    """Sanity check: confirms the tagged/relocated site atom is
    chemically the right one for a handful of known molecules. Run
    this before trusting features_with_site or the experiment script
    for anything real - if any row says FAIL, stop and tell me.
    """
    cases = [
        ("CC(=O)O", "O"),     # acetic acid -> carboxylic O
        ("Oc1ccccc1", "O"),   # phenol -> phenolic O
        ("CCN", "N"),         # ethylamine -> amine N
        ("c1ccncc1", "N"),    # pyridine -> ring N
    ]
    print(f"{'SMILES':<14}{'expected':>10}{'prot atom':>12}{'deprot atom':>14}{'result':>8}")
    all_ok = True
    for smi, expected_elem in cases:
        prot, deprot = protonation_pair_with_site(smi)
        _, site_p = smiles_to_atoms_with_site(prot)
        _, site_d = smiles_to_atoms_with_site(deprot)
        elem_p = Chem.MolFromSmiles(prot).GetAtomWithIdx(site_p).GetSymbol() if site_p is not None else "?"
        elem_d = Chem.MolFromSmiles(deprot).GetAtomWithIdx(site_d).GetSymbol() if site_d is not None else "?"
        ok = (elem_p == expected_elem and elem_d == expected_elem)
        all_ok &= ok
        print(f"{smi:<14}{expected_elem:>10}{elem_p:>12}{elem_d:>14}{'OK' if ok else 'FAIL':>8}")
    print("\nALL PASS - safe to proceed" if all_ok else
          "\nSOME FAILED - do not trust results below this; tell Claude")


if __name__ == "__main__":
    self_test()
