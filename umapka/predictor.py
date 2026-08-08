"""
umapka - pKa prediction from UMA foundation-model embeddings.

Uses Meta's UMA universal atomistic model as a frozen feature
extractor: per-atom embeddings are pulled from the input to the
energy head, pooled, and combined as paired protonated/deprotonated
difference features. A gradient-boosted regressor maps those to pKa.

Scope (please read before using):
  - Validated for MONOPROTIC acids and bases in the range pKa 2-12.
  - Also handles the FIRST ionization of simple polyprotic acids and
    bases in that range.
  - NOT reliable for: pKa2 and beyond, zwitterionic carboxyls
    (amino acids), or pKa outside 2-12.
See README for benchmark numbers and known limitations.
"""

from __future__ import annotations
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms

from . import site_finder

__all__ = ["PkaPredictor", "ACID_SITES", "BASE_SITES",
           "neutralize", "protonation_pair"]


# ---------------------------------------------------------------------
# Titratable-site definitions (SMARTS, atom index within the match)
# ---------------------------------------------------------------------
ACID_SITES = [
    ("carboxylic_acid", "[CX3](=O)[OX2H1]", 2),
    ("sulfonic_acid",   "[SX4](=O)(=O)[OX2H1]", 3),
    ("phosphoric_acid", "[PX4](=O)[OX2H1]", 2),
    ("tetrazole",       "c1nnn[nH]1", 0),
    ("tetrazole_2",     "c1nn[nH]n1", 0),
    ("sulfonamide_2",   "[SX4](=O)(=O)[NX3H1]", 3),
    ("sulfonamide_1",   "[SX4](=O)(=O)[NX3H2]", 3),
    ("thiol",           "[SX2H1]", 0),
    ("hydroxamic_acid",  "[CX3](=O)[NX3][OX2H1]", 3),
    ("phenol",          "[c][OX2H1]", 1),
    ("imide",           "[CX3](=O)[NX3H1][CX3]=O", 2),
    # Aromatic lactam N-H (2-pyridone, 4-pyridone, uracil-type rings):
    # deprotonation gives an aromatic, resonance-delocalized anion,
    # similar in mechanism to phenol - NOT the same as a plain acyclic
    # amide N-H (pKa ~15-17+, far out of range, deliberately NOT
    # matched here). Narrowly targeted to aromatic ring N-H directly
    # bonded to an aromatic ring carbon bearing an exocyclic C=O, so
    # ordinary amides stay unmatched. Found missing when 2-pyridone
    # (O=c1cccc[nH]1) raised "no titratable site found" during
    # predict_smart() testing - previously had neither an acid nor
    # base match at all for this real, drug-relevant scaffold.
    ("aromatic_lactam", "[nX3H1]c(=O)", 0),
    # Active-methylene C-H flanked by two electron-withdrawing groups.
    # Common and previously MISSING entirely: malonate/acetoacetate
    # esters, acetylacetone, Meldrum's acid, and real drugs such as
    # phenylbutazone/oxyphenbutazone (pKa ~4.4, C-H between an amide
    # C=O and a ketone C=O in the pyrazolidinedione ring). Without
    # these, protonation_pair() raised "no titratable site found" for
    # this whole common molecule class.
    ("malonate_dicarbonyl", "[CX4;H1,H2]([CX3]=[OX1])[CX3]=[OX1]", 0),
    ("cyanoacetate",        "[CX4;H1,H2]([CX3]=[OX1])C#N", 0),
    ("malononitrile",       "[CX4;H1,H2](C#N)C#N", 0),
    # Oxime O-H (pKa ~11-12) - distinct from hydroxamic acid (C(=O)N-OH);
    # oximes have no carbonyl, just C=N-OH, so the hydroxamic pattern
    # above never matches them.
    ("oxime", "[CX3]=[NX2][OX2H1]", 2),
    # --- expanded coverage ---
    ("boronic_acid", "[BX3]([OX2H1])[OX2H1]", 1),
    ("phosphonic_acid", "[PX4](=O)([OX2H1])[OX2H1]", 2),
    ("phosphinic_acid", "[PX4](=O)([OX2H1])[#6]", 2),
    ("sulfinic_acid", "[SX3](=O)[OX2H1]", 2),
    ("acyl_sulfonamide", "[CX3](=O)[NX3H1][SX4](=O)(=O)", 2),
    ("sulfonimide", "[SX4](=O)(=O)[NX3H1][SX4](=O)(=O)", 3),
    ("hydantoin", "[NX3H1]([CX3]=O)[CX3](=O)[NX3]", 0),
    ("barbiturate_NH", "O=[CX3][NX3H1][CX3]=O", 2),
    ("thiourea_NH", "[NX3H1][CX3](=[SX1])[NX3]", 0),
    ("carbamate_NH", "[NX3H1][CX3](=O)[OX2]", 0),
    ("enol", "[CX3]=[CX3][OX2H1]", 2),
    ("thiophenol", "[c][SX2H1]", 1),
    ("hydroxylamine_OH", "[NX3][OX2H1]", 1),
    ("pyrazole_NH", "[nX3H1]1[nX2][cX3][cX3][cX3]1", 0),
    ("imidazole_NH", "[nX3H1]1[cX3][nX2][cX3][cX3]1", 0),
    ("triazole_NH", "[nX3H1]1[nX2][nX2][cX3][cX3]1", 0),
    ("benzimidazole_NH", "[nX3H1]1[cX3][nX2]c2ccccc21", 0),
    ("purine_NH", "[nX3H1]1[cX3][nX2][cX3]2[cX3]1[nX2][cX3][nX2][cX3]2", 0),
    ("squaric_acid", "[OX2H1][CX3]1=[CX3][CX3](=O)[CX3]1=O", 0),
    ("vinylogous_acid", "[OX2H1][CX3]=[CX3][CX3]=O", 0),
    ("alpha_nitro_CH", "[CX4;H1,H2][NX3](=O)=O", 0),
    ("sulfone_CH", "[CX4;H1,H2][SX4](=O)(=O)", 0),
    ("nitramide", "[NX3H1][NX3](=O)=O", 0),
]

BASE_SITES = [
    ("guanidine",  "[NX3][CX3](=[NX2])[NX3]", 2),
    ("amidine",    "[NX3][CX3]=[NX2]", 2),
    ("prim_amine", "[NX3;H2;!$(N[C,S]=[O,S,N]);!$(N-a)]", 0),
    ("sec_amine",  "[NX3;H1;!$(N[C,S]=[O,S,N]);!$(N-a)]", 0),
    ("tert_amine", "[NX3;H0;!$(N[C,S]=[O,S,N]);!$(N-a)]", 0),
    # pyridine-like only: aromatic N, no H, NOT adjacent to another
    # aromatic N (excludes tetrazole/triazole/imidazole ring nitrogens,
    # which are not basic in this sense)
    ("pyridine_N", "[nX2;H0;!$(n~n)]", 0),
    # +0 charge constraint added: without it these three also match a
    # nitro group's nitrogen ([N+](=O)[O-] attached to an aromatic ring
    # is NX3;H0 with no other exclusion here), wrongly flagging nitro-
    # substituted arenes (e.g. nitrophenols) as having a second basic
    # site - the root cause of several of the worst outliers in
    # polyprotic_benchmark_v16.csv (2-nitrophenol, nitro-substituted
    # salicylic acid, etc.)
    ("aniline",    "[NX3;H2;+0]-a", 0),
    ("aniline_sec", "[NX3;H1;+0]-a", 0),
    ("aniline_tert","[NX3;H0;+0]-a", 0),
    # --- expanded coverage ---
    ("imidazole_N", "[nX2]1[cX3][nX3H1][cX3][cX3]1", 0),
    ("pyrimidine_N", "[nX2]1[cX3][nX2][cX3][cX3][cX3]1", 0),
    ("pyrazine_N", "[nX2]1[cX3][cX3][nX2][cX3][cX3]1", 0),
    ("oxazole_N", "[nX2]1[cX3][oX2][cX3][cX3]1", 0),
    ("thiazole_N", "[nX2]1[cX3][sX2][cX3][cX3]1", 0),
    ("triazine_N", "[nX2]1[cX3][nX2][cX3][nX2][cX3]1", 0),
    ("imine_N", "[NX2;H0,H1]=[CX3]", 0),
    ("hydrazine_N", "[NX3;H1,H2][NX3;H1,H2]", 0),
    ("hydroxylamine_N", "[NX3;H1,H2][OX2H1]", 0),
    ("phosphazene_N", "[NX2]=[PX4]", 0),
    ("piperazine_N", "[NX3;H1]1[CX4][CX4][NX3][CX4][CX4]1", 0),
    ("morpholine_N", "[NX3;H0,H1]1[CX4][CX4][OX2][CX4][CX4]1", 0),
]

_NEUTRALIZE_PATTERN = Chem.MolFromSmarts(
    "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
)


# ---------------------------------------------------------------------
# molecule utilities
# ---------------------------------------------------------------------
def neutralize(mol: Chem.Mol) -> Chem.Mol:
    """Strip formal charges where chemically reasonable.

    Public pKa datasets frequently store molecules already ionized,
    which prevents the neutral-form SMARTS above from matching.
    """
    rw = Chem.RWMol(mol)
    for (idx,) in rw.GetSubstructMatches(_NEUTRALIZE_PATTERN):
        atom = rw.GetAtomWithIdx(idx)
        charge, n_h = atom.GetFormalCharge(), atom.GetTotalNumHs()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(n_h - charge)
        atom.SetNoImplicit(True)
        atom.UpdatePropertyCache(strict=False)
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return out
    except Exception:
        return mol


def _shift_hydrogen(mol, idx, d_h, d_charge):
    """Change hydrogen count and formal charge at one atom.

    Round-trips through SMILES so hydrogen bookkeeping stays correct
    for any subsequent SMARTS matching.
    """
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(idx)
    n_h = atom.GetTotalNumHs() + d_h
    if n_h < 0:
        return None
    atom.SetNumExplicitHs(n_h)
    atom.SetNoImplicit(True)
    atom.SetFormalCharge(atom.GetFormalCharge() + d_charge)
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        round_trip = Chem.MolFromSmiles(Chem.MolToSmiles(out))
        return round_trip if round_trip is not None else out
    except Exception:
        return None


def protonation_pair(smiles: str) -> tuple[str, str]:
    """Return (protonated_smiles, deprotonated_smiles) differing by one proton.

    Site selection: the learned ranker (umapka.site_finder, trained on
    real ChemAxon site labels - 97-99% out-of-fold atom accuracy vs
    56-78% for fixed SMARTS-priority order, see site_finder.py) picks
    the atom among ALL SMARTS-matched candidates, acid or base, that
    it scores highest. Falls back to the original first-match priority
    walk below if the model file is missing or scoring fails for any
    reason, so this never hard-fails on molecules the ranker doesn't
    handle cleanly.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles}")
    mol = neutralize(mol)

    best = site_finder.best_site_atom(mol, ACID_SITES, BASE_SITES)
    if best is not None:
        idx, kind = best
        if kind == "acid":
            dep = _shift_hydrogen(mol, idx, -1, -1)
            if dep is not None:
                return Chem.MolToSmiles(mol), Chem.MolToSmiles(dep)
        else:
            pro = _shift_hydrogen(mol, idx, +1, +1)
            if pro is not None:
                return Chem.MolToSmiles(pro), Chem.MolToSmiles(mol)

    for _, smarts, ai in ACID_SITES:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            dep = _shift_hydrogen(mol, matches[0][ai], -1, -1)
            if dep is not None:
                return Chem.MolToSmiles(mol), Chem.MolToSmiles(dep)

    for _, smarts, ai in BASE_SITES:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            pro = _shift_hydrogen(mol, matches[0][ai], +1, +1)
            if pro is not None:
                return Chem.MolToSmiles(pro), Chem.MolToSmiles(mol)

    raise RuntimeError(f"no titratable site found in {smiles}")


# ---------------------------------------------------------------------
# site-tagged variant, for site-LOCAL embedding features (see
# PkaPredictor.features_local below). Kept separate from
# protonation_pair()/​_shift_hydrogen() above rather than modifying them,
# so existing behaviour/tests for those two functions are untouched.
# ---------------------------------------------------------------------
_SITE_TAG = 99  # temporary atom-map number; always cleared before the
                # SMILES is handed back to the caller


def _strip_map_tag(smiles: str) -> str:
    """Remove the temporary ':_SITE_TAG' atom-map annotation from a
    SMILES string via plain TEXT substitution - not by clearing the
    RDKit atom's map number and calling Chem.MolToSmiles() again.

    That second approach was tried and is WRONG: re-canonicalizing
    after clearing an atom's map number can silently reorder atoms in
    the output text relative to the first canonicalization pass (bug
    found via test_site_tagging.py - it passed for asymmetric sites
    like a carboxyl O, but failed for phenol/aniline/imidazole/
    benzenesulfonamide, where ring symmetry means the tag's presence
    or absence changes canonical tie-breaking). Since text substitution
    doesn't touch atom order at all, an index found by parsing the
    TAGGED string stays valid for parsing this STRIPPED string too -
    they're identical text except for the removed ":99".
    """
    return smiles.replace(f":{_SITE_TAG}]", "]")


def _tag_and_reparse(mol, idx):
    """Tag atom `idx` of `mol`, canonicalize ONCE, and return
    (clean_smiles, site_idx) where site_idx is valid when clean_smiles
    is independently re-parsed by a caller - see _strip_map_tag for why
    this must not canonicalize a second time after clearing the tag.
    """
    tagged = Chem.RWMol(mol)
    tagged.GetAtomWithIdx(idx).SetAtomMapNum(_SITE_TAG)
    try:
        tagged_smi = Chem.MolToSmiles(tagged)
    except Exception:
        return None, None
    rt = Chem.MolFromSmiles(tagged_smi)
    if rt is None:
        return None, None
    new_idx = None
    for a in rt.GetAtoms():
        if a.GetAtomMapNum() == _SITE_TAG:
            new_idx = a.GetIdx()
            break
    return (_strip_map_tag(tagged_smi), new_idx) if new_idx is not None else (None, None)


def _shift_hydrogen_tagged(mol, idx, d_h, d_charge):
    """Like _shift_hydrogen, but ALSO tags atom `idx` with a temporary
    atom-map number before canonicalizing, so its identity survives
    canonicalization (which does NOT preserve atom order/index - RDKit
    re-ranks atoms when producing canonical SMILES, so the original
    `idx` is meaningless in the returned molecule without this).

    Returns (clean_smiles, site_idx) or (None, None), where site_idx is
    valid when clean_smiles is independently re-parsed. Canonicalizes
    only ONCE and strips the tag via text substitution afterward (see
    _strip_map_tag) - NOT by clearing the atom's map number and calling
    Chem.MolToSmiles() again, which can silently reorder atoms and was
    an actual bug caught by test_site_tagging.py.
    """
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(idx)
    n_h = atom.GetTotalNumHs() + d_h
    if n_h < 0:
        return None, None
    atom.SetNumExplicitHs(n_h)
    atom.SetNoImplicit(True)
    atom.SetFormalCharge(atom.GetFormalCharge() + d_charge)
    atom.SetAtomMapNum(_SITE_TAG)
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        tagged_smi = Chem.MolToSmiles(out)
        rt = Chem.MolFromSmiles(tagged_smi)
        if rt is None:
            return None, None
        site_idx = None
        for a in rt.GetAtoms():
            if a.GetAtomMapNum() == _SITE_TAG:
                site_idx = a.GetIdx()
                break
        return (_strip_map_tag(tagged_smi), site_idx) if site_idx is not None else (None, None)
    except Exception:
        return None, None


def protonation_pair_site_tagged(smiles: str, return_kind: bool = False):
    """Like protonation_pair(), but also returns the ionizable atom's
    index within EACH final molecule (protonated and deprotonated),
    for site-local embedding features.

    Returns (protonated_smiles, prot_site_idx, deprotonated_smiles,
    deprot_site_idx) by default. With return_kind=True, returns a
    5-tuple with "acid" or "base" appended - existing callers using the
    4-tuple form are unaffected since this is opt-in.

Raises the same exceptions as protonation_pair() for unparseable/
    untitratable input - kept in sync with it by trying the same
    learned-ranker-then-SMARTS-priority selection, so callers get the
    same site choice either function would report.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles}")
    mol = neutralize(mol)

    best = site_finder.best_site_atom(mol, ACID_SITES, BASE_SITES)
    if best is not None:
        idx, kind = best
        if kind == "acid":
            prot_smi, prot_idx = _tag_and_reparse(mol, idx)
            dep_smi, dep_idx = _shift_hydrogen_tagged(mol, idx, -1, -1)
            if prot_smi is not None and dep_smi is not None:
                if return_kind:
                    return (prot_smi, prot_idx, dep_smi, dep_idx, "acid")
                return (prot_smi, prot_idx, dep_smi, dep_idx)
        else:
            base_smi, base_idx = _tag_and_reparse(mol, idx)
            pro_smi, pro_idx = _shift_hydrogen_tagged(mol, idx, +1, +1)
            if pro_smi is not None and base_smi is not None:
                if return_kind:
                    return (pro_smi, pro_idx, base_smi, base_idx, "base")
                return (pro_smi, pro_idx, base_smi, base_idx)

    for _, smarts, ai in ACID_SITES:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            site_atom_idx = matches[0][ai]
            # tag the UNCHANGED partner (the neutral acid) too, via the
            # SAME text-substitution-based helper as the shifted
            # partner below - see _tag_and_reparse/_strip_map_tag for
            # why a second canonicalization pass must be avoided.
            prot_smi, prot_idx = _tag_and_reparse(mol, site_atom_idx)
            dep_smi, dep_idx = _shift_hydrogen_tagged(mol, site_atom_idx, -1, -1)
            if prot_smi is not None and dep_smi is not None:
                if return_kind:
                    return (prot_smi, prot_idx, dep_smi, dep_idx, "acid")
                return (prot_smi, prot_idx, dep_smi, dep_idx)

    for _, smarts, ai in BASE_SITES:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            site_atom_idx = matches[0][ai]
            base_smi, base_idx = _tag_and_reparse(mol, site_atom_idx)
            pro_smi, pro_idx = _shift_hydrogen_tagged(mol, site_atom_idx, +1, +1)
            if pro_smi is not None and base_smi is not None:
                if return_kind:
                    return (pro_smi, pro_idx, base_smi, base_idx, "base")
                return (pro_smi, pro_idx, base_smi, base_idx)

    raise RuntimeError(f"no titratable site found in {smiles}")


def _smiles_to_atoms(smiles: str, seed: int = 42) -> Atoms:
    """SMILES -> single MMFF-optimized 3D conformer as an ASE Atoms.

    Net formal charge is read from the structure, never assigned by hand.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles}")
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
    return atoms


def _smiles_to_atoms_with_site(smiles: str, site_idx: int, seed: int = 42):
    """Like _smiles_to_atoms, but ALSO returns the site atom's index in
    the final ASE Atoms object, plus the RDKit mol-with-Hs (for
    topological neighbor-shell lookups in pool_local()).

    Safe because Chem.AddHs() APPENDS new H atoms after the existing
    heavy atoms and does not renumber them - so `site_idx`, which is
    valid on the pre-AddHs mol (from protonation_pair_site_tagged),
    stays valid after AddHs and in the resulting Atoms array, since
    both are built by iterating mol.GetAtoms() in the same order.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles}")
    if site_idx is None or site_idx >= mol.GetNumAtoms():
        raise ValueError(f"invalid site_idx {site_idx} for {smiles}")
    charge = Chem.GetFormalCharge(mol)
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        raise RuntimeError(f"3D embedding failed for {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol_h)
    except Exception:
        pass
    atoms = Atoms(
        symbols=[a.GetSymbol() for a in mol_h.GetAtoms()],
        positions=mol_h.GetConformer().GetPositions(),
    )
    atoms.info = {"charge": int(charge), "spin": 1}
    return atoms, site_idx, mol_h


def _smiles_to_atoms_with_site_multiconf(smiles: str, site_idx: int,
                                          n_confs: int = 3, seed: int = 42):
    """Like _smiles_to_atoms_with_site, but generates n_confs distinct
    MMFF-optimized conformers instead of one, returning a LIST of Atoms
    (one per conformer) sharing the same site_idx and mol_h (only the
    3D positions differ between conformers - the topology, and so the
    site atom's index, does not change).

    Intended for base sites specifically: a basic nitrogen's lone pair
    has real conformational freedom a single arbitrary conformer can
    misrepresent, unlike a rigid carboxylate/phenol O-H. Averaging the
    pooled embedding across a few conformers is a cheap way to reduce
    that noise - at ~n_confs x the UMA compute cost for that molecule,
    so use only where the site type actually warrants it.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles}")
    if site_idx is None or site_idx >= mol.GetNumAtoms():
        raise ValueError(f"invalid site_idx {site_idx} for {smiles}")
    charge = Chem.GetFormalCharge(mol)
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_confs, params=params)
    if len(cids) == 0:
        raise RuntimeError(f"3D embedding failed for {smiles}")
    atoms_list = []
    for cid in cids:
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, confId=cid)
        except Exception:
            pass
        atoms = Atoms(
            symbols=[a.GetSymbol() for a in mol_h.GetAtoms()],
            positions=mol_h.GetConformer(cid).GetPositions(),
        )
        atoms.info = {"charge": int(charge), "spin": 1}
        atoms_list.append(atoms)
    return atoms_list, site_idx, mol_h


def _smiles_to_atoms_with_site_bestconf(smiles: str, site_idx: int,
                                         n_confs: int = 10, seed: int = 42):
    """Like _smiles_to_atoms_with_site, but generates n_confs candidate
    conformers via ETKDGv3 and keeps only the LOWEST-MMFF-energy one,
    instead of trusting a single arbitrary ETKDG seed.

    Motivated by check_conformer_quality.py: across a 400-molecule
    sample, a single-shot conformer differs from the best of 10 by a
    median 2.28 Angstrom RMSD and mean 3.3 kcal/mol energy gap, BOTH
    growing with molecule size (2.0 kcal/mol / 2.1 A at <15 atoms vs
    5.0 kcal/mol / 4.3 A at >30 atoms) - exactly the bucket where
    prediction error is worst. Unlike UMA-based relaxation (tested via
    relax_with_uma/pilot_relaxed_geometry.py and found both far too
    expensive - ~134h projected for a full training re-embed - AND
    directionally negative, 0.950->0.990 MAE in that pilot), this stays
    entirely classical (RDKit/MMFF) for the SEARCH step. UMA cost stays
    at exactly ONE embedding pass per molecule state, identical to v3
    today, since only the single best-scoring geometry is ever handed
    to UMA.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles}")
    if site_idx is None or site_idx >= mol.GetNumAtoms():
        raise ValueError(f"invalid site_idx {site_idx} for {smiles}")
    charge = Chem.GetFormalCharge(mol)
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_confs, params=params)
    if len(cids) == 0:
        raise RuntimeError(f"3D embedding failed for {smiles}")

    best_cid, best_energy = None, float("inf")
    props = None
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol_h)
    except Exception:
        props = None
    for cid in cids:
        try:
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=cid)
            else:
                ff = None
            if ff is None:
                AllChem.MMFFOptimizeMolecule(mol_h, confId=cid)
                continue
            ff.Minimize()
            e = ff.CalcEnergy()
            if e < best_energy:
                best_energy, best_cid = e, cid
        except Exception:
            continue
    if best_cid is None:
        best_cid = cids[0]   # fallback: first conformer, best-effort only

    atoms = Atoms(
        symbols=[a.GetSymbol() for a in mol_h.GetAtoms()],
        positions=mol_h.GetConformer(best_cid).GetPositions(),
    )
    atoms.info = {"charge": int(charge), "spin": 1}
    return atoms, site_idx, mol_h


# ---------------------------------------------------------------------
# predictor
# ---------------------------------------------------------------------
class PkaPredictor:
    """pKa predictor built on frozen UMA embeddings.

    Parameters
    ----------
    model_path : str
        Path to the trained regressor (joblib .pkl).
    uma_model : str
        UMA checkpoint name, e.g. ``"uma-s-1p1"``. Requires a
        HuggingFace account with access to ``facebook/UMA``.
    device : str
        ``"cuda"`` or ``"cpu"``.

    Examples
    --------
    >>> p = PkaPredictor("model_core.pkl")
    >>> p.predict("CC(=O)O")          # acetic acid
    4.23
    """

    def __init__(self, model_path: str,
                 uma_model: str = "uma-s-1p1",
                 device: str | None = None,
                 free_energy_model_path: str | None = None,
                 multisolvent_model_path: str | None = "models/multisolvent_tuned.pkl"):
        import torch, joblib
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._torch = torch

        predictor = pretrained_mlip.get_predict_unit(uma_model, device=device)
        self._calc = FAIRChemCalculator(predictor, task_name="omol")

        # the model is built lazily; one forward pass materializes it
        probe = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
        probe.info = {"charge": 0, "spin": 1}
        probe.calc = self._calc
        probe.get_potential_energy()

        module = predictor.tracked_modules()["model"]
        self._energy_head = (
            module.module.output_heads["energyandforcehead"].head.energy_block
        )
        try:
            self._calc.use_cache = False   # caching would skip the forward pass
        except Exception:
            pass

        self._buffer = {}
        self.regressor = joblib.load(model_path)

        # OPTIONAL second regressor for predict_chain() - a DIFFERENT
        # model trained by train_free_energy_model.py on single-state
        # embedding differences. Not the same file as model_path above;
        # predict()/predict_site()/predict_all_sites() never touch this.
        self._free_energy_model = (
            joblib.load(free_energy_model_path)
            if free_energy_model_path else None
        )

        # OPTIONAL third regressor for non-aqueous solvents (`solvent=`
        # argument on predict()/predict_site()/predict_detailed()).
        # Loaded LAZILY on first non-water use, not here, so importing
        # this class doesn't require multisolvent_tuned.pkl to exist if
        # you only ever predict in water. See _load_multisolvent().
        self._multisolvent_model_path = multisolvent_model_path
        self._multisolvent_bundle = None

    # -- embedding extraction ------------------------------------------
    def _hook(self, module, inputs):
        if inputs and self._torch.is_tensor(inputs[0]):
            self._buffer["h"] = inputs[0].detach().float().cpu().numpy()

    def embeddings(self, atoms: Atoms) -> np.ndarray:
        """(n_atoms, 128) per-atom embeddings from the energy-head input."""
        self._buffer.clear()
        handle = self._energy_head[0].register_forward_pre_hook(self._hook)
        try:
            a = atoms.copy()
            a.calc = self._calc
            self._calc.reset()
            a.get_potential_energy()
        finally:
            handle.remove()
        emb = self._buffer.get("h")
        if emb is None or emb.shape[0] != len(atoms):
            raise RuntimeError("embedding extraction failed")
        return emb

    def get_energy(self, atoms: Atoms) -> float:
        """Raw UMA-computed potential energy (eV) for a given structure.

        Unlike embeddings() - which deliberately hooks an internal
        layer and never uses the final energy output, because Section
        III of the paper showed UMA's absolute energies for CHARGED
        species are unreliable (scatter 2.5-3.5x the pKa signal itself)
        - this DOES use the energy output directly, for tautomer
        ranking. That's a fundamentally different, easier comparison:
        neutral, isoelectronic structures differing only in proton
        placement, typically a few kcal/mol apart, not the charged-
        species free energy differences that were shown to fail.
        """
        a = atoms.copy()
        a.calc = self._calc
        self._calc.reset()
        return float(a.get_potential_energy())

    def rank_tautomers(self, smiles: str, max_tautomers: int = 8) -> dict:
        """Enumerate tautomers (RDKit's TautomerEnumerator) and rank
        them by UMA-computed GAS-PHASE energy, identifying the lowest-
        energy one as the canonical representative to run pKa site-
        detection on instead of whatever tautomeric form the input
        SMILES happened to be written in.

        HONEST LIMITATION, read before trusting a "changed" result:
        this is gas-phase only - no implicit solvation model. Real
        tautomer preference can differ between gas phase and aqueous
        solution; the classic textbook case is 2-hydroxypyridine
        (mildly favored gas-phase) vs 2-pyridone (favored in water).
        Treat a "changed" result as informative, not definitive,
        especially for zwitterionic or highly polar tautomers where
        solvation is known to matter most - this is a real, acknowledged
        gap relative to tools that explicitly model implicit solvation
        at this step (e.g. QupKake's GFN2-xTB + implicit water).

        NOT wired into predict()/predict_site() automatically - this is
        opt-in and adds real cost (up to max_tautomers extra UMA energy
        evaluations). Call it yourself first and feed the returned
        canonical_smiles into predict() if you want it applied.

        Returns: {"canonical_smiles": str (lowest-energy tautomer),
                  "original_smiles": str (input, canonicalized),
                  "changed": bool,
                  "n_tautomers": int (enumerated, before the cap),
                  "n_scored": int (successfully energy-evaluated),
                  "energies": list[(smiles, energy_eV)] sorted
                              ascending, or None if only one tautomer
                              was found}
        """
        from rdkit.Chem.MolStandardize import rdMolStandardize

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"could not parse SMILES: {smiles}")
        original_canonical = Chem.MolToSmiles(mol)

        enumerator = rdMolStandardize.TautomerEnumerator()
        tautomers = list(enumerator.Enumerate(mol))
        if len(tautomers) <= 1:
            return {"canonical_smiles": original_canonical,
                    "original_smiles": original_canonical,
                    "changed": False, "n_tautomers": 1, "n_scored": 1,
                    "energies": None}

        n_enumerated = len(tautomers)
        if n_enumerated > max_tautomers:
            tautomers = tautomers[:max_tautomers]

        results = []
        for taut in tautomers:
            try:
                taut_smiles = Chem.MolToSmiles(taut)
                atoms = _smiles_to_atoms(taut_smiles)
                energy = self.get_energy(atoms)
                results.append((taut_smiles, energy))
            except Exception:
                continue

        if not results:
            return {"canonical_smiles": original_canonical,
                    "original_smiles": original_canonical,
                    "changed": False, "n_tautomers": n_enumerated,
                    "n_scored": 0, "energies": None,
                    "warning": "all tautomer energy calculations failed"}

        results.sort(key=lambda x: x[1])
        best_smiles, best_energy = results[0]
        return {"canonical_smiles": best_smiles,
                "original_smiles": original_canonical,
                "changed": best_smiles != original_canonical,
                "n_tautomers": n_enumerated, "n_scored": len(results),
                "energies": results}

    def rank_tautomers_multiconf(self, smiles: str, max_tautomers: int = 8,
                                  n_confs: int = 5) -> dict:
        """Like rank_tautomers(), but samples n_confs conformers PER
        tautomer and keeps each tautomer's LOWEST UMA energy across
        them, instead of trusting a single arbitrary ETKDG conformer.

        Built specifically because rank_tautomers() got acetylacetone's
        keto/enol preference backwards relative to well-established
        gas-phase chemistry (real gas-phase equilibrium is ~92% enol,
        stabilized by a resonance-assisted intramolecular H-bond
        forming a 6-membered ring). Leading hypothesis: a single
        arbitrary MMFF conformer per tautomer SMILES has no reason to
        find that specific H-bonded geometry the enol's stability
        depends on - consistent with check_conformer_quality.py's
        finding that single-shot conformers can differ from a best-of-10
        search by up to 130 kcal/mol. This tests that hypothesis
        directly rather than assuming it.

        Costs n_confs x more UMA energy evaluations per tautomer than
        rank_tautomers() - still cheap in absolute terms (seconds) for
        a handful of tautomers on a few molecules, but time it before
        applying to anything larger-scale.
        """
        from rdkit.Chem.MolStandardize import rdMolStandardize
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"could not parse SMILES: {smiles}")
        original_canonical = Chem.MolToSmiles(mol)

        enumerator = rdMolStandardize.TautomerEnumerator()
        tautomers = list(enumerator.Enumerate(mol))
        if len(tautomers) <= 1:
            return {"canonical_smiles": original_canonical,
                    "original_smiles": original_canonical,
                    "changed": False, "n_tautomers": 1, "n_scored": 1,
                    "energies": None}

        n_enumerated = len(tautomers)
        if n_enumerated > max_tautomers:
            tautomers = tautomers[:max_tautomers]

        results = []
        for taut in tautomers:
            try:
                taut_smiles = Chem.MolToSmiles(taut)
                taut_mol = Chem.MolFromSmiles(taut_smiles)
                charge = Chem.GetFormalCharge(taut_mol)
                mol_h = Chem.AddHs(taut_mol)
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_confs, params=params)
                if len(cids) == 0:
                    continue
                best_energy = float("inf")
                for cid in cids:
                    try:
                        AllChem.MMFFOptimizeMolecule(mol_h, confId=cid)
                    except Exception:
                        pass
                    atoms = Atoms(
                        symbols=[a.GetSymbol() for a in mol_h.GetAtoms()],
                        positions=mol_h.GetConformer(cid).GetPositions(),
                    )
                    atoms.info = {"charge": int(charge), "spin": 1}
                    e = self.get_energy(atoms)
                    if e < best_energy:
                        best_energy = e
                results.append((taut_smiles, best_energy))
            except Exception:
                continue

        if not results:
            return {"canonical_smiles": original_canonical,
                    "original_smiles": original_canonical,
                    "changed": False, "n_tautomers": n_enumerated,
                    "n_scored": 0, "energies": None,
                    "warning": "all tautomer energy calculations failed"}

        results.sort(key=lambda x: x[1])
        best_smiles, best_energy = results[0]
        return {"canonical_smiles": best_smiles,
                "original_smiles": original_canonical,
                "changed": best_smiles != original_canonical,
                "n_tautomers": n_enumerated, "n_scored": len(results),
                "energies": results}

    def rank_tautomers_safe(self, smiles: str, max_tautomers: int = 8,
                             n_confs: int = 5) -> dict:
        """Gated wrapper around rank_tautomers_multiconf: only trusts
        (safe_to_apply=True) a tautomer correction when it does NOT
        change the aromatic ring system.

        WHY THIS GATE, SPECIFICALLY: direct testing found a clean
        pattern - gas-phase UMA ranking was correct on every NON-
        aromatic case tried (acetylacetone enol preference matching
        real ~92% gas-phase enol content; ethyl acetoacetate correctly
        flipping to keto-preferred, consistent with an ester being a
        weaker H-bond acceptor than a ketone; acetamide correctly and
        heavily favoring the amide form). But 2-/4-hydroxypyridine -
        both AROMATIC ring tautomers - are KNOWN cases where gas-phase
        preference disagrees with real aqueous behavior (pKa is
        inherently an aqueous-phase question, and this tool has no
        implicit solvation model). Rather than silently applying a
        correction that's validated for one chemical class but known-
        risky for another, this checks whether the winning tautomer
        actually changed the count of aromatic atoms, and only sets
        safe_to_apply=True when it didn't.

        This is a heuristic based on the specific cases tested so far,
        not a proof - if you find a non-aromatic case where gas-phase
        ranking is wrong, or an aromatic case where it's actually
        fine, that's real information that should update this gate,
        not be discarded.

        Returns the same dict as rank_tautomers_multiconf, plus:
          "safe_to_apply": bool
          "warning": str, present when safe_to_apply is False
        """
        result = self.rank_tautomers_multiconf(smiles, max_tautomers, n_confs)
        if not result["changed"]:
            result["safe_to_apply"] = True
            return result

        orig_mol = Chem.MolFromSmiles(result["original_smiles"])
        new_mol = Chem.MolFromSmiles(result["canonical_smiles"])
        if orig_mol is None or new_mol is None:
            result["safe_to_apply"] = False
            result["warning"] = "could not re-parse for aromaticity check"
            return result

        # FIXED (was comparing aromatic-atom COUNT before vs after,
        # which never fires for ring tautomers like pyridone/
        # hydroxypyridine - RDKit perceives BOTH forms as fully
        # aromatic with the SAME atom count, since the SMILES for
        # both is written in lowercase; that's exactly why the
        # 2-pyridone test case slipped through as a false "safe").
        # What actually distinguishes the validated-safe cases
        # (acetylacetone, ethyl acetoacetate, acetamide - zero
        # aromatic atoms in any tautomer) from the validated-risky
        # ones (2-/4-hydroxypyridine - fully aromatic ring in every
        # tautomer) is simply whether aromaticity is PRESENT at all,
        # not whether it changed.
        has_aromatic = (any(a.GetIsAromatic() for a in orig_mol.GetAtoms()) or
                         any(a.GetIsAromatic() for a in new_mol.GetAtoms()))
        result["safe_to_apply"] = not has_aromatic
        if not result["safe_to_apply"]:
            result["warning"] = (
                "this molecule involves an aromatic ring - gas-phase "
                "ranking is validated as potentially unreliable for this "
                "class (see 2-/4-hydroxypyridine testing, and the "
                "2-pyridone case that first caught this exact bug); NOT "
                "auto-applying, flagging only. canonical_smiles still "
                "reports the gas-phase-lowest-energy tautomer for "
                "reference, but do not use it automatically."
            )
        return result

    @staticmethod
    def pool(emb: np.ndarray) -> np.ndarray:
        """L2-normalize per atom, then concatenate mean and max pooling.

        Per-atom normalization matters: raw mean-pooling is dominated
        by a few high-magnitude atoms.
        """
        norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        return np.concatenate([norm.mean(0), norm.max(0)])

    @staticmethod
    def pool_local(emb: np.ndarray, site_idx: int, mol_with_hs,
                    max_shell: int = 2) -> np.ndarray:
        """L2-normalize per atom, then concatenate the ionizable atom's
        OWN vector with the mean over its `max_shell`-bond neighborhood
        (site atom included).

        Complements the global pool() above, which mean/max-pools over
        EVERY atom in the molecule - fine for small molecules, but the
        site's signal gets diluted against dozens of unrelated atoms in
        large ones (see RESULTS.md: "global mean-pooling dilutes local
        pKa signal on large molecules" / error grows monotonically with
        molecule size). This gives the regressor a second, size-
        invariant view centered on the actual atom gaining/losing the
        proton.
        """
        norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        site_vec = norm[site_idx]
        dmat = Chem.GetDistanceMatrix(mol_with_hs)
        shell = np.where(dmat[site_idx] <= max_shell)[0]
        shell_mean = norm[shell].mean(0)
        return np.concatenate([site_vec, shell_mean])

    @staticmethod
    def pool_local_multiscale(emb: np.ndarray, site_idx: int, mol_with_hs,
                               shells: tuple = (1, 2, 3)) -> np.ndarray:
        """Like pool_local, but concatenates shell means at SEVERAL bond
        radii instead of one fixed radius (default 2).

        v3's single-radius (2-bond) local pooling helped size-related
        error a lot but barely moved the 3+-rings bucket - conjugation
        in fused/polycyclic systems can extend further than 2 bonds, so
        this lets the regressor use whichever radius actually matters
        per molecule instead of a single guessed value. Shape:
        128 (site) + 128*len(shells) - e.g. 128+128*3=512 for the
        default 3 shells, vs pool_local's 256.
        """
        norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        site_vec = norm[site_idx]
        dmat = Chem.GetDistanceMatrix(mol_with_hs)
        shell_means = []
        for r in shells:
            shell = np.where(dmat[site_idx] <= r)[0]
            shell_means.append(norm[shell].mean(0))
        return np.concatenate([site_vec] + shell_means)

    def _regressor_expected_dim(self) -> int:
        """How many features the loaded aqueous regressor expects.
        Falls back to 768 (the original global-only dimension) if it
        can't be read off the model, so anything trained the old way
        keeps working exactly as before.
        """
        reg = self.regressor
        if isinstance(reg, dict) and "regressor" in reg:
            reg = reg["regressor"]
        n = getattr(reg, "n_features_in_", None)
        return int(n) if n else 768

    def _state_features(self, smiles: str, site_idx: int | None,
                         need_local: bool):
        """One UMA forward pass per molecule state, producing the
        global pooled vector and (if needed) the local pooled vector
        from the SAME per-atom embeddings - not a second forward pass.
        """
        if need_local:
            if site_idx is None:
                raise ValueError(
                    "site-local features requested but site_idx is None - "
                    "use protonation_pair_site_tagged() (or sites()/"
                    "predict_site()'s internal tagging) to get one"
                )
            atoms, s_idx, mol_h = _smiles_to_atoms_with_site(smiles, site_idx)
            emb = self.embeddings(atoms)
            return self.pool(emb), self.pool_local(emb, s_idx, mol_h)
        return self.pool(self.embeddings(_smiles_to_atoms(smiles))), None

    def features(self, protonated: str, deprotonated: str,
                 prot_site: int | None = None,
                 deprot_site: int | None = None) -> np.ndarray:
        """[h_prot ; h_deprot ; h_prot - h_deprot], shape (1, 768).

        pKa describes a transition rather than a molecule, so both
        charge states and their difference are encoded.

        If the loaded aqueous regressor was trained on the combined
        global+local feature space (n_features_in_ == 1536, e.g.
        model_core_v3-style), `prot_site`/`deprot_site` (atom indices
        from protonation_pair_site_tagged()) are required and the
        site-local block (see pool_local/features_local) is appended,
        giving shape (1, 1536) instead. Existing 768-dim models
        (model_core.pkl, model_core_v2.pkl, multisolvent_tuned.pkl)
        are completely unaffected - this only activates for a
        regressor that actually expects the larger input.
        """
        expected = self._regressor_expected_dim()
        need_local = (expected == 1536)
        hg_p, hl_p = self._state_features(protonated, prot_site, need_local)
        hg_d, hl_d = self._state_features(deprotonated, deprot_site, need_local)
        global_feat = np.concatenate([hg_p, hg_d, hg_p - hg_d])
        if not need_local:
            return global_feat.reshape(1, -1)
        local_feat = np.concatenate([hl_p, hl_d, hl_p - hl_d])
        return np.concatenate([global_feat, local_feat]).reshape(1, -1)

    def features_local(self, protonated: str, prot_site: int,
                        deprotonated: str, deprot_site: int) -> np.ndarray:
        """Site-local counterpart to features(): [local_prot ;
        local_deprot ; local_prot - local_deprot], shape (1, 768) -
        same shape as features()'s global block, so
        np.concatenate([features(...), features_local(...)]) gives the
        1536-dim vector model_core_v3-style training expects. Exposed
        directly (rather than only via features()'s auto-detection)
        for use in the re-embedding script, which needs the local block
        on its own to build feat_train_v3.pkl.
        """
        atoms_p, idx_p, mol_p = _smiles_to_atoms_with_site(protonated, prot_site)
        atoms_d, idx_d, mol_d = _smiles_to_atoms_with_site(deprotonated, deprot_site)
        h_p = self.pool_local(self.embeddings(atoms_p), idx_p, mol_p)
        h_d = self.pool_local(self.embeddings(atoms_d), idx_d, mol_d)
        return np.concatenate([h_p, h_d, h_p - h_d]).reshape(1, -1)

    def state_features_v4(self, smiles: str, site_idx: int, kind: str,
                           n_confs_base: int = 3):
        """v4 experiment: multi-scale shell pooling for everything, PLUS
        multi-conformer averaging specifically for base sites (kind ==
        "base") - see pool_local_multiscale/_smiles_to_atoms_with_site_
        multiconf docstrings for the reasoning. Acid sites stay
        single-conformer (rigid O-H/COOH, not expected to need it),
        keeping the extra compute cost targeted rather than blanket.

        Returns (global_pooled, local_multiscale_pooled) for ONE
        molecule state (caller combines protonated/deprotonated/diff
        the same way features()/features_local() do). Kept separate
        from those - not wired into features()'s auto-dimension
        dispatch - so this is purely additive: v2/v3 inference via
        predict_pka.py is completely unaffected regardless of whether
        this experiment pans out.
        """
        if kind == "base":
            atoms_list, s_idx, mol_h = _smiles_to_atoms_with_site_multiconf(
                smiles, site_idx, n_confs=n_confs_base)
            globals_, locals_ = [], []
            for atoms in atoms_list:
                emb = self.embeddings(atoms)
                globals_.append(self.pool(emb))
                locals_.append(self.pool_local_multiscale(emb, s_idx, mol_h))
            return np.mean(globals_, axis=0), np.mean(locals_, axis=0)
        else:
            atoms, s_idx, mol_h = _smiles_to_atoms_with_site(smiles, site_idx)
            emb = self.embeddings(atoms)
            return self.pool(emb), self.pool_local_multiscale(emb, s_idx, mol_h)

    def relax_with_uma(self, atoms: Atoms, fmax: float = 0.05,
                        max_steps: int = 60) -> Atoms:
        """Refine an already MMFF-optimized geometry using UMA's OWN
        forces, via ASE's FIRE optimizer.

        Motivation: our own benchmarking (see paper Section III) found
        UMA's predicted STRUCTURES and forces reliable even though its
        absolute energies for charged/partially-charged species were
        not. MMFF94 is a generic classical force field with no knowledge
        of what UMA's own learned potential energy surface actually
        looks like - a geometry that's a good MMFF minimum is not
        necessarily a good UMA minimum, and the embeddings we extract
        are conditioned on whatever geometry we hand UMA. This uses
        exactly the part of UMA already shown trustworthy (forces/
        structure) to move the starting geometry toward UMA's own idea
        of a minimum before embedding extraction, without touching the
        part shown NOT trustworthy (absolute energies).

        Falls back gracefully: if optimization raises (rare, e.g. a
        numerically difficult step), returns whatever geometry was
        reached rather than failing the whole embedding.
        """
        from ase.optimize import FIRE
        a = atoms.copy()
        a.info = dict(atoms.info)
        a.calc = self._calc
        try:
            opt = FIRE(a, logfile=None)
            opt.run(fmax=fmax, steps=max_steps)
        except Exception:
            pass
        return a

    def state_features_relaxed(self, smiles: str, site_idx: int,
                                fmax: float = 0.05, max_steps: int = 60):
        """v6 experiment: same global+local-2-bond-shell pooling as v3
        (state_features_v4's simpler predecessor), but on a UMA-relaxed
        geometry instead of raw MMFF. Isolates the "better 3D structure"
        variable cleanly against the v3 baseline (0.832 external MAE),
        without also stacking v4's multi-conformer/multi-scale changes,
        which our own ablations showed were flat-to-negative - so any
        change here is attributable to geometry quality specifically.

        Returns (global_pooled, local_pooled) for ONE molecule state,
        same shapes as features()/features_local()'s per-state outputs
        (256-dim global, 256-dim local) so this drops into the same
        768+768=1536-dim combination v3 used.
        """
        atoms, s_idx, mol_h = _smiles_to_atoms_with_site(smiles, site_idx)
        relaxed = self.relax_with_uma(atoms, fmax=fmax, max_steps=max_steps)
        emb = self.embeddings(relaxed)
        return self.pool(emb), self.pool_local(emb, s_idx, mol_h)

    def state_features_bestconf(self, smiles: str, site_idx: int,
                                 n_confs: int = 10):
        """v7 experiment: same global+local-2-bond-shell pooling as v3,
        but on the LOWEST-MMFF-ENERGY conformer out of n_confs candidates
        instead of a single arbitrary ETKDG seed's result - see
        _smiles_to_atoms_with_site_bestconf's docstring for why this
        replaces the (failed, too-expensive) UMA-relaxation approach.

        UMA cost is IDENTICAL to v3 (one embedding pass per state) - the
        extra n_confs work is pure classical MMFF, not UMA, so this
        should be roughly as fast as v3's original re-embed, not the
        134-hour-projected cost the relaxation approach had.

        Returns (global_pooled, local_pooled) for ONE molecule state,
        same shapes/combination convention as state_features_relaxed.
        """
        atoms, s_idx, mol_h = _smiles_to_atoms_with_site_bestconf(
            smiles, site_idx, n_confs=n_confs)
        emb = self.embeddings(atoms)
        return self.pool(emb), self.pool_local(emb, s_idx, mol_h)

    # -- multi-solvent support -------------------------------------------
    def _load_multisolvent(self):
        """Lazily load the multisolvent regressor bundle on first use.
        Not loaded in __init__ so predicting only-ever-in-water never
        requires multisolvent_tuned.pkl to exist.
        """
        if self._multisolvent_bundle is None:
            import joblib
            if not self._multisolvent_model_path:
                raise RuntimeError(
                    "no multisolvent_model_path configured - construct "
                    "PkaPredictor(..., multisolvent_model_path="
                    "'models/multisolvent_tuned.pkl') to enable solvent!='water'"
                )
            self._multisolvent_bundle = joblib.load(self._multisolvent_model_path)
        return self._multisolvent_bundle

    def _base_pka(self, pair_feat: np.ndarray, solvent: str) -> float:
        """Route a 768-dim pair feature through the right regressor for
        `solvent`. Water uses the dedicated aqueous regressor
        (self.regressor - the more accurate, water-only model); any
        other solvent uses the multisolvent regressor with the two
        extra solvent-descriptor features it was trained on, in the
        EXACT encoding tune_multisolvent.py used (see umapka.solvents -
        NOT raw physical constants).
        """
        from . import solvents as _solvents
        info = _solvents.resolve_solvent(solvent)

        # Force contiguous float64 arrays before every LightGBM .predict()
        # call - prevents a Windows-only memory access violation with
        # non-contiguous arrays. Applied to every regressor path (water
        # dict-bundle, plain water regressor, and multisolvent), not just
        # one of them.
        feat_arr = np.ascontiguousarray(pair_feat, dtype=np.float64)

        if info.name == "Water":
            # model_core.pkl is a plain regressor. model_core_v2.pkl (the
            # RECOMMENDED aqueous model - RESULTS.md: leakage-fixed,
            # Novartis MAE 1.16 vs 1.41 for model_core.pkl) is instead a
            # {"regressor":..., "calibrator":...} bundle: raw regressor
            # output is passed through a calibrator before it's a real
            # pKa. Handle both shapes rather than silently mis-scoring
            # one of them.
            if isinstance(self.regressor, dict) and "regressor" in self.regressor:
                raw = float(self.regressor["regressor"].predict(feat_arr)[0])
                if "calibrator" in self.regressor and self.regressor["calibrator"] is not None:
                    return float(self.regressor["calibrator"].predict([raw])[0])
                return raw
            return float(self.regressor.predict(feat_arr)[0])
        bundle = self._load_multisolvent()
        # multisolvent_tuned.pkl was trained on the 768-dim GLOBAL-ONLY
        # feature layout + 2 solvent descriptors (770 total). features()
        # may return 1536-dim (global+site-local) if the loaded aqueous
        # regressor is a v3-style model - global_feat is always the
        # first 768 columns of that (see features()'s concatenation
        # order: global block built and returned/extended first), so
        # slice it down regardless of which aqueous regressor produced
        # pair_feat. Without this, non-water solvents silently broke
        # (770 vs 1538 feature-count mismatch) as soon as the default
        # aqueous model switched from 768-dim to 1536-dim.
        feat_global = feat_arr.reshape(-1)[:768]
        feat = np.concatenate([feat_global, [info.eps_norm, info.protic]]).reshape(1, -1)
        feat = np.ascontiguousarray(feat, dtype=np.float64)
        return float(bundle["model"].predict(feat)[0])

    # -- prediction -----------------------------------------------------
    def predict(self, smiles: str, solvent: str = "water",
                salt: str | None = None,
                salt_concentration: float | None = None) -> float:
        """Predict pKa for the first titratable site found, in `solvent`
        (default water - the best-validated case, MAE 0.994 scaffold
        split). Other solvents route through the multisolvent regressor;
        see umapka.solvents.SOLVENTS for which are supported and their
        held-out MAE. For solvent MIXTURES use
        ``umapka.mixtures.predict_mixed_solvent_pka`` instead - a single
        `solvent` string here always means one pure solvent.

        `salt` / `salt_concentration` (mol/L) apply a physics-based
        ionic-strength correction (see ``umapka.solvation``) on top of
        the base prediction for whichever solvent you picked; they
        don't change which solvent the base prediction is for. Use
        ``predict_detailed`` if you want the correction tier/warnings
        rather than just the final float.
        """
        return self.predict_detailed(smiles, solvent, salt, salt_concentration)["pKa"]

    def predict_detailed(self, smiles: str, solvent: str = "water",
                          salt: str | None = None,
                          salt_concentration: float | None = None) -> dict:
        """Like ``predict``, but returns
        {"pKa": float, "base_pKa": float, "solvent": str, "correction": dict}
        where ``correction`` is the raw dict from
        ``solvation.predict_salt_correction`` - includes which tier
        fired (davies / davies+ion-pairing / pitzer / none) and any
        validity warnings. Inspect this rather than trusting the
        headline number blindly when a salt is specified.
        """
        from . import solvents as _solvents
        info = _solvents.resolve_solvent(solvent)
        prot, prot_idx, deprot, deprot_idx = protonation_pair_site_tagged(smiles)
        base = self._base_pka(
            self.features(prot, deprot, prot_idx, deprot_idx), solvent)
        if salt is None:
            return {"pKa": base, "base_pKa": base, "solvent": info.name,
                    "correction": {"shift": 0.0, "tier": "none",
                                   "note": "no salt specified"}}
        if info.solvation_key is None:
            raise ValueError(
                f"no ionic-strength model available for {info.name} - "
                f"umapka.solvation only covers "
                f"{sorted(s.solvation_key for s in _solvents.SOLVENTS.values() if s.solvation_key)}"
            )

        import importlib
        solvation = importlib.import_module(".solvation", __package__)
        # the site actually used by protonation_pair() is, by
        # construction, sites(smiles)[0] - same priority-ordered scan
        site_kind = self.sites(smiles)[0]["kind"]
        correction = solvation.predict_salt_correction(
            site_kind, salt, salt_concentration, solvent=info.solvation_key)
        return {"pKa": base + correction["shift"], "base_pKa": base,
                "solvent": info.name, "correction": correction}

    def predict_smart(self, smiles: str, solvent: str = "water",
                       salt: str | None = None,
                       salt_concentration: float | None = None,
                       use_tautomer_correction: bool = True,
                       use_site_disambiguation: bool = True) -> dict:
        """The integrated 3-step workflow: tautomer normalization ->
        site selection -> pKa prediction - the same overall structure
        used by leading pKa tools, built from the pieces validated
        tonight rather than a supervised site-classifier (which would
        need per-site training labels this project's data doesn't
        have).

        1. Tautomer correction (rank_tautomers_safe): if the input
           isn't already the lowest-gas-phase-energy tautomer, and
           switching is validated-safe (does not touch an aromatic
           ring - see rank_tautomers_safe), the corrected tautomer
           becomes the working structure.
        2. Same-type site disambiguation (rank_same_type_sites): if
           the site protonation_pair_site_tagged() would pick by
           default belongs to a group with multiple chemically
           inequivalent candidates (e.g. two inequivalent phenols),
           and a DIFFERENT candidate in that group is UMA-favored,
           that candidate is used instead of the arbitrary first
           SMARTS match.
        3. Standard prediction (predict_site/predict_detailed) on the
           resulting structure and site.

        Fully opt-in - predict()/predict_site()/predict_detailed() are
        completely unchanged and still use plain first-match behavior.
        Disable either step to isolate its individual effect.

        Returns the normal predict_detailed()-shaped dict, plus:
          "working_smiles": SMILES actually used (may differ from
                             input if tautomer-corrected)
          "tautomer_applied": bool
          "tautomer_info": dict from rank_tautomers_safe, or None
          "site_disambiguation_applied": bool
          "site_info": dict from rank_same_type_sites, or None
        """
        working_smiles = smiles
        tautomer_info = None
        tautomer_applied = False
        if use_tautomer_correction:
            try:
                tautomer_info = self.rank_tautomers_safe(smiles)
                if tautomer_info["changed"] and tautomer_info["safe_to_apply"]:
                    working_smiles = tautomer_info["canonical_smiles"]
                    tautomer_applied = True
            except Exception:
                pass

        site_info = None
        site_disambiguation_applied = False
        chosen_site_idx = None
        if use_site_disambiguation:
            try:
                site_info = self.rank_same_type_sites(working_smiles)
                if site_info.get("has_ambiguity"):
                    all_sites = self.sites(working_smiles)
                    if all_sites:
                        default_site = all_sites[0]
                        group = default_site["group"]
                        if group in site_info["results"]:
                            best = site_info["results"][group][0]
                            if best["index"] != default_site["index"]:
                                site_disambiguation_applied = True
                                chosen_site_idx = best["index"]
            except Exception:
                pass

        if site_disambiguation_applied:
            base_pka = self.predict_site(
                working_smiles, chosen_site_idx, solvent=solvent,
                salt=salt, salt_concentration=salt_concentration)
            from . import solvents as _solvents
            info = _solvents.resolve_solvent(solvent)
            result = {"pKa": base_pka, "base_pKa": base_pka,
                      "solvent": info.name,
                      "correction": {"shift": 0.0, "tier": "n/a",
                                     "note": "see predict_site for salt handling"}}
        else:
            result = self.predict_detailed(
                working_smiles, solvent=solvent, salt=salt,
                salt_concentration=salt_concentration)

        result["working_smiles"] = working_smiles
        result["tautomer_applied"] = tautomer_applied
        result["tautomer_info"] = tautomer_info
        result["site_disambiguation_applied"] = site_disambiguation_applied
        result["site_info"] = site_info
        return result

    def sites(self, smiles: str) -> list[dict]:
        """List titratable sites, each with an index for ``predict_site``."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"could not parse SMILES: {smiles}")
        mol = neutralize(mol)
        found, seen = [], set()
        for kind, table in (("acid", ACID_SITES), ("base", BASE_SITES)):
            for group, smarts, ai in table:
                patt = Chem.MolFromSmarts(smarts)
                if patt is None:
                    continue
                for match in mol.GetSubstructMatches(patt):
                    idx = match[ai]
                    if idx in seen:
                        continue
                    seen.add(idx)
                    found.append({
                        "index": len(found), "atom": idx, "group": group,
                        "kind": kind,
                        "element": mol.GetAtomWithIdx(idx).GetSymbol(),
                    })
        return found

    def rank_same_type_sites(self, smiles: str) -> dict:
        """When a molecule has MULTIPLE candidate sites of the SAME
        chemical group (e.g., two chemically inequivalent phenol -OH
        groups), rank them by raw UMA energy of the resulting ionized
        structure - which specific site is most energetically
        favorable to ionize.

        WHY THIS IS LEGITIMATE despite Section III showing UMA's
        charged-species energies are unreliable for ABSOLUTE free
        energies: that finding was about comparing DIFFERENT charge
        states (neutral vs. ion) to get an absolute pKa - UMA's
        charged-species error (200-500 meV) dwarfs the pKa signal
        itself there. This compares structures at the SAME net charge
        (e.g., several different singly-charged anions of the same
        molecule) - a RELATIVE comparison, the same category as
        rank_tautomers (also same-charge-state), which was validated
        directly on real cases. Systematic charged-species error is
        expected to be largely common across closely related, same-
        charge structures and cancel out in a relative comparison the
        way it does not in an absolute one.

        Does NOT change which site predict()/predict_site() use by
        default (still first-SMARTS-match priority order, for
        backward compatibility) - this is a diagnostic/opt-in tool,
        same pattern as rank_tautomers.

        Returns {"has_ambiguity": bool,
                 "results": {group_name: [site_dict_with_energy, ...]
                             sorted lowest-energy first, ...}}
        Only groups with more than one candidate are included.
        """
        all_sites = self.sites(smiles)
        from collections import defaultdict
        by_group = defaultdict(list)
        for s in all_sites:
            by_group[s["group"]].append(s)
        multi_groups = {g: c for g, c in by_group.items() if len(c) > 1}
        if not multi_groups:
            return {"has_ambiguity": False, "results": {}}

        mol = neutralize(Chem.MolFromSmiles(smiles))
        results = {}
        for group, candidates in multi_groups.items():
            scored = []
            for c in candidates:
                if c["kind"] == "acid":
                    shifted_smi, _ = _shift_hydrogen_tagged(mol, c["atom"], -1, -1)
                else:
                    shifted_smi, _ = _shift_hydrogen_tagged(mol, c["atom"], +1, +1)
                if shifted_smi is None:
                    continue
                try:
                    atoms = _smiles_to_atoms(shifted_smi)
                    energy = self.get_energy(atoms)
                    scored.append({**c, "ionized_smiles": shifted_smi,
                                    "energy_eV": energy})
                except Exception:
                    continue
            scored.sort(key=lambda x: x["energy_eV"])
            results[group] = scored
        return {"has_ambiguity": True, "results": results}

    def predict_site(self, smiles: str, site_index: int,
                      solvent: str = "water",
                      salt: str | None = None,
                      salt_concentration: float | None = None) -> float:
        """Predict pKa for one chosen site, other sites left neutral,
        in `solvent` (see ``predict`` for supported solvents and what
        ``salt``/``salt_concentration`` do).
        """
        from . import solvents as _solvents
        info = _solvents.resolve_solvent(solvent)
        mol = neutralize(Chem.MolFromSmiles(smiles))
        sites = self.sites(smiles)
        if site_index >= len(sites):
            raise IndexError(f"molecule has only {len(sites)} sites")
        site = sites[site_index]
        if site["kind"] == "acid":
            other_smi, other_idx = _shift_hydrogen_tagged(mol, site["atom"], -1, -1)
            this_smi, this_idx = _tag_and_reparse(mol, site["atom"])
            prot_smi, prot_idx = this_smi, this_idx
            deprot_smi, deprot_idx = other_smi, other_idx
        else:
            other_smi, other_idx = _shift_hydrogen_tagged(mol, site["atom"], +1, +1)
            this_smi, this_idx = _tag_and_reparse(mol, site["atom"])
            prot_smi, prot_idx = other_smi, other_idx
            deprot_smi, deprot_idx = this_smi, this_idx
        if prot_smi is None or deprot_smi is None:
            raise RuntimeError("could not construct the protonation pair")
        pair_feat = self.features(prot_smi, deprot_smi, prot_idx, deprot_idx)
        base = self._base_pka(pair_feat, solvent)
        if salt is None:
            return base
        if info.solvation_key is None:
            raise ValueError(
                f"no ionic-strength model available for {info.name} - see "
                f"umapka.solvents.SOLVENTS for which solvents support it"
            )
        import importlib
        solvation = importlib.import_module(".solvation", __package__)
        correction = solvation.predict_salt_correction(
            site["kind"], salt, salt_concentration, solvent=info.solvation_key)
        return base + correction["shift"]

    def predict_all_sites(self, smiles: str, solvent: str = "water",
                           salt: str | None = None,
                           salt_concentration: float | None = None) -> list[dict]:
        """Score EVERY detected titratable site, not just the first one
        found by SMARTS priority order (which is what ``predict()`` uses).

        Borrowed from pKalculator's strategy: since umapka's regressor
        can already evaluate any single site via ``predict_site()``, and
        ``sites()`` already enumerates every candidate, there's no reason
        to only ever look at the first match. Useful for molecules with
        multiple plausible acid/base sites where you want to know which
        one is actually most acidic/basic, not just which one the
        priority table happened to find first.

        Returns a list of site dicts (same shape as ``sites()``) each
        with an added "pKa" key, sorted by pKa ascending. Note this does
        NOT solve sequential multi-deprotonation (that's a different,
        harder problem - see ``predict_detailed`` discussion and the
        free-energy-difference approach); this only ranks INDEPENDENT
        single-site predictions for one fixed molecule, each computed
        with all other sites left neutral.
        """
        sites = self.sites(smiles)
        results = []
        for site in sites:
            try:
                pka = self.predict_site(smiles, site["index"], solvent=solvent,
                                         salt=salt,
                                         salt_concentration=salt_concentration)
                results.append({**site, "pKa": pka})
            except Exception as e:
                results.append({**site, "pKa": None, "error": str(e)})
        scored = [r for r in results if r["pKa"] is not None]
        unscored = [r for r in results if r["pKa"] is None]
        return sorted(scored, key=lambda r: r["pKa"]) + unscored

    _PROTONATED_BASE_PATTERN = Chem.MolFromSmarts("[#7+;H1,H2,H3]")

    def _sites_on_current_mol(self, mol) -> list[tuple[str, int]]:
        """Like sites(), but works DIRECTLY on the given mol object
        without calling neutralize() first, AND includes protonated
        base sites (ammonium-type: N+ with a remaining H) as candidate
        removable-proton sites, not just ACID_SITES.

        Why: once a base site (amine, guanidine, pyridine, ...) has
        been protonated - either by _build_fully_protonated() at the
        start of a chain, or in principle at any point - it becomes,
        chemically, just another site that can lose a proton. There is
        no meaningful difference at that point between "a carboxylic
        acid losing H+" and "an ammonium losing H+"; both are acid
        dissociations. The generic pattern [#7+;H1,H2,H3] catches
        protonated primary/secondary/tertiary amines, anilinium,
        pyridinium, protonated guanidinium/amidinium, etc. without
        needing to track which specific atom got protonated earlier.

        Also does NOT call neutralize() first, for the same reason as
        before: after removing one proton, the resulting anion (or,
        now, after removing a proton from an ammonium, the resulting
        neutral amine) has a real, intentional charge state that
        neutralize()'s cleanup SMARTS would otherwise undo.
        """
        found = []
        for group, smarts, ai in ACID_SITES:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            for match in mol.GetSubstructMatches(patt):
                found.append((group, match[ai]))
        if self._PROTONATED_BASE_PATTERN is not None:
            for match in mol.GetSubstructMatches(self._PROTONATED_BASE_PATTERN):
                found.append(("protonated_base", match[0]))
        return found

    def _build_fully_protonated(self, mol):
        """From a NEUTRAL mol (as neutralize() returns), protonate every
        detected BASE_SITES match to reach the fully-protonated
        macrostate - the correct starting point for a complete
        deprotonation chain covering both acid AND base sites.

        Acid sites are already at their fully-protonated H-count by
        convention (neutralize() gives COOH, not COO-, an amine as NH2
        not NH3+), so only base sites need this extra step. Without it,
        a molecule's own amine/guanidine/etc. ionization would be
        silently skipped entirely - which is exactly the gap found when
        testing this method on amino-diacid molecules.
        """
        current = mol
        for _ in range(10):  # generous cap; molecules rarely have >10 base sites
            found_idx = None
            for group, smarts, ai in BASE_SITES:
                patt = Chem.MolFromSmarts(smarts)
                if patt is None:
                    continue
                matches = current.GetSubstructMatches(patt)
                if matches:
                    found_idx = matches[0][ai]
                    break
            if found_idx is None:
                break
            protonated = _shift_hydrogen(current, found_idx, +1, +1)
            if protonated is None:
                break
            current = protonated
        return current

    def _predict_free_energy(self, feat: np.ndarray) -> float:
        """Predict pKa from a feature row, transparently supporting
        either a single regressor or a list of regressors (ensemble -
        averages predictions). free_energy_ensemble.pkl (validated:
        1.454 MAE, 96.1%/90.5% ordering for 2/3-step chains) is a list
        of 5 models; older single-model files still work unchanged.
        """
        model = self._free_energy_model
        if isinstance(model, (list, tuple)):
            return float(np.mean([float(m.predict(feat)[0]) for m in model]))
        return float(model.predict(feat)[0])

    def predict_chain(self, smiles: str, max_steps: int = 4) -> dict:
        """Predict a genuine SEQUENCE of deprotonations across BOTH acid
        and base sites - find the most acidic removable proton, remove
        it, find the next most acidic on the resulting structure, and
        so on. Starts from the fully-protonated macrostate (every
        detected base site protonated, e.g. an amine as NH3+), not the
        plain neutral molecule - otherwise a molecule's own amine/
        guanidine/etc. ionization would be silently skipped, which is
        exactly the gap found when this method was first tested on
        amino-diacid molecules (it only walked ACID_SITES).

        This is a different problem from predict_all_sites(), which
        ranks independent single-site predictions on the SAME starting
        molecule with every other site left neutral; predict_chain()
        actually walks the ionization path.

        Requires a free-energy-difference model loaded via
        PkaPredictor(..., free_energy_model_path=...) - see
        train_free_energy_model.py. Raises if none was loaded.

        IMPORTANT - read before trusting results:
          - This code has been run-tested end-to-end (base version) and
            the underlying model has been validated with a tuned,
            feature-enriched, 5-model ensemble (free_energy_ensemble.pkl):
            scaffold-split MAE 1.454, single-site subset 1.347. Don't
            use this for single-site molecules; use predict() instead
            (0.994 MAE, still meaningfully better for that case).
          - Ordering accuracy on held-out chains: 96.1% for 2-step
            (173/180), 90.5% for 3-step (19/21) - a large improvement
            over earlier versions (88.9%/61.9% baseline,
            83.9%/61.9% with a labeling bug found and fixed along the
            way). Achieved via hyperparameter tuning, an explicit
            step-number feature, and 5-model ensembling - NOT via a
            more complex architecture, which was tried and made things
            worse (a small neural net scored 1.632 MAE, 78.3%/66.7% -
            classic small-data regime where gradient-boosted trees
            beat neural nets).
          - Known failure modes, from direct inspection of held-out
            mis-ordered chains: near-degenerate true pKa gaps (<0.5
            apart - often not meaningful orderings to begin with),
            triprotic amino-diacids, and fused heteroaromatic systems
            with coupled tautomers.
          - This method does NOT sort the output to force monotonic
            ordering. Check the "monotonic" key - if False, treat the
            chain's ordering as unreliable rather than trusting it.
          - The starting state is the fully-protonated macrostate,
            not plain neutralize() output - the FIRST entry in
            "states" may therefore carry a positive charge (e.g. an
            amino acid's ammonium/diacid form), which is intentional.
          - No phosphonic/phosphoric acid site coverage - molecules
            with only this chemistry will return an empty chain.

        Returns {"states": [SMILES, ...], "pKas": [float, ...],
                 "monotonic": bool, "warning": str | None}
        """
        if self._free_energy_model is None:
            raise RuntimeError(
                "predict_chain() needs a free-energy-difference model. "
                "Construct PkaPredictor(..., free_energy_model_path="
                "'free_energy_model_scaffold.pkl') (or similar) - this "
                "is a SEPARATE file from the main model_core.pkl."
            )

        mol = neutralize(Chem.MolFromSmiles(smiles))
        if mol is None:
            raise ValueError(f"could not parse SMILES: {smiles}")
        mol = self._build_fully_protonated(mol)

        states_smiles = [Chem.MolToSmiles(mol)]
        pkas = []
        current_mol = mol
        # single-state embedding of the starting structure
        h_current = self.pool(self.embeddings(_smiles_to_atoms(Chem.MolToSmiles(current_mol))))

        for _ in range(max_steps):
            candidates = self._sites_on_current_mol(current_mol)
            if not candidates:
                break

            step_number = len(pkas) + 1  # 1-indexed, matches training exactly
            best = None  # (pka, deprot_mol, h_deprot)
            for group, atom_idx in candidates:
                deprot = _shift_hydrogen(current_mol, atom_idx, -1, -1)
                if deprot is None:
                    continue
                h_deprot = self.pool(self.embeddings(
                    _smiles_to_atoms(Chem.MolToSmiles(deprot))))
                # SAME feature convention as train_tuned_ensemble.py:
                # [h_before, h_after, h_after - h_before, step_number] -
                # note this is the OPPOSITE order from features()'s
                # [h_prot, h_deprot, h_prot-h_deprot]. Getting this
                # backwards would silently feed the model mis-signed
                # input.
                feat = np.concatenate(
                    [h_current, h_deprot, h_deprot - h_current, [step_number]]
                ).reshape(1, -1)
                pka = self._predict_free_energy(feat)
                if best is None or pka < best[0]:
                    best = (pka, deprot, h_deprot)

            if best is None:
                break
            pka, current_mol, h_current = best
            pkas.append(pka)
            states_smiles.append(Chem.MolToSmiles(current_mol))

        monotonic = all(pkas[i] <= pkas[i + 1] for i in range(len(pkas) - 1))
        return {
            "states": states_smiles,
            "pKas": pkas,
            "monotonic": monotonic,
            "warning": None if monotonic else (
                "predicted pKa sequence is not monotonically increasing "
                "- this chain may involve a hard chemotype (see "
                "predict_chain docstring); do not trust the ordering"
            ),
        }
