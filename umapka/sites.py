"""
umapka.sites - titratable-site definitions and site detection.

RDKit-only (no UMA / ASE / torch), so the thermodynamic layer in
``umapka.microstates`` can be used and tested without a GPU or model
weights. ``umapka.predictor`` re-exports ACID_SITES, BASE_SITES and
``neutralize`` from here, so existing imports keep working.

Each table entry is (group, SMARTS, site_atom_index_in_match).
``CHARGE_SHARE`` says over which atoms of the same match the charge
of the *ionized* form is spread (e.g. a carboxylate's -1 is shared by
both oxygens). This matters for the site-site electrostatic coupling
in ``umapka.microstates``: putting a carboxylate's whole charge on the
single former-OH oxygen over-estimates short-range interactions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from rdkit import Chem

__all__ = ["ACID_SITES", "BASE_SITES", "CHARGE_SHARE", "Site",
           "neutralize", "find_sites"]


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
    ("aniline",    "[NX3;H2]-a", 0),
    ("aniline_sec", "[NX3;H1]-a", 0),
    ("aniline_tert","[NX3;H0]-a", 0),
]

# match positions carrying the ionized form's charge (equal shares).
# Groups not listed keep the whole charge on the site atom.
CHARGE_SHARE = {
    "carboxylic_acid": (1, 2),
    "sulfonic_acid":   (1, 2, 3),
    "phosphoric_acid": (1, 2),
    "tetrazole":       (1, 2, 3, 4),
    "tetrazole_2":     (1, 2, 3, 4),
    "imide":           (1, 4),
    "guanidine":       (0, 2, 3),
    "amidine":         (0, 2),
}

_NEUTRALIZE_PATTERN = Chem.MolFromSmarts(
    "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
)


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


@dataclass
class Site:
    """One titratable site of a neutral molecule.

    ``kind`` is "acid" (neutral HA -> A-, charge 0 -> -1) or "base"
    (BH+ -> B, charge +1 -> 0). ``charge_atoms`` are (atom index,
    share) pairs over which the ionized form's unit charge is spread.
    """
    index: int
    atom: int
    group: str
    kind: str
    element: str
    charge_atoms: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"index": self.index, "atom": self.atom, "group": self.group,
                "kind": self.kind, "element": self.element}


def find_sites(mol: Chem.Mol) -> list[Site]:
    """Every titratable site in a (neutralized) molecule, deduplicated
    by site atom, acids first then bases, each in table priority order.
    Same enumeration as ``PkaPredictor.sites``.
    """
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
                share = CHARGE_SHARE.get(group)
                atoms = [match[k] for k in share] if share else [idx]
                w = 1.0 / len(atoms)
                found.append(Site(
                    index=len(found), atom=idx, group=group, kind=kind,
                    element=mol.GetAtomWithIdx(idx).GetSymbol(),
                    charge_atoms=[(a, w) for a in atoms],
                ))
    return found
