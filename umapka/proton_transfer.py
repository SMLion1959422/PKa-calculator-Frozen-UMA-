"""
umapka.proton_transfer - Gibbs energies of transfer of the proton from
water to organic solvents.

The thermodynamic cycle for an acid HA in solvent S, relative to water:

    pKa(S) - pKa(W) = [dG_tr(A-) - dG_tr(HA) + dG_tr(H+)] / (RT ln10)

The first two terms depend on the solute; dG_tr(H+) is ONE CONSTANT PER
SOLVENT. dev/proton_offset_test.py tests whether the per-solvent offset
fitted from measured pKa data is that constant.

Values below are standard molar Gibbs energies of transfer of the proton
from water at 298 K, on the TATB (tetraphenylarsonium tetraphenylborate)
extrathermodynamic assumption, which is the usual basis for single-ion
transfer energies.

ACCURACY WARNING. Single-ion transfer energies are not measurable
without an extrathermodynamic assumption; compilations disagree by
several kJ/mol, and by more for less-studied solvents. Each entry
carries a `verified` flag:

  verified=True   value cross-checked against a citable compilation
                  (Kalidas, Hefter & Marcus, Chem. Rev. 2000, 100, 819,
                  as reported in secondary sources)
  verified=False  commonly quoted value entered from general knowledge
                  and NOT re-checked against the primary source; treat
                  as indicative only and re-verify before publication

Anyone using this for a paper should replace the whole table with values
read from a primary compilation, and say which one. A modern
reassessment is Stroh et al., ChemPhysChem 2026, doi
10.1002/cphc.202500349 ("Proton Solvation in Water and Selected Organic
Solvents"), which was not reachable from the environment this file was
written in.
"""

from __future__ import annotations
from dataclasses import dataclass

__all__ = ["ProtonTransfer", "DG_TR_PROTON", "pk_units", "PK_PER_KCAL"]

PK_PER_KCAL = 1.0 / 1.3637      # 1 pK unit = RT ln10 = 1.3637 kcal/mol at 298 K


@dataclass(frozen=True)
class ProtonTransfer:
    solvent: str
    kcal_per_mol: float     # dG_tr(H+), water -> solvent, 298 K, TATB
    verified: bool
    note: str = ""


DG_TR_PROTON: dict[str, ProtonTransfer] = {
    "Water":        ProtonTransfer("Water", 0.0, True, "reference state, exact by definition"),
    "Methanol":     ProtonTransfer("Methanol", 2.1, True, "Kalidas/Hefter/Marcus"),
    "Acetonitrile": ProtonTransfer("Acetonitrile", 10.7, True,
                                   "Kalidas/Hefter/Marcus; the large positive value is why "
                                   "MeCN pKa scales sit ~8 pK units above DMSO"),
    "DMSO":         ProtonTransfer("DMSO", -4.6, True, "Kalidas/Hefter/Marcus"),
    "DMF":          ProtonTransfer("DMF", -4.4, False, "commonly quoted ~ -18 kJ/mol; NOT re-verified"),
    "Ethanol":      ProtonTransfer("Ethanol", 2.6, False, "commonly quoted ~ +11 kJ/mol; NOT re-verified"),
    "NMP":          ProtonTransfer("NMP", -4.8, False, "amide solvent, assumed close to DMF; NOT re-verified"),
    # Ethylene glycol: no value entered rather than a guess.
}


def pk_units(solvent: str) -> float | None:
    """dG_tr(H+) expressed in pKa units (kcal/mol / 1.3637)."""
    e = DG_TR_PROTON.get(solvent)
    return None if e is None else e.kcal_per_mol * PK_PER_KCAL
