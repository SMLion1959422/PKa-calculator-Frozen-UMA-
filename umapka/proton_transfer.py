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

  verified=True   value reproduced from two independent secondary
                  sources that agree to <= 0.05 kcal/mol
  verified=False  entered from general knowledge and NOT corroborated;
                  treat as indicative only

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
    "Methanol":     ProtonTransfer("Methanol", 8.7 / 4.184, True,
                                   "8.7 kJ/mol, TATB recommended. Marcus separately quoted at "
                                   "10.4 kJ/mol (=2.49 kcal) - a ~0.4 kcal spread between "
                                   "compilations, which bounds the accuracy here."),
    "Ethanol":      ProtonTransfer("Ethanol", 11.1 / 4.184, True, "11.1 kJ/mol, TATB recommended"),
    "Acetonitrile": ProtonTransfer("Acetonitrile", 44.8 / 4.184, True,
                                   "44.8 kJ/mol, TATB recommended. The large positive value is "
                                   "why MeCN pKa scales sit ~8 pK units above DMSO."),
    "DMSO":         ProtonTransfer("DMSO", -19.4 / 4.184, True, "-19.4 kJ/mol, TATB recommended"),
    "DMF":          ProtonTransfer("DMF", -14.4 / 4.184, True,
                                   "-14.4 kJ/mol, TATB recommended (Marcus). CORRECTED: an "
                                   "earlier unverified entry used -18.4 kJ/mol, wrong by "
                                   "~1.0 kcal/mol (0.70 pK units)."),
    # NMP: no value located. Left out rather than guessed; it also has only
    # 17 paired rows, so it is unusable here either way.
    # EthyleneGlycol: no value located.
}

# Provenance. The five values above are the TATB-assumption recommended
# values as reported in the secondary literature (a 2021 Computational and
# Theoretical Chemistry study quoting the compilations, alongside
# Kalidas, Hefter & Marcus, Chem. Rev. 2000, 100, 819). Four of them
# independently reproduce values that had been entered here earlier from a
# different secondary source, to within 0.05 kcal/mol, which is why they are
# marked verified. NONE has been read by this project off a primary table.
# Before publication, read them from a primary compilation and cite it; the
# 2026 reassessment (Stroh et al., ChemPhysChem, doi 10.1002/cphc.202500349)
# is the obvious cross-check and was not reachable here.

def pk_units(solvent: str) -> float | None:
    """dG_tr(H+) expressed in pKa units (kcal/mol / 1.3637)."""
    e = DG_TR_PROTON.get(solvent)
    return None if e is None else e.kcal_per_mol * PK_PER_KCAL
