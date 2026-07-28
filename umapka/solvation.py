"""
umapka.solvation - salt/ionic-strength and solvent-dielectric corrections
for apparent pKa.

This module is deliberately independent of UMA, rdkit-embedding features,
and the trained regressor: it applies classical electrolyte theory as a
POST-HOC correction on top of a base pKa prediction (whatever solvent
that base prediction came from). It answers a different question than
`PkaPredictor`: not "what is the intrinsic pKa in this solvent" but
"how does the intrinsic pKa shift when I add this salt at this
concentration."

Honesty about scope (read before trusting numbers):

  - Tier 1 (Debye-Hueckel / Davies): general, but only rigorously valid
    for dilute aqueous solutions, roughly I <~ 0.1-0.5 mol/L. It only
    depends on ionic strength and ion charge, NOT on which specific
    salt you used - i.e. NaCl and KCl at the same concentration give
    the same correction in this tier. If you need salt-specific
    behaviour, that's Tier 3, and it usually isn't available.
  - Tier 2 (solvent dielectric + Bjerrum/Fuoss ion pairing): extends
    Tier 1 into non-aqueous solvents, where simple Debye-Hueckel
    theory becomes QUALITATIVELY wrong because ions associate into
    neutral pairs well before I = 0.1 M. This is an approximate,
    electrostatics-only correction - it captures the generic physics
    of ion pairing, not specific salt chemistry (complexation,
    Hofmeister effects, etc).
  - Tier 3 (Pitzer): the accurate option for concentrated solutions,
    but requires empirically fit interaction parameters PER SALT.
    These exist in the literature for maybe a few dozen well-studied
    salts in WATER. They essentially do not exist for organic
    solvents. The `PITZER_PARAMS` table below is a small, clearly
    labelled starting point, not a comprehensive database - verify
    any entry against its cited primary source before relying on it.
  - Complexing/chelating salts (e.g. many transition-metal or
    lanthanide salts binding directly to the analyte) are OUT OF
    SCOPE entirely: that's specific binding equilibria, not ionic
    atmosphere screening, and needs binding constants this module
    does not attempt to supply.

None of the numbers below are fit to any experimental pKa shift
dataset. They are textbook physical-chemistry constants and formulas
applied directly - there is no ML model here, and none is needed for
this particular sub-problem.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

__all__ = [
    "Ion", "SolventProperties", "SOLVENTS", "IONS", "SALTS",
    "ionic_strength", "debye_huckel_A", "debye_huckel_B",
    "davies_log_gamma", "pka_shift_debye_huckel",
    "bjerrum_length", "ion_pair_fraction", "pka_shift_ion_pairing",
    "PitzerParams", "PITZER_PARAMS", "pka_shift_pitzer",
    "predict_salt_correction",
]

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------
_E = 1.602176634e-19        # elementary charge, C
_EPS0 = 8.8541878128e-12    # vacuum permittivity, F/m
_KB = 1.380649e-23          # Boltzmann constant, J/K
_NA = 6.02214076e23         # Avogadro's number
_LN10 = math.log(10)


# ---------------------------------------------------------------------
# solvent properties
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class SolventProperties:
    name: str
    dielectric_constant: float   # relative permittivity, ~25 C, unitless
    # source: standard physical-chemistry reference values (CRC-type
    # compilations). Re-verify before use in anything precision-critical -
    # dielectric constants are somewhat temperature- and
    # measurement-method-sensitive, especially for less common solvents.


SOLVENTS: dict[str, SolventProperties] = {
    "water":        SolventProperties("water", 78.4),
    "dmso":         SolventProperties("dmso", 46.7),
    "acetonitrile": SolventProperties("acetonitrile", 37.5),
    "methanol":     SolventProperties("methanol", 32.7),
    "ethanol":      SolventProperties("ethanol", 24.6),
    "dmf":          SolventProperties("dmf", 36.7),
    "acetone":      SolventProperties("acetone", 20.7),
    "thf":          SolventProperties("thf", 7.6),
    "dcm":          SolventProperties("dcm", 8.9),
}


# ---------------------------------------------------------------------
# ion registry
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Ion:
    symbol: str
    charge: int
    kielland_radius_pm: float
    # "effective hydrated radius" used in the extended Debye-Hueckel
    # equation's B-parameter term. Values follow the classic Kielland
    # (1937) ion-size-parameter table commonly reproduced in analytical
    # chemistry texts. Treat as approximate - re-check specific values
    # against a primary compilation before precision-critical use.


IONS: dict[str, Ion] = {
    "Na+":  Ion("Na+", +1, 400),
    "K+":   Ion("K+", +1, 300),
    "Li+":  Ion("Li+", +1, 600),
    "NH4+": Ion("NH4+", +1, 250),
    "Ca2+": Ion("Ca2+", +2, 600),
    "Mg2+": Ion("Mg2+", +2, 800),
    "Ba2+": Ion("Ba2+", +2, 500),
    "Cl-":  Ion("Cl-", -1, 300),
    "Br-":  Ion("Br-", -1, 300),
    "I-":   Ion("I-", -1, 300),
    "NO3-": Ion("NO3-", -1, 300),
    "SO4-2":Ion("SO4-2", -2, 400),
    "PO4-3":Ion("PO4-3", -3, 400),
    "OAc-": Ion("OAc-", -1, 425),   # acetate
}


# ---------------------------------------------------------------------
# salt registry - a CURATED lookup, not a general formula parser.
#
# Parsing an arbitrary typed formula ("Na2SO4", "CaCl2", ...) into
# constituent ions unambiguously is itself nontrivial (polyatomic ions,
# hydrates, etc.), so this deliberately supports a known list rather
# than guessing. Unrecognized input should fail loudly and say so,
# not silently guess wrong.
# ---------------------------------------------------------------------
SALTS: dict[str, dict[str, int]] = {
    # salt formula (as the user would type it) -> {ion_symbol: stoichiometry}
    "NaCl":   {"Na+": 1, "Cl-": 1},
    "KCl":    {"K+": 1, "Cl-": 1},
    "LiCl":   {"Li+": 1, "Cl-": 1},
    "NaBr":   {"Na+": 1, "Br-": 1},
    "KBr":    {"K+": 1, "Br-": 1},
    "NaI":    {"Na+": 1, "I-": 1},
    "NH4Cl":  {"NH4+": 1, "Cl-": 1},
    "NaNO3":  {"Na+": 1, "NO3-": 1},
    "KNO3":   {"K+": 1, "NO3-": 1},
    "NaOAc":  {"Na+": 1, "OAc-": 1},
    "Na2SO4": {"Na+": 2, "SO4-2": 1},
    "K2SO4":  {"K+": 2, "SO4-2": 1},
    "MgSO4":  {"Mg2+": 1, "SO4-2": 1},
    "CaCl2":  {"Ca2+": 1, "Cl-": 2},
    "MgCl2":  {"Mg2+": 1, "Cl-": 2},
    "BaCl2":  {"Ba2+": 1, "Cl-": 2},
    "Na3PO4": {"Na+": 3, "PO4-3": 1},
}


def _resolve_salt(salt: str) -> dict[str, int]:
    if salt not in SALTS:
        raise ValueError(
            f"salt '{salt}' is not in the curated registry. Known salts: "
            f"{sorted(SALTS)}. Add it to SALTS (with correct ion "
            f"stoichiometry) rather than guessing a parse."
        )
    return SALTS[salt]


# ---------------------------------------------------------------------
# Tier 1: ionic strength + Debye-Hueckel / Davies
# ---------------------------------------------------------------------
def ionic_strength(salt: str, concentration_M: float) -> float:
    """Ionic strength I = 0.5 * sum(c_i * z_i^2) for a fully-dissociated
    salt at the given total concentration (mol/L). Does not itself
    account for ion pairing - see `ion_pair_fraction` for that
    correction, which effectively reduces the free-ion concentration
    entering this formula.
    """
    if concentration_M < 0:
        raise ValueError("concentration must be >= 0")
    ions = _resolve_salt(salt)
    total = 0.0
    for sym, stoich in ions.items():
        z = IONS[sym].charge
        c_i = stoich * concentration_M
        total += c_i * z * z
    return 0.5 * total


def debye_huckel_A(epsilon_r: float, T_K: float = 298.15) -> float:
    """Debye-Hueckel A constant (base-10 log form), solvent- and
    temperature-dependent. Standard form:
        A = 1.8246e6 / (epsilon_r * T)^1.5      [units: (mol/L)^-0.5]
    Reduces to the familiar A ~ 0.509 (mol/L)^-0.5 for water at 25 C.
    """
    return 1.8246e6 / (epsilon_r * T_K) ** 1.5


def debye_huckel_B(epsilon_r: float, T_K: float = 298.15) -> float:
    """Debye-Hueckel B constant for the extended (Kielland-radius) term.
    Standard form: B = 50.29 / sqrt(epsilon_r * T)   [units: pm^-1 * (mol/L)^-0.5,
    consistent with kielland_radius_pm above]. Reduces to B ~ 0.328 A^-1
    (mol/L)^-0.5 for water at 25 C in the traditional Angstrom form.
    """
    return 50.29 / math.sqrt(epsilon_r * T_K)


def davies_log_gamma(z: int, I: float, A: float) -> float:
    """log10(activity coefficient) via the Davies equation - an
    extended Debye-Hueckel form valid further into moderate ionic
    strength (roughly I <~ 0.5 M) than the bare limiting law, at the
    cost of an empirical linear term with no strong theoretical
    justification beyond "it fits data reasonably well".
        log10(gamma) = -A z^2 [ sqrt(I)/(1+sqrt(I)) - 0.3 I ]
    """
    sqrtI = math.sqrt(I)
    return -A * z * z * (sqrtI / (1 + sqrtI) - 0.3 * I)


def pka_shift_debye_huckel(site_kind: str, I: float,
                            solvent: str = "water",
                            T_K: float = 298.15) -> float:
    """Apparent pKa shift from ionic strength alone (Tier 1/Davies).

    site_kind: "acid" (neutral acid -> anion + H+, e.g. carboxylic
        acid, phenol, thiol - most of umapka's ACID_SITES) or "base"
        (cationic acid -> neutral base + H+, e.g. protonated amine -
        most of umapka's BASE_SITES after protonation).

    Neutral acids show a MUCH larger ionic-strength dependence than
    cationic acids/ammonium-type bases, because the charge-squared
    change on dissociation is larger (0 -> 1,1 vs 1 -> 0,1). This
    distinction is exactly the acid/base "kind" umapka's own
    PkaPredictor.sites() already tags each site with.
    """
    if I < 0:
        raise ValueError("ionic strength must be >= 0")
    props = SOLVENTS.get(solvent)
    if props is None:
        raise ValueError(f"unknown solvent '{solvent}'. Known: {sorted(SOLVENTS)}")
    A = debye_huckel_A(props.dielectric_constant, T_K)

    # pKa = -log10(Ka); Ka activity ratio involves log(gamma_H+) +
    # log(gamma_conjugate) - log(gamma_acid). For a neutral acid HA:
    # z(H+)=1, z(A-)=1, z(HA)=0 -> the correction below.
    # For a cationic acid BH+: z(H+)=1, z(B)=0, z(BH+)=1 -> the charge
    # squared terms mostly cancel, leaving a much smaller shift.
    if site_kind == "acid":
        # neutral HA -> H+ + A-
        d_log_gamma = davies_log_gamma(1, I, A) + davies_log_gamma(1, I, A) \
                      - davies_log_gamma(0, I, A)
    elif site_kind == "base":
        # BH+ -> H+ + B (neutral)
        d_log_gamma = davies_log_gamma(1, I, A) + davies_log_gamma(0, I, A) \
                      - davies_log_gamma(1, I, A)
    else:
        raise ValueError("site_kind must be 'acid' or 'base'")

    # pKa_apparent = pKa_thermodynamic - d_log_gamma  (sign convention:
    # stabilizing the products relative to reactant lowers apparent pKa
    # for an acid dissociation)
    return -d_log_gamma


# ---------------------------------------------------------------------
# Tier 2: Bjerrum length + Fuoss-style ion-pairing correction
# ---------------------------------------------------------------------
def bjerrum_length(epsilon_r: float, T_K: float = 298.15) -> float:
    """Bjerrum length (m): the separation at which the electrostatic
    interaction energy between two unit charges equals kT. Large
    Bjerrum length (low-dielectric solvent) means electrostatic
    attraction dominates thermal motion over a long range, i.e. ions
    associate into pairs readily. lambda_B = e^2 / (4 pi eps0 epsilon_r kB T)
    """
    return (_E ** 2) / (4 * math.pi * _EPS0 * epsilon_r * _KB * T_K)


def ion_pair_fraction(z1: int, z2: int, ion_size_m: float,
                       concentration_M: float, epsilon_r: float,
                       T_K: float = 298.15) -> float:
    """Rough estimate of the fraction of ion pairs formed, via a
    Fuoss-style association constant built from the Bjerrum length.
    This is an ELECTROSTATICS-ONLY estimate (no specific chemistry) -
    treat it as indicating whether ion pairing is a first-order or
    negligible effect for a given solvent/ion combination, not as a
    precise number. Returns a value in [0, 1).
    """
    lB = bjerrum_length(epsilon_r, T_K)
    # Fuoss association constant (L/mol), simplified form:
    # K_A = (4 pi N_A a^3 / 3000) * exp(|z1 z2| * lB / a)
    a = ion_size_m
    exponent = abs(z1 * z2) * lB / a
    exponent = min(exponent, 50)  # guard against overflow for extreme cases
    K_A = (4 * math.pi * _NA * a ** 3 / 3000) * math.exp(exponent)
    # fraction paired, from mass-action K_A = [pair] / ([M+][X-]),
    # solved for a 1:1 approximation at total concentration c
    c = concentration_M
    # x = fraction paired solves K_A * c * (1-x)^2 = x  (1:1 salt approx)
    # -> quadratic in x
    a_q = K_A * c
    b_q = -(2 * K_A * c + 1)
    c_q = K_A * c
    disc = b_q ** 2 - 4 * a_q * c_q
    if a_q == 0 or disc < 0:
        return 0.0
    x = (-b_q - math.sqrt(disc)) / (2 * a_q)
    return max(0.0, min(x, 0.999))


def pka_shift_ion_pairing(site_kind: str, salt: str, concentration_M: float,
                           solvent: str, T_K: float = 298.15) -> float:
    """Tier 2 correction: Debye-Hueckel/Davies shift (Tier 1) computed
    on the FREE-ION concentration after removing the estimated
    ion-paired fraction, rather than the total salt concentration.
    In high-dielectric solvents (water) the paired fraction is
    typically negligible and this reduces to Tier 1. In low-dielectric
    solvents it can matter a lot - and the correction itself becomes
    less trustworthy, since ion pairing is exactly where simple
    electrostatic theory is weakest.
    """
    props = SOLVENTS.get(solvent)
    if props is None:
        raise ValueError(f"unknown solvent '{solvent}'. Known: {sorted(SOLVENTS)}")
    ions = _resolve_salt(salt)
    syms = list(ions)
    if len(syms) != 2:
        # pairing model here is a simple 1:1 approximation; skip for
        # anything more exotic and fall back to Tier 1 unmodified.
        I = ionic_strength(salt, concentration_M)
        return pka_shift_debye_huckel(site_kind, I, solvent, T_K)

    (sym1, sym2) = syms
    ion1, ion2 = IONS[sym1], IONS[sym2]
    avg_radius_m = ((ion1.kielland_radius_pm + ion2.kielland_radius_pm)
                     / 2) * 1e-12
    paired = ion_pair_fraction(ion1.charge, ion2.charge, avg_radius_m,
                                concentration_M, props.dielectric_constant, T_K)
    free_conc = concentration_M * (1 - paired)
    I_free = ionic_strength(salt, free_conc)
    return pka_shift_debye_huckel(site_kind, I_free, solvent, T_K)


# ---------------------------------------------------------------------
# Tier 3: Pitzer parameters (sparse, aqueous-only in practice)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class PitzerParams:
    beta0: float
    beta1: float
    c_phi: float
    source_note: str = "UNVERIFIED - check against primary Pitzer literature before relying on this."


PITZER_PARAMS: dict[str, PitzerParams] = {
    # Deliberately left as a small, explicitly-unverified starting
    # point rather than populated with unchecked numbers pulled from
    # memory. Populate this table from a primary source (e.g. Pitzer &
    # Mayorga 1973, or a validated compilation) before using Tier 3 for
    # anything that matters. Structure shown for NaCl as an example of
    # the expected shape only:
    # "NaCl": PitzerParams(beta0=0.0765, beta1=0.2664, c_phi=0.00127,
    #                       source_note="Pitzer & Mayorga (1973), Table II"),
}


def pka_shift_pitzer(site_kind: str, salt: str, concentration_M: float,
                      solvent: str = "water", T_K: float = 298.15) -> float:
    """Tier 3: not implemented pending a verified parameter table (see
    PITZER_PARAMS). Raises rather than returning a plausible-looking
    but unvalidated number.
    """
    if salt not in PITZER_PARAMS:
        raise NotImplementedError(
            f"no verified Pitzer parameters for '{salt}'. This tier is "
            f"only meaningful for salts with a checked entry in "
            f"PITZER_PARAMS - falling back to Tier 1/2 is the honest "
            f"default, not a Pitzer-quality estimate with this salt."
        )
    raise NotImplementedError("Pitzer correction formula not yet implemented")


# ---------------------------------------------------------------------
# top-level entry point: fallback chain across tiers
# ---------------------------------------------------------------------
def predict_salt_correction(site_kind: str, salt: str | None,
                             concentration_M: float | None,
                             solvent: str = "water",
                             T_K: float = 298.15) -> dict:
    """Compute the pKa shift from adding `salt` at `concentration_M`
    (mol/L) in `solvent`, using the highest-fidelity tier available,
    and report which tier was actually used.

    Returns a dict: {"shift": float, "tier": str, "note": str}
    `shift` is added to a base (salt-free) pKa prediction:
        pKa_in_solution = pKa_base + shift

    If salt is None or concentration_M is None/0, returns a zero shift
    with tier "none".
    """
    if salt is None or not concentration_M:
        return {"shift": 0.0, "tier": "none", "note": "no salt specified"}
    if solvent not in SOLVENTS:
        raise ValueError(f"unknown solvent '{solvent}'. Known: {sorted(SOLVENTS)}")

    try:
        if salt in PITZER_PARAMS:
            shift = pka_shift_pitzer(site_kind, salt, concentration_M, solvent, T_K)
            return {"shift": shift, "tier": "pitzer",
                     "note": "verified salt-specific parameters used"}
    except NotImplementedError:
        pass

    if SOLVENTS[solvent].dielectric_constant < 40:
        # low-dielectric regime: ion pairing likely non-negligible
        ions = _resolve_salt(salt)
        syms = list(ions)
        paired_frac = None
        if len(syms) == 2:
            ion1, ion2 = IONS[syms[0]], IONS[syms[1]]
            avg_radius_m = ((ion1.kielland_radius_pm + ion2.kielland_radius_pm)
                             / 2) * 1e-12
            paired_frac = ion_pair_fraction(
                ion1.charge, ion2.charge, avg_radius_m,
                concentration_M, SOLVENTS[solvent].dielectric_constant, T_K)
        shift = pka_shift_ion_pairing(site_kind, salt, concentration_M, solvent, T_K)
        note = ("low-dielectric solvent: generic electrostatics-only "
                "ion-pairing estimate, not salt-specific chemistry. "
                "Treat as order-of-magnitude, not precise.")
        if paired_frac is not None and paired_frac > 0.5:
            note = (f"WARNING: model predicts ~{paired_frac:.0%} of this salt is "
                     f"ion-paired (not free ions) in {solvent} at this "
                     f"concentration - this usually means the salt doesn't "
                     f"meaningfully dissociate here at all (often it barely "
                     f"dissolves). The large shift below is a symptom of "
                     f"leaving the model's valid regime, not a trustworthy "
                     f"prediction. Verify the salt is actually soluble and "
                     f"ionic in {solvent} before using this number.")
        return {"shift": shift, "tier": "davies+ion-pairing", "note": note}

    I = ionic_strength(salt, concentration_M)
    shift = pka_shift_debye_huckel(site_kind, I, solvent, T_K)
    note = "Davies equation"
    if I > 0.5:
        note += (f" - WARNING: ionic strength {I:.2f} mol/L exceeds the "
                 f"~0.5 mol/L regime Davies is reasonably valid for; "
                 f"treat this number with real skepticism")
    return {"shift": shift, "tier": "davies", "note": note}
