"""
umapka.microstates - thermodynamically consistent multi-site pKa.

Why this module exists
----------------------
The regressor in ``umapka.predictor`` predicts ONE proton transfer at a
time, with every other site held neutral. That is a *microscopic*
constant in one particular context. Experiments, and anything else you
would use a pKa for (charge at pH 7.4, logD, isoelectric point,
solubility, docking protonation states), are governed by the
*macroscopic* equilibrium over all protonation microstates at once.
Treating sites independently is what produces the failures listed in
the README: zwitterionic amino acids predicted at pKa 5-8 instead of
~2.3, non-monotonic pKa2/pKa3, and no titration curve at all.

What it does
------------
Standard site-site interaction model from protein electrostatics
(Tanford & Kirkwood 1957; Bashford & Karplus 1990; Ullmann 2003),
applied to small molecules:

    G(x) / (RT ln10) = sum_i x_i (pH - pKa_i^intr)
                     + sum_{i<j} W_ij q_i(x_i) q_j(x_j)
                     - log10 gamma(z(x))

  x_i            1 if site i carries its proton, else 0
  pKa_i^intr     intrinsic micro-pKa of site i with every OTHER site
                 neutral - exactly what PkaPredictor.predict_site()
                 returns, so the trained model plugs in unchanged
  q_i            site charge: acid 0 (HA) / -1 (A-); base +1 (BH+) / 0 (B)
  W_ij           charge-charge coupling in pK units (see below)
  gamma(z)       Davies activity coefficient of the microstate's net
                 charge z (ionic strength; pH is then -log10 a_H+)

Populations are Boltzmann weights 10^(-G). Macroscopic pKas follow from
the binding polynomial (Ullmann 2003): pK_n = log10(Z_n / Z_{n-1}),
where Z_n sums the pH-independent weights of every microstate holding n
protons. Everything else - titration curve, net charge vs pH,
isoelectric point, neutral fraction for logD, site occupancies - is a
sum over the same populations, so it is all mutually consistent.

The coupling W_ij
-----------------
Screened Coulomb interaction between the two sites' ionized-form charge
distributions (``umapka.sites.CHARGE_SHARE``), averaged over an RDKit
conformer ensemble, with a distance-dependent dielectric capped at the
bulk solvent value:

    W_ij = < sum_ab w_a w_b * 332.06 exp(-kappa r_ab) / (eps(r_ab) r_ab) >
           / (RT ln10)            eps(r) = min(D * r, eps_solvent)

``D`` is the only fitted constant. It was fitted on experimental
pKa1/pKa2 of 17 symmetric diacids and diamines, where the intrinsic
pKa cancels exactly: pKa2 - pKa1 = log10(4) + W. So the fit does not
depend on the ML model at all. See ``dev/validate_coupling.py`` and
``data/symmetric_polyprotic.csv``: leave-one-out MAE 0.32 pK units,
vs 0.80 for a constant-gap baseline and 1.01 for independent sites.
Held-out check on glycine (not in the fit): macro pKas 1.81 / 10.32
(exp 2.35 / 9.78; independent sites give 4.38 / 7.75) and pI 6.06
(exp 6.06). The NH3+/COO- coupling is overestimated by ~0.5 pK.

Known approximations (read before trusting numbers)
---------------------------------------------------
  - One conformer ensemble (the neutral molecule's) is used for every
    microstate. Salt bridges in zwitterions are therefore not favoured
    specially.
  - Tautomers other than those implied by moving protons between the
    detected sites are not enumerated.
  - Temperature dependence of the intrinsic pKa uses group rules
    (Perrin's rule for bases, ~0 for carboxylic-type acids); accurate to
    roughly +-0.1 pK over 15-45 C, not a replacement for measured dH.
  - Non-aqueous solvents: the dielectric cap and Davies constant are
    scaled with the solvent's permittivity, but ``D`` was fitted in
    water only.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .sites import Site, find_sites, neutralize

__all__ = [
    "COULOMB_KCAL_A", "COUPLING_D", "water_dielectric", "rt_ln10",
    "debye_kappa", "davies_log_gamma", "dpka_dT", "SiteGeometry",
    "site_geometry", "coupling_matrix", "MicrostateModel", "build_model",
]

COULOMB_KCAL_A = 332.0637          # e^2 / (4 pi eps0), kcal/mol * Angstrom
_R_KCAL = 1.987204e-3              # kcal/(mol K)
_R_J = 8.314462618                 # J/(mol K)
_LN10 = math.log(10.0)
_E = 1.602176634e-19
_EPS0 = 8.8541878128e-12
_KB = 1.380649e-23
_NA = 6.02214076e23

# Slope of the distance-dependent dielectric eps(r) = D*r. Fitted by
# dev/validate_coupling.py on data/symmetric_polyprotic.csv (water,
# 25 C, I -> 0). Re-run that script if you change the charge model.
COUPLING_D = 9.60

MAX_SITES = 12                     # 2^12 = 4096 microstates


# ---------------------------------------------------------------------
# physical helpers
# ---------------------------------------------------------------------
def water_dielectric(T_K: float = 298.15) -> float:
    """Relative permittivity of water, Malmberg & Maryott (1956),
    valid 0-100 C."""
    t = T_K - 273.15
    return 87.740 - 0.40008 * t + 9.398e-4 * t * t - 1.410e-6 * t ** 3


def rt_ln10(T_K: float = 298.15) -> float:
    """RT ln10 in kcal/mol (1.364 at 25 C): the free energy of 1 pK unit."""
    return _R_KCAL * T_K * _LN10


def debye_kappa(I: float, eps_r: float, T_K: float = 298.15) -> float:
    """Inverse Debye length in 1/Angstrom for ionic strength I (mol/L).
    Water at 25 C: 1/kappa = 3.04 A / sqrt(I)."""
    if I <= 0:
        return 0.0
    k2 = 2 * _NA * _E ** 2 * I * 1000.0 / (_EPS0 * eps_r * _KB * T_K)
    return math.sqrt(k2) * 1e-10


def davies_log_gamma(z: int, I: float, eps_r: float = 78.4,
                     T_K: float = 298.15) -> float:
    """log10 activity coefficient of an ion of charge z (Davies)."""
    if I <= 0 or z == 0:
        return 0.0
    A = 1.8246e6 / (eps_r * T_K) ** 1.5
    s = math.sqrt(I)
    return -A * z * z * (s / (1 + s) - 0.3 * I)


# Perrin, Dempsey & Serjeant (1981) group rules for d(pKa)/dT, per K.
# Bases: Perrin's rule -dpKa/dT = (pKa - 0.9)/T (reproduces e.g.
# methylammonium dH = 55 kJ/mol). Carboxylic / sulfonic / phosphoric /
# tetrazole acids: ~0 (|dH| < 5 kJ/mol). Phenol-type O-H, S-H and N-H
# acids: about -0.012/K (phenol dH ~ 24 kJ/mol).
_ACID_DPKA_DT = {
    "carboxylic_acid": 0.0, "sulfonic_acid": 0.0, "phosphoric_acid": 0.0,
    "tetrazole": 0.0, "tetrazole_2": 0.0,
    "phenol": -0.012, "thiol": -0.012, "hydroxamic_acid": -0.012,
    "imide": -0.012, "sulfonamide_1": -0.012, "sulfonamide_2": -0.012,
}


def dpka_dT(site: Site, pka_25: float, T_K: float = 298.15) -> float:
    """Approximate d(pKa)/dT (per K) for one site."""
    if site.kind == "base":
        return -(pka_25 - 0.9) / T_K
    return _ACID_DPKA_DT.get(site.group, -0.006)


# ---------------------------------------------------------------------
# geometry and coupling
# ---------------------------------------------------------------------
@dataclass
class SiteGeometry:
    """Conformer-ensemble distances between every pair of sites'
    charge-bearing atoms. ``pairs[(i, j)]`` is a list of
    (weight_a * weight_b, distances over conformers) tuples."""
    n_conformers: int
    pairs: dict = field(default_factory=dict)


def site_geometry(mol: Chem.Mol, sites: list[Site], n_conformers: int = 20,
                  seed: int = 42) -> SiteGeometry:
    """Embed a conformer ensemble of the (neutral) molecule and record
    inter-site charge-atom distances. Falls back to topological
    distance x 1.25 A if embedding fails."""
    mh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = list(AllChem.EmbedMultipleConfs(mh, n_conformers, params))
    if cids:
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mh)
        except Exception:
            pass
        pos = np.array([mh.GetConformer(c).GetPositions() for c in cids])
    else:
        pos = None
        topo = Chem.GetDistanceMatrix(mol)
    geom = SiteGeometry(len(cids) if cids else 0)
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            terms = []
            for a, wa in sites[i].charge_atoms:
                for b, wb in sites[j].charge_atoms:
                    if pos is not None:
                        d = np.linalg.norm(pos[:, a] - pos[:, b], axis=1)
                    else:
                        d = np.array([1.25 * topo[a, b]])
                    terms.append((wa * wb, np.maximum(d, 1.0)))
            geom.pairs[(i, j)] = terms
    return geom


def coupling_matrix(geom: SiteGeometry, n_sites: int,
                    eps_solvent: float = 78.4, T_K: float = 298.15,
                    ionic_strength: float = 0.0,
                    D: float = COUPLING_D) -> np.ndarray:
    """W_ij in pK units for unit charges (sign applied later via q_i q_j)."""
    kappa = debye_kappa(ionic_strength, eps_solvent, T_K)
    W = np.zeros((n_sites, n_sites))
    for (i, j), terms in geom.pairs.items():
        e = 0.0
        for w, d in terms:
            eps = np.minimum(D * d, eps_solvent)
            e += w * np.mean(COULOMB_KCAL_A * np.exp(-kappa * d) / (eps * d))
        W[i, j] = W[j, i] = e / rt_ln10(T_K)
    return W


# ---------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------
def _logsumexp10(v: np.ndarray) -> float:
    m = np.max(v)
    return float(m + np.log10(np.sum(10.0 ** (v - m))))


@dataclass
class MicrostateModel:
    """Microstate ensemble for one molecule under fixed conditions
    (temperature, solvent, ionic strength). pH is a free argument of
    every method."""
    smiles: str
    sites: list[Site]
    pka_intrinsic: np.ndarray       # at the model's temperature
    W: np.ndarray
    ionic_strength: float = 0.0
    eps_solvent: float = 78.4
    T_K: float = 298.15
    _mol: Chem.Mol | None = None

    def __post_init__(self):
        n = len(self.sites)
        self.states = np.array([[(s >> k) & 1 for k in range(n)]
                                for s in range(2 ** n)], dtype=int)
        is_base = np.array([s.kind == "base" for s in self.sites])
        # site charge: acid 0/-1 (prot/deprot), base +1/0
        self.q = np.where(is_base, self.states, self.states - 1)
        self.n_protons = self.states.sum(1)
        self.net_charge_state = self.q.sum(1)
        # pH-independent log10 weight of each microstate
        pair = np.einsum("si,ij,sj->s", self.q, np.triu(self.W, 1), self.q)
        act = np.array([davies_log_gamma(int(z), self.ionic_strength,
                                         self.eps_solvent, self.T_K)
                        for z in self.net_charge_state])
        self.log_w0 = self.states @ self.pka_intrinsic - pair - act

    # -- populations ---------------------------------------------------
    def populations(self, pH: float) -> np.ndarray:
        lw = self.log_w0 - self.n_protons * pH
        lw = lw - lw.max()
        p = 10.0 ** lw
        return p / p.sum()

    def macro_pkas(self) -> list[float]:
        """Macroscopic pKas, most acidic first (pKa1, pKa2, ...)."""
        n = len(self.sites)
        logZ = [_logsumexp10(self.log_w0[self.n_protons == k])
                for k in range(n + 1)]
        return [logZ[k] - logZ[k - 1] for k in range(n, 0, -1)]

    def net_charge(self, pH: float) -> float:
        return float(self.populations(pH) @ self.net_charge_state)

    def site_occupancy(self, pH: float) -> np.ndarray:
        """Probability that each site carries its proton."""
        return self.populations(pH) @ self.states

    def apparent_site_pkas(self) -> list[float | None]:
        """pH at which each site is half protonated (None if it never
        crosses 0.5 in -5..20)."""
        grid = np.linspace(-5, 20, 2501)
        occ = np.array([self.site_occupancy(ph) for ph in grid])
        out = []
        for k in range(len(self.sites)):
            below = np.where(occ[:, k] < 0.5)[0]
            if len(below) == 0 or below[0] == 0:
                out.append(None)
                continue
            i = below[0]
            y0, y1 = occ[i - 1, k], occ[i, k]
            out.append(float(grid[i - 1] + (y0 - 0.5) / (y0 - y1)
                             * (grid[i] - grid[i - 1])))
        return out

    def isoelectric_point(self) -> float | None:
        """pH of zero net charge, or None if the molecule never changes
        sign of charge (e.g. a plain carboxylic acid)."""
        lo, hi = -5.0, 20.0
        f_lo, f_hi = self.net_charge(lo), self.net_charge(hi)
        if f_lo * f_hi > 0:
            return None
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.net_charge(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def fraction_uncharged(self, pH: float) -> float:
        """Population with NO charged site (not just zero net charge -
        zwitterions are excluded). Use for logD = logP + log10(f)."""
        mask = np.all(self.q == 0, axis=1)
        return float(self.populations(pH)[mask].sum())

    def microstate_smiles(self, k: int) -> str:
        mol = Chem.RWMol(self._mol)
        for i, s in enumerate(self.sites):
            x = self.states[k, i]
            a = mol.GetAtomWithIdx(s.atom)
            if s.kind == "acid" and x == 0:
                a.SetNumExplicitHs(a.GetTotalNumHs() - 1)
                a.SetFormalCharge(-1)
                a.SetNoImplicit(True)
            elif s.kind == "base" and x == 1:
                a.SetNumExplicitHs(a.GetTotalNumHs() + 1)
                a.SetFormalCharge(+1)
                a.SetNoImplicit(True)
        m = mol.GetMol()
        try:
            Chem.SanitizeMol(m)
        except Exception:
            pass
        return Chem.MolToSmiles(m)

    def species(self, pH: float, min_fraction: float = 0.01) -> list[dict]:
        """Microstates above ``min_fraction`` at this pH, most populated first."""
        p = self.populations(pH)
        order = np.argsort(-p)
        return [{"smiles": self.microstate_smiles(int(k)),
                 "fraction": float(p[k]),
                 "net_charge": int(self.net_charge_state[k])}
                for k in order if p[k] >= min_fraction]

    def titration_curve(self, pH_grid=None) -> dict:
        grid = np.linspace(0, 14, 141) if pH_grid is None else np.asarray(pH_grid)
        return {"pH": grid.tolist(),
                "net_charge": [self.net_charge(ph) for ph in grid],
                "mean_protons": [float(self.populations(ph) @ self.n_protons)
                                 for ph in grid],
                "fraction_uncharged": [self.fraction_uncharged(ph) for ph in grid]}

    def summary(self, pH: float = 7.4) -> dict:
        return {
            "smiles": self.smiles,
            "sites": [{**s.as_dict(), "pKa_intrinsic": float(p),
                       "pKa_apparent": ap}
                      for s, p, ap in zip(self.sites, self.pka_intrinsic,
                                          self.apparent_site_pkas())],
            "macro_pKas": self.macro_pkas(),
            "isoelectric_point": self.isoelectric_point(),
            "pH": pH,
            "net_charge": self.net_charge(pH),
            "fraction_uncharged": self.fraction_uncharged(pH),
            "dominant_species": self.species(pH, 0.05),
            "conditions": {"T_K": self.T_K, "ionic_strength": self.ionic_strength,
                           "eps_solvent": self.eps_solvent},
        }


def build_model(smiles: str,
                intrinsic_pkas: Mapping[int, float] | Callable[[Site], float],
                T_K: float = 298.15,
                ionic_strength: float = 0.0,
                eps_solvent: float | None = None,
                n_conformers: int = 20,
                seed: int = 42,
                temperature_correct: bool = True,
                D: float = COUPLING_D) -> MicrostateModel:
    """Build a microstate model for ``smiles``.

    ``intrinsic_pkas`` gives each site's micro-pKa at 25 C with the
    other sites neutral: either a mapping site.index -> pKa, or a
    callable taking a ``Site`` (e.g. a wrapper round
    ``PkaPredictor.predict_site``). Sites missing from a mapping are
    dropped from the model, so you can restrict it to sites you trust.

    ``eps_solvent`` defaults to water at ``T_K``. ``ionic_strength`` is
    in mol/L and enters both the Davies activity term and Debye
    screening of W.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles}")
    mol = neutralize(mol)
    sites = find_sites(mol)
    if isinstance(intrinsic_pkas, Mapping):
        sites = [s for s in sites if s.index in intrinsic_pkas]
        pk = [float(intrinsic_pkas[s.index]) for s in sites]
    else:
        pk = [float(intrinsic_pkas(s)) for s in sites]
    if not sites:
        raise RuntimeError(f"no titratable site found in {smiles}")
    if len(sites) > MAX_SITES:
        raise ValueError(f"{len(sites)} sites > MAX_SITES={MAX_SITES}; pass a "
                         "mapping restricted to the sites you care about")
    pk = np.array(pk)
    if temperature_correct and abs(T_K - 298.15) > 1e-6:
        pk = pk + np.array([dpka_dT(s, p) for s, p in zip(sites, pk)]) * (T_K - 298.15)
    if eps_solvent is None:
        eps_solvent = water_dielectric(T_K)
    geom = site_geometry(mol, sites, n_conformers, seed)
    W = coupling_matrix(geom, len(sites), eps_solvent, T_K, ionic_strength, D)
    return MicrostateModel(Chem.MolToSmiles(mol), sites, pk, W,
                           ionic_strength, eps_solvent, T_K, _mol=mol)
