"""
umapka.solvent_descriptors - continuous physical descriptors of solvents.

Used by the solvent-shift experiment (dev/solvent_shift_experiment.py)
in place of the (eps/78.4, protic flag) encoding of multisolvent_tuned.pkl.
Continuous descriptors are what allow a model to say anything about a
solvent it has not seen in training.

Columns
  eps     relative permittivity, 25 C
  alpha   Kamlet-Taft hydrogen-bond donor acidity
  beta    Kamlet-Taft hydrogen-bond acceptor basicity
  pi      Kamlet-Taft dipolarity/polarizability (pi*)
  ET30    Reichardt E_T(30), kcal/mol

Values are the commonly tabulated ones (Marcus, Chem. Soc. Rev. 1993,
22, 409; Reichardt & Welton, Solvents and Solvent Effects in Organic
Chemistry). Kamlet-Taft values differ between compilations by up to
~0.1, most notably beta for water (0.18 in the original scale, 0.47 in
Marcus); Marcus is used here. Treat them as +-0.05-0.1.
"""

from __future__ import annotations
import numpy as np

__all__ = ["DESCRIPTORS", "DESCRIPTOR_NAMES", "descriptor_vector", "SMILES_TO_NAME"]

DESCRIPTOR_NAMES = ("eps", "alpha", "beta", "pi", "ET30")

#                 eps    alpha  beta   pi*    ET(30)
DESCRIPTORS = {
    "Water":          (78.4, 1.17, 0.47, 1.09, 63.1),
    "DMSO":           (46.7, 0.00, 0.76, 1.00, 45.1),
    "Acetonitrile":   (35.9, 0.19, 0.40, 0.75, 45.6),
    "DMF":            (36.7, 0.00, 0.69, 0.88, 43.2),
    "Methanol":       (32.7, 0.98, 0.66, 0.60, 55.4),
    "Ethanol":        (24.5, 0.86, 0.75, 0.54, 51.9),
    "NMP":            (32.2, 0.00, 0.77, 0.92, 42.2),
    "EthyleneGlycol": (37.7, 0.90, 0.52, 0.92, 56.3),
}

SMILES_TO_NAME = {
    "O": "Water", "CS(C)=O": "DMSO", "CC#N": "Acetonitrile",
    "CN(C)C=O": "DMF", "CO": "Methanol", "CCO": "Ethanol",
    "CN1CCCC1=O": "NMP", "OCCO": "EthyleneGlycol", "C(CO)O": "EthyleneGlycol",
}


def descriptor_vector(name: str) -> np.ndarray:
    """[1/eps, alpha, beta, pi*, ET30/100]. 1/eps rather than eps because
    electrostatic solvation terms (Born) scale with 1/eps."""
    eps, a, b, p, et = DESCRIPTORS[name]
    return np.array([1.0 / eps, a, b, p, et / 100.0])
