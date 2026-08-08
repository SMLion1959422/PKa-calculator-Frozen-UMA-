"""
umapka.solvents - single source of truth for solvent identity and the
exact feature encoding `multisolvent_tuned.pkl` was trained on.

Previously this table existed in THREE places with different encodings
(solvent_features.py used raw dielectric constants; predict_pka.py and
tune_multisolvent.py used eps/78.4 normalized values - the one that
actually matches what the shipped model was trained on). Predicting
with the wrong encoding silently gives a wrong-but-plausible-looking
number, since nothing errors - the regressor just gets a different
input. This module is now the only place that encoding lives.

`eps_norm` and `protic` below are copied VERBATIM from
`tune_multisolvent.py`'s `SOLVENTS` dict (the actual training script for
`multisolvent_tuned.pkl`) - do not "fix" these to more accurate physical
values without retraining the model, or predictions will be silently
wrong.
"""

from __future__ import annotations
from dataclasses import dataclass

__all__ = ["SolventInfo", "SOLVENTS", "resolve_solvent"]


@dataclass(frozen=True)
class SolventInfo:
    name: str                 # canonical display name, e.g. "Acetonitrile"
    smiles: str                # neutral solvent SMILES, used as a dict key upstream
    eps_norm: float             # normalized dielectric feature, AS TRAINED (not raw epsilon)
    protic: float               # protic-character feature, AS TRAINED
    eps_raw: float               # actual dielectric constant, ~25C (for solvation.py / mixtures.py)
    test_mae: float | None       # held-out MAE from RESULTS.md, or None if untested
    solvation_key: str | None    # matching key in solvation.SOLVENTS, or None if not modeled there


SOLVENTS: dict[str, SolventInfo] = {
    "water":          SolventInfo("Water", "O", 1.00, 1.0, 78.4, 0.78, "water"),
    "dmso":           SolventInfo("DMSO", "CS(C)=O", 0.60, 0.0, 46.7, 1.15, "dmso"),
    "acetonitrile":   SolventInfo("Acetonitrile", "CC#N", 0.48, 0.0, 37.5, 0.71, "acetonitrile"),
    "dmf":            SolventInfo("DMF", "CN(C)C=O", 0.47, 0.0, 36.7, 0.40, "dmf"),
    "methanol":       SolventInfo("Methanol", "CO", 0.42, 0.5, 32.7, 0.62, "methanol"),
    "ethanol":        SolventInfo("Ethanol", "CCO", 0.31, 0.5, 24.5, 0.18, "ethanol"),
    "nmp":            SolventInfo("NMP", "CN1CCCC1=O", 0.42, 0.0, 32.2, None, None),
    "ethyleneglycol": SolventInfo("EthyleneGlycol", "C(CO)O", 0.51, 0.5, 37.0, None, None),
}

_ALIASES = {
    "h2o": "water", "w": "water",
    "mecn": "acetonitrile", "acn": "acetonitrile",
    "meoh": "methanol",
    "etoh": "ethanol",
    "eg": "ethyleneglycol", "ethylene glycol": "ethyleneglycol",
}


def resolve_solvent(name: str) -> SolventInfo:
    """Look up a solvent by canonical name, alias, or SMILES.
    Raises ValueError (with the supported list) if unrecognized -
    this deliberately never guesses.
    """
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key in SOLVENTS:
        return SOLVENTS[key]
    for info in SOLVENTS.values():
        if name.strip() == info.smiles:
            return info
    raise ValueError(
        f"unknown solvent '{name}'. Supported: "
        f"{sorted(s.name for s in SOLVENTS.values())}"
    )
