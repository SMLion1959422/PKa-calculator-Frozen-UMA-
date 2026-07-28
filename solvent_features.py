# solvent_features.py - SUPERSEDED, kept only for history.
#
# DO NOT USE THIS FILE'S ENCODING TO BUILD FEATURES FOR
# multisolvent_tuned.pkl. It uses RAW dielectric constants (eps/78.4,
# 3 components), which is NOT the encoding that model was actually
# trained with - tune_multisolvent.py used a hand-picked (eps_norm,
# protic) pair with different numbers per solvent (e.g. DMSO here would
# be 46.7/78.4=0.60 by coincidence, but Acetonitrile is 37.5/78.4=0.478
# here vs 0.48 in training - close but not guaranteed identical for
# every solvent, and NMP/EthyleneGlycol aren't in this table at all).
# Use `umapka.solvents.resolve_solvent()` instead, which reproduces
# tune_multisolvent.py's table exactly. This file is left in place
# only so old notebooks referencing it don't immediately break.
#
# ORIGINAL COMMENT (for context, now historical):
# the ONE new concept for multi-solvent: encode solvent as features.
# These 5 physical constants are what let one model span solvents. Fit on data later.
# Values: dielectric constant (eps), and normalized solvent descriptors.

SOLVENT_PARAMS = {
    # solvent:      [dielectric, autoprotolysis_pKauto, is_aqueous]
    "H2O":   [78.4, 14.0, 1.0],
    "DMSO":  [46.7, 35.0, 0.0],
    "MeCN":  [37.5, 33.0, 0.0],   # acetonitrile
    "MeOH":  [32.7, 17.2, 0.0],   # methanol
    "DMF":   [36.7, 23.0, 0.0],
    "EtOH":  [24.5, 19.0, 0.0],
}

import numpy as np

def solvent_vector(solvent_name):
    """Return the physical-descriptor vector for a solvent, or None if unknown."""
    p = SOLVENT_PARAMS.get(solvent_name)
    if p is None:
        return None
    eps, pkauto, aqueous = p
    # normalize roughly to O(1) so it plays well with the 768-dim embedding features
    return np.array([eps / 78.4, pkauto / 35.0, aqueous])

def combined_feature(mol_feature_768, solvent_name):
    """Concatenate the existing 768-dim molecular feature with the solvent vector.
    This is the whole trick: same molecule, different solvent -> different feature -> different pKa."""
    sv = solvent_vector(solvent_name)
    if sv is None:
        return None
    return np.concatenate([mol_feature_768, sv])   # -> 771-dim
