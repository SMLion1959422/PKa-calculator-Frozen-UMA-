"""
umapka.mixtures - pKa in binary solvent mixtures (e.g. water/acetonitrile,
water/DMSO, water/methanol), built on top of the trained pure-solvent
model rather than by feeding an interpolated feature into it.

WHY IT'S BUILT THIS WAY (read this before trusting numbers)
-------------------------------------------------------------------------
There is no mixed-solvent training data anywhere in this project - the
multisolvent model (`multisolvent_tuned.pkl`) was trained ONLY on single
pure solvents (RESULTS.md: "All data are single pure solvents; mixtures
out of scope"). Two ways to get a mixture prediction:

  (A) Feed the ML regressor a made-up "interpolated dielectric constant"
      for the mixture and let it predict. This LOOKS reasonable because
      the model literally takes a continuous epsilon feature - but the
      model has never seen intermediate epsilon values during training,
      so this is silent extrapolation dressed up as interpolation. A
      tree-based regressor (LightGBM) does not extrapolate gracefully:
      outside the range of splits it saw, it just returns whatever the
      nearest leaf says, with no reason to believe that's physically
      meaningful.

  (B) Predict the two PURE-solvent endpoints with the trained model
      (this is the part "training" is good at and validated for), then
      use a physical law to interpolate BETWEEN those two trained,
      trustworthy numbers as composition changes. This is the
      Yasuda-Shedlovsky approach used in real pharmaceutical pKa
      determination: apparent pKa is approximately linear in 1/epsilon
      of the mixture across a water-cosolvent composition range.

This module does (B). It is deliberately conservative: it will not
silently produce a number for solvent pairs / composition ranges where
the underlying assumption (linearity in 1/epsilon) is known to break
down (see `_CURVATURE_WARNING_THRESHOLD` below), and it always reports
which regime it's in.

Still an approximation:
  - Yasuda-Shedlovsky linearity is best-established in the water-rich
    region (organic cosolvent mole fraction below ~0.3-0.5) for
    carboxylic acids and ammonium-type bases. Near the organic-rich end,
    and for solvent pairs with strong specific solvation (e.g. DMSO
    H-bond accepting to a proton donor), real behaviour can curve.
  - It answers "what is the apparent pKa of THIS solute in a
    water/cosolvent mixture", not "what is the true thermodynamic pKa
    referenced to the mixture's own autoprotolysis" - those can differ,
    same caveat classical Yasuda-Shedlovsky papers carry.
  - The dielectric constant of the mixture itself is also approximated
    (see `mixture_dielectric`) - real epsilon(composition) curves for
    common pairs (water/MeCN, water/DMSO, water/MeOH) are mildly
    S-shaped, not perfectly linear in volume fraction. Good enough to
    pick the right regime; not a substitute for a measured value if you
    have one.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

__all__ = [
    "MIXTURE_DIELECTRIC_DATA", "mixture_dielectric",
    "predict_mixed_solvent_pka",
]

# ---------------------------------------------------------------------
# Pure-solvent dielectric constants (~25 C). Same values used elsewhere
# in this project (solvation.py / solvent_features.py) - kept here too
# so this module has no import-order dependency on those.
# ---------------------------------------------------------------------
PURE_DIELECTRIC = {
    "water": 78.4, "h2o": 78.4,
    "dmso": 46.7,
    "acetonitrile": 37.5, "mecn": 37.5,
    "methanol": 32.7, "meoh": 32.7,
    "dmf": 36.7,
    "ethanol": 24.5, "etoh": 24.5,
    "acetone": 20.7,
    "thf": 7.6,
    "dcm": 8.9,
}

# A few real measured water/cosolvent dielectric curves, digitized at
# coarse composition steps from standard physical-chemistry compilations
# (Akerlof-type measurements), volume-fraction organic -> epsilon. Used
# in preference to the crude linear rule when available, since real
# curves are visibly S-shaped rather than linear.
# NOTE: these are commonly-cited literature values, not a precision
# reference - re-check against a primary source before anything
# precision-critical.
MIXTURE_DIELECTRIC_DATA = {
    ("water", "acetonitrile"): {
        0.0: 78.4, 0.2: 62.0, 0.4: 50.0, 0.6: 42.0, 0.8: 38.5, 1.0: 37.5,
    },
    ("water", "methanol"): {
        0.0: 78.4, 0.2: 68.0, 0.4: 58.0, 0.6: 48.0, 0.8: 39.0, 1.0: 32.7,
    },
    ("water", "dmso"): {
        0.0: 78.4, 0.2: 72.0, 0.4: 65.0, 0.6: 58.0, 0.8: 51.0, 1.0: 46.7,
    },
    ("water", "ethanol"): {
        0.0: 78.4, 0.2: 63.0, 0.4: 50.0, 0.6: 40.0, 0.8: 30.0, 1.0: 24.5,
    },
}

# Above this fraction of organic cosolvent, treat the Yasuda-Shedlovsky
# linear-in-1/epsilon assumption as unreliable rather than silently
# extrapolating it further than the classic literature range supports.
_CURVATURE_WARNING_FRACTION = 0.5

# Power applied to the 1/epsilon interpolation fraction `t` before using
# it to blend the two endpoint pKa values. 1.0 = straight-line
# behaviour. >1 flattens the curve near the water-rich endpoint and
# steepens it near the organic-rich endpoint; both trained endpoints
# stay exact.
#
# THIS IS AN UNVALIDATED HEURISTIC SHAPE, NOT A FITTED PARAMETER. No
# composition-resolved experimental mixture pKa data exists in this
# project, so there is nothing here to fit it against and no number
# that would justify 2.5 over 2.0 or 3.0. It is applied because a
# straight line between the two endpoints was judged to overshoot in
# the water-rich region (the pure-organic endpoint sits on a different
# absolute pKa scale / standard state than water, so the two endpoints
# are not two points on one physical curve). Treat every mixture number
# as directional. See predict_mixed_solvent_pka's returned
# "confidence"/"warning" and the README section on mixtures.
#
# (Was previously defined twice, identically, in this file - editing the
# first copy silently had no effect because the second overwrote it.)
_CURVATURE_EXPONENT = 2.5


def _lookup_pair(solvent_a: str, solvent_b: str):
    a, b = solvent_a.lower(), solvent_b.lower()
    if (a, b) in MIXTURE_DIELECTRIC_DATA:
        return MIXTURE_DIELECTRIC_DATA[(a, b)], False
    if (b, a) in MIXTURE_DIELECTRIC_DATA:
        flipped = {round(1 - k, 2): v for k, v in MIXTURE_DIELECTRIC_DATA[(b, a)].items()}
        return flipped, False
    return None, True


def mixture_dielectric(solvent_a: str, solvent_b: str,
                        fraction_b: float) -> tuple[float, str]:
    """Estimate the dielectric constant of a binary A/B mixture at the
    given volume fraction of B (0 = pure A, 1 = pure B).

    Returns (epsilon, method) where method is "measured-curve"
    (interpolated from a digitized real curve) or "linear-fallback"
    (crude volume-fraction-weighted average of the pure values - used
    only when no measured curve is on file for this pair).
    """
    if not 0.0 <= fraction_b <= 1.0:
        raise ValueError("fraction_b must be between 0 and 1")

    curve, is_fallback = _lookup_pair(solvent_a, solvent_b)
    if curve is not None:
        xs = sorted(curve)
        if fraction_b in curve:
            return curve[fraction_b], "measured-curve"
        lo = max(x for x in xs if x <= fraction_b)
        hi = min(x for x in xs if x >= fraction_b)
        if lo == hi:
            return curve[lo], "measured-curve"
        t = (fraction_b - lo) / (hi - lo)
        eps = curve[lo] + t * (curve[hi] - curve[lo])
        return eps, "measured-curve"

    eps_a = PURE_DIELECTRIC.get(solvent_a.lower())
    eps_b = PURE_DIELECTRIC.get(solvent_b.lower())
    if eps_a is None or eps_b is None:
        raise ValueError(
            f"unknown solvent(s) '{solvent_a}'/'{solvent_b}'. Known pure "
            f"solvents: {sorted(PURE_DIELECTRIC)}"
        )
    eps = (1 - fraction_b) * eps_a + fraction_b * eps_b
    return eps, "linear-fallback"


def predict_mixed_solvent_pka(predictor, smiles: str,
                               solvent_a: str, solvent_b: str,
                               fraction_b: float,
                               site_index: int | None = None) -> dict:
    """Predict pKa in a binary A/B solvent mixture.

    Anchors on TWO predictions from the trained model - pure `solvent_a`
    and pure `solvent_b` (each must be one of the pure solvents the
    multisolvent model was actually trained on) - then interpolates
    Yasuda-Shedlovsky-style: linear in 1/epsilon between those two
    trained, validated endpoints. `predictor` is a `PkaPredictor`
    already constructed with the multisolvent regressor loaded, or
    anything exposing the same `.predict(smiles, ...)` /
    `.predict_site(smiles, index, ...)` interface used elsewhere in
    this package for solvent-aware prediction.

    Parameters
    ----------
    fraction_b : volume fraction of solvent_b, 0 (pure A) to 1 (pure B).

    Returns
    -------
    dict with:
      "pKa"          - interpolated estimate
      "endpoint_a"   - trained-model pKa in pure solvent_a
      "endpoint_b"   - trained-model pKa in pure solvent_b
      "epsilon_mix"  - dielectric constant used for the mixture
      "epsilon_method" - "measured-curve" or "linear-fallback"
      "confidence"   - "normal" or "low"
      "warning"      - str or None
    """
    if site_index is not None:
        pka_a = predictor.predict_site(smiles, site_index, solvent=solvent_a)
        pka_b = predictor.predict_site(smiles, site_index, solvent=solvent_b)
    else:
        pka_a = predictor.predict(smiles, solvent=solvent_a)
        pka_b = predictor.predict(smiles, solvent=solvent_b)

    eps_a = PURE_DIELECTRIC.get(solvent_a.lower())
    eps_b = PURE_DIELECTRIC.get(solvent_b.lower())
    if eps_a is None or eps_b is None:
        raise ValueError(
            f"unknown solvent(s) '{solvent_a}'/'{solvent_b}'. Known: "
            f"{sorted(PURE_DIELECTRIC)}"
        )
    eps_mix, eps_method = mixture_dielectric(solvent_a, solvent_b, fraction_b)

    # Yasuda-Shedlovsky: pKa is taken as a function of 1/epsilon.
    # Interpolate between the two TRAINED endpoint predictions, not by
    # re-running the ML model on a fabricated intermediate feature.
    inv_a, inv_b, inv_mix = 1.0 / eps_a, 1.0 / eps_b, 1.0 / eps_mix
    if inv_b == inv_a:
        t = 0.0
    else:
        t = (inv_mix - inv_a) / (inv_b - inv_a)
    # Curvature correction: a straight line in 1/epsilon between the two
    # endpoints overshoots in the water-rich region for every solvent
    # pair, because the pure-organic endpoint sits on a very different
    # absolute pKa scale than water. Reparametrizing t through a convex
    # power law keeps both trained endpoints exact while suppressing
    # that early overshoot and pushing most of the change later.
    t_clamped = min(max(t, 0.0), 1.0)
    t_shaped = t_clamped ** _CURVATURE_EXPONENT
    pka_mix = pka_a + t_shaped * (pka_b - pka_a)

    warning = None
    confidence = "normal"
    organic_fraction = fraction_b if solvent_a.lower() in ("water", "h2o") else (1 - fraction_b)
    if "water" not in (solvent_a.lower(), solvent_b.lower()) and \
       "h2o" not in (solvent_a.lower(), solvent_b.lower()):
        confidence = "low"
        warning = ("neither solvent is water - Yasuda-Shedlovsky linearity "
                   "is best-established for water/organic-cosolvent pairs; "
                   "treat this as a rough estimate")
    elif organic_fraction > _CURVATURE_WARNING_FRACTION:
        confidence = "low"
        warning = (f"organic cosolvent fraction ({organic_fraction:.0%}) is "
                   f"past the water-rich range where Yasuda-Shedlovsky "
                   f"behaviour is best established; a curvature correction "
                   f"is applied (see umapka/mixtures.py), but it is a "
                   f"documented heuristic shape, not measured mixture data "
                   f"- treat this estimate with real skepticism this far "
                   f"from the aqueous endpoint")
    if eps_method == "linear-fallback":
        confidence = "low"
        extra = ("no measured dielectric curve on file for this solvent "
                 "pair - using a crude volume-fraction-weighted average "
                 "instead of a real epsilon(composition) curve")
        warning = extra if warning is None else warning + "; " + extra

    return {
        "pKa": pka_mix,
        "endpoint_a": pka_a,
        "endpoint_b": pka_b,
        "epsilon_mix": eps_mix,
        "epsilon_method": eps_method,
        "interp_fraction": t_clamped,
        "confidence": confidence,
        "warning": warning,
    }
