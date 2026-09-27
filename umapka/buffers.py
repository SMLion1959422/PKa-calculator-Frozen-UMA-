"""
umapka.buffers - mobile-phase / buffer selection from a molecule's pKa.

The practical question a chromatographer asks is not "what is the pKa" but
"what pH should I run at, and what buffer holds it there". Those follow from
the pKa by three well-established rules:

  1. RETENTION REPRODUCIBILITY. Near its pKa an analyte's ionized fraction
     swings steeply with pH, so small pH errors move retention a lot. Work
     at least ~2 pH units away from every pKa, which puts the analyte >=99%
     in one form. (Snyder, Kirkland & Dolan, *Practical HPLC Method
     Development*; the 1.5-2 unit rule is standard practice.)
  2. BUFFER CAPACITY. A buffer only buffers within about +-1 pH unit of its
     OWN pKa. Outside that it is salt, not buffer.
  3. COLUMN AND DETECTOR LIMITS. Bare-silica C18 dissolves above ~pH 8 and
     loses bonded phase below ~pH 2; hybrid particles go further. LC-MS
     needs VOLATILE buffers - phosphate and citrate contaminate the source
     and suppress ionization.

Rule 1 and rule 3 often disagree for basic analytes in positive-mode ESI,
and this module reports that tension rather than hiding it: a base run at
low pH is protonated (great for positive ESI, poor reversed-phase
retention), while running it neutral needs a high-pH-tolerant column.

Speciation is exact for the macroscopic constants given: the fraction with
j protons removed is proportional to 10^(j*pH - sum of the first j pKas)
(the binding polynomial), so polyprotic analytes are handled properly
rather than one pKa at a time.

NOT a substitute for method development. Retention also depends on organic
modifier, temperature, ionic strength and the stationary phase, none of
which are modelled here. Treat the output as a starting point.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Buffer", "BUFFERS", "COLUMN_CHEMISTRIES", "speciation",
           "clean_windows", "recommend", "format_report", "charge_label",
           "advise_smiles"]


def charge_label(z):
    """'neutral' reads better than '+0' in a report."""
    return "neutral" if z == 0 else f"{z:+d}"


# ---------------------------------------------------------------------
# buffer registry
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Buffer:
    name: str
    pkas: tuple            # buffering pKa(s) at ~25 C
    volatile: bool         # usable with MS
    uv_cutoff_nm: float | None   # approximate; None = no meaningful UV limit
    note: str = ""

    def ranges(self, width=1.0):
        return [(p - width, p + width) for p in self.pkas]


BUFFERS: list[Buffer] = [
    Buffer("formic acid / ammonium formate", (3.75,), True, 210,
           "the LC-MS workhorse at low pH; ~0.1% formic acid is ~pH 2.7"),
    Buffer("acetic acid / ammonium acetate", (4.76,), True, 210,
           "volatile, mild; weaker UV absorbance than formate"),
    Buffer("ammonium bicarbonate", (6.35, 10.33), True, 220,
           "volatile high-pH option; needs a pH-stable column, and it "
           "outgasses CO2 so prepare fresh"),
    Buffer("ammonium hydroxide / ammonia", (9.25,), True, None,
           "volatile basic modifier, common for negative-mode ESI"),
    Buffer("trifluoroacetic acid (TFA)", (0.30,), True, 210,
           "an ion-pairing agent more than a buffer; strongly suppresses "
           "ESI - avoid for MS, excellent peak shape for UV"),
    Buffer("phosphate", (2.15, 7.20, 12.35), False, 200,
           "the classic UV buffer, widest useful range - NOT MS compatible"),
    Buffer("citrate", (3.13, 4.76, 6.40), False, 230,
           "strong buffering across a wide span; non-volatile and it "
           "chelates metals"),
    Buffer("MES", (6.15,), False, 230, "zwitterionic Good's buffer"),
    Buffer("HEPES", (7.55,), False, 230, "zwitterionic Good's buffer"),
    Buffer("tris", (8.06,), False, 210,
           "pKa shifts strongly with temperature (~-0.03/K)"),
    Buffer("borate", (9.24,), False, 200, "non-volatile; can complex diols"),
]

# usable pH range by column chemistry
COLUMN_CHEMISTRIES = {
    "silica-C18": (2.0, 8.0),
    "hybrid-BEH": (1.0, 12.0),
    "polymeric": (1.0, 13.0),
}


# ---------------------------------------------------------------------
# speciation
# ---------------------------------------------------------------------
def speciation(pkas, pH, z_protonated=0):
    """Fractions of each protonation level at `pH`, most-protonated first.

    `pkas` are MACROSCOPIC constants in ascending order. `z_protonated` is
    the charge of the fully protonated species (0 for a neutral acid HA,
    +1 for a protonated base BH+). Returns (fractions, charges).
    """
    pkas = sorted(float(p) for p in pkas)
    cum = np.concatenate([[0.0], np.cumsum(pkas)])
    j = np.arange(len(pkas) + 1)
    logw = j * pH - cum
    w = 10.0 ** (logw - logw.max())
    return w / w.sum(), z_protonated - j


def clean_windows(pkas, z_protonated=0, purity=0.99, lo=1.0, hi=13.0, step=0.02):
    """pH intervals where ONE protonation state holds >= `purity`.

    purity 0.99 corresponds to ~2 pH units from the nearest pKa, which is
    the standard reproducibility rule.
    """
    grid = np.arange(lo, hi + 1e-9, step)
    out, start, cur = [], None, None
    for pH in grid:
        f, z = speciation(pkas, pH, z_protonated)
        k = int(np.argmax(f))
        ok = f[k] >= purity
        if ok and start is None:
            start, cur = pH, k
        elif start is not None and (not ok or k != cur):
            out.append((round(start, 2), round(pH - step, 2), int(z[cur])))
            start, cur = (pH, k) if ok else (None, None)
    if start is not None:
        out.append((round(start, 2), round(grid[-1], 2), int(z[cur])))
    return [w for w in out if w[1] - w[0] >= 0.3]


# ---------------------------------------------------------------------
# recommendation
# ---------------------------------------------------------------------
def _overlap(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if hi - lo > 1e-9 else None


def recommend(pkas, z_protonated=0, ms=True, column="silica-C18",
              min_purity=0.95, buffer_width=1.0, max_results=6, step=0.05,
              zwitterionic=False):
    """Rank (pH, buffer) conditions for a molecule with these macro pKas.

    Scans every pH each buffer can actually hold (its pKa +-`buffer_width`,
    intersected with the column limits) and keeps those where the analyte is
    at least `min_purity` in one protonation state. Scanning rather than
    intersecting with the strict >=99% windows matters in practice: 0.1%
    formic acid sits at ~pH 2.7, which is only ~1.5 units below a pKa-4.2
    acid (~97% neutral) - strictly outside the 2-unit rule, and the standard
    answer anyway.

    ms=True restricts to volatile buffers and reports a suggested ESI
    polarity. `column` keys COLUMN_CHEMISTRIES.

    `zwitterionic=True` matters for amino acids and anything with both an
    acid and a base site: NET charge zero there means NH3+/COO-, not an
    uncharged molecule, and it retains poorly on reversed phase. Net charge
    alone cannot tell the two apart - umapka.microstates.fraction_uncharged
    can, which is what `advise_smiles` passes in.
    """
    if column not in COLUMN_CHEMISTRIES:
        raise ValueError(f"unknown column '{column}'. "
                         f"Known: {sorted(COLUMN_CHEMISTRIES)}")
    col = COLUMN_CHEMISTRIES[column]
    out = []
    for buf in BUFFERS:
        if ms and not buf.volatile:
            continue
        for centre, br in zip(buf.pkas, buf.ranges(buffer_width)):
            usable = _overlap(br, col)
            if usable is None:
                continue
            best = None
            for pH in np.arange(usable[0], usable[1] + 1e-9, step):
                f, z = speciation(pkas, pH, z_protonated)
                k = int(np.argmax(f))
                if f[k] < min_purity:
                    continue
                charge = int(z[k])
                margin = min(abs(pH - p) for p in pkas) if pkas else 9.9
                neutral_bonus = 0.0 if charge != 0 else (0.8 if zwitterionic else 3.0)
                score = (
                    neutral_bonus                       # uncharged -> best RP retention
                    + min(margin, 2.5)                  # distance from the analyte's pKa
                    - abs(pH - centre)                  # buffer capacity
                    - (1.5 if "TFA" in buf.name and ms else 0.0)
                )
                if best is None or score > best[0]:
                    best = (score, float(pH), charge, float(f[k]), float(margin))
            if best is None:
                continue
            score, pH, charge, purity, margin = best
            warn = []
            if not (col[0] + 0.5 <= pH <= col[1] - 0.5):
                warn.append(f"pH {pH:.1f} is close to the {column} limit "
                            f"({col[0]:g}-{col[1]:g})")
            if charge != 0:
                warn.append("analyte is ionized here: weak reversed-phase "
                            "retention - consider ion-pairing or HILIC")
            elif zwitterionic:
                warn.append("NET charge is zero, but this molecule has both an "
                            "acid and a base site - it is most likely a "
                            "ZWITTERION here, which still retains poorly on "
                            "reversed phase. HILIC or ion-pairing may suit better")
            if purity < 0.99:
                warn.append(f"{purity:.0%} of one species, {round(margin, 2):.1f} pH "
                            f"units from a pKa - below the 99%/2-unit rule, so "
                            f"retention is more sensitive to pH error")
            if ms and "TFA" in buf.name:
                warn.append("TFA strongly suppresses ESI")
            out.append(dict(pH=round(pH, 2), buffer=buf.name, buffer_pka=centre,
                            volatile=buf.volatile, uv_cutoff_nm=buf.uv_cutoff_nm,
                            analyte_charge=charge, purity=purity,
                            margin=round(margin, 2),
                            esi=("positive" if charge > 0 else
                                 "negative" if charge < 0 else
                                 "either (neutral in solution)"),
                            score=round(score, 3), note=buf.note, warnings=warn))
    out.sort(key=lambda r: -r["score"])
    seen, kept = set(), []
    for r in out:
        key = (r["buffer"], r["analyte_charge"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept[:max_results]


def format_report(pkas, z_protonated=0, ms=True, column="silica-C18",
                  min_purity=0.95, name=None, zwitterionic=False):
    """Human-readable report; returns a string."""
    L = []
    head = f"Buffer / mobile-phase guidance{'' if name is None else ' for ' + name}"
    L.append(head)
    L.append("=" * len(head))
    L.append(f"macro pKa: {', '.join(f'{p:.2f}' for p in sorted(pkas))}"
             f"    fully-protonated charge {z_protonated:+d}")
    L.append(f"column: {column} (pH {COLUMN_CHEMISTRIES[column][0]:g}–"
             f"{COLUMN_CHEMISTRIES[column][1]:g})"
             f"    detection: {'LC-MS (volatile buffers only)' if ms else 'UV / any buffer'}")
    L.append("")
    w = clean_windows(pkas, z_protonated, 0.99)
    L.append("pH windows meeting the strict rule (>= 99% one species, "
             "~2 pH units from every pKa):")
    if w:
        for lo, hi, z in w:
            L.append(f"   pH {lo:5.2f} – {hi:5.2f}   analyte {charge_label(z)}"
                     + ("   (neutral: best reversed-phase retention)" if z == 0 else ""))
    else:
        L.append("   none in pH 1–13 — the pKas are too close together to isolate\n"
                 "   a single species that cleanly; the options below trade purity\n"
                 "   for practicality.")
    L.append("")
    recs = recommend(pkas, z_protonated, ms, column, min_purity,
                     zwitterionic=zwitterionic)
    if not recs:
        L.append("No buffer satisfies every constraint. Try ms=False (opens\n"
                 "phosphate and citrate), a wider-range column such as\n"
                 "hybrid-BEH, or a lower min_purity.")
        return "\n".join(L)
    L.append("Recommended conditions, best first:")
    for i, r in enumerate(recs, 1):
        L.append(f"\n  {i}. pH {r['pH']:.1f}  with  {r['buffer']}"
                 f"   (buffer pKa {r['buffer_pka']:.2f})")
        L.append(f"     analyte {charge_label(r['analyte_charge'])}, "
                 f"{r['purity']:.1%} of one species, "
                 f"{r['margin']:.1f} pH units from the nearest pKa")
        L.append(f"     ESI polarity: {r['esi']}"
                 + (f"   ·   UV cutoff ~{r['uv_cutoff_nm']:.0f} nm"
                    if r["uv_cutoff_nm"] else ""))
        if r["note"]:
            L.append(f"     note: {r['note']}")
        for wn in r["warnings"]:
            L.append(f"     ! {wn}")
    if not any(r["analyte_charge"] == 0 for r in recs):
        neutral = [x for x in clean_windows(pkas, z_protonated, 0.99) if x[2] == 0]
        if neutral:
            lo, hi = neutral[0][0], neutral[0][1]
            better = [k for k, (a, b) in COLUMN_CHEMISTRIES.items()
                      if a <= hi and b >= lo and k != column]
            L.append("")
            L.append(f"  NOTE: no option above leaves the analyte uncharged. It is "
                     f"uncharged at pH {lo:.1f}–{hi:.1f},")
            L.append(f"  which is outside the {column} limit "
                     f"({COLUMN_CHEMISTRIES[column][0]:g}–{COLUMN_CHEMISTRIES[column][1]:g})."
                     + (f" A {' or '.join(better)} column would reach it."
                        if better else ""))
    L.append("")
    L.append("Concentration 10–25 mM is typical; keep >= 95% aqueous buffer "
             "capacity in mind when the organic fraction is high.")
    L.append("Starting point only — organic modifier, temperature and "
             "stationary phase are not modelled here.")
    return "\n".join(L)


def advise_smiles(predictor, smiles, ms=True, column="silica-C18",
                  min_purity=0.95, **macro_kw):
    """End-to-end: predict the macro pKas of `smiles`, then advise on buffers.

    `predictor` is a PkaPredictor. The fully-protonated charge is the number
    of basic sites, and a molecule carrying BOTH an acid and a base site is
    flagged zwitterionic so a net-neutral state is not mistaken for an
    uncharged one.
    """
    out = predictor.predict_macro(smiles, **macro_kw)
    kinds = [s_["kind"] for s_ in out["sites"]]
    z0 = sum(1 for k in kinds if k == "base")
    zwit = ("acid" in kinds) and ("base" in kinds)
    report = format_report(out["macro_pKas"], z0, ms=ms, column=column,
                           min_purity=min_purity, name=smiles, zwitterionic=zwit)
    return {"smiles": smiles, "macro_pKas": out["macro_pKas"],
            "z_protonated": z0, "zwitterionic": zwit,
            "recommendations": recommend(out["macro_pKas"], z0, ms, column,
                                         min_purity, zwitterionic=zwit),
            "report": report, "pka_detail": out}
