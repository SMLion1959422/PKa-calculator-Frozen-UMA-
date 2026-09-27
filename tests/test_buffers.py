"""Buffer / mobile-phase recommendation from pKa."""
import numpy as np
import pytest
from umapka.buffers import (speciation, clean_windows, recommend,
                            format_report, charge_label, BUFFERS)


def test_speciation_half_and_half_at_the_pka():
    f, z = speciation([4.76], 4.76)
    assert f == pytest.approx([0.5, 0.5], abs=1e-9)
    assert list(z) == [0, -1]


def test_speciation_two_units_away_is_99_percent():
    f, _ = speciation([4.76], 6.76)
    assert f[1] == pytest.approx(0.99, abs=0.005)
    f, _ = speciation([4.76], 2.76)
    assert f[0] == pytest.approx(0.99, abs=0.005)


def test_speciation_sums_to_one_and_handles_polyprotic():
    for pH in (1.0, 4.0, 7.4, 11.0):
        f, z = speciation([2.15, 7.20, 12.35], pH, z_protonated=0)
        assert f.sum() == pytest.approx(1.0)
        assert len(f) == 4 and list(z) == [0, -1, -2, -3]


def test_clean_windows_bracket_the_pka():
    w = clean_windows([4.20], 0, purity=0.99)
    neutral = [x for x in w if x[2] == 0][0]
    anion = [x for x in w if x[2] == -1][0]
    assert neutral[1] < 4.20 < anion[0]
    # ~2 pH units either side
    assert 4.20 - neutral[1] == pytest.approx(2.0, abs=0.15)
    assert anion[0] - 4.20 == pytest.approx(2.0, abs=0.15)


def test_ms_mode_only_returns_volatile_buffers():
    for r in recommend([4.20], 0, ms=True):
        assert r["volatile"], r["buffer"]


def test_uv_mode_opens_up_phosphate():
    names = {r["buffer"] for r in recommend([7.0], 0, ms=False, max_results=12)}
    assert any("phosphate" in n for n in names)


def test_benzoic_acid_gets_low_pH_formate_neutral():
    """The textbook answer: ~0.1% formic acid, analyte neutral."""
    top = recommend([4.20], 0, ms=True)[0]
    assert "formic" in top["buffer"]
    assert top["analyte_charge"] == 0
    assert 2.0 <= top["pH"] <= 3.5


def test_every_recommendation_respects_the_column_range():
    for col, (lo, hi) in [("silica-C18", (2.0, 8.0)), ("hybrid-BEH", (1.0, 12.0))]:
        for r in recommend([4.2, 9.5], 1, ms=True, column=col, max_results=12):
            assert lo <= r["pH"] <= hi


def test_buffer_pH_is_within_capacity_of_its_own_pka():
    for r in recommend([4.20], 0, ms=True, max_results=12):
        assert abs(r["pH"] - r["buffer_pka"]) <= 1.0 + 1e-9


def test_purity_threshold_is_respected():
    for r in recommend([4.20], 0, ms=True, min_purity=0.99, max_results=12):
        assert r["purity"] >= 0.99


def test_zwitterion_is_flagged_and_demoted():
    plain = recommend([2.35, 9.78], 1, ms=True, column="hybrid-BEH")
    zwit = recommend([2.35, 9.78], 1, ms=True, column="hybrid-BEH", zwitterionic=True)
    neutral = [r for r in zwit if r["analyte_charge"] == 0]
    assert neutral, "expected a net-neutral option"
    assert any("ZWITTERION" in w for w in neutral[0]["warnings"])
    # the neutral state should not be rewarded as much as a true uncharged one
    assert zwit[0]["score"] < plain[0]["score"]


def test_report_suggests_a_wider_column_when_neutral_is_out_of_range():
    r = format_report([9.53], 1, ms=True, column="silica-C18")
    assert "hybrid-BEH" in r and "uncharged at pH" in r


def test_charge_label():
    assert charge_label(0) == "neutral"
    assert charge_label(-1) == "-1" and charge_label(2) == "+2"


def test_buffer_table_is_sane():
    for b in BUFFERS:
        assert b.pkas and all(-1 < p < 14 for p in b.pkas), b.name
        assert isinstance(b.volatile, bool)
