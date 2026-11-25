import numpy as np
import pandas as pd
import pytest

from analysis.guidance import extract_figures, extract_filing, fiscal_period_end
from analysis.valuation_model import guided_growth, held_guidance, model_params, run_model
from tests.test_valuation_model import make_close, make_quarters


def figures(sentence):
    return [(f["period"], f["metric"], round(f["low"], 2), round(f["high"], 2)) for f in extract_figures(sentence)]


def test_reads_the_common_ways_guidance_is_written():
    assert figures("[Outlook] Revenue is expected to be $130.0 billion, plus or minus 3%.") == \
        [("quarter", "revenue", 126.1e9, 133.9e9)]
    assert figures("[Guidance for Full Year and Fourth Quarter 2026] The company now estimates net sales growth for "
                   "the full year 2026, versus the prior year period, to be approximately 4.25 percent on a reported "
                   "basis and approximately 3.75 percent on an organic basis.") == [("year", "revenue_growth", 4.25, 4.25)]
    assert figures("[Guidance for Full Year and Third Quarter 2026] The company estimates net sales growth for the third "
                   "quarter of 2026, versus the prior year period, to be approximately 3 to 5 percent on a reported "
                   "and organic basis.") == [("quarter", "revenue_growth", 3.0, 5.0)]
    # A table line with the previous and the updated range: the updated one.
    assert figures("[2026 Financial Guidance] Revenue $85 to $87 billion $91 to $92 billion") == \
        [("year", "revenue", 91e9, 92e9)]
    # The unit stated once in the row label.
    assert figures("[Business Outlook] Revenue ($B) $15.0 - $16.0") == [("quarter", "revenue", 15e9, 16e9)]
    # "FY27 Q1" is a quarter of fiscal 2027, not the year.
    assert figures("[FY27 Q1 Adjusted Outlook] Total Company Revenue of $89.85 to $90.95 billion Productivity and "
                   "Business Processes Revenue of $36.7 to $37.0 billion")[0] == ("quarter", "revenue", 89.85e9, 90.95e9)


def test_leaves_out_results_and_comparisons_with_guidance():
    assert figures("[Outlook] Third-quarter revenue was $107.6 billion, up 21% from the previous quarter.") == []
    assert figures("Reported net sales of $5.442 billion, representing an increase of 7.5 percent on a reported basis, "
                   "compared to the company's guidance range of 5.5 to 7.5 percent.") == []
    assert figures("We are pleased to deliver fourth quarter non-GAAP revenues at the high end of our guidance, along "
                   "with 36% year-over-year growth.") == []


def test_one_row_per_period_and_metric_with_the_period_end():
    text = ("Outlook\nRevenue is expected to be $10.0 billion, plus or minus 2%.\n"
            "Revenue is expected to be $9.0 billion.\nAbout the company\n")
    rows = extract_filing(text, "2026-02-20", "0131", "XYZ", "0001")
    assert len(rows) == 1 and rows[0]["low"] == pytest.approx(9.8e9)
    assert rows[0]["period_end"] == pd.Timestamp("2026-04-30")
    assert fiscal_period_end("2026-02-20", "year", "0131") == pd.Timestamp("2027-01-31")
    assert fiscal_period_end("2026-10-28", "year", "1231") == pd.Timestamp("2026-12-31")


def guidance_rows(rows):
    return pd.DataFrame(rows, columns=["date", "period", "period_end", "metric", "low", "high"])


def test_guided_amounts_become_growth_against_the_year_before():
    quarters = make_quarters(count=12, growth=0.0)  # revenue 1000 a quarter
    release = quarters.index[-1]
    guided = guided_growth(guidance_rows([
        [release, "quarter", release + pd.Timedelta(days=60), "revenue", 1090.0, 1110.0],
        [release, "year", release + pd.Timedelta(days=150), "revenue", 4400.0, 4400.0]]), quarters)
    # The quarter wins over the year on the same day: 1100 against 1000 a year earlier.
    assert guided["growth"].iloc[-1] == pytest.approx(10.0) and guided["period"].iloc[-1] == "quarter"
    year_only = guided_growth(guidance_rows([[release, "year", release + pd.Timedelta(days=150), "revenue", 4400, 4400]]),
                              quarters)
    assert year_only["growth"].iloc[-1] == pytest.approx(10.0)
    # A figure that would mean growth beyond the plausible range is a misread and left out.
    assert guided_growth(guidance_rows([[release, "quarter", release + pd.Timedelta(days=60), "revenue", 30.0, 30.0]]),
                         quarters).empty


def test_guidance_is_held_until_the_next_release():
    releases = pd.DatetimeIndex(["2026-01-30", "2026-04-30", "2026-07-30"])
    days = pd.bdate_range("2026-01-01", "2026-08-31")
    guided = pd.DataFrame({"growth": [12.0, 8.0]}, index=pd.DatetimeIndex(["2026-01-30", "2026-03-15"]))
    held = held_guidance(guided, releases, days)
    assert np.isnan(held.loc["2026-01-29"])
    assert held.loc["2026-01-30"] == 12.0 and held.loc["2026-03-16"] == 8.0
    # The April release gave no guidance: from then on nothing is in force (the model holds the reported quarter).
    assert np.isnan(held.loc["2026-04-30"]) and np.isnan(held.loc["2026-08-31"])


def test_model_uses_guidance_in_force_and_holds_reported_growth_without_it():
    quarters = make_quarters(count=16, growth=0.0)
    close = make_close(quarters)
    release = quarters.index[-1]
    guidance = guidance_rows([[release, "quarter", release + pd.Timedelta(days=60), "revenue", 1200.0, 1200.0]])
    base = {"anchor_source": "Fundamentals", "peer_weight": 1.0, "pe_growth_latest_share": 1.0, "use_guidance": True}
    without = run_model(quarters, close, None, model_params(base))
    with_guidance = run_model(quarters, close, None, model_params(base), guidance=guidance)
    switched_off = run_model(quarters, close, None, model_params({**base, "use_guidance": False}), guidance=guidance)
    assert with_guidance["daily"]["held_guided_growth"].iloc[-1] == pytest.approx(20.0)
    assert with_guidance["daily"]["rule_growth"].iloc[-1] == pytest.approx(20.0, abs=0.5)
    assert without["daily"]["rule_growth"].iloc[-1] == pytest.approx(0.0, abs=0.5)
    assert switched_off["daily"]["rule_growth"].iloc[-1] == pytest.approx(0.0, abs=0.5)
    # Before the guidance day the reported growth is used.
    earlier = with_guidance["daily"].loc[: release - pd.Timedelta(days=1)]
    assert earlier["held_guided_growth"].isna().all()
