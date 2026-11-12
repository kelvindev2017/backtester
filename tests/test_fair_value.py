"""Fair value method, checked against the ASMPT worked example (HK$, 2027 scenarios)."""

import numpy as np
import pandas as pd
import pytest

from analysis.fair_value import (
    acceptable_multiple,
    common_size,
    default_risks,
    default_scenarios,
    earnings_quality,
    earnings_risks,
    implied_bear_probability,
    market_check,
    multiple_deductions,
    operating_leverage,
    period_frame,
    run_rate_eps,
    run_rate_pe,
    scenario_earnings,
    sensitivity,
    suggested_deductions,
    triage,
    valuation,
    weighted,
    yoy_growth,
)

EPS = "EPS after share-based pay"


def asmpt_scenarios():
    """The worked example: EPS after share-based pay 3.6 / 7.4 / 9.7 at 25/50/25, multiples 14x / 17x / 22x."""

    return pd.DataFrame({"Probability %": [25.0, 50.0, 25.0], EPS: [3.6, 7.4, 9.7],
                         "Multiple +/-": [-3.0, 0.0, 5.0]}, index=pd.Index(["Bear", "Base", "Bull"], name="Scenario"))


def quarters(n=8):
    ends = pd.date_range("2024-03-31", periods=n, freq="QE")
    revenue = np.linspace(1000, 1700, n)
    income = pd.DataFrame({
        "Total Revenue": revenue, "Cost Of Revenue": revenue * 0.58, "Gross Profit": revenue * 0.42,
        "Research And Development": revenue * 0.11, "Selling General And Administration": revenue * 0.16,
        "Operating Income": revenue * 0.15, "Pretax Income": revenue * 0.15, "Tax Provision": revenue * 0.15 * 0.31,
        "Net Income": revenue * 0.10, "Normalized Income": revenue * 0.11,
        "Total Unusual Items": [0, -20, 0, -15, 0, -30, -12, 0][:n], "EBITDA": revenue * 0.19,
        "Diluted EPS": revenue * 0.10 / 418, "Diluted Average Shares": [418.0] * n,
    }, index=ends).T
    cash = pd.DataFrame({"Stock Based Compensation": revenue * 0.012, "Depreciation And Amortization": revenue * 0.04,
                         "Operating Cash Flow": revenue * 0.12}, index=ends).T
    # Yahoo lists the newest period first.
    return income[income.columns[::-1]], cash[cash.columns[::-1]]


def test_risk_adjusted_value_matches_the_worked_example():
    scenarios = asmpt_scenarios()
    assert weighted(scenarios, EPS) == pytest.approx(7.025)
    table, risk_adjusted = valuation(scenarios, acceptable_multiple(23.5, 3 + 2 + 1))
    assert list(table["Multiple"]) == [14.5, 17.5, 22.5]
    table, risk_adjusted = valuation(scenarios, 17.0)
    assert list(table["Multiple"]) == [14.0, 17.0, 22.0]
    assert table["Value per share"].round(0).tolist() == [50.0, 126.0, 213.0]
    assert risk_adjusted["Value per share"] == pytest.approx(119.4, abs=0.1)
    # Per ASMVY ADR: three shares at HK$7.84 to the dollar gives about US$45.7.
    _, adr = valuation(scenarios, 17.0, per_share=3 / 7.84)
    assert adr["Value per share"] == pytest.approx(45.7, abs=0.1)


def test_market_check_and_sensitivity_match_the_worked_example():
    scenarios = asmpt_scenarios()
    check = market_check(scenarios, 98.0, 17.0)
    assert check["Implied EPS"] == pytest.approx(5.76, abs=0.01)
    assert check["Implied multiple"] == pytest.approx(13.95, abs=0.01)
    # "The price fits my scenarios only if the bear case is about 50% likely."
    assert check["Implied bear probability"] == pytest.approx(0.53, abs=0.01)
    assert check["Value vs price %"] == pytest.approx(21.9, abs=0.1)
    moves = sensitivity(scenarios, 17.0)
    assert moves["Per 1x P/E"] == pytest.approx(7.0, abs=0.05)
    assert moves["Per 10 points to bear"] == pytest.approx(-7.8, abs=0.05)
    # A price above the bull case cannot be reached by any bear probability.
    assert implied_bear_probability(scenarios, 500.0, 17.0) is None


def test_scenario_earnings_build_eps_from_revenue_and_margin():
    scenarios = pd.DataFrame({"Probability %": [25.0, 50.0, 25.0], "Revenue growth %": [0.0, 26.0, 44.0],
                              "Operating margin %": [15.0, 21.0, 24.0], "Multiple +/-": [-3.0, 0.0, 5.0]},
                             index=["Bear", "Base", "Bull"])
    table = scenario_earnings(scenarios, base_revenue=1000.0, shares=10.0, net_share_pct=75.0, share_based_pay=20.0)
    assert table.at["Base", "Revenue"] == pytest.approx(1260.0)
    assert table.at["Base", "Operating profit"] == pytest.approx(264.6)
    assert table.at["Base", "Adjusted EPS"] == pytest.approx(264.6 * 0.75 / 10)
    assert table.at["Base", EPS] == pytest.approx((264.6 * 0.75 - 20) / 10)
    assert scenario_earnings(scenarios, 1000.0, 0, 75.0)[EPS].isna().all()


def test_statements_become_common_size_and_quality_figures():
    income, cash = quarters()
    frame = period_frame(income, cash)
    assert frame.index.is_monotonic_increasing and len(frame) == 8
    shares = common_size(frame)
    assert shares["Cost of sales"].iloc[-1] == pytest.approx(58.0)
    assert shares["Operating margin"].iloc[-1] == pytest.approx(15.0)
    assert shares["Tax rate"].iloc[-1] == pytest.approx(31.0)
    assert len(yoy_growth(frame["Revenue"])) == 4
    # Revenue and operating profit grow together at a fixed margin, so leverage is 1.
    assert operating_leverage(frame) == pytest.approx(1.0)
    quality = earnings_quality(frame)
    revenue_ttm = frame["Revenue"].iloc[-4:].sum()
    assert quality["Clean net income"] == pytest.approx(revenue_ttm * (0.11 - 0.012))
    assert quality["One-off periods"] == 2
    assert any("recur" in flag for flag in quality["Flags"])
    assert any("Tax took 31%" in flag for flag in quality["Flags"])
    assert run_rate_eps(frame) == pytest.approx(1700 * 0.11 / 418 * 4)
    assert run_rate_pe(10.0, -1.0) is None


def test_default_scenarios_and_risk_register():
    income, cash = quarters()
    frame = period_frame(income, cash)
    scenarios = default_scenarios(frame, analyst_growth=20.0)
    assert list(scenarios.index) == ["Bear", "Base", "Bull"]
    assert scenarios["Probability %"].tolist() == [25.0, 50.0, 25.0]
    assert scenarios.at["Base", "Revenue growth %"] == 20.0
    assert scenarios.at["Bull", "Revenue growth %"] == 34.0
    assert scenarios["Multiple +/-"].tolist() == [-3.0, 0.0, 5.0]
    # Without analysts: half the latest year-on-year growth.
    assert default_scenarios(frame).at["Base", "Revenue growth %"] == pytest.approx(
        round(yoy_growth(frame["Revenue"]).iloc[-1] / 2, 1))

    annual = pd.DataFrame({pd.Timestamp("2022-12-31"): [100.0], pd.Timestamp("2023-12-31"): [120.0],
                           pd.Timestamp("2024-12-31"): [110.0]}, index=["Normalized Income"])
    quality = earnings_quality(frame)
    suggestions = suggested_deductions(frame, quality, annual=annual, foreign_listing=True)
    names = [item["Risk"] for item in suggestions]
    assert names == ["Earnings may be near the top of the cycle", "Earnings quality", "Listing and liquidity"]
    risks = default_risks(suggestions)
    risks.loc[len(risks)] = ["HBM delays", "Korea revenue down 64%", "Earnings: bear and base", "High", 0.0]
    risks.loc[len(risks)] = ["Minor FX", "", "Multiple", "Low", 0.5]
    active, parked = triage(risks, "Medium")
    assert parked["Risk"].tolist() == ["Minor FX"]
    assert multiple_deductions(active) == pytest.approx(3 + 2 + 1)
    assert earnings_risks(active, "Bear") == ["HBM delays"]
    assert earnings_risks(active, "Base") == ["HBM delays"]
    # A high bar parks the medium risks too.
    active, parked = triage(risks, "High")
    assert multiple_deductions(active) == pytest.approx(3)


def test_foreign_filer_with_empty_quarters_keeps_reported_periods():
    income, cash = quarters(5)
    income.loc["Total Revenue", income.columns[:3]] = np.nan
    income.loc["Diluted EPS", income.columns[:2]] = np.nan
    frame = period_frame(income, cash)
    assert len(frame) == 3
    assert earnings_quality(frame)["Reported net income"] is None


def test_adrs_use_their_home_listing_and_hong_kong_quarters_give_a_run_rate():
    from analysis.fair_value import analyst_operating_margin, home_listing

    assert home_listing("asmvy") == ("0522.HK", 3.0)
    assert home_listing("BESIY") == ("BESI.AS", 1.0)
    assert home_listing("NVDA") is None
    # Yahoo's Hong Kong quarters carry EPS and shares but no revenue: the newest EPS sets the run rate.
    income, cash = quarters(5)
    income.loc["Total Revenue", income.columns[:3]] = np.nan
    income.loc["Normalized Income", income.columns[:3]] = np.nan
    income.loc["Diluted EPS", income.columns[0]] = 1.27
    assert run_rate_eps(period_frame(income, cash)) == pytest.approx(5.08)
    # Analysts' HK$6.06 EPS on 419M shares and HK$23.0B revenue is a 11% net margin, 20% operating at 55%.
    margin = analyst_operating_margin(23.0e9, 6.06, 419e6, 55.0)
    assert margin == pytest.approx(20.07, abs=0.01)
    assert analyst_operating_margin(None, 6.06, 419e6, 55.0) is None
    scenarios = default_scenarios(period_frame(*quarters()), analyst_growth=43.8, base_margin=margin)
    assert scenarios["Operating margin %"].tolist() == [15.1, 20.1, 23.1]
