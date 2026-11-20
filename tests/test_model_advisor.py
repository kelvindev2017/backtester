import numpy as np
import pandas as pd
import pytest

from analysis.model_advisor import (
    fundamental_pe_with,
    growth_steadiness,
    market_beta,
    model_checks,
    steady_half_life,
    suggested_risk_premium,
    suggestions,
)
from analysis.valuation_model import latest_values, model_params, run_model
from tests.test_valuation_model import make_close, make_quarters


def steady_retailer():
    # Revenue up a steady ~8% a year, a 12% gross margin and a 3% net margin: a Costco-like business.
    quarters = make_quarters()
    quarters["Revenue"] = [1000 * 1.0194 ** step for step in range(len(quarters))]
    quarters["Gross_Profit"] = quarters["Revenue"] * 0.12
    quarters["Net_Income"] = quarters["Revenue"] * 0.03
    return quarters


def test_steady_growth_earns_a_longer_half_life_than_a_low_gross_margin_gives():
    params = model_params({})
    assert steady_half_life(2.0, 8.0, params) == pytest.approx(6.0)
    assert steady_half_life(15.0, 8.0, params) == pytest.approx(params["growth_half_life_min"])
    assert np.isnan(steady_half_life(1.0, -3.0, params)) and np.isnan(steady_half_life(np.nan, 8.0, params))
    quarters = steady_retailer()
    close = make_close(quarters)
    result = run_model(quarters, close, None, model_params({"anchor_source": "Fundamentals", "peer_weight": 1.0}))
    assert growth_steadiness(result["quarterly"]) < 2.0
    proposed = suggestions(result).set_index("Parameter")
    assert proposed.loc["forward_half_life", "Suggested"] == pytest.approx(6.0)
    assert "steady" in proposed.loc["forward_half_life", "Why"]
    # The default bear case (-3 points) would take all of a 3% margin: the suggestion keeps it to a third.
    assert proposed.loc["bear_margin_pts", "Suggested"] == pytest.approx(-1.0)
    checks = model_checks(result)
    assert checks["Level"].iloc[0] == "Fix"
    assert {"Scenario engine", "Multiple builder"} <= set(checks["Where"])
    # Applying the half-life lifts the fundamental P/E.
    before = latest_values(result)["outside_forward"]
    assert fundamental_pe_with(result, {"forward_half_life": 6.0}) > before


def test_beta_sets_the_risk_premium():
    days = pd.bdate_range("2022-01-03", "2025-12-31")
    rng = np.random.default_rng(1)
    market = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))), index=days)
    calm = pd.Series(50 * np.exp(0.5 * np.log(market / 100)), index=days)
    assert market_beta(calm, market) == pytest.approx(0.5, abs=0.02)
    assert np.isnan(market_beta(calm.tail(100), market.tail(100)))
    params = model_params({})
    assert suggested_risk_premium(0.5, params) == pytest.approx(2.5)
    assert suggested_risk_premium(1.2, params) == pytest.approx(5.4)
    quarters = steady_retailer()
    close = make_close(quarters)
    result = run_model(quarters, close, None, model_params({"anchor_source": "Fundamentals", "peer_weight": 1.0}))
    proposed = suggestions(result, beta=0.5).set_index("Parameter")
    assert proposed.loc["forward_equity_premium", "Suggested"] == pytest.approx(2.5)
    assert any("Beta" in what for what in model_checks(result, beta=0.5)["What"])


def test_mismatched_peers_and_missing_data_are_flagged():
    quarters = make_quarters()
    close = make_close(quarters)
    peers = pd.DataFrame({"GM": 6.0, "F": 7.0}, index=close.index)
    result = run_model(quarters, close, None, model_params({"anchor_source": "Peers", "peer_weight": 0.5,
                                                            "pe_basis": "Trailing"}), peers)
    own = latest_values(result)["own_anchor"]
    checks = model_checks(result)
    if own / 6.5 > 3:
        assert any("peers' P/E" in what for what in checks["What"])
    no_revenue = run_model(quarters[["EPS"]], close, None, model_params({}), peers)
    assert any("fallback" in what for what in model_checks(no_revenue)["What"])
