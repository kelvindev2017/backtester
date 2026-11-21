import numpy as np
import pandas as pd
import pytest

from analysis.model_advisor import growth_steadiness, market_beta, model_checks, suggested_params, suggestions
from analysis.valuation_model import latest_values, model_params, run_model
from tests.test_valuation_model import make_close, make_quarters


def steady_retailer():
    # Revenue up a steady ~8% a year, a 12% gross margin and a 3% net margin: a Costco-like business.
    quarters = make_quarters()
    quarters["Revenue"] = [1000 * 1.0194 ** step for step in range(len(quarters))]
    quarters["Gross_Profit"] = quarters["Revenue"] * 0.12
    quarters["Net_Income"] = quarters["Revenue"] * 0.03
    return quarters


def test_suggestions_turn_on_the_rules_for_every_quarter_and_fix_the_bear_case():
    quarters = steady_retailer()
    close = make_close(quarters)
    off = {"anchor_source": "Fundamentals", "peer_weight": 1.0, "half_life_from_steadiness": False,
           "erp_from_beta": False}
    result = run_model(quarters, close, None, model_params(off))
    assert growth_steadiness(result["quarterly"]) < 2.0
    proposed = suggestions(result, beta=0.5).set_index("Parameter")
    assert proposed.loc["half_life_from_steadiness", "Suggested"] == 1.0
    assert proposed.loc["half_life_from_steadiness", "Applies to"] == "every quarter"
    assert proposed.loc["erp_from_beta", "Suggested"] == 1.0
    # The default bear case (-3 points) would take all of a 3% margin: the suggestion keeps it to a third.
    assert proposed.loc["bear_margin_pts", "Suggested"] == pytest.approx(-1.0)
    checks = model_checks(result, beta=0.5)
    assert checks["Level"].iloc[0] == "Fix" and {"Scenario engine", "Multiple builder"} <= set(checks["Where"])
    # Applied, the rules lift the P/E on every day they are known, not only from the latest release.
    applied = run_model(quarters, close, None, suggested_params(result["params"], proposed.reset_index()))
    assert applied["params"]["half_life_from_steadiness"] is True
    known = applied["daily"]["held_growth_spread"].notna()
    assert known.sum() > 50
    assert (applied["daily"].loc[known, "fundamental_pe"] > result["daily"].loc[known, "fundamental_pe"]).all()
    # With the rule on there is nothing left to suggest for it.
    assert "half_life_from_steadiness" not in suggestions(applied, beta=0.5)["Parameter"].tolist()


def test_analysts_far_from_the_history_get_a_forward_suggestion():
    quarters = steady_retailer()
    result = run_model(quarters, make_close(quarters), None, model_params({"anchor_source": "Fundamentals"}))
    growth = latest_values(result)["pe_growth"]
    proposed = suggestions(result, analyst_growth=growth + 20).set_index("Parameter")
    assert proposed.loc["forward_pe_growth", "Suggested"] == pytest.approx(round(growth + 10, 1))
    assert proposed.loc["forward_pe_growth", "Applies to"] == "from the latest release"


def test_beta_and_missing_data():
    days = pd.bdate_range("2022-01-03", "2025-12-31")
    rng = np.random.default_rng(1)
    market = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))), index=days)
    calm = pd.Series(50 * np.exp(0.5 * np.log(market / 100)), index=days)
    assert market_beta(calm, market) == pytest.approx(0.5, abs=0.02)
    assert np.isnan(market_beta(calm.tail(100), market.tail(100)))
    quarters = make_quarters()
    no_revenue = run_model(quarters[["EPS"]], make_close(quarters), None, model_params({}))
    assert any("fallback" in what for what in model_checks(no_revenue)["What"])
