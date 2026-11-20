import numpy as np
import pandas as pd
import pytest

from analysis.valuation_model import (
    BLOCKS, MODELLED, SIGNALS, TO_MULTIPLE, TO_SCENARIOS, clean_register, implied_dials, item_effects, latest_values,
    model_params, preset_params, preset_register, quarterly_inputs, register_changes, route_items, run_model,
    scenario_point, scenario_weights, sensitivities, track_items,
)


def make_quarters(count=14, growth=0.20, margin=0.20, shares=100.0, end="2026-10-28"):
    """Quarterly results by release date: revenue up ``growth`` a year, a steady net margin."""

    dates = pd.date_range(end=end, periods=count, freq="91D")
    revenue = 1000.0 * (1 + growth) ** (np.arange(count) / 4)
    net_income = revenue * margin
    return pd.DataFrame({"Revenue": revenue, "Net_Income": net_income, "EPS": net_income / shares}, index=dates)


def make_close(quarters, pe=20.0, end="2026-11-10"):
    days = pd.bdate_range(quarters.index[0], end)
    ttm = quarters["EPS"].rolling(4).sum()
    held = ttm.reindex(ttm.index.union(days)).ffill().reindex(days)
    return (held * pe).fillna(quarters["EPS"].iloc[0] * 4 * pe)


def register(rows):
    return pd.DataFrame(rows, columns=["Item", "Type", "Kind", "Timing", "Probability %", "EPS impact %", "P/E points",
                                       "Status", "Since", "Keywords"])


def test_scenario_point_grows_eps_with_revenue_and_scales_it_by_margin():
    point = scenario_point(4.0, 20.0, 20.0, model_params())
    # Base growth is half the trailing 20%; bear is 25 points below it and loses 3 margin points of 20.
    assert point["growth_base"] == pytest.approx(10.0) and point["growth_bear"] == pytest.approx(-15.0)
    assert point["eps_bear"] == pytest.approx(4.0 * 0.85 * 17 / 20)
    assert point["eps_base"] == pytest.approx(4.4)
    assert point["eps_bull"] == pytest.approx(4.0 * 1.25 * 22 / 20)
    assert point["risk_adjusted_eps"] == pytest.approx(0.25 * 2.89 + 0.5 * 4.4 + 0.25 * 5.5)


def test_haircut_events_and_fallbacks():
    params = model_params({"eps_haircut_pct": 10.0, "fallback_yoy": 30.0, "fallback_margin": 12.0})
    point = scenario_point(4.0, np.nan, np.nan, params, event_effect=-0.05)
    assert point["clean_eps"] == pytest.approx(3.6)
    assert point["growth_base"] == pytest.approx(15.0)
    assert point["eps_base"] == pytest.approx(3.6 * 1.15 * 0.95)
    # Growth is capped, and weights are normalised.
    assert scenario_point(4.0, 500.0, 20.0, model_params())["growth_base"] == pytest.approx(40.0)
    assert scenario_weights(model_params({"weight_bear": 1, "weight_base": 1, "weight_bull": 2})).tolist() == [0.25, 0.25, 0.5]
    assert scenario_weights(model_params({"weight_bear": 0, "weight_base": 0, "weight_bull": 0})).sum() == pytest.approx(1)


def test_each_item_enters_once_through_the_gate_and_router():
    routed = route_items(register([
        ["Export rules", "Risk", "Event", "In horizon", 20, 15, None, "Modelled", "", ""],
        ["Cyclicality", "risk", "general", "Beyond horizon", None, None, 3, "Modelled", "", ""],
        ["New product", "Product", "Event", "Beyond horizon", 50, None, 2, "Modelled", "", ""],
        ["Tiny risk", "Risk", "Event", "In horizon", 5, 2, None, "Modelled", "", ""],
        ["Unsized", "Risk", "Event", "In horizon", None, None, None, "Modelled", "", ""],
        ["Watching", "Risk", "General", "In horizon", None, None, 4, "Parked", "", ""],
    ]))
    by_item = routed.set_index("Item")
    assert by_item.at["Export rules", "Route"] == TO_SCENARIOS and by_item.at["Export rules", "Effect"] == pytest.approx(-3.0)
    assert by_item.at["Cyclicality", "Route"] == TO_MULTIPLE and by_item.at["Cyclicality", "Effect"] == pytest.approx(-3.0)
    # A product beyond the horizon lifts the multiple, weighted by its probability.
    assert by_item.at["New product", "Route"] == TO_MULTIPLE and by_item.at["New product", "Effect"] == pytest.approx(1.0)
    assert by_item.at["Tiny risk", "Gate"].startswith("Parked: below the materiality bar")
    assert by_item.at["Unsized", "Gate"].startswith("Parked: not sized")
    assert by_item.at["Watching", "Gate"] == "Parked by you"
    events, points = item_effects(routed)
    assert events == pytest.approx(-0.03) and points == pytest.approx(-2.0)


def test_income_map_uses_trailing_quarters():
    inputs = quarterly_inputs(make_quarters())
    assert inputs["revenue_yoy"].iloc[-1] == pytest.approx(20.0)
    assert inputs["net_margin"].iloc[-1] == pytest.approx(20.0)
    assert inputs["ttm_eps"].iloc[-1] == pytest.approx(inputs["eps"].iloc[-4:].sum())
    # With five quarters there is no trailing-year growth yet: the latest quarter against a year earlier.
    assert quarterly_inputs(make_quarters(5))["revenue_yoy"].iloc[-1] == pytest.approx(20.0)
    assert quarterly_inputs(pd.DataFrame()).empty


def test_model_price_is_held_eps_times_acceptable_pe():
    quarters = make_quarters()
    close = make_close(quarters)
    peers = pd.DataFrame({"AAA": 24.0, "BBB": 16.0, "CCC": 20.0}, index=close.index)
    items = register([["Cyclicality", "Risk", "General", "Beyond horizon", None, None, 3, "Modelled", "", ""]])
    result = run_model(quarters, close, items, model_params({"peer_weight": 1.0}), peers)
    daily, quarterly = result["daily"], result["quarterly"]
    last = daily.iloc[-1]
    assert last["peer_anchor"] == pytest.approx(20.0) and last["acceptable_pe"] == pytest.approx(17.0)
    assert last["held_eps"] == pytest.approx(quarterly["risk_adjusted_eps"].iloc[-1])
    assert last["model_price"] == pytest.approx(last["held_eps"] * 17.0)
    assert last["residual_pct"] == pytest.approx((last["model_price"] / last["close"] - 1) * 100)
    # Between releases the quarterly outputs are held.
    release = quarterly.index[-2]
    held = daily.loc[release:quarterly.index[-1] - pd.Timedelta(days=1), "held_eps"]
    assert held.nunique() == 1 and held.iloc[0] == pytest.approx(quarterly.at[release, "risk_adjusted_eps"])
    # Every signal a block reads or writes exists in the result.
    for block in BLOCKS.values():
        for name in block["inputs"] + block["outputs"]:
            assert name in result[SIGNALS[name][0]], name


def test_nothing_uses_data_from_later():
    quarters = make_quarters()
    close = make_close(quarters)
    full = run_model(quarters, close, None, model_params())
    cut = quarters.index[-3]
    early = run_model(quarters.loc[:cut], close.loc[:cut + pd.Timedelta(days=30)], None, model_params())
    shared = early["daily"].index
    pd.testing.assert_series_equal(full["daily"].loc[shared, "model_price"], early["daily"]["model_price"])
    # The own anchor ends the day before, so today's close does not set today's multiple.
    spiked = close.copy()
    spiked.iloc[-1] *= 3
    assert run_model(quarters, spiked, None, model_params())["daily"]["acceptable_pe"].iloc[-1] == pytest.approx(
        full["daily"]["acceptable_pe"].iloc[-1])


def test_an_item_starts_on_its_date_between_releases():
    quarters = make_quarters()
    close = make_close(quarters)
    since = quarters.index[-1] + pd.Timedelta(days=5)
    items = register([["Export rules", "Risk", "Event", "In horizon", 50, 20, None, "Modelled", f"{since:%Y-%m-%d}", ""]])
    result = run_model(quarters, close, items, model_params())
    plain = run_model(quarters, close, None, model_params())
    assert since in result["quarterly"].index
    before, after = since - pd.Timedelta(days=1), close.index[-1]
    assert result["daily"].loc[:before, "held_eps"].iloc[-1] == pytest.approx(plain["daily"].loc[:before, "held_eps"].iloc[-1])
    assert result["daily"].at[after, "held_eps"] == pytest.approx(plain["daily"].at[after, "held_eps"] * 0.9)
    assert result["quarterly"].at[since, "items_modelled"] == 1


def test_implied_dials_match_the_market_price():
    quarters = make_quarters()
    close = make_close(quarters, pe=14.0)
    peers = pd.DataFrame({"AAA": 20.0}, index=close.index)
    items = register([
        ["Cyclicality", "Risk", "General", "Beyond horizon", None, None, 2, "Modelled", "", ""],
        ["Export rules", "Risk", "Event", "In horizon", 20, 60, None, "Parked", "", ""],
        ["Small risk", "Risk", "Event", "In horizon", 20, 5, None, "Parked", "", ""],
        ["New product", "Product", "Event", "In horizon", 30, 10, None, "Parked", "", ""],
    ])
    params = model_params({"peer_weight": 1.0})
    result = run_model(quarters, close, items, params, peers)
    values = latest_values(result)
    assert values["model_price"] > values["close"]
    dials = implied_dials(result).set_index("Dial")

    # Re-running the model with the needed bear weight lands on the close.
    needed = dials.at["Bear weight", "Needed"]
    rest = (100 - needed) / 75
    again = run_model(quarters, close, items, {**params, "weight_bear": needed, "weight_base": 50 * rest,
                                               "weight_bull": 25 * rest}, peers)
    assert latest_values(again)["model_price"] == pytest.approx(values["close"], rel=1e-6)
    assert dials.at["Acceptable P/E", "Needed"] == pytest.approx(values["close"] / values["held_eps"], rel=1e-6)
    assert dials.at["Acceptable P/E", "Now"] == pytest.approx(18.0)
    # The modelled general risk needs more points; the parked risk could explain the gap, the parked product cannot.
    assert dials.at["Cyclicality: P/E points", "Needed"] == pytest.approx(2 + 18.0 - values["close"] / values["held_eps"], rel=1e-6)
    assert 0 < dials.at["Export rules (parked): probability", "Needed"] < 100
    # A risk that removes 5% of EPS at most cannot close the gap however likely it is.
    assert dials.at["Small risk (parked): probability", "Read"] == "Cannot match alone"
    assert dials.at["New product (parked): probability", "Read"] == "Wrong direction"
    assert set(dials["Read"]) <= {"Plausible", "A stretch", "Cannot match alone", "Wrong direction"}
    steps = sensitivities(result)
    assert steps["10 points more bear weight"] < 0 < steps["1 point more growth"]
    assert steps["1x more P/E"] == pytest.approx(values["held_eps"])


def test_item_tracker_counts_mentions_and_collects_unexplained_stories():
    now = pd.Timestamp("2026-11-10", tz="UTC")
    news = pd.DataFrame({
        "title": ["New export rules hit chip tools", "Company wins big optics order", "Shares drift", "Plant fire halts output"],
        "summary": ["", "", "", ""],
        "published": [now - pd.Timedelta(days=2), now - pd.Timedelta(days=1), now, now - pd.Timedelta(days=3)],
        "tone": ["Negative", "Positive", "Neutral", "Negative"], "publisher": "Wire", "url": "https://example.com",
    })
    items = register([
        ["Export rules", "Risk", "Event", "In horizon", 20, 15, None, "Modelled", "", "export, sanction"],
        ["HBM delays", "Risk", "Event", "In horizon", 30, 8, None, "Modelled", "", "hbm"],
        ["Optics", "Product", "Event", "Beyond horizon", 50, None, 1, "Parked", "", "optics"],
    ])
    tracked, candidates = track_items(items, news, now=now)
    by_item = tracked.set_index("Item")
    assert by_item["Mentions"].to_dict() == {"Export rules": 1, "HBM delays": 0, "Optics": 1}
    assert by_item["Stale"].to_dict() == {"Export rules": False, "HBM delays": True, "Optics": False}
    assert candidates["title"].tolist() == ["Plant fire halts output"]
    # Without stories nothing is called stale.
    assert not track_items(items)[0]["Stale"].any() and track_items(items)[1].empty


def test_asmvy_preset_runs_on_a_typed_in_eps():
    params, items = preset_params("asmvy"), preset_register("ASMVY")
    assert params["manual_eps"] == pytest.approx(1.64) and params["weight_base"] == 50
    routed = route_items(items, params)
    assert (routed["Gate"] == MODELLED).sum() == 3 and item_effects(routed)[1] == pytest.approx(-6.0)
    days = pd.bdate_range("2026-01-02", "2026-11-10")
    close = pd.Series(np.linspace(30, 37.5, len(days)), index=days)
    peers = pd.DataFrame({"KLIC": 14.4, "CAMT": 26.5, "AMAT": 26.5}, index=days)
    result = run_model(pd.DataFrame(), close, items, params, peers)
    values = latest_values(result)
    # 1.64 less 11%, then bear 0% / base 28% / bull 44% growth on margins of 8.8 / 13.8 / 16.8 against 9.8.
    clean = 1.64 * 0.89
    expected = 0.25 * clean * 8.8 / 9.8 + 0.5 * clean * 1.28 * 13.8 / 9.8 + 0.25 * clean * 1.44 * 16.8 / 9.8
    assert values["risk_adjusted_eps"] == pytest.approx(expected)
    assert values["acceptable_pe"] == pytest.approx(26.5 - 6) and values["model_price"] == pytest.approx(expected * 20.5)
    assert preset_register("NVDA").empty and preset_params("NVDA") == model_params()


def test_register_is_cleaned_and_changes_are_listed():
    before = clean_register(register([
        ["Export rules", "Risk", "Event", "In horizon", 20, 15, None, "Parked", "", ""],
        ["", "Risk", "Event", "In horizon", 20, 15, None, "Parked", "", ""],
        ["Old item", "weird", "General", "In horizon", None, None, -2, "Modelled", "not a date", ""],
    ]))
    assert before["Item"].tolist() == ["Export rules", "Old item"]
    assert before.at[1, "Type"] == "Risk" and before.at[1, "P/E points"] == 2 and before.at[1, "Since"] == ""
    after = before.copy()
    after.at[0, "Probability %"], after.at[0, "Status"] = 35.0, "Modelled"
    after = pd.concat([after.iloc[:1], clean_register(register([
        ["HBM delays", "Risk", "Event", "In horizon", 30, 8, None, "Parked", "", ""]]))], ignore_index=True)
    assert register_changes(before, after) == [
        "Added: HBM delays", "Removed: Old item", "Export rules: Probability % 20 -> 35",
        "Export rules: Status Parked -> Modelled"]
    assert register_changes(before, before) == []


def test_forward_basis_anchors_on_the_given_own_pe():
    quarters = make_quarters()
    close = make_close(quarters)
    trailing = run_model(quarters, close, None, model_params({"peer_weight": 0.0}))
    own_forward = pd.Series(15.0, index=close.index)
    forward = run_model(quarters, close, None, model_params({"peer_weight": 0.0}),
                        own_pe=own_forward)
    assert trailing["daily"]["own_anchor"].iloc[-1] == pytest.approx(20.0)
    assert forward["daily"]["own_anchor"].iloc[-1] == pytest.approx(15.0)
    # Same EPS, lower multiple: the price scales with the anchor.
    last_t, last_f = trailing["daily"].iloc[-1], forward["daily"].iloc[-1]
    assert last_f["held_eps"] == pytest.approx(last_t["held_eps"])
    assert last_f["model_price"] == pytest.approx(last_t["model_price"] * 15 / 20)


def test_earnings_quality_takes_points_off_the_acceptable_pe():
    quarters = make_quarters()
    close = make_close(quarters)
    plain = run_model(quarters, close, None, model_params())
    weak = run_model(quarters, close, None, model_params({"quality_pe_pts": 3.0}))
    before, after = plain["daily"].iloc[-1], weak["daily"].iloc[-1]
    assert after["held_quality_pe"] == pytest.approx(-3.0)
    assert after["acceptable_pe"] == pytest.approx(before["acceptable_pe"] - 3.0)
    assert after["model_price"] == pytest.approx(after["held_eps"] * after["acceptable_pe"])
    dials = implied_dials(weak).set_index("Dial")
    assert dials.at["Acceptable P/E", "Now"] == pytest.approx(after["acceptable_pe"])


def test_stock_pay_haircut_uses_the_last_quarters_of_both():
    from analysis.model_equations import stock_pay_haircut
    dates = pd.date_range("2025-03-31", periods=5, freq="QE")
    pay = pd.Series([9.0, 1.0, 1.0, 1.0, 1.0], index=dates)
    income = pd.Series([1.0, 10.0, 10.0, 10.0, 10.0], index=dates)
    # Last four quarters: pay 4, net income 40, so 4 / (40 + 4) of adjusted earnings.
    assert stock_pay_haircut(pay, income) == pytest.approx(4 / 44 * 100)
    assert np.isnan(stock_pay_haircut(pay, -income))
    assert np.isnan(stock_pay_haircut(pay.iloc[:2], income))
    assert stock_pay_haircut(pay * 100, income) == 90
    # Stock pay as large as GAAP profit (Tesla, Q3 2026) takes off half of adjusted EPS, not all of it.
    assert stock_pay_haircut(pay.iloc[1:], income.iloc[1:] / 10) == pytest.approx(50.0)


def test_worked_equations_end_in_the_model_values():
    from analysis.model_equations import EQUATIONS, worked_equations
    quarters = make_quarters()
    close = make_close(quarters)
    peers = pd.DataFrame({"AAA": 24.0, "BBB": 16.0}, index=close.index)
    result = run_model(quarters, close, None, model_params({"eps_haircut_pct": 5.0, "quality_pe_pts": 2.0}), peers)
    values = latest_values(result)
    for block in BLOCKS:
        assert EQUATIONS[block]
    assert f"& {values['risk_adjusted_eps']:,.2f} \\end{{aligned}}" in worked_equations(result, "risk_weighting")[0]
    assert worked_equations(result, "product")[0].endswith(f"{values['model_price']:,.2f}")
    assert f"& {values['acceptable_pe']:,.1f} \\end{{aligned}}" in worked_equations(result, "multiple_builder")[1]
    for name in ["bear", "base", "bull"]:
        line = [eq for eq in worked_equations(result, "scenario_engine") if rf"\text{{EPS}}_{{\text{{{name}}}}}" in eq][0]
        assert f"& {values[f'eps_{name}']:,.2f} \\end{{aligned}}" in line


def test_rate_model_turns_a_policy_rate_into_a_yield_and_a_pe():
    from analysis.valuation_model import model_yield, rate_adjusted_pe, rate_weight
    # A 10-year yield with a 2-year half-life carries about 28% of a policy rate change; a 0-year one all of it.
    assert rate_weight(10, 2) == pytest.approx((1 - np.exp(-np.log(2) * 5)) / (np.log(2) * 5))
    assert rate_weight(10, 2) == pytest.approx(0.2795, abs=1e-4)
    assert rate_weight(0, 2) == 1.0 and rate_weight(1, 1000) == pytest.approx(1.0, abs=1e-3)
    params = model_params({"neutral_rate": 3.0, "term_premium": 0.5})
    assert model_yield(3.0, params) == pytest.approx(3.5)
    assert model_yield(5.0, params) - model_yield(4.5, params) == pytest.approx(0.5 * rate_weight(10, 2))
    # The earnings yield moves point for point: 25x is a 4% earnings yield, +0.15 points makes it 4.15%.
    assert rate_adjusted_pe(25.0, 0.15) == pytest.approx(1 / 0.0415)
    assert rate_adjusted_pe(25.0, 0.15, pass_through=0) == pytest.approx(25.0)
    assert rate_adjusted_pe(25.0, -5.0) == np.inf and np.isnan(rate_adjusted_pe(np.nan, 0.1))
    series = rate_adjusted_pe(pd.Series([20.0, np.nan]), pd.Series([1.0, 1.0]))
    assert series.iloc[0] == pytest.approx(1 / 0.06) and np.isnan(series.iloc[1])


def rate_run(policy_rate, **overrides):
    quarters = make_quarters()
    close = make_close(quarters)
    peers = pd.DataFrame({"AAA": 24.0, "BBB": 16.0}, index=close.index)
    rates = pd.Series(policy_rate, index=close.index) if np.ndim(policy_rate) == 0 else policy_rate
    return run_model(quarters, close, None, model_params(overrides), peers, policy_rate=rates,
                     market_yield=pd.Series(4.2, index=close.index))


def test_a_steady_policy_rate_leaves_the_price_alone_and_a_forward_rate_moves_it():
    quarters = make_quarters()
    close = make_close(quarters)
    peers = pd.DataFrame({"AAA": 24.0, "BBB": 16.0}, index=close.index)
    plain = latest_values(run_model(quarters, close, None, model_params(), peers))
    steady = latest_values(rate_run(4.0))
    assert steady["model_price"] == pytest.approx(plain["model_price"])
    assert steady["yield_gap"] == 0 and steady["forward_gap"] == 0 and steady["market_yield"] == 4.2

    hiked = rate_run(4.0, use_forward_rate=True, forward_rate=4.5)
    values = latest_values(hiked)
    shift = 0.5 * 0.2795
    assert values["forward_rate"] == 4.5 and values["policy_rate"] == 4.0
    assert values["forward_gap"] == pytest.approx(shift, abs=1e-3) and values["yield_gap"] == pytest.approx(shift, abs=1e-3)
    # Both anchors move: own and peers' P/E each get the higher earnings yield, then the half-half blend.
    daily = hiked["daily"].iloc[-1]
    expected = 0.5 * (1 / (1 / daily["peer_anchor"] + shift / 100)) + 0.5 * (1 / (1 / daily["own_anchor"] + shift / 100))
    assert values["rate_anchor"] == pytest.approx(expected, rel=1e-3)
    assert values["model_price"] < plain["model_price"]
    # Only the last day uses the forward rate: the day before is untouched.
    assert hiked["daily"]["model_price"].iloc[-2] == pytest.approx(plain_series(quarters, close, peers).iloc[-2])


def plain_series(quarters, close, peers):
    return run_model(quarters, close, None, model_params(), peers)["daily"]["model_price"]


def test_a_rate_above_the_look_back_lowers_the_own_anchor_only():
    quarters = make_quarters()
    close = make_close(quarters)
    rates = pd.Series(np.where(close.index < close.index[-20], 2.0, 5.0), index=close.index)
    result = rate_run(rates, peer_weight=0.0)
    values = latest_values(result)
    # The own P/E's look-back sat mostly at 2%: today's 5% is a higher yield, so a lower own anchor.
    assert values["yield_gap"] == pytest.approx(3.0 * 0.2795, abs=1e-3) and values["forward_gap"] == 0
    assert values["rate_anchor"] < result["daily"]["own_anchor"].iloc[-1]
    # Peers price today's rate already: with all of the anchor from peers, only a forward rate would move it.
    peers_only = latest_values(rate_run(rates, peer_weight=1.0))
    assert peers_only["rate_anchor"] == pytest.approx(20.0)


def test_rate_dial_sensitivity_and_cases():
    from analysis.valuation_model import default_rate_cases, rate_cases
    result = rate_run(4.0)
    values = latest_values(result)
    dials = implied_dials(result).set_index("Dial")
    assert "Policy rate" in dials.index and dials.loc["Policy rate", "Block"] == "Rate model"
    # The model price is above the close here, so the rate that explains the close is higher than today's.
    if values["model_price"] > values["close"]:
        assert dials.loc["Policy rate", "Needed"] > 4.0
    steps = sensitivities(result)
    assert steps["0.5 point higher policy rate"] < 0
    table, expected = rate_cases(result, default_rate_cases(4.0))
    assert table["Policy rate %"].tolist() == [4.0, 4.25, 4.5]
    assert table["Model price"].iloc[0] == pytest.approx(values["model_price"])
    assert table["Model price"].is_monotonic_decreasing
    assert expected == pytest.approx((table["Model price"] * table["Probability %"]).sum() / 100)
    # Without a policy rate there is no rate dial and no cases.
    off = run_model(make_quarters(), make_close(make_quarters()), None, model_params(),
                    pd.DataFrame({"AAA": 20.0}, index=make_close(make_quarters()).index))
    assert "Policy rate" not in implied_dials(off)["Dial"].tolist()
    assert rate_cases(off, default_rate_cases(4.0))[0].empty


def test_now_is_the_chain_without_forward_assumptions():
    from analysis.valuation_model import FORWARD_PARAMS, now_and_forward
    flat = {"growth_follow": 0.0, "bear_growth_gap": 0.0, "bull_growth_gap": 0.0, "bear_margin_pts": 0.0,
            "base_margin_pts": 0.0, "bull_margin_pts": 0.0}
    assert {"bear_growth_gap", "bull_growth_gap", "bear_margin_pts", "bull_margin_pts", "forward_rate", "forward_growth",
            "forward_margin"} <= set(FORWARD_PARAMS)
    # With no growth, margin change, items or forward rate, now and forward are the same price.
    values = latest_values(rate_run(4.0, **flat))
    assert values["now_price"] == pytest.approx(values["model_price"])
    assert values["held_clean_eps"] == pytest.approx(values["held_eps"]) and values["now_pe"] == pytest.approx(values["acceptable_pe"])
    # A forward rate moves only the forward side.
    hiked = rate_run(4.0, use_forward_rate=True, forward_rate=4.5, **flat)
    moved = latest_values(hiked)
    assert moved["now_price"] == pytest.approx(values["now_price"]) and moved["model_price"] < values["model_price"]
    table = now_and_forward(hiked).set_index("Step")
    assert table.loc["Policy rate %", "Now"] == 4.0 and table.loc["Policy rate %", "Forward"] == 4.5
    assert table.loc["Model price", "Change"] == pytest.approx(moved["model_price"] - moved["now_price"])
    # Growth is forward too: the default cases lift the forward EPS above the clean trailing EPS.
    grown = latest_values(rate_run(4.0))
    assert grown["held_eps"] > grown["held_clean_eps"] and grown["now_price"] == pytest.approx(values["now_price"])
    # Register items count only forward.
    items = register([["Cyclicality", "Risk", "General", "Beyond horizon", None, None, 3, "Modelled", "", ""]])
    quarters = make_quarters()
    close = make_close(quarters)
    with_items = latest_values(run_model(quarters, close, items, model_params(flat), pd.DataFrame({"AAA": 20.0}, index=close.index)))
    assert with_items["now_pe"] - with_items["acceptable_pe"] == pytest.approx(3.0)


def test_forward_growth_and_margin_apply_to_the_latest_quarter_only():
    from analysis.valuation_model import case_table, case_table_changes
    auto = rate_run(4.0)
    mine = rate_run(4.0, forward_growth=30.0, forward_margin=25.0, bull_growth_gap=10.0, bear_growth_gap=15.0,
                    bull_margin_pts=1.0, bear_margin_pts=-2.0)
    last = mine["quarterly"].iloc[-1]
    assert last["growth_base"] == 30.0 and last["growth_bull"] == 40.0 and last["growth_bear"] == 15.0
    assert last["margin_base"] == 25.0
    clean, margin = last["clean_eps"], last["net_margin"]
    assert last["eps_bear"] == pytest.approx(clean * 1.15 * 23.0 / margin)
    assert last["eps_bull"] == pytest.approx(clean * 1.40 * 26.0 / margin)
    # Earlier quarters keep the auto rule (the history stays a backtest).
    earlier = mine["quarterly"].iloc[:-1]
    assert (earlier["growth_base"] == auto["quarterly"]["growth_base"].iloc[:-1]).all()
    # The table: current, forward (auto or yours), bull and bear adjustments (bear growth gap shown negative).
    drawn = case_table(mine).set_index("Metric")
    assert drawn.loc["Revenue growth %", "Forward"] == 30.0 and drawn.loc["Revenue growth %", "Bear adj"] == -15.0
    assert drawn.loc["Net margin %", "Forward from"] == "you"
    auto_table = case_table(auto)
    assert (auto_table["Forward from"] == "auto rule").all()
    assert auto_table.set_index("Metric").loc["Revenue growth %", "Forward"] == pytest.approx(
        auto["quarterly"]["growth_base"].iloc[-1])
    # Editing only an adjustment keeps forward on auto; a new forward becomes yours; a cleared one goes back to auto.
    edited = auto_table.copy()
    edited.loc[0, "Bear adj"] = -20.0
    assert case_table_changes(auto_table, edited, auto["params"]) == {"bear_growth_gap": 20.0}
    edited.loc[0, "Forward"] = 60.0
    assert case_table_changes(auto_table, edited, auto["params"])["forward_growth"] == 60.0
    cleared = case_table(mine)
    cleared.loc[1, "Forward"] = np.nan
    assert np.isnan(case_table_changes(case_table(mine), cleared, mine["params"])["forward_margin"])


def test_item_probability_moves_with_the_case_and_the_market_implies_one():
    from analysis.valuation_model import market_implied_probabilities
    quarters = make_quarters()
    close = make_close(quarters)
    peers = pd.DataFrame({"AAA": 20.0}, index=close.index)
    flat = model_params({"bear_growth_gap": 0.0, "bull_growth_gap": 0.0, "bear_margin_pts": 0.0, "bull_margin_pts": 0.0})
    items = register([["Export rules", "Risk", "Event", "In horizon", 40, 20, None, "Modelled", "", ""]])
    items["Bull prob adj"], items["Bear prob adj"] = -20.0, 30.0
    result = run_model(quarters, close, items, flat, peers)
    last = result["quarterly"].iloc[-1]
    # Bear: 70% x 20% = -14% of EPS, base 40% -> -8%, bull 20% -> -4%.
    assert last["event_bear"] == pytest.approx(-14.0) and last["event_base"] == pytest.approx(-8.0)
    assert last["event_bull"] == pytest.approx(-4.0)
    weights = scenario_weights(flat)
    assert last["event_effect"] == pytest.approx(weights @ np.array([-14.0, -8.0, -4.0]))
    # The market-implied probability prices the model at the close.
    implied = market_implied_probabilities(result)
    values = latest_values(result)
    assert np.isfinite(implied.iloc[0])
    # A risk: a model below the close needs a lower probability, above it a higher one.
    assert (implied.iloc[0] < 40) == (values["model_price"] < values["close"])
    if True:
        trial = items.copy()
        trial["Probability %"] = implied.iloc[0]
        again = latest_values(run_model(quarters, close, trial, flat, peers))
        assert again["model_price"] == pytest.approx(values["close"], rel=1e-3)
    # A general item has no probability to imply.
    general = register([["Cycle", "Risk", "General", "In horizon", None, None, 2, "Modelled", "", ""]])
    assert market_implied_probabilities(run_model(quarters, close, general, flat, peers)).isna().all()
