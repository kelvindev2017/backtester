import numpy as np
import pandas as pd
import pytest

from analysis.valuation_model import (
    BLOCKS, MODELLED, SIGNALS, TO_MULTIPLE, TO_SCENARIOS, clean_register, implied_dials, item_effects, latest_values,
    model_params, preset_params, preset_register, quarterly_inputs, register_changes, route_items, run_model,
    scenario_point, through_cycle_eps, scenario_weights, sensitivities, track_items,
)


@pytest.fixture(autouse=True)
def peer_anchor_defaults(monkeypatch):
    """Most tests here were written for the peer-median anchor (half peers, half own): keep that as their default.
    Tests of the fundamental P/E set ``anchor_source`` themselves."""

    import analysis.valuation_model as model
    monkeypatch.setitem(model.DEFAULT_PARAMS, "anchor_source", "Peers")
    monkeypatch.setitem(model.DEFAULT_PARAMS, "peer_weight", 0.5)



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
    point = scenario_point(4.0, 20.0, 20.0, model_params({"case_spread": "Fixed gaps"}))
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
    # Next year's EPS rolls forward from the release at the rate it set.
    assert last["held_eps"] == pytest.approx(quarterly["risk_adjusted_eps"].iloc[-1] * last["eps_roll"])
    ratio = quarterly["risk_adjusted_eps"].iloc[-1] / quarterly["clean_eps"].iloc[-1]
    assert last["eps_roll"] == pytest.approx(ratio ** last["roll_years"]) and last["roll_years"] > 0
    assert last["model_price"] == pytest.approx(last["held_eps"] * 17.0)
    assert last["residual_pct"] == pytest.approx((last["model_price"] / last["close"] - 1) * 100)
    # Between releases the quarterly outputs are held.
    release = quarterly.index[-2]
    held = (daily["held_eps"] / daily["eps_roll"]).loc[release:quarterly.index[-1] - pd.Timedelta(days=1)]
    assert held.round(9).nunique() == 1 and held.iloc[0] == pytest.approx(quarterly.at[release, "risk_adjusted_eps"])
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
    unrolled = lambda run: run["daily"].at[after, "held_eps"] / run["daily"].at[after, "eps_roll"]
    assert unrolled(result) == pytest.approx(unrolled(plain) * 0.9)
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
    params = model_params({"peer_weight": 1.0, "case_spread": "Fixed gaps"})
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
    assert dials.at["Acceptable P/E", "Current"] == pytest.approx(18.0)
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
    # The preset takes its results from the Hong Kong listing; a typed-in EPS still works when you set one.
    params, items = preset_params("asmvy"), preset_register("ASMVY")
    assert params["manual_eps"] == 0 and params["weight_base"] == 50
    params = {**params, "manual_eps": 1.64}
    routed = route_items(items, params)
    assert (routed["Gate"] == MODELLED).sum() == 2 and item_effects(routed)[1] == pytest.approx(-3.0)
    days = pd.bdate_range("2026-01-02", "2026-11-10")
    close = pd.Series(np.linspace(30, 37.5, len(days)), index=days)
    peers = pd.DataFrame({"KLIC": 14.4, "CAMT": 26.5, "AMAT": 26.5}, index=days)
    result = run_model(pd.DataFrame(), close, items, params, peers)
    values = latest_values(result)
    # 1.64 less 11%, then bear 0% / base 28% / bull 44% growth on margins of 8.8 / 13.8 / 16.8 against 9.8.
    clean = 1.64 * 0.89
    expected = 0.25 * clean * 8.8 / 9.8 + 0.5 * clean * 1.28 * 13.8 / 9.8 + 0.25 * clean * 1.44 * 16.8 / 9.8
    assert values["risk_adjusted_eps"] == pytest.approx(expected)
    assert values["acceptable_pe"] == pytest.approx(26.5 - 3) and values["model_price"] == pytest.approx(expected * 23.5)
    assert preset_register("NVDA").empty and preset_params("NVDA") == model_params()


def test_deep_cycle_is_valued_on_through_cycle_eps():
    # Yearly EPS 4, 6, 8, 2, 1, 1.5 and next year 3: it fell over 50% from a high with no trend, so Auto turns it on.
    past = pd.Series(np.repeat([4.0, 6.0, 8.0, 2.0, 1.0, 1.5], 4))
    params = model_params({"cycle_eps": "Auto", "cycle_years": 7})
    cycle, trend, on = through_cycle_eps(past, 3.0, 1.0, params)
    assert on and cycle == pytest.approx((4 + 6 + 8 + 2 + 1 + 1.5 + 3) / 7) and trend < 10
    # A steady grower is left on next year's EPS.
    grower = pd.Series(np.repeat([1.0, 1.3, 1.7, 2.2, 2.9, 3.7], 4))
    assert not through_cycle_eps(grower, 4.8, 1.0, params)[2]
    assert not through_cycle_eps(past, 3.0, 1.0, model_params({"cycle_eps": "Off"}))[2]
    assert through_cycle_eps(grower, 4.8, 1.0, model_params({"cycle_eps": "On"}))[2]
    # Too little history: off.
    assert not through_cycle_eps(past.iloc[-6:], 3.0, 1.0, params)[2]


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
    assert dials.at["Acceptable P/E", "Current"] == pytest.approx(after["acceptable_pe"])


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
    # These tests are about the policy-rate model; the market yield is only shown as a check.
    overrides = {"yield_source": "Policy model", "case_spread": "Fixed gaps", **overrides}
    return run_model(quarters, close, None, model_params(overrides), peers, policy_rate=rates,
                     market_yield=pd.Series(4.2, index=close.index))


def test_a_steady_policy_rate_leaves_the_price_alone_and_a_forward_rate_moves_it():
    quarters = make_quarters()
    close = make_close(quarters)
    peers = pd.DataFrame({"AAA": 24.0, "BBB": 16.0}, index=close.index)
    plain = latest_values(run_model(quarters, close, None, model_params({"case_spread": "Fixed gaps"}), peers))
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
    return run_model(quarters, close, None, model_params({"case_spread": "Fixed gaps"}), peers)["daily"]["model_price"]


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
    # The modeled fair value is above the close here, so the rate that explains the close is higher than today's.
    if values["model_price"] > values["close"]:
        assert dials.loc["Policy rate", "Needed"] > 4.0
    steps = sensitivities(result)
    assert steps["0.5 point higher policy rate"] < 0
    table, expected = rate_cases(result, default_rate_cases(4.0))
    assert table["Policy rate %"].tolist() == [4.0, 4.25, 4.5]
    assert table["Modeled fair value"].iloc[0] == pytest.approx(values["model_price"])
    assert table["Modeled fair value"].is_monotonic_decreasing
    assert expected == pytest.approx((table["Modeled fair value"] * table["Probability %"]).sum() / 100)
    # Without a policy rate there is no rate dial and no cases.
    off = run_model(make_quarters(), make_close(make_quarters()), None, model_params(),
                    pd.DataFrame({"AAA": 20.0}, index=make_close(make_quarters()).index))
    assert "Policy rate" not in implied_dials(off)["Dial"].tolist()
    assert rate_cases(off, default_rate_cases(4.0))[0].empty


def test_now_is_the_chain_without_forward_assumptions():
    from analysis.valuation_model import FORWARD_PARAMS, now_and_forward
    flat = {"case_spread": "Fixed gaps", "growth_follow": 0.0, "bear_growth_gap": 0.0, "bull_growth_gap": 0.0, "bear_margin_pts": 0.0,
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
    assert table.loc["Policy rate %", "Last quarter"] == 4.0 and table.loc["Policy rate %", "Forward"] == 4.5
    assert table.loc["Modeled fair value", "Change"] == pytest.approx(moved["model_price"] - moved["now_price"])
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


def test_a_peer_p_e_above_the_cap_sits_out_the_median():
    from analysis.valuation_model import case_table
    quarters = make_quarters()
    close = make_close(quarters)
    peers = pd.DataFrame({"AAA": 20.0, "BBB": 30.0, "CCC": 400.0}, index=close.index)
    result = run_model(quarters, close, None, model_params({"pe_cap": 100.0}), peers)
    values = latest_values(result)
    assert values["peer_anchor"] == pytest.approx(25.0) and values["peers_used"] == 2
    assert list(result["peer_pe"].columns) == ["AAA", "BBB", "CCC"] and result["peer_pe"]["CCC"].iloc[-1] == 400.0
    # Without a reported growth or margin, Current shows the fallback the model runs on, and says so.
    no_revenue = run_model(quarters[["EPS"]], close, None, model_params({"fallback_yoy": 40.0, "fallback_margin": 9.8}), peers)
    table = case_table(no_revenue).set_index("Metric")
    assert table.loc["Revenue growth %", "Current"] == 40.0 and table.loc["Net margin %", "Current"] == 9.8
    assert (table["Current from"] == "fallback").all()
    assert (case_table(result)["Current from"] == "reported").all()


def test_fundamental_pe_from_growth_and_gross_margin():
    from analysis.valuation_model import fundamental_pe, growth_half_life, quarterly_inputs
    params = model_params({"anchor_source": "Fundamentals"})
    # A higher gross margin lets growth last longer: a half-life of 0.5 years at 20% or less, 2 at 60% or more.
    assert growth_half_life(10, params) == 0.5 and growth_half_life(60, params) == 2.0
    assert growth_half_life(40, params) == 1.25 and growth_half_life(np.nan, params) == 1.25
    # No growth, then 3% for ever, at 4% + 4.5%: close to 1 / (8.5% - 3%) = 18x.
    flat = fundamental_pe(0, 18, 4.0, params)
    assert 15 < flat < 19
    # Faster growth and a higher gross margin are worth more; a higher yield is worth less.
    assert fundamental_pe(30, 60, 4.0, params) > fundamental_pe(30, 18, 4.0, params) > fundamental_pe(8, 18, 4.0, params)
    assert fundamental_pe(30, 60, 6.0, params) < fundamental_pe(30, 60, 4.0, params)
    # Growth over 100% counts, and the P/E keeps rising with it without exploding.
    steps = [fundamental_pe(g, 75, 4.3, params) for g in (50, 100, 200, 300)]
    assert steps == sorted(steps) and 60 < steps[1] < 80 and steps[-1] < 400
    assert fundamental_pe(1000, 75, 4.3, params) == fundamental_pe(300, 75, 4.3, params)
    assert np.isnan(fundamental_pe(np.nan, 50, 4.0, params))
    # The income map gives the growth a year over the history and the gross margin.
    quarters = make_quarters().assign(Gross_Profit=lambda frame: frame["Revenue"] * 0.45)
    inputs = quarterly_inputs(quarters, 3)
    assert inputs["gross_margin"].dropna().iloc[-1] == pytest.approx(45.0)
    ttm = inputs["ttm_revenue"].dropna()
    # 14 quarters give 11 trailing sums: not enough for 3 years back, so the growth a year is over 2 years.
    assert len(ttm) < 13
    assert inputs["revenue_cagr"].iloc[-1] == pytest.approx(((ttm.iloc[-1] / ttm.iloc[-9]) ** (1 / 2) - 1) * 100)


def test_the_fundamental_anchor_replaces_the_peers_and_follows_the_rate():
    quarters = make_quarters().assign(Gross_Profit=lambda frame: frame["Revenue"] * 0.45)
    close = make_close(quarters)
    peers = pd.DataFrame({"AAA": 6.0}, index=close.index)
    rates = pd.Series(4.0, index=close.index)
    base = {"anchor_source": "Fundamentals", "peer_weight": 1.0, "pe_growth_latest_share": 0.0,
            "half_life_from_steadiness": False, "market_implied_erp": False}
    result = run_model(quarters, close, None, model_params(base), peers, policy_rate=rates)
    values = latest_values(result)
    # The peers are not used: the outside view is the fundamental P/E, worked out at the day's yield.
    from analysis.valuation_model import fundamental_pe
    expected = fundamental_pe(values["held_cagr"], values["held_gross_margin"], values["now_yield"], model_params(base))
    assert values["fundamental_pe"] == pytest.approx(expected) and values["outside_anchor"] == pytest.approx(expected)
    assert values["rate_anchor"] == pytest.approx(expected)
    # A forward rate lifts the yield and lowers the fundamental P/E; 'now' stays at today's rate.
    hiked = latest_values(run_model(quarters, close, None, model_params({**base, "use_forward_rate": True,
                                                                         "forward_rate": 6.0}), peers, policy_rate=rates))
    assert hiked["rate_anchor"] < values["rate_anchor"] and hiked["now_anchor"] == pytest.approx(values["now_anchor"])
    # The rate dial and the rate cases redo the fundamental P/E at each rate.
    from analysis.valuation_model import default_rate_cases, rate_cases
    table, _ = rate_cases(result, default_rate_cases(4.0))
    assert table["P/E anchor"].iloc[0] == pytest.approx(values["rate_anchor"]) and table["P/E anchor"].is_monotonic_decreasing


def test_forward_view_of_the_p_e_and_a_separate_market_premium():
    from analysis.valuation_model import fundamental_pe, pe_table, pe_table_changes
    quarters = make_quarters().assign(Gross_Profit=lambda frame: frame["Revenue"] * 0.12)
    close = make_close(quarters)
    rates = pd.Series(4.0, index=close.index)
    base = {"anchor_source": "Fundamentals", "peer_weight": 1.0, "pe_growth_latest_share": 0.0,
            "half_life_from_steadiness": False}
    plain = run_model(quarters, close, None, model_params(base), None, policy_rate=rates)
    before = latest_values(plain)
    assert before["premium_pct"] == 0 and before["acceptable_pe"] == pytest.approx(before["pe_before_premium"])
    # The market premium: what the close pays over the model before any premium of yours.
    assert before["implied_premium_pct"] == pytest.approx((before["close"] / before["model_price"] - 1) * 100)
    # Your forward growth and half-life replace the history from the latest release; the premium multiplies the P/E.
    mine = run_model(quarters, close, None, model_params({**base, "forward_pe_growth": 12.0, "forward_half_life": 6.0,
                                                          "forward_equity_premium": 3.5, "market_premium": 50.0}),
                     None, policy_rate=rates)
    after = latest_values(mine)
    params = model_params({**base, "equity_premium": 3.5})
    expected = fundamental_pe(12.0, None, after["model_yield"], params, half_life=6.0)
    assert after["rate_anchor"] == pytest.approx(expected)
    assert after["acceptable_pe"] == pytest.approx(after["pe_before_premium"] * 1.5)
    assert after["model_price"] == pytest.approx(after["held_eps"] * after["acceptable_pe"])
    # 'Now' keeps neither: the history's fundamental P/E, no premium.
    assert after["now_pe"] == pytest.approx(before["now_pe"])
    # Earlier quarters keep the history: the day before the latest release is unchanged.
    day = mine["forward_from"] - pd.Timedelta(days=1)
    earlier = mine["daily"].loc[:day].iloc[-1]
    assert earlier["acceptable_pe"] == pytest.approx(plain["daily"].loc[:day].iloc[-1]["acceptable_pe"])
    # The dials and rate cases see the same price.
    assert sensitivities(mine)
    # The table: current from the history (and the market's premium), forward yours; edits round-trip.
    table = pe_table(mine).set_index("Metric")
    assert table.loc["Growth for the P/E %", "Forward"] == 12.0 and table.loc["Market premium %", "Forward"] == 50.0
    assert table.loc["Growth half-life, years", "Current"] == pytest.approx(0.5)
    drawn = pe_table(plain)
    edited = drawn.copy()
    edited.loc[edited["Metric"] == "Market premium %", "Forward"] = 20.0
    assert pe_table_changes(drawn, edited, plain["params"]) == {"market_premium": 20.0}
    cleared = pe_table(mine)
    cleared.loc[cleared["Metric"] == "Growth for the P/E %", "Forward"] = np.nan
    cleared.loc[cleared["Metric"] == "Market premium %", "Forward"] = np.nan
    changes = pe_table_changes(pe_table(mine), cleared, mine["params"])
    assert np.isnan(changes["forward_pe_growth"]) and changes["market_premium"] == 0.0
    # With peers as the outside view only the premium row is shown.
    peers = run_model(quarters, close, None, model_params({"anchor_source": "Peers"}), pd.DataFrame({"A": 20.0}, index=close.index))
    assert pe_table(peers)["Metric"].tolist() == ["Market premium %"]


def test_rules_from_data_known_at_the_time_hold_for_the_whole_history():
    from analysis.valuation_model import fundamental_pe, rolling_beta
    quarters = make_quarters()
    quarters["Revenue"] = [1000 * 1.0194 ** step for step in range(len(quarters))]
    quarters["Gross_Profit"] = quarters["Revenue"] * 0.12
    close = make_close(quarters)
    rng = np.random.default_rng(3)
    market = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(close)))), index=close.index)
    stock = pd.Series(50 * np.exp(0.6 * np.log(market / 100)), index=close.index)
    on = {"anchor_source": "Fundamentals", "peer_weight": 1.0, "market_implied_erp": False}
    off = {**on, "half_life_from_steadiness": False, "erp_from_beta": False, "pe_growth_latest_share": 0.0}
    ruled = run_model(quarters, stock, None, model_params(on), market_close=market)
    plain = run_model(quarters, stock, None, model_params(off), market_close=market)
    daily = ruled["daily"]
    # Steady ~8% growth earns a long half-life on every day it is known, not only the latest quarter.
    known = daily["held_growth_spread"].notna()
    assert known.sum() > 50
    assert (daily.loc[known, "growth_half_life"] > plain["daily"].loc[known, "growth_half_life"] + 1).all()
    # Beta ~0.6 lowers the risk premium (rolling, as known on each day).
    beta = rolling_beta(stock, market, daily.index)
    assert beta.dropna().iloc[-1] == pytest.approx(0.6, abs=0.03)
    # Blume-adjusted: 0.67 x 0.6 + 0.33 = 0.73.
    assert daily["rule_erp"].iloc[-1] == pytest.approx(4.5 * (0.67 * beta.iloc[-1] + 0.33), rel=1e-6)
    # Growth for the P/E blends the latest year and the history.
    last = daily.iloc[-1]
    assert last["rule_growth"] == pytest.approx(0.5 * last["held_yoy"] + 0.5 * last["held_cagr"])
    expected = fundamental_pe(last["rule_growth"], None, 4.0, {**model_params(on), "equity_premium": last["rule_erp"]},
                              half_life=last["growth_half_life"])
    assert last["fundamental_pe"] == pytest.approx(expected)
    assert daily.loc[known, "fundamental_pe"].mean() > plain["daily"].loc[known, "fundamental_pe"].mean()


def test_worked_fundamental_pe_shows_the_risk_premium_the_run_used():
    # With beta scaling on, the shown required return must be the yield plus the beta-scaled premium, not the plain one.
    from analysis.model_equations import worked_equations
    quarters = make_quarters()
    stock = make_close(quarters)
    rng = np.random.default_rng(3)
    market = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(stock)))), index=stock.index)
    stock = stock * (market / market.iloc[0]) ** 1.5
    result = run_model(quarters, stock, None, model_params({"anchor_source": "Fundamentals"}), market_close=market)
    values = latest_values(result)
    assert abs(values["pe_erp"] - model_params()["equity_premium"]) > 0.1
    line = [eq for eq in worked_equations(result, "multiple_builder") if r"\text{P/E}_{\text{fund}}" in eq][0]
    required = model_params()["base_yield"] + values["pe_erp"]
    assert f"= {required:,.2f}\\%)" in line


def test_market_yield_is_the_discount_rate_and_a_forward_rate_moves_it_by_the_model_weight():
    from analysis.valuation_model import rate_weight
    quarters = make_quarters()
    close = make_close(quarters)
    market = pd.Series(5.0, index=close.index)
    policy = pd.Series(3.9, index=close.index)
    params = {"anchor_source": "Fundamentals", "erp_from_beta": False, "market_implied_erp": False}
    result = run_model(quarters, close, None, model_params(params), policy_rate=policy, market_yield=market)
    values = latest_values(result)
    assert values["now_yield"] == pytest.approx(5.0) and values["model_yield"] == pytest.approx(5.0)
    old = run_model(quarters, close, None, model_params({**params, "yield_source": "Policy model"}),
                    policy_rate=policy, market_yield=market)
    # A higher discount rate gives a lower fundamental P/E than the policy model's 3.9 -> ~3.75% yield.
    assert values["fundamental_pe"] < latest_values(old)["fundamental_pe"]
    forward = run_model(quarters, close, None, model_params({**params, "use_forward_rate": True, "forward_rate": 4.9}),
                        policy_rate=policy, market_yield=market)
    assert latest_values(forward)["model_yield"] == pytest.approx(5.0 + 1.0 * rate_weight(10, 2.0))
    # No policy rate at all: still discounted at the market yield.
    alone = run_model(quarters, close, None, model_params(params), market_yield=market)
    assert latest_values(alone)["now_yield"] == pytest.approx(5.0)


def test_analysts_base_case_makes_base_eps_the_analysts_eps_less_the_haircut():
    from analysis.valuation_model import case_table
    quarters = make_quarters()
    close = make_close(quarters)
    params = model_params({"eps_haircut_pct": 10.0, "base_margin_pts": 2.0})
    result = run_model(quarters, close, None, params, analyst_growth=30.0)
    values = latest_values(result)
    assert values["eps_base"] == pytest.approx(values["ttm_eps"] * 1.3 * 0.9)
    assert values["eps_bear"] < values["eps_base"] < values["eps_bull"]
    assert case_table(result).set_index("Metric").loc["Revenue growth %", "Forward from"] == "analysts"
    own = run_model(quarters, close, None, model_params({**params, "base_eps_source": "Own rule"}), analyst_growth=30.0)
    assert latest_values(own)["growth_base"] != pytest.approx(30.0)
    # Earlier quarters keep the auto rule: only the latest one knows today's estimates.
    assert result["quarterly"]["growth_base"].iloc[0] == pytest.approx(own["quarterly"]["growth_base"].iloc[0])


def test_per_share_haircut_skips_gaap_reporters_and_ignores_a_gaap_loss():
    from analysis.model_equations import stock_pay_haircut_per_share
    ends = pd.date_range("2025-03-31", periods=4, freq="QE")
    announced = ends + pd.Timedelta(days=30)
    pay, shares = pd.Series(100.0, index=ends), pd.Series(1000.0, index=ends)  # 0.10 a share a quarter
    reported = pd.Series(1.0, index=announced)  # 4.00 over the year
    # Reported equals GAAP: the pay is already a cost (Apple-like), nothing to take off.
    assert stock_pay_haircut_per_share(pay, shares, reported, pd.Series(1.0, index=ends))[0] == 0.0
    # Adjusted above GAAP: 0.40 a share x (1 - 25%) = 0.30 of 4.00.
    share, reason = stock_pay_haircut_per_share(pay, shares, reported, pd.Series(0.8, index=ends), tax_rate=0.25)
    assert share == pytest.approx(7.5) and "after 25% tax" in reason
    # A GAAP loss from write-offs (Intel-like) does not blow the haircut up to the cap.
    assert stock_pay_haircut_per_share(pay, shares, reported, pd.Series(-0.6, index=ends), tax_rate=0.25)[0] == pytest.approx(7.5)
    # Never more than the gap between reported and GAAP EPS.
    assert stock_pay_haircut_per_share(pay, shares, reported, pd.Series(0.99, index=ends) * 1.0, tax_rate=0.0)[0] == pytest.approx(0.0)
    assert stock_pay_haircut_per_share(pay, shares, reported, pd.Series(0.95, index=ends), tax_rate=0.0)[0] == pytest.approx(5.0)
    assert np.isnan(stock_pay_haircut_per_share(pay, shares, -reported)[0])


def test_eps_volatility_cases_sit_k_spreads_either_side_of_base():
    from analysis.valuation_model import case_spread
    params = model_params()
    assert params["case_spread"] == "EPS volatility"
    # Own volatility 0.30: s = 0.25 x 0.30 + 0.75 x 0.073.
    point = scenario_point(4.0, 20.0, 20.0, params, eps_spread=0.30)
    s = 0.25 * 0.30 + 0.75 * 0.073
    assert point["case_spread"] == pytest.approx(s)
    assert point["eps_base"] == pytest.approx(4.0 * 1.10)
    assert point["eps_bear"] == pytest.approx(4.0 * 1.10 * np.exp(-2.4 * s))
    assert point["eps_bull"] == pytest.approx(4.0 * 1.10 * np.exp(2.4 * s))
    # No history: the typical volatility alone; a calmer company gets narrower cases than a volatile one.
    assert case_spread(np.nan, params) == pytest.approx(0.073)
    calm, wild = scenario_point(4.0, 20.0, 20.0, params, eps_spread=0.01), scenario_point(4.0, 20.0, 20.0, params, eps_spread=0.6)
    assert calm["eps_bull"] - calm["eps_bear"] < wild["eps_bull"] - wild["eps_bear"]
    # The run feeds each quarter its own EPS volatility, known at that release.
    quarters = make_quarters(count=24)
    quarters["EPS"] = quarters["EPS"] * (1 + 0.3 * np.sin(np.arange(len(quarters))))
    result = run_model(quarters, make_close(quarters), None, params)
    last = result["quarterly"].iloc[-1]
    assert np.isfinite(last["eps_spread"]) and last["case_spread"] == pytest.approx(case_spread(last["eps_spread"], params))
    from analysis.model_equations import worked_equations
    assert any(line.startswith("s = ") for line in worked_equations(result, "scenario_engine"))


def test_market_implied_premium_respects_the_usual_market_pe_and_passes_rate_changes():
    from analysis.valuation_model import fundamental_pe, implied_erp
    params = model_params()
    FUND = {"anchor_source": "Fundamentals", "case_spread": "Fixed gaps"}
    # The solver inverts the fundamental P/E.
    erp = implied_erp(30.0, 12.0, 3.0, 4.5, params)
    assert fundamental_pe(12.0, None, 4.5, {**params, "equity_premium": erp}, half_life=3.0) == pytest.approx(30.0, rel=1e-4)
    # A stock the market always prices at 1.5x what the rules justify: holding its usual premium puts the model
    # near the close, and a jump in the yield on the last days lowers the P/E.
    quarters = make_quarters(count=40, end="2026-10-28")
    close = make_close(quarters, pe=40.0)
    plain = run_model(quarters, close, None, model_params({**FUND, "market_implied_erp": False}))
    implied = run_model(quarters, close, None, model_params(FUND))
    last_plain, last = latest_values(plain), latest_values(implied)
    assert np.isfinite(last["usual_erp"]) and np.isfinite(last["implied_erp"])
    assert abs(last["residual_pct"]) < abs(last_plain["residual_pct"])
    market = pd.Series(4.0, index=close.index)
    market.iloc[-5:] = 5.0
    risen = latest_values(run_model(quarters, close, None, model_params(FUND),
                                    market_yield=market))
    steady = latest_values(run_model(quarters, close, None, model_params(FUND),
                                     market_yield=pd.Series(4.0, index=close.index)))
    assert risen["acceptable_pe"] < steady["acceptable_pe"]


def test_usual_premium_follows_the_market_even_below_zero():
    from analysis.valuation_model import usual_implied_erp
    days = pd.bdate_range("2023-01-02", "2026-11-20")
    growth, half = pd.Series(40.0, index=days), pd.Series(2.0, index=days)
    yields = pd.Series(4.5, index=days)
    # A market P/E of 300 implies a tiny premium, and the usual one follows it: no floor.
    _, usual = usual_implied_erp(pd.Series(300.0, index=days), growth, half, yields, model_params(), 2)
    assert usual.dropna().iloc[-1] < 2.5


def test_past_quarters_follow_the_eps_trend_and_eps_rolls_forward():
    quarters = make_quarters()
    close = make_close(quarters)
    trend = run_model(quarters, close, None, model_params())
    revenue = run_model(quarters, close, None, model_params({"past_base": "Revenue rule", "roll_forward_eps": False}))
    q = trend["quarterly"]
    assert q["past_growth"].iloc[:-1].notna().any() and np.isnan(q["past_growth"].iloc[-1])
    assert (revenue["daily"]["eps_roll"] == 1.0).all() and trend["daily"]["eps_roll"].max() > 1.0


def test_through_cycle_auto_has_hysteresis():
    # Trend about 12% a year after a deep fall: above the 10% cap when off, but kept on once on (cap x 1.5).
    past = pd.Series(np.repeat([2.0, 4.0, 8.0, 3.0, 3.5, 4.5], 4))
    params = model_params({"cycle_eps": "Auto", "cycle_years": 7})
    _, trend, on = through_cycle_eps(past, 6.0, 1.0, params)
    assert 10 < trend < 15 and not on
    assert through_cycle_eps(past, 6.0, 1.0, params, was_on=True)[2]


def test_premium_warning_flags_a_low_or_negative_usual_premium():
    from analysis.valuation_model import premium_warning
    params = model_params({"anchor_source": "Fundamentals"})
    assert premium_warning({"pe_erp": 4.0, "implied_erp": 3.0, "now_yield": 4.5}, params) is None
    assert "only 1.00%" in premium_warning({"pe_erp": 1.0, "implied_erp": 3.0, "now_yield": 4.5}, params)
    text = premium_warning({"pe_erp": -0.5, "implied_erp": -0.5, "now_yield": 4.5}, params)
    assert "negative (-0.50%)" in text and "lower premium than the model can solve" in text
    assert premium_warning({"pe_erp": -0.5}, model_params({"anchor_source": "Peers"})) is None


def test_growth_is_per_share_over_consecutive_quarters_and_split_adjusted():
    from analysis.valuation_model import quarterly_inputs, split_adjusted_shares
    ends = pd.date_range("2023-03-31", periods=12, freq="QE")
    # Revenue doubles in 2025 by an all-share deal that doubles the shares: no growth per share.
    revenue = pd.Series([100.0] * 8 + [200.0] * 4, index=ends)
    shares = pd.Series([10.0] * 8 + [20.0] * 4, index=ends)
    frame = quarterly_inputs(pd.DataFrame({"Revenue": revenue, "Shares": shares, "EPS": 1.0}))
    assert frame["revenue_yoy"].iloc[-1] == pytest.approx(0.0)
    assert quarterly_inputs(pd.DataFrame({"Revenue": revenue, "EPS": 1.0}))["revenue_yoy"].iloc[-1] == pytest.approx(100.0)
    # A gap in the data: four reported quarters spread over years give no trailing sum.
    gappy = revenue.drop(ends[3:7])
    assert np.isnan(quarterly_inputs(pd.DataFrame({"Revenue": gappy, "EPS": 1.0}))["ttm_revenue"].loc[ends[7]])
    # A 10-for-1 split: earlier counts are put on today's basis; a deal's share issue is left alone.
    counts = pd.Series([2.5, 2.5, 25.0, 25.0], index=ends[:4])
    assert split_adjusted_shares(counts, {ends[1] + pd.Timedelta(days=10): 10.0}).tolist() == [25.0] * 4
    assert split_adjusted_shares(shares, {ends[7] + pd.Timedelta(days=10): 10.0}).tolist() == shares.tolist()


def test_share_counts_filed_in_millions_are_put_on_one_scale():
    from analysis.valuation_model import split_adjusted_shares
    ends = pd.date_range("2018-03-31", periods=6, freq="QE")
    # KO's older filings state 4,306 (millions) next to 4.305e9 in later ones.
    counts = pd.Series([4306.0, 4290.0, 4295.0, 4.305e9, 4.321e9, 4.325e9], index=ends)
    fixed = split_adjusted_shares(counts, None)
    assert fixed.iloc[:3].tolist() == pytest.approx([4.306e9, 4.29e9, 4.295e9])
    assert fixed.iloc[3:].tolist() == counts.iloc[3:].tolist()


def test_a_one_off_slump_is_not_projected_forward():
    # Flat EPS of 1 a quarter, one quarter at 0.2 (a pandemic quarter), then back to 1.
    quarters = make_quarters(count=24, growth=0.0)
    eps = np.ones(24)
    eps[12] = 0.2
    quarters["EPS"] = eps
    close = make_close(quarters)
    with_recovery = run_model(quarters, close, None, model_params())["quarterly"]["past_growth"]
    without = run_model(quarters, close, None, model_params({"history_recovery": 0.0}))["quarterly"]["past_growth"]
    slump, rebound = quarters.index[13], quarters.index[17]
    # In the slump: the old rule projects the fall forward; the recovery view expects part of the gap back.
    assert without.loc[slump] < 0 < with_recovery.loc[slump]
    # Climbing back out: growth is measured from the normal level, not from the dip.
    assert without.loc[rebound] > 10 and abs(with_recovery.loc[rebound]) < 5


def test_through_cycle_auto_needs_revenue_to_fall_too():
    from analysis.valuation_model import through_cycle_eps
    params = model_params({"cycle_eps": "Auto", "cycle_years": 7})
    past = pd.Series(np.repeat([2.0, 4.0, 3.0, 1.5, 1.4, 1.5], 4))
    flat = pd.Series(np.linspace(100, 104, 24))
    cyclical = pd.Series(np.concatenate([np.linspace(100, 140, 12), np.linspace(140, 100, 12)]))
    # EPS more than halved: with flat revenue it is a margin squeeze, with falling revenue a cycle.
    assert not through_cycle_eps(past, 1.5, 1.0, params, past_revenue=flat)[2]
    assert through_cycle_eps(past, 1.5, 1.0, params, past_revenue=cyclical)[2]
    assert through_cycle_eps(past, 1.5, 1.0, params)[2]


def test_market_check_pe_is_searched_past_the_cap():
    quarters = make_quarters()
    close = make_close(quarters, pe=400.0)
    dials = implied_dials(run_model(quarters, close, None, model_params({"market_implied_erp": False})))
    row = dials[dials["Dial"] == "Acceptable P/E"].iloc[0]
    assert row["Needed"] > 300 and row["Read"] == "Beyond your 300x cap"


def test_latest_quarter_growth_is_held_daily():
    quarters = make_quarters(count=12, growth=0.0)
    quarters.loc[quarters.index[-1], ["Revenue", "Net_Income"]] *= 1.2
    result = run_model(quarters, make_close(quarters), None, model_params())
    assert result["quarterly"]["quarter_yoy"].iloc[-1] == pytest.approx(20.0)
    # The 4-quarter growth moves only a quarter as much.
    assert result["quarterly"]["revenue_yoy"].iloc[-1] == pytest.approx(5.0)
    assert result["daily"]["held_quarter_yoy"].iloc[-1] == pytest.approx(20.0)


def test_pe_growth_can_use_the_latest_quarter():
    quarters = make_quarters(count=16, growth=0.0)
    quarters.loc[quarters.index[-1], ["Revenue", "Net_Income"]] *= 1.2
    close = make_close(quarters)
    base = {"anchor_source": "Fundamentals", "peer_weight": 1.0, "pe_growth_latest_share": 1.0}
    quarter = run_model(quarters, close, None, model_params({**base, "pe_growth_latest": "Latest quarter"}))
    year = run_model(quarters, close, None, model_params({**base, "pe_growth_latest": "4 quarters"}))
    assert quarter["daily"]["rule_growth"].iloc[-1] == pytest.approx(20.0, abs=0.5)
    assert year["daily"]["rule_growth"].iloc[-1] == pytest.approx(5.0, abs=0.5)


def test_growth_history_is_set_in_quarters():
    from analysis.valuation_model import quarterly_inputs
    quarters = make_quarters(count=24, growth=0.0)
    quarters["Revenue"] = 100 * 1.1 ** (np.arange(len(quarters)) / 4)  # 10% a year, steadily
    for count in (4, 6, 12):
        assert quarterly_inputs(quarters, count)["revenue_cagr"].iloc[-1] == pytest.approx(10.0, abs=0.01)
    # Settings saved as years still work.
    assert model_params({"growth_history_years": 2})["growth_history_quarters"] == 8
