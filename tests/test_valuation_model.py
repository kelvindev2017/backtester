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
