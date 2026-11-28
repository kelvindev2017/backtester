import numpy as np
import pandas as pd
import pytest

from analysis import portfolio as pf


def closes(n=800, seed=1):
    index = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(seed)
    data = {
        "UP": 100 * np.exp(np.cumsum(np.full(n, 0.0005))),
        "CALM": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.003, n))),
        "WILD": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.03, n))),
    }
    frame = pd.DataFrame(data, index=index)
    frame["NEW"] = frame["UP"].where(frame.index >= index[400])    # starts later
    frame["OLD"] = frame["UP"]
    return frame


def test_fixed_mix_without_rules_tracks_its_assets():
    px = closes()
    spec = pf.PortfolioSpec([pf.Holding("UP", 1.0)], rebalance="M", cost=0.0)
    returns, held, _, begin = pf.simulate(px[["UP"]], pd.Series(0.0, index=px.index), spec)
    assert begin == px.index[0]
    assert np.isclose((1 + returns).prod(), px["UP"].iloc[-1] / px["UP"].iloc[0])
    assert np.allclose(held["UP"], 1.0)


def test_stand_in_fills_the_years_before_a_ticker_existed():
    px = closes()
    level, source = pf.splice(px["NEW"], [px["OLD"]])
    assert level.first_valid_index() == px.index[0]
    assert source.iloc[100] == "OLD" and source.iloc[500] == "NEW"
    assert np.isclose(level.iloc[-1] / level.iloc[0], px["UP"].iloc[-1] / px["UP"].iloc[0])
    spec = pf.PortfolioSpec([pf.Holding("NEW", 1.0, stand_ins=("OLD",))])
    prices, table = pf.build_prices(px, spec)
    assert table.loc[0, "Stand-ins used"] == "OLD"
    assert prices["NEW"].first_valid_index() == px.index[0]


def test_inverse_volatility_gives_the_calm_asset_more():
    px = closes()
    spec = pf.PortfolioSpec([pf.Holding("CALM", 0.5, sleeve="rp"), pf.Holding("WILD", 0.5, sleeve="rp")],
                            sleeves={"rp": pf.SleeveRule(weighting="Inverse volatility")})
    targets = pf.target_weights(px, spec).iloc[-1]
    assert targets["CALM"] > 0.8 and np.isclose(targets.sum(), 1.0)


def test_trend_filter_moves_to_cash_below_the_average():
    px = closes()
    falling = px["UP"].iloc[::-1].to_numpy()
    px["DOWN"] = falling
    spec = pf.PortfolioSpec([pf.Holding("DOWN", 1.0, trend_days=50)])
    targets = pf.target_weights(px, spec)
    assert targets["DOWN"].iloc[100:].eq(0).all()
    cash = pd.Series(0.0001, index=px.index)
    returns, *_ = pf.simulate(px[["DOWN"]], cash, spec)
    assert returns.iloc[5:].gt(0).all()          # earns cash every day, never holds the falling asset


def test_volatility_target_scales_down_a_wild_asset():
    px = closes()
    spec = pf.PortfolioSpec([pf.Holding("WILD", 1.0, sleeve="v")], sleeves={"v": pf.SleeveRule(vol_target=0.10)})
    weight = pf.target_weights(px, spec)["WILD"].iloc[100:]
    assert weight.max() <= 1.0 and weight.median() < 0.4


def test_leverage_and_rebalance_days():
    px = closes()
    spec = pf.PortfolioSpec([pf.Holding("UP", 1.0)], leverage=1.5, rebalance="W")
    assert np.isclose(pf.target_weights(px, spec)["UP"].iloc[-1], 1.5)
    flags = pf.rebalance_flags(px.index, "W")
    assert all(day.weekday() == 4 for day in px.index[flags][:-1])
    assert pf.next_rebalance_day(pd.Timestamp("2026-11-24"), "W") == pd.Timestamp("2026-11-27")
    assert pf.next_rebalance_day(pd.Timestamp("2026-11-27"), "W") == pd.Timestamp("2026-12-04")


def test_stats_and_targets():
    index = pd.bdate_range("2020-01-01", periods=504)
    returns = pd.Series(0.0004, index=index)
    returns.iloc[300:310] = -0.012                     # about an 11% fall in 2021
    row = pf.stats(returns)
    assert row["Max drawdown %"] == pytest.approx(-11.4, abs=0.3)
    checks = pf.target_check(row, 10, 10, 15)
    assert checks == {"Return": row["Return %/yr"] > 10, "Max drawdown": False, "Yearly drawdown": True}


@pytest.mark.parametrize("name", list(pf.PRESETS))
def test_presets_build(name):
    spec = pf.preset_spec(name)
    assert spec.holdings and abs(sum(h.weight for h in spec.holdings) - 1) < 1e-9
