import numpy as np
import pandas as pd

from analysis.optimizer import (
    SEARCH_SPACES,
    Param,
    SearchSpace,
    neighbourhood_scores,
    params_of,
    period_metrics,
    pick_settings,
    run_search,
    split_date,
)


def toy_backtest(index, daily_return):
    equity = 100 * np.cumprod(np.full(len(index), 1 + daily_return))
    buys = np.zeros(len(index), dtype=bool)
    buys[::20] = True
    return pd.DataFrame({"Equity": equity, "Buy_Signal": buys}, index=index)


def test_period_metrics_rebases_each_window():
    index = pd.bdate_range("2020-01-01", periods=504)
    backtest = toy_backtest(index, 0.001)
    first = period_metrics(backtest, index[0], index[251])
    second = period_metrics(backtest, index[252], index[-1])
    assert abs(first["return"] - (1.001 ** 251 - 1)) < 1e-9
    assert abs(second["return"] - (1.001 ** 251 - 1)) < 1e-9
    assert first["max_dd"] == 0
    assert first["trades"] == 13


def test_split_date_keeps_a_forward_period():
    index = pd.bdate_range("2020-01-01", periods=100)
    assert split_date(index, 0.7) == index[69]
    assert split_date(index, 1.0) == index[98]


def test_search_finds_the_peak_and_respects_ordering():
    space = SearchSpace((Param("a", "A", "int", 0, 50, 1), Param("b", "B", "int", 0, 50, 1)), less_than=(("a", "b"),))
    index = pd.bdate_range("2020-01-01", periods=300)

    def evaluate(params):
        # Best near a = 20, b = 40; noise-free so the search should get close.
        quality = -((params["a"] - 20) ** 2 + (params["b"] - 40) ** 2) / 1e6
        return toy_backtest(index, 0.0005 + quality)

    trials = run_search(evaluate, space, index[0], index[199], objective="cagr", n_trials=60, min_trades=1)
    assert len(trials) == 60
    ok = trials.loc[trials["status"].eq("ok")]
    assert (ok["a"] < ok["b"]).all()
    assert trials.loc[trials["status"].ne("ok"), "score"].isna().all()
    best, stable = pick_settings(trials, space)
    assert abs(best["a"] - 20) <= 8 and abs(best["b"] - 40) <= 8
    assert stable["neighbourhood"] <= best["score"] + 1e-12


def test_fixed_params_are_not_searched_and_min_trades_skips():
    space = SEARCH_SPACES["breakout"]
    index = pd.bdate_range("2020-01-01", periods=300)
    seen = []

    def evaluate(params):
        seen.append(params)
        backtest = toy_backtest(index, 0.0003)
        if params["breakout_lookback"] > 50:
            backtest["Buy_Signal"] = False
        return backtest

    trials = run_search(evaluate, space, index[0], index[199], n_trials=20, min_trades=2,
                        fixed={"breakout_exit_ma": 200})
    assert all(params["breakout_exit_ma"] == 200 for params in seen)
    skipped = trials.loc[trials["breakout_lookback"] > 50]
    assert skipped["status"].str.startswith("fewer than").all()


def test_stable_pick_prefers_a_plateau_over_a_spike():
    space = SearchSpace((Param("x", "X", "int", 0, 100, 1),))
    # A broad area scoring 1.0, and a lone 1.4 with poor values either side of it.
    xs = list(range(0, 21)) + [74, 77, 80, 83, 86]
    scores = [1.0] * 21 + [0.1, 0.2, 1.4, 0.2, 0.1]
    trials = pd.DataFrame({"trial": range(1, len(xs) + 1), "x": xs, "score": scores, "status": "ok"})
    best, stable = pick_settings(trials, space)
    assert best["x"] == 80
    assert stable["x"] <= 20
    assert len(neighbourhood_scores(trials, space)) == len(xs)
    assert params_of(stable, space) == {"x": int(stable["x"])}


def test_trials_record_phase_best_so_far_and_importance():
    from analysis.optimizer import param_importance

    space = SearchSpace((Param("a", "A", "int", 0, 50, 1), Param("b", "B", "int", 0, 50, 1)))
    index = pd.bdate_range("2020-01-01", periods=300)

    def evaluate(params):
        # Only a matters.
        return toy_backtest(index, 0.0005 - abs(params["a"] - 25) / 1e5)

    trials = run_search(evaluate, space, index[0], index[199], objective="cagr", n_trials=30, min_trades=1,
                        n_startup_trials=5)
    assert list(trials["phase"].iloc[:6]) == ["Latin hypercube start"] * 5 + ["guided"]
    assert trials["best_so_far"].is_monotonic_increasing
    importance = param_importance(trials, space, ["a", "b"])
    assert set(importance) == {"a", "b"}
    assert importance["a"] > importance["b"]
    assert param_importance(trials, space, ["a"]) == {}


def test_nearest_trials_excludes_itself_and_sorts_by_distance():
    from analysis.optimizer import nearest_trials

    space = SearchSpace((Param("x", "X", "int", 0, 100, 1),))
    trials = pd.DataFrame({"trial": [1, 2, 3, 4], "x": [10, 12, 50, 11], "score": [1.0, 0.9, 0.2, 0.8],
                           "status": "ok"})
    neighbours = nearest_trials(trials, space, trials.loc[0], k=2)
    assert list(neighbours["x"]) == [11, 12]
    assert list(neighbours["distance"].round(2)) == [0.01, 0.02]


def test_repeated_settings_are_not_rerun_or_counted_twice():
    space = SearchSpace((Param("x", "X", "int", 0, 4, 1),))
    index = pd.bdate_range("2020-01-01", periods=300)
    calls = []

    def evaluate(params):
        calls.append(params["x"])
        return toy_backtest(index, 0.0001 * (1 + params["x"]))

    trials = run_search(evaluate, space, index[0], index[199], objective="cagr", n_trials=20, min_trades=1)
    assert len(calls) == len(set(calls)) <= 5
    repeats = trials.loc[trials["repeat_of"].notna()]
    assert len(repeats) == 20 - len(calls)
    best, stable = pick_settings(trials, space)
    assert pd.isna(best["repeat_of"]) and pd.isna(stable["repeat_of"])
    assert len(neighbourhood_scores(trials, space)) == len(calls)


def test_latin_hypercube_uses_every_slice_once():
    from analysis.optimizer import latin_hypercube

    params = (Param("a", "A", "int", 0, 90, 10), Param("b", "B", "float", 0.0, 0.9, 0.1),
              Param("c", "C", "choice", choices=("x", "y")))
    points = latin_hypercube(params, 10, seed=3)
    assert sorted(point["a"] for point in points) == list(range(0, 100, 10))
    assert len({round(point["b"], 1) for point in points}) == 10
    assert sorted(point["c"] for point in points) == ["x"] * 5 + ["y"] * 5


def test_search_starts_with_the_latin_hypercube_points():
    from analysis.optimizer import latin_hypercube

    space = SearchSpace((Param("a", "A", "int", 0, 90, 10),))
    index = pd.bdate_range("2020-01-01", periods=300)
    trials = run_search(lambda params: toy_backtest(index, 0.0003), space, index[0], index[199], n_trials=12,
                        min_trades=1, n_startup_trials=10, seed=7)
    assert list(trials["a"].iloc[:10]) == [point["a"] for point in latin_hypercube(space.params, 10, 7)]


def test_weighted_objective_and_drawdown_limit():
    from analysis.optimizer import objective_latex, objective_score

    metrics = {"sharpe": 1.5, "cagr_pct": 12.0, "max_dd_pct": 20.0}
    assert objective_score(metrics, {"sharpe": 2, "cagr_pct": 0.1, "max_dd_pct": 0.05}) == 3 + 1.2 - 1.0
    assert objective_latex({"sharpe": 1, "max_dd_pct": 0.5}) == r"\text{score} = \text{Sharpe} - 0.5\times \text{max DD}\%"

    index = pd.bdate_range("2020-01-01", periods=300)
    falling = toy_backtest(index, -0.002)
    trials = run_search(lambda params: falling, SEARCH_SPACES["breakout"], index[0], index[199], n_trials=3,
                        min_trades=1, max_dd_limit=10)
    assert trials["status"].eq("drawdown over 10%").all()


def test_shortest_pullback_window_runs():
    from analysis.patterns import detect_signals
    from tests.conftest import make_bars
    df = make_bars(np.linspace(50, 100, 400))
    out = detect_signals(df, "pullback", pullback_bars=2)
    assert "Signal_Up" in out


def test_search_survives_a_setting_that_errors():
    space = SearchSpace((Param("x", "X", "int", 1, 10, 1),))
    index = pd.bdate_range("2020-01-01", periods=300)

    def evaluate(params):
        if params["x"] == 3:
            raise ValueError("edge")
        return toy_backtest(index, 0.0001 * params["x"])

    trials = run_search(evaluate, space, index[0], index[199], objective="cagr", n_trials=30, min_trades=1, seed=1)
    assert trials.loc[trials["x"].eq(3), "score"].isna().all()
    assert trials["score"].notna().any()


def test_search_on_several_stocks_scores_the_median():
    from analysis.optimizer import combined_metrics
    index = pd.bdate_range("2020-01-01", periods=300)
    one = period_metrics(toy_backtest(index, 0.001), index[0], index[199])
    metrics, score = combined_metrics([one], {"cagr": 1.0})
    assert score == one["cagr"]
    space = SearchSpace((Param("x", "X", "int", 1, 10, 1),))

    def evaluate(params):
        # Stock A likes high x, stock B low x, stock C is indifferent: the median favours neither extreme alone.
        return [toy_backtest(index, 0.0001 * params["x"]), toy_backtest(index, 0.0001 * (11 - params["x"])),
                toy_backtest(index, 0.0005)]

    trials = run_search(evaluate, space, index[0], index[199], objective="cagr", n_trials=20, min_trades=1, seed=1)
    ok = trials.loc[trials["status"].eq("ok")]
    for _, row in ok.iterrows():
        cagrs = sorted([(1 + 0.0001 * row["x"]) ** 252 - 1, (1 + 0.0001 * (11 - row["x"])) ** 252 - 1,
                        1.0005 ** 252 - 1])
        assert abs(row["score"] - cagrs[1]) < 1e-3
