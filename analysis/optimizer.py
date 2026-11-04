"""Bayesian parameter search for the Backtest tab's signal rules, with a forward test.

The date range is split in two. The search (Optuna's TPE sampler, a form of Bayesian optimization) only ever sees
the score of the first part, the tuning period. The second part, the forward period, is scored once, after the
search, for the settings it picked. A setting that only fits the tuning period's noise shows up there.

The search does not run backtests itself: the caller passes ``evaluate(params)``, which returns the backtest of
those settings over the whole date range (columns ``Equity`` and ``Buy_Signal``), so the Backtest tab's own
engine, costs included, is the one being tuned.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Param:
    """One searchable setting. ``key`` is the setting's Session State key, so a result can be applied as is."""

    key: str
    label: str
    kind: str  # "int", "float" or "choice"
    low: float = 0
    high: float = 0
    step: float = 1
    choices: tuple = ()


@dataclass(frozen=True)
class SearchSpace:
    params: tuple
    # Pairs (a, b) of keys that must satisfy params[a] < params[b], e.g. the fast MA below the slow MA.
    less_than: tuple = field(default=())


SEARCH_SPACES = {
    "ma_rule": SearchSpace((
        Param("fast_period", "Fast MA period", "int", 5, 100, 5),
        Param("slow_period", "Slow MA period", "int", 50, 300, 10),
        Param("fast_type", "Fast MA type", "choice", choices=("SMA", "EMA")),
        Param("slow_type", "Slow MA type", "choice", choices=("SMA", "EMA")),
    ), less_than=(("fast_period", "slow_period"),)),
    "breakout": SearchSpace((
        Param("breakout_lookback", "Breakout lookback (bars)", "int", 5, 100, 5),
        Param("breakout_exit_ma", "Exit SMA (bars)", "int", 20, 300, 10),
    )),
    "pullback": SearchSpace((
        Param("trend_mid", "Trend fast MA", "int", 20, 100, 5),
        Param("trend_slow", "Trend slow MA", "int", 100, 300, 10),
        Param("pullback_ema", "Pullback EMA", "int", 5, 50, 1),
        Param("swing_bars", "Swing high lookback", "int", 5, 60, 1),
        Param("pullback_bars", "Pullback window", "int", 2, 15, 1),
        Param("min_depth_atr", "Min dip depth (ATR)", "float", 0.5, 4.0, 0.25),
    ), less_than=(("trend_mid", "trend_slow"),)),
    "reversal": SearchSpace((
        Param("breakdown_lookback", "Breakdown lookback (bars)", "int", 5, 60, 1),
        Param("trend_mid", "Trend fast MA", "int", 20, 100, 5),
        Param("trend_slow", "Trend slow MA", "int", 100, 300, 10),
        Param("reversal_max_hold", "Max bars held", "int", 2, 40, 1),
    ), less_than=(("trend_mid", "trend_slow"),)),
    "day_breakout": SearchSpace((
        Param("signal_day_breakout_lookback", "Prior N-day high", "int", 2, 60, 1),
        Param("signal_day_breakout_exit_pct", "Exit below high (% of range)", "int", 10, 90, 5),
    )),
    "macd_bollinger": SearchSpace((
        Param("macd_ma", "MA and Bollinger period", "int", 5, 60, 1),
        Param("macd_widen_pct", "Bands % wider", "float", 0.0, 50.0, 5.0),
        Param("macd_widen_days", "than N days before", "int", 1, 15, 1),
        Param("macd_stop_pct", "Stop % below entry", "float", 0.0, 15.0, 0.5),
        Param("macd_cooldown", "Days to wait after a sell", "int", 0, 10, 1),
    )),
}

OBJECTIVES = {
    "Sharpe ratio": "sharpe",
    "Return / max drawdown": "calmar",
    "Annual return (CAGR)": "cagr",
}
PERIODS_PER_YEAR = {"Daily": 252, "Weekly": 52, "Monthly": 12}


def period_metrics(backtest, start, end, periods_per_year=252):
    """Return, drawdown, Sharpe and trade count of ``backtest`` (``Equity``, ``Buy_Signal``) between two dates.

    The equity is rebased at ``start``, so a period is scored on its own, wherever the earlier period left off.
    """

    window = backtest.loc[(backtest.index >= start) & (backtest.index <= end)]
    equity = window["Equity"].astype(float)
    if len(equity) < 2 or not np.isfinite(equity.iloc[0]) or equity.iloc[0] <= 0:
        return {"return": np.nan, "cagr": np.nan, "max_dd": np.nan, "sharpe": np.nan, "calmar": np.nan, "trades": 0}
    returns = equity.pct_change().dropna()
    total = equity.iloc[-1] / equity.iloc[0] - 1
    years = max(len(returns) / periods_per_year, 1e-9)
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    max_dd = float((equity / equity.cummax() - 1).min())
    std = returns.std()
    sharpe = float(returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    calmar = cagr / abs(max_dd) if max_dd < 0 else (cagr * 100 if cagr > 0 else 0.0)
    trades = int(window["Buy_Signal"].sum()) if "Buy_Signal" in window else 0
    return {"return": float(total), "cagr": float(cagr), "max_dd": max_dd, "sharpe": sharpe, "calmar": float(calmar),
            "trades": trades}


def split_date(index, tune_fraction):
    """The last date of the tuning period: ``tune_fraction`` of the bars from the start."""

    position = int(np.clip(round(len(index) * tune_fraction), 1, len(index) - 1)) - 1
    return index[position]


def _suggest(trial, param, fixed):
    if param.key in fixed:
        return fixed[param.key]
    if param.kind == "choice":
        return trial.suggest_categorical(param.key, list(param.choices))
    if param.kind == "int":
        return trial.suggest_int(param.key, int(param.low), int(param.high), step=int(param.step))
    return trial.suggest_float(param.key, float(param.low), float(param.high), step=float(param.step))


def run_search(evaluate, space, tune_start, tune_end, objective="sharpe", n_trials=60, min_trades=3,
               fixed=None, periods_per_year=252, seed=0, progress=None):
    """Search ``space`` with Optuna's TPE sampler, scoring each trial on the tuning period only.

    ``fixed`` maps keys to values that are held, not searched. Trials that break an ordering rule (fast MA above the
    slow MA) or trade fewer than ``min_trades`` times are recorded without a score and steer the search away.

    Returns a DataFrame with one row per trial: the parameters, ``score``, the tuning-period metrics and ``status``.
    """

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fixed = dict(fixed or {})
    rows = []

    def objective_fn(trial):
        params = {param.key: _suggest(trial, param, fixed) for param in space.params}
        row = {"trial": trial.number + 1, **params}
        broken = [(a, b) for a, b in space.less_than if params[a] >= params[b]]
        if broken:
            rows.append({**row, "score": np.nan, "status": "fast MA not below slow MA"})
            raise optuna.TrialPruned()
        metrics = period_metrics(evaluate(params), tune_start, tune_end, periods_per_year)
        if metrics["trades"] < min_trades:
            rows.append({**row, **metrics, "score": np.nan, "status": f"fewer than {min_trades} trades"})
            raise optuna.TrialPruned()
        score = metrics[objective]
        if not np.isfinite(score):
            rows.append({**row, **metrics, "score": np.nan, "status": "no score"})
            raise optuna.TrialPruned()
        rows.append({**row, **metrics, "score": float(score), "status": "ok"})
        return float(score)

    callbacks = [lambda study, trial: progress(trial.number + 1, n_trials)] if progress else []
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective_fn, n_trials=n_trials, callbacks=callbacks)
    return pd.DataFrame(rows)


def neighbourhood_scores(trials, space, k=5):
    """Each scored trial's average score over itself and its ``k`` nearest scored trials in parameter space.

    Numeric parameters are scaled to 0-1 over their search range; a different choice counts as distance 1. A high
    neighbourhood score means the setting sits on a plateau of good results rather than a lone spike.
    """

    scored = trials.loc[trials["status"].eq("ok")].copy()
    if scored.empty:
        return pd.Series(dtype=float)
    columns = []
    for param in space.params:
        if param.key not in scored:
            continue
        if param.kind == "choice":
            for choice in param.choices:
                columns.append(scored[param.key].eq(choice).astype(float).to_numpy() / np.sqrt(2))
        else:
            span = (param.high - param.low) or 1
            columns.append((scored[param.key].astype(float).to_numpy() - param.low) / span)
    points = np.column_stack(columns) if columns else np.zeros((len(scored), 1))
    distances = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    scores = scored["score"].to_numpy(dtype=float)
    count = min(k + 1, len(scored))
    nearest = np.argsort(distances, axis=1)[:, :count]
    return pd.Series(scores[nearest].mean(axis=1), index=scored.index)


def pick_settings(trials, space, k=5):
    """The best single trial and the most stable one (highest neighbourhood score), as rows of ``trials``."""

    scored = trials.loc[trials["status"].eq("ok")]
    if scored.empty:
        return None, None
    best = scored.loc[scored["score"].idxmax()]
    neighbourhood = neighbourhood_scores(trials, space, k)
    stable = trials.loc[neighbourhood.idxmax()].copy()
    stable["neighbourhood"] = float(neighbourhood.max())
    best = best.copy()
    best["neighbourhood"] = float(neighbourhood.loc[best.name])
    return best, stable


def params_of(row, space):
    """The parameter values of a trial row, with numpy types turned into plain Python ones."""

    params = {}
    for param in space.params:
        value = row[param.key]
        if param.kind == "int":
            value = int(value)
        elif param.kind == "float":
            value = float(value)
        params[param.key] = value
    return params
