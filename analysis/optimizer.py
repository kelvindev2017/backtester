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

# The score is a weighted sum of these terms, each measured over the tuning period. Max drawdown enters as a penalty.
SCORE_TERMS = {
    "sharpe": "Sharpe ratio",
    "sortino": "Sortino ratio",
    "calmar": "Return / max drawdown",
    "cagr_pct": "Annual return, CAGR (%)",
    "max_dd_pct": "Max drawdown (%)",
}
PENALTY_TERMS = {"max_dd_pct"}
OBJECTIVES = {
    "Sharpe ratio": {"sharpe": 1.0},
    "Sortino ratio": {"sortino": 1.0},
    "Return / max drawdown": {"calmar": 1.0},
    "Annual return (CAGR)": {"cagr_pct": 1.0},
    "Custom (weighted)": None,
}
# How each term is calculated, for the app to show (LaTeX). r = the strategy's bar returns, N = bars per year.
TERM_FORMULAS = {
    "sharpe": r"\text{Sharpe} = \frac{\overline{r}}{\sigma(r)}\sqrt{N}",
    "sortino": r"\text{Sortino} = \frac{\overline{r}}{\sqrt{\overline{\min(r,0)^2}}}\sqrt{N}",
    "calmar": r"\text{Return / max DD} = \frac{\text{CAGR}}{|\text{max drawdown}|}",
    "cagr_pct": r"\text{CAGR} = 100\left[\left(\frac{E_{end}}{E_{start}}\right)^{1/\text{years}} - 1\right]",
    "max_dd_pct": r"\text{Max DD} = 100\,\max_t\left(1 - \frac{E_t}{\max_{s \le t} E_s}\right)",
}


def objective_score(metrics, weights):
    """The weighted score: the sum of weight x term, with max drawdown (%) subtracted."""

    total = 0.0
    for term, weight in weights.items():
        if not weight:
            continue
        value = metrics.get(term, np.nan)
        total += (-weight if term in PENALTY_TERMS else weight) * value
    return float(total)


def objective_latex(weights):
    """The score's formula with the chosen weights, as LaTeX."""

    names = {"sharpe": r"\text{Sharpe}", "sortino": r"\text{Sortino}", "calmar": r"\frac{\text{CAGR}}{|\text{max DD}|}",
             "cagr_pct": r"\text{CAGR}\%", "max_dd_pct": r"\text{max DD}\%"}
    parts = []
    for term, weight in weights.items():
        if not weight:
            continue
        sign = "-" if term in PENALTY_TERMS else "+"
        factor = "" if weight == 1 else f"{weight:g}" + r"\times "
        parts.append(f" {sign} {factor}{names[term]}")
    formula = "".join(parts).strip()
    if formula.startswith("+"):
        formula = formula[1:].strip()
    return r"\text{score} = " + (formula or "0")
PERIODS_PER_YEAR = {"Daily": 252, "Weekly": 52, "Monthly": 12}


def period_metrics(backtest, start, end, periods_per_year=252):
    """Return, drawdown, Sharpe and trade count of ``backtest`` (``Equity``, ``Buy_Signal``) between two dates.

    The equity is rebased at ``start``, so a period is scored on its own, wherever the earlier period left off.
    """

    window = backtest.loc[(backtest.index >= start) & (backtest.index <= end)]
    equity = window["Equity"].astype(float)
    if len(equity) < 2 or not np.isfinite(equity.iloc[0]) or equity.iloc[0] <= 0:
        return {"return": np.nan, "cagr": np.nan, "max_dd": np.nan, "sharpe": np.nan, "sortino": np.nan,
                "calmar": np.nan, "cagr_pct": np.nan, "max_dd_pct": np.nan, "trades": 0}
    returns = equity.pct_change().dropna()
    total = equity.iloc[-1] / equity.iloc[0] - 1
    years = max(len(returns) / periods_per_year, 1e-9)
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    max_dd = float((equity / equity.cummax() - 1).min())
    std = returns.std()
    sharpe = float(returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    downside = float(np.sqrt((np.minimum(returns.to_numpy(dtype=float), 0) ** 2).mean()))
    sortino = float(returns.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else 0.0
    calmar = cagr / abs(max_dd) if max_dd < 0 else (cagr * 100 if cagr > 0 else 0.0)
    trades = int(window["Buy_Signal"].sum()) if "Buy_Signal" in window else 0
    return {"return": float(total), "cagr": float(cagr), "max_dd": max_dd, "sharpe": sharpe, "sortino": sortino,
            "calmar": float(calmar), "cagr_pct": float(cagr) * 100, "max_dd_pct": abs(max_dd) * 100,
            "trades": trades}


def split_date(index, tune_fraction):
    """The last date of the tuning period: ``tune_fraction`` of the bars from the start."""

    position = int(np.clip(round(len(index) * tune_fraction), 1, len(index) - 1)) - 1
    return index[position]


def latin_hypercube(params, n, seed=0):
    """``n`` starting settings spread by Latin hypercube sampling: each setting's range is cut into ``n`` equal slices
    and every slice is used exactly once, so the start covers each range evenly instead of clumping by chance.

    Values are snapped to the setting's step; a choice setting cycles through its choices the same way.
    """

    rng = np.random.default_rng(seed)
    points = [{} for _ in range(n)]
    for param in params:
        positions = (rng.permutation(n) + rng.uniform(size=n)) / n  # one point in each of n slices, shuffled
        for point, position in zip(points, positions):
            if param.kind == "choice":
                point[param.key] = param.choices[min(int(position * len(param.choices)), len(param.choices) - 1)]
                continue
            steps = int(round((param.high - param.low) / param.step))
            value = param.low + min(int(position * (steps + 1)), steps) * param.step
            point[param.key] = int(round(value)) if param.kind == "int" else round(float(value), 10)
    return points


def _suggest(trial, param, fixed):
    if param.key in fixed:
        return fixed[param.key]
    if param.kind == "choice":
        return trial.suggest_categorical(param.key, list(param.choices))
    if param.kind == "int":
        return trial.suggest_int(param.key, int(param.low), int(param.high), step=int(param.step))
    return trial.suggest_float(param.key, float(param.low), float(param.high), step=float(param.step))


N_STARTUP_TRIALS = 10  # Optuna's TPE default: this many random trials before the model starts guiding.


def run_search(evaluate, space, tune_start, tune_end, objective="sharpe", n_trials=60, min_trades=3,
               fixed=None, periods_per_year=252, seed=0, progress=None, n_startup_trials=N_STARTUP_TRIALS,
               start="lhs", max_dd_limit=None):
    """Search ``space`` with Optuna's TPE sampler, scoring each trial on the tuning period only.

    ``objective`` is a term name ("sharpe") or a dict of term weights (see ``SCORE_TERMS``). ``fixed`` maps keys to
    values that are held, not searched. Trials that break an ordering rule (fast MA above the slow MA), trade fewer
    than ``min_trades`` times or fall further than ``max_dd_limit`` % from a peak are recorded without a score and
    steer the search away. The first ``n_startup_trials`` are a Latin hypercube (``start="lhs"``) or random.

    Returns a DataFrame with one row per trial: the parameters, ``score``, the tuning-period metrics and ``status``.
    """

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    weights = {objective: 1.0} if isinstance(objective, str) else dict(objective)
    fixed = dict(fixed or {})
    rows = []
    start_label = "Latin hypercube start" if start == "lhs" else "random start"
    # TPE often proposes a setting it has already tried (the steps are discrete). The backtest would give the same
    # result, so a repeat is answered from this cache and marked, and counts once when picking settings.
    seen = {}

    def objective_fn(trial):
        params = {param.key: _suggest(trial, param, fixed) for param in space.params}
        phase = start_label if trial.number < n_startup_trials else "guided"
        row = {"trial": trial.number + 1, "phase": phase, **params}
        signature = tuple(sorted(params.items()))
        if signature in seen:
            first = seen[signature]
            rows.append({**first, **row, "repeat_of": first["trial"]})
            if first["status"] != "ok":
                raise optuna.TrialPruned()
            return first["score"]
        try:
            return _score(row, params)
        finally:
            if rows and rows[-1]["trial"] == row["trial"]:
                seen[signature] = rows[-1]

    def _score(row, params):
        broken = [(a, b) for a, b in space.less_than if params[a] >= params[b]]
        if broken:
            rows.append({**row, "score": np.nan, "status": "fast MA not below slow MA"})
            raise optuna.TrialPruned()
        metrics = period_metrics(evaluate(params), tune_start, tune_end, periods_per_year)
        if metrics["trades"] < min_trades:
            rows.append({**row, **metrics, "score": np.nan, "status": f"fewer than {min_trades} trades"})
            raise optuna.TrialPruned()
        if max_dd_limit and metrics["max_dd_pct"] > max_dd_limit:
            rows.append({**row, **metrics, "score": np.nan, "status": f"drawdown over {max_dd_limit:g}%"})
            raise optuna.TrialPruned()
        score = objective_score(metrics, weights)
        if not np.isfinite(score):
            rows.append({**row, **metrics, "score": np.nan, "status": "no score"})
            raise optuna.TrialPruned()
        rows.append({**row, **metrics, "score": float(score), "status": "ok"})
        return float(score)

    callbacks = [lambda study, trial: progress(trial.number + 1, n_trials)] if progress else []
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    if start == "lhs" and n_startup_trials:
        searched = [param for param in space.params if param.key not in fixed]
        for point in latin_hypercube(searched, min(n_startup_trials, n_trials), seed):
            study.enqueue_trial(point, skip_if_exists=False)
    study.optimize(objective_fn, n_trials=n_trials, callbacks=callbacks)
    trials = pd.DataFrame(rows)
    if "repeat_of" not in trials:
        trials["repeat_of"] = np.nan
    if not trials.empty:
        # The best tuning score found up to each trial: how fast the search closed in.
        trials["best_so_far"] = trials["score"].cummax()
    return trials


def unique_scored(trials):
    """Scored trials, each distinct setting once (repeats of an earlier trial dropped)."""

    scored = trials.loc[trials["status"].eq("ok")]
    if "repeat_of" in scored:
        scored = scored.loc[scored["repeat_of"].isna()]
    return scored


def neighbourhood_scores(trials, space, k=5):
    """Each scored trial's average score over itself and its ``k`` nearest scored trials in parameter space.

    Numeric parameters are scaled to 0-1 over their search range; a different choice counts as distance 1. A high
    neighbourhood score means the setting sits on a plateau of good results rather than a lone spike.
    """

    scored = unique_scored(trials).copy()
    if scored.empty:
        return pd.Series(dtype=float)
    distances = _distances(scored, space)
    scores = scored["score"].to_numpy(dtype=float)
    count = min(k + 1, len(scored))
    nearest = np.argsort(distances, axis=1)[:, :count]
    return pd.Series(scores[nearest].mean(axis=1), index=scored.index)


def _distances(scored, space):
    """Distances between trials: numeric settings scaled 0-1 over their range, a different choice counts 1."""

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
    return np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))


def nearest_trials(trials, space, row, k=5):
    """The ``k`` scored trials nearest to ``row`` (a scored trial), with their ``distance`` (0 = the same values)."""

    scored = unique_scored(trials)
    distances = _distances(scored, space)
    position = scored.index.get_loc(row.name)
    order = [index for index in np.argsort(distances[position], kind="stable") if index != position][:k]
    result = scored.iloc[order].copy()
    result["distance"] = distances[position][order]
    return result


def param_importance(trials, space, searched):
    """How much each searched setting moved the tuning score (Optuna's PED-ANOVA importances, summing to 1).

    Empty when fewer than two settings were searched or too few trials were scored to tell.
    """

    import optuna
    from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution

    params = [param for param in space.params if param.key in searched]
    scored = unique_scored(trials)
    if len(params) < 2 or len(scored) < 10:
        return {}
    distributions = {}
    for param in params:
        if param.kind == "choice":
            distributions[param.key] = CategoricalDistribution(list(param.choices))
        elif param.kind == "int":
            distributions[param.key] = IntDistribution(int(param.low), int(param.high), step=int(param.step))
        else:
            distributions[param.key] = FloatDistribution(float(param.low), float(param.high), step=float(param.step))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    for _, row in scored.iterrows():
        values = params_of(row, SearchSpace(tuple(params)))
        study.add_trial(optuna.trial.create_trial(params=values, distributions=distributions, value=float(row["score"])))
    try:
        return dict(optuna.importance.get_param_importances(study))
    except Exception:
        return {}


def pick_settings(trials, space, k=5):
    """The best single trial and the most stable one (highest neighbourhood score), as rows of ``trials``."""

    scored = unique_scored(trials)
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
