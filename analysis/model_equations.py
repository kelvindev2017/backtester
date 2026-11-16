"""The equation of each Valuation Model block, as written and with the latest numbers filled in.

``EQUATIONS`` holds each block's equations in LaTeX (for ``st.latex``); ``worked_equations`` puts a model run's
latest values into them, so a reader can follow one number from the results to the price. ``stock_pay_haircut``
is the default EPS haircut: stock-based pay as a share of net income.
"""

import numpy as np
import pandas as pd

from analysis.valuation_model import SCENARIOS, item_effects, latest_values, scenario_weights

# Each string is one display line, kept short enough for a phone; aligned blocks break longer ones.
EQUATIONS = {
    "income_map": [
        r"g = \left(\frac{\text{Revenue}_{\text{last 4q}}}{\text{Revenue}_{\text{4q before}}} - 1\right) \times 100",
        r"m = \frac{\text{Net income}_{\text{last 4q}}}{\text{Revenue}_{\text{last 4q}}} \times 100",
    ],
    "earnings_quality": [
        r"\text{EPS}_{\text{clean}} = \text{EPS}_{\text{last 4q}} \times \left(1 - \frac{h}{100}\right)",
        r"h_{\text{default}} = \frac{\text{Stock-based pay}_{\text{last 4q}}}{\text{Net income}_{\text{last 4q}}} \times 100",
    ],
    "item_tracker": [r"\text{Items} = \text{modelled} + \text{parked}"],
    "triage_gate": [
        r"\begin{aligned} \text{EPS item in if } & p \times |\text{impact}| \ge \text{bar}_{\text{EPS}} \\ "
        r"\text{P/E item in if } & |\text{points}| \ge \text{bar}_{\text{P/E}} \end{aligned}",
    ],
    "item_router": [
        r"e = \sum \pm\, p \times \text{impact} \qquad a = \sum \pm\, \text{points}",
    ],
    "scenario_engine": [
        r"g_{\text{base}} = \min\!\big(\text{cap},\ \max(\text{floor},\ s \times g)\big)",
        r"g_{\text{bear}} = g_{\text{base}} - \text{gap}_{\text{bear}}",
        r"g_{\text{bull}} = g_{\text{base}} + \text{gap}_{\text{bull}}",
        r"\begin{aligned} \text{EPS}_{c} = {} & \text{EPS}_{\text{clean}} \times \left(1 + \frac{g_c}{100}\right) \\ "
        r"& \times \frac{m + \Delta m_c}{m} \times \left(1 + \frac{e}{100}\right) \end{aligned}",
    ],
    "risk_weighting": [
        r"\begin{aligned} \text{EPS}_{\text{risk}} = {} & w_{\text{bear}} \times \text{EPS}_{\text{bear}} \\ "
        r"& + w_{\text{base}} \times \text{EPS}_{\text{base}} \\ & + w_{\text{bull}} \times \text{EPS}_{\text{bull}} \end{aligned}",
    ],
    "multiple_builder": [
        r"\begin{aligned} \text{anchor} = {} & k \times \text{P/E}_{\text{peers}} \\ & + (1 - k) \times \text{P/E}_{\text{own}} \end{aligned}",
        r"\begin{aligned} \text{P/E}_{\text{acc}} = {} & \text{anchor} + a - d_q, \\ & \text{kept between floor and cap} \end{aligned}",
    ],
    "product": [r"\text{Model price} = \text{EPS}_{\text{risk}} \times \text{P/E}_{\text{acc}}"],
    "market_check": [r"\text{Gap} = \left(\frac{\text{Model price}}{\text{Close}} - 1\right) \times 100"],
}

# What each symbol stands for, shown under the equations.
SYMBOLS = {
    "income_map": "g: revenue growth %, m: net margin %.",
    "earnings_quality": "h: EPS haircut %. The P/E deduction d_q goes to the Multiple builder.",
    "triage_gate": "p: probability, impact: EPS impact %, points: P/E points.",
    "item_router": "e: expected EPS change % from event items, a: P/E points from general items (+ product, - risk).",
    "scenario_engine": "s: base growth as a share of trailing growth, g: revenue growth %, c: bear, base or bull, "
                       "Δm_c: that case's net margin change in points, m: net margin %, e: event items' EPS change %.",
    "risk_weighting": "w: scenario weights, rescaled to add up to 1.",
    "multiple_builder": "k: share of the anchor from peers, P/E peers and own: their medians over the look-back, "
                        "a: P/E points from register items, d_q: earnings quality P/E deduction. P/E is trailing or "
                        "forward, as chosen.",
}


def stock_pay_haircut(stock_pay, net_income, periods=4):
    """Stock-based pay of the last ``periods`` reports as a % of net income over the same reports.

    Both are series by period end. NaN when either is missing or net income is not positive (no profit to compare
    with); capped at 90, the haircut box's limit.
    """

    pay = pd.to_numeric(pd.Series(stock_pay, dtype=float), errors="coerce").dropna().sort_index()
    income = pd.to_numeric(pd.Series(net_income, dtype=float), errors="coerce").dropna().sort_index()
    shared = pay.index.intersection(income.index)
    if len(shared) < periods:
        return np.nan
    shared = shared.sort_values()[-periods:]
    total_income = income.loc[shared].sum()
    if not np.isfinite(total_income) or total_income <= 0:
        return np.nan
    return float(np.clip(abs(pay.loc[shared].sum()) / total_income * 100, 0, 90))


def _n(value, digits=2):
    return f"{value:,.{digits}f}" if value is not None and np.isfinite(value) else r"\text{n/a}"


def worked_equations(result, block, on=None):
    """The block's equations with the run's values on ``on`` (default: the last close) filled in."""

    values = latest_values(result, on)
    params = result["params"]
    if not values:
        return []
    get = lambda name: values.get(name, np.nan)
    growth, margin = get("revenue_yoy"), get("net_margin")
    used_growth = growth if np.isfinite(growth) else params["fallback_yoy"]
    used_margin = margin if np.isfinite(margin) and margin > 0 else params["fallback_margin"]
    events = get("event_effect")
    events = events if np.isfinite(events) else 0.0
    if block == "income_map":
        return [rf"g = {_n(growth, 1)}\% \qquad m = {_n(margin, 1)}\%"]
    if block == "earnings_quality":
        return [rf"\text{{EPS}}_{{\text{{clean}}}} = {_n(get('ttm_eps'))} \times \left(1 - \frac{{{_n(params['eps_haircut_pct'], 1)}}}{{100}}\right)"
                rf" = {_n(get('clean_eps'))}", rf"d_q = {_n(params['quality_pe_pts'], 1)} \text{{ P/E points}}"]
    if block == "scenario_engine":
        base = float(np.clip(params["growth_follow"] * used_growth, params["growth_floor"], params["growth_cap"]))
        lines = [rf"\begin{{aligned}} g_{{\text{{base}}}} = {{}} & \min\!\big({_n(params['growth_cap'], 0)},\ \max({_n(params['growth_floor'], 0)},\ "
                 rf"{_n(params['growth_follow'])} \times {_n(used_growth, 1)})\big) \\ = {{}} & {_n(base, 1)}\% \end{{aligned}}"]
        for name in SCENARIOS:
            key = name.lower()
            points = params[f"{key}_margin_pts"]
            lines.append(
                rf"\begin{{aligned}} \text{{EPS}}_{{\text{{{key}}}}} = {{}} & {_n(get('clean_eps'))} \times {_n(1 + get(f'growth_{key}') / 100, 3)} \\ "
                rf"& \times \frac{{{_n(used_margin, 1)} {'+' if points >= 0 else '-'} {_n(abs(points), 1)}}}{{{_n(used_margin, 1)}}} "
                rf"\times {_n(1 + events / 100, 3)} \\ = {{}} & {_n(get(f'eps_{key}'))} \end{{aligned}}")
        return lines
    if block == "risk_weighting":
        weights = scenario_weights(params)
        terms = r" \\ & + ".join(rf"{_n(weight, 2)} \times {_n(get(f'eps_{name.lower()}'))}" for weight, name in zip(weights, SCENARIOS))
        return [rf"\begin{{aligned}} \text{{EPS}}_{{\text{{risk}}}} = {{}} & {terms} \\ = {{}} & {_n(get('risk_adjusted_eps'))} \end{{aligned}}"]
    if block in ("item_router", "triage_gate", "item_tracker"):
        _, points = item_effects(result["items"], result["daily"].index[-1] if on is None else on)
        return [rf"e = {_n(events, 1)}\% \qquad a = {_n(points, 1)} \text{{ pts}}",
                rf"\text{{{int(get('items_modelled') or 0)} modelled, {int(get('items_parked') or 0)} parked}}"]
    if block == "multiple_builder":
        share = float(np.clip(params["peer_weight"], 0, 1))
        anchor = result["daily"]["anchor"].dropna()
        anchor = float(anchor.iloc[-1]) if len(anchor) else np.nan
        sign = '+' if not get('held_pe_adjust') < 0 else '-'
        peer, own = get('peer_anchor'), get('own_anchor')
        if np.isfinite(peer) and np.isfinite(own):
            mix = rf"{_n(share)} \times {_n(peer, 1)} + {_n(1 - share)} \times {_n(own, 1)}"
        else:  # only one of them exists: the anchor is that one
            mix = rf"\text{{{'peers' if np.isfinite(peer) else 'own'} only: }} {_n(peer if np.isfinite(peer) else own, 1)}"
        return [rf"\begin{{aligned}} \text{{anchor}} = {{}} & {mix} \\ = {{}} & {_n(anchor, 1)} \end{{aligned}}",
                rf"\begin{{aligned}} \text{{P/E}}_{{\text{{acc}}}} = {{}} & {_n(anchor, 1)} {sign} {_n(abs(get('held_pe_adjust')), 1)} - {_n(abs(get('held_quality_pe')), 1)} \\ "
                rf"& \text{{kept in }} {_n(params['pe_floor'], 0)} \text{{ to }} {_n(params['pe_cap'], 0)} \\ = {{}} & {_n(get('acceptable_pe'), 1)} \end{{aligned}}"]
    if block == "product":
        return [rf"{_n(get('held_eps'))} \times {_n(get('acceptable_pe'), 1)} = {_n(get('model_price'))}"]
    if block == "market_check":
        return [rf"\left(\frac{{{_n(get('model_price'))}}}{{{_n(get('close'))}}} - 1\right) \times 100 = {_n(get('residual_pct'), 1)}\%"]
    return []
