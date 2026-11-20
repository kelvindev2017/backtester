"""The equation of each Valuation Model block, as written and with the latest numbers filled in.

``EQUATIONS`` holds each block's equations in LaTeX (for ``st.latex``); ``worked_equations`` puts a model run's
latest values into them, so a reader can follow one number from the results to the price. ``stock_pay_haircut``
is the default EPS haircut: stock-based pay as a share of net income.
"""

import numpy as np
import pandas as pd

from analysis.valuation_model import (
    SCENARIOS, case_effects, latest_values, rate_adjusted_pe, rate_weight, scenario_weights,
)

# Each string is one display line, kept short enough for a phone; aligned blocks break longer ones.
EQUATIONS = {
    "income_map": [
        r"g = \left(\frac{\text{Revenue}_{\text{last 4q}}}{\text{Revenue}_{\text{4q before}}} - 1\right) \times 100",
        r"m = \frac{\text{Net income}_{\text{last 4q}}}{\text{Revenue}_{\text{last 4q}}} \times 100",
    ],
    "earnings_quality": [
        r"\text{EPS}_{\text{clean}} = \text{EPS}_{\text{last 4q}} \times \left(1 - \frac{h}{100}\right)",
        r"h_{\text{default}} = \frac{\text{SBC}_{\text{last 4q}}}{\text{Net income}_{\text{last 4q}} + \text{SBC}_{\text{last 4q}}} \times 100",
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
        r"g_{\text{base}} = g_{\text{fwd}} \ \text{(yours) or } \min\!\big(\text{cap},\ \max(\text{floor},\ s \times g)\big)",
        r"m_{\text{base}} = m_{\text{fwd}} \ \text{(yours) or } m + \Delta m_{\text{base}}",
        r"g_c = g_{\text{base}} + \text{adj}_c \qquad m_c = m_{\text{base}} + \Delta m_c",
        r"\begin{aligned} \text{EPS}_{c} = {} & \text{EPS}_{\text{clean}} \times \left(1 + \frac{g_c}{100}\right) \\ "
        r"& \times \frac{m_c}{m} \times \left(1 + \frac{e_c}{100}\right) \end{aligned}",
    ],
    "risk_weighting": [
        r"\begin{aligned} \text{EPS}_{\text{risk}} = {} & w_{\text{bear}} \times \text{EPS}_{\text{bear}} \\ "
        r"& + w_{\text{base}} \times \text{EPS}_{\text{base}} \\ & + w_{\text{bull}} \times \text{EPS}_{\text{bull}} \end{aligned}",
    ],
    "rate_model": [
        r"w_N = \frac{1 - e^{-\lambda N}}{\lambda N}, \qquad \lambda = \frac{\ln 2}{h}",
        r"y_N = r_n + (r_f - r_n) \times w_N + tp",
        r"\Delta y_{\text{own}} = y_N - \operatorname{median}_{\text{look-back}}(y_N)",
        r"\Delta y_{\text{peers}} = y_N(r_f) - y_N(r_{\text{today}})",
    ],
    "multiple_builder": [
        r"n = n_{\min} + (n_{\max} - n_{\min}) \times \operatorname{clip}\!\left(\frac{GM - GM_{\text{low}}}{GM_{\text{high}} - GM_{\text{low}}}, 0, 1\right)",
        r"g_t = g + (g_T - g)\,\frac{t}{n}, \qquad E_1 = 1, \quad E_{t+1} = E_t (1 + g_t)",
        r"\text{P/E}_{\text{fund}} = \sum_{t=1}^{n} \frac{E_t}{(1+r)^t} + \frac{E_{n+1}}{(r - g_T)(1+r)^n}, \qquad r = y_N + \text{ERP}",
        r"\text{P/E}^{r}_{\text{own}} = \frac{1}{1 / \text{P/E}_{\text{own}} + t \times \Delta y_{\text{own}} / 100}",
        r"\begin{aligned} \text{anchor} = {} & k \times \text{P/E}_{\text{outside}} \\ & + (1 - k) \times \text{P/E}^{r}_{\text{own}} \end{aligned}",
        r"\begin{aligned} \text{P/E}_{\text{acc}} = {} & \text{anchor} + a - d_q, \\ & \text{kept between floor and cap} \end{aligned}",
    ],
    "product": [r"\text{Model price} = \text{EPS}_{\text{risk}} \times \text{P/E}_{\text{acc}}"],
    "market_check": [r"\text{Gap} = \left(\frac{\text{Model price}}{\text{Close}} - 1\right) \times 100"],
}

# What each symbol stands for, shown under the equations.
SYMBOLS = {
    "income_map": "g: revenue growth %, m: net margin %.",
    "earnings_quality": "h: EPS haircut %, SBC: stock-based pay (adjusted EPS leaves it out, so h_default takes it back). The P/E deduction d_q goes to the Multiple builder.",
    "triage_gate": "p: probability, impact: EPS impact %, points: P/E points.",
    "item_router": "e: expected EPS change % from event items (per case: the forward probability plus the item's bull "
                   "or bear adjustment; shown weighted), a: P/E points (+ product, - risk), weighted over the cases.",
    "scenario_engine": "g, m: current (trailing) revenue growth and net margin %, g_fwd, m_fwd: your forward values "
                       "(latest quarter only; blank = the auto rule with s: share of trailing growth), c: bear, base or "
                       "bull, adj_c and Δm_c: the case's growth and margin adjustment in points (0 for base), e_c: event "
                       "items' EPS change % at that case's probability.",
    "risk_weighting": "w: scenario weights, rescaled to add up to 1.",
    "rate_model": "r_f: policy rate used (your forward rate on the last day, the actual rate before), r_n: neutral "
                  "rate, h: half-life of a rate's gap to neutral in years, N: yield maturity in years, w_N: share of "
                  "the policy rate that reaches the N-year yield, tp: term premium (it cancels out of both Δy). "
                  "Δy_own compares with the years the own P/E median covers; Δy_peers is only the forward rate's "
                  "surprise, because the peers' P/E is today's and already prices today's rate.",
    "multiple_builder": "Outside view: P/E_fund (default) or the peers' median P/E (moved by the forward rate's yield change "
                        "like the own one). g: revenue growth a year over the growth history (within floor and cap), GM: gross "
                        "margin %, n: years the growth lasts, g_T: terminal growth, y_N: the Rate model's yield (or the "
                        "base yield), ERP: equity risk premium. x: own or peers, Δy_x: yield change from the Rate model, t: share of it passed to the P/E "
                        "(the earnings yield 1/P/E moves point for point with the required return). k: share of the anchor from peers, P/E peers and own: their medians over the look-back, "
                        "a: P/E points from register items, d_q: earnings quality P/E deduction. P/E is trailing or "
                        "forward, as chosen.",
}


def stock_pay_haircut(stock_pay, net_income, periods=4):
    """Stock-based pay of the last ``periods`` reports as a % of net income plus that pay, over the same reports.

    The haircut is applied to reported EPS, which for most companies is the adjusted figure that leaves stock-based
    pay out: adjusted earnings are about GAAP net income + stock-based pay, so the share of them that is stock pay is
    pay / (net income + pay), and removing it lands near GAAP earnings. (Pay / net income alone would take off too
    much: for a company whose stock pay is as large as its GAAP profit it would remove all of the adjusted EPS.)

    Both are series by period end. NaN when either is missing or the sum is not positive; capped at 90, the haircut
    box's limit.
    """

    pay = pd.to_numeric(pd.Series(stock_pay, dtype=float), errors="coerce").dropna().sort_index()
    income = pd.to_numeric(pd.Series(net_income, dtype=float), errors="coerce").dropna().sort_index()
    shared = pay.index.intersection(income.index)
    if len(shared) < periods:
        return np.nan
    shared = shared.sort_values()[-periods:]
    total_pay = abs(pay.loc[shared].sum())
    adjusted = income.loc[shared].sum() + total_pay
    if not np.isfinite(adjusted) or adjusted <= 0:
        return np.nan
    return float(np.clip(total_pay / adjusted * 100, 0, 90))


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
        base, base_margin = get("growth_base"), get("margin_base")
        forward = result.get("forward_from") is None or (on is None or pd.Timestamp(on) >= result["forward_from"])
        if forward and np.isfinite(params.get("forward_growth", np.nan)):
            lines = [rf"g_{{\text{{base}}}} = {_n(base, 1)}\% \ \text{{(yours)}}"]
        else:
            lines = [rf"\begin{{aligned}} g_{{\text{{base}}}} = {{}} & \min\!\big({_n(params['growth_cap'], 0)},\ \max({_n(params['growth_floor'], 0)},\ "
                     rf"{_n(params['growth_follow'])} \times {_n(used_growth, 1)})\big) \\ = {{}} & {_n(base, 1)}\% \end{{aligned}}"]
        source = r"\ \text{(yours)}" if forward and np.isfinite(params.get("forward_margin", np.nan)) else (
            rf" = {_n(used_margin, 1)} + {_n(params['base_margin_pts'], 1)}")
        lines.append(rf"m_{{\text{{base}}}}{source} = {_n(base_margin, 1)}\%")
        for name in SCENARIOS:
            key = name.lower()
            case_margin = base_margin + (params[f"{key}_margin_pts"] if key != "base" else 0.0)
            case_event = get(f"event_{key}")
            case_event = case_event if np.isfinite(case_event) else 0.0
            lines.append(
                rf"\begin{{aligned}} \text{{EPS}}_{{\text{{{key}}}}} = {{}} & {_n(get('clean_eps'))} \times {_n(1 + get(f'growth_{key}') / 100, 3)} \\ "
                rf"& \times \frac{{{_n(case_margin, 1)}}}{{{_n(used_margin, 1)}}} "
                rf"\times {_n(1 + case_event / 100, 3)} \\ = {{}} & {_n(get(f'eps_{key}'))} \end{{aligned}}")
        return lines
    if block == "risk_weighting":
        weights = scenario_weights(params)
        terms = r" \\ & + ".join(rf"{_n(weight, 2)} \times {_n(get(f'eps_{name.lower()}'))}" for weight, name in zip(weights, SCENARIOS))
        return [rf"\begin{{aligned}} \text{{EPS}}_{{\text{{risk}}}} = {{}} & {terms} \\ = {{}} & {_n(get('risk_adjusted_eps'))} \end{{aligned}}"]
    if block in ("item_router", "triage_gate", "item_tracker"):
        _, points = case_effects(result["items"], result["daily"].index[-1] if on is None else on, params)
        return [rf"e = {_n(events, 1)}\% \qquad a = {_n(points, 1)} \text{{ pts}}",
                rf"\text{{{int(get('items_modelled') or 0)} modelled, {int(get('items_parked') or 0)} parked}}"]
    if block == "multiple_builder":
        share = float(np.clip(params["peer_weight"], 0, 1))
        anchor = get("rate_anchor")
        sign = '+' if not get('held_pe_adjust') < 0 else '-'
        passed = params["rate_pass_through"]
        fundamentals = params["anchor_source"] == "Fundamentals"
        raw_peer, raw_own = get('outside_anchor'), get('own_anchor')
        # The anchors after the rate model (the same as before it when no policy rate is loaded).
        peer = (get("fundamental_pe") if fundamentals else
                rate_adjusted_pe(raw_peer, get("forward_gap") if np.isfinite(get("forward_gap")) else 0.0, passed))
        if fundamentals and np.isfinite(get("forward_gap")) and get("forward_gap") != 0:
            peer = np.nan  # shown below: the fundamental P/E at the forward yield
        own = rate_adjusted_pe(raw_own, get("yield_gap") if np.isfinite(get("yield_gap")) else 0.0, passed)
        if np.isfinite(peer) and np.isfinite(own):
            mix = rf"{_n(share)} \times {_n(peer, 1)} + {_n(1 - share)} \times {_n(own, 1)}"
        else:  # only one of them exists: the anchor is that one
            mix = rf"\text{{{'outside' if np.isfinite(peer) else 'own'} only: }} {_n(peer if np.isfinite(peer) else own, 1)}"
        lines = [rf"\begin{{aligned}} \text{{anchor}} = {{}} & {mix} \\ = {{}} & {_n(anchor, 1)} \end{{aligned}}",
                rf"\begin{{aligned}} \text{{P/E}}_{{\text{{acc}}}} = {{}} & {_n(anchor, 1)} {sign} {_n(abs(get('held_pe_adjust')), 1)} - {_n(abs(get('held_quality_pe')), 1)} \\ "
                rf"& \text{{kept in }} {_n(params['pe_floor'], 0)} \text{{ to }} {_n(params['pe_cap'], 0)} \\ = {{}} & {_n(get('acceptable_pe'), 1)} \end{{aligned}}"]
        if fundamentals and np.isfinite(get("fundamental_pe")):
            margin = get("held_gross_margin")
            margin_text = _n(margin, 1) + r"\%" if np.isfinite(margin) else r"\text{n/a}"
            growth = get("held_cagr") if np.isfinite(get("held_cagr")) else params["fallback_yoy"]
            required = (get("now_yield") if np.isfinite(get("now_yield")) else params["base_yield"]) + params["equity_premium"]
            years_text, pe_text = _n(get("growth_years"), 1), _n(get("fundamental_pe"), 1)
            lines.insert(0, rf"n = {years_text} \text{{ years}} \quad (GM = {margin_text}), \qquad g = {_n(growth, 1)}\%")
            lines.insert(1, rf"\text{{P/E}}_{{\text{{fund}}}} = {pe_text} \quad (r = {_n(required, 2)}\%)")
        pairs = [("own", raw_own, get("yield_gap"))] + ([] if fundamentals else [("peers", raw_peer, get("forward_gap"))])
        for label, raw, gap in pairs:
            if np.isfinite(raw) and np.isfinite(get("policy_rate")):
                lines.append(rf"\text{{P/E}}^{{r}}_{{\text{{{label}}}}} = \frac{{1}}{{1 / {_n(raw, 1)} + {_n(passed)} \times {_n(gap, 2)} / 100}}"
                             rf" = {_n(rate_adjusted_pe(raw, gap, passed), 1)}")
        return lines
    if block == "rate_model":
        if not np.isfinite(get("policy_rate")):
            return [r"\text{No policy rate loaded: the rate model is off } (\Delta y = 0)"]
        weight = rate_weight(params["yield_years"], params["rate_half_life"])
        lines = [rf"w_{{{int(params['yield_years'])}}} = {_n(weight, 3)} \qquad r_f = {_n(get('forward_rate'))}\%",
                 rf"\begin{{aligned}} y_N = {{}} & {_n(params['neutral_rate'])} + ({_n(get('forward_rate'))} - {_n(params['neutral_rate'])}) \times {_n(weight, 3)} + {_n(params['term_premium'])} \\ = {{}} & {_n(get('model_yield'))}\% \end{{aligned}}",
                 rf"\Delta y_{{\text{{own}}}} = {_n(get('model_yield'))} - {_n(get('yield_ref'))} = {_n(get('yield_gap'))}",
                 rf"\Delta y_{{\text{{peers}}}} = {_n(get('forward_gap'))}"]
        if np.isfinite(get("market_yield")):
            lines.append(rf"\text{{Market }} {int(params['yield_years'])}\text{{-year yield: }} {_n(get('market_yield'))}\%")
        return lines
    if block == "product":
        return [rf"{_n(get('held_eps'))} \times {_n(get('acceptable_pe'), 1)} = {_n(get('model_price'))}"]
    if block == "market_check":
        return [rf"\left(\frac{{{_n(get('model_price'))}}}{{{_n(get('close'))}}} - 1\right) \times 100 = {_n(get('residual_pct'), 1)}\%"]
    return []
