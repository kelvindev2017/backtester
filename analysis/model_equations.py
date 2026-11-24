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
        r"h_{\text{default}} = \frac{\min\!\big(\sum \frac{\text{SBC}_q}{\text{shares}_q}(1 - \tau),\ \text{EPS}_{\text{rep}} - \text{EPS}_{\text{GAAP}}\big)}{\text{EPS}_{\text{rep}}} \times 100",
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
        r"\begin{aligned} \text{EPS}_{\text{base}} = {} & \text{EPS}_{\text{clean}} \times \left(1 + \frac{g_{\text{base}}}{100}\right) \\ "
        r"& \times \frac{m_{\text{base}}}{m} \times \left(1 + \frac{e_{\text{base}}}{100}\right) \end{aligned}",
        r"\text{EPS volatility (default): } s = w\,\sigma_{\text{own}} + (1 - w)\,\sigma_{\text{typical}}",
        r"\text{EPS}_{\text{bear, bull}} = \text{EPS}_{\text{base, before items}} \times e^{\mp k s} \times \left(1 + \frac{e_c}{100}\right)",
        r"\text{Fixed gaps: } g_c = g_{\text{base}} + \text{adj}_c, \quad m_c = m_{\text{base}} + \Delta m_c",
        r"\begin{aligned} \text{EPS}_{c} = {} & \text{EPS}_{\text{clean}} \times \left(1 + \frac{g_c}{100}\right) \\ "
        r"& \times \frac{m_c}{m} \times \left(1 + \frac{e_c}{100}\right) \end{aligned}",
    ],
    "risk_weighting": [
        r"\begin{aligned} \text{EPS}_{\text{risk}} = {} & w_{\text{bear}} \times \text{EPS}_{\text{bear}} \\ "
        r"& + w_{\text{base}} \times \text{EPS}_{\text{base}} \\ & + w_{\text{bull}} \times \text{EPS}_{\text{bull}} \end{aligned}",
        r"\text{Through-cycle (deep cycles): } \text{EPS}_{\text{cycle}} = \frac{1}{Y}\Big(\sum_{j=1}^{Y-1} \text{EPS}_{\text{4q}, t-4j+4} "
        r"\times (1 - h) + \text{EPS}_{\text{risk}}\Big)",
        r"\text{Auto: on when } \min_t \frac{\text{EPS}_t}{\max_{s \le t} \text{EPS}_s} \le 1 - \frac{F}{100} "
        r"\text{ and trend } g_{\text{cycle}} < G",
    ],
    "rate_model": [
        r"w_N = \frac{1 - e^{-\lambda N}}{\lambda N}, \qquad \lambda = \frac{\ln 2}{h}",
        r"\text{Market (default): } y_N = y_{\text{market}} + (r_f - r_{\text{today}}) \times w_N",
        r"\text{Policy model: } y_N = r_n + (r_f - r_n) \times w_N + tp",
        r"\Delta y_{\text{own}} = y_N - \operatorname{median}_{\text{look-back}}(y_N)",
        r"\Delta y_{\text{peers}} = y_N(r_f) - y_N(r_{\text{today}})",
    ],
    "multiple_builder": [
        r"g = s_g \times g_{\text{latest}} + (1 - s_g) \times g_{\text{history}}",
        r"h_{GM} = h_{\min} + (h_{\max} - h_{\min}) \times \operatorname{clip}\!\left(\frac{GM - GM_{\text{low}}}{GM_{\text{high}} - GM_{\text{low}}}, 0, 1\right)",
        r"h = \max(h_{GM},\ h_{\text{steady}})",
        r"\text{Beta rule: ERP} = \operatorname{clip}\big(\text{ERP}_0 \times \operatorname{clip}(0.67 \beta + 0.33,\ 0.5,\ 1.8),\ 2.5,\ 8\big)",
        r"\text{Market-implied (default): } \text{P/E}_{\text{fund}}(\text{ERP}^{*}_m) = \frac{\text{Close}_m}{\text{EPS}_{\text{risk}, m}} \text{ at each month-end } m",
        r"\text{ERP} = \operatorname{median}_{\text{look-back}}\big(\text{ERP}^{*}_m\big)",
        r"\ln(1 + g_t) = \ln(1 + g_T) + \big(\ln(1 + g) - \ln(1 + g_T)\big)\, 0.5^{t/h}",
        r"E_1 = 1, \quad E_{t+1} = E_t (1 + g_t)",
        r"\text{P/E}_{\text{fund}} = \sum_{t=1}^{40} \frac{E_t}{(1+r)^t} + \frac{E_{41}}{(r - g_T)(1+r)^{40}}, \qquad r = y_N + \text{ERP}",
        r"\text{P/E}^{r}_{\text{own}} = \frac{1}{1 / \text{P/E}_{\text{own}} + t \times \Delta y_{\text{own}} / 100}",
        r"\begin{aligned} \text{anchor} = {} & k \times \text{P/E}_{\text{outside}} \\ & + (1 - k) \times \text{P/E}^{r}_{\text{own}} \end{aligned}",
        r"\begin{aligned} \text{P/E}_{\text{acc}} = {} & \operatorname{clip}(\text{anchor} + a - d_q) \\ & \times (1 + p / 100) \end{aligned}",
    ],
    "product": [r"\text{Modeled fair value} = \text{EPS}_{\text{risk}} \times \text{P/E}_{\text{acc}}"],
    "market_check": [r"\text{Gap} = \left(\frac{\text{Modeled fair value}}{\text{Close}} - 1\right) \times 100"],
}

# What each symbol stands for, shown under the equations.
SYMBOLS = {
    "income_map": "g: revenue growth %, m: net margin %.",
    "earnings_quality": "h: EPS haircut %, SBC_q: a quarter's stock-based pay, shares_q: its diluted shares, τ: tax rate, EPS_rep and EPS_GAAP: reported (street) and GAAP EPS over the last 4 quarters. h_default = 0 when EPS_rep is not above EPS_GAAP: the company's figure already counts the pay. The P/E deduction d_q goes to the Multiple builder.",
    "triage_gate": "p: probability, impact: EPS impact %, points: P/E points.",
    "item_router": "e: expected EPS change % from event items (per case: the forward probability plus the item's bull "
                   "or bear adjustment; shown weighted), a: P/E points (+ product, - risk), weighted over the cases.",
    "scenario_engine": "g, m: current (trailing) revenue growth and net margin %, g_fwd, m_fwd: your forward values "
                       "(latest quarter only; blank = the auto rule with s: share of trailing growth), c: bear, base or "
                       "bull, adj_c and Δm_c: the case's growth and margin adjustment in points (Fixed gaps only), e_c: event "
                       "items' EPS change % at that case's probability. σ_own: the company's own EPS volatility (standard "
                       "deviation of the yearly log change in trailing EPS over the last 12 releases, at least 8), "
                       "σ_typical: the typical one, w: share of the own one, k: how many s the bear and bull cases sit "
                       "from base. Defaults (k 2.4, w 0.25, σ_typical 0.073) come from a forward test on 60 stocks: "
                       "trained on 2010-2018, they beat the fixed gaps in each year 2019-2025.",
    "risk_weighting": "w: scenario weights, rescaled to add up to 1. Through-cycle EPS: for a deep-cycle company "
                      "the model values the mean of Y yearly EPS figures (the past Y - 1 years' trailing EPS after the "
                      "haircut h, and next year's risk-adjusted EPS), and its fundamental P/E rests on their trend "
                      "growth g_cycle (0 to G). F: the fall from an earlier high, G: the trend cap; Auto turns it on "
                      "when the EPS fell at least F% and the trend is below G% a year.",
    "rate_model": "y_market: the market's N-year Treasury yield (FRED), r_today: today's policy rate. r_f: policy rate used (your forward rate on the last day, the actual rate before), r_n: neutral "
                  "rate, h: half-life of a rate's gap to neutral in years, N: yield maturity in years, w_N: share of "
                  "the policy rate that reaches the N-year yield, tp: term premium (it cancels out of both Δy). "
                  "Δy_own compares with the years the own P/E median covers; Δy_peers is only the forward rate's "
                  "surprise, because the peers' P/E is today's and already prices today's rate.",
    "multiple_builder": "Outside view: P/E_fund (default) or the peers' median P/E (moved by the forward rate's yield change "
                        "like the own one). g: growth for the P/E %, s_g: share from the latest revenue growth (latest quarter vs a year ago, or the last 4 quarters vs the 4 before, per the Latest growth setting), g_history: revenue growth a year over the "
                        "growth history, (TTM revenue per share now / TTM N quarters earlier)^(4/N) - 1 with N the Growth history "
                        "(quarters) setting (within floor and cap), GM: gross margin %, h: half-life in years of the growth above g_T (longer with "
                        "a higher gross margin, or with steady growth: h_steady runs from the low half-life for choppy growth to the steady "
                        "half-life), g_T: terminal growth, y_N: the Rate model's yield (or the base yield), ERP: equity risk premium, ERP_0 the "
                        "plain one, β: the stock's 3-year weekly beta to the S&P 500 (when beta scaling is on); ERP*_m: the premium at which the fundamental P/E equals the market's P/E on that month-end (solved within the floor and cap); with the market-implied premium on, the ERP is its median over the look-back months before the day, even when negative (the tab warns when it is below 2.5%), so a change in the yield moves the P/E as the market usually prices it, and the gap to the close says how far today's premium strays from the usual one; from the latest release g, h and "
                        "ERP are your forward values where set. "
                        "p: your market premium % (kept apart from the fundamentals). x: own or peers, Δy_x: yield change from the Rate model, t: share of it passed to the P/E "
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


def stock_pay_haircut_per_share(stock_pay, shares, reported_eps, gaap_eps=None, tax_rate=0.21, periods=4):
    """(haircut %, reason): the after-tax stock-based pay per share that reported EPS leaves out, as a % of it.

    ``stock_pay`` and ``shares`` (diluted average shares) are by quarter end, ``reported_eps`` (the street figure
    Yahoo reports, usually adjusted) by announcement date and ``gaap_eps`` (diluted GAAP EPS) by quarter end; the
    last ``periods`` of each are summed. When reported EPS is not above GAAP EPS (within 2%), the company's figure
    already counts stock pay as a cost (Apple, Microsoft, Coca-Cola), so nothing is taken off. Otherwise the
    haircut is the stock pay per share after tax, no more than the gap between reported and GAAP EPS. A GAAP loss
    from write-offs therefore cannot inflate it the way a share of net income does. NaN when the data are missing
    or reported EPS is not positive; capped at 90, the haircut box's limit.
    """

    pay = pd.to_numeric(pd.Series(stock_pay, dtype=float), errors="coerce").dropna().sort_index()
    count = pd.to_numeric(pd.Series(shares, dtype=float), errors="coerce").dropna().sort_index()
    reported = pd.to_numeric(pd.Series(reported_eps, dtype=float), errors="coerce").dropna().sort_index()
    gaap = pd.to_numeric(pd.Series(gaap_eps, dtype=float), errors="coerce").dropna().sort_index() if gaap_eps is not None else pd.Series(dtype=float)
    shared = pay.index.intersection(count.index[count > 0])
    if len(shared) < periods or len(reported) < periods:
        return np.nan, "not enough quarters of stock-based pay, shares or reported EPS"
    reported_sum = float(reported.iloc[-periods:].sum())
    if not reported_sum > 0:
        return np.nan, "reported EPS over the last 4 quarters is not positive"
    gaap_sum = float(gaap.iloc[-periods:].sum()) if len(gaap) >= periods else np.nan
    if np.isfinite(gaap_sum) and reported_sum <= gaap_sum + 0.02 * abs(gaap_sum):
        return 0.0, (f"reported EPS ({reported_sum:.2f}) is not above GAAP EPS ({gaap_sum:.2f}), so it already counts "
                     "stock-based pay")
    shared = shared.sort_values()[-periods:]
    tax = float(np.clip(tax_rate, 0, 0.5)) if tax_rate is not None and np.isfinite(tax_rate) else 0.21
    per_share = float((pay.loc[shared].abs() / count.loc[shared]).sum()) * (1 - tax)
    reason = f"stock-based pay of {per_share:.2f} a share after {tax * 100:.0f}% tax over reported EPS of {reported_sum:.2f}"
    if np.isfinite(gaap_sum) and reported_sum - gaap_sum < per_share:
        per_share = reported_sum - gaap_sum
        reason += f", limited to the gap to GAAP EPS ({gaap_sum:.2f})"
    return float(np.clip(per_share / reported_sum * 100, 0, 90)), reason + " (last 4 quarters)"


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
        if forward and params.get("analyst_base"):
            lines = [rf"g_{{\text{{base}}}} = {_n(base, 1)}\% \ \text{{(analysts' EPS, next 4q vs last 4q)}}"]
        elif forward and np.isfinite(params.get("forward_growth", np.nan)):
            lines = [rf"g_{{\text{{base}}}} = {_n(base, 1)}\% \ \text{{(yours)}}"]
        else:
            lines = [rf"\begin{{aligned}} g_{{\text{{base}}}} = {{}} & \min\!\big({_n(params['growth_cap'], 0)},\ \max({_n(params['growth_floor'], 0)},\ "
                     rf"{_n(params['growth_follow'])} \times {_n(used_growth, 1)})\big) \\ = {{}} & {_n(base, 1)}\% \end{{aligned}}"]
        source = (r"\ \text{(held)}" if forward and params.get("analyst_base") else
                  r"\ \text{(yours)}" if forward and np.isfinite(params.get("forward_margin", np.nan)) else None)
        source = source or (
            rf" = {_n(used_margin, 1)} + {_n(params['base_margin_pts'], 1)}")
        lines.append(rf"m_{{\text{{base}}}}{source} = {_n(base_margin, 1)}\%")
        if params.get("case_spread") == "EPS volatility" and np.isfinite(get("case_spread")):
            spread, k = get("case_spread"), params["spread_k"]
            own = get("eps_spread")
            w = float(np.clip(params["spread_own_share"], 0, 1))
            own_text = _n(own, 3) if np.isfinite(own) and params["manual_eps"] <= 0 else rf"{_n(params['spread_typical'], 3)}\ \text{{(no history)}}"
            lines.append(rf"s = {_n(w)} \times {own_text} + {_n(1 - w)} \times {_n(params['spread_typical'], 3)} = {_n(spread, 3)}")
            for name in SCENARIOS:
                key = name.lower()
                case_event = get(f"event_{key}")
                case_event = case_event if np.isfinite(case_event) else 0.0
                factor = {"bear": np.exp(-k * spread), "base": 1.0, "bull": np.exp(k * spread)}[key]
                lines.append(
                    rf"\begin{{aligned}} \text{{EPS}}_{{\text{{{key}}}}} = {{}} & {_n(get('clean_eps'))} \times {_n(1 + get('growth_base') / 100, 3)} "
                    rf"\times \frac{{{_n(base_margin, 1)}}}{{{_n(used_margin, 1)}}} \\ & \times {_n(factor, 3)} "
                    rf"\times {_n(1 + case_event / 100, 3)} \\ = {{}} & {_n(get(f'eps_{key}'))} \end{{aligned}}")
            return lines
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
        if get("cycle_on") == 1:
            years = int(get("cycle_years_used")) if np.isfinite(get("cycle_years_used")) else int(params["cycle_years"])
            return [rf"\begin{{aligned}} \text{{EPS}}_{{\text{{risk}}}} = {{}} & {terms} \\ = {{}} & {_n(get('next_year_eps'))} \end{{aligned}}",
                    rf"\begin{{aligned}} \text{{EPS}}_{{\text{{cycle}}}} = {{}} & ({_n(get('cycle_past_sum'))} + {_n(get('next_year_eps'))}) / {years} "
                    rf"\\ = {{}} & {_n(get('cycle_eps'))} \quad (\text{{trend }} {_n(get('cycle_trend'), 1)}\%,\ "
                    rf"g_{{\text{{cycle}}}} = {_n(float(np.clip(get('cycle_trend'), 0, params['cycle_trend_cap'])), 1)}\%) \end{{aligned}}"]
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
        peer = (get("outside_forward") if fundamentals else
                rate_adjusted_pe(raw_peer, get("forward_gap") if np.isfinite(get("forward_gap")) else 0.0, passed))
        own = rate_adjusted_pe(raw_own, get("yield_gap") if np.isfinite(get("yield_gap")) else 0.0, passed)
        if np.isfinite(peer) and np.isfinite(own):
            mix = rf"{_n(share)} \times {_n(peer, 1)} + {_n(1 - share)} \times {_n(own, 1)}"
        else:  # only one of them exists: the anchor is that one
            mix = rf"\text{{{'outside' if np.isfinite(peer) else 'own'} only: }} {_n(peer if np.isfinite(peer) else own, 1)}"
        lines = [rf"\begin{{aligned}} \text{{anchor}} = {{}} & {mix} \\ = {{}} & {_n(anchor, 1)} \end{{aligned}}",
                rf"\begin{{aligned}} \text{{P/E}}_{{\text{{acc}}}} = {{}} & {_n(anchor, 1)} {sign} {_n(abs(get('held_pe_adjust')), 1)} - {_n(abs(get('held_quality_pe')), 1)} \\ "
                rf"& \text{{kept in }} {_n(params['pe_floor'], 0)} \text{{ to }} {_n(params['pe_cap'], 0)} \\ "
                rf"& \times (1 + {_n(get('premium_pct') if np.isfinite(get('premium_pct')) else 0.0, 1)} / 100) \\ = {{}} & {_n(get('acceptable_pe'), 1)} \end{{aligned}}"]
        if fundamentals and np.isfinite(get("fundamental_pe")):
            margin = get("held_gross_margin")
            margin_text = _n(margin, 1) + r"\%" if np.isfinite(margin) else r"\text{n/a}"
            growth = get("pe_growth") if np.isfinite(get("pe_growth")) else params["fallback_yoy"]
            # The premium the run used: yours from the latest release, else the plain one scaled by beta.
            erp = get("pe_erp")
            if not np.isfinite(erp):
                erp = (params["forward_equity_premium"] if np.isfinite(params.get("forward_equity_premium", np.nan))
                       else params["equity_premium"])
            required = (get("model_yield") if np.isfinite(get("model_yield")) else params["base_yield"]) + erp
            years_text = _n(get("pe_half_life") if np.isfinite(get("pe_half_life")) else get("growth_half_life"), 2)
            pe_text = _n(get("outside_forward"), 1)
            lines.insert(0, rf"h = {years_text} \text{{ years}} \quad (GM = {margin_text}), \qquad g = {_n(growth, 1)}\%")
            lines.insert(1, rf"\text{{P/E}}_{{\text{{fund}}}} = {pe_text} \quad (r = {_n(required - erp, 2)} + {_n(erp, 2)} = {_n(required, 2)}\%)")
            if np.isfinite(get("usual_erp")):
                lines.insert(2, rf"\text{{ERP}} = \text{{usual implied }} {_n(get('usual_erp'), 2)}\% \quad (\text{{today's close implies }} {_n(get('implied_erp'), 2)}\%)")
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
        lines = [rf"w_{{{int(params['yield_years'])}}} = {_n(weight, 3)} \qquad r_f = {_n(get('forward_rate'))}\%"]
        if params.get("yield_source") == "Market" and np.isfinite(get("market_yield")):
            lines.append(rf"\begin{{aligned}} y_N = {{}} & y_{{\text{{market}}}} + (r_f - r_{{\text{{today}}}}) \times w_N \\ = {{}} & {_n(get('market_yield'))} + ({_n(get('forward_rate'))} - {_n(get('policy_rate'))}) \times {_n(weight, 3)} \\ = {{}} & {_n(get('model_yield'))}\% \end{{aligned}}")
        else:
            lines.append(rf"\begin{{aligned}} y_N = {{}} & {_n(params['neutral_rate'])} + ({_n(get('forward_rate'))} - {_n(params['neutral_rate'])}) \times {_n(weight, 3)} + {_n(params['term_premium'])} \\ = {{}} & {_n(get('model_yield'))}\% \end{{aligned}}")
        lines += [
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
