"""Model check: where the valuation model is likely to go wrong for this stock, and suggested forward values.

Every stock needs a few judgement calls (how long growth lasts, how risky the stock is, how hard the bear case
bites). This module reads the stock's own data and the model run and

- ``model_checks`` lists what looks off, block by block, with what to change;
- ``suggestions`` proposes forward values worked out from the data, each with the reason, ready to apply.

Suggestions are starting points, not answers: they keep the market price out on purpose (fitting the model to the
price would only ever say the stock is fairly priced).
"""

import numpy as np
import pandas as pd

from analysis.valuation_model import (
    fundamental_pe,
    growth_half_life,
    implied_dials,
    latest_values,
    model_params,
)

CHECK_COLUMNS = ["Level", "Where", "What", "Suggestion"]
SUGGESTION_COLUMNS = ["Parameter", "Label", "Now", "Suggested", "Why"]

# How steady the growth must be to last: the spread (standard deviation) of the yearly revenue growth across the
# last releases, in points. At STEADY or less growth halves only every LONG_HALF_LIFE years; at CHOPPY or more the
# gross-margin rule stands alone.
STEADY, CHOPPY = 2.0, 15.0
LONG_HALF_LIFE = 6.0
MARKET_SYMBOL = "SPY"


def growth_steadiness(quarterly, releases=12):
    """Spread in points of the yearly revenue growth over the last ``releases`` releases (NaN with fewer than 8)."""

    if quarterly is None or quarterly.empty or "revenue_yoy" not in quarterly:
        return np.nan
    growth = quarterly["revenue_yoy"].dropna()
    growth = growth[~growth.index.duplicated(keep="last")].tail(releases)
    return float(growth.std()) if len(growth) >= 8 else np.nan


def steady_half_life(spread, growth, params):
    """Growth half-life that steady growth earns: LONG_HALF_LIFE at a spread of STEADY points or less, falling
    to the low gross-margin half-life at CHOPPY or more. NaN without a spread or without positive growth."""

    params = model_params(params)
    if not np.isfinite(spread) or not np.isfinite(growth) or growth <= 0:
        return np.nan
    share = float(np.clip((CHOPPY - spread) / (CHOPPY - STEADY), 0, 1))
    return params["growth_half_life_min"] + (LONG_HALF_LIFE - params["growth_half_life_min"]) * share


def market_beta(close, market_close, years=3):
    """The stock's beta to the market from weekly returns over the last ``years`` years (NaN with too little)."""

    if close is None or market_close is None or not len(close) or not len(market_close):
        return np.nan
    both = pd.concat([pd.Series(close, dtype=float), pd.Series(market_close, dtype=float)], axis=1, join="inner").dropna()
    both = both[both.index >= both.index.max() - pd.DateOffset(years=years)]
    weekly = np.log(both.resample("W-FRI").last()).diff().dropna()
    if len(weekly) < 52:
        return np.nan
    variance = weekly.iloc[:, 1].var()
    return float(weekly.iloc[:, 0].cov(weekly.iloc[:, 1]) / variance) if variance > 0 else np.nan


def suggested_risk_premium(beta, params):
    """Equity risk premium scaled by beta (a calmer stock needs a lower return), kept within 2.5% and 8%."""

    params = model_params(params)
    if not np.isfinite(beta):
        return np.nan
    return float(np.clip(params["equity_premium"] * np.clip(beta, 0.5, 1.8), 2.5, 8.0))


def _value(values, name):
    value = values.get(name, np.nan)
    return float(value) if value is not None and np.isfinite(value) else np.nan


def suggestions(result, beta=np.nan, analyst_growth=np.nan):
    """Forward values worked out from the stock's data, each with the reason. Only rows that differ from today's.

    - Growth for the P/E: the middle of the history's growth a year, the latest year's growth and (when known) the
      analysts' expected EPS growth.
    - Growth half-life: the longer of the gross-margin rule and what steady growth earns.
    - Equity risk premium: the default scaled by the stock's beta to the market.
    - Bear net margin change: no more than a third of the current margin, so the bear case cannot wipe out earnings.
    """

    params, values = result["params"], latest_values(result)
    rows = []
    if not values:
        return pd.DataFrame(columns=SUGGESTION_COLUMNS)

    def add(key, label, now, suggested, why, digits=1):
        if np.isfinite(suggested) and (not np.isfinite(now) or abs(round(suggested, digits) - round(now, digits)) > 0):
            rows.append({"Parameter": key, "Label": label, "Now": now, "Suggested": round(suggested, digits), "Why": why})

    fundamentals = params["anchor_source"] == "Fundamentals"
    cagr, yoy = _value(values, "held_cagr"), _value(values, "revenue_yoy")
    if fundamentals:
        sources = [value for value in (cagr, yoy, analyst_growth) if np.isfinite(value)]
        if len(sources) >= 2:
            middle = float(np.median(sources))
            names = ["history " + f"{cagr:.1f}%" if np.isfinite(cagr) else "", "latest year " + f"{yoy:.1f}%" if np.isfinite(yoy) else "",
                     "analysts " + f"{analyst_growth:.1f}%" if np.isfinite(analyst_growth) else ""]
            add("forward_pe_growth", "Growth for the P/E %", _value(values, "pe_growth"), middle,
                "Middle of " + ", ".join(name for name in names if name) + ": less swayed by one odd year.")
        spread = growth_steadiness(result["quarterly"])
        margin_half = _value(values, "growth_half_life")
        steady = steady_half_life(spread, cagr if np.isfinite(cagr) else yoy, params)
        if np.isfinite(steady) and steady > margin_half + 0.25:
            add("forward_half_life", "Growth half-life, years", _value(values, "pe_half_life"), steady,
                f"Growth has been steady (yearly growth varies by only {spread:.1f} points), which the gross-margin "
                f"rule ({margin_half:.1f} years at a {_value(values, 'held_gross_margin'):.0f}% gross margin) misses: "
                "a low-margin business can still have a strong moat.")
        erp = suggested_risk_premium(beta, params)
        now_erp = params["forward_equity_premium"] if np.isfinite(params["forward_equity_premium"]) else params["equity_premium"]
        if np.isfinite(erp) and abs(erp - now_erp) >= 0.5:
            add("forward_equity_premium", "Equity risk premium %", now_erp, erp,
                f"The stock's beta to the market is {beta:.2f} over 3 years: "
                + ("it moves less than the market, so investors accept a lower return." if beta < 1 else
                   "it moves more than the market, so investors want a higher return."))
    margin = _value(values, "net_margin") if np.isfinite(_value(values, "net_margin")) else params["fallback_margin"]
    if margin > 0 and params["bear_margin_pts"] < -margin / 3:
        add("bear_margin_pts", "Bear: margin points vs forward", params["bear_margin_pts"], -margin / 3,
            f"A bear case of {params['bear_margin_pts']:+.1f} points on a {margin:.1f}% net margin takes away "
            f"{-params['bear_margin_pts'] / margin:.0%} of earnings; a third of the margin is already a hard downturn.")
    return pd.DataFrame(rows, columns=SUGGESTION_COLUMNS)


def model_checks(result, beta=np.nan, analyst_growth=np.nan):
    """What looks off in this run, most important first: Level (Fix / Check / Note), Where (block), What, Suggestion."""

    params, values = result["params"], latest_values(result)
    rows = []
    if not values:
        return pd.DataFrame(columns=CHECK_COLUMNS)

    def add(level, where, what, suggestion):
        rows.append({"Level": level, "Where": where, "What": what, "Suggestion": suggestion})

    # Data the model runs without.
    if not np.isfinite(_value(values, "revenue_yoy")) or not np.isfinite(_value(values, "net_margin")):
        add("Check", "Income and cost map", "No reported revenue or net margin: the model runs on the fallback "
            f"growth ({params['fallback_yoy']:g}%) and margin ({params['fallback_margin']:g}%).",
            "Check the fallbacks in the Scenario engine settings against the company's latest report.")
    if params["anchor_source"] == "Fundamentals" and not np.isfinite(_value(values, "held_gross_margin")):
        add("Note", "Multiple builder", "No gross margin: the growth half-life sits in the middle of its range.",
            "Set a forward half-life in the P/E table if you know the business.")
    # Earnings quality.
    if params["eps_haircut_pct"] >= 40:
        add("Check", "Earnings quality", f"The EPS haircut is {params['eps_haircut_pct']:.0f}%: most of the reported "
            "EPS is set aside.", "Make sure the reported EPS is the adjusted one; if it is GAAP, the stock pay is "
            "already deducted and the haircut should be near 0.")
    # Scenario engine.
    margin = _value(values, "net_margin") if np.isfinite(_value(values, "net_margin")) else params["fallback_margin"]
    if margin > 0 and params["bear_margin_pts"] <= -margin:
        add("Fix", "Scenario engine", f"The bear case takes {abs(params['bear_margin_pts']):g} margin points off a "
            f"{margin:.1f}% margin: no earnings at all in the bear case.", "Use the suggested bear adjustment.")
    elif margin > 0 and params["bear_margin_pts"] < -margin / 3:
        add("Check", "Scenario engine", f"The bear case takes {-params['bear_margin_pts'] / margin:.0%} of the net margin.",
            "A third of the margin is already a hard downturn; see the suggestion.")
    # Growth.
    cagr, yoy = _value(values, "held_cagr"), _value(values, "revenue_yoy")
    if np.isfinite(cagr) and np.isfinite(yoy) and abs(cagr - yoy) >= 10:
        add("Check", "Multiple builder", f"Growth has changed pace: {yoy:.1f}% in the latest year against {cagr:.1f}% "
            "a year over the history.", "Decide which one lasts and put it in Growth for the P/E (P/E table).")
    if np.isfinite(analyst_growth) and np.isfinite(cagr) and abs(analyst_growth - cagr) >= 15:
        add("Note", "Multiple builder", f"Analysts expect {analyst_growth:+.0f}% EPS growth, far from the history's "
            f"{cagr:.1f}% revenue growth.", "A turnaround or a one-off year? The suggestion takes the middle.")
    spread = growth_steadiness(result["quarterly"])
    if params["anchor_source"] == "Fundamentals":
        steady = steady_half_life(spread, cagr if np.isfinite(cagr) else yoy, params)
        margin_half = _value(values, "growth_half_life")
        if np.isfinite(steady) and steady > margin_half + 0.25 and not np.isfinite(params["forward_half_life"]):
            add("Fix", "Multiple builder", f"Growth is steady (varies by {spread:.1f} points) but the low gross margin "
                f"gives it only a {margin_half:.1f}-year half-life.", "Use the suggested half-life.")
    if np.isfinite(beta) and abs(suggested_risk_premium(beta, params) - params["equity_premium"]) >= 0.5 \
            and not np.isfinite(params["forward_equity_premium"]):
        add("Check", "Multiple builder", f"Beta to the market is {beta:.2f}; every stock gets the same "
            f"{params['equity_premium']:g}% risk premium by default.", "Use the suggested risk premium.")
    # Peers.
    if params["anchor_source"] == "Peers":
        own, peers = _value(values, "own_anchor"), _value(values, "peer_anchor")
        if np.isfinite(own) and np.isfinite(peers) and peers > 0 and not 1 / 3 <= own / peers <= 3:
            add("Fix", "Multiple builder", f"The peers' P/E ({peers:.1f}x) is far from the stock's own ({own:.1f}x): "
                "they are priced as a different business.", "Switch the outside view to Fundamentals.")
    # Limits.
    before = _value(values, "pe_before_premium")
    if np.isfinite(before) and before >= params["pe_cap"] - 1e-9:
        add("Check", "Multiple builder", f"The P/E is held at the cap ({params['pe_cap']:g}x).",
            "Raise the cap only if the growth really supports more.")
    elif np.isfinite(before) and before <= params["pe_floor"] + 1e-9:
        add("Check", "Multiple builder", f"The P/E is held at the floor ({params['pe_floor']:g}x).",
            "Look at the growth and the margin the P/E rests on.")
    # The premium.
    premium = _value(values, "implied_premium_pct")
    if np.isfinite(premium) and abs(premium) >= 50:
        dials = implied_dials(result)
        plausible = dials[dials["Read"] == "Plausible"]["Dial"].head(3).tolist() if not dials.empty else []
        add("Note", "Market check", f"The market pays {premium:+.0f}% over the model before any premium of yours.",
            "Work through the suggestions first; what is left is a premium the market pays for things outside the "
            "model" + (f". Dials that could explain it: {', '.join(plausible)}." if plausible else "."))
    order = {"Fix": 0, "Check": 1, "Note": 2}
    frame = pd.DataFrame(rows, columns=CHECK_COLUMNS)
    return frame.sort_values("Level", key=lambda column: column.map(order), kind="stable").reset_index(drop=True)


def fundamental_pe_with(result, overrides):
    """The fundamental P/E today with some forward values changed (to show what the suggestions do)."""

    params = model_params({**result["params"], **overrides})
    values = latest_values(result)
    growth = overrides.get("forward_pe_growth", _value(values, "pe_growth"))
    half_life = overrides.get("forward_half_life", _value(values, "pe_half_life"))
    erp = overrides.get("forward_equity_premium", params["forward_equity_premium"])
    if np.isfinite(erp):
        params = {**params, "equity_premium": erp}
    if not np.isfinite(half_life):
        half_life = growth_half_life(_value(values, "held_gross_margin"), params)
    return fundamental_pe(growth, None, _value(values, "model_yield"), params, half_life=half_life)
