"""Model check: where the valuation model is likely to go wrong for this stock, and suggested settings.

Every stock needs a few judgement calls (how long growth lasts, how risky the stock is, how hard the bear case
bites). This module reads the stock's own data and the model run and

- ``model_checks`` lists what looks off, block by block, with what to change;
- ``suggestions`` proposes settings worked out from the data, each with the reason and whether it applies to every
  quarter (a rule from data known at the time) or from the latest release (a forward guess), ready to apply.

Suggestions are starting points, not answers: they keep the market price out on purpose (fitting the model to the
price would only ever say the stock is fairly priced).
"""

import numpy as np
import pandas as pd

from analysis.valuation_model import (
    implied_dials,
    latest_values,
    model_params,
    risk_premium_from_beta,
    rolling_beta,
    steady_half_life,
)

CHECK_COLUMNS = ["Level", "Where", "What", "Suggestion"]
SUGGESTION_COLUMNS = ["Parameter", "Label", "Current", "Suggested", "Applies to", "Why"]
MARKET_SYMBOL = "SPY"
ALL_QUARTERS, LATEST = "every quarter", "from the latest release"


def growth_steadiness(quarterly, releases=12):
    """Spread in points of the yearly revenue growth over the last ``releases`` releases (NaN with fewer than 8)."""

    if quarterly is None or quarterly.empty or "revenue_yoy" not in quarterly:
        return np.nan
    growth = quarterly["revenue_yoy"].dropna()
    growth = growth[~growth.index.duplicated(keep="last")].tail(releases)
    return float(growth.std()) if len(growth) >= 8 else np.nan


def market_beta(close, market_close, years=3):
    """The stock's beta to the market today, from weekly returns over the last ``years`` years (NaN with too little)."""

    if close is None or not len(close):
        return np.nan
    beta = rolling_beta(close, market_close, pd.DatetimeIndex([pd.Series(close).index.max()]), years)
    return float(beta.iloc[-1]) if len(beta) else np.nan


def _value(values, name):
    value = values.get(name, np.nan)
    return float(value) if value is not None and np.isfinite(value) else np.nan


def suggestions(result, beta=np.nan, analyst_growth=np.nan):
    """Settings worked out from the stock's data, each with the reason. Only rows that change something.

    Rules for every quarter (they use only data known at the time): half-life from growth steadiness, risk premium
    from beta. Forward guesses from the latest release: growth for the P/E halfway to the analysts' expected growth.
    The bear case's margin cut (every quarter): at most a third of the current margin.
    """

    params, values = result["params"], latest_values(result)
    rows = []
    if not values:
        return pd.DataFrame(columns=SUGGESTION_COLUMNS)

    def add(key, label, now, suggested, applies, why):
        rows.append({"Parameter": key, "Label": label, "Current": now, "Suggested": suggested, "Applies to": applies,
                     "Why": why})

    if params["anchor_source"] == "Fundamentals":
        spread = growth_steadiness(result["quarterly"])
        growth = _value(values, "pe_growth")
        steady = steady_half_life(spread, growth, params)
        if not params["half_life_from_steadiness"] and np.isfinite(steady) and steady > _value(values, "growth_half_life") + 0.25:
            add("half_life_from_steadiness", "Half-life from growth steadiness (rule)", 0.0, 1.0, ALL_QUARTERS,
                f"Growth has been steady (yearly growth varies by {spread:.1f} points), which the gross-margin rule "
                "misses: a low-margin business can still have a strong moat.")
        if not params["erp_from_beta"] and np.isfinite(beta):
            erp = float(risk_premium_from_beta([beta], params).iloc[0])
            if abs(erp - params["equity_premium"]) >= 0.5:
                add("erp_from_beta", "Risk premium from beta (rule)", 0.0, 1.0, ALL_QUARTERS,
                    f"Beta to the market is {beta:.2f}: the risk premium would be {erp:.1f}% instead of "
                    f"{params['equity_premium']:g}%.")
        if np.isfinite(analyst_growth) and np.isfinite(growth) and abs(analyst_growth - growth) >= 3 \
                and not np.isfinite(params["forward_pe_growth"]):
            middle = round((analyst_growth + growth) / 2, 1)
            add("forward_pe_growth", "Growth for the P/E %", growth, middle, LATEST,
                f"Halfway between the history's {growth:.1f}% and the analysts' expected {analyst_growth:.1f}% "
                "for the next four quarters.")
    margin = _value(values, "net_margin") if np.isfinite(_value(values, "net_margin")) else params["fallback_margin"]
    if margin > 0 and params["bear_margin_pts"] < -margin / 3:
        add("bear_margin_pts", "Bear: margin points vs forward", params["bear_margin_pts"], round(-margin / 3, 1),
            ALL_QUARTERS, f"A bear case of {params['bear_margin_pts']:+.1f} points on a {margin:.1f}% net margin "
            f"takes away {-params['bear_margin_pts'] / margin:.0%} of earnings; a third of the margin is already a "
            "hard downturn.")
    return pd.DataFrame(rows, columns=SUGGESTION_COLUMNS)


def model_checks(result, beta=np.nan, analyst_growth=np.nan):
    """What looks off in this run, most important first: Level (Fix / Check / Note), Where (block), What, Suggestion."""

    params, values = result["params"], latest_values(result)
    rows = []
    if not values:
        return pd.DataFrame(columns=CHECK_COLUMNS)

    def add(level, where, what, suggestion):
        rows.append({"Level": level, "Where": where, "What": what, "Suggestion": suggestion})

    proposed = suggestions(result, beta, analyst_growth).set_index("Parameter")
    # Data the model runs without.
    if not np.isfinite(_value(values, "revenue_yoy")) or not np.isfinite(_value(values, "net_margin")):
        add("Check", "Income and cost map", "No reported revenue or net margin: the model runs on the fallback "
            f"growth ({params['fallback_yoy']:g}%) and margin ({params['fallback_margin']:g}%).",
            "Check the fallbacks in the Scenario engine settings against the company's latest report.")
    if params["anchor_source"] == "Fundamentals" and not np.isfinite(_value(values, "held_gross_margin")):
        add("Note", "Multiple builder", "No gross margin: the growth half-life sits in the middle of its range.",
            "Set a forward half-life in the P/E table if you know the business.")
    if params["eps_haircut_pct"] >= 40:
        add("Check", "Earnings quality", f"The EPS haircut is {params['eps_haircut_pct']:.0f}%: most of the reported "
            "EPS is set aside.", "Make sure the reported EPS is the adjusted one; if it is GAAP, the stock pay is "
            "already deducted and the haircut should be near 0.")
    # Scenario engine.
    margin = _value(values, "net_margin") if np.isfinite(_value(values, "net_margin")) else params["fallback_margin"]
    if margin > 0 and params["bear_margin_pts"] <= -margin:
        add("Fix", "Scenario engine", f"The bear case takes {abs(params['bear_margin_pts']):g} margin points off a "
            f"{margin:.1f}% margin: no earnings at all in the bear case.", "Apply the suggested bear adjustment.")
    elif margin > 0 and params["bear_margin_pts"] < -margin / 3:
        add("Check", "Scenario engine", f"The bear case takes {-params['bear_margin_pts'] / margin:.0%} of the net margin.",
            "A third of the margin is already a hard downturn; see the suggestion.")
    # Growth.
    cagr, yoy = _value(values, "held_cagr"), _value(values, "revenue_yoy")
    if np.isfinite(cagr) and np.isfinite(yoy) and abs(cagr - yoy) >= 10:
        add("Check", "Multiple builder", f"Growth has changed pace: {yoy:.1f}% in the latest year against {cagr:.1f}% "
            "a year over the history.", "Decide which one lasts and put it in Growth for the P/E (P/E table).")
    if "half_life_from_steadiness" in proposed.index:
        add("Fix", "Multiple builder", "Growth is steady but the low gross margin gives it a short half-life.",
            "Turn on the half-life from growth steadiness (suggested).")
    if "erp_from_beta" in proposed.index:
        add("Check", "Multiple builder", f"Beta to the market is {beta:.2f}, yet every stock gets the same "
            f"{params['equity_premium']:g}% risk premium.", "Turn on the risk premium from beta (suggested).")
    if "forward_pe_growth" in proposed.index:
        add("Note", "Multiple builder", f"Analysts expect {analyst_growth:+.0f}% EPS growth, away from the "
            f"{_value(values, 'pe_growth'):.1f}% the P/E rests on.", "A turnaround or a one-off year? See the suggestion.")
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


def suggested_params(params, proposed):
    """``params`` with the suggestions applied (to preview them)."""

    updated = dict(params)
    for row in pd.DataFrame(proposed).to_dict("records"):
        value = row["Suggested"]
        updated[row["Parameter"]] = bool(value) if row["Parameter"] in ("half_life_from_steadiness", "erp_from_beta") else float(value)
    return model_params(updated)
