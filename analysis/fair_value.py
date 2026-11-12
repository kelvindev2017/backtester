"""Fair value of a stock: risk-adjusted earnings times an acceptable multiple.

The method has two speeds.

Quarterly, once per results release:

1. Income map: revenue, growth and how much of it is left as profit.
2. Costs and margins: each cost line as a share of revenue (a common-size statement) and operating leverage.
3. Earnings quality: reported, adjusted ("normalized") and clean profit. Clean profit deducts share-based pay,
   which adjusted figures leave out although it is a real cost. One-offs that recur are flagged.
4. Risk register: each risk goes into the model once. Specific events change earnings (bear or base case);
   general uncertainty lowers the multiple. Risks below the materiality bar are parked with no price effect.
5. Scenarios: bear, base and bull revenue and margins, weighted by probability into one risk-adjusted EPS.

Daily, as the price moves:

6. Acceptable multiple: a starting P/E (peer average) less a deduction for each general risk.
7. Value per share: risk-adjusted EPS times the acceptable multiple, and the bear / base / bull range.
8. Market check: what today's price implies (EPS at the acceptable multiple, multiple on the risk-adjusted
   EPS, and how likely the bear case would have to be).

Every function here is pure; the app fetches the statements and draws the tab.
"""

import numpy as np
import pandas as pd

SCENARIOS = ["Bear", "Base", "Bull"]
SCENARIO_COLUMNS = ["Probability %", "Revenue growth %", "Operating margin %", "Multiple +/-"]
DEFAULT_PROBABILITIES = {"Bear": 25.0, "Base": 50.0, "Bull": 25.0}
# The bull case grows about 1.7 times as fast as the base case in the worked example (44% against 26%).
BULL_GROWTH_FACTOR = 1.7
BULL_MARGIN_UPLIFT = 3.0
BEAR_MULTIPLE_GAP = 3.0
BULL_MULTIPLE_GAP = 5.0

DESTINATIONS = ["Earnings: bear", "Earnings: base", "Earnings: bear and base", "Multiple"]
IMPACTS = ["Low", "Medium", "High"]
RISK_COLUMNS = ["Risk", "Evidence", "Goes to", "Impact", "P/E deduction"]

# Yahoo's statement rows, first match wins.
INCOME_ROWS = {
    "Revenue": ["Total Revenue", "Operating Revenue"],
    "Cost of sales": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "Gross profit": ["Gross Profit"],
    "R&D": ["Research And Development"],
    "SG&A": ["Selling General And Administration"],
    "Operating income": ["Operating Income", "Total Operating Income As Reported"],
    "Pretax income": ["Pretax Income"],
    "Tax": ["Tax Provision"],
    "Net income": ["Net Income Common Stockholders", "Net Income"],
    "Adjusted net income": ["Normalized Income"],
    "Unusual items": ["Total Unusual Items", "Total Unusual Items Excluding Goodwill"],
    "EBITDA": ["EBITDA"],
    "Adjusted EBITDA": ["Normalized EBITDA"],
    "D&A (income)": ["Reconciled Depreciation"],
    "Diluted EPS": ["Diluted EPS"],
    "Shares": ["Diluted Average Shares", "Basic Average Shares"],
}
CASH_FLOW_ROWS = {
    "Share-based pay": ["Stock Based Compensation"],
    "D&A": ["Depreciation And Amortization", "Depreciation Amortization Depletion"],
    "Operating cash flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
}
COST_LINES = ["Cost of sales", "R&D", "SG&A"]

# US over-the-counter ADRs whose home listing has fuller data in the company's own currency:
# ADR -> (home listing, home shares per ADR).
HOME_LISTINGS = {
    "ASMVY": ("0522.HK", 3.0),
    "BESIY": ("BESI.AS", 1.0),
}


def home_listing(ticker):
    """(home listing, home shares per ADR) for an ADR in HOME_LISTINGS, else None."""

    return HOME_LISTINGS.get((ticker or "").upper())


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def statement_row(statement, names):
    """The first of ``names`` present in a Yahoo statement (rows = items, columns = period ends), oldest first."""

    if statement is None or getattr(statement, "empty", True):
        return pd.Series(dtype=float)
    for name in names:
        if name in statement.index:
            row = pd.to_numeric(statement.loc[name], errors="coerce")
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            row.index = pd.to_datetime(row.index)
            return row.sort_index()
    return pd.Series(dtype=float)


def period_frame(income, cash_flow=None):
    """One row per period (oldest first) with the items the tab uses, from Yahoo's income and cash-flow
    statements. Periods with no revenue and no EPS are dropped (Yahoo pads foreign filers with empty
    quarters)."""

    columns = {name: statement_row(income, rows) for name, rows in INCOME_ROWS.items()}
    columns.update({name: statement_row(cash_flow, rows) for name, rows in CASH_FLOW_ROWS.items()})
    frame = pd.DataFrame(columns).sort_index()
    if frame.empty:
        return frame
    if frame["D&A"].isna().all():
        frame["D&A"] = frame["D&A (income)"]
    frame = frame.drop(columns="D&A (income)")
    if frame["Gross profit"].isna().all() and not frame["Cost of sales"].isna().all():
        frame["Gross profit"] = frame["Revenue"] - frame["Cost of sales"]
    if frame["Adjusted net income"].isna().all():
        frame["Adjusted net income"] = frame["Net income"]
    return frame.loc[frame["Revenue"].notna() | frame["Diluted EPS"].notna()]


def common_size(frame):
    """Cost lines, gross and operating margin and tax rate as % of revenue (tax as % of pre-tax profit)."""

    revenue = frame["Revenue"].where(frame["Revenue"] > 0)
    table = pd.DataFrame(index=frame.index)
    for line in COST_LINES:
        table[line] = frame[line] / revenue * 100
    table["Gross margin"] = frame["Gross profit"] / revenue * 100
    table["Operating margin"] = frame["Operating income"] / revenue * 100
    table["Net margin"] = frame["Net income"] / revenue * 100
    pretax = frame["Pretax income"].where(frame["Pretax income"] > 0)
    table["Tax rate"] = frame["Tax"] / pretax * 100
    return table.dropna(axis=1, how="all")


def yoy_growth(series, periods_per_year=4):
    """% change on the same period a year earlier (four quarters back), matched by date, not position, so a
    missing quarter does not shift the comparison."""

    series = series.dropna()
    out = {}
    for when, value in series.items():
        earlier = series.loc[(series.index >= when - pd.Timedelta(days=380)) &
                             (series.index <= when - pd.Timedelta(days=350))]
        if not earlier.empty and earlier.iloc[-1] > 0:
            out[when] = (value / earlier.iloc[-1] - 1) * 100
    return pd.Series(out, dtype=float)


def operating_leverage(frame):
    """Latest year-on-year % change in operating income divided by that of revenue, or None.

    Above 1 means profit moves more than sales, both up and down."""

    revenue = yoy_growth(frame["Revenue"])
    operating = frame["Operating income"].dropna()
    if revenue.empty or operating.empty:
        return None
    when = revenue.index[-1]
    earlier = operating.loc[(operating.index >= when - pd.Timedelta(days=380)) &
                            (operating.index <= when - pd.Timedelta(days=350))]
    if when not in operating.index or earlier.empty or earlier.iloc[-1] <= 0 or revenue.iloc[-1] == 0:
        return None
    return _finite((operating.loc[when] / earlier.iloc[-1] - 1) * 100 / revenue.iloc[-1])


def trailing(frame, column, periods=4):
    """Sum of the last ``periods`` values of ``column``, or None when fewer are reported."""

    values = frame[column].dropna() if column in frame else pd.Series(dtype=float)
    if len(values) < periods:
        return None
    return _finite(values.iloc[-periods:].sum())


def earnings_quality(frame, periods=4):
    """Reported, adjusted and clean profit over the last ``periods`` periods (a trailing year for quarters),
    with EBITDA, tax and the flags that weaken the adjusted figure.

    Clean profit = adjusted profit less share-based pay.
    """

    result = {name: trailing(frame, column, periods) for name, column in [
        ("Reported net income", "Net income"), ("Adjusted net income", "Adjusted net income"),
        ("Share-based pay", "Share-based pay"), ("EBITDA", "EBITDA"), ("Adjusted EBITDA", "Adjusted EBITDA"),
        ("Revenue", "Revenue"), ("Operating income", "Operating income"), ("Tax", "Tax"),
        ("Pretax income", "Pretax income"), ("Operating cash flow", "Operating cash flow"),
    ]}
    adjusted = result["Adjusted net income"]
    sbc = result["Share-based pay"]
    result["Clean net income"] = adjusted - sbc if adjusted is not None and sbc is not None else adjusted
    pretax, tax = result["Pretax income"], result["Tax"]
    result["Tax rate %"] = tax / pretax * 100 if pretax and tax is not None and pretax > 0 else None
    unusual = frame["Unusual items"].dropna().iloc[-periods:] if "Unusual items" in frame else pd.Series(dtype=float)
    revenue = result["Revenue"]
    # An item counts as a one-off charge when it is at least 0.5% of a quarter's revenue.
    threshold = abs(revenue) / periods * 0.005 if revenue else 0
    result["One-off periods"] = int((unusual.abs() > threshold).sum()) if not unusual.empty else None
    result["Flags"] = quality_flags(result)
    return result


def quality_flags(quality):
    """Plain-language reasons the adjusted profit overstates what reaches shareholders."""

    flags = []
    adjusted, sbc = quality.get("Adjusted net income"), quality.get("Share-based pay")
    if adjusted and sbc and adjusted > 0 and sbc / adjusted >= 0.05:
        flags.append(f"Share-based pay is {sbc / adjusted:.0%} of adjusted profit; it is a real cost, so clean "
                     "EPS deducts it.")
    if (quality.get("One-off periods") or 0) >= 2:
        flags.append(f"\"One-off\" items appeared in {quality['One-off periods']} of the last 4 periods, so they "
                     "recur.")
    tax = quality.get("Tax rate %")
    if tax is not None and tax >= 28:
        flags.append(f"Tax took {tax:.0f}% of pre-tax profit, more than most peers pay.")
    reported = quality.get("Reported net income")
    if adjusted and reported is not None and adjusted > 0 and reported < 0.7 * adjusted:
        flags.append(f"Reported profit is only {reported / adjusted:.0%} of adjusted profit: check what the "
                     "adjustments remove.")
    cash = quality.get("Operating cash flow")
    if cash is not None and reported and reported > 0 and cash < 0.8 * reported:
        flags.append(f"Operating cash flow is {cash / reported:.0%} of reported profit.")
    return flags


def run_rate_eps(frame, column="Adjusted net income"):
    """Four times the latest quarter's per-share profit (``column`` / diluted shares), or None.

    When the latest quarter has only its reported EPS (Yahoo's data for Hong Kong listings), that EPS is used.
    """

    rows = frame[[column, "Shares"]].dropna()
    rows = rows.loc[rows["Shares"] > 0]
    eps = frame["Diluted EPS"].dropna() if "Diluted EPS" in frame else pd.Series(dtype=float)
    if not eps.empty and (rows.empty or eps.index[-1] > rows.index[-1]):
        return _finite(eps.iloc[-1] * 4)
    if rows.empty:
        return None
    return _finite(rows[column].iloc[-1] / rows["Shares"].iloc[-1] * 4)


def analyst_operating_margin(revenue_estimate, eps_estimate, shares, net_share_pct):
    """Operating margin implied by analysts' next-year EPS and revenue: EPS x shares / revenue, grossed up
    from net to operating profit with ``net_share_pct``. None when an input is missing."""

    revenue_estimate, eps_estimate, shares = _finite(revenue_estimate), _finite(eps_estimate), _finite(shares)
    if not revenue_estimate or eps_estimate is None or not shares or not net_share_pct:
        return None
    return eps_estimate * shares / revenue_estimate * 100 / (net_share_pct / 100)


def run_rate_pe(price, eps):
    """Price over a run-rate EPS, or None for a loss."""

    price, eps = _finite(price), _finite(eps)
    if price is None or not eps or eps <= 0:
        return None
    return price / eps


def default_scenarios(frame, analyst_growth=None, base_margin=None):
    """Bear / base / bull starting assumptions for next year from the reported periods.

    Base: analysts' next-year revenue growth when known (else half the latest year-on-year growth), at the
    latest quarter's operating margin. Bear: flat revenue at the lowest operating margin of the last two
    years. Bull: growth 1.7 times the base at three points more margin. "Multiple +/-" is each case's P/E
    against the acceptable multiple: 3 lower for the bear case and 5 higher for the bull case (14x / 17x / 22x
    in the worked example).

    ``base_margin`` (analysts' implied margin, used when the reported figures are stale) replaces the latest
    margin; the bear case is then 5 points lower.
    """

    margins = common_size(frame).get("Operating margin", pd.Series(dtype=float)).dropna()
    latest_margin = float(margins.iloc[-1]) if not margins.empty else 10.0
    low_margin = float(margins.iloc[-8:].min()) if not margins.empty else latest_margin - 5
    if _finite(base_margin) is not None:
        latest_margin, low_margin = float(base_margin), float(base_margin) - 5
    growth = _finite(analyst_growth)
    if growth is None:
        recent = yoy_growth(frame["Revenue"])
        growth = float(recent.iloc[-1]) / 2 if not recent.empty else 5.0
    growth = float(np.clip(growth, -50, 150))
    rows = {
        "Bear": [DEFAULT_PROBABILITIES["Bear"], min(0.0, growth), min(low_margin, latest_margin), -BEAR_MULTIPLE_GAP],
        "Base": [DEFAULT_PROBABILITIES["Base"], growth, latest_margin, 0.0],
        "Bull": [DEFAULT_PROBABILITIES["Bull"], max(growth * BULL_GROWTH_FACTOR, growth + 5),
                 latest_margin + BULL_MARGIN_UPLIFT, BULL_MULTIPLE_GAP],
    }
    table = pd.DataFrame.from_dict(rows, orient="index", columns=SCENARIO_COLUMNS)
    table.index.name = "Scenario"
    return table.round(1)


def net_from_operating(quality):
    """Share of operating profit left as adjusted net profit (after interest, tax and minorities), as a %
    (72-75% in the worked example). Falls back to 75%."""

    operating, adjusted = quality.get("Operating income"), quality.get("Adjusted net income")
    if operating and adjusted is not None and operating > 0:
        return float(np.clip(adjusted / operating * 100, 30, 120))
    return 75.0


def scenario_earnings(scenarios, base_revenue, shares, net_share_pct, share_based_pay=0.0):
    """Next year's revenue, operating profit, adjusted EPS and EPS after share-based pay per scenario.

    revenue = base revenue x (1 + growth); operating profit = revenue x margin; adjusted net profit =
    operating profit x net share; EPS after share-based pay deducts the pay before dividing by the shares.
    """

    table = scenarios.copy()
    shares = _finite(shares)
    if not shares or shares <= 0 or _finite(base_revenue) is None:
        for column in ["Revenue", "Operating profit", "Adjusted EPS", "EPS after share-based pay"]:
            table[column] = np.nan
        return table
    table["Revenue"] = base_revenue * (1 + table["Revenue growth %"] / 100)
    table["Operating profit"] = table["Revenue"] * table["Operating margin %"] / 100
    net = table["Operating profit"] * net_share_pct / 100
    table["Adjusted EPS"] = net / shares
    table["EPS after share-based pay"] = (net - (share_based_pay or 0.0)) / shares
    return table


def normalized_probabilities(scenarios):
    """Scenario probabilities as fractions that add up to 1 (equal weights when they are all zero)."""

    probabilities = scenarios["Probability %"].clip(lower=0).fillna(0)
    total = probabilities.sum()
    if total <= 0:
        return pd.Series(1 / len(probabilities), index=probabilities.index)
    return probabilities / total


def weighted(scenarios, column):
    """Probability-weighted value of ``column`` over the scenarios."""

    values = scenarios[column]
    if values.isna().any():
        return None
    return _finite((values * normalized_probabilities(scenarios)).sum())


def default_risks(suggestions):
    """The register's starting rows: one per suggested multiple deduction."""

    rows = [{"Risk": item["Risk"], "Evidence": item["Evidence"], "Goes to": "Multiple",
             "Impact": item.get("Impact", "Medium"), "P/E deduction": float(item["Deduction"])}
            for item in suggestions]
    return pd.DataFrame(rows, columns=RISK_COLUMNS)


def suggested_deductions(frame, quality, annual=None, foreign_listing=False):
    """Starting multiple deductions for general risks, each with its evidence.

    Cyclicality: last year's adjusted profit against the average of the earlier full years (near the top of
    the cycle earns a lower multiple). Earnings quality: one point per quality flag, at most 2. Listing and
    liquidity: one point for a foreign company traded through an ADR or another listing.
    """

    suggestions = []
    ttm = quality.get("Adjusted net income")
    years = statement_row(annual, INCOME_ROWS["Adjusted net income"]).dropna() if annual is not None else None
    if years is None or years.empty:
        years = statement_row(annual, INCOME_ROWS["Net income"]).dropna() if annual is not None else None
    if ttm and years is not None and len(years) >= 2:
        history = years.iloc[:-1] if len(years) > 2 else years
        average = float(history.mean())
        if average > 0:
            ratio = ttm / average
            deduction = 3.0 if ratio >= 1.5 else 1.0 if ratio >= 1.2 else 0.0
            if deduction:
                suggestions.append({
                    "Risk": "Earnings may be near the top of the cycle",
                    "Evidence": f"Last year's adjusted profit is {ratio:.1f}x the average of the earlier full "
                                "years",
                    "Impact": "High" if deduction >= 3 else "Medium", "Deduction": deduction})
    flags = quality.get("Flags") or []
    if flags:
        suggestions.append({"Risk": "Earnings quality", "Evidence": " ".join(flags[:2]),
                            "Impact": "Medium", "Deduction": float(min(len(flags), 2))})
    if foreign_listing:
        suggestions.append({"Risk": "Listing and liquidity",
                            "Evidence": "Foreign company traded through an ADR or second listing",
                            "Impact": "Medium", "Deduction": 1.0})
    return suggestions


def triage(risks, bar="Medium"):
    """Split the register at the materiality bar: rows rated below it are parked (no price effect).

    Returns (active rows, parked rows)."""

    if risks is None or risks.empty:
        empty = pd.DataFrame(columns=RISK_COLUMNS)
        return empty, empty
    level = {name: rank for rank, name in enumerate(IMPACTS)}
    ranks = risks["Impact"].map(level).fillna(level["Medium"])
    material = ranks >= level.get(bar, 1)
    named = risks["Risk"].fillna("").astype(str).str.strip() != ""
    return risks.loc[material & named], risks.loc[~material & named]


def multiple_deductions(active_risks):
    """Total P/E points taken off by the active risks routed to the multiple (positive number)."""

    if active_risks is None or active_risks.empty:
        return 0.0
    rows = active_risks.loc[active_risks["Goes to"] == "Multiple", "P/E deduction"]
    return float(pd.to_numeric(rows, errors="coerce").fillna(0).abs().sum())


def earnings_risks(active_risks, scenario):
    """Names of the active risks the register routes into ``scenario``'s earnings ("Bear" or "Base")."""

    if active_risks is None or active_risks.empty:
        return []
    goes = active_risks["Goes to"].fillna("")
    mask = goes.str.startswith("Earnings") & goes.str.contains(scenario.lower())
    return active_risks.loc[mask, "Risk"].tolist()


def acceptable_multiple(start, deductions):
    """Starting P/E less the deductions, never below 1."""

    return max(float(start) - float(deductions), 1.0)


def valuation(scenarios, acceptable, eps_column="EPS after share-based pay", per_share=1.0):
    """Value per share by scenario (EPS x the scenario's multiple, the acceptable multiple plus its
    "Multiple +/-") and the risk-adjusted value (weighted EPS x the acceptable multiple), in the price's
    currency per listed share.

    ``per_share`` converts a statement-currency value per ordinary share to the listing (exchange rate times
    ordinary shares per listed share)."""

    table = scenarios[["Probability %", eps_column]].copy()
    table["Multiple"] = (float(acceptable) + scenarios["Multiple +/-"].fillna(0)).clip(lower=1.0)
    table["Value per share"] = table[eps_column] * table["Multiple"] * per_share
    eps = weighted(scenarios, eps_column)
    risk_adjusted = {
        "EPS": eps, "Multiple": float(acceptable),
        "Value per share": eps * acceptable * per_share if eps is not None else None,
    }
    return table, risk_adjusted


def implied_bear_probability(scenarios, price, multiple, eps_column="EPS after share-based pay", per_share=1.0):
    """Bear-case probability at which risk-adjusted value equals the price, moving probability from the base
    and bull cases in proportion. None when the price lies outside what the scenarios can reach."""

    price, multiple = _finite(price), _finite(multiple)
    if price is None or not multiple:
        return None
    probabilities = normalized_probabilities(scenarios)
    eps = scenarios[eps_column]
    rest = probabilities.drop("Bear")
    if rest.sum() <= 0 or eps.isna().any():
        return None
    other = float((eps.drop("Bear") * rest).sum() / rest.sum())
    bear = float(eps.loc["Bear"])
    target = price / (multiple * per_share)
    if other == bear:
        return None
    share = (other - target) / (other - bear)
    return share if 0 <= share <= 1 else None


def market_check(scenarios, price, acceptable, eps_column="EPS after share-based pay", per_share=1.0):
    """What today's price implies: EPS at the acceptable multiple, multiple on the risk-adjusted EPS, the
    bear probability that fits the price, and the gap between value and price."""

    price = _finite(price)
    eps = weighted(scenarios, eps_column)
    result = {"Price": price, "Implied EPS": None, "Implied multiple": None, "Implied bear probability": None,
              "Value vs price %": None}
    if price is None:
        return result
    result["Implied EPS"] = price / (acceptable * per_share) if acceptable else None
    result["Implied multiple"] = price / (eps * per_share) if eps and eps > 0 else None
    result["Implied bear probability"] = implied_bear_probability(scenarios, price, acceptable, eps_column,
                                                                  per_share)
    if eps is not None and price > 0:
        result["Value vs price %"] = (eps * acceptable * per_share / price - 1) * 100
    return result


def sensitivity(scenarios, acceptable, eps_column="EPS after share-based pay", per_share=1.0, shift=10.0):
    """How much the risk-adjusted value moves for one point of P/E, and for ``shift`` points of probability
    moved into the bear case (from base and bull in proportion)."""

    eps = weighted(scenarios, eps_column)
    if eps is None:
        return {"Per 1x P/E": None, f"Per {shift:g} points to bear": None}
    probabilities = normalized_probabilities(scenarios)
    rest = probabilities.drop("Bear")
    values = scenarios[eps_column]
    other = float((values.drop("Bear") * rest).sum() / rest.sum()) if rest.sum() > 0 else float(values.loc["Bear"])
    per_bear = -(other - float(values.loc["Bear"])) * shift / 100 * acceptable * per_share
    return {"Per 1x P/E": eps * per_share, f"Per {shift:g} points to bear": per_bear}
