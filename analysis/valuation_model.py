"""Block model of a share's fair price: inputs and assumptions in, an estimated price per share out.

The model has two sample rates. The quarterly part runs on every earnings release (and on the day a risk or product
item starts to apply): income and cost map -> earnings quality -> scenario engine -> risk weighting. Its outputs are
held between releases. The daily part runs on every close: multiple builder -> product -> market check.

Signals are what the world supplies (quarterly results, prices, the list of risk and product items). Parameters are
what the analyst chooses (adjustment rules, growth and margin cases, scenario weights, P/E deductions). Every item
enters once: an event inside the forecast year changes earnings, anything general or further out changes the P/E.

Pure pandas / numpy, no network and no Streamlit, so it runs on past dates with only the data known then.
"""

import numpy as np
import pandas as pd

SCENARIOS = ["Bear", "Base", "Bull"]
RISK, PRODUCT = "Risk", "Product"
EVENT, GENERAL = "Event", "General"
IN_HORIZON, BEYOND = "In horizon", "Beyond horizon"
MODELLED, PARKED = "Modelled", "Parked"
TO_SCENARIOS, TO_MULTIPLE = "Scenario engine", "Multiple builder"

REGISTER_COLUMNS = ["Item", "Type", "Kind", "Timing", "Probability %", "EPS impact %", "P/E points", "Status",
                    "Since", "Keywords", "Source"]
REGISTER_DEFAULTS = {"Item": "", "Type": RISK, "Kind": EVENT, "Timing": IN_HORIZON, "Probability %": np.nan,
                     "EPS impact %": np.nan, "P/E points": np.nan, "Status": PARKED, "Since": "", "Keywords": "",
                     "Source": ""}

DEFAULT_PARAMS = {
    # Earnings quality
    "eps_haircut_pct": 0.0,      # share of reported EPS removed (share-based pay, one-offs that keep coming back)
    "manual_eps": 0.0,           # trailing 12-month EPS per share typed in; 0 = use the reported quarters
    # Scenario engine
    "growth_follow": 0.5,        # base growth = this share of the trailing revenue growth
    "growth_floor": -20.0,
    "growth_cap": 40.0,
    "fallback_yoy": 0.0,         # trailing revenue growth % assumed when there is no revenue history
    "fallback_margin": 10.0,     # net margin % assumed when there is no revenue or net income
    "bear_growth_gap": 25.0,     # bear growth = base growth minus this many points
    "bull_growth_gap": 15.0,     # bull growth = base growth plus this many points
    "bear_margin_pts": -3.0,     # net margin change in each case, in points
    "base_margin_pts": 0.0,
    "bull_margin_pts": 2.0,
    # Risk weighting
    "weight_bear": 25.0,
    "weight_base": 50.0,
    "weight_bull": 25.0,
    # Multiple builder
    "pe_lookback_years": 5,
    "peer_weight": 0.5,          # share of the P/E anchor taken from the peers' median; the rest is the own median
    "pe_floor": 5.0,
    "pe_cap": 40.0,
    # Triage gate
    "materiality_eps_pct": 1.0,  # smallest expected EPS effect (probability x impact) that enters the model
    "materiality_pe": 0.5,       # smallest P/E effect, in points, that enters the model
}

# name -> (frame, label, unit, axis). Signals with the same axis can share a chart panel.
SIGNALS = {
    "revenue": ("quarterly", "Quarterly revenue", "", "amount"),
    "net_income": ("quarterly", "Quarterly net income", "", "amount"),
    "eps": ("quarterly", "Quarterly reported EPS", "", "eps"),
    "ttm_revenue": ("quarterly", "Revenue, last 4 quarters", "", "amount"),
    "revenue_yoy": ("quarterly", "Revenue growth, year on year", "%", "percent"),
    "net_margin": ("quarterly", "Net margin, last 4 quarters", "%", "share"),
    "ttm_eps": ("quarterly", "Reported EPS, last 4 quarters", "", "eps"),
    "clean_eps": ("quarterly", "Clean base EPS", "", "eps"),
    "growth_bear": ("quarterly", "Bear growth", "%", "percent"),
    "growth_base": ("quarterly", "Base growth", "%", "percent"),
    "growth_bull": ("quarterly", "Bull growth", "%", "percent"),
    "event_effect": ("quarterly", "Expected effect of event items", "%", "percent"),
    "eps_bear": ("quarterly", "Bear EPS", "", "eps"),
    "eps_base": ("quarterly", "Base EPS", "", "eps"),
    "eps_bull": ("quarterly", "Bull EPS", "", "eps"),
    "risk_adjusted_eps": ("quarterly", "Risk-adjusted EPS", "", "eps"),
    "pe_adjust": ("quarterly", "P/E points from general items", "pts", "points"),
    "items_modelled": ("quarterly", "Items in the model", "", "count"),
    "items_parked": ("quarterly", "Items parked", "", "count"),
    "close": ("daily", "Close", "", "price"),
    "held_eps": ("daily", "Risk-adjusted EPS (held)", "", "eps"),
    "held_pe_adjust": ("daily", "P/E points (held)", "pts", "points"),
    "own_anchor": ("daily", "Own median P/E", "x", "multiple"),
    "peer_anchor": ("daily", "Peers' median P/E", "x", "multiple"),
    "acceptable_pe": ("daily", "Acceptable P/E", "x", "multiple"),
    "model_price": ("daily", "Model price per share", "", "price"),
    "market_pe": ("daily", "Market price / risk-adjusted EPS", "x", "multiple"),
    "residual_pct": ("daily", "Model price vs close", "%", "percent"),
}

# Each block: its sample rate, what it reads, what it writes and the parameters it owns (key -> label).
BLOCKS = {
    "income_map": dict(
        title="Income and cost map", rate="quarterly", role="Logic: parsing only",
        inputs=["revenue", "net_income"], outputs=["ttm_revenue", "revenue_yoy", "net_margin"],
        params={"fallback_yoy": "Growth % when there is no revenue history",
                "fallback_margin": "Net margin % when there is no net income"}),
    "earnings_quality": dict(
        title="Earnings quality", rate="quarterly", role="Param: adjustment rules",
        inputs=["eps"], outputs=["ttm_eps", "clean_eps"],
        params={"eps_haircut_pct": "EPS haircut %", "manual_eps": "Typed-in trailing EPS (0 = reported)"}),
    "item_tracker": dict(
        title="Item tracker", rate="quarterly", role="Logic: detect and park",
        inputs=[], outputs=["items_modelled", "items_parked"], params={}),
    "triage_gate": dict(
        title="Triage gate", rate="quarterly", role="Param: materiality bar",
        inputs=["items_modelled", "items_parked"], outputs=["items_modelled"],
        params={"materiality_eps_pct": "Smallest expected EPS effect %", "materiality_pe": "Smallest P/E effect, points"}),
    "item_router": dict(
        title="Item router", rate="quarterly", role="Logic: type and timing",
        inputs=["items_modelled"], outputs=["event_effect", "pe_adjust"], params={}),
    "scenario_engine": dict(
        title="Scenario engine", rate="quarterly", role="Param: growth, margins",
        inputs=["clean_eps", "revenue_yoy", "net_margin", "event_effect"],
        outputs=["growth_bear", "growth_base", "growth_bull", "eps_bear", "eps_base", "eps_bull"],
        params={"growth_follow": "Base growth as a share of trailing growth", "growth_floor": "Base growth floor %",
                "growth_cap": "Base growth cap %", "bear_growth_gap": "Bear: growth points below base",
                "bull_growth_gap": "Bull: growth points above base", "bear_margin_pts": "Bear: margin change, points",
                "base_margin_pts": "Base: margin change, points", "bull_margin_pts": "Bull: margin change, points"}),
    "risk_weighting": dict(
        title="Risk weighting", rate="quarterly", role="Param: scenario weights",
        inputs=["eps_bear", "eps_base", "eps_bull"], outputs=["risk_adjusted_eps"],
        params={"weight_bear": "Bear weight", "weight_base": "Base weight", "weight_bull": "Bull weight"}),
    "multiple_builder": dict(
        title="Multiple builder", rate="daily", role="Param: P/E deductions",
        inputs=["own_anchor", "peer_anchor", "held_pe_adjust"], outputs=["acceptable_pe"],
        params={"pe_lookback_years": "Own P/E look-back, years", "peer_weight": "Share of the anchor from peers",
                "pe_floor": "P/E floor", "pe_cap": "P/E cap"}),
    "product": dict(
        title="Product", rate="daily", role="EPS x acceptable P/E",
        inputs=["held_eps", "acceptable_pe"], outputs=["model_price"], params={}),
    "market_check": dict(
        title="Market check", rate="daily", role="Out: implied dials",
        inputs=["model_price", "close"], outputs=["residual_pct", "market_pe"], params={}),
}

# The doc's ASMPT case, per ADR in US dollars (1 ADR = 3 Hong Kong shares). The event items stay parked because the
# doc bundles them in the bear case: modelling them as well would count them twice.
PRESETS = {
    "ASMVY": {
        "peers": ["BESIY", "KLIC", "AMAT", "COHU", "CAMT"],
        "note": ("ASMPT preset from the worked example: trailing adjusted EPS of HK$4.28 a share is typed in as "
                 "US$1.64 per ADR, because Yahoo's ADR financials are in Hong Kong dollars. The event risks are parked: "
                 "the bear case already carries them."),
        "params": {
            "manual_eps": 1.64, "eps_haircut_pct": 11.0, "growth_follow": 0.7, "fallback_yoy": 40.0,
            "fallback_margin": 9.8, "bear_growth_gap": 28.0, "bull_growth_gap": 16.0, "bear_margin_pts": -1.0,
            "base_margin_pts": 4.0, "bull_margin_pts": 7.0, "peer_weight": 1.0,
        },
        "register": [
            ["Cyclicality: 2027 is probably near peak earnings", RISK, GENERAL, BEYOND, np.nan, np.nan, 3.0, MODELLED,
             "", "cycle, downturn, peak", "ASMPT doc"],
            ["Earnings quality and mix", RISK, GENERAL, IN_HORIZON, np.nan, np.nan, 2.0, MODELLED,
             "", "restructuring, write-off, impairment", "ASMPT doc"],
            ["Listing and liquidity (Hong Kong mid-cap, OTC ADR)", RISK, GENERAL, IN_HORIZON, np.nan, np.nan, 1.0,
             MODELLED, "", "", "ASMPT doc"],
            ["AI spending pause, pulled-forward orders reverse", RISK, EVENT, IN_HORIZON, 25.0, 20.0, np.nan, PARKED,
             "", "capex, ai spending, pull-in, order cut, digestion", "ASMPT doc: in the bear case"],
            ["Tighter US chip export rules", RISK, EVENT, IN_HORIZON, 20.0, 15.0, np.nan, PARKED,
             "", "export, china, sanction, restriction, chip security", "ASMPT doc: in the bear case"],
            ["HBM investment delays", RISK, EVENT, IN_HORIZON, 30.0, 8.0, np.nan, PARKED,
             "", "hbm, memory, sk hynix, samsung", "ASMPT doc: in the bear case"],
            ["SMT targets missed", RISK, EVENT, IN_HORIZON, 40.0, 6.0, np.nan, PARKED,
             "", "smt, surface mount", "ASMPT doc: base case haircut"],
            ["Hybrid bonding replaces TCB sooner", RISK, GENERAL, BEYOND, np.nan, np.nan, 1.0, PARKED,
             "", "hybrid bonding, besi", "ASMPT doc"],
            ["Panel-level packaging", PRODUCT, EVENT, BEYOND, 50.0, np.nan, 1.0, PARKED,
             "", "panel-level, panel level", "ASMPT doc: revenue from 2028"],
            ["Co-packaged optics", PRODUCT, EVENT, BEYOND, 50.0, np.nan, 1.0, PARKED,
             "", "optics, cpo, photonics", "ASMPT doc"],
        ],
    },
}


def model_params(overrides=None):
    """The default parameters with ``overrides`` on top; unknown keys are ignored."""

    params = dict(DEFAULT_PARAMS)
    for key, value in (overrides or {}).items():
        if key in params and value is not None:
            params[key] = value
    return params


def preset_params(ticker):
    return model_params(PRESETS.get(str(ticker).upper(), {}).get("params"))


def empty_register():
    return pd.DataFrame(columns=REGISTER_COLUMNS).astype({column: float for column in
                                                          ["Probability %", "EPS impact %", "P/E points"]})


def preset_register(ticker):
    rows = PRESETS.get(str(ticker).upper(), {}).get("register")
    return clean_register(pd.DataFrame(rows, columns=REGISTER_COLUMNS)) if rows else empty_register()


def _choice(value, choices, default):
    text = str(value).strip().lower() if value is not None and not (isinstance(value, float) and np.isnan(value)) else ""
    for choice in choices:
        if text == choice.lower():
            return choice
    return default


def clean_register(register):
    """A register with every column present and typed; rows without a name are dropped."""

    if register is None or len(register) == 0:
        return empty_register()
    frame = pd.DataFrame(register).copy()
    for column, default in REGISTER_DEFAULTS.items():
        if column not in frame:
            frame[column] = default
    frame = frame[REGISTER_COLUMNS]
    for column in ["Item", "Since", "Keywords", "Source"]:
        frame[column] = [("" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)).strip()
                         for value in frame[column]]
    frame["Since"] = [text if text and pd.notna(pd.to_datetime(text, errors="coerce")) else "" for text in frame["Since"]]
    frame["Type"] = [_choice(value, [RISK, PRODUCT], RISK) for value in frame["Type"]]
    frame["Kind"] = [_choice(value, [EVENT, GENERAL], EVENT) for value in frame["Kind"]]
    frame["Timing"] = [_choice(value, [IN_HORIZON, BEYOND], IN_HORIZON) for value in frame["Timing"]]
    frame["Status"] = [_choice(value, [MODELLED, PARKED], PARKED) for value in frame["Status"]]
    for column in ["Probability %", "EPS impact %", "P/E points"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").abs()
    frame["Probability %"] = frame["Probability %"].clip(upper=100)
    return frame[frame["Item"] != ""].reset_index(drop=True)


def route_items(register, params=None):
    """Triage gate and item router for every item.

    Adds ``Route`` (which block the item feeds), ``Effect`` (expected EPS change in % for the scenario engine, P/E
    points for the multiple builder; negative for a risk, positive for a product) and ``Gate`` (``Modelled`` or the
    reason the item is parked). An event inside the forecast year goes to the scenario engine; a general item or
    anything beyond the horizon goes to the multiple builder.
    """

    params = model_params(params)
    frame = clean_register(register)
    routes, effects, gates = [], [], []
    for row in frame.to_dict("records"):
        sign = -1.0 if row["Type"] == RISK else 1.0
        probability = row["Probability %"] / 100 if pd.notna(row["Probability %"]) else np.nan
        if row["Kind"] == EVENT and row["Timing"] == IN_HORIZON:
            route, bar, unit = TO_SCENARIOS, params["materiality_eps_pct"], "% of EPS"
            effect = sign * probability * row["EPS impact %"]
            needs = "a probability and an EPS impact"
        else:
            route, bar, unit = TO_MULTIPLE, params["materiality_pe"], "P/E points"
            weight = probability if row["Kind"] == EVENT and pd.notna(probability) else 1.0
            effect = sign * weight * row["P/E points"]
            needs = "P/E points"
        if not np.isfinite(effect):
            gate, effect = f"Parked: not sized (needs {needs})", np.nan
        elif row["Status"] != MODELLED:
            gate = "Parked by you"
        elif abs(effect) < bar:
            gate = f"Parked: below the materiality bar ({bar:g} {unit})"
        else:
            gate = MODELLED
        routes.append(route)
        effects.append(effect)
        gates.append(gate)
    frame["Route"], frame["Effect"], frame["Gate"] = routes, effects, gates
    return frame


def _since_dates(items):
    """The day each item starts to apply (NaT = always)."""

    text = items["Since"].astype(object)
    return pd.to_datetime(text.where(text != "", None), errors="coerce")


def item_effects(routed, on=None):
    """(expected EPS change as a fraction, P/E points) from the modelled items in force on ``on`` (None = all)."""

    if routed is None or routed.empty:
        return 0.0, 0.0
    active = routed[routed["Gate"] == MODELLED]
    if on is not None and not active.empty:
        since = _since_dates(active)
        active = active[since.isna() | (since <= pd.Timestamp(on))]
    events = active.loc[active["Route"] == TO_SCENARIOS, "Effect"].sum() / 100
    points = active.loc[active["Route"] == TO_MULTIPLE, "Effect"].sum()
    return max(-1.0, float(events)), float(points)


def scenario_weights(params):
    """Bear, base and bull weights as fractions that sum to 1 (equal weights when all are zero)."""

    raw = np.array([max(0.0, float(params[f"weight_{name.lower()}"])) for name in SCENARIOS])
    return raw / raw.sum() if raw.sum() > 0 else np.full(len(SCENARIOS), 1 / len(SCENARIOS))


def scenario_point(ttm_eps, revenue_yoy, net_margin, params, event_effect=0.0):
    """The quarterly chain for one date: earnings quality, scenario engine and risk weighting.

    ``revenue_yoy`` and ``net_margin`` are in %; ``event_effect`` is the fraction of EPS the modelled event items are
    expected to add or remove. Each case grows the clean EPS with revenue and scales it by the change in net margin.
    """

    clean = ttm_eps * (1 - params["eps_haircut_pct"] / 100)
    yoy = revenue_yoy if np.isfinite(revenue_yoy) else params["fallback_yoy"]
    margin = net_margin if np.isfinite(net_margin) and net_margin > 0 else params["fallback_margin"]
    base_growth = float(np.clip(params["growth_follow"] * yoy, params["growth_floor"], params["growth_cap"]))
    growth = {"Bear": base_growth - params["bear_growth_gap"], "Base": base_growth,
              "Bull": base_growth + params["bull_growth_gap"]}
    point = {"clean_eps": clean, "event_effect": event_effect * 100}
    for name in SCENARIOS:
        key = name.lower()
        margin_ratio = max(0.0, margin + params[f"{key}_margin_pts"]) / margin if margin > 0 else 1.0
        point[f"growth_{key}"] = growth[name]
        point[f"eps_{key}"] = clean * (1 + growth[name] / 100) * margin_ratio * (1 + event_effect)
    weights = scenario_weights(params)
    point["risk_adjusted_eps"] = float(sum(weight * point[f"eps_{name.lower()}"]
                                           for weight, name in zip(weights, SCENARIOS)))
    return point


def _trailing(series, periods=4):
    series = pd.to_numeric(series, errors="coerce").dropna()
    return series.rolling(periods, min_periods=periods).sum()


def quarterly_inputs(quarters):
    """Income and cost map: trailing revenue, its growth and the net margin at each release date.

    ``quarters`` is indexed by the date each quarter became public, with any of the columns Revenue, Net_Income and
    EPS. Growth is the last four quarters against the four before; with under eight quarters it is the latest
    quarter against the same quarter a year earlier.
    """

    frame = pd.DataFrame(index=pd.DatetimeIndex(quarters.index).sort_values().unique()) if quarters is not None else pd.DataFrame()
    if quarters is None or quarters.empty:
        return pd.DataFrame(columns=["revenue", "net_income", "eps", "ttm_revenue", "revenue_yoy", "net_margin", "ttm_eps"])
    quarters = quarters.sort_index()
    quarters = quarters.loc[~quarters.index.duplicated(keep="last")]

    def column(name):
        return pd.to_numeric(quarters[name], errors="coerce").dropna() if name in quarters else pd.Series(dtype=float)

    revenue, net_income, eps = column("Revenue"), column("Net_Income"), column("EPS")
    ttm_revenue, ttm_income = _trailing(revenue), _trailing(net_income)
    yoy = (ttm_revenue / ttm_revenue.shift(4).where(ttm_revenue.shift(4) > 0) - 1) * 100
    quarter_yoy = (revenue / revenue.shift(4).where(revenue.shift(4) > 0) - 1) * 100
    yoy = yoy.combine_first(quarter_yoy)
    margin = (ttm_income.reindex(ttm_revenue.index) / ttm_revenue.where(ttm_revenue > 0)) * 100
    frame["revenue"], frame["net_income"], frame["eps"] = revenue, net_income, eps
    for name, series in [("ttm_revenue", ttm_revenue), ("revenue_yoy", yoy), ("net_margin", margin),
                         ("ttm_eps", _trailing(eps))]:
        # A quarter known for one measure but not another keeps the last value of the other.
        frame[name] = series.reindex(frame.index.union(series.index)).ffill().reindex(frame.index)
    return frame


def run_model(quarters, close, register=None, params=None, peer_pe=None, start=None):
    """Run the whole model over history.

    quarters: results by the date they became public (columns Revenue, Net_Income, EPS; any may be missing).
    close: daily close of the share. register: the risk and product items. peer_pe: daily trailing P/E of each peer,
    one column per peer. start: first day to report (earlier data still feeds the trailing sums and medians).

    Returns a dict with ``quarterly`` (one row per update of the quarterly part), ``daily`` (one row per close),
    ``items`` (the register after the triage gate and router) and ``params``. Nothing on a date uses data from later.
    """

    params = model_params(params)
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    close = close.loc[~close.index.duplicated(keep="last")]
    routed = route_items(register, params)
    inputs = quarterly_inputs(quarters)

    # The quarterly part also runs on the day an item starts to apply, with the last results held.
    item_dates = _since_dates(routed[routed["Gate"] == MODELLED]).dropna()
    first_day = close.index[0] if len(close) else None
    updates = pd.DatetimeIndex(inputs.index).union(pd.DatetimeIndex(item_dates))
    if params["manual_eps"] > 0 and first_day is not None:
        updates = updates.union(pd.DatetimeIndex([first_day]))
    updates = updates.sort_values().unique()
    held = inputs.reindex(inputs.index.union(updates)).ffill().reindex(updates) if len(updates) else inputs
    rows = []
    for when, row in held.iterrows():
        ttm_eps = params["manual_eps"] if params["manual_eps"] > 0 else row.get("ttm_eps", np.nan)
        events, points = item_effects(routed, when)
        point = scenario_point(ttm_eps, row.get("revenue_yoy", np.nan), row.get("net_margin", np.nan), params, events)
        since = _since_dates(routed)
        in_force = (routed["Gate"] == MODELLED) & (since.isna() | (since <= when))
        rows.append({**row.to_dict(), "ttm_eps": ttm_eps, **point, "pe_adjust": points,
                     "items_modelled": int(in_force.sum()), "items_parked": int(len(routed) - in_force.sum())})
    quarterly = pd.DataFrame(rows, index=updates) if rows else pd.DataFrame(
        columns=[name for name, (frame, *_) in SIGNALS.items() if frame == "quarterly"])

    daily = pd.DataFrame(index=close.index)
    daily["close"] = close
    if quarterly.empty:
        daily["held_eps"], daily["held_pe_adjust"], held_ttm = np.nan, 0.0, pd.Series(np.nan, index=close.index)
    else:
        # Hold blocks: each quarterly output keeps its last value until the next update.
        both = quarterly.reindex(quarterly.index.union(close.index)).ffill().reindex(close.index)
        daily["held_eps"], daily["held_pe_adjust"], held_ttm = both["risk_adjusted_eps"], both["pe_adjust"].fillna(0.0), both["ttm_eps"]

    # Multiple builder. The own anchor is the median trailing P/E of the look-back, up to the day before, so the
    # day's own price does not set its own fair multiple.
    own_pe = (close / held_ttm.where(held_ttm > 0)).replace([np.inf, -np.inf], np.nan)
    window = max(2, int(round(252 * float(params["pe_lookback_years"]))))
    daily["own_anchor"] = own_pe.shift(1).rolling(window, min_periods=min(window, 60)).median()
    if peer_pe is not None and not pd.DataFrame(peer_pe).empty:
        peers = pd.DataFrame(peer_pe).apply(pd.to_numeric, errors="coerce")
        peers = peers.where(peers > 0).reindex(peers.index.union(close.index)).ffill(limit=5).reindex(close.index)
        daily["peer_anchor"] = peers.median(axis=1, skipna=True)
    else:
        daily["peer_anchor"] = np.nan
    share = float(np.clip(params["peer_weight"], 0, 1))
    blended = share * daily["peer_anchor"] + (1 - share) * daily["own_anchor"]
    anchor = blended.where(daily["peer_anchor"].notna() & daily["own_anchor"].notna(),
                           daily["peer_anchor"].combine_first(daily["own_anchor"]))
    if share >= 1:
        anchor = daily["peer_anchor"].combine_first(daily["own_anchor"])
    elif share <= 0:
        anchor = daily["own_anchor"].combine_first(daily["peer_anchor"])
    daily["anchor"] = anchor
    daily["acceptable_pe"] = (anchor + daily["held_pe_adjust"]).clip(lower=params["pe_floor"], upper=params["pe_cap"])

    # Product and market check.
    positive_eps = daily["held_eps"].where(daily["held_eps"] > 0)
    daily["model_price"] = positive_eps * daily["acceptable_pe"]
    daily["market_pe"] = daily["close"] / positive_eps
    daily["residual_pct"] = (daily["model_price"] / daily["close"] - 1) * 100
    if start is not None:
        daily = daily.loc[pd.Timestamp(start):]
        # Keep the update in force on the first day shown.
        before = quarterly.loc[:pd.Timestamp(start)]
        quarterly = quarterly.loc[before.index[-1]:] if len(before) else quarterly
    return {"quarterly": quarterly, "daily": daily, "items": routed, "params": params}


def latest_values(result, on=None):
    """The value of every signal on ``on`` (default: the last close), quarterly ones held from their last update."""

    daily, quarterly = result["daily"], result["quarterly"]
    values = {}
    if daily.empty:
        return values
    on = pd.Timestamp(on) if on is not None else daily.index[-1]
    day = daily.loc[:on]
    update = quarterly.loc[:on]
    for name, (frame, *_) in SIGNALS.items():
        source = day if frame == "daily" else update
        if name in source and len(source):
            # The latest quarter can lack a value another supplier has (e.g. revenue from SEC, EPS from Yahoo).
            known = source[name].dropna()
            values[name] = float(known.iloc[-1]) if len(known) else np.nan
    return values


def _point_price(state, params, routed):
    """Model price from one day's held inputs: the whole chain as one function of the parameters and items."""

    events, points = item_effects(routed, state["on"])
    point = scenario_point(state["ttm_eps"], state["revenue_yoy"], state["net_margin"], params, events)
    eps = point["risk_adjusted_eps"]
    if not np.isfinite(eps) or not np.isfinite(state["anchor"]):
        return np.nan
    if eps <= 0:
        return 0.0  # no earnings, no price: lets the solver see the sign change at the edge of a dial's range
    return eps * float(np.clip(state["anchor"] + points, params["pe_floor"], params["pe_cap"]))


def _solve(price_at, low, high, target, steps=60):
    """The value in [low, high] where ``price_at`` equals ``target``, or None when it is never reached."""

    f_low, f_high = price_at(low) - target, price_at(high) - target
    if not (np.isfinite(f_low) and np.isfinite(f_high)) or f_low * f_high > 0:
        return None
    for _ in range(steps):
        middle = (low + high) / 2
        f_middle = price_at(middle) - target
        if not np.isfinite(f_middle):
            return None
        if f_low * f_middle <= 0:
            high, f_high = middle, f_middle
        else:
            low, f_low = middle, f_middle
    return (low + high) / 2


DIAL_COLUMNS = ["Dial", "Block", "Now", "Needed", "Change", "Unit", "Read"]


def _read(needed, now, low, high):
    if needed is None:
        return "Cannot match alone"
    return "Plausible" if low <= needed <= high else "A stretch"


def implied_dials(result, on=None):
    """What the market price implies, one dial at a time.

    For each parameter and each risk or product item: holding everything else as it is, the value that makes the
    model price equal the close on ``on`` (default: the last day), and whether that value is believable. A parked
    item is tried too: its row shows what it would have to be to explain the gap on its own.
    """

    values = latest_values(result, on)
    if not values or not np.isfinite(values.get("close", np.nan)) or not np.isfinite(values.get("model_price", np.nan)):
        return pd.DataFrame(columns=DIAL_COLUMNS)
    params, routed = result["params"], result["items"]
    on = pd.Timestamp(on) if on is not None else result["daily"].index[-1]
    anchor = result["daily"].loc[:on, "anchor"].iloc[-1]
    state = {"on": on, "ttm_eps": values.get("ttm_eps", np.nan), "revenue_yoy": values.get("revenue_yoy", np.nan),
             "net_margin": values.get("net_margin", np.nan), "anchor": anchor}
    target = values["close"]
    rows = []

    def add(name, block, now, low, high, plausible, unit, price_at):
        needed = _solve(price_at, low, high, target)
        rows.append({"Dial": name, "Block": block, "Now": now, "Needed": needed if needed is not None else np.nan,
                     "Change": needed - now if needed is not None else np.nan, "Unit": unit,
                     "Read": _read(needed, now, *plausible)})

    weights = scenario_weights(params) * 100

    def with_bear_weight(weight):
        # The base and bull weights keep their ratio.
        rest = weights[1] + weights[2]
        scale = (100 - weight) / rest if rest > 0 else 0
        return _point_price(state, {**params, "weight_bear": weight, "weight_base": weights[1] * scale,
                                    "weight_bull": weights[2] * scale}, routed)

    add("Bear weight", "Risk weighting", weights[0], 0, 100, (5, 60), "%", with_bear_weight)
    add("Growth in every case", "Scenario engine", 0.0, -60, 60, (-15, 15), "pts",
        lambda shift: _shifted_growth_price(state, params, routed, shift))
    add("Net margin in every case", "Scenario engine", 0.0, -30, 30, (-5, 5), "pts",
        lambda shift: _point_price(state, {**params, **{f"{name.lower()}_margin_pts": params[f"{name.lower()}_margin_pts"] + shift
                                                        for name in SCENARIOS}}, routed))
    add("EPS haircut", "Earnings quality", params["eps_haircut_pct"], 0, 90, (0, 30), "%",
        lambda haircut: _point_price(state, {**params, "eps_haircut_pct": haircut}, routed))
    events, points = item_effects(routed, on)
    add("Acceptable P/E", "Multiple builder", float(np.clip(anchor + points, params["pe_floor"], params["pe_cap"])),
        1, 100, (params["pe_floor"], params["pe_cap"]), "x",
        lambda multiple: _point_price(state, {**params, "pe_floor": multiple, "pe_cap": multiple}, routed))

    model_above = values["model_price"] > target
    for position, row in routed.iterrows():
        parked = row["Gate"] != MODELLED
        label = f"{row['Item']} (parked)" if parked else row["Item"]
        # A risk can only pull the model price down, a product can only lift it.
        helps = (row["Type"] == RISK) == model_above if parked else True
        if row["Route"] == TO_SCENARIOS:
            impact = row["EPS impact %"] if pd.notna(row["EPS impact %"]) else 10.0
            now = 0.0 if parked else row["Probability %"]

            def with_probability(probability, position=position, impact=impact):
                trial = routed.copy()
                sign = -1.0 if trial.at[position, "Type"] == RISK else 1.0
                trial.at[position, "Effect"], trial.at[position, "Gate"] = sign * probability / 100 * impact, MODELLED
                trial.at[position, "Since"] = ""
                return _point_price(state, params, trial)

            if helps:
                add(f"{label}: probability", TO_SCENARIOS, now, 0, 100, (max(0, now - 30), min(100, now + 30)), "%",
                    with_probability)
            else:
                rows.append({"Dial": f"{label}: probability", "Block": TO_SCENARIOS, "Now": now, "Needed": np.nan,
                             "Change": np.nan, "Unit": "%", "Read": "Wrong direction"})
        else:
            now = 0.0 if parked else abs(row["Effect"])

            def with_points(size, position=position):
                trial = routed.copy()
                sign = -1.0 if trial.at[position, "Type"] == RISK else 1.0
                trial.at[position, "Effect"], trial.at[position, "Gate"] = sign * size, MODELLED
                trial.at[position, "Since"] = ""
                return _point_price(state, params, trial)

            if helps:
                add(f"{label}: P/E points", TO_MULTIPLE, now, 0, 30, (0, now + 3), "pts", with_points)
            else:
                rows.append({"Dial": f"{label}: P/E points", "Block": TO_MULTIPLE, "Now": now, "Needed": np.nan,
                             "Change": np.nan, "Unit": "pts", "Read": "Wrong direction"})
    return pd.DataFrame(rows, columns=DIAL_COLUMNS)


def _shifted_growth_price(state, params, routed, shift):
    """Model price with ``shift`` points added to the growth of every case."""

    yoy = state["revenue_yoy"] if np.isfinite(state["revenue_yoy"]) else params["fallback_yoy"]
    base = float(np.clip(params["growth_follow"] * yoy, params["growth_floor"], params["growth_cap"])) + shift
    fixed = {**params, "growth_follow": 1.0, "growth_floor": base, "growth_cap": base}
    return _point_price(state, fixed, routed)


def sensitivities(result, on=None):
    """How far the model price moves for one step of each main dial (10 points of bear weight, 1x of P/E ...)."""

    dials = implied_dials(result, on)
    values = latest_values(result, on)
    if dials.empty or not values:
        return {}
    params, routed = result["params"], result["items"]
    on = pd.Timestamp(on) if on is not None else result["daily"].index[-1]
    state = {"on": on, "ttm_eps": values.get("ttm_eps", np.nan), "revenue_yoy": values.get("revenue_yoy", np.nan),
             "net_margin": values.get("net_margin", np.nan), "anchor": result["daily"].loc[:on, "anchor"].iloc[-1]}
    base = _point_price(state, params, routed)
    weights = scenario_weights(params) * 100
    rest = weights[1] + weights[2]
    scale = (100 - weights[0] - 10) / rest if rest > 0 else 0
    more_bear = {**params, "weight_bear": weights[0] + 10, "weight_base": weights[1] * scale, "weight_bull": weights[2] * scale}
    return {
        "10 points more bear weight": _point_price(state, more_bear, routed) - base,
        "1x more P/E": values.get("held_eps", np.nan),
        "1 point more growth": _shifted_growth_price(state, params, routed, 1.0) - base,
        "1 point more net margin": _point_price(state, {**params, **{
            f"{name.lower()}_margin_pts": params[f"{name.lower()}_margin_pts"] + 1 for name in SCENARIOS}}, routed) - base,
    }


def track_items(register, news=None, now=None, stale_days=90):
    """Item tracker: match stories to the register and collect the ones no item explains.

    Returns ``(register, candidates)``. The register gains ``Mentions``, ``Last mention`` and ``Stale`` (a modelled
    item with keywords that no story mentions; left False when there are no stories to judge by). ``candidates``
    are stories with a clear positive or negative tone that match no item's keywords: possible new risks or products.
    Matching is on keywords in the headline and summary, so it finds mentions, not meaning.
    """

    frame = clean_register(register)
    columns = ["published", "tone", "title", "publisher", "url"]
    if news is None or len(news) == 0:
        frame["Mentions"], frame["Last mention"], frame["Stale"] = 0, pd.NaT, False
        return frame, pd.DataFrame(columns=columns)
    stories = news.copy()
    text = (stories["title"].fillna("").astype(str) + " " + stories.get("summary", "").fillna("").astype(str)).str.lower()
    published = pd.to_datetime(stories["published"], utc=True, errors="coerce")
    matched = pd.Series(False, index=stories.index)
    mentions, last, stale = [], [], []
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    for row in frame.to_dict("records"):
        keywords = [word.strip().lower() for word in row["Keywords"].split(",") if word.strip()]
        hit = pd.Series(False, index=stories.index)
        for word in keywords:
            hit |= text.str.contains(word, regex=False)
        matched |= hit
        mentions.append(int(hit.sum()))
        latest = published[hit].max() if hit.any() else pd.NaT
        last.append(latest)
        recent = pd.notna(latest) and (now - latest).days <= stale_days
        stale.append(bool(keywords) and row["Status"] == MODELLED and not recent)
    frame["Mentions"], frame["Last mention"], frame["Stale"] = mentions, last, stale
    tone = stories["tone"] if "tone" in stories else pd.Series("Neutral", index=stories.index)
    candidates = stories[~matched & tone.isin(["Positive", "Negative"])].copy()
    candidates["tone"] = tone[candidates.index]
    for column in columns:
        if column not in candidates:
            candidates[column] = ""
    return frame, candidates[columns].reset_index(drop=True)


def register_changes(before, after):
    """Plain-language list of what changed between two registers (items added, removed or re-sized)."""

    before, after = clean_register(before), clean_register(after)
    old = {row["Item"]: row for row in before.to_dict("records")}
    new = {row["Item"]: row for row in after.to_dict("records")}
    changes = [f"Added: {name}" for name in new if name not in old]
    changes += [f"Removed: {name}" for name in old if name not in new]
    for name in new:
        if name not in old:
            continue
        for column in ["Type", "Kind", "Timing", "Probability %", "EPS impact %", "P/E points", "Status", "Since"]:
            was, now = old[name][column], new[name][column]
            both_empty = _is_empty(was) and _is_empty(now)
            if not both_empty and was != now:
                changes.append(f"{name}: {column} {_show(was)} -> {_show(now)}")
    return changes


def _is_empty(value):
    return value is None or value == "" or (isinstance(value, float) and np.isnan(value))


def _show(value):
    if _is_empty(value):
        return "-"
    return f"{value:g}" if isinstance(value, (float, np.floating)) else str(value)
