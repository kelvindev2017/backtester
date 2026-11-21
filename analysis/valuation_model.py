"""Block model of a share's fair price: inputs and assumptions in, an estimated price per share out.

The model has two sample rates. The quarterly part runs on every earnings release (and on the day a risk or product
item starts to apply): income and cost map -> earnings quality -> scenario engine -> risk weighting. Its outputs are
held between releases. The daily part runs on every close: rate model -> multiple builder -> product -> market check.
The rate model turns a forward policy rate into an N-year yield and moves the P/E anchor by the change in yield.

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
                    "Since", "Keywords", "Source", "Bull prob adj", "Bear prob adj"]
REGISTER_DEFAULTS = {"Item": "", "Type": RISK, "Kind": EVENT, "Timing": IN_HORIZON, "Probability %": np.nan,
                     "EPS impact %": np.nan, "P/E points": np.nan, "Status": PARKED, "Since": "", "Keywords": "",
                     "Source": "", "Bull prob adj": 0.0, "Bear prob adj": 0.0}
# The probability of an event item in the bull and bear case: the forward probability plus these points (0 to 100).
CASE_ADJUST = {"Bull": "Bull prob adj", "Bear": "Bear prob adj"}

DEFAULT_PARAMS = {
    # Earnings quality
    "eps_haircut_pct": 0.0,      # share of reported EPS removed (share-based pay, one-offs that keep coming back)
    "manual_eps": 0.0,           # trailing 12-month EPS per share typed in; 0 = use the reported quarters
    "quality_pe_pts": 0.0,       # P/E points taken off the acceptable P/E for weak earnings quality
    # Scenario engine
    "analyst_base": False,       # set by run_model: True when the latest base case is the analysts' EPS
    "base_eps_source": "Analysts",  # latest quarter's base EPS: Analysts (next 4 quarters' estimates) or Own rule
    "forward_growth": np.nan,    # base (forward) revenue growth % for the latest quarter; NaN = the auto rule below
    "forward_margin": np.nan,    # base (forward) net margin % for the latest quarter; NaN = current + base_margin_pts
    "growth_follow": 0.5,        # auto rule: base growth = this share of the trailing revenue growth
    "growth_floor": -20.0,
    "growth_cap": 40.0,
    "fallback_yoy": 0.0,         # trailing revenue growth % assumed when there is no revenue history
    "fallback_margin": 10.0,     # net margin % assumed when there is no revenue or net income
    "case_spread": "EPS volatility",  # how bear and bull differ from base: EPS volatility or Fixed gaps (below)
    "spread_k": 2.4,             # EPS volatility: bear/bull EPS = base EPS x exp(-/+ k x s)
    "spread_own_share": 0.25,    # ... s = this share of the company's own EPS volatility + the rest of the typical one
    "spread_typical": 0.073,     # ... typical (median) volatility of the yearly log change in trailing EPS
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
    "pe_basis": "Forward",       # Trailing: P/E on the last four quarters' EPS; Forward: on the next four quarters'
    "pe_lookback_years": 5,
    "peer_weight": 1.0,          # share of the P/E anchor from the outside view (fundamentals or peers); rest: own median
    "anchor_source": "Fundamentals",  # the outside view: Fundamentals (growth, gross margin, yield) or Peers' median P/E
    "growth_history_years": 3,   # fundamental P/E: revenue growth = the yearly growth (CAGR) over this many years
    "fund_growth_cap": 300.0,    # ... kept within this cap (high: some companies grow over 100% a year)
    "fund_growth_floor": -50.0,  # ... and floor, in %
    "equity_premium": 4.5,       # ... required return = N-year yield + this, in %
    "base_yield": 4.0,           # ... the yield used when no rates are loaded, in %
    "terminal_growth": 3.0,      # ... growth after the growth years, in %
    "growth_half_life_min": 0.5,  # ... years for half of the extra growth to fade, at a gross margin of gm_low or less
    "growth_half_life_max": 2.0,  # ... and at gm_high or more: a high gross margin is a moat, so growth lasts longer
    "gm_low": 20.0,
    "gm_high": 60.0,
    # Rules worked out from the data known on each day, so they hold for the whole history, not only the latest quarter.
    "pe_growth_latest_share": 0.5,   # growth for the P/E = this share of the latest year's growth + the rest of the history's
    "half_life_from_steadiness": True,  # steady growth earns a longer half-life than a low gross margin gives
    "steady_spread": 2.0,            # ... yearly growth varying by this many points or less earns steady_half_life
    "choppy_spread": 15.0,           # ... and by this many or more earns nothing beyond the gross-margin rule
    "steady_half_life": 6.0,
    "erp_from_beta": True,           # equity risk premium x the stock's beta to the market (3 years, weekly)
    "market_implied_erp": True,      # use the risk premium the market usually prices this stock at (below)
    "erp_lookback_years": 5,         # ... the usual premium is the median over this many years of month-ends
    "implied_erp_floor": -2.0,       # ... the implied premium is solved within this floor
    "implied_erp_cap": 20.0,         # ... and this cap, in %
    # Forward view of the P/E (latest quarter on; NaN = the historical value) and a premium kept apart from it.
    "forward_pe_growth": np.nan,      # growth the fundamental P/E rests on, %
    "forward_half_life": np.nan,      # years for half of the extra growth to fade
    "forward_equity_premium": np.nan,  # equity risk premium, %
    "market_premium": 0.0,            # % on top of the P/E for what no fundamental explains (brand, scarcity, story)
    "history_growth_follow": 0.5,  # Forward basis, before the latest report: expected EPS growth = this x past year's
    "history_growth_floor": -20.0,  # ... kept within this floor
    "history_growth_cap": 100.0,    # ... and this cap (%), so a fast grower is not cut to an ordinary one
    "history_recovery": 0.5,        # ... or, below its 3-year average EPS, this share of the gap back (the higher)
    "pe_floor": 5.0,
    "pe_cap": 100.0,
    # Triage gate
    "materiality_eps_pct": 1.0,  # smallest expected EPS effect (probability x impact) that enters the model
    "materiality_pe": 0.5,       # smallest P/E effect, in points, that enters the model
    # Rate model
    "yield_source": "Market",    # Market: the actual N-year Treasury yield; Policy model: built from the policy rate
    "use_forward_rate": False,   # True: the last day uses forward_rate instead of the policy rate on the day
    "forward_rate": 0.0,         # the policy rate % you expect (e.g. after the next central bank meeting)
    "yield_years": 10,           # N: maturity of the yield the stock is valued against
    "neutral_rate": 3.0,         # policy rate % the short rate drifts back to in the long run
    "rate_half_life": 2.0,       # years for half of a rate's gap to the neutral rate to fade
    "term_premium": 0.5,         # extra % a year for holding an N-year bond; cancels out of the yield changes
    "rate_pass_through": 1.0,    # share of a yield change that reaches the stock's required return
}

YIELD_MATURITIES = [1, 2, 3, 5, 7, 10, 20, 30]

# Assumptions about the future, set together in the Forward section; the other parameters describe today's data.
FORWARD_PARAMS = ["use_forward_rate", "forward_rate", "forward_growth", "forward_margin", "bear_growth_gap",
                  "bull_growth_gap", "bear_margin_pts", "bull_margin_pts", "forward_pe_growth", "forward_half_life",
                  "forward_equity_premium", "market_premium"]

# name -> (frame, label, unit, axis). Signals with the same axis can share a chart panel.
SIGNALS = {
    "revenue": ("quarterly", "Quarterly revenue", "", "amount"),
    "net_income": ("quarterly", "Quarterly net income", "", "amount"),
    "eps": ("quarterly", "Quarterly reported EPS", "", "eps"),
    "ttm_revenue": ("quarterly", "Revenue, last 4 quarters", "", "amount"),
    "revenue_yoy": ("quarterly", "Revenue growth, year on year", "%", "percent"),
    "net_margin": ("quarterly", "Net margin, last 4 quarters", "%", "share"),
    "gross_margin": ("quarterly", "Gross margin, last 4 quarters", "%", "share"),
    "revenue_cagr": ("quarterly", "Revenue growth a year over the growth history", "%", "percent"),
    "ttm_eps": ("quarterly", "Reported EPS, last 4 quarters", "", "eps"),
    "eps_spread": ("quarterly", "Own EPS volatility (std of the yearly log change, last 12 releases)", "", "count"),
    "case_spread": ("quarterly", "Case spread s used (EPS volatility)", "", "count"),
    "clean_eps": ("quarterly", "Clean base EPS", "", "eps"),
    "growth_bear": ("quarterly", "Bear growth", "%", "percent"),
    "growth_base": ("quarterly", "Base growth", "%", "percent"),
    "growth_bull": ("quarterly", "Bull growth", "%", "percent"),
    "event_effect": ("quarterly", "Expected effect of event items", "%", "percent"),
    "event_bear": ("quarterly", "Event items' effect, bear case", "%", "percent"),
    "event_base": ("quarterly", "Event items' effect, base case", "%", "percent"),
    "event_bull": ("quarterly", "Event items' effect, bull case", "%", "percent"),
    "eps_bear": ("quarterly", "Bear EPS", "", "eps"),
    "eps_base": ("quarterly", "Base EPS", "", "eps"),
    "eps_bull": ("quarterly", "Bull EPS", "", "eps"),
    "risk_adjusted_eps": ("quarterly", "Risk-adjusted EPS", "", "eps"),
    "pe_adjust": ("quarterly", "P/E points from general items", "pts", "points"),
    "quality_pe": ("quarterly", "P/E points from earnings quality", "pts", "points"),
    "items_modelled": ("quarterly", "Items in the model", "", "count"),
    "items_parked": ("quarterly", "Items parked", "", "count"),
    "close": ("daily", "Close", "", "price"),
    "policy_rate": ("daily", "Policy rate", "%", "rate"),
    "market_yield": ("daily", "Market N-year yield", "%", "rate"),
    "forward_rate": ("daily", "Policy rate used (forward on the last day)", "%", "rate"),
    "model_yield": ("daily", "Model N-year yield", "%", "rate"),
    "yield_ref": ("daily", "Model yield, median over the P/E look-back", "%", "rate"),
    "yield_gap": ("daily", "Yield vs the own P/E's look-back", "%", "rate_change"),
    "forward_gap": ("daily", "Yield change from the forward rate", "%", "rate_change"),
    "held_eps": ("daily", "Risk-adjusted EPS (held)", "", "eps"),
    "held_pe_adjust": ("daily", "P/E points (held)", "pts", "points"),
    "held_quality_pe": ("daily", "Earnings quality P/E points (held)", "pts", "points"),
    "own_anchor": ("daily", "Own median P/E", "x", "multiple"),
    "peer_anchor": ("daily", "Peers' median P/E", "x", "multiple"),
    "held_cagr": ("daily", "Revenue growth a year over the growth history (held)", "%", "percent"),
    "held_gross_margin": ("daily", "Gross margin, last 4 quarters (held)", "%", "share"),
    "growth_half_life": ("daily", "Half-life of the extra growth, years (gross margin and steadiness)", "", "count"),
    "held_yoy": ("daily", "Revenue growth, latest year (held)", "%", "percent"),
    "held_growth_spread": ("daily", "How much the yearly growth varies, points (held)", "", "count"),
    "beta": ("daily", "Beta to the market, 3 years weekly (raw)", "", "count"),
    "implied_erp": ("daily", "Risk premium the close implies (month-end)", "%", "percent"),
    "usual_erp": ("daily", "Usual implied risk premium (median over the P/E look-back)", "%", "percent"),
    "pe_erp": ("daily", "Equity risk premium used", "%", "percent"),
    "rule_growth": ("daily", "Growth for the P/E by the rule (history and latest year)", "%", "percent"),
    "rule_erp": ("daily", "Equity risk premium by the rule", "%", "percent"),
    "fundamental_pe": ("daily", "Fundamental P/E (growth, gross margin, yield)", "x", "multiple"),
    "outside_anchor": ("daily", "Outside view P/E (fundamental or peers)", "x", "multiple"),
    "outside_forward": ("daily", "Outside view P/E with your forward view and rate", "x", "multiple"),
    "pe_growth": ("daily", "Growth the fundamental P/E rests on (yours from the latest release)", "%", "percent"),
    "pe_half_life": ("daily", "Growth half-life used, years", "", "count"),
    "pe_before_premium": ("daily", "P/E before the market premium", "x", "multiple"),
    "premium_pct": ("daily", "Market premium applied", "%", "percent"),
    "implied_premium_pct": ("daily", "Premium the market pays over the model", "%", "percent"),
    "rate_anchor": ("daily", "P/E anchor after the rate model", "x", "multiple"),
    "peers_used": ("daily", "Peers in the median (P/E up to the cap)", "", "count"),
    "acceptable_pe": ("daily", "Acceptable P/E", "x", "multiple"),
    "margin_base": ("quarterly", "Base (forward) net margin", "%", "share"),
    "held_clean_eps": ("daily", "Clean trailing EPS (held)", "", "eps"),
    "now_yield": ("daily", "Model N-year yield at today's rate", "%", "rate"),
    "now_anchor": ("daily", "P/E anchor at today's rate", "x", "multiple"),
    "now_pe": ("daily", "P/E now (today's rate, no register items)", "x", "multiple"),
    "now_price": ("daily", "Model price, last quarter (no forward assumptions)", "", "price"),
    "now_residual_pct": ("daily", "Model price, last quarter, vs close", "%", "percent"),
    "model_price": ("daily", "Model price per share", "", "price"),
    "market_pe": ("daily", "Market price / risk-adjusted EPS", "x", "multiple"),
    "residual_pct": ("daily", "Model price vs close", "%", "percent"),
}

# Each block: its sample rate, what it reads, what it writes and the parameters it owns (key -> label).
BLOCKS = {
    "income_map": dict(
        title="Income and cost map", rate="quarterly", role="Logic: parsing only",
        inputs=["revenue", "net_income"], outputs=["ttm_revenue", "revenue_yoy", "net_margin", "gross_margin",
                                                     "revenue_cagr"],
        params={"fallback_yoy": "Growth % when there is no revenue history",
                "fallback_margin": "Net margin % when there is no net income"}),
    "earnings_quality": dict(
        title="Earnings quality", rate="quarterly", role="Param: adjustment rules",
        inputs=["eps"], outputs=["ttm_eps", "clean_eps", "quality_pe"],
        params={"eps_haircut_pct": "EPS haircut %", "manual_eps": "Typed-in trailing EPS (0 = reported)",
                "quality_pe_pts": "P/E deduction, points"}),
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
        inputs=["clean_eps", "revenue_yoy", "net_margin", "event_effect", "eps_spread"],
        outputs=["growth_bear", "growth_base", "growth_bull", "margin_base", "case_spread", "eps_bear", "eps_base",
                 "eps_bull"],
        params={"case_spread": "Bear and bull from: EPS volatility or Fixed gaps",
                "spread_k": "EPS volatility: k (cases at exp(-/+ k x s))",
                "spread_own_share": "EPS volatility: share of the own volatility in s",
                "spread_typical": "EPS volatility: typical volatility",
                "forward_growth": "Forward revenue growth % (blank = auto rule)",
                "forward_margin": "Forward net margin % (blank = auto rule)",
                "bear_growth_gap": "Bear: growth points below forward", "bull_growth_gap": "Bull: growth points above forward",
                "bear_margin_pts": "Bear: margin points vs forward", "bull_margin_pts": "Bull: margin points vs forward",
                "growth_follow": "Auto rule: base growth as a share of trailing growth",
                "growth_floor": "Auto rule: base growth floor %", "growth_cap": "Auto rule: base growth cap %",
                "base_margin_pts": "Auto rule: base margin change, points"}),
    "risk_weighting": dict(
        title="Risk weighting", rate="quarterly", role="Param: scenario weights",
        inputs=["eps_bear", "eps_base", "eps_bull"], outputs=["risk_adjusted_eps"],
        params={"weight_bear": "Bear weight", "weight_base": "Base weight", "weight_bull": "Bull weight"}),
    "rate_model": dict(
        title="Rate model", rate="daily", role="Param: forward rate, N",
        inputs=["policy_rate", "market_yield"], outputs=["forward_rate", "now_yield", "model_yield", "yield_ref",
                                                         "yield_gap", "forward_gap"],
        params={"use_forward_rate": "Use the forward rate on the last day (1 = yes)",
                "forward_rate": "Forward policy rate %", "yield_years": "Yield maturity N, years",
                "neutral_rate": "Neutral policy rate %", "rate_half_life": "Rate half-life, years",
                "term_premium": "Term premium %", "rate_pass_through": "Share of a yield change passed to the P/E"}),
    "multiple_builder": dict(
        title="Multiple builder", rate="daily", role="Param: P/E deductions",
        inputs=["own_anchor", "held_cagr", "held_gross_margin", "now_yield", "peer_anchor", "peers_used", "yield_gap",
                "forward_gap", "held_pe_adjust", "held_quality_pe"],
        outputs=["growth_half_life", "implied_erp", "usual_erp", "pe_growth", "pe_half_life", "fundamental_pe", "pe_before_premium", "premium_pct",
                 "implied_premium_pct", "outside_anchor", "now_pe", "rate_anchor", "acceptable_pe"],
        params={"pe_lookback_years": "Own P/E look-back, years",
                "peer_weight": "Share of the anchor from the outside view (rest: own P/E history)",
                "anchor_source": "Outside view: Fundamentals or Peers",
                "growth_history_years": "Fundamental P/E: growth history, years",
                "fund_growth_cap": "Fundamental P/E: growth cap %", "fund_growth_floor": "Fundamental P/E: growth floor %",
                "equity_premium": "Fundamental P/E: equity risk premium %",
                "market_implied_erp": "Fundamental P/E: use the market's usual implied risk premium (1 = yes)",
                "erp_lookback_years": "Fundamental P/E: look-back for the usual implied premium, years",
                "base_yield": "Fundamental P/E: yield without rate data %",
                "terminal_growth": "Fundamental P/E: growth after the growth years %",
                "growth_half_life_min": "Fundamental P/E: growth half-life at a low gross margin, years",
                "growth_half_life_max": "Fundamental P/E: growth half-life at a high gross margin, years",
                "gm_low": "Fundamental P/E: low gross margin %", "gm_high": "Fundamental P/E: high gross margin %",
                "forward_pe_growth": "Forward: growth for the P/E % (blank = history)",
                "forward_half_life": "Forward: growth half-life, years (blank = from gross margin)",
                "forward_equity_premium": "Forward: equity risk premium % (blank = the default)",
                "market_premium": "Forward: market premium % on the P/E",
                "pe_floor": "P/E floor", "pe_cap": "P/E cap",
                "history_growth_follow": "Forward P/E history: share of past EPS growth expected",
                "history_growth_floor": "Forward P/E history: expected growth floor %",
                "history_growth_cap": "Forward P/E history: expected growth cap %",
                "history_recovery": "Forward P/E history: share of a slump expected back"}),
    "product": dict(
        title="Product", rate="daily", role="EPS x acceptable P/E",
        inputs=["held_clean_eps", "held_eps", "now_pe", "acceptable_pe"], outputs=["now_price", "model_price"], params={}),
    "market_check": dict(
        title="Market check", rate="daily", role="Out: implied dials",
        inputs=["model_price", "close"], outputs=["now_residual_pct", "residual_pct", "market_pe"], params={}),
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
            "fallback_margin": 9.8, "bear_growth_gap": 28.0, "bull_growth_gap": 16.0, "bear_margin_pts": -5.0,
            "base_margin_pts": 4.0, "bull_margin_pts": 3.0, "peer_weight": 1.0, "case_spread": "Fixed gaps",
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
                                                          ["Probability %", "EPS impact %", "P/E points",
                                                           "Bull prob adj", "Bear prob adj"]})


def preset_register(ticker):
    rows = PRESETS.get(str(ticker).upper(), {}).get("register")
    return clean_register(pd.DataFrame(rows, columns=REGISTER_COLUMNS[:len(rows[0])])) if rows else empty_register()


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
    for column in CASE_ADJUST.values():
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(-100, 100)
    return frame[frame["Item"] != ""].reset_index(drop=True)


def route_items(register, params=None):
    """Triage gate and item router for every item.

    Adds ``Route`` (which block the item feeds), ``Effect`` (expected EPS change in % for the scenario engine, P/E
    points for the multiple builder; negative for a risk, positive for a product), ``Effect Bear/Base/Bull`` (the
    same at each case's probability: the forward one plus the item's bull or bear adjustment) and ``Gate`` (``Modelled`` or the
    reason the item is parked). An event inside the forecast year goes to the scenario engine; a general item or
    anything beyond the horizon goes to the multiple builder.
    """

    params = model_params(params)
    frame = clean_register(register)
    routes, effects, gates, by_case = [], [], [], {name: [] for name in SCENARIOS}
    for row in frame.to_dict("records"):
        sign = -1.0 if row["Type"] == RISK else 1.0
        forward = row["Probability %"]
        case_probability = {"Base": forward, **{name: float(np.clip(forward + row[column], 0, 100))
                                                if pd.notna(forward) else np.nan
                                                for name, column in CASE_ADJUST.items()}}
        if row["Kind"] == EVENT and row["Timing"] == IN_HORIZON:
            route, bar, unit = TO_SCENARIOS, params["materiality_eps_pct"], "% of EPS"
            case_effect = {name: sign * value / 100 * row["EPS impact %"] for name, value in case_probability.items()}
            needs = "a probability and an EPS impact"
        else:
            route, bar, unit = TO_MULTIPLE, params["materiality_pe"], "P/E points"
            event = row["Kind"] == EVENT and pd.notna(forward)
            case_effect = {name: sign * (value / 100 if event else 1.0) * row["P/E points"]
                           for name, value in case_probability.items()}
            needs = "P/E points"
        effect = case_effect["Base"]
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
        for name in SCENARIOS:
            by_case[name].append(case_effect[name] if np.isfinite(effect) else np.nan)
    frame["Route"], frame["Effect"], frame["Gate"] = routes, effects, gates
    for name in SCENARIOS:
        frame[f"Effect {name}"] = by_case[name]
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


def case_effects(routed, on, params):
    """Modelled items in force on ``on``, per case: ({case: EPS change as a fraction}, P/E points).

    The EPS change differs by case (each case has its own probability); the P/E points are the scenario-weighted
    average over the cases, because there is one acceptable P/E.
    """

    if routed is None or routed.empty:
        return {name: 0.0 for name in SCENARIOS}, 0.0
    active = routed[routed["Gate"] == MODELLED]
    if on is not None and not active.empty:
        since = _since_dates(active)
        active = active[since.isna() | (since <= pd.Timestamp(on))]
    column = {name: f"Effect {name}" if f"Effect {name}" in active else "Effect" for name in SCENARIOS}
    events = {name: max(-1.0, float(active.loc[active["Route"] == TO_SCENARIOS, column[name]].sum() / 100))
              for name in SCENARIOS}
    points = sum(weight * float(active.loc[active["Route"] == TO_MULTIPLE, column[name]].sum())
                 for weight, name in zip(scenario_weights(params), SCENARIOS))
    return events, float(points)


def scenario_weights(params):
    """Bear, base and bull weights as fractions that sum to 1 (equal weights when all are zero)."""

    raw = np.array([max(0.0, float(params[f"weight_{name.lower()}"])) for name in SCENARIOS])
    return raw / raw.sum() if raw.sum() > 0 else np.full(len(SCENARIOS), 1 / len(SCENARIOS))


def base_growth_and_margin(revenue_yoy, net_margin, params, forward=False):
    """(trailing growth, base growth, current margin, base margin), all in %.

    The base (forward) growth and margin are yours on the latest quarter (``forward``) when set; otherwise, and on
    every earlier quarter, the auto rule: a share of the trailing growth within a floor and cap, and the current margin
    plus the base margin change.
    """

    yoy = revenue_yoy if np.isfinite(revenue_yoy) else params["fallback_yoy"]
    margin = net_margin if np.isfinite(net_margin) and net_margin > 0 else params["fallback_margin"]
    growth = float(np.clip(params["growth_follow"] * yoy, params["growth_floor"], params["growth_cap"]))
    base_margin = margin + params["base_margin_pts"]
    if forward and np.isfinite(params.get("forward_growth", np.nan)):
        growth = float(params["forward_growth"])
    if forward and np.isfinite(params.get("forward_margin", np.nan)):
        base_margin = float(params["forward_margin"])
    return yoy, growth, margin, base_margin


def case_spread(eps_spread, params):
    """The case spread s: a share of the company's own EPS volatility plus the rest of the typical one (the typical
    one alone without enough history)."""

    typical = float(params["spread_typical"])
    if eps_spread is None or not np.isfinite(eps_spread) or eps_spread < 0:
        return typical
    share = float(np.clip(params["spread_own_share"], 0, 1))
    return share * float(eps_spread) + (1 - share) * typical


def scenario_point(ttm_eps, revenue_yoy, net_margin, params, event_effect=0.0, forward=False, eps_spread=np.nan):
    """The quarterly chain for one date: earnings quality, scenario engine and risk weighting.

    ``revenue_yoy`` and ``net_margin`` are in %; ``event_effect`` is the fraction of EPS the modelled event items are
    expected to add or remove, one number or one per case. The base case grows the clean EPS with the base revenue
    growth and scales it by the base net margin over the current one. With ``case_spread`` Fixed gaps, bear and bull
    do the same with their own growth and margin (the base plus or minus the gaps); with EPS volatility, they are the
    base EPS times exp(-k s) and exp(+k s), s from ``case_spread`` and ``eps_spread`` (the company's own EPS
    volatility; a forward test on 60 stocks, 2019-2025, set k, the share and the typical volatility). ``forward``: the latest quarter, where your forward growth and margin apply.
    """

    clean = ttm_eps * (1 - params["eps_haircut_pct"] / 100)
    _, base_growth, margin, base_margin = base_growth_and_margin(revenue_yoy, net_margin, params, forward)
    growth = {"Bear": base_growth - params["bear_growth_gap"], "Base": base_growth,
              "Bull": base_growth + params["bull_growth_gap"]}
    case_margin = {"Bear": base_margin + params["bear_margin_pts"], "Base": base_margin,
                   "Bull": base_margin + params["bull_margin_pts"]}
    events = event_effect if isinstance(event_effect, dict) else {name: event_effect for name in SCENARIOS}
    weights = scenario_weights(params)
    point = {"clean_eps": clean, "event_effect": float(sum(w * events[n] for w, n in zip(weights, SCENARIOS))) * 100,
             "quality_pe": -abs(float(params["quality_pe_pts"])), "margin_base": base_margin}
    by_volatility = params.get("case_spread", "Fixed gaps") == "EPS volatility"
    spread = case_spread(eps_spread, params)
    point["case_spread"] = spread if by_volatility else np.nan
    factor = {"Bear": np.exp(-params["spread_k"] * spread), "Base": 1.0, "Bull": np.exp(params["spread_k"] * spread)}
    for name in SCENARIOS:
        key = name.lower()
        if by_volatility:
            # One spread on EPS, from how much the company's earnings have moved in a year.
            margin_ratio = max(0.0, base_margin) / margin if margin > 0 else 1.0
            point[f"growth_{key}"] = base_growth
            case_eps = clean * (1 + base_growth / 100) * margin_ratio * factor[name]
        else:
            margin_ratio = max(0.0, case_margin[name]) / margin if margin > 0 else 1.0
            point[f"growth_{key}"] = growth[name]
            case_eps = clean * (1 + growth[name] / 100) * margin_ratio
        point[f"event_{key}"] = events[name] * 100
        point[f"eps_{key}"] = case_eps * (1 + events[name])
    point["risk_adjusted_eps"] = float(sum(weight * point[f"eps_{name.lower()}"]
                                           for weight, name in zip(weights, SCENARIOS)))
    return point


def _trailing(series, periods=4):
    series = pd.to_numeric(series, errors="coerce").dropna()
    return series.rolling(periods, min_periods=periods).sum()


def quarterly_inputs(quarters, growth_years=3):
    """Income and cost map: trailing revenue, its growth and the net margin at each release date.

    ``quarters`` is indexed by the date each quarter became public, with any of the columns Revenue, Net_Income and
    EPS. Growth is the last four quarters against the four before; with under eight quarters it is the latest
    quarter against the same quarter a year earlier.
    """

    frame = pd.DataFrame(index=pd.DatetimeIndex(quarters.index).sort_values().unique()) if quarters is not None else pd.DataFrame()
    if quarters is None or quarters.empty:
        return pd.DataFrame(columns=["revenue", "net_income", "eps", "ttm_revenue", "revenue_yoy", "net_margin", "ttm_eps",
                                     "gross_margin", "revenue_cagr", "growth_spread", "eps_spread"])
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
    ttm_gross = _trailing(column("Gross_Profit"))
    gross_margin = (ttm_gross.reindex(ttm_revenue.index) / ttm_revenue.where(ttm_revenue > 0)) * 100
    # Yearly growth over the growth history; fewer years when the history is shorter (at least one).
    cagr = pd.Series(np.nan, index=ttm_revenue.index, dtype=float)
    for years in range(max(1, int(growth_years)), 0, -1):
        earlier = ttm_revenue.shift(4 * years)
        ratio = ttm_revenue / earlier.where(earlier > 0)
        cagr = cagr.combine_first((ratio.where(ratio > 0) ** (1 / years) - 1) * 100)
    frame["revenue"], frame["net_income"], frame["eps"] = revenue, net_income, eps
    ttm_eps = _trailing(eps)
    log_eps = np.log(ttm_eps.where(ttm_eps > 0))
    # How much trailing EPS moved in a year (log change), its spread over the last 12 releases (at least 8).
    eps_spread = (log_eps - log_eps.shift(4)).rolling(12, min_periods=8).std()
    for name, series in [("ttm_revenue", ttm_revenue), ("revenue_yoy", yoy), ("net_margin", margin),
                         ("ttm_eps", _trailing(eps)), ("gross_margin", gross_margin.dropna()),
                         ("revenue_cagr", cagr.dropna()),
                         # How much the yearly growth varied over the last 12 releases (at least 8), as known then.
                         ("growth_spread", yoy.dropna().rolling(12, min_periods=8).std().dropna()),
                         ("eps_spread", eps_spread.dropna())]:
        # A quarter known for one measure but not another keeps the last value of the other.
        frame[name] = series.reindex(frame.index.union(series.index)).ffill().reindex(frame.index)
    return frame


def rate_weight(years, half_life):
    """Share of a policy rate change that shows up in the N-year yield.

    The expected short rate starts at the policy rate and drifts back to the neutral rate, losing half its gap every
    ``half_life`` years; the N-year yield is the average of that path (plus a term premium). The weight is the
    average of exp(-lambda t) over 0..N: 1 for an instant maturity or a rate that never fades, about 0.28 for a
    10-year yield with a 2-year half-life.
    """

    years, half_life = float(years), float(half_life)
    if years <= 0 or half_life <= 0:
        return 1.0 if years <= 0 else 0.0
    decay = np.log(2) / half_life * years
    return float((1 - np.exp(-decay)) / decay)


def model_yield(policy_rate, params):
    """The N-year yield implied by a policy rate (a number or a Series), in %."""

    params = model_params(params)
    weight = rate_weight(params["yield_years"], params["rate_half_life"])
    return params["neutral_rate"] + (policy_rate - params["neutral_rate"]) * weight + params["term_premium"]


def rate_adjusted_pe(pe, yield_change, pass_through=1.0):
    """A P/E after a change in yield (in % points): the earnings yield 1/P/E moves by the same amount.

    For a stock valued as earnings / (required return - growth), the earnings yield is required return minus growth,
    so a rise in the required return lifts it point for point. A long-duration stock (high P/E) loses more: at 25x a
    yield up 0.15 points takes the P/E to 24.1x, at 50x to 46.5x. An earnings yield pushed to zero or below leaves
    the P/E unbounded (the multiple builder's cap then applies).
    """

    if isinstance(pe, pd.Series):
        change = pd.Series(yield_change, index=pe.index, dtype=float) if np.ndim(yield_change) else yield_change
        earnings_yield = 1 / pe + float(pass_through) * change / 100
        return (1 / earnings_yield).where(earnings_yield > 0, np.inf).where(pe.notna())
    if pe is None or not np.isfinite(pe) or pe <= 0:
        return np.nan
    earnings_yield = 1 / pe + float(pass_through) * yield_change / 100
    return 1 / earnings_yield if earnings_yield > 0 else np.inf


def growth_half_life(gross_margin, params):
    """Years for half of the growth above the terminal rate to fade: longer with a higher gross margin."""

    params = model_params(params)
    low, high = params["growth_half_life_min"], params["growth_half_life_max"]
    if gross_margin is None or not np.isfinite(gross_margin):
        return (low + high) / 2
    span = max(1e-9, params["gm_high"] - params["gm_low"])
    share = float(np.clip((gross_margin - params["gm_low"]) / span, 0, 1))
    return low + (high - low) * share


def fundamental_pe(growth, gross_margin, yield_pct, params, horizon=40, half_life=None):
    """The forward P/E the company's own growth and gross margin justify at a given yield.

    Next year's earnings are 1. They grow at ``growth`` % (the company's revenue growth a year over its growth
    history, within a floor and cap) and the growth above the terminal rate fades away, halving every half-life
    years; the gross margin sets the half-life (a high gross margin is pricing power, so growth lasts longer). The
    fade is in log terms, so 100% or 300% growth is handled: the P/E rises with it without exploding (at a 60% gross
    margin and a 4.3% yield: 30% -> ~28x, 100% -> ~71x, 300% -> ~340x). The P/E is the value of that stream
    discounted at the N-year yield plus the equity risk premium (at least 1 point above the terminal growth), with the
    terminal growth for ever after ``horizon`` years. NaN without a growth rate.
    """

    params = model_params(params)
    if growth is None or not np.isfinite(growth):
        return np.nan
    g = float(np.clip(growth, params["fund_growth_floor"], params["fund_growth_cap"])) / 100
    terminal = params["terminal_growth"] / 100
    yield_pct = yield_pct if yield_pct is not None and np.isfinite(yield_pct) else params["base_yield"]
    k = max(yield_pct / 100 + params["equity_premium"] / 100, terminal + 0.01)
    half_life = max(1e-6, half_life if half_life is not None and np.isfinite(half_life)
                    else growth_half_life(gross_margin, params))
    start, end = np.log1p(g), np.log1p(terminal)
    earnings, value = 1.0, 0.0
    for t in range(1, horizon + 1):
        value += earnings / (1 + k) ** t
        earnings *= np.exp(end + (start - end) * 0.5 ** (t / half_life))
    value += earnings / (k - terminal) / (1 + k) ** horizon
    return float(value)


def _fundamental_series(growth, half_lives, yields, params, erps=None):
    """``fundamental_pe`` day by day (computed once per distinct growth, half-life, yield and risk premium)."""

    params = model_params(params)
    erps = [params["equity_premium"]] * len(growth) if erps is None else list(erps)
    out, cache = [], {}
    for g, h, y, e in zip(growth, half_lives, yields, erps):
        e = float(e) if e is not None and np.isfinite(e) else params["equity_premium"]
        key = (round(g, 4) if np.isfinite(g) else None, round(h, 4), round(y, 4) if np.isfinite(y) else None, round(e, 4))
        if key not in cache:
            cache[key] = fundamental_pe(g, None, y, {**params, "equity_premium": e}, half_life=h)
        out.append(cache[key])
    return pd.Series(out, index=growth.index, dtype=float)


def steady_half_life(spread, growth, params):
    """Growth half-life that steady growth earns: ``steady_half_life`` years when the yearly growth varies by
    ``steady_spread`` points or less, falling to the low gross-margin half-life at ``choppy_spread`` or more.
    NaN without a spread or without positive growth."""

    params = model_params(params)
    if spread is None or growth is None or not np.isfinite(spread) or not np.isfinite(growth) or growth <= 0:
        return np.nan
    span = max(1e-9, params["choppy_spread"] - params["steady_spread"])
    share = float(np.clip((params["choppy_spread"] - spread) / span, 0, 1))
    low = params["growth_half_life_min"]
    return low + (params["steady_half_life"] - low) * share


def rolling_beta(close, market_close, days, years=3):
    """The stock's beta to the market by day: weekly log returns over the past ``years`` years (156 weeks, at least
    52), as known on each day. NaN without market prices."""

    out = pd.Series(np.nan, index=days, dtype=float)
    if close is None or market_close is None or not len(close) or not len(pd.Series(market_close).dropna()):
        return out
    both = pd.concat([pd.Series(close, dtype=float), pd.Series(market_close, dtype=float)], axis=1, join="inner").dropna()
    if both.empty:
        return out
    weekly = np.log(both.resample("W-FRI").last()).diff().dropna()
    window = 52 * years
    beta = (weekly.iloc[:, 0].rolling(window, min_periods=52).cov(weekly.iloc[:, 1])
            / weekly.iloc[:, 1].rolling(window, min_periods=52).var())
    return beta.reindex(beta.index.union(days)).ffill().reindex(days)


def adjusted_beta(beta):
    """Blume-adjusted beta, 0.67 x raw + 0.33 x 1, as data vendors report it: a raw beta measured over a few years
    drifts towards 1, and a 3-year raw beta can be near 0 for a defensive stock (KO, PG, PEP were ~0.1 in 2023-26)."""

    return 0.67 * pd.Series(beta, dtype=float) + 0.33


def risk_premium_from_beta(beta, params):
    """The equity risk premium scaled by the adjusted beta (a calmer stock needs a lower return), within 2.5% and
    8%; the plain premium where beta is unknown."""

    params = model_params(params)
    base = float(params["equity_premium"])
    return (base * adjusted_beta(beta).clip(0.5, 1.8)).clip(2.5, 8.0).fillna(base)


def implied_erp(market_pe, growth, half_life, yield_pct, params):
    """The equity risk premium (%) at which the fundamental P/E equals ``market_pe``: what the price implies.

    The fundamental P/E falls as the premium rises, so it is found by bisection within the floor and cap (and above
    the premium that would put the required return at the terminal growth + 1 point). NaN without a market P/E or
    growth; the floor or cap when the market P/E lies beyond what the range can reach.
    """

    params = model_params(params)
    if not (market_pe is not None and np.isfinite(market_pe) and market_pe > 0 and growth is not None
            and np.isfinite(growth) and half_life is not None and np.isfinite(half_life)):
        return np.nan
    yield_pct = yield_pct if yield_pct is not None and np.isfinite(yield_pct) else params["base_yield"]
    low = max(float(params["implied_erp_floor"]), params["terminal_growth"] + 1.0 - yield_pct)
    high = max(low, float(params["implied_erp_cap"]))

    def pe_at(erp):
        return fundamental_pe(growth, None, yield_pct, {**params, "equity_premium": erp}, half_life=half_life)

    if market_pe >= pe_at(low):
        return low
    if market_pe <= pe_at(high):
        return high
    for _ in range(40):
        middle = (low + high) / 2
        if pe_at(middle) > market_pe:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def usual_implied_erp(market_pe, growth, half_life, yields, params, lookback_years):
    """(implied premium at each month-end, its median over the look-back known the day before, by day).

    The usual premium is what the market has been asking of this stock: holding it and moving the yield moves the
    P/E as the market would for a change in rates, while today's own implied premium says how far the close strays.
    """

    days = market_pe.index
    if not len(days):
        empty = pd.Series(np.nan, index=days, dtype=float)
        return empty, empty
    ends = pd.Series(days, index=days).groupby(days.to_period("M")).last()
    values = pd.Series([implied_erp(market_pe.get(day), growth.get(day), half_life.get(day), yields.get(day), params)
                        for day in ends], index=pd.DatetimeIndex(ends.values), dtype=float)
    months = max(1, int(round(12 * float(lookback_years))))
    usual = values.shift(1).rolling(months, min_periods=min(months, 12)).median()
    on_day = lambda series: series.reindex(series.index.union(days)).ffill().reindex(days)
    return on_day(values), on_day(usual)


def forward_pe_params(params):
    """The parameters for the forward fundamental P/E: your equity risk premium where you set one."""

    params = model_params(params)
    erp = params.get("forward_equity_premium", np.nan)
    return {**params, "equity_premium": float(erp)} if np.isfinite(erp) else params


def blend_anchor(own, peer, share):
    """The P/E anchor: ``share`` of the peers' P/E and the rest of the own one; either alone when the other is missing."""

    share = float(np.clip(share, 0, 1))
    own_series, peer_series = pd.Series(own, dtype=float), pd.Series(peer, dtype=float)
    if share >= 1:
        anchor = peer_series.combine_first(own_series)
    elif share <= 0:
        anchor = own_series.combine_first(peer_series)
    else:
        blended = share * peer_series + (1 - share) * own_series
        anchor = blended.where(peer_series.notna() & own_series.notna(), peer_series.combine_first(own_series))
    return anchor if np.ndim(own) or np.ndim(peer) else float(anchor.iloc[0])


def run_model(quarters, close, register=None, params=None, peer_pe=None, start=None, own_pe=None, policy_rate=None,
              market_yield=None, market_close=None, analyst_growth=None):
    """Run the whole model over history.

    quarters: results by the date they became public (columns Revenue, Net_Income, EPS; any may be missing).
    close: daily close of the share. register: the risk and product items. peer_pe: daily P/E of each peer, one
    column per peer, on the basis the anchor uses. start: first day to report (earlier data still feeds the trailing
    sums and medians). market_close: the market's close (SPY) for the stock's beta. own_pe: the share's own daily
    P/E for its anchor; None uses the trailing P/E from ``quarters``
    (pass a forward P/E here, with the peers' forward P/E, for a forward anchor). policy_rate: the central bank rate
    in % by date; None (or empty) switches the rate model off. market_yield: the market's N-year yield in %, shown
    next to the model's yield as a check; with ``yield_source`` Market it is the yield the stock is discounted at, and
    the policy-rate model only adds the change a forward rate would make. analyst_growth: analysts' EPS growth %,
    next four quarters over the last four; with ``base_eps_source`` Analysts it sets the latest quarter's base case
    (growth = that, net margin held), so base EPS = analysts' EPS less the haircut.

    Returns a dict with ``quarterly`` (one row per update of the quarterly part), ``daily`` (one row per close),
    ``items`` (the register after the triage gate and router) and ``params``. Nothing on a date uses data from later.
    """

    params = model_params(params)
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    close = close.loc[~close.index.duplicated(keep="last")]
    routed = route_items(register, params)
    inputs = quarterly_inputs(quarters, params["growth_history_years"])
    params["analyst_base"] = False
    if (params["base_eps_source"] == "Analysts" and analyst_growth is not None and np.isfinite(analyst_growth)
            and len(inputs)):
        # The base case is the analysts' EPS: their growth, with the net margin held (margin ratio 1).
        latest_margin = inputs["net_margin"].dropna() if "net_margin" in inputs else pd.Series(dtype=float)
        params["forward_growth"] = float(analyst_growth)
        params["forward_margin"] = (float(latest_margin.iloc[-1]) if len(latest_margin) and latest_margin.iloc[-1] > 0
                                    else float(params["fallback_margin"]))
        params["analyst_base"] = True

    # The quarterly part also runs on the day an item starts to apply, with the last results held.
    item_dates = _since_dates(routed[routed["Gate"] == MODELLED]).dropna()
    first_day = close.index[0] if len(close) else None
    updates = pd.DatetimeIndex(inputs.index).union(pd.DatetimeIndex(item_dates))
    if params["manual_eps"] > 0 and first_day is not None:
        updates = updates.union(pd.DatetimeIndex([first_day]))
    updates = updates.sort_values().unique()
    held = inputs.reindex(inputs.index.union(updates)).ffill().reindex(updates) if len(updates) else inputs
    rows = []
    last_release = inputs.index.max() if len(inputs) else None
    for when, row in held.iterrows():
        ttm_eps = params["manual_eps"] if params["manual_eps"] > 0 else row.get("ttm_eps", np.nan)
        events, points = case_effects(routed, when, params)
        forward = last_release is None or when >= last_release
        point = scenario_point(ttm_eps, row.get("revenue_yoy", np.nan), row.get("net_margin", np.nan), params, events,
                               forward, row.get("eps_spread", np.nan) if params["manual_eps"] <= 0 else np.nan)
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
        daily["held_quality_pe"] = -abs(float(params["quality_pe_pts"]))
        daily["held_clean_eps"] = np.nan
        daily["held_cagr"], daily["held_gross_margin"] = np.nan, np.nan
        daily["held_yoy"], daily["held_growth_spread"] = np.nan, np.nan
    else:
        # Hold blocks: each quarterly output keeps its last value until the next update.
        both = quarterly.reindex(quarterly.index.union(close.index)).ffill().reindex(close.index)
        daily["held_eps"], daily["held_pe_adjust"], held_ttm = both["risk_adjusted_eps"], both["pe_adjust"].fillna(0.0), both["ttm_eps"]
        daily["held_quality_pe"] = both["quality_pe"].fillna(0.0)
        daily["held_clean_eps"] = both["clean_eps"]
        daily["held_cagr"] = both["revenue_cagr"] if "revenue_cagr" in both else np.nan
        daily["held_yoy"] = both["revenue_yoy"] if "revenue_yoy" in both else np.nan
        daily["held_growth_spread"] = both["growth_spread"] if "growth_spread" in both else np.nan
        daily["held_gross_margin"] = both["gross_margin"] if "gross_margin" in both else np.nan

    # Multiple builder. The own anchor is the median trailing P/E of the look-back, up to the day before, so the
    # day's own price does not set its own fair multiple.
    if own_pe is None:
        own_pe = (close / held_ttm.where(held_ttm > 0)).replace([np.inf, -np.inf], np.nan)
    else:
        own_pe = pd.to_numeric(pd.Series(own_pe), errors="coerce")
        own_pe = own_pe.where(own_pe > 0).reindex(own_pe.index.union(close.index)).ffill(limit=5).reindex(close.index)
    window = max(2, int(round(252 * float(params["pe_lookback_years"]))))
    daily["own_anchor"] = own_pe.shift(1).rolling(window, min_periods=min(window, 60)).median()
    if peer_pe is not None and not pd.DataFrame(peer_pe).empty:
        peers = pd.DataFrame(peer_pe).apply(pd.to_numeric, errors="coerce")
        peers = peers.where(peers > 0).reindex(peers.index.union(close.index)).ffill(limit=5).reindex(close.index)
        # A P/E above the cap rests on earnings near zero and says nothing about a fair multiple: like a loss, it
        # sits out that day's median.
        usable = peers.where(peers <= float(params["pe_cap"]))
        daily["peer_anchor"] = usable.median(axis=1, skipna=True)
        daily["peers_used"] = usable.notna().sum(axis=1).astype(float)
    else:
        peers = pd.DataFrame(index=close.index)
        daily["peer_anchor"] = np.nan
        daily["peers_used"] = 0.0

    # Rate model. The own anchor is a median over years of past rates, so it moves with the yield against the median
    # yield of those years; the peers' P/E is today's and already prices today's rate, so it moves only with the
    # forward rate's surprise.
    for name in ["policy_rate", "forward_rate", "model_yield", "now_yield", "yield_ref", "market_yield"]:
        daily[name] = np.nan
    daily["yield_gap"], daily["forward_gap"] = 0.0, 0.0
    rates = pd.to_numeric(pd.Series(policy_rate, dtype=float), errors="coerce").dropna() if policy_rate is not None else pd.Series(dtype=float)
    if len(rates) and len(close):
        rates = rates.sort_index()
        rates = rates.loc[~rates.index.duplicated(keep="last")]
        on_day = rates.reindex(rates.index.union(close.index)).ffill().reindex(close.index)
        used = on_day.copy()
        if params["use_forward_rate"]:
            used.iloc[-1] = float(params["forward_rate"])
        actual_yield = model_yield(on_day, params)
        market_now = _market_yield_on(market_yield, close.index) if params["yield_source"] == "Market" else None
        if market_now is not None:
            # Discount at the market's yield; the policy-rate model only adds what a forward rate would change.
            actual_yield = market_now.combine_first(actual_yield)
        daily["policy_rate"], daily["forward_rate"] = on_day, used
        daily["model_yield"] = actual_yield + (model_yield(used, params) - model_yield(on_day, params))
        daily["now_yield"] = actual_yield
        daily["yield_ref"] = actual_yield.shift(1).rolling(window, min_periods=min(window, 60)).median()
        daily["yield_gap"] = (daily["model_yield"] - daily["yield_ref"]).fillna(0.0)
        daily["forward_gap"] = (daily["model_yield"] - actual_yield).fillna(0.0)
    market_on_day = _market_yield_on(market_yield, close.index)
    if market_on_day is not None:
        daily["market_yield"] = market_on_day
        if not len(rates) and params["yield_source"] == "Market":
            # No policy rate: the rate model is off, but the stock is still discounted at the market's yield.
            daily["now_yield"], daily["model_yield"] = market_on_day, market_on_day
    passed = float(params["rate_pass_through"])
    # The outside view: the P/E the company's growth and gross margin justify at the day's yield, or the peers'.
    # The rules below use only what was known on each day, so they hold for the whole history.
    margin_half = pd.Series([growth_half_life(margin, params) for margin in daily["held_gross_margin"].astype(float)],
                            index=daily.index, dtype=float)
    cagr, latest = daily["held_cagr"].astype(float), daily["held_yoy"].astype(float)
    share = float(np.clip(params["pe_growth_latest_share"], 0, 1))
    growth = (share * latest + (1 - share) * cagr).combine_first(cagr).combine_first(latest)
    if growth.isna().all():
        growth = growth.fillna(float(params["fallback_yoy"]))
    if params["half_life_from_steadiness"]:
        steady = pd.Series([steady_half_life(spread, g, params) for spread, g in
                            zip(daily["held_growth_spread"].astype(float), growth)], index=daily.index, dtype=float)
        margin_half = pd.concat([margin_half, steady], axis=1).max(axis=1)
    daily["growth_half_life"], daily["rule_growth"] = margin_half, growth
    daily["beta"] = rolling_beta(close, market_close, daily.index)
    base_erp = (risk_premium_from_beta(daily["beta"], params) if params["erp_from_beta"]
                else pd.Series(float(params["equity_premium"]), index=daily.index))
    now_yield = daily["now_yield"].astype(float)
    daily["implied_erp"], daily["usual_erp"] = np.nan, np.nan
    if params["market_implied_erp"] and params["anchor_source"] == "Fundamentals":
        # Respect the market: the premium it has usually asked of this stock, with today's yield.
        market_pe = daily["close"] / daily["held_eps"].where(daily["held_eps"] > 0)
        implied, usual = usual_implied_erp(market_pe, growth, daily["growth_half_life"], now_yield, params,
                                           params["erp_lookback_years"])
        daily["implied_erp"], daily["usual_erp"] = implied, usual
        base_erp = usual.combine_first(base_erp)
    daily["rule_erp"] = base_erp
    daily["fundamental_pe"] = _fundamental_series(growth, daily["growth_half_life"], now_yield, params, base_erp)
    # Your forward view of the P/E applies from the latest release, like the other forward assumptions.
    forward_days = (pd.Series(daily.index >= last_release, index=daily.index) if last_release is not None
                    else pd.Series(True, index=daily.index))
    pe_growth, pe_half_life = growth.copy(), daily["growth_half_life"].astype(float).copy()
    if np.isfinite(params["forward_pe_growth"]):
        pe_growth[forward_days] = float(params["forward_pe_growth"])
    if np.isfinite(params["forward_half_life"]):
        pe_half_life[forward_days] = float(params["forward_half_life"])
    pe_erp = base_erp.copy()
    if np.isfinite(params["forward_equity_premium"]):
        pe_erp[forward_days] = float(params["forward_equity_premium"])
    daily["pe_growth"], daily["pe_half_life"], daily["pe_erp"] = pe_growth, pe_half_life, pe_erp
    daily["premium_pct"] = np.where(forward_days, float(params["market_premium"]), 0.0)
    fundamentals = params["anchor_source"] == "Fundamentals"
    if fundamentals:
        outside_now = daily["fundamental_pe"]
        rate_yield = daily["model_yield"].astype(float)
        outside_rate = _fundamental_series(pe_growth, pe_half_life, rate_yield, params, pe_erp)
    else:
        outside_now = daily["peer_anchor"]
        outside_rate = rate_adjusted_pe(daily["peer_anchor"], daily["forward_gap"], passed)
    daily["outside_anchor"], daily["outside_forward"] = outside_now, outside_rate
    own_rate = rate_adjusted_pe(daily["own_anchor"], daily["yield_gap"], passed)
    daily["anchor"] = blend_anchor(daily["own_anchor"], outside_now, params["peer_weight"])
    daily["rate_anchor"] = blend_anchor(own_rate, outside_rate, params["peer_weight"])
    anchor = daily["rate_anchor"]
    daily["pe_before_premium"] = (anchor + daily["held_pe_adjust"] + daily["held_quality_pe"]).clip(
        lower=params["pe_floor"], upper=params["pe_cap"])
    daily["acceptable_pe"] = (daily["pe_before_premium"] * (1 + daily["premium_pct"] / 100)).clip(lower=0)
    # "Now": the same chain with no forward assumption, i.e. today's policy rate, no register items and the clean
    # trailing EPS with no growth or margin change. Next to the model price it shows what the forward inputs add.
    own_now = rate_adjusted_pe(daily["own_anchor"], daily["yield_gap"] - daily["forward_gap"], passed)
    daily["now_anchor"] = blend_anchor(own_now, outside_now, params["peer_weight"])
    daily["now_pe"] = (daily["now_anchor"] + daily["held_quality_pe"]).clip(lower=params["pe_floor"], upper=params["pe_cap"])

    # Product and market check.
    positive_eps = daily["held_eps"].where(daily["held_eps"] > 0)
    daily["model_price"] = positive_eps * daily["acceptable_pe"]
    daily["market_pe"] = daily["close"] / positive_eps
    daily["residual_pct"] = (daily["model_price"] / daily["close"] - 1) * 100
    daily["now_price"] = daily["held_clean_eps"].where(daily["held_clean_eps"] > 0) * daily["now_pe"]
    daily["now_residual_pct"] = (daily["now_price"] / daily["close"] - 1) * 100
    # What the market pays on top of the model before your premium: the part no fundamental here explains.
    before = daily["held_eps"].where(daily["held_eps"] > 0) * daily["pe_before_premium"]
    daily["implied_premium_pct"] = (daily["close"] / before - 1) * 100
    if start is not None:
        daily = daily.loc[pd.Timestamp(start):]
        # Keep the update in force on the first day shown.
        before = quarterly.loc[:pd.Timestamp(start)]
        quarterly = quarterly.loc[before.index[-1]:] if len(before) else quarterly
    return {"quarterly": quarterly, "daily": daily, "items": routed, "params": params, "forward_from": last_release,
            "peer_pe": peers.reindex(daily.index)}


def _market_yield_on(market_yield, days):
    """The market's N-year yield on each of ``days`` (held up to 10 days), or None when there is none."""

    if market_yield is None or not len(days):
        return None
    market = pd.to_numeric(pd.Series(market_yield, dtype=float), errors="coerce").dropna().sort_index()
    if market.empty:
        return None
    market = market.loc[~market.index.duplicated(keep="last")]
    return market.reindex(market.index.union(days)).ffill(limit=10).reindex(days)


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

    events, points = case_effects(routed, state["on"], params)
    point = scenario_point(state["ttm_eps"], state["revenue_yoy"], state["net_margin"], params, events,
                           state.get("forward", True), state.get("eps_spread", np.nan))
    eps = point["risk_adjusted_eps"]
    if not np.isfinite(eps) or not np.isfinite(state["anchor"]):
        return np.nan
    if eps <= 0:
        return 0.0  # no earnings, no price: lets the solver see the sign change at the edge of a dial's range
    pe = float(np.clip(state["anchor"] + points + point["quality_pe"], params["pe_floor"], params["pe_cap"]))
    return eps * pe * (1 + state.get("premium", 0.0) / 100)


def _day_state(result, values, on):
    """One day's held inputs for ``_point_price``, plus what the rate model needs to redo the anchor."""

    day = result["daily"].loc[:on].iloc[-1]
    anchor = day["rate_anchor"] if "rate_anchor" in day else day["anchor"]
    return {"on": on, "ttm_eps": values.get("ttm_eps", np.nan), "revenue_yoy": values.get("revenue_yoy", np.nan),
            "net_margin": values.get("net_margin", np.nan), "anchor": anchor,
            "eps_spread": values.get("eps_spread", np.nan) if result["params"]["manual_eps"] <= 0 else np.nan,
            "own_anchor": day.get("own_anchor", np.nan),
            "peer_anchor": day.get("outside_anchor", day.get("peer_anchor", np.nan)),
            "held_cagr": day.get("pe_growth", day.get("held_cagr", np.nan)),
            "growth_half_life": day.get("pe_half_life", day.get("growth_half_life", np.nan)),
            "premium": float(day.get("premium_pct", 0.0)) if np.isfinite(day.get("premium_pct", np.nan)) else 0.0,
            "pe_erp": day.get("pe_erp", np.nan),
            "policy_rate": day.get("policy_rate", np.nan), "yield_ref": day.get("yield_ref", np.nan),
            "now_yield": day.get("now_yield", np.nan),
            "forward": _is_forward(result, on)}


def _is_forward(result, on):
    """True on and after the latest release, where your forward growth and margin apply."""

    start = result.get("forward_from")
    return start is None or pd.Timestamp(on) >= start


def has_rates(result):
    """True when the run had a policy rate on its last day, so the rate model is live."""

    daily = result["daily"]
    return "policy_rate" in daily and len(daily) > 0 and bool(np.isfinite(daily["policy_rate"].iloc[-1]))


def anchor_at_rate(state, params, rate):
    """The day's P/E anchor (after the rate model) if the policy rate were ``rate`` instead."""

    if not np.isfinite(state.get("policy_rate", np.nan)):
        return state["anchor"]
    expected = model_yield(rate, params)
    actual = model_yield(state["policy_rate"], params)
    if np.isfinite(state.get("now_yield", np.nan)):
        # The day's yield (the market's, when that is the source) moved by what the other rate would change.
        expected, actual = state["now_yield"] + expected - actual, state["now_yield"]
    own_gap = expected - state["yield_ref"] if np.isfinite(state["yield_ref"]) else 0.0
    passed = params["rate_pass_through"]
    own = rate_adjusted_pe(state["own_anchor"], own_gap, passed)
    if params["anchor_source"] == "Fundamentals" and np.isfinite(state.get("growth_half_life", np.nan)):
        growth = state["held_cagr"] if np.isfinite(state.get("held_cagr", np.nan)) else params["fallback_yoy"]
        peer = _fundamental_series(pd.Series([growth]), [state["growth_half_life"]], [expected], params,
                                   [state.get("pe_erp", params["equity_premium"])]).iloc[0]
    else:
        peer = rate_adjusted_pe(state["peer_anchor"], expected - actual, passed)
    return blend_anchor(own, peer, params["peer_weight"])


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


DIAL_COLUMNS = ["Dial", "Block", "Current", "Needed", "Change", "Unit", "Read"]


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
    state = _day_state(result, values, on)
    anchor = state["anchor"]
    target = values["close"]
    rows = []

    def add(name, block, now, low, high, plausible, unit, price_at):
        needed = _solve(price_at, low, high, target)
        rows.append({"Dial": name, "Block": block, "Current": now, "Needed": needed if needed is not None else np.nan,
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
        lambda shift: _point_price(state, _shifted_margin_params(state, params, shift), routed))
    add("EPS haircut", "Earnings quality", params["eps_haircut_pct"], 0, 90, (0, 30), "%",
        lambda haircut: _point_price(state, {**params, "eps_haircut_pct": haircut}, routed))
    events, points = case_effects(routed, on, params)
    points -= abs(float(params["quality_pe_pts"]))
    add("Acceptable P/E", "Multiple builder", float(np.clip(anchor + points, params["pe_floor"], params["pe_cap"])),
        1, 100, (params["pe_floor"], params["pe_cap"]), "x",
        lambda multiple: _point_price(state, {**params, "pe_floor": multiple, "pe_cap": multiple}, routed))

    if has_rates(result):
        used = result["daily"].loc[:on, "forward_rate"].iloc[-1]
        add("Policy rate", "Rate model", used, -1, 20, (max(0.0, used - 1.5), used + 1.5), "%",
            lambda rate: _point_price({**state, "anchor": anchor_at_rate(state, params, rate)}, params, routed))

    model_above = values["model_price"] > target
    for position, row in routed.iterrows():
        parked = row["Gate"] != MODELLED
        label = f"{row['Item']} (parked)" if parked else row["Item"]
        # A risk can only pull the model price down, a product can only lift it.
        helps = (row["Type"] == RISK) == model_above if parked else True
        if row["Route"] == TO_SCENARIOS:
            now = 0.0 if parked else row["Probability %"]

            def with_probability(probability, position=position):
                return _point_price(state, params, _reroute(routed, position, params, probability=probability))

            if helps:
                add(f"{label}: probability", TO_SCENARIOS, now, 0, 100, (max(0, now - 30), min(100, now + 30)), "%",
                    with_probability)
            else:
                rows.append({"Dial": f"{label}: probability", "Block": TO_SCENARIOS, "Current": now, "Needed": np.nan,
                             "Change": np.nan, "Unit": "%", "Read": "Wrong direction"})
        else:
            now = 0.0 if parked else abs(row["Effect"])

            def with_points(size, position=position):
                return _point_price(state, params, _reroute(routed, position, params, effect=size))

            if helps:
                add(f"{label}: P/E points", TO_MULTIPLE, now, 0, 30, (0, now + 3), "pts", with_points)
            else:
                rows.append({"Dial": f"{label}: P/E points", "Block": TO_MULTIPLE, "Current": now, "Needed": np.nan,
                             "Change": np.nan, "Unit": "pts", "Read": "Wrong direction"})
    return pd.DataFrame(rows, columns=DIAL_COLUMNS)


def _reroute(routed, position, params, probability=None, effect=None):
    """The routed items with one item modelled from the day it is tried, at a trial probability or P/E size.

    ``probability``: the item's forward probability (its bull and bear adjustments still apply; an item without an
    EPS impact is tried at 10%). ``effect``: the item's P/E points at the forward probability, the cases scaled alike.
    """

    trial = routed.copy()
    row = trial.loc[[position]].copy()
    row["Status"], row["Since"] = MODELLED, ""
    if probability is not None:
        row["Probability %"] = probability
        if row.at[position, "Route"] == TO_SCENARIOS and not pd.notna(row.at[position, "EPS impact %"]):
            row["EPS impact %"] = 10.0
    redone = route_items(row, params).iloc[0]
    columns = ["Effect"] + [f"Effect {name}" for name in SCENARIOS]
    values = {column: redone[column] for column in columns}
    if effect is not None:
        sign = -1.0 if row.at[position, "Type"] == RISK else 1.0
        base = values["Effect"]
        scale = {column: (values[column] / base if np.isfinite(base) and base != 0 and np.isfinite(values[column]) else 1.0)
                 for column in columns}
        values = {column: sign * abs(effect) * scale[column] for column in columns}
    for column, value in values.items():
        trial.at[position, column] = value
    trial.at[position, "Gate"], trial.at[position, "Since"] = MODELLED, ""
    return trial


def market_implied_probabilities(result, on=None):
    """For each item with a probability to set: the probability that alone makes the model price equal the close.

    Everything else stays as set (the item's own bull and bear adjustments included). NaN when no probability from 0
    to 100% does it alone, or when the item has no probability (a general item, or no EPS impact or P/E points).
    Indexed like the register.
    """

    values = latest_values(result, on)
    routed = result["items"]
    out = pd.Series(np.nan, index=routed.index, dtype=float)
    if not values or not np.isfinite(values.get("close", np.nan)) or not np.isfinite(values.get("model_price", np.nan)):
        return out
    params = result["params"]
    on = pd.Timestamp(on) if on is not None else result["daily"].index[-1]
    state = _day_state(result, values, on)
    for position, row in routed.iterrows():
        sized = (row["Route"] == TO_SCENARIOS
                 or (row["Kind"] == EVENT and pd.notna(row["P/E points"])))
        if not sized:
            continue
        needed = _solve(lambda p, position=position: _point_price(state, params, _reroute(routed, position, params, probability=p)),
                        0, 100, values["close"])
        out[position] = needed if needed is not None else np.nan
    return out


CASE_ROWS = {"Revenue growth %": ("revenue_yoy", "growth_base", "forward_growth", "bull_growth_gap", "bear_growth_gap"),
             "Net margin %": ("net_margin", "margin_base", "forward_margin", "bull_margin_pts", "bear_margin_pts")}
CASE_COLUMNS = ["Metric", "Current", "Forward", "Bull adj", "Bear adj", "Current from", "Forward from"]


def case_table(result, params=None):
    """Growth and margin now and forward, with the bull and bear adjustments, as the Forward table shows them.

    Current: the latest reported trailing value, or the fallback the model uses without one. Forward: yours if set,
    else the auto rule. Bull and bear adjustments
    are points added to the forward value (the bear growth gap shows as a negative number).
    """

    params = model_params(params if params is not None else result["params"])
    values = latest_values(result)
    yoy, growth, margin, base_margin = base_growth_and_margin(values.get("revenue_yoy", np.nan),
                                                              values.get("net_margin", np.nan), params, forward=True)
    reported = {"Revenue growth %": values.get("revenue_yoy", np.nan), "Net margin %": values.get("net_margin", np.nan)}
    # Without a reported value the model runs on the fallback (Scenario engine settings): show that, and say so.
    current = {"Revenue growth %": yoy, "Net margin %": margin}
    forward = {"Revenue growth %": growth, "Net margin %": base_margin}
    rows = []
    for metric, (_, _, key, bull, bear) in CASE_ROWS.items():
        bear_value = -abs(params[bear]) if key == "forward_growth" else params[bear]
        rows.append({"Metric": metric, "Current": current[metric], "Forward": forward[metric],
                     "Bull adj": params[bull], "Bear adj": bear_value,
                     "Current from": "reported" if np.isfinite(reported[metric]) and (
                         metric == "Revenue growth %" or reported[metric] > 0) else "fallback",
                     "Forward from": ("analysts" if params.get("analyst_base") else
                                      "you" if np.isfinite(params.get(key, np.nan)) else "auto rule")})
    return pd.DataFrame(rows, columns=CASE_COLUMNS)


def case_table_changes(before, after, params):
    """Parameter updates from an edited case table (``before`` as ``case_table`` drew it, ``after`` as edited).

    A forward value left as drawn keeps its source (auto stays auto); a changed one becomes yours; a cleared one goes
    back to the auto rule. Returns {parameter: value}.
    """

    updates = {}
    before, after = before.set_index("Metric"), pd.DataFrame(after).set_index("Metric")
    for metric, (_, _, key, bull, bear) in CASE_ROWS.items():
        if metric not in after.index:
            continue
        old, new = before.loc[metric], after.loc[metric]
        if pd.isna(new["Forward"]):
            if np.isfinite(params.get(key, np.nan)):
                updates[key] = np.nan
        elif pd.isna(old["Forward"]) or abs(float(new["Forward"]) - float(old["Forward"])) > 1e-9:
            updates[key] = float(new["Forward"])
        for column, name in [("Bull adj", bull), ("Bear adj", bear)]:
            value = new[column]
            if pd.notna(value) and abs(float(value) - float(old[column])) > 1e-9:
                updates[name] = abs(float(value)) if key == "forward_growth" else float(value)
    return updates


PE_ROWS = {"Growth for the P/E %": ("rule_growth", "forward_pe_growth"),
           "Growth half-life, years": ("growth_half_life", "forward_half_life"),
           "Equity risk premium %": (None, "forward_equity_premium"),
           "Market premium %": ("implied_premium_pct", "market_premium")}
PE_COLUMNS = ["Metric", "Current", "Forward", "Forward from"]


def pe_table(result, params=None):
    """The fundamental P/E's inputs now and forward, and the market premium kept apart from them.

    Current: from the rules on data known today (growth from the history and the latest year, half-life from the
    gross margin and growth steadiness, risk premium from beta) and, for the premium, what the market pays over the model before your premium. Forward: yours,
    used from the latest release. Only the premium row when the outside view is the peers.
    """

    params = model_params(params if params is not None else result["params"])
    values = latest_values(result)
    growth = values.get("rule_growth", np.nan)
    growth = growth if np.isfinite(growth) else params["fallback_yoy"]
    erp = values.get("rule_erp", np.nan)
    current = {"Growth for the P/E %": growth, "Growth half-life, years": values.get("growth_half_life", np.nan),
               "Equity risk premium %": erp if np.isfinite(erp) else params["equity_premium"],
               "Market premium %": values.get("implied_premium_pct", np.nan)}
    rows = []
    for metric, (_, key) in PE_ROWS.items():
        if params["anchor_source"] != "Fundamentals" and metric != "Market premium %":
            continue
        mine = params.get(key, np.nan)
        if metric == "Market premium %":
            rows.append({"Metric": metric, "Current": current[metric], "Forward": float(mine),
                         "Forward from": "you" if mine else "none"})
        else:
            rows.append({"Metric": metric, "Current": current[metric],
                         "Forward": float(mine) if np.isfinite(mine) else current[metric],
                         "Forward from": "you" if np.isfinite(mine) else "history"})
    return pd.DataFrame(rows, columns=PE_COLUMNS)


def pe_table_changes(before, after, params):
    """Parameter updates from an edited P/E table: a changed Forward becomes yours, a cleared one goes back to the
    history (the premium to 0); a Forward left as drawn keeps its source."""

    updates = {}
    before, after = before.set_index("Metric"), pd.DataFrame(after).set_index("Metric")
    for metric, (_, key) in PE_ROWS.items():
        if metric not in after.index or metric not in before.index:
            continue
        old, new = before.loc[metric, "Forward"], after.loc[metric, "Forward"]
        if pd.isna(new):
            reset = 0.0 if key == "market_premium" else np.nan
            current = params.get(key, np.nan)
            if (key == "market_premium" and current) or (key != "market_premium" and np.isfinite(current)):
                updates[key] = reset
        elif pd.isna(old) or abs(float(new) - float(old)) > 1e-9:
            updates[key] = float(new)
    return updates


def _shifted_growth_price(state, params, routed, shift):
    """Model price with ``shift`` points added to the growth of every case."""

    _, base, _, _ = base_growth_and_margin(state["revenue_yoy"], state["net_margin"], params, state.get("forward", True))
    base += shift
    fixed = {**params, "growth_follow": 1.0, "growth_floor": base, "growth_cap": base, "fallback_yoy": base,
             "forward_growth": base}
    state = {**state, "revenue_yoy": base}
    return _point_price(state, fixed, routed)


def _shifted_margin_params(state, params, shift):
    """Parameters with ``shift`` points added to the net margin of every case (the base, which the cases follow)."""

    if state.get("forward", True) and np.isfinite(params.get("forward_margin", np.nan)):
        return {**params, "forward_margin": params["forward_margin"] + shift}
    return {**params, "base_margin_pts": params["base_margin_pts"] + shift}


def sensitivities(result, on=None):
    """How far the model price moves for one step of each main dial (10 points of bear weight, 1x of P/E ...)."""

    dials = implied_dials(result, on)
    values = latest_values(result, on)
    if dials.empty or not values:
        return {}
    params, routed = result["params"], result["items"]
    on = pd.Timestamp(on) if on is not None else result["daily"].index[-1]
    state = _day_state(result, values, on)
    base = _point_price(state, params, routed)
    weights = scenario_weights(params) * 100
    rest = weights[1] + weights[2]
    scale = (100 - weights[0] - 10) / rest if rest > 0 else 0
    more_bear = {**params, "weight_bear": weights[0] + 10, "weight_base": weights[1] * scale, "weight_bull": weights[2] * scale}
    steps = {
        "10 points more bear weight": _point_price(state, more_bear, routed) - base,
        "1x more P/E": values.get("held_eps", np.nan),
        "1 point more growth": _shifted_growth_price(state, params, routed, 1.0) - base,
        "1 point more net margin": _point_price(state, _shifted_margin_params(state, params, 1.0), routed) - base,
    }
    if has_rates(result):
        used = result["daily"].loc[:on, "forward_rate"].iloc[-1]
        higher = {**state, "anchor": anchor_at_rate(state, params, used + 0.5)}
        steps["0.5 point higher policy rate"] = _point_price(higher, params, routed) - base
    return steps


RATE_CASE_COLUMNS = ["Case", "Policy rate %", "Probability %"]


def default_rate_cases(rate):
    """Three cases around a policy rate: hold, +0.25 and +0.50 points."""

    return pd.DataFrame([["Hold", rate, 50.0], ["+0.25", rate + 0.25, 30.0], ["+0.50", rate + 0.5, 20.0]],
                        columns=RATE_CASE_COLUMNS)


def rate_cases(result, cases, on=None):
    """The model price in each policy rate case and their probability-weighted expected value.

    ``cases`` has the columns Case, Policy rate % and Probability % (rescaled to add up to 100). Each case reruns the
    rate model and the multiple builder for the day with that rate on it; everything else is held. Returns
    ``(table, expected)``: the table adds the case's N-year yield, anchor, model price and change vs the close;
    ``expected`` is the weighted model price (NaN when there is nothing to weigh).
    """

    columns = RATE_CASE_COLUMNS + ["N-year yield %", "P/E anchor", "Model price", "vs close %"]
    values = latest_values(result, on)
    frame = pd.DataFrame(cases, columns=RATE_CASE_COLUMNS) if cases is not None else pd.DataFrame(columns=RATE_CASE_COLUMNS)
    frame = frame.dropna(subset=["Policy rate %"]).reset_index(drop=True)
    if not values or frame.empty or not has_rates(result):
        return pd.DataFrame(columns=columns), np.nan
    params, routed = result["params"], result["items"]
    on = pd.Timestamp(on) if on is not None else result["daily"].index[-1]
    state = _day_state(result, values, on)
    rows = []
    for row in frame.to_dict("records"):
        rate = float(row["Policy rate %"])
        anchor = anchor_at_rate(state, params, rate)
        price = _point_price({**state, "anchor": anchor}, params, routed)
        rows.append({**row, "N-year yield %": float(model_yield(rate, params)), "P/E anchor": anchor,
                     "Model price": price, "vs close %": (price / values["close"] - 1) * 100})
    table = pd.DataFrame(rows, columns=columns)
    weights = pd.to_numeric(table["Probability %"], errors="coerce").fillna(0).clip(lower=0)
    usable = weights.where(table["Model price"].notna(), 0)
    expected = float((usable * table["Model price"].fillna(0)).sum() / usable.sum()) if usable.sum() > 0 else np.nan
    return table, expected


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
        for column in ["Type", "Kind", "Timing", "Probability %", "Bull prob adj", "Bear prob adj", "EPS impact %",
                       "P/E points", "Status", "Since"]:
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


def now_and_forward(result, on=None):
    """Each step of the chain now (today's data, no forward assumption) and forward (with them), side by side.

    Rows: policy rate, N-year yield, EPS, P/E, model price and its gap to the close. Columns: Now, Forward, Change.
    """

    values = latest_values(result, on)
    columns = ["Step", "Last quarter", "Forward", "Change"]
    if not values:
        return pd.DataFrame(columns=columns)
    rows = [("Policy rate %", "policy_rate", "forward_rate"), ("N-year yield %", "now_yield", "model_yield"),
            ("EPS", "held_clean_eps", "held_eps"), ("P/E", "now_pe", "acceptable_pe"),
            ("Model price", "now_price", "model_price"), ("vs close %", "now_residual_pct", "residual_pct")]
    table = [{"Step": label, "Last quarter": values.get(now, np.nan), "Forward": values.get(forward, np.nan)}
             for label, now, forward in rows]
    frame = pd.DataFrame(table, columns=columns[:3])
    frame = frame[frame[["Last quarter", "Forward"]].notna().any(axis=1)].reset_index(drop=True)
    frame["Change"] = frame["Forward"] - frame["Last quarter"]
    return frame
