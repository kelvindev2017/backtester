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
    "quality_pe_pts": 0.0,       # P/E points taken off the risk-adjusted P/E for weak earnings quality
    # Scenario engine
    "analyst_base": False,       # set by run_model: True when the latest base case is the analysts' EPS
    "eps_model": "Build-up",     # base EPS: Build-up (revenue -> EBIT -> interest, tax, preferred -> per share after
                                 # stock-pay dilution and buybacks) or Scaled EPS (reported EPS x growth x margin ratio)
    "fallback_tax_rate": 21.0,   # Build-up: tax rate % when the pre-tax income is not positive or no tax is reported
    "share_creep_cap": 5.0,
    "exclude_one_offs": True,    # Build-up: leave one-time costs (acquired intangibles' amortization, impairments,
                                 # acquired R&D, deal costs) out of operating income
    "one_off_floor": 0.8,        # Build-up: the EBIT margin used is at least this share of the margin the reported EPS
                                 # (after the haircut) implies, so a one-off accounting hit (acquisition amortization,
                                 # a write-down) cannot sink it; 0 = the GAAP margin as it is      # Build-up: highest yearly share creep % from stock-based pay (more is an acquisition)
    "base_eps_source": "Own rule",  # latest quarter's base EPS: Analysts (next 4 quarters' estimates) or Own rule
    "analyst_share": 1.0,        # with Analysts: their share of the latest base growth; the rest is the EPS trend rule
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
    "growth_history_quarters": 12,  # fundamental P/E: revenue growth = the yearly growth (CAGR) over this many quarters
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
    "pe_growth_latest": "Latest quarter",  # latest growth: Latest quarter vs the same quarter a year ago, or 4 quarters
    "use_guidance": False,           # company guidance, when a source gives some, replaces the reported growth
    "quarter_base_effect": True,     # latest-quarter growth vs two years ago when the year-ago quarter had dipped
    "pe_growth_latest_share": 0.5,   # growth for the P/E = this share of the latest year's growth + the rest of the history's
    "half_life_from_steadiness": True,  # steady growth earns a longer half-life than a low gross margin gives
    "steady_spread": 2.0,            # ... yearly growth varying by this many points or less earns steady_half_life
    "choppy_spread": 15.0,           # ... and by this many or more earns nothing beyond the gross-margin rule
    "steady_half_life": 6.0,
    "erp_from_beta": True,           # equity risk premium x the stock's beta to the market (3 years, weekly)
    "market_implied_erp": True,      # use the risk premium the market usually prices this stock at (below)
    "erp_lookback_years": 2,         # ... the usual premium is the median over this many years of month-ends
    "implied_erp_floor": -2.0,       # ... the implied premium is solved within this floor
    "implied_erp_cap": 20.0,         # ... and this cap, in %
    "roll_forward_eps": True,
    "past_base": "EPS trend",        # base growth before the latest release: EPS trend (history rule) or Revenue rule        # next year's EPS rolls forward between releases (the next 12 months, each day)
    "erp_flag_level": 2.5,           # ... the market's premium is used even when negative; below this % it is flagged
    # Through-cycle EPS: a deep-cycle company is valued on its average EPS over a whole cycle, not one year's.
    "cycle_eps": "Auto",             # Auto (when the EPS fell deeply and has no strong trend), On or Off
    "cycle_years": 7,                # ... the average of this many yearly EPS figures (the past years + next year)
    "cycle_fall": 50.0,              # ... Auto: the EPS fell at least this % below an earlier high in those years
    "cycle_trend_cap": 10.0,         # ... Auto: and its trend growth is below this % a year
    "cycle_revenue_fall": 15.0,      # ... Auto: and revenue fell at least this % from a high (a cycle, not a margin squeeze)
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
    "pe_cap": 300.0,                # room for a fast grower; the market premium floor keeps it from copying a bubble
    "peer_pe_max": 100.0,           # a peer's P/E above this rests on earnings near zero: it sits out the median
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

# Quarterly columns the EPS build-up reads (besides Revenue and Shares), as the SEC loader names them.
BUILD_UP_COLUMNS = ["Operating_Income", "Pretax_Income", "Income_Tax", "Interest_Expense", "Interest_Income",
                    "Net_Interest_Nonoperating", "Preferred_Dividends", "Stock_Comp", "Buybacks", "One_Off_Costs"]
# The build-up's held inputs at each release (quarterly signals), in model units (revenue and money in dollars).
BUILD_UP_INPUTS = ["bu_revenue", "bu_net_interest", "bu_tax_rate", "bu_preferred", "bu_shares", "bu_stock_comp",
                   "bu_buybacks", "bu_price", "bu_creep_pct"]

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
    "revenue_yoy": ("quarterly", "Revenue per share growth, year on year", "%", "percent"),
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
    "next_year_eps": ("quarterly", "Next year's EPS (risk-adjusted)", "", "eps"),
    "cycle_eps": ("quarterly", "Through-cycle EPS", "", "eps"),
    "cycle_trend": ("quarterly", "Through-cycle EPS trend", "%", "percent"),
    "cycle_on": ("quarterly", "Through-cycle EPS on (1 = yes)", "", "count"),
    "cycle_past_sum": ("quarterly", "Past years' EPS, summed", "", "eps"),
    "cycle_years_used": ("quarterly", "Years in the through-cycle mean", "", "count"),
    "ebit_margin": ("quarterly", "EBIT (operating) margin, last 4 quarters", "%", "share"),
    "revenue_total_yoy": ("quarterly", "Revenue growth, year on year (total, not per share)", "%", "percent"),
    "bu_revenue": ("quarterly", "Build-up: revenue, last 4 quarters", "", "amount"),
    "bu_net_interest": ("quarterly", "Build-up: net interest expense, last 4 quarters", "", "amount"),
    "bu_tax_rate": ("quarterly", "Build-up: tax rate", "%", "share"),
    "bu_preferred": ("quarterly", "Build-up: preferred dividends, last 4 quarters", "", "amount"),
    "bu_shares": ("quarterly", "Build-up: diluted shares, latest quarter", "", "count"),
    "bu_stock_comp": ("quarterly", "Build-up: stock-based pay, last 4 quarters", "", "amount"),
    "bu_buybacks": ("quarterly", "Build-up: buybacks, last 4 quarters", "", "amount"),
    "bu_price": ("quarterly", "Build-up: share price for dilution and buybacks (close on the release)", "", "price"),
    "ebit_margin_gaap": ("quarterly", "EBIT margin used before the floor, last 4 quarters", "%", "share"),
    "bu_one_offs": ("quarterly", "Build-up: one-time costs left out of EBIT, last 4 quarters", "", "amount"),
    "bu_margin_floor": ("quarterly", "Build-up: EBIT margin floor from the reported EPS", "%", "share"),
    "bu_creep_pct": ("quarterly", "Build-up: yearly share creep from stock-based pay", "%", "percent"),
    "bu_on": ("quarterly", "EPS from the build-up (1) or scaled reported EPS (0)", "", "count"),
    "past_growth": ("quarterly", "Base EPS growth before the latest release (EPS trend)", "%", "percent"),
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
    "held_eps": ("daily", "Risk-adjusted EPS (rolled forward daily)", "", "eps"),
    "eps_roll": ("daily", "Roll-forward factor on next year's EPS", "", "count"),
    "roll_years": ("daily", "Years since the release (roll-forward)", "", "count"),
    "held_pe_adjust": ("daily", "P/E points (held)", "pts", "points"),
    "held_quality_pe": ("daily", "Earnings quality P/E points (held)", "pts", "points"),
    "own_anchor": ("daily", "Own median P/E", "x", "multiple"),
    "peer_anchor": ("daily", "Peers' median P/E", "x", "multiple"),
    "held_cagr": ("daily", "Revenue per share growth a year over the growth history (held)", "%", "percent"),
    "held_guided_growth": ("daily", "Revenue per share growth the company guided (in force until the next release)", "%",
                           "percent"),
    "held_quarter_yoy": ("daily", "Revenue per share growth, latest quarter vs the same quarter a year ago (held)", "%",
                         "percent"),
    "held_gross_margin": ("daily", "Gross margin, last 4 quarters (held)", "%", "share"),
    "growth_half_life": ("daily", "Half-life of the extra growth, years (gross margin and steadiness)", "", "count"),
    "held_yoy": ("daily", "Revenue per share growth, latest year (held)", "%", "percent"),
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
    "acceptable_pe": ("daily", "Risk-adjusted P/E", "x", "multiple"),
    "margin_base": ("quarterly", "Base (forward) net margin", "%", "share"),
    "held_clean_eps": ("daily", "Clean trailing EPS (held)", "", "eps"),
    "now_yield": ("daily", "Model N-year yield at today's rate", "%", "rate"),
    "now_anchor": ("daily", "P/E anchor at today's rate", "x", "multiple"),
    "now_pe": ("daily", "Model P/E (today's rate, no register items)", "x", "multiple"),
    "now_price": ("daily", "Model fair value (no forward assumptions)", "", "price"),
    "now_residual_pct": ("daily", "Model fair value, vs close", "%", "percent"),
    "model_price": ("daily", "Risk-adjusted fair value per share", "", "price"),
    "market_pe": ("daily", "Market price / risk-adjusted EPS", "x", "multiple"),
    "residual_pct": ("daily", "Risk-adjusted fair value vs close", "%", "percent"),
}

# Each block: its sample rate, what it reads, what it writes and the parameters it owns (key -> label).
BLOCKS = {
    "income_map": dict(
        title="Income and cost map", rate="quarterly", role="Logic: parsing only",
        inputs=["revenue", "net_income"], outputs=["ttm_revenue", "revenue_yoy", "net_margin", "gross_margin",
                                                     "revenue_cagr"],
        params={"fallback_yoy": "Growth % when there is no revenue history",
                "fallback_margin": "Net margin % when there is no net income",
                "fallback_tax_rate": "Build-up: tax rate % when pre-tax income is not positive",
                "share_creep_cap": "Build-up: highest yearly share creep %",
                "exclude_one_offs": "Build-up: leave one-time costs out of EBIT",
                "one_off_floor": "Build-up: lowest EBIT margin, share of the reported EPS's (0 = off)"}),
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
        params={"eps_model": "Base EPS: Build-up (revenue to EPS) or Scaled EPS",
                "case_spread": "Bear and bull from: EPS volatility or Fixed gaps",
                "spread_k": "EPS volatility: k (cases at exp(-/+ k x s))",
                "spread_own_share": "EPS volatility: share of the own volatility in s",
                "spread_typical": "EPS volatility: typical volatility",
                "analyst_share": "Analysts' share of the latest base growth (rest: EPS trend rule)",
                "forward_growth": "Forward revenue growth % (blank = auto rule)",
                "forward_margin": "Forward net margin % (blank = auto rule)",
                "bear_growth_gap": "Bear: growth points below forward", "bull_growth_gap": "Bull: growth points above forward",
                "bear_margin_pts": "Bear: margin points vs forward", "bull_margin_pts": "Bull: margin points vs forward",
                "growth_follow": "Auto rule: base growth as a share of trailing growth",
                "growth_floor": "Auto rule: base growth floor %", "growth_cap": "Auto rule: base growth cap %",
                "base_margin_pts": "Auto rule: base margin change, points"}),
    "risk_weighting": dict(
        title="Risk weighting", rate="quarterly", role="Param: scenario weights",
        inputs=["eps_bear", "eps_base", "eps_bull", "ttm_eps"],
        outputs=["risk_adjusted_eps", "next_year_eps", "cycle_eps", "cycle_trend", "cycle_on", "cycle_past_sum", "cycle_years_used"],
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
                "growth_history_quarters": "Fundamental P/E: growth history, quarters",
                "quarter_base_effect": "Latest quarter: skip a dipped year-ago quarter",
                "use_guidance": "Use company guidance",
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
        title="Product", rate="daily", role="EPS x risk-adjusted P/E",
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
        "note": ("ASMPT preset from the worked example. Results come from the Hong Kong listing (0522.HK), converted "
                 "to US$ per ADR (3 shares) at each day's rate; the doc's trailing adjusted EPS was HK$4.28 a share "
                 "(US$1.64 per ADR). It is valued on through-cycle EPS, so the cyclicality item is parked; the event risks "
                 "are parked too: the bear case already carries them."),
        "params": {
            "eps_haircut_pct": 11.0, "growth_follow": 0.7, "fallback_yoy": 40.0,
            "fallback_margin": 9.8, "bear_growth_gap": 28.0, "bull_growth_gap": 16.0, "bear_margin_pts": -5.0,
            "base_margin_pts": 4.0, "bull_margin_pts": 3.0, "peer_weight": 1.0, "case_spread": "Fixed gaps",
        },
        "register": [
            ["Cyclicality: 2027 is probably near peak earnings", RISK, GENERAL, BEYOND, np.nan, np.nan, 3.0, PARKED,
             "", "cycle, downturn, peak", "ASMPT doc; parked: the through-cycle EPS already averages the cycle"],
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
    overrides = dict(overrides or {})
    if "growth_history_years" in overrides and "growth_history_quarters" not in overrides:
        overrides["growth_history_quarters"] = 4 * int(overrides.pop("growth_history_years"))  # settings saved before
    for key, value in overrides.items():
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
    average over the cases, because there is one risk-adjusted P/E.
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


def through_cycle_eps(past_ttm, forward_eps, clean_share, params, was_on=False, past_revenue=None):
    """(through-cycle EPS, its trend growth %, on) for one date.

    The yearly figures are the trailing-4-quarter EPS at the latest release and every 4th release before it (the
    past ``cycle_years`` - 1 years, at least 3, after the haircut) and next year's risk-adjusted EPS. The through-cycle EPS is
    their mean; the trend is the log-linear growth through them (0 when any is not positive). Auto turns it on when
    a figure fell at least ``cycle_fall`` % below an earlier high and the trend is below ``cycle_trend_cap``: a
    deep cycle, where one year's EPS says little about the next five. Auto also needs trailing revenue
    (``past_revenue``) to have fallen ``cycle_revenue_fall`` % below a high over those years: EPS falling on flat revenue
    is a margin squeeze, not a cycle. Without revenue that test is skipped. Off, or too little history: (NaN, NaN, False).
    """

    mode = str(params.get("cycle_eps", "Off"))
    years = int(params.get("cycle_years", 7))
    none = (np.nan, np.nan, False)
    if mode == "Off" or years < 3 or forward_eps is None or not np.isfinite(forward_eps):
        return none
    past = pd.to_numeric(pd.Series(past_ttm), errors="coerce").dropna()
    yearly = past.iloc[::-1].iloc[::4].iloc[:years - 1].iloc[::-1] * clean_share
    if len(yearly) < min(3, years - 1):
        return none
    values = np.append(yearly.to_numpy(dtype=float), float(forward_eps))
    mean = float(values.mean())
    if not np.isfinite(mean) or mean <= 0:
        return none
    trend = float(np.expm1(np.polyfit(np.arange(len(values)), np.log(values), 1)[0]) * 100) if (values > 0).all() else 0.0
    if mode == "On":
        return mean, trend, True
    high = np.maximum.accumulate(values)
    fell = bool((values / high).min() <= 1 - float(params["cycle_fall"]) / 100) if (high > 0).all() else True
    # Hysteresis: once on, it stays on until the trend clears the cap by half again, so a stock near the line does
    # not flip between bases (and P/Es) from one release to the next.
    cap = float(params["cycle_trend_cap"]) * (1.5 if was_on else 1.0)
    revenue_fell = True
    if past_revenue is not None and not was_on:
        revenue = pd.to_numeric(pd.Series(past_revenue), errors="coerce").dropna()
        revenue = revenue[revenue > 0].iloc[-(4 * (years - 1) + 1):]
        if len(revenue) >= 8:
            revenue_fell = bool((revenue / revenue.cummax()).min() <= 1 - float(params.get("cycle_revenue_fall", 15)) / 100)
    on = fell and revenue_fell and trend < cap
    return (mean, trend, True) if on else (mean, trend, False)


def build_up_eps(revenue, growth, ebit_margin, net_interest, tax_rate, preferred, shares, dilution, buybacks, price):
    """Estimated EPS = ([revenue x (1 + g) x EBIT margin - net interest] x (1 - t) - preferred dividends)
    / (shares + SBC dilution - buyback $ / share price).

    ``growth``, ``ebit_margin`` and ``tax_rate`` in %; money in one currency unit; ``dilution`` is next year's new
    shares from stock-based pay and the buyback retires shares at ``price``. NaN when the shares left are not positive.
    """

    ebit = revenue * (1 + growth / 100) * ebit_margin / 100
    earnings = (ebit - net_interest) * (1 - tax_rate / 100) - preferred
    retired = buybacks / price if price > 0 else 0.0
    count = shares + dilution - retired
    return earnings / count if count > 0 else np.nan


def build_up_inputs(values):
    """The build-up's inputs from one release's held values, or None when the essentials (revenue, EBIT margin,
    shares) are missing."""

    inputs = {name: float(values.get(name, np.nan)) for name in BUILD_UP_INPUTS}
    inputs["ebit_margin"] = float(values.get("ebit_margin", np.nan))
    if not (inputs["bu_revenue"] > 0 and inputs["bu_shares"] > 0 and np.isfinite(inputs["ebit_margin"])):
        return None
    for name in ["bu_net_interest", "bu_preferred", "bu_stock_comp", "bu_buybacks", "bu_creep_pct"]:
        if not np.isfinite(inputs[name]):
            inputs[name] = 0.0
    if not np.isfinite(inputs["bu_tax_rate"]):
        inputs["bu_tax_rate"] = np.nan
    if not np.isfinite(inputs["bu_price"]):
        inputs["bu_price"] = 0.0
    return inputs


def _build_up_case(inputs, growth, margin, params):
    tax = inputs["bu_tax_rate"] if np.isfinite(inputs["bu_tax_rate"]) else float(params["fallback_tax_rate"])
    return build_up_eps(inputs["bu_revenue"], growth, margin, inputs["bu_net_interest"], tax, inputs["bu_preferred"],
                        inputs["bu_shares"], inputs["bu_shares"] * inputs["bu_creep_pct"] / 100, inputs["bu_buybacks"],
                        inputs["bu_price"])


def implied_ebit_margin(eps, inputs, params):
    """The EBIT margin % at which the build-up, with no growth and today's shares, gives ``eps``: the margin behind
    the reported EPS. NaN when ``eps`` is not positive."""

    if not (np.isfinite(eps) and eps > 0):
        return np.nan
    tax = inputs["bu_tax_rate"] if np.isfinite(inputs["bu_tax_rate"]) else float(params["fallback_tax_rate"])
    if tax >= 100:
        return np.nan
    earnings = (eps * inputs["bu_shares"] + inputs["bu_preferred"]) / (1 - tax / 100) + inputs["bu_net_interest"]
    return earnings / inputs["bu_revenue"] * 100


def share_creep_pct(shares, shares_year_ago, buybacks, average_price, stock_comp, price, cap):
    """Yearly share creep from stock-based pay, % of today's shares: the historical one (the change in diluted shares
    over the last year plus the shares bought back, at the year's average price) when a year-ago count exists, else
    the stock-based pay over the price. Kept within 0 and ``cap`` (a jump beyond it is an acquisition paid in shares)."""

    if shares > 0 and shares_year_ago > 0 and average_price > 0:
        creep = (shares - shares_year_ago + buybacks / average_price) / shares_year_ago * 100
    elif shares > 0 and price > 0:
        creep = stock_comp / price / shares * 100
    else:
        return 0.0
    return float(np.clip(creep, 0.0, cap)) if np.isfinite(creep) else 0.0


def scenario_point(ttm_eps, revenue_yoy, net_margin, params, event_effect=0.0, forward=False, eps_spread=np.nan,
                   build_up=None):
    """The quarterly chain for one date: earnings quality, scenario engine and risk weighting.

    ``revenue_yoy`` and ``net_margin`` are in %; ``event_effect`` is the fraction of EPS the modelled event items are
    expected to add or remove, one number or one per case. The base case grows the clean EPS with the base revenue
    growth and scales it by the base net margin over the current one. With ``case_spread`` Fixed gaps, bear and bull
    do the same with their own growth and margin (the base plus or minus the gaps); with EPS volatility, they are the
    base EPS times exp(-k s) and exp(+k s), s from ``case_spread`` and ``eps_spread`` (the company's own EPS
    volatility; a forward test on 60 stocks, 2019-2025, set k, the share and the typical volatility). ``forward``: the latest quarter, where your forward growth and margin apply.

    ``build_up`` (from ``build_up_inputs``): each case's EPS comes from the build-up instead (``build_up_eps``), with
    ``revenue_yoy`` the total revenue growth and ``net_margin`` the EBIT margin. Its EPS is after stock-based pay
    already, so the haircut does not apply; the clean EPS is the build-up today (no growth, today's shares).
    """

    clean = ttm_eps * (1 - params["eps_haircut_pct"] / 100)
    if build_up is not None:
        today = dict(build_up, bu_creep_pct=0.0, bu_buybacks=0.0)
        clean = _build_up_case(today, 0.0, build_up["ebit_margin"], params)
    _, base_growth, margin, base_margin = base_growth_and_margin(revenue_yoy, net_margin, params, forward)
    if build_up is not None:
        # The EBIT margin is used as it is, a loss included (the net-margin fallback is for a missing figure).
        margin = build_up["ebit_margin"]
        base_margin = (float(params["forward_margin"]) if forward and np.isfinite(params.get("forward_margin", np.nan))
                       else margin + params["base_margin_pts"])
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
        if build_up is not None:
            case_growth, case_margin_pct = ((base_growth, base_margin) if by_volatility
                                            else (growth[name], case_margin[name]))
            point[f"growth_{key}"] = case_growth
            case_eps = _build_up_case(build_up, case_growth, case_margin_pct, params)
            if by_volatility:
                # The spread scales the base EPS; a loss stays a loss (the spread widens it in the bear case).
                case_eps = case_eps * (factor[name] if case_eps >= 0 else 1 / factor[name])
        elif by_volatility:
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
    """Sum of the last ``periods`` quarters, only where they are consecutive (spanning under about a year): a gap in
    the data must not make four quarters out of several years."""

    series = pd.to_numeric(series, errors="coerce").dropna()
    total = series.rolling(periods, min_periods=periods).sum()
    if len(series) < periods or not isinstance(series.index, pd.DatetimeIndex):
        return total
    dates = pd.Series(series.index, index=series.index)
    span = (dates - dates.shift(periods - 1)).dt.days
    return total.where(span <= int(periods * 92 + 30))


def split_adjusted_shares(shares, splits):
    """Share counts on today's split basis and one unit scale. ``splits`` maps a split date to its ratio (2 for 2-for-1). Filings often
    keep the old count for quarters before a split; each split multiplies those earlier counts by its ratio, but only
    when the count really jumps across it (about the ratio), so counts a filing already restated are left alone."""

    shares = pd.to_numeric(pd.Series(shares, dtype=float), errors="coerce").dropna().sort_index()
    shares = shares[shares > 0]
    if len(shares):
        # Some filings state the count in thousands or millions (4,306 for 4.306 billion): put every count on the
        # scale of the typical one, in steps of 1,000.
        steps = np.round(np.log10(shares.median() / shares) / 3)
        shares = shares * 1000.0 ** steps
    if shares.empty or splits is None or len(splits) == 0:
        return shares
    for day, ratio in pd.Series(splits, dtype=float).sort_index().items():
        if not np.isfinite(ratio) or ratio <= 0 or abs(ratio - 1) < 0.05:
            continue
        day = pd.Timestamp(day).tz_localize(None) if pd.Timestamp(day).tzinfo else pd.Timestamp(day)
        before, after = shares[shares.index < day], shares[shares.index >= day]
        if before.empty or after.empty:
            continue
        jump = after.iloc[0] / before.iloc[-1]
        if abs(np.log(jump / ratio)) < np.log(1.25):
            shares.loc[before.index] = before * ratio
    return shares


BASE_EFFECT_POINTS = 15.0  # a one-off dip: growth this many points below both neighbouring quarters'


def _year_ago(series, periods=4, years=1):
    """The value about ``years`` years earlier: the entry dated nearest to that day, within 65 days a year, so a
    quarter missing from the data spoils only the comparisons that need it (not the next four by position).
    Without dates, the value ``periods`` entries back."""

    if not isinstance(series.index, pd.DatetimeIndex) or series.empty:
        return series.shift(periods)
    series = series[~series.index.duplicated(keep="last")]
    target = series.index - pd.Timedelta(days=round(365.25 * years))
    position = series.index.get_indexer(target, method="nearest")
    found = series.index[position]
    near = np.abs((found - target).days) <= 65 * years
    values = series.to_numpy()[position]
    return pd.Series(np.where(near, values, np.nan), index=series.index, dtype=float)


def quarterly_inputs(quarters, growth_quarters=12, base_effect=True):
    """Income and cost map: trailing revenue, its growth and the net margin at each release date.

    ``quarters`` is indexed by the date each quarter became public, with any of the columns Revenue, Net_Income and
    EPS. Growth is the last four quarters against the four before; with under eight quarters it is the latest
    quarter against the same quarter a year earlier.
    """

    frame = pd.DataFrame(index=pd.DatetimeIndex(quarters.index).sort_values().unique()) if quarters is not None else pd.DataFrame()
    if quarters is None or quarters.empty:
        return pd.DataFrame(columns=["revenue", "net_income", "eps", "ttm_revenue", "revenue_yoy", "quarter_yoy", "net_margin", "ttm_eps",
                                     "gross_margin", "revenue_cagr", "growth_spread", "eps_spread"])
    quarters = quarters.sort_index()
    quarters = quarters.loc[~quarters.index.duplicated(keep="last")]

    def column(name):
        return pd.to_numeric(quarters[name], errors="coerce").dropna() if name in quarters else pd.Series(dtype=float)

    revenue, net_income, eps = column("Revenue"), column("Net_Income"), column("EPS")
    ttm_revenue, ttm_income = _trailing(revenue), _trailing(net_income)
    # Growth is per share: a company that doubles its revenue by buying another one with new shares has not
    # doubled what each share earns. The share count is the quarter's diluted average, held when a quarter lacks it.
    shares = column("Shares")
    if len(shares) >= 4:
        # Quarters before the first reported count take the earliest one, so the series never mixes bases.
        held_shares = shares.reindex(shares.index.union(revenue.index)).ffill().bfill()
        # Each quarter's revenue over that quarter's shares, then summed, so a share issue mid-year counts only
        # from the quarter it happened.
        growth_quarter = revenue / held_shares.reindex(revenue.index)
        growth_revenue = _trailing(growth_quarter)
    else:
        growth_revenue, growth_quarter = ttm_revenue, revenue
    growth_revenue, growth_quarter = growth_revenue.dropna(), growth_quarter.dropna()
    before, quarter_before = _year_ago(growth_revenue), _year_ago(growth_quarter)
    yoy = (growth_revenue / before.where(before > 0) - 1) * 100
    quarter_yoy = (growth_quarter / quarter_before.where(quarter_before > 0) - 1) * 100
    yoy = yoy.combine_first(quarter_yoy)
    if base_effect:
        # Base effect: when the year-ago quarter was a one-off dip (BSX's 2020 lockdown quarter: -24% while the
        # quarters either side were about flat), growth against it overstates the pace (+51% in mid-2021). Then
        # compare with two years ago instead, as a yearly rate. A dip of several quarters (a cycle) is left alone.
        known = quarter_yoy.dropna()
        neighbours = pd.concat([known.shift(1), known.shift(-1)], axis=1).min(axis=1, skipna=False)
        one_off = (known <= neighbours - BASE_EFFECT_POINTS).astype(float)
        two_before = _year_ago(growth_quarter, 8, years=2)
        two_year = ((growth_quarter / two_before.where(two_before > 0)) ** 0.5 - 1) * 100
        dipped = (_year_ago(one_off).reindex(quarter_yoy.index) == 1) & two_year.reindex(quarter_yoy.index).notna()
        quarter_yoy = quarter_yoy.where(~dipped, two_year)
    margin = (ttm_income.reindex(ttm_revenue.index) / ttm_revenue.where(ttm_revenue > 0)) * 100
    ttm_gross = _trailing(column("Gross_Profit"))
    gross_margin = (ttm_gross.reindex(ttm_revenue.index) / ttm_revenue.where(ttm_revenue > 0)) * 100
    # Yearly growth over the growth history: the last 4 quarters against the 4 quarters ending N quarters earlier,
    # made a yearly rate. Fewer quarters when the history is shorter (at least 4).
    cagr = pd.Series(np.nan, index=ttm_revenue.index, dtype=float)
    for count in range(max(4, int(growth_quarters)), 3, -1):
        earlier = growth_revenue.shift(count)
        ratio = growth_revenue / earlier.where(earlier > 0)
        cagr = cagr.combine_first((ratio.where(ratio > 0) ** (4 / count) - 1) * 100)
    frame["revenue"], frame["net_income"], frame["eps"] = revenue, net_income, eps
    build_up = _build_up_series(quarters, column, revenue, ttm_revenue, shares)
    ttm_eps = _trailing(eps)
    log_eps = np.log(ttm_eps.where(ttm_eps > 0))
    # How much trailing EPS moved in a year (log change), its spread over the last 12 releases (at least 8).
    eps_spread = (log_eps - log_eps.shift(4)).rolling(12, min_periods=8).std()
    for name, series in [("ttm_revenue", ttm_revenue), ("revenue_yoy", yoy), ("quarter_yoy", quarter_yoy.dropna()),
                         ("net_margin", margin),
                         ("ttm_eps", _trailing(eps)), ("gross_margin", gross_margin.dropna()),
                         ("revenue_cagr", cagr.dropna()),
                         # How much the yearly growth varied over the last 12 releases (at least 8), as known then.
                         ("growth_spread", yoy.dropna().rolling(12, min_periods=8).std().dropna()),
                         ("eps_spread", eps_spread.dropna()), *build_up.items()]:
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


# --- Company guidance interface --------------------------------------------------------------------------------
# The model does not read filings itself. A guidance source (analysis/guidance.py, or anything else) hands it a
# table with these columns, one row per guided figure:
#   date        the day the guidance became public (the model uses it from that day on)
#   period      "quarter" (the next quarter) or "year" (the fiscal year)
#   period_end  the last day of the guided quarter or fiscal year
#   metric      "revenue" (an amount), "revenue_growth" (% against the same period a year earlier) or "eps"
#   low, high   the guided range (a single figure has low = high)
# Other columns (ticker, basis, text, ...) are carried along for display.
GUIDANCE_COLUMNS = ["date", "ticker", "accession", "period", "period_end", "metric", "low", "high", "basis", "text"]
GUIDANCE_GROWTH_RANGE = (-60.0, 200.0)  # a guided growth outside this is taken as a misread and left out


def guided_growth(guidance, quarters, base_effect=True):
    """Revenue growth per share (%) the company guided, one row per guidance day: the next quarter against the
    same quarter a year earlier where the company guided the quarter, else its fiscal year against the year
    before. Amounts are turned into growth with the reported revenue the model has; growth stated by the company
    is used as it is. Columns: growth, period, metric, low, high, base (the year-earlier revenue an amount was
    compared with), text."""

    columns = ["growth", "period", "metric", "low", "high", "base", "text"]
    if guidance is None or len(guidance) == 0 or quarters is None or quarters.empty or "Revenue" not in quarters:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(guidance).copy()
    frame = frame[frame["metric"].isin(["revenue", "revenue_growth"])].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    frame["mid"] = (pd.to_numeric(frame["low"], errors="coerce") + pd.to_numeric(frame["high"], errors="coerce")) / 2
    quarters = quarters.sort_index()
    revenue = pd.to_numeric(quarters["Revenue"], errors="coerce").dropna()
    ttm = _trailing(revenue)
    shares = pd.to_numeric(quarters["Shares"], errors="coerce").dropna() if "Shares" in quarters else pd.Series(dtype=float)
    released = pd.DatetimeIndex(revenue.index).normalize()

    def first_release_after(series, day):
        """The value released 5 to 100 days after a period that ended on ``day``, and its release date."""
        dates = pd.DatetimeIndex(series.index).normalize()
        window = (dates > day + pd.Timedelta(days=5)) & (dates <= day + pd.Timedelta(days=100))
        values = series[window].dropna()
        return (float(values.iloc[0]), values.index[0]) if len(values) else (np.nan, None)

    # One-off dips (as in quarterly_inputs): a quarter whose growth is 15 points or more below both neighbours'.
    quarter_growth = (revenue / _year_ago(revenue).where(lambda v: v > 0) - 1) * 100
    known = quarter_growth.dropna()
    neighbours = pd.concat([known.shift(1), known.shift(-1)], axis=1).min(axis=1, skipna=False)
    one_off = known <= neighbours - BASE_EFFECT_POINTS

    rows = []
    for row in frame.itertuples():
        if not np.isfinite(row.mid) or pd.isna(row.period_end):
            continue
        year_before = row.period_end - pd.Timedelta(days=365)
        base = np.nan
        base, base_day = first_release_after(revenue if row.period == "quarter" else ttm, year_before)
        if row.metric == "revenue_growth":
            growth = row.mid
        else:
            growth = (row.mid / base - 1) * 100 if np.isfinite(base) and base > 0 else np.nan
        if not np.isfinite(growth):
            continue
        if base_effect and row.period == "quarter" and base_day is not None and bool(one_off.get(base_day, False)):
            # Guided against a one-off dip a year ago: as a yearly rate against two years ago instead.
            two, _ = first_release_after(revenue, year_before - pd.Timedelta(days=365))
            if np.isfinite(two) and two > 0 and np.isfinite(base) and base > 0:
                growth = (np.sqrt((1 + growth / 100) * base / two) - 1) * 100
        # Per share, like the model's own growth: take off the change in the share count over the last year.
        known = shares[pd.DatetimeIndex(shares.index).normalize() <= row.date]
        if len(known) >= 5 and known.iloc[-5] > 0 and (known.index[-1] - known.index[-5]).days <= 430:
            growth = ((1 + growth / 100) / (known.iloc[-1] / known.iloc[-5]) - 1) * 100
        if not GUIDANCE_GROWTH_RANGE[0] <= growth <= GUIDANCE_GROWTH_RANGE[1]:
            continue
        rank = {("quarter", "revenue"): 0, ("quarter", "revenue_growth"): 1, ("year", "revenue"): 2,
                ("year", "revenue_growth"): 3}.get((row.period, row.metric), 4)
        rows.append({"date": row.date, "rank": rank, "growth": growth, "period": row.period, "metric": row.metric,
                     "low": row.low, "high": row.high, "base": base, "text": getattr(row, "text", "")})
    if not rows:
        return pd.DataFrame(columns=columns)
    found = pd.DataFrame(rows).sort_values(["date", "rank"]).drop_duplicates("date")
    return found.set_index("date")[columns]


def held_guidance(guided, releases, days):
    """Guided growth on each day: from the guidance day until the next quarterly release after it (a release
    the same day as the guidance does not end it). Days with no guidance in force are NaN, so the model holds the
    last reported quarter."""

    out = pd.Series(np.nan, index=days, dtype=float)
    if guided is None or guided.empty:
        return out
    releases = pd.DatetimeIndex(releases).normalize().unique().sort_values()
    events = pd.concat([pd.Series(guided["growth"].astype(float).values, index=guided.index),
                        pd.Series(np.nan, index=releases[~releases.isin(guided.index)])]).sort_index()
    events = events[~events.index.duplicated(keep="first")]
    # A release ends the guidance in force; NaN events are kept (not forward-filled) by marking them first.
    marker = events.isna()
    filled = events.reindex(events.index.union(days)).ffill()
    ended = marker.astype(float).reindex(events.index.union(days)).ffill().fillna(1.0) > 0
    out = filled.where(~ended).reindex(days)
    return out


def _build_up_series(quarters, column, revenue, ttm_revenue, shares):
    """The EPS build-up's trailing inputs at each quarter: EBIT margin, total revenue growth, net interest, tax rate,
    preferred dividends, stock-based pay, buybacks and the latest share count. Empty when there is no EBIT at all.

    EBIT is the operating income; without one (banks, insurers: interest is their business), the pre-tax income with
    no separate interest. Items a company does not report (no preferred shares, no buybacks) count as zero.
    """

    operating, pretax = column("Operating_Income"), column("Pretax_Income")
    if (operating.empty and pretax.empty) or revenue.empty:
        return {}
    dates = revenue.index

    def optional(name):
        return _trailing(column(name).reindex(dates).fillna(0.0))

    ttm_operating = _trailing(operating.reindex(dates)) if len(operating) else pd.Series(np.nan, index=dates)
    ttm_pretax = _trailing(pretax.reindex(dates)) if len(pretax) else pd.Series(np.nan, index=dates)
    has_operating = ttm_operating.notna()
    # One-time costs (acquired intangibles' amortization, impairments, acquired R&D, deal costs) are left out:
    # they are in GAAP operating and pre-tax income but say nothing about next year's earnings.
    one_offs = optional("One_Off_Costs")
    ebit = ttm_operating.where(has_operating, ttm_pretax) + one_offs.reindex(dates).fillna(0.0)
    net_tagged = column("Net_Interest_Nonoperating").reindex(dates)
    quarter_interest = (-net_tagged).where(net_tagged.notna(), column("Interest_Expense").reindex(dates).fillna(0.0)
                                          - column("Interest_Income").reindex(dates).fillna(0.0))
    net_interest = _trailing(quarter_interest).where(has_operating, 0.0)
    ttm_tax = _trailing(column("Income_Tax").reindex(dates)) if "Income_Tax" in quarters else pd.Series(np.nan, index=dates)
    # Tax on the income without the one-offs: their tax saving (at 21%) is taken back out.
    adjusted_pretax = ttm_pretax + one_offs.reindex(dates).fillna(0.0)
    tax_rate = ((ttm_tax + 0.21 * one_offs.reindex(dates).fillna(0.0)) / adjusted_pretax.where(adjusted_pretax > 0)
                * 100).clip(0, 50)
    total_before = _year_ago(ttm_revenue.dropna())
    total_yoy = (ttm_revenue / total_before.where(total_before > 0) - 1) * 100
    series = {
        "ebit_margin": (ebit / ttm_revenue.where(ttm_revenue > 0) * 100).dropna(),
        "revenue_total_yoy": total_yoy.dropna(),
        "bu_revenue": ttm_revenue.dropna(),
        "bu_net_interest": net_interest.dropna(),
        "bu_tax_rate": tax_rate.dropna(),
        "bu_preferred": optional("Preferred_Dividends").dropna(),
        "bu_stock_comp": optional("Stock_Comp").dropna(),
        "bu_buybacks": optional("Buybacks").dropna(),
        "bu_one_offs": one_offs.dropna(),
        "bu_shares": shares,
        "bu_shares_year_ago": _year_ago(shares) if len(shares) else shares,
    }
    return {name: value for name, value in series.items() if len(value)}


def run_model(quarters, close, register=None, params=None, peer_pe=None, start=None, own_pe=None, policy_rate=None,
              market_yield=None, market_close=None, analyst_growth=None, guidance=None):
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
    if not params["exclude_one_offs"] and quarters is not None and "One_Off_Costs" in quarters:
        quarters = quarters.drop(columns=["One_Off_Costs"])
    inputs = quarterly_inputs(quarters, params["growth_history_quarters"], bool(params["quarter_base_effect"]))
    # The build-up grows revenue, not EPS: analysts' EPS growth does not fit it, only your own forward values do.
    own_forward = {key: params[key] for key in ["forward_growth", "forward_margin"]}
    use_build_up = params["eps_model"] == "Build-up" and params["manual_eps"] <= 0
    price_on = close.reindex(close.index.union(pd.DatetimeIndex(inputs.index))).ffill() if len(close) else close
    params["analyst_base"] = False
    if (params["base_eps_source"] == "Analysts" and analyst_growth is not None and np.isfinite(analyst_growth)
            and len(inputs)):
        # The base case is the analysts' EPS: their growth, with the net margin held (margin ratio 1).
        latest_margin = inputs["net_margin"].dropna() if "net_margin" in inputs else pd.Series(dtype=float)
        params["forward_growth"] = float(analyst_growth)  # blended with the EPS trend rule below (analyst_share)
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
    eps_trend = None
    if params["past_base"] == "EPS trend" and params["manual_eps"] <= 0 and "ttm_eps" in held:
        ttm = held["ttm_eps"].astype(float)
        known = inputs["ttm_eps"].astype(float) if "ttm_eps" in inputs else ttm
        recovery = float(params["history_recovery"])
        # Normal EPS: the average of the last 3 years (12 releases) as known on each release.
        normal_known = known.rolling(12, min_periods=8).mean()
        reference = known.shift(4)
        if recovery > 0:
            # A rebound is measured from the normal level a year ago, not from the slump, so the climb back out of
            # a one-off dip is not projected forward as growth.
            reference = pd.concat([reference, normal_known.shift(4)], axis=1).max(axis=1, skipna=False).fillna(reference)
        year_ago = reference.reindex(known.index.union(held.index)).ffill().reindex(held.index)
        change = (ttm / year_ago.where(year_ago > 0) - 1) * 100
        eps_trend = (float(params["history_growth_follow"]) * change).clip(
            lower=float(params["history_growth_floor"]), upper=float(params["history_growth_cap"])).where(ttm > 0)
        if recovery > 0:
            # Below its normal level, a share of the gap is expected back (the forward P/E history rule's recovery
            # view), so a one-off slump (a pandemic quarter) is not projected forward as a trend.
            normal = normal_known.reindex(normal_known.index.union(held.index)).ffill().reindex(held.index)
            back = (recovery * (normal / ttm - 1) * 100).where((normal > ttm) & (ttm > 0))
            eps_trend = pd.concat([eps_trend, back], axis=1).max(axis=1, skipna=True).where(ttm > 0)
    if params["analyst_base"]:
        # Analysts' growth on the latest release only, blended with the rule every earlier release uses (the EPS
        # trend, else the auto rule), so one bold estimate does not make a step the past line could never show.
        share = float(np.clip(params["analyst_share"], 0, 1))
        rule = eps_trend.get(last_release, np.nan) if eps_trend is not None and last_release is not None else np.nan
        if not np.isfinite(rule):
            latest = held.loc[last_release] if last_release in held.index else held.iloc[-1]
            rule = base_growth_and_margin(latest.get("revenue_yoy", np.nan), latest.get("net_margin", np.nan),
                                          {**params, "forward_growth": np.nan})[1]
        params["analyst_growth"], params["rule_growth_latest"] = float(analyst_growth), float(rule)
        params["forward_growth"] = share * float(analyst_growth) + (1 - share) * float(rule)
    was_on = False
    for when, row in held.iterrows():
        ttm_eps = params["manual_eps"] if params["manual_eps"] > 0 else row.get("ttm_eps", np.nan)
        events, points = case_effects(routed, when, params)
        forward = last_release is None or when >= last_release
        row_params, row_forward = params, forward
        past_growth = eps_trend.get(when, np.nan) if eps_trend is not None else np.nan
        if not forward and np.isfinite(past_growth):
            # Before the latest release: the EPS trend known then (the forward P/E history rule), margin held, so a
            # company whose margins are widening is not cut to its revenue growth.
            margin = row.get("net_margin", np.nan)
            row_params = {**params, "forward_growth": past_growth,
                          "forward_margin": margin if np.isfinite(margin) and margin > 0 else params["fallback_margin"]}
            row_forward = True
        build_up = None
        if use_build_up:
            price = price_on.asof(when) if len(price_on) else np.nan
            year = close.loc[when - pd.Timedelta(days=365):when] if len(close) else close
            creep = share_creep_pct(row.get("bu_shares", np.nan), row.get("bu_shares_year_ago", np.nan),
                                    row.get("bu_buybacks", 0.0), float(year.mean()) if len(year) else np.nan,
                                    row.get("bu_stock_comp", 0.0), price, float(params["share_creep_cap"]))
            build_up = build_up_inputs({**row.to_dict(), "bu_price": price, "bu_creep_pct": creep})
            if build_up is not None:
                # GAAP EBIT can carry non-cash one-offs the reported EPS leaves out (AMD: Xilinx amortization took
                # its operating margin to about 0 in 2023). The margin used is at least a share of the one behind
                # the reported EPS after the haircut (which keeps stock-based pay in).
                clean_eps = ttm_eps * (1 - params["eps_haircut_pct"] / 100)
                floor = float(params["one_off_floor"]) * implied_ebit_margin(clean_eps, build_up, params)
                build_up["ebit_margin_gaap"], build_up["bu_margin_floor"] = build_up["ebit_margin"], floor
                if np.isfinite(floor) and floor > build_up["ebit_margin"]:
                    build_up["ebit_margin"] = floor
        if build_up is not None:
            # Build-up: revenue growth from the auto rule (or your forward growth on the latest release), on total
            # revenue (the share count changes on its own), and the EBIT margin.
            row_params, row_forward = {**params, **own_forward}, forward
            past_growth = np.nan
            point = scenario_point(ttm_eps, row.get("revenue_total_yoy", np.nan), build_up["ebit_margin"], row_params,
                                   events, row_forward, row.get("eps_spread", np.nan), build_up=build_up)
            point.update({name: build_up[name] for name in BUILD_UP_INPUTS})
            point.update({name: build_up.get(name, np.nan) for name in ["ebit_margin", "ebit_margin_gaap", "bu_margin_floor"]})
            point["bu_on"] = 1.0
        else:
            point = scenario_point(ttm_eps, row.get("revenue_yoy", np.nan), row.get("net_margin", np.nan), row_params,
                                   events, row_forward,
                                   row.get("eps_spread", np.nan) if params["manual_eps"] <= 0 else np.nan)
            point["bu_on"] = 0.0
        point["past_growth"] = past_growth if not forward else np.nan
        point["cycle_eps"], point["cycle_trend"], point["cycle_on"] = np.nan, np.nan, 0.0
        point["next_year_eps"], point["cycle_past_sum"] = point.get("risk_adjusted_eps", np.nan), np.nan
        point["cycle_years_used"] = np.nan
        if params["manual_eps"] <= 0 and "ttm_eps" in inputs:
            cycle, trend, on = through_cycle_eps(inputs["ttm_eps"].loc[:when], point.get("risk_adjusted_eps", np.nan),
                                                 1 - params["eps_haircut_pct"] / 100, params, was_on,
                                                 inputs["ttm_revenue"].loc[:when]
                                                 if "ttm_revenue" in inputs else None)
            was_on = on
            point["cycle_eps"], point["cycle_trend"] = cycle, trend
            if on:
                # Deep cycle: value the average EPS over the cycle; next year's EPS is one of its years.
                point["cycle_years_used"] = float(min(int(params["cycle_years"]),
                                                      1 + len(inputs["ttm_eps"].loc[:when].dropna().iloc[::-4])))
                point["cycle_past_sum"] = cycle * point["cycle_years_used"] - point["risk_adjusted_eps"]
                point["risk_adjusted_eps"], point["cycle_on"] = cycle, 1.0
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
        daily["held_quarter_yoy"] = np.nan
        daily["held_cycle_on"], daily["held_cycle_trend"] = 0.0, np.nan
    else:
        # Hold blocks: each quarterly output keeps its last value until the next update.
        both = quarterly.reindex(quarterly.index.union(close.index)).ffill().reindex(close.index)
        daily["held_eps"], daily["held_pe_adjust"], held_ttm = both["risk_adjusted_eps"], both["pe_adjust"].fillna(0.0), both["ttm_eps"]
        daily["held_quality_pe"] = both["quality_pe"].fillna(0.0)
        daily["held_clean_eps"] = both["clean_eps"]
        daily["held_cagr"] = both["revenue_cagr"] if "revenue_cagr" in both else np.nan
        daily["held_yoy"] = both["revenue_yoy"] if "revenue_yoy" in both else np.nan
        daily["held_quarter_yoy"] = both["quarter_yoy"] if "quarter_yoy" in both else np.nan
        daily["held_growth_spread"] = both["growth_spread"] if "growth_spread" in both else np.nan
        daily["held_gross_margin"] = both["gross_margin"] if "gross_margin" in both else np.nan
        daily["held_cycle_on"] = both["cycle_on"].fillna(0.0) if "cycle_on" in both else 0.0
        daily["held_cycle_trend"] = both["cycle_trend"] if "cycle_trend" in both else np.nan
    # Rolling forward: next year's EPS is the next 12 months from the release. Each day after it the window covers
    # later months, so the EPS grows at the rate the release set (risk-adjusted EPS over the clean trailing EPS),
    # instead of jumping once a quarter. Only what was known at the release is used. Through-cycle EPS stays put.
    daily["eps_roll"], daily["roll_years"] = 1.0, 0.0
    if params["roll_forward_eps"] and not quarterly.empty and params["manual_eps"] <= 0:
        since = pd.Series(quarterly.index, index=quarterly.index).reindex(
            quarterly.index.union(close.index)).ffill().reindex(close.index)
        years = ((close.index.to_series() - since).dt.days / 365.25).clip(lower=0, upper=1)
        ratio = (daily["held_eps"] / daily["held_clean_eps"]).where(
            (daily["held_eps"] > 0) & (daily["held_clean_eps"] > 0))
        roll = (ratio.clip(0.5, 3.0) ** years).where(daily["held_cycle_on"].astype(float) <= 0)
        daily["eps_roll"] = roll.fillna(1.0)
        daily["roll_years"] = years.where(roll.notna(), 0.0)
        daily["held_eps"] = daily["held_eps"] * daily["eps_roll"]

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
        usable = peers.where(peers <= float(params["peer_pe_max"]))
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
    # Company guidance (when a source gave some): in force from its day until the next quarterly release.
    guided = (guided_growth(guidance, quarters, bool(params["quarter_base_effect"])) if params.get("use_guidance", False)
              else guided_growth(None, None))
    daily["held_guided_growth"] = held_guidance(guided, quarterly.index if not quarterly.empty else [], daily.index)
    cagr, latest = daily["held_cagr"].astype(float), daily["held_yoy"].astype(float)
    if params.get("pe_growth_latest", "Latest quarter") == "Latest quarter" and "held_quarter_yoy" in daily:
        # The latest quarter against the same quarter a year earlier: the figure the market reacts to.
        latest = daily["held_quarter_yoy"].astype(float).clip(-50, 150).combine_first(latest)
    # Guidance in force replaces the reported growth; with none, the last reported quarter is held.
    latest = daily["held_guided_growth"].astype(float).clip(-50, 150).combine_first(latest)
    share = float(np.clip(params["pe_growth_latest_share"], 0, 1))
    growth = (share * latest + (1 - share) * cagr).combine_first(cagr).combine_first(latest)
    if growth.isna().all():
        growth = growth.fillna(float(params["fallback_yoy"]))
    # On through-cycle EPS, the P/E rests on the cycle's trend growth, not on one year's swing.
    cycle_days = daily["held_cycle_on"].astype(float) > 0
    cycle_growth = daily["held_cycle_trend"].astype(float).clip(lower=0.0, upper=float(params["cycle_trend_cap"]))
    growth = growth.where(~cycle_days, cycle_growth).combine_first(growth)
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
    # A typed-in EPS is held over the whole history, so the market P/E it gives in the past is not real: no usual
    # premium from it (the beta rule applies).
    if params["market_implied_erp"] and params["anchor_source"] == "Fundamentals" and params["manual_eps"] <= 0:
        # Respect the market: the premium it has usually asked of this stock, with today's yield.
        market_pe = daily["close"] / daily["held_eps"].where(daily["held_eps"] > 0)
        implied, usual = usual_implied_erp(market_pe, growth, daily["growth_half_life"], now_yield, params,
                                           params["erp_lookback_years"])
        if cycle_days.any():
            # Like with like: a day on through-cycle EPS takes the usual premium of the months on it, and a day on
            # one year's EPS that of the months on one year's.
            on_cycle = usual_implied_erp(market_pe.where(cycle_days), growth, daily["growth_half_life"], now_yield,
                                         params, params["erp_lookback_years"])[1]
            off_cycle = usual_implied_erp(market_pe.where(~cycle_days), growth, daily["growth_half_life"], now_yield,
                                          params, params["erp_lookback_years"])[1]
            # A basis with no months of its own yet borrows the other's usual premium rather than the beta rule.
            usual = on_cycle.where(cycle_days, off_cycle).combine_first(usual)
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
    # trailing EPS with no growth or margin change. Next to the risk-adjusted fair value it shows what the forward inputs add.
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
            "guidance": guided,
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


def build_up_view(values):
    """(revenue growth %, margin %, build-up inputs or None) the scenario engine used on the latest release: the
    total revenue growth and EBIT margin when its EPS came from the build-up, else the per-share growth and net margin."""

    build_up = build_up_inputs(values) if values.get("bu_on", 0) == 1 else None
    if build_up is None:
        return values.get("revenue_yoy", np.nan), values.get("net_margin", np.nan), None
    return values.get("revenue_total_yoy", np.nan), build_up["ebit_margin"], build_up


def _point_price(state, params, routed):
    """Risk-adjusted fair value from one day's held inputs: the whole chain as one function of the parameters and items."""

    events, points = case_effects(routed, state["on"], params)
    point = scenario_point(state["ttm_eps"], state["revenue_yoy"], state["net_margin"], params, events,
                           state.get("forward", True), state.get("eps_spread", np.nan), build_up=state.get("build_up"))
    eps = point["risk_adjusted_eps"]
    years = state.get("roll_years", 0.0)
    if years > 0 and eps > 0 and point["clean_eps"] > 0:
        eps = eps * float(np.clip(eps / point["clean_eps"], 0.5, 3.0)) ** years
    if state.get("cycle_on") and np.isfinite(state.get("cycle_past_sum", np.nan)):
        eps = (state["cycle_past_sum"] + eps) / state.get("cycle_years_used", params["cycle_years"])
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
    growth, margin, build_up = build_up_view(values)
    return {"on": on, "ttm_eps": values.get("ttm_eps", np.nan), "revenue_yoy": growth,
            "net_margin": margin, "anchor": anchor, "build_up": build_up,
            "eps_spread": values.get("eps_spread", np.nan) if result["params"]["manual_eps"] <= 0 else np.nan,
            "own_anchor": day.get("own_anchor", np.nan),
            "peer_anchor": day.get("outside_anchor", day.get("peer_anchor", np.nan)),
            "held_cagr": day.get("pe_growth", day.get("held_cagr", np.nan)),
            "growth_half_life": day.get("pe_half_life", day.get("growth_half_life", np.nan)),
            "premium": float(day.get("premium_pct", 0.0)) if np.isfinite(day.get("premium_pct", np.nan)) else 0.0,
            "pe_erp": day.get("pe_erp", np.nan),
            "policy_rate": day.get("policy_rate", np.nan), "yield_ref": day.get("yield_ref", np.nan),
            "now_yield": day.get("now_yield", np.nan),
            "cycle_on": bool(values.get("cycle_on", 0) or 0) and result["params"]["manual_eps"] <= 0,
            "cycle_past_sum": values.get("cycle_past_sum", np.nan),
            "cycle_years_used": values.get("cycle_years_used", np.nan),
            "roll_years": float(day.get("roll_years", 0.0)) if np.isfinite(day.get("roll_years", np.nan)) else 0.0,
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
    risk-adjusted fair value equal the close on ``on`` (default: the last day), and whether that value is believable. A parked
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
    # The P/E is searched past your cap, so the row can say how far the market is beyond it; plausible is within
    # a third below to half above the stock's own usual P/E (or the current one when that is unknown).
    pe_now = float(np.clip(anchor + points, params["pe_floor"], params["pe_cap"]))
    usual_pe = values.get("own_anchor", np.nan)
    reference = usual_pe if np.isfinite(usual_pe) and usual_pe > 0 else pe_now
    cap = float(params["pe_cap"])
    add("Risk-adjusted P/E", "Multiple builder", pe_now, 1, max(cap, 1000.0), (reference / 1.5, reference * 1.5), "x",
        lambda multiple: _point_price(state, {**params, "pe_floor": multiple, "pe_cap": multiple}, routed))
    if np.isfinite(rows[-1]["Needed"]) and rows[-1]["Needed"] > cap:
        rows[-1]["Read"] = f"Beyond your {cap:.0f}x cap"
    if values.get("cycle_on", 0) > 0:
        # Through-cycle EPS is the average over the cycle: these dials only move next year's share of it.
        for row in rows:
            if row["Dial"] in ("Bear weight", "Growth in every case", "Net margin in every case") \
                    and row["Read"] == "Cannot match alone":
                row["Read"] = "Little effect (through-cycle EPS)"

    if has_rates(result):
        used = result["daily"].loc[:on, "forward_rate"].iloc[-1]
        add("Policy rate", "Rate model", used, -1, 20, (max(0.0, used - 1.5), used + 1.5), "%",
            lambda rate: _point_price({**state, "anchor": anchor_at_rate(state, params, rate)}, params, routed))

    model_above = values["model_price"] > target
    for position, row in routed.iterrows():
        parked = row["Gate"] != MODELLED
        label = f"{row['Item']} (parked)" if parked else row["Item"]
        # A risk can only pull the risk-adjusted fair value down, a product can only lift it.
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
    """For each item with a probability to set: the probability that alone makes the risk-adjusted fair value equal the close.

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
    view_growth, view_margin, build_up = build_up_view(values)
    yoy, growth, margin, base_margin = base_growth_and_margin(view_growth, view_margin, params, forward=True)
    if build_up is not None:
        margin = view_margin
        base_margin = (float(params["forward_margin"]) if np.isfinite(params.get("forward_margin", np.nan))
                       else margin + params["base_margin_pts"])
    reported = {"Revenue growth %": view_growth, "Net margin %": view_margin}
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
                     "Forward from": ("analysts" if params.get("analyst_base") and build_up is None else
                                      "you" if np.isfinite(params.get(key, np.nan)) else "auto rule")})
    table = pd.DataFrame(rows, columns=CASE_COLUMNS)
    if build_up is not None:
        # The build-up grows total revenue and applies the EBIT margin.
        table["Metric"] = table["Metric"].replace(BUILD_UP_METRICS)
        table.loc[table["Metric"] == BUILD_UP_METRICS["Net margin %"], "Current from"] = "reported"
    return table


BUILD_UP_METRICS = {"Revenue growth %": "Revenue growth % (total)", "Net margin %": "EBIT margin %"}


def case_table_changes(before, after, params):
    """Parameter updates from an edited case table (``before`` as ``case_table`` drew it, ``after`` as edited).

    A forward value left as drawn keeps its source (auto stays auto); a changed one becomes yours; a cleared one goes
    back to the auto rule. Returns {parameter: value}.
    """

    updates = {}
    names = {shown: metric for metric, shown in BUILD_UP_METRICS.items()}
    before = before.assign(Metric=before["Metric"].replace(names)).set_index("Metric")
    after = pd.DataFrame(after)
    after = after.assign(Metric=after["Metric"].replace(names)).set_index("Metric")
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
    """Risk-adjusted fair value with ``shift`` points added to the growth of every case."""

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
    """How far the risk-adjusted fair value moves for one step of each main dial (10 points of bear weight, 1x of P/E ...)."""

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
    """The risk-adjusted fair value in each policy rate case and their probability-weighted expected value.

    ``cases`` has the columns Case, Policy rate % and Probability % (rescaled to add up to 100). Each case reruns the
    rate model and the multiple builder for the day with that rate on it; everything else is held. Returns
    ``(table, expected)``: the table adds the case's N-year yield, anchor, risk-adjusted fair value and change vs the close;
    ``expected`` is the weighted risk-adjusted fair value (NaN when there is nothing to weigh).
    """

    columns = RATE_CASE_COLUMNS + ["N-year yield %", "P/E anchor", "Risk-adjusted fair value", "vs close %"]
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
                     "Risk-adjusted fair value": price, "vs close %": (price / values["close"] - 1) * 100})
    table = pd.DataFrame(rows, columns=columns)
    weights = pd.to_numeric(table["Probability %"], errors="coerce").fillna(0).clip(lower=0)
    usable = weights.where(table["Risk-adjusted fair value"].notna(), 0)
    expected = float((usable * table["Risk-adjusted fair value"].fillna(0)).sum() / usable.sum()) if usable.sum() > 0 else np.nan
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

    Rows: policy rate, N-year yield, EPS, P/E, risk-adjusted fair value and its gap to the close. Columns: Now, Forward, Change.
    """

    values = latest_values(result, on)
    columns = ["Step", "Model", "Risk-adjusted", "Change"]
    if not values:
        return pd.DataFrame(columns=columns)
    rows = [("Policy rate %", "policy_rate", "forward_rate"), ("N-year yield %", "now_yield", "model_yield"),
            ("EPS", "held_clean_eps", "held_eps"), ("P/E", "now_pe", "acceptable_pe"),
            ("Fair value", "now_price", "model_price"), ("vs close %", "now_residual_pct", "residual_pct")]
    table = [{"Step": label, "Model": values.get(now, np.nan), "Risk-adjusted": values.get(forward, np.nan)}
             for label, now, forward in rows]
    frame = pd.DataFrame(table, columns=columns[:3])
    frame = frame[frame[["Model", "Risk-adjusted"]].notna().any(axis=1)].reset_index(drop=True)
    frame["Change"] = frame["Risk-adjusted"] - frame["Model"]
    return frame


def premium_warning(values, params):
    """A plain warning when the risk premium the model uses is below the flag level or negative, or when today's
    close lies beyond the lowest premium the model can solve; None otherwise."""

    params = model_params(params)
    if not (params["market_implied_erp"] and params["anchor_source"] == "Fundamentals"):
        return None
    used, today = values.get("pe_erp", np.nan), values.get("implied_erp", np.nan)
    level, yield_pct = float(params["erp_flag_level"]), values.get("now_yield", np.nan)
    notes = []
    if np.isfinite(used) and used < level:
        if used < 0:
            notes.append(f"The risk premium the market has usually asked of this stock is negative ({used:.2f}%): "
                         "investors have accepted a return below the Treasury yield"
                         + (f" ({yield_pct:.2f}%)" if np.isfinite(yield_pct) else "") + ".")
            notes.append("The model follows it, so its price rests on the market's story, not on the earnings "
                         "alone.")
        else:
            notes.append(f"The risk premium the market has usually asked of this stock is only {used:.2f}%, below "
                         f"{level:g}%: the market prices it richly, and the model follows.")
    low = max(float(params["implied_erp_floor"]), params["terminal_growth"] + 1.0 - (
        yield_pct if np.isfinite(yield_pct) else params["base_yield"]))
    if np.isfinite(today) and today <= low + 1e-6:
        notes.append("Today's close needs an even lower premium than the model can solve (the required return "
                     "would fall to the long-run growth rate): what is left shows as market premium.")
    return " ".join(notes) or None
