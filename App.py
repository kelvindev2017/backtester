import dataclasses
import hashlib
from datetime import date, timedelta
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import unquote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from plotly.subplots import make_subplots

from analysis.opening_bar import (
    EXIT_FILLS as OPENING_BAR_FILLS,
    EXIT_MODES as OPENING_BAR_EXITS,
    EXIT_OPERATORS as OPENING_BAR_OPERATORS,
    EXIT_VALUES as OPENING_BAR_VALUES,
    CustomExit,
    COMMISSION_PLANS as OPENING_BAR_COMMISSIONS,
    OpeningBarSettings,
    order_costs,
    run_opening_bar,
    summarize_opening_bar,
)
from analysis.optimizer import (
    N_STARTUP_TRIALS,
    OBJECTIVES,
    PENALTY_TERMS,
    PERIODS_PER_YEAR,
    SCORE_TERMS,
    SEARCH_SPACES,
    TERM_FORMULAS,
    nearest_trials,
    objective_latex,
    objective_score,
    param_importance,
    params_of,
    period_metrics,
    pick_settings,
    run_search,
    split_date,
    unique_scored,
)
from analysis.chart_fit import fit_y_axes
from analysis.economy import (
    ASSETS,
    BENCHMARK as ECONOMY_BENCHMARK,
    DEFAULT_ASSETS,
    DEFAULT_INDICATORS,
    HISTORY_CHOICES,
    INDICATORS,
    RECESSION_SERIES,
    SP500_INDEX,
    asset_table,
    assets_figure,
    fred_series,
    indicator_values,
    latest_text,
    macro_figure,
    rebase,
    window_start,
)
from analysis.earnings_dates import (
    EASTERN,
    change_since_release,
    describe_release,
    merge_releases,
    next_release_text,
    past_and_next,
    reaction_summary,
    release_table,
    releases_with_reaction,
    sec_releases,
    surprise_held,
    with_valuation_changes,
    yahoo_releases,
)
from analysis.engine import AnalysisSettings, analyze, report_to_dict, report_to_markdown
from analysis.indicators import resample_ohlcv
from analysis.notify import send_webhook, short_summary
from analysis.patterns import (
    MA_RULE_OPERATORS,
    MA_RULE_SERIES,
    describe_ma_rule,
    detect_signals,
    ma_rule_conditions,
    moving_average,
    rule_condition,
)
from analysis.risk import RiskSettings
from analysis.rule_diagrams import (
    breakout_figure,
    day_breakout_figure,
    ma_rule_figure,
    macd_bollinger_figure,
    opening_bar_figure,
    pullback_figure,
    reversal_figure,
)
from analysis.condition_table import (
    ALL_OPERATORS as CONDITION_OPERATORS,
    COLUMNS as CONDITION_COLUMNS,
    SIGNALS as CONDITION_SIGNALS,
    default_row as condition_default_row,
    default_table as default_condition_table,
    normalize as normalize_condition_table,
    row_condition,
)
from analysis.screener import (
    MARKET_CAP_BUCKETS,
    RATE_OF_CHANGE,
    TRADING_DAYS_PER_YEAR,
    breadth_history,
    close_sparkline,
    condition_label,
    fundamental_trend,
    index_trend,
    leaders_health,
    market_gate,
    screen_stocks,
    snapshot_table,
    stock_snapshot,
    vix_summary,
)
from analysis.scoring import COMPONENT_WEIGHTS
from analysis.supply_chain import (
    CHAINS as SUPPLY_CHAINS,
    COMPANIES as SUPPLY_COMPANIES,
    LEVEL_SHORT_NAMES as SUPPLY_LEVEL_SHORT_NAMES,
    LEVELS as SUPPLY_LEVELS,
    NODE_METRICS as SUPPLY_NODE_METRICS,
    chain_tickers,
    linked_tickers,
    neighbors,
    node_positions,
    node_table,
)
from analysis.signal_groups import (
    ANALYST_RATINGS,
    DEFAULT_INDUSTRIES,
    FUNDAMENTAL_COLUMNS,
    GICS_SECTORS,
    INDUSTRY_INDEXES,
    INDUSTRY_TOP_HOLDINGS,
    MOVING_AVERAGE,
    SIGNAL_GROUPS,
    condition_column,
    fundamental_row,
    overlay_frame,
    period_start,
)
from analysis.valuation_chart import (
    DOWN_COLOR,
    UP_COLOR,
    VALUATION_HISTORY_YEARS,
    add_volume_overlay,
    correlation_text,
    earnings_signal_figure,
    eps_estimate_quarters,
    forward_eps,
    forward_valuation,
    growth_price_correlation,
    legends_under_panels,
    pe_summary,
    pe_summary_text,
    peg_gap_reasons,
    peg_summary_text,
    price_yoy,
    quarterly_eps,
    revenue_estimate_quarters,
    revenue_pe_figure,
    trailing_peg,
    yoy_change,
)
from analysis.what_happened_next import (
    CHECK_FREQUENCIES,
    DEFAULT_BAND,
    DEFAULT_LOOKBACK_YEARS,
    LOOK_AHEADS,
    LOOKBACK_CHOICES,
    observations,
    scatter_figure,
    similar_mask,
    summary_table,
    today_text,
    walkthrough_figure,
)
from analysis.universe import (
    MARKETS as SCREENER_MARKETS,
    load_us_index_members,
    normalize_symbol,
    normalize_ticker,
    parse_ticker_list,
)
from analysis.trade_journal import (
    BREAKDOWNS as JOURNAL_BREAKDOWNS,
    CASH_LIKE as JOURNAL_CASH_LIKE,
    CHART_TIMEFRAMES,
    ESTIMATED_ACTIONS,
    IMPROVEMENT_ACTIONS,
    INTRADAY_CHART_INTERVALS,
    NO_ACTION,
    WhatIfRules,
    apply_filters as journal_filter,
    breakdown as journal_breakdown,
    chart_request,
    chart_timeframes,
    chart_window,
    equity_curve as journal_equity,
    action_rules,
    fill_bar_positions,
    improvement_summary,
    load_ibkr_trades,
    notes_frame,
    notes_from_csv,
    prepare as journal_prepare,
    round_trips,
    simulate as journal_simulate,
    stop_sweep,
    summarize as journal_summary,
)
from analysis.validation import ValidationSettings, base_rates, exit_signal_stats, summarize, verdict, walk_forward


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Trade Strategy Backtest",
    page_icon=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.png"),
    layout="wide",
    initial_sidebar_state="auto",
)

# Touch screens: Plotly's own touch handling (drag to zoom) is replaced, so that
# - a swipe over a chart scrolls the page, like anywhere else;
# - a tap shows the values under the finger, and a double-tap resets the zoom;
# - holding a finger still for HOLD_MS and then moving it moves the hover line with the finger;
# - two fingers zoom (pinch) and pan the dates; the values fit themselves to the dates in view (CHART_FIT_Y_JS);
# - after a tap on the chart's own Pan button, one finger pans the chart instead of scrolling the page,
#   until the Zoom button is tapped.
# The gestures reach Plotly as the mouse and wheel events it already understands (wheel zoom needs the
# chart's scrollZoom, which show_chart turns on in the mobile layout). Mice and touchpads are untouched.
# Charts that turn dragging off themselves (the supply chain graph) are left alone.
# Runs in the app page itself (not this component's iframe), so it outlives reruns; installed once.
CHART_TOUCH_JS = """
(function () {
  const HOLD_MS = 200, SLOP_PX = 10, DOUBLE_TAP_MS = 350;
  let gesture = null, lastTap = null;

  // The chart under a touch on its plot area (Plotly's drag layers), unless the chart turned dragging off.
  function chartOf(target) {
    const gd = target.closest && target.closest(".drag") ? target.closest(".js-plotly-plot") : null;
    const layout = gd && (gd._fullLayout || gd.layout);
    return gd && !(layout && layout.dragmode === false) ? gd : null;
  }
  function mouse(type, target, x, y, buttons) {
    target.dispatchEvent(new MouseEvent(type, { clientX: x, clientY: y, button: 0, buttons: buttons || 0,
                                                bubbles: true, cancelable: true, view: window }));
  }
  function wheel(target, x, y, deltaY) {
    target.dispatchEvent(new WheelEvent("wheel", { clientX: x, clientY: y, deltaY: deltaY, deltaMode: 0,
                                                   bubbles: true, cancelable: true, view: window }));
  }
  function modebar(gd, attr, values) {  // press one of the chart's own toolbar buttons
    for (const value of values) {
      const button = gd.querySelector('.modebar-btn[data-attr="' + attr + '"][data-val="' + value + '"]');
      if (button) { button.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true })); return; }
    }
  }
  // Hover at a point afresh: Plotly sends no hover event for the point it thinks is still hovered, and
  // the labels were cleared when the touch began.
  function hoverAt(target, x, y) {
    mouse("mouseout", target, x, y);
    mouse("mousemove", target, x, y);
  }
  function centre(touches) {
    return { x: (touches[0].clientX + touches[1].clientX) / 2, y: (touches[0].clientY + touches[1].clientY) / 2 };
  }
  function spread(touches) {
    return Math.hypot(touches[0].clientX - touches[1].clientX, touches[0].clientY - touches[1].clientY);
  }
  // The drag mode the person chose on the chart's toolbar. The pinch presses Pan itself, which must not
  // stop swipes from scrolling the page, so only a real tap counts.
  document.addEventListener("click", function (event) {
    const button = event.isTrusted && event.target.closest && event.target.closest('.modebar-btn[data-attr="dragmode"]');
    const gd = button && button.closest(".js-plotly-plot");
    if (gd) gd.dataset.chosenDrag = button.getAttribute("data-val");
  }, true);
  function span(touches, axis) {
    return axis === "x" ? Math.abs(touches[0].clientX - touches[1].clientX) : Math.abs(touches[0].clientY - touches[1].clientY);
  }
  // The x and y axes of the panel under a touch, from its drag box ("x2y3" names xaxis2 and yaxis3). The y-axis
  // may be fixed (it fits itself to the dates in view); the dates must not be.
  function axesOf(gd, target) {
    const box = target.closest && target.closest(".drag");
    const subplot = box && box.getAttribute("data-subplot");
    const layout = gd._fullLayout, match = subplot && /^(x\\d*)(y\\d*)$/.exec(subplot);
    if (!match || !window.Plotly) return null;
    const xa = layout[match[1].replace("x", "xaxis")], ya = layout[match[2].replace("y", "yaxis")];
    return xa && ya && !xa.fixedrange ? { x: xa, y: ya } : null;
  }
  function axisRange(axis) { return axis.range.map(axis.r2l); }
  // Zoom one axis by how far apart the fingers moved along it, and pan it as their centre moves: the value under
  // the fingers when they landed stays under their centre, so both fingers moving the same way pan.
  function zoomAxis(gesture, touches) {
    const axis = gesture.axes[gesture.axis], rect = gesture.gd.getBoundingClientRect(), length = axis._length;
    function along(c) {  // 0 at the axis's range[0] end, 1 at range[1]
      const pixel = (gesture.axis === "x" ? c.x - rect.left : c.y - rect.top) - axis._offset;
      return gesture.axis === "x" ? pixel / length : 1 - pixel / length;
    }
    const start = gesture.start, width = start[1] - start[0];
    const at = start[0] + along(gesture.centre) * width;
    const newWidth = width * gesture.span / Math.max(span(touches, gesture.axis), 1);
    const low = at - along(centre(touches)) * newWidth;
    const update = {};
    update[axis._name + ".range"] = [low, low + newWidth].map(function (v) { return axis.l2r(v); });
    if (gesture.frame) cancelAnimationFrame(gesture.frame);
    gesture.frame = requestAnimationFrame(function () { window.Plotly.relayout(gesture.gd, update); });
  }
  function endPinch() {
    if (gesture && (gesture.mode === "pinch" || gesture.mode === "pan")) mouse("mouseup", gesture.target, gesture.centre.x, gesture.centre.y, 0);
  }

  document.addEventListener("touchstart", function (event) {
    if (!event.isTrusted) return;
    const multi = event.touches.length >= 2;
    const gd = chartOf(event.target) || (multi && gesture && gesture.gd) || null;
    if (!gd) { endPinch(); gesture = null; return; }
    event.stopPropagation();  // Plotly never sees the touch itself
    if (multi) {
      // A second finger: pinch to zoom, move both to pan. This holds whatever the first finger did (a swipe that
      // started scrolling the page, or a touch beside the plot): left to the browser, a pinch zooms the whole
      // page, every chart with it, and springs back when the fingers lift.
      if (gesture) clearTimeout(gesture.timer);
      if (event.cancelable) event.preventDefault();
      if (gesture && (gesture.mode === "pinch" || gesture.mode === "axis-pinch")) return;
      const c = centre(event.touches);
      const target = gesture && gesture.gd === gd && gesture.target.closest(".drag") ? gesture.target : event.target;
      if (gesture && gesture.mode === "pan") mouse("mouseup", gesture.target, c.x, c.y, 0);
      // Two fingers zoom and pan the time only, however they are held: the values fit themselves to it.
      const axis = "x";
      const axes = axesOf(gd, target);
      if (axes) {
        gesture = { gd: gd, target: target, mode: "axis-pinch", axis: axis, axes: axes, centre: c,
                    span: Math.max(span(event.touches, axis), 1), start: axisRange(axes[axis]) };
        return;
      }
      modebar(gd, "dragmode", ["pan"]);
      // Not on a panel's plot area (e.g. on an axis): zoom and pan through Plotly's own wheel zoom and pan, which
      // leave the fixed y-axes alone.
      gesture = { gd: gd, target: target, mode: "pinch", centre: c, spread: spread(event.touches) };
      mouse("mousedown", target, c.x, c.y, 1);
      return;
    }
    if (event.touches.length !== 1) return;
    const touch = event.touches[0];
    gesture = { gd: gd, target: event.target, mode: "pending", x: touch.clientX, y: touch.clientY };
    const held = gesture;
    held.timer = setTimeout(function () {
      if (gesture !== held || held.mode !== "pending") return;
      held.mode = "scrub";
      if (navigator.vibrate) navigator.vibrate(15);
      hoverAt(held.target, held.x, held.y);
    }, HOLD_MS);
  }, { capture: true, passive: false });

  document.addEventListener("touchmove", function (event) {
    if (!gesture || !event.isTrusted) return;
    if (gesture.mode === "pending") {
      const touch = event.touches[0];
      if (Math.abs(touch.clientX - gesture.x) > SLOP_PX || Math.abs(touch.clientY - gesture.y) > SLOP_PX) {
        clearTimeout(gesture.timer);
        if (gesture.gd.dataset.chosenDrag === "pan") {
          // Pan chosen on the toolbar: the chart follows the finger.
          gesture.mode = "pan";
          modebar(gesture.gd, "dragmode", ["pan"]);
          mouse("mousedown", gesture.target, gesture.x, gesture.y, 1);
        } else {
          gesture.mode = "scroll";  // a swipe: the browser scrolls the page
          return;
        }
      } else {
        return;
      }
    }
    if (gesture.mode === "scroll") return;
    event.stopPropagation();
    if (event.cancelable) event.preventDefault();  // the page stays still while the chart follows the finger
    if (gesture.mode === "pan") {
      mouse("mousemove", gesture.target, event.touches[0].clientX, event.touches[0].clientY, 1);
    } else if (gesture.mode === "scrub") {
      const touch = event.touches[0];
      const target = document.elementFromPoint(touch.clientX, touch.clientY);
      mouse("mousemove", target && target.closest && target.closest(".drag") ? target : gesture.target,
            touch.clientX, touch.clientY);
    } else if (gesture.mode === "axis-pinch" && event.touches.length >= 2) {
      zoomAxis(gesture, event.touches);
    } else if (gesture.mode === "pinch" && event.touches.length >= 2) {
      const c = centre(event.touches), s = spread(event.touches);
      mouse("mousemove", gesture.target, c.x, c.y, 1);  // pan with the fingers
      // Zoom by the change in finger spread. Plotly zooms at most ~10% per wheel step of 20.
      let delta = -200 * Math.log(s / gesture.spread);
      while (Math.abs(delta) > 0.5) {
        const step = Math.max(-20, Math.min(20, delta));
        wheel(gesture.target, c.x, c.y, step);
        delta -= step;
      }
      gesture.centre = c;
      gesture.spread = s;
    }
  }, { capture: true, passive: false });

  document.addEventListener("touchend", function (event) {
    if (!gesture || !event.isTrusted) return;
    if (gesture.mode === "pinch" || gesture.mode === "axis-pinch") {
      if (event.touches.length < 2) { endPinch(); gesture.mode = "done"; }
      return;
    }
    if (event.touches.length) return;
    clearTimeout(gesture.timer);
    if (gesture.mode === "pan") {
      const touch = event.changedTouches[0];
      mouse("mouseup", gesture.target, touch.clientX, touch.clientY, 0);
      gesture = null;
      return;
    }
    if (gesture.mode === "pending") {
      // A tap: show the values under the finger; a second tap soon after resets the zoom. Cancel the
      // browser's follow-up mouse events, whose mouseout would hide the values again.
      if (event.cancelable) event.preventDefault();
      const now = Date.now();
      if (lastTap && lastTap.gd === gesture.gd && now - lastTap.time < DOUBLE_TAP_MS) {
        modebar(gesture.gd, "zoom", ["reset", "auto"]);
        lastTap = null;
      } else {
        lastTap = { gd: gesture.gd, time: now };
      }
      hoverAt(gesture.target, gesture.x, gesture.y);
    }
    gesture = null;
  }, { capture: true, passive: false });
  document.addEventListener("touchcancel", function () { endPinch(); gesture = null; }, { capture: true, passive: true });
})();
"""

# Every chart zooms its dates (x) only; show_chart (analysis.chart_fit.fit_y_axes) fixes the y-axes and lists them
# in layout meta "fit_y". After each zoom or pan of the dates, each listed y-axis is refitted to the lowest and
# highest value drawn in view on it (candles by their low and high). An axis whose range the chart chose itself
# ("cap") stays inside that range and opens on it; volume bars ("volume") keep to the bottom quarter of their panel.
# When the dates are back where the chart opened (double-tap, reset button), every axis goes back to its first range.
CHART_FIT_Y_JS = """
(function () {
  const PAD = 0.05;
  function near(a, b) {
    const scale = Math.max(Math.abs(a[1] - a[0]), 1e-9);
    return Math.abs(a[0] - b[0]) / scale < 1e-6 && Math.abs(a[1] - b[1]) / scale < 1e-6;
  }
  function rangeOf(axis) { return axis.range.map(axis.r2l); }
  // A trace's x values in the axis's linear units (dates as milliseconds, categories as slots), worked out once.
  function xValues(gd, index, trace, xa) {
    const cache = gd._fitXCache || (gd._fitXCache = {});
    const hit = cache[index];
    if (hit && hit.source === trace.x && hit.axis === xa.type) return hit.values;
    const n = (trace.x || trace.y || trace.low || []).length, values = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const v = trace.x ? xa.d2l(trace.x[i]) : i;
      values[i] = typeof v === "number" ? v : NaN;
    }
    cache[index] = { source: trace.x, axis: xa.type, values: values };
    return values;
  }
  // Lowest and highest value (in the y-axis's linear units: log10 on a log axis) drawn in view on each fitted axis.
  function visibleExtents(gd, specs) {
    const layout = gd._fullLayout, out = {};
    gd._fullData.forEach(function (trace, index) {
      if (trace.visible !== true) return;
      const yname = (trace.yaxis || "y").replace("y", "yaxis"), xname = (trace.xaxis || "x").replace("x", "xaxis");
      if (!(yname in specs)) return;
      const xa = layout[xname], ya = layout[yname];
      if (!xa || !ya || !xa.range) return;
      const r = rangeOf(xa), lo = Math.min(r[0], r[1]), hi = Math.max(r[0], r[1]);
      const xs = xValues(gd, index, trace, xa);
      const candle = trace.type === "candlestick" || trace.type === "ohlc";
      const lows = candle ? trace.low : trace.y, highs = candle ? trace.high : trace.y;
      if (!lows) return;
      const e = out[yname] || (out[yname] = { low: Infinity, high: -Infinity, zero: false });
      if (trace.type === "bar" || trace.fill === "tozeroy") e.zero = true;
      for (let i = 0; i < xs.length; i++) {
        const x = xs[i];
        if (!(x >= lo && x <= hi)) continue;
        const a = ya.c2l(+lows[i]), b = ya.c2l(+highs[i]);
        if (isFinite(a) && a < e.low) e.low = a;
        if (isFinite(b) && b > e.high) e.high = b;
      }
    });
    return out;
  }
  function capture(gd) {
    const layout = gd._fullLayout, start = {};
    for (const key of Object.keys(layout)) {
      if (/^[xy]axis\\d*$/.test(key) && layout[key] && layout[key].range) start[key] = rangeOf(layout[key]).slice();
    }
    gd._fitStart = start;
    gd._fitMeta = gd.layout.meta;
    gd._fitXCache = {};
  }
  function fit(gd) {
    const meta = gd.layout && gd.layout.meta, specs = meta && meta.fit_y;
    if (!specs || !window.Plotly || !gd._fullLayout) return;
    if (gd._fitMeta !== meta) capture(gd);  // a new figure (the app reran)
    const layout = gd._fullLayout, start = gd._fitStart, extents = visibleExtents(gd, specs), update = {};
    // The x-axes of each fitted y-axis, from its traces.
    const xsOf = {};
    for (const trace of gd._fullData) {
      const yname = (trace.yaxis || "y").replace("y", "yaxis");
      (xsOf[yname] = xsOf[yname] || new Set()).add((trace.xaxis || "x").replace("x", "xaxis"));
    }
    for (const yname of Object.keys(specs)) {
      const ya = layout[yname], spec = specs[yname] || {};
      if (!ya || !ya.range) continue;
      const xnames = Array.from(xsOf[yname] || []);
      const atStart = xnames.every(function (x) { return layout[x] && start[x] && near(rangeOf(layout[x]), start[x]); });
      // Dates where the chart opened: an axis the chart sized itself keeps that size, and so does one opened on
      // every date (Plotly already fitted it). A chart that opens on part of its dates gets the other axes fitted.
      const openedOnAll = xnames.every(function (x) { const a = gd.layout[x]; return !a || !a.range; });
      let target = null;
      if (atStart && (spec.cap || spec.volume || openedOnAll)) {
        target = start[yname];
      } else {
        const e = extents[yname];
        if (!e || !isFinite(e.low) || !isFinite(e.high)) continue;
        let low = e.low, high = e.high;
        if (e.zero && ya.type !== "log") { low = Math.min(low, 0); high = Math.max(high, 0); }
        if (spec.volume) {
          target = [0, (high || 1) * 4];
        } else {
          const pad = (high - low) * PAD || Math.abs(high) * PAD || 1;
          low = e.zero && low === 0 ? 0 : low - pad;
          high = e.zero && high === 0 ? 0 : high + pad;
          if (spec.cap) {
            const capLow = Math.min(spec.cap[0], spec.cap[1]), capHigh = Math.max(spec.cap[0], spec.cap[1]);
            low = Math.max(low, capLow);
            high = Math.min(high, capHigh);
            if (!(high > low)) { low = capLow; high = capHigh; }
          }
          target = [low, high];
        }
      }
      if (!target) continue;
      const now = rangeOf(ya), reversed = now[0] > now[1];
      const wanted = reversed ? [target[1], target[0]] : target;
      if (!near(now, wanted)) update[yname + ".range"] = wanted.map(ya.l2r);
    }
    if (Object.keys(update).length) window.Plotly.relayout(gd, update);
  }
  function attach(gd) {
    if (gd.dataset.fitY || !gd.on) return;
    gd.dataset.fitY = "1";
    gd.on("plotly_relayout", function (event) {
      if (Object.keys(event).some(function (key) { return /^xaxis\\d*\\.(range|autorange)/.test(key) || /^xaxis\\d*$/.test(key); })) fit(gd);
    });
    // A new figure (first draw or a rerun) that opens on part of its dates: fit its axes to them.
    gd.on("plotly_afterplot", function () {
      const meta = gd.layout && gd.layout.meta;
      if (meta && meta.fit_y && gd._fitMeta !== meta) { capture(gd); fit(gd); }
    });
    const meta = gd.layout && gd.layout.meta;
    if (meta && meta.fit_y && gd._fullLayout) { capture(gd); fit(gd); }
  }
  function scan() { document.querySelectorAll(".js-plotly-plot").forEach(attach); }
  new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
  scan();
})();
"""

# Charts whose layout has meta={"axis_hover": True} drop Plotly's hover box, which covers the lines, for
# TradingView-style labels: each hovered value sits on its own y-axis (left or right) in the series' colour,
# with a dashed line across the panel at that level, and the date sits on the x-axis.
CHART_AXIS_HOVER_JS = """
(function () {
  const style = document.createElement("style");
  style.textContent = ".axis-hover .hoverlayer .legend, .axis-hover .hoverlayer .hovertext{display:none}"
    + ".axis-hover-overlay{position:absolute;inset:0;pointer-events:none;z-index:5}"
    + ".axis-hover-overlay div{position:absolute;font:11px sans-serif;line-height:16px;white-space:nowrap}";
  document.head.appendChild(style);

  function format(value, axis) {
    const digits = Math.abs(value) >= 1000 ? 0 : 2;
    return value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })
      + (axis.ticksuffix || "");
  }
  function pixel(axis, value) {
    const position = axis.type === "date" || axis.type === "category" ? axis.d2p(value) : axis.c2p(value);
    return axis._offset + position;
  }
  function box(overlay, css, text) {
    const el = document.createElement("div");
    Object.assign(el.style, css);
    if (text) el.textContent = text;
    overlay.appendChild(el);
    return el;
  }
  function show(gd, points, xvals) {
    let overlay = gd.querySelector(".axis-hover-overlay");
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.className = "axis-hover-overlay";
      gd.style.position = "relative";
      gd.appendChild(overlay);
    }
    overlay.textContent = "";
    const labels = { left: [], right: [] }, dates = new Map(), notes = [];
    for (const point of points) {
      const xa = point.xaxis, ya = point.yaxis;
      // A candle is labelled at its close.
      const value = typeof point.y === "number" ? point.y : point.close;
      if (!xa || !ya || typeof value !== "number" || !isFinite(value)) continue;
      const trace = point.fullData || point.data;  // the full trace has the default colours filled in
      // Earnings release markers show their text in a box near the top of the panel instead of a value label.
      if (trace.meta && trace.meta.earnings) {
        if (!xvals || !xvals.length || Math.abs(pixel(xa, point.x) - pixel(xa, xvals[0])) <= 12) notes.push(point);
        continue;
      }
      // A line with no point near the hovered date (a gap, or a line that starts later) gets no label: Plotly
      // offers its nearest point, which can be months away.
      if (xvals && xvals.length && trace.mode === "lines" && xa.type === "date"
          && Math.abs(pixel(xa, point.x) - pixel(xa, xvals[0])) > 6) continue;
      const bottom = ya._offset + ya._length;
      if (!dates.has(xa) || dates.get(xa).bottom < bottom) dates.set(xa, { bottom: bottom, x: point.x });
      const y = pixel(ya, value);
      if (y < ya._offset - 1 || y > ya._offset + ya._length + 1) continue;  // off the visible range
      let color = trace.type === "candlestick"
        ? ((point.close >= point.open ? trace.increasing : trace.decreasing) || {}).line
        : null;
      color = (color && color.color) || (trace.line && trace.line.color) || (trace.marker && trace.marker.color);
      if (Array.isArray(color)) color = color[point.pointNumber];
      color = typeof color === "string" ? color : "#555";
      // Markers (trade signals) carry their name, since the hover box with it is gone.
      const markersOnly = trace.mode && trace.mode.indexOf("markers") >= 0 && trace.mode.indexOf("lines") < 0;
      box(overlay, { left: xa._offset + "px", width: xa._length + "px", top: y + "px",
                     borderTop: "1px dashed " + color, opacity: "0.8" });
      labels[ya.side === "right" ? "right" : "left"].push(
        { y: y, xa: xa, color: color, text: (markersOnly && trace.name ? trace.name + " " : "") + format(value, ya) });
    }
    for (const side of ["left", "right"]) {
      // Keep labels on the same axis from overlapping: push each one at least a label's height down.
      labels[side].sort((a, b) => a.y - b.y);
      let last = -Infinity;
      for (const label of labels[side]) {
        const top = Math.max(label.y - 8, last + 17);
        last = top;
        const css = { top: top + "px", background: label.color, color: "#fff", padding: "0 4px", borderRadius: "2px" };
        if (side === "left") css.right = (gd.clientWidth - label.xa._offset + 1) + "px";
        else css.left = (label.xa._offset + label.xa._length + 1) + "px";
        box(overlay, css, label.text);
      }
    }
    for (const point of notes) {
      const xa = point.xaxis, ya = point.yaxis, x = pixel(xa, point.x), width = 210;
      const text = String(point.text || (point.data && point.data.name) || "");
      // Beside the line, on whichever side has room, kept inside the panel.
      const left = x + 8 + width <= xa._offset + xa._length ? x + 8 : Math.max(xa._offset, x - 8 - width);
      box(overlay, { left: left + "px", top: (ya._offset + 4) + "px", width: width + "px", whiteSpace: "pre-wrap",
                     background: "rgba(35,35,35,0.88)", color: "#fff", padding: "3px 6px", borderRadius: "3px",
                     boxSizing: "border-box" }, text);
      box(overlay, { left: x + "px", top: ya._offset + "px", height: ya._length + "px",
                     borderLeft: "1px dashed " + ((point.fullData.marker || {}).color || "#555") });
    }
    for (const [xa, date] of dates) {
      const x = pixel(xa, date.x);
      // The time too on intraday charts.
      const full = String(date.x), time = full.slice(11, 16);
      // A category axis (one slot per bar) already names its bars, e.g. "Oct 28 09:35".
      const text = xa.type === "category" ? full : time && time !== "00:00" ? full.slice(0, 10) + " " + time : full.slice(0, 10);
      const width = text.length > 10 ? 108 : 72;
      box(overlay, { top: (date.bottom + 2) + "px", left: (x - width / 2) + "px", width: width + "px",
                     textAlign: "center", background: "#444", color: "#fff", borderRadius: "2px" }, text);
    }
  }
  function attach(gd) {
    if (gd.dataset.axisHover || !gd.layout || !gd.layout.meta || !gd.layout.meta.axis_hover || !gd.on) return;
    gd.dataset.axisHover = "1";
    gd.classList.add("axis-hover");
    gd.on("plotly_hover", function (event) { show(gd, event.points, event.xvals); });
    // A tap on a touch screen hovers without a hover event; Plotly reports it as a click instead.
    gd.on("plotly_click", function (event) { show(gd, event.points, event.xvals); });
    gd.on("plotly_unhover", function () { const o = gd.querySelector(".axis-hover-overlay"); if (o) o.textContent = ""; });
    gd.on("plotly_relayout", function () { const o = gd.querySelector(".axis-hover-overlay"); if (o) o.textContent = ""; });
  }
  function scan() { document.querySelectorAll(".js-plotly-plot").forEach(attach); }
  // Touch screens get no unhover: clear the labels when the next touch starts (a tap draws them again).
  document.addEventListener("touchstart", function () {
    document.querySelectorAll(".axis-hover-overlay").forEach(function (o) { o.textContent = ""; });
  }, { capture: true, passive: true });
  new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
  scan();
})();
"""

# Tables (st.dataframe / st.data_editor) keep their column order. Streamlit's grid lets a user drag a column
# header to another place and has no option to turn that off, so when a press that began on a header row ends
# after moving (or the browser cancels it because the finger scrolled), the script swallows the release, hovers
# the pointer back over the starting column so the move is onto itself, and releases it outside the grid.
# A plain tap on a header (sorting), resizing a column at its edge and cell selection are untouched.
TABLE_COLUMN_LOCK_JS = """
(function () {
  const HEADER_PX = 36;  // table header row height
  let press = null;
  let swallowClick = false;
  function gridCanvas(target) {
    const grid = target instanceof Element ? target.closest("[data-testid='stDataFrame']") : null;
    return grid ? grid.querySelector("canvas") : null;
  }
  function fire(target, type, x, y) {
    target.dispatchEvent(new PointerEvent(type, {
      bubbles: true, cancelable: true, composed: true, pointerId: 1, pointerType: "mouse", isPrimary: true,
      clientX: x, clientY: y, button: 0, buttons: type === "pointerup" ? 0 : 1,
    }));
  }
  document.addEventListener("pointerdown", function (e) {
    press = null;
    const canvas = gridCanvas(e.target);
    if (!canvas || e.button !== 0) return;
    const box = canvas.getBoundingClientRect();
    if (e.clientY - box.top > HEADER_PX) return;            // a cell, not the header
    if (canvas.style.cursor === "col-resize") return;       // dragging a column edge resizes it
    press = { canvas: canvas, x: e.clientX, y: e.clientY, moved: false };
  }, true);
  document.addEventListener("pointermove", function (e) {
    if (press && e.isTrusted && Math.abs(e.clientX - press.x) + Math.abs(e.clientY - press.y) > 4) press.moved = true;
  }, true);
  function snapBack(p) {
    fire(p.canvas, "pointermove", p.x, p.y);
    // Release once the grid has re-rendered with the pointer back over the starting column.
    setTimeout(function () { fire(p.canvas, "pointerup", -9999, -9999); }, 80);
  }
  document.addEventListener("pointerup", function (e) {
    if (!press || !e.isTrusted) return;
    const p = press;
    press = null;
    if (!p.moved) return;
    e.stopPropagation();
    swallowClick = true;
    snapBack(p);
  }, true);
  // On a touch screen a swipe that starts on a header scrolls instead, and the browser cancels the press.
  // The grid never hears it end and keeps the column "picked up": the next tap on another column would drop it there.
  document.addEventListener("pointercancel", function (e) {
    if (!press || !e.isTrusted) return;
    const p = press;
    press = null;
    snapBack(p);
  }, true);
  document.addEventListener("click", function (e) {
    if (swallowClick && e.isTrusted) { swallowClick = false; e.stopPropagation(); }
  }, true);
})();
"""

# Streamlit Community Cloud wraps the app in its own page, whose install
# manifest and home-screen icon are Streamlit's. Both pages share an origin,
# so point the wrapper's manifest and apple-touch-icon at our favicon.
# The manifest's id is ours too: see the comment on it below.
components.html(
    """
<script>
(function () {
  try {
    const top = window.top.document;
    const favicon = window.parent.document.querySelector("link[rel~='icon']");
    if (!favicon || top.documentElement.dataset.backtestIcon) return;
    const icon = favicon.href;
    const origin = window.top.location.origin;
    const manifest = {
      // An id of our own, so Chrome treats Streamlit's manifest (which it sometimes reads when it checks an
      // installed app for updates, before this script has run) as a different app instead of an update
      // that swaps our icon and splash screen back to Streamlit's.
      id: "/backtest-app",
      name: "Trade Strategy Backtest",
      short_name: "Backtest",
      icons: [{ src: icon, type: "image/png", sizes: "512x512", purpose: "any" }],
      start_url: origin + "/",
      scope: origin + "/",
      display: "standalone",
      theme_color: "#000000",
      background_color: "#000000",
    };
    const blob = new Blob([JSON.stringify(manifest)], { type: "application/manifest+json" });
    let link = top.querySelector("link[rel='manifest']");
    if (!link) {
      link = top.createElement("link");
      link.rel = "manifest";
      top.head.appendChild(link);
    }
    link.href = URL.createObjectURL(blob);
    top.querySelectorAll("link[rel='apple-touch-icon']").forEach((el) => (el.href = icon));
    top.documentElement.dataset.backtestIcon = "1";
  } catch (e) {
    // Not same-origin (e.g. embedded elsewhere): keep the defaults.
  }
})();
(function () {
  const app = window.parent.document;
  if (app.documentElement.dataset.chartTouch) return;
  app.documentElement.dataset.chartTouch = "1";
  const script = app.createElement("script");
  script.textContent = %s;
  app.head.appendChild(script);
})();
</script>
""" % json.dumps(CHART_TOUCH_JS + CHART_AXIS_HOVER_JS + CHART_FIT_Y_JS + TABLE_COLUMN_LOCK_JS),
    height=0,
)

st.title("Trade Strategy Backtester")

# Streamlit sizes a setting's help tooltip to its text and can place it past the right edge of a phone
# screen; keep it inside the screen and let the text wrap.
st.markdown(
    "<style>div[data-baseweb='tooltip'], div[data-baseweb='popover'] {max-width: calc(100vw - 1rem)}"
    "[data-testid='stTooltipContent'] {max-width: calc(100vw - 2rem); white-space: normal; "
    "overflow-wrap: anywhere}</style>",
    unsafe_allow_html=True,
)


# ============================================================
# Password protection
# ============================================================

def check_password():
    """Return True after the correct password has been entered."""

    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    password = st.text_input("Enter Password", type="password")

    if password:
        try:
            correct_password = st.secrets["password"]
        except KeyError:
            st.error("The password is not configured in Streamlit secrets.")
            return False

        if password == correct_password:
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("Password incorrect")

    return False


# ============================================================
# Indicators and signal conditions
# ============================================================

compute_ma = moving_average
build_condition = rule_condition


# ============================================================
# Price-data loader
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def load_data(symbol, start_date_str, end_date_str, timeframe):
    """Download prices. Yahoo Finance treats the end date as exclusive."""

    start_timestamp = pd.Timestamp(start_date_str).normalize()
    selected_end_timestamp = pd.Timestamp(end_date_str).normalize()
    download_end_timestamp = selected_end_timestamp + pd.Timedelta(days=1)

    try:
        price_data = yf.download(
            symbol,
            start=start_timestamp.strftime("%Y-%m-%d"),
            end=download_end_timestamp.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if price_data is None or price_data.empty:
            return pd.DataFrame()

        price_data = price_data.copy()

        if isinstance(price_data.columns, pd.MultiIndex):
            final_level = price_data.columns.get_level_values(-1)
            if symbol in final_level:
                price_data = price_data.xs(symbol, axis=1, level=-1)
            else:
                price_data.columns = price_data.columns.get_level_values(0)

        price_data.index = pd.to_datetime(price_data.index)
        if getattr(price_data.index, "tz", None) is not None:
            price_data.index = price_data.index.tz_localize(None)

        price_data = price_data.sort_index()
        price_data = price_data.loc[~price_data.index.duplicated(keep="last")]
        price_data = price_data.loc[
            (price_data.index >= start_timestamp)
            & (price_data.index <= selected_end_timestamp)
        ]

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        if any(column not in price_data.columns for column in required_columns):
            return pd.DataFrame()

        price_data = price_data[required_columns].copy()
        price_data = price_data.dropna(subset=["Close"])

        if timeframe == "Weekly":
            price_data = (
                price_data.resample("W-FRI")
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
                .dropna(subset=["Close"])
            )
            price_data = price_data.loc[price_data.index <= selected_end_timestamp]

        elif timeframe == "Monthly":
            price_data = (
                price_data.resample("ME")
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
                .dropna(subset=["Close"])
            )
            price_data = price_data.loc[price_data.index <= selected_end_timestamp]

        return price_data

    except Exception:
        return pd.DataFrame()


# ============================================================
# Earnings loader
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_earnings_dates(symbol):
    """Yahoo's earnings dates (past and scheduled), as Yahoo returns them; raises when empty so failures are not cached."""

    earnings = yf.Ticker(symbol).get_earnings_dates(limit=100)
    if earnings is None or earnings.empty:
        raise ValueError(f"Yahoo returned no earnings dates for {symbol}")
    return earnings


@st.cache_data(ttl=3600, show_spinner=False)
def load_earnings_data(symbol, start_date_str, end_date_str):
    """Retrieve historical earnings dates and EPS values."""

    try:
        earnings = fetch_earnings_dates(symbol)
        earnings = earnings.copy().reset_index()
        date_column = earnings.columns[0]
        earnings[date_column] = pd.to_datetime(
            earnings[date_column], errors="coerce", utc=True
        ).dt.tz_convert(None)
        earnings = earnings.rename(columns={date_column: "Earnings_Date"})

        for column in ["EPS Estimate", "Reported EPS", "Surprise(%)"]:
            if column in earnings.columns:
                earnings[column] = pd.to_numeric(earnings[column], errors="coerce")

        start_timestamp = pd.Timestamp(start_date_str).normalize()
        end_exclusive = pd.Timestamp(end_date_str).normalize() + pd.Timedelta(days=1)
        earnings = earnings.loc[
            (earnings["Earnings_Date"] >= start_timestamp)
            & (earnings["Earnings_Date"] < end_exclusive)
        ]
        earnings = earnings.dropna(subset=["Earnings_Date"])
        earnings = earnings.sort_values("Earnings_Date")
        earnings = earnings.drop_duplicates(subset=["Earnings_Date"], keep="last")
        return earnings.set_index("Earnings_Date")

    except Exception:
        return pd.DataFrame()


# ============================================================
# Quarterly fundamentals loader
# ============================================================

def first_available_statement(stock, attribute_names):
    """Return the first non-empty financial statement available."""

    for attribute_name in attribute_names:
        try:
            statement = getattr(stock, attribute_name)
            if callable(statement):
                statement = statement()
            if statement is not None and not statement.empty:
                return statement.copy()
        except Exception:
            continue
    return pd.DataFrame()


def extract_statement_row(statement, candidate_names):
    """Extract the first available statement row using name fallbacks."""

    if statement is None or statement.empty:
        return pd.Series(dtype=float)

    normalized_lookup = {
        str(index_value).strip().lower(): index_value
        for index_value in statement.index
    }

    for candidate in candidate_names:
        original_name = normalized_lookup.get(candidate.strip().lower())
        if original_name is not None:
            values = statement.loc[original_name]
            if isinstance(values, pd.DataFrame):
                values = values.iloc[0]
            values = pd.to_numeric(values, errors="coerce")
            values.index = pd.to_datetime(values.index, errors="coerce")
            values = values.loc[~values.index.isna()]
            return values.sort_index()

    return pd.Series(dtype=float)


def load_yahoo_quarterly_fundamentals(symbol):
    """Load the recent quarters available through Yahoo Finance."""

    try:
        stock = yf.Ticker(symbol)
        income_statement = first_available_statement(
            stock,
            ["quarterly_income_stmt", "quarterly_financials"],
        )
        cash_flow_statement = first_available_statement(
            stock,
            ["quarterly_cashflow", "quarterly_cash_flow"],
        )

        revenue = extract_statement_row(
            income_statement,
            ["Total Revenue", "Operating Revenue", "Revenue"],
        )
        operating_expense = extract_statement_row(
            income_statement,
            ["Operating Expense", "Total Operating Expenses", "Operating Expenses"],
        )
        total_expense = extract_statement_row(
            income_statement,
            ["Total Expenses", "Costs And Expenses", "Total Costs And Expenses"],
        )
        operating_income = extract_statement_row(
            income_statement,
            ["Operating Income", "Operating Income Loss"],
        )
        net_income = extract_statement_row(
            income_statement,
            [
                "Net Income",
                "Net Income Common Stockholders",
                "Net Income Including Noncontrolling Interests",
            ],
        )
        capex = extract_statement_row(
            cash_flow_statement,
            [
                "Capital Expenditure",
                "Capital Expenditures",
                "Purchase Of PPE",
                "Investments In Property Plant And Equipment",
            ],
        ).abs()

        all_dates = revenue.index.union(operating_expense.index)
        all_dates = all_dates.union(total_expense.index).union(capex.index)
        all_dates = all_dates.union(net_income.index).sort_values()
        if len(all_dates) == 0:
            return pd.DataFrame()

        result = pd.DataFrame(index=all_dates)
        result["Revenue"] = revenue.reindex(all_dates)
        result["Operating_Expense"] = operating_expense.reindex(all_dates)
        result["Total_Expense"] = total_expense.reindex(all_dates)
        result["CapEx"] = capex.reindex(all_dates)
        result["Net_Income"] = net_income.reindex(all_dates)

        # If Yahoo omits total expenses, derive it as revenue minus operating income.
        derived_total_expense = revenue.reindex(all_dates) - operating_income.reindex(all_dates)
        result["Total_Expense"] = result["Total_Expense"].combine_first(
            derived_total_expense
        )
        return result.dropna(how="all")

    except Exception:
        return pd.DataFrame()


def get_sec_user_agent():
    """Return an SEC-compliant identifying User-Agent."""

    try:
        configured_value = st.secrets.get("SEC_USER_AGENT", "")
    except Exception:
        configured_value = ""

    return (
        configured_value
        or os.getenv("SEC_USER_AGENT", "")
        or "StreamlitBacktester/1.0 kelvin@example.com"
    )


def sec_request_json(url):
    """Read JSON from an SEC endpoint with an identifying User-Agent."""

    request = Request(
        url,
        headers={
            "User-Agent": get_sec_user_agent(),
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        },
    )

    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sec_submissions(cik):
    """A company's SEC filing index (the latest 1,000 filings), downloaded once a day."""

    return sec_request_json(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")


def load_earnings_releases(symbol):
    """Earnings releases from Yahoo, plus SEC 8-K (item 2.02) releases Yahoo is missing, by US/Eastern time.

    Either source may fail (Yahoo rate limits, foreign filers have no 8-K); what is left is returned.
    """

    try:
        yahoo = yahoo_releases(fetch_earnings_dates(symbol))
    except Exception:
        yahoo = None
    try:
        cik = get_sec_cik(symbol)
        sec = sec_releases(fetch_sec_submissions(cik)) if cik else None
    except Exception:
        sec = None
    return merge_releases(yahoo_releases(None) if yahoo is None else yahoo, sec)


def next_earnings(symbol, releases=None):
    """``(past releases, next release)`` for a ticker; see :func:`past_and_next`."""

    releases = load_earnings_releases(symbol) if releases is None else releases
    now = pd.Timestamp.now(tz=EASTERN).tz_localize(None)
    return past_and_next(releases, now, load_fundamental_snapshot(symbol))


@st.cache_data(ttl=86400, show_spinner=False)
def load_sec_ticker_map():
    """SEC ticker-to-CIK map (about 1 MB), downloaded once a day."""

    return sec_request_json("https://www.sec.gov/files/company_tickers.json")


def get_sec_cik(symbol):
    """Resolve a US-listed ticker to its SEC CIK."""

    ticker_map = load_sec_ticker_map()

    requested_symbol = symbol.upper().replace(".", "-")

    for item in ticker_map.values():
        sec_symbol = str(item.get("ticker", "")).upper()
        if sec_symbol == requested_symbol:
            return int(item["cik_str"])

    return None


def get_sec_fact_entries(company_facts, tag):
    """Return the USD fact entries reported under one SEC tag."""

    us_gaap_facts = company_facts.get("facts", {}).get("us-gaap", {})
    return us_gaap_facts.get(tag, {}).get("units", {}).get("USD", [])


def combine_sec_tags(company_facts, candidate_tags, extract):
    """Quarterly series merged across tags, earlier tags taking priority.

    Companies switch tags over time (NVIDIA reported revenue under
    RevenueFromContractWithCustomerExcludingAssessedTax until 2022, then
    Revenues), so taking only the first tag with data drops whole years.
    """

    combined = pd.Series(dtype=float)
    for tag in candidate_tags:
        series = extract(get_sec_fact_entries(company_facts, tag))
        if series.empty:
            continue
        # Treat quarter ends a few days apart as the same quarter.
        new_dates = [
            timestamp for timestamp in series.index
            if combined.empty or np.abs(combined.index - timestamp).min() > pd.Timedelta(days=14)
        ]
        combined = pd.concat([combined, series.loc[new_dates]]).sort_index()
    return combined


def prepare_sec_duration_entries(entries):
    """Clean SEC duration facts and keep quarterly/annual filing observations."""

    records = []

    for entry in entries:
        if entry.get("form") not in {"10-Q", "10-K", "20-F", "40-F"}:
            continue

        start_date = pd.to_datetime(entry.get("start"), errors="coerce")
        end_date = pd.to_datetime(entry.get("end"), errors="coerce")
        filed_date = pd.to_datetime(entry.get("filed"), errors="coerce")
        value = pd.to_numeric(entry.get("val"), errors="coerce")
        fiscal_year = pd.to_numeric(entry.get("fy"), errors="coerce")
        fiscal_period = str(entry.get("fp", "")).upper()

        if (
            pd.isna(start_date)
            or pd.isna(end_date)
            or pd.isna(value)
            or pd.isna(fiscal_year)
        ):
            continue

        duration_days = int((end_date - start_date).days) + 1

        if duration_days < 60 or duration_days > 390:
            continue

        records.append(
            {
                "Start_Date": start_date,
                "End_Date": end_date,
                "Filed_Date": filed_date,
                "Value": float(value),
                "Fiscal_Year": int(fiscal_year),
                "Fiscal_Period": fiscal_period,
                "Duration_Days": duration_days,
                "Form": entry.get("form"),
            }
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values(
        ["Fiscal_Year", "End_Date", "Duration_Days", "Filed_Date"]
    )


def select_latest_shortest(group, minimum_days, maximum_days):
    """Select the latest-filed value among the shortest matching durations."""

    candidates = group.loc[
        (group["Duration_Days"] >= minimum_days)
        & (group["Duration_Days"] <= maximum_days)
    ].copy()

    if candidates.empty:
        return None

    shortest_duration = candidates["Duration_Days"].min()
    candidates = candidates.loc[
        candidates["Duration_Days"] <= shortest_duration + 7
    ]
    candidates = candidates.sort_values("Filed_Date")
    return candidates.iloc[-1]


def select_latest_longest(group, minimum_days, maximum_days):
    """Select the latest-filed value among the longest matching durations."""

    candidates = group.loc[
        (group["Duration_Days"] >= minimum_days)
        & (group["Duration_Days"] <= maximum_days)
    ].copy()

    if candidates.empty:
        return None

    longest_duration = candidates["Duration_Days"].max()
    candidates = candidates.loc[
        candidates["Duration_Days"] >= longest_duration - 7
    ]
    candidates = candidates.sort_values("Filed_Date")
    return candidates.iloc[-1]


def extract_sec_fiscal_quarters(company_facts, candidate_tags):
    """Fiscal-quarter values from SEC duration facts, merged across ``candidate_tags``."""

    return combine_sec_tags(company_facts, candidate_tags, fiscal_quarters_from_entries)


def fiscal_quarters_from_entries(entries):
    """
    Build fiscal-quarter values from SEC duration facts.

    Q1-Q3 use standalone approximately three-month facts whenever available.
    Q4 is derived from the full fiscal-year value less Q1, Q2 and Q3. This
    avoids relying on CYyyyyQn frame labels, which omit many fiscal quarters.
    """

    facts = prepare_sec_duration_entries(entries)

    if facts.empty:
        return pd.Series(dtype=float)

    quarter_records = []

    for fiscal_year, fiscal_year_group in facts.groupby("Fiscal_Year"):
        selected_quarters = {}

        for quarter_name in ["Q1", "Q2", "Q3"]:
            period_group = fiscal_year_group.loc[
                fiscal_year_group["Fiscal_Period"] == quarter_name
            ]
            selected = select_latest_shortest(period_group, 60, 120)

            if selected is not None:
                selected_quarters[quarter_name] = selected
                quarter_records.append(
                    (
                        selected["End_Date"],
                        selected["Value"],
                        fiscal_year,
                        quarter_name,
                    )
                )

        annual_group = fiscal_year_group.loc[
            fiscal_year_group["Fiscal_Period"] == "FY"
        ]
        annual = select_latest_longest(annual_group, 300, 390)

        if annual is not None and all(
            quarter_name in selected_quarters
            for quarter_name in ["Q1", "Q2", "Q3"]
        ):
            q4_value = annual["Value"] - sum(
                selected_quarters[quarter_name]["Value"]
                for quarter_name in ["Q1", "Q2", "Q3"]
            )

            # Accept ordinary positive/negative accounting values but reject
            # implausibly large derivation errors caused by mixed contexts.
            comparison_scale = max(abs(annual["Value"]), 1.0)
            if abs(q4_value) <= comparison_scale * 1.5:
                quarter_records.append(
                    (annual["End_Date"], q4_value, fiscal_year, "Q4")
                )

    if not quarter_records:
        return pd.Series(dtype=float)

    quarter_df = pd.DataFrame(
        quarter_records,
        columns=["End_Date", "Value", "Fiscal_Year", "Fiscal_Quarter"],
    )
    quarter_df = quarter_df.sort_values(
        ["End_Date", "Fiscal_Year", "Fiscal_Quarter"]
    )
    quarter_df = quarter_df.drop_duplicates(
        subset=["End_Date"],
        keep="last",
    )

    return pd.Series(
        quarter_df["Value"].to_numpy(),
        index=pd.DatetimeIndex(quarter_df["End_Date"]),
        dtype=float,
    ).sort_index()


def extract_sec_cumulative_quarters(company_facts, candidate_tags):
    """Fiscal quarters from cumulative SEC facts, merged across ``candidate_tags``."""

    return combine_sec_tags(company_facts, candidate_tags, cumulative_quarters_from_entries)


def cumulative_quarters_from_entries(entries):
    """
    Convert cumulative SEC cash-flow facts into individual fiscal quarters.

    This is used for CapEx because 10-Q cash-flow statements commonly report
    year-to-date values rather than standalone Q2 and Q3 amounts.
    """

    facts = prepare_sec_duration_entries(entries)

    if facts.empty:
        return pd.Series(dtype=float)

    quarter_records = []

    for fiscal_year, fiscal_year_group in facts.groupby("Fiscal_Year"):
        cumulative = {}

        period_ranges = {
            "Q1": (60, 120),
            "Q2": (120, 220),
            "Q3": (200, 310),
            "FY": (300, 390),
        }

        for period_name, (minimum_days, maximum_days) in period_ranges.items():
            period_group = fiscal_year_group.loc[
                fiscal_year_group["Fiscal_Period"] == period_name
            ]
            selected = select_latest_longest(
                period_group,
                minimum_days,
                maximum_days,
            )
            if selected is not None:
                cumulative[period_name] = selected

        if "Q1" in cumulative:
            quarter_records.append(
                (cumulative["Q1"]["End_Date"], cumulative["Q1"]["Value"])
            )

        if "Q2" in cumulative and "Q1" in cumulative:
            quarter_records.append(
                (
                    cumulative["Q2"]["End_Date"],
                    cumulative["Q2"]["Value"] - cumulative["Q1"]["Value"],
                )
            )

        if "Q3" in cumulative and "Q2" in cumulative:
            quarter_records.append(
                (
                    cumulative["Q3"]["End_Date"],
                    cumulative["Q3"]["Value"] - cumulative["Q2"]["Value"],
                )
            )

        if "FY" in cumulative and "Q3" in cumulative:
            quarter_records.append(
                (
                    cumulative["FY"]["End_Date"],
                    cumulative["FY"]["Value"] - cumulative["Q3"]["Value"],
                )
            )

    if not quarter_records:
        return pd.Series(dtype=float)

    quarter_df = pd.DataFrame(
        quarter_records,
        columns=["End_Date", "Value"],
    ).sort_values("End_Date")
    quarter_df = quarter_df.drop_duplicates(subset=["End_Date"], keep="last")

    return pd.Series(
        quarter_df["Value"].abs().to_numpy(),
        index=pd.DatetimeIndex(quarter_df["End_Date"]),
        dtype=float,
    ).sort_index()


def load_sec_quarterly_fundamentals(symbol):
    """Load extended quarterly financial history from SEC Company Facts."""

    try:
        cik = get_sec_cik(symbol)
        if cik is None:
            return pd.DataFrame(), "Ticker is not mapped to an SEC CIK."

        company_facts = sec_request_json(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
        )

        revenue = extract_sec_fiscal_quarters(
            company_facts,
            [
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet",
                # Banks report net revenue (after interest expense) here.
                "RevenuesNetOfInterestExpense",
                "Revenues",
            ],
        )
        operating_expense = extract_sec_fiscal_quarters(
            company_facts,
            ["OperatingExpenses"],
        )
        total_expense = extract_sec_fiscal_quarters(
            company_facts,
            ["CostsAndExpenses"],
        )
        operating_income = extract_sec_fiscal_quarters(
            company_facts,
            ["OperatingIncomeLoss"],
        )
        capex = extract_sec_cumulative_quarters(
            company_facts,
            [
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "PaymentsForAdditionsToPropertyPlantAndEquipment",
            ],
        )
        net_income = extract_sec_fiscal_quarters(
            company_facts,
            ["NetIncomeLoss", "ProfitLoss"],
        )

        all_dates = revenue.index.union(operating_expense.index)
        all_dates = all_dates.union(total_expense.index).union(capex.index)
        all_dates = all_dates.union(net_income.index).sort_values()

        if len(all_dates) == 0:
            return pd.DataFrame(), "SEC returned no usable quarterly duration facts."

        result = pd.DataFrame(index=all_dates)
        result["Revenue"] = revenue.reindex(all_dates)
        result["Operating_Expense"] = operating_expense.reindex(all_dates)
        result["Total_Expense"] = total_expense.reindex(all_dates)
        result["CapEx"] = capex.reindex(all_dates)
        result["Net_Income"] = net_income.reindex(all_dates)

        derived_total_expense = (
            revenue.reindex(all_dates) - operating_income.reindex(all_dates)
        )
        result["Total_Expense"] = result["Total_Expense"].combine_first(
            derived_total_expense
        )

        return result.dropna(how="all"), ""

    except Exception as error:
        return pd.DataFrame(), f"{type(error).__name__}: {error}"


@st.cache_data(ttl=21600, show_spinner=False)
def load_quarterly_fundamentals(
    symbol,
    start_date_str,
    end_date_str,
    cache_version="2026-10-12-v6",
):
    """Combine long SEC history with recent Yahoo Finance statements."""

    del cache_version

    sec_data, sec_error = load_sec_quarterly_fundamentals(symbol)
    yahoo_data = load_yahoo_quarterly_fundamentals(symbol)

    if sec_data.empty and yahoo_data.empty:
        return pd.DataFrame(), "No SEC or Yahoo quarterly fundamentals were returned.", "None"

    if sec_data.empty:
        fundamentals = yahoo_data.copy()
        source = "Yahoo Finance only"
    elif yahoo_data.empty:
        fundamentals = sec_data.copy()
        source = "SEC Company Facts only"
    else:
        # Quarter-end dates can differ by a few days between data suppliers.
        # Keep SEC as the historical base, then use Yahoo only for dates not
        # already represented within 14 days of an SEC fiscal quarter end.
        fundamentals = sec_data.copy()

        for yahoo_date, yahoo_row in yahoo_data.iterrows():
            if fundamentals.empty:
                fundamentals.loc[yahoo_date] = yahoo_row
                continue

            # Convert the Yahoo date into a timezone-naive Timestamp.
            yahoo_timestamp = pd.Timestamp(yahoo_date)
            
            if yahoo_timestamp.tzinfo is not None:
                yahoo_timestamp = yahoo_timestamp.tz_localize(None)
            
            # Ensure the SEC fundamentals index contains timezone-naive timestamps.
            fundamentals.index = pd.DatetimeIndex(
                pd.to_datetime(
                    fundamentals.index,
                    errors="coerce"
                )
            )
            
            # Remove invalid dates.
            fundamentals = fundamentals.loc[
                ~fundamentals.index.isna()
            ]
            
            if fundamentals.index.tz is not None:
                fundamentals.index = (
                    fundamentals.index.tz_localize(None)
                )
            
            # Calculate distance from every SEC quarter date to the Yahoo date.
            # The index of this Series must contain the actual quarter dates.
            date_distances = pd.Series(
                np.abs(
                    fundamentals.index
                    - yahoo_timestamp
                ),
                index=fundamentals.index
            )
            
            # idxmin now returns the actual nearest Timestamp.
            nearest_date = pd.Timestamp(
                date_distances.idxmin()
            )
            
            nearest_distance_days = abs(
                (
                    nearest_date
                    - yahoo_timestamp
                ).days
            )
            
            # Match Yahoo and SEC records when their quarter-end dates
            # differ by no more than 14 days.
            if nearest_distance_days <= 14:
                for column in yahoo_data.columns:
                    yahoo_value = yahoo_row.get(column)
            
                    if pd.notna(yahoo_value):
                        fundamentals.loc[
                            nearest_date,
                            column
                        ] = yahoo_value
            
            else:
                # No nearby SEC quarter exists, so add the Yahoo row.
                fundamentals.loc[
                    yahoo_timestamp
                ] = yahoo_row

        source = "SEC Company Facts plus Yahoo Finance"

    start_timestamp = pd.Timestamp(start_date_str).normalize()
    end_exclusive = pd.Timestamp(end_date_str).normalize() + pd.Timedelta(days=1)
    fundamentals = fundamentals.loc[
        (fundamentals.index >= start_timestamp)
        & (fundamentals.index < end_exclusive)
    ]
    fundamentals = fundamentals.sort_index()
    fundamentals = fundamentals.loc[
        ~fundamentals.index.duplicated(keep="last")
    ]

    return fundamentals.dropna(how="all"), sec_error, source


def choose_fundamental_scale(fundamentals_df):
    """Choose USD millions or billions based on the largest absolute value."""

    if fundamentals_df is None or fundamentals_df.empty:
        return 1e9, "USD billions"
    numeric_values = fundamentals_df.select_dtypes(include=[np.number])
    if numeric_values.empty:
        return 1e9, "USD billions"
    maximum_value = numeric_values.abs().max().max()
    if pd.isna(maximum_value) or maximum_value >= 1e9:
        return 1e9, "USD billions"
    return 1e6, "USD millions"

def align_fundamentals_to_earnings_dates(
    fundamentals_df,
    earnings_df,
    maximum_days=120,
):
    """
    Align fiscal quarter-end fundamentals with earnings announcement dates.

    Fundamentals from SEC and Yahoo statements normally use fiscal
    quarter-end dates, while the EPS plot uses earnings announcement
    dates. This function maps each fundamentals quarter to the first
    earnings announcement occurring after that quarter-end date.

    If no suitable earnings announcement is found, the original
    fiscal quarter-end date is retained.
    """

    if fundamentals_df is None or fundamentals_df.empty:
        return pd.DataFrame()

    aligned_fundamentals = (
        fundamentals_df
        .copy()
        .sort_index()
    )

    aligned_fundamentals.index = pd.DatetimeIndex(
        pd.to_datetime(
            aligned_fundamentals.index,
            errors="coerce",
        )
    )

    aligned_fundamentals = aligned_fundamentals.loc[
        ~aligned_fundamentals.index.isna()
    ]

    if aligned_fundamentals.index.tz is not None:
        aligned_fundamentals.index = (
            aligned_fundamentals.index.tz_localize(None)
        )

    # Keep the original fiscal quarter-end date for hover information.
    aligned_fundamentals["Fiscal_Quarter_End"] = (
        aligned_fundamentals.index
    )

    if earnings_df is None or earnings_df.empty:
        aligned_fundamentals["Display_Date"] = (
            aligned_fundamentals.index
        )

        return aligned_fundamentals.set_index(
            "Display_Date"
        )

    earnings_dates = pd.DatetimeIndex(
        pd.to_datetime(
            earnings_df.index,
            errors="coerce",
        )
    )

    earnings_dates = earnings_dates[
        ~earnings_dates.isna()
    ]

    if earnings_dates.tz is not None:
        earnings_dates = (
            earnings_dates.tz_localize(None)
        )

    earnings_dates = (
        earnings_dates
        .sort_values()
        .unique()
    )

    display_dates = []

    for fiscal_quarter_end in aligned_fundamentals.index:
        # An earnings announcement should normally occur after
        # the corresponding fiscal quarter has ended.
        later_earnings_dates = earnings_dates[
            earnings_dates >= fiscal_quarter_end
        ]

        if len(later_earnings_dates) == 0:
            display_dates.append(
                fiscal_quarter_end
            )
            continue

        matching_earnings_date = pd.Timestamp(
            later_earnings_dates[0]
        )

        date_difference_days = (
            matching_earnings_date
            - fiscal_quarter_end
        ).days

        if (
            0
            <= date_difference_days
            <= maximum_days
        ):
            display_dates.append(
                matching_earnings_date
            )
        else:
            display_dates.append(
                fiscal_quarter_end
            )

    aligned_fundamentals["Display_Date"] = (
        pd.DatetimeIndex(display_dates)
    )

    aligned_fundamentals = (
        aligned_fundamentals
        .set_index("Display_Date")
        .sort_index()
    )

    return aligned_fundamentals


# ============================================================
# EPS and valuation calculations
# ============================================================

def build_valuation_data(price_df, earnings_history_df):
    """Calculate TTM EPS, trailing P/E and an estimate-based P/E proxy."""

    valuation = pd.DataFrame(index=price_df.index)
    valuation["Close"] = pd.to_numeric(price_df["Close"], errors="coerce")
    valuation["TTM_Reported_EPS"] = np.nan
    valuation["TTM_Estimated_EPS"] = np.nan
    valuation["Trailing_PE"] = np.nan
    valuation["Estimate_Based_PE_Proxy"] = np.nan

    if earnings_history_df is None or earnings_history_df.empty:
        valuation["PE_Reconstruction_Error"] = np.nan
        return valuation, pd.DataFrame()

    earnings = earnings_history_df.copy().sort_index()

    for column in ["Reported EPS", "EPS Estimate"]:
        if column in earnings.columns:
            earnings[column] = pd.to_numeric(earnings[column], errors="coerce")
        else:
            earnings[column] = np.nan

    earnings["TTM_Reported_EPS"] = (
        earnings["Reported EPS"].rolling(window=4, min_periods=4).sum()
    )
    earnings["TTM_Estimated_EPS"] = (
        earnings["EPS Estimate"].rolling(window=4, min_periods=4).sum()
    )

    eps_columns = earnings[["TTM_Reported_EPS", "TTM_Estimated_EPS"]].copy()
    combined_index = valuation.index.union(eps_columns.index).sort_values()
    earnings_daily = eps_columns.reindex(combined_index).ffill().reindex(valuation.index)

    valuation["TTM_Reported_EPS"] = earnings_daily["TTM_Reported_EPS"]
    valuation["TTM_Estimated_EPS"] = earnings_daily["TTM_Estimated_EPS"]
    valuation["Trailing_PE"] = np.where(
        valuation["TTM_Reported_EPS"] > 0,
        valuation["Close"] / valuation["TTM_Reported_EPS"],
        np.nan,
    )
    valuation["Estimate_Based_PE_Proxy"] = np.where(
        valuation["TTM_Estimated_EPS"] > 0,
        valuation["Close"] / valuation["TTM_Estimated_EPS"],
        np.nan,
    )
    valuation = valuation.replace([np.inf, -np.inf], np.nan)
    valuation["Implied_Close_From_PE"] = (
        valuation["Trailing_PE"] * valuation["TTM_Reported_EPS"]
    )
    valuation["PE_Reconstruction_Error"] = (
        valuation["Implied_Close_From_PE"] - valuation["Close"]
    ).abs()
    return valuation, earnings


# ============================================================
# Benchmark loader
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def load_benchmark_data(start_date_str, end_date_str, timeframe):
    """Download S&P 500 and Nasdaq-100 and normalize each to 100."""

    symbols = ["^GSPC", "^NDX"]
    start_timestamp = pd.Timestamp(start_date_str).normalize()
    selected_end_timestamp = pd.Timestamp(end_date_str).normalize()
    download_end_timestamp = selected_end_timestamp + pd.Timedelta(days=1)

    try:
        data = yf.download(
            symbols,
            start=start_timestamp.strftime("%Y-%m-%d"),
            end=download_end_timestamp.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if data is None or data.empty:
            return pd.DataFrame()

        close = data["Close"].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
        close.index = pd.to_datetime(close.index)
        if getattr(close.index, "tz", None) is not None:
            close.index = close.index.tz_localize(None)
        close = close.sort_index()
        close = close.loc[
            (close.index >= start_timestamp) & (close.index <= selected_end_timestamp)
        ]
        close = close.rename(columns={"^GSPC": "S&P 500", "^NDX": "Nasdaq-100"})

        if timeframe == "Weekly":
            close = close.resample("W-FRI").last().dropna(how="all")
            close = close.loc[close.index <= selected_end_timestamp]
        elif timeframe == "Monthly":
            close = close.resample("ME").last().dropna(how="all")
            close = close.loc[close.index <= selected_end_timestamp]

        normalized = pd.DataFrame(index=close.index)
        for column in close.columns:
            valid = close[column].dropna()
            if not valid.empty and valid.iloc[0] != 0:
                normalized[column] = close[column] / valid.iloc[0] * 100
        return normalized

    except Exception:
        return pd.DataFrame()


# ============================================================
# Position engine
# ============================================================

def trading_cost_settings():
    """Broker fees and slippage, charged on every buy and every sell of every Backtest rule."""

    with settings_panel("Trading costs"):
        plan_label = st.selectbox(
            "Broker fees", list(OPENING_BAR_COMMISSIONS), key="cost_commissions",
            help="IBKR Pro Tiered: $0.0035 a share (min $0.35 an order) plus exchange (~$0.003 a share for a market "
            "order), clearing and pass-through fees. Fixed: $0.005 a share, min $1 an order. Sales also pay the SEC "
            "and FINRA fees. On a $10,000 position Tiered comes to about $1.30 a round trip.")
        slippage = float(st.number_input(
            "Slippage per share, each buy and sell ($)", min_value=0.0, max_value=5.0, value=0.01, step=0.01,
            format="%.2f", key="cost_slippage",
            help="How much worse than the chart price each order fills. About 1 cent is typical for liquid large "
            "caps; use more for thinner stocks or fast markets."))
        st.caption("Charged on every buy and every sell, for every signal rule.")
    return OPENING_BAR_COMMISSIONS[plan_label], slippage


def equity_with_costs(df, account_size, commission_plan="tiered", slippage=0.01):
    """Equity of a full-position backtest (``Position``, ``Buy_Signal``, ``Sell_Signal`` from the position engine),
    paying broker fees and ``slippage`` $ a share on every buy and sell at the signal bar's close.

    Returns (equity series, total costs $, number of orders).
    """

    closes = df["Close"].to_numpy(dtype=float)
    returns = (df["Position"] * df["Close"].pct_change().fillna(0.0)).to_numpy(dtype=float)
    buys, sells = df["Buy_Signal"].to_numpy(dtype=bool), df["Sell_Signal"].to_numpy(dtype=bool)
    equity, shares, total_costs, orders = float(account_size), 0.0, 0.0, 0
    values = []
    for row in range(len(df)):
        equity *= 1.0 + returns[row]
        if buys[row] or sells[row]:
            if buys[row]:
                shares = equity / closes[row]
            cost = order_costs(shares, closes[row], sell=bool(sells[row]), plan=commission_plan) + slippage * shares
            equity -= cost
            total_costs += cost
            orders += 1
        values.append(equity)
    return pd.Series(values, index=df.index), total_costs, orders


def run_position_engine(price_df, buy_condition, sell_condition, max_hold=0, stop_pct=0.0, cooldown=0):
    """Generate long-only positions and entry/exit markers.

    ``max_hold`` (bars, 0 = no limit) also sells a position that has been held that long. ``stop_pct`` (0 = off)
    sells at the first close at least that % below the entry close. ``cooldown`` bars must pass after a sell
    before the next buy.
    """

    positions, buy_markers, sell_markers = [], [], []
    in_trade = False
    entry_row = 0
    exit_row = -10**9
    closes = price_df["Close"].to_numpy(dtype=float)
    # Plain arrays: the parameter search runs this many times, and .iloc per bar is slow.
    buys = np.asarray(buy_condition, dtype=bool)
    sells = np.asarray(sell_condition, dtype=bool)

    for row_number in range(len(price_df)):
        buy_signal = False
        sell_signal = False
        if (not in_trade and buys[row_number]
                and row_number - exit_row > cooldown):
            in_trade = True
            buy_signal = True
            entry_row = row_number
        elif in_trade and (sells[row_number]
                           or (max_hold and row_number - entry_row >= max_hold)
                           or (stop_pct and closes[row_number] <= closes[entry_row] * (1 - stop_pct / 100))):
            in_trade = False
            sell_signal = True
            exit_row = row_number

        positions.append(1 if in_trade else 0)
        buy_markers.append(buy_signal)
        sell_markers.append(sell_signal)

    result = price_df.copy()
    result["Raw_Position"] = pd.Series(positions, index=result.index, dtype=int)
    result["Buy_Signal"] = pd.Series(buy_markers, index=result.index, dtype=bool)
    result["Sell_Signal"] = pd.Series(sell_markers, index=result.index, dtype=bool)
    result["Position"] = result["Raw_Position"].shift(1).fillna(0).astype(int)
    return result


# ============================================================
# Signal analysis: data loaders
# ============================================================

def get_secret(name, default=""):
    """Read a Streamlit secret or environment variable without raising."""

    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return value or os.getenv(name, default)


@st.cache_data(ttl=300, show_spinner=False)
def load_intraday_data(symbol, interval="5m", period="10d", start=None, end=None):
    """Download regular-session intraday bars (Yahoo keeps ~60 days of 5m data, ~30 days of 1m data).

    ``start`` and ``end`` (dates as text), when given, replace ``period``.

    The bar that is still forming is dropped, and timestamps are kept in the
    exchange's local time so that 09:30 is the open for US listings.
    """

    try:
        window = dict(start=start, end=end) if start else dict(period=period)
        bars = yf.download(
            symbol,
            **window,
            interval=interval,
            auto_adjust=False,
            prepost=False,
            progress=False,
            threads=False,
        )
        if bars is None or bars.empty:
            return pd.DataFrame()

        bars = bars.copy()
        if isinstance(bars.columns, pd.MultiIndex):
            final_level = bars.columns.get_level_values(-1)
            if symbol in final_level:
                bars = bars.xs(symbol, axis=1, level=-1)
            else:
                bars.columns = bars.columns.get_level_values(0)

        bars.index = pd.to_datetime(bars.index)
        if bars.index.tz is not None:
            bar_length = pd.Timedelta(interval.replace("m", "min"))
            now = pd.Timestamp.now(tz=bars.index.tz)
            bars = bars.loc[bars.index + bar_length <= now]
            bars.index = bars.index.tz_localize(None)

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        if any(column not in bars.columns for column in required_columns):
            return pd.DataFrame()
        bars = bars[required_columns].dropna(subset=["Close"]).sort_index()
        return bars.loc[~bars.index.duplicated(keep="last")]

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamental_snapshot(symbol):
    """Yahoo valuation fields; raises when Yahoo returns nothing so failures are not cached."""

    keys = [
        "trailingPE", "forwardPE", "pegRatio", "trailingPegRatio", "earningsGrowth",
        "revenueGrowth", "profitMargins", "priceToBook", "debtToEquity", "sector", "industry", "shortName",
        "nextFiscalYearEnd", "earningsTimestamp", "earningsTimestampStart", "isEarningsDateEstimate",
    ]
    info = yf.Ticker(symbol).info or {}
    snapshot = {key: info.get(key) for key in keys if info.get(key) is not None}
    if not snapshot:
        raise ValueError(f"Yahoo returned no valuation fields for {symbol}")
    return snapshot


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_revenue_estimates(symbol):
    """Yahoo's analyst revenue estimates (rows 0q, +1q, 0y, +1y); raises when empty so failures are not cached."""

    estimates = yf.Ticker(symbol).revenue_estimate
    if estimates is None or estimates.empty:
        raise ValueError(f"Yahoo returned no revenue estimates for {symbol}")
    return estimates


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_earnings_estimates(symbol):
    """Yahoo's analyst EPS estimates (rows 0q, +1q, 0y, +1y; columns avg, growth ...); raises when empty."""

    estimates = yf.Ticker(symbol).earnings_estimate
    if estimates is None or estimates.empty:
        raise ValueError(f"Yahoo returned no earnings estimates for {symbol}")
    return estimates


def load_earnings_estimates(symbol):
    """Analyst EPS estimates from Yahoo, or an empty frame when unavailable."""

    try:
        return fetch_earnings_estimates(symbol)
    except Exception:
        return pd.DataFrame()


def next_year_eps_growth(symbol):
    """Analysts' expected EPS growth for next fiscal year (0.70 = +70%), or None when unavailable."""

    try:
        value = float(fetch_earnings_estimates(symbol).at["+1y", "growth"])
    except Exception:
        return None
    return value if np.isfinite(value) else None


@st.cache_data(ttl=3600, show_spinner=False)
def load_total_return(symbol, start_date_str, end_date_str):
    """Daily close adjusted for splits and dividends (a total-return series), or an empty Series."""

    try:
        data = yf.download(symbol, start=start_date_str,
                           end=(pd.Timestamp(end_date_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                           auto_adjust=True, progress=False, threads=False)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
        close.index = pd.to_datetime(close.index)
        if getattr(close.index, "tz", None) is not None:
            close.index = close.index.tz_localize(None)
        return close.loc[~close.index.duplicated(keep="last")]
    except Exception:
        return pd.Series(dtype=float)


def load_revenue_estimates(symbol):
    """Analyst revenue estimates from Yahoo, or an empty frame when unavailable."""

    try:
        return fetch_revenue_estimates(symbol)
    except Exception:
        return pd.DataFrame()


def load_fundamental_snapshot(symbol):
    """Current valuation fields from Yahoo (forward P/E, PEG, growth ...), or {} when unavailable."""

    try:
        return fetch_fundamental_snapshot(symbol)
    except Exception:
        return {}


def fundamentals_by_release_date(fundamentals_df, earnings_df, fallback_lag_days=45):
    """Re-index quarterly fundamentals by the date they became public.

    Quarters are matched to their earnings announcement; a quarter with no
    matching announcement is assumed public ``fallback_lag_days`` after its
    fiscal quarter end. This keeps historical analysis free of look-ahead.
    """

    if fundamentals_df is None or fundamentals_df.empty:
        return pd.DataFrame()
    aligned = align_fundamentals_to_earnings_dates(fundamentals_df, earnings_df, maximum_days=120)
    unmatched = aligned.index == pd.DatetimeIndex(aligned["Fiscal_Quarter_End"])
    release_dates = aligned.index.where(
        ~unmatched, aligned.index + pd.Timedelta(days=fallback_lag_days)
    )
    aligned.index = pd.DatetimeIndex(release_dates)
    aligned = aligned.drop(columns=["Fiscal_Quarter_End"]).sort_index()
    return aligned.loc[~aligned.index.duplicated(keep="last")]


def drop_in_progress_daily_bar(daily):
    """Exclude today's daily bar while the US session is still open."""

    if daily.empty:
        return daily, False
    now_new_york = pd.Timestamp.now(tz="America/New_York")
    today = now_new_york.normalize().tz_localize(None)
    if daily.index[-1].normalize() == today and now_new_york.hour < 16:
        return daily.iloc[:-1], True
    return daily, False


def resample_weekly(daily):
    """Weekly bars ending Friday."""

    return resample_ohlcv(daily, "W-FRI")


# ============================================================
# Signal analysis: charts and report rendering
# ============================================================

def build_signal_chart(daily, report, bars_to_show=180):
    """Daily candles with S/R zones, breakout markers and the trade plan."""

    view = daily.iloc[-bars_to_show:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(
        go.Candlestick(
            x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
            name="Daily", increasing_line_color="seagreen", decreasing_line_color="indianred",
        ),
        row=1, col=1,
    )

    visible_low = view["Low"].min() * 0.9
    visible_high = view["High"].max() * 1.1
    for zones, color, label in [(report.supports, "green", "Support"), (report.resistances, "red", "Resistance")]:
        for zone in zones[:4]:
            if zone.high < visible_low or zone.low > visible_high:
                continue
            fig.add_hrect(
                y0=zone.low, y1=zone.high, fillcolor=color, opacity=0.12, line_width=0,
                annotation_text=f"{label} {zone.low:,.2f}-{zone.high:,.2f}", annotation_position="top left",
                annotation_font_size=10, row=1, col=1,
            )

    signals = report.signals.reindex(view.index) if report.signals is not None else None
    if signals is not None:
        for column, symbol, color, name in [
            ("Signal_Up", "triangle-up", "lime", "Bullish signal (vol. confirmed)"),
            ("Signal_Down", "triangle-down", "red", "Bearish signal (vol. confirmed)"),
        ]:
            mask = signals[column].fillna(False).astype(bool) & signals["Volume_Confirmed"].fillna(False).astype(bool)
            points = view.loc[mask]
            if not points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=points.index, y=points["Close"], mode="markers", name=name,
                        marker=dict(symbol=symbol, size=12, color=color, line=dict(color="black", width=1)),
                    ),
                    row=1, col=1,
                )

    if report.signal:
        fig.add_hline(
            y=report.signal["trigger"], line_dash="dot", line_color="orange",
            annotation_text=f"Trigger {report.signal['trigger']:,.2f}", row=1, col=1,
        )
    if report.plan is not None:
        for value, color, label in [
            (report.plan.entry, "royalblue", "Entry"),
            (report.plan.stop, "red", "Stop"),
            (report.plan.target, "green", "Target"),
        ]:
            fig.add_hline(
                y=value, line_dash="dash", line_color=color,
                annotation_text=f"{label} {value:,.2f}", annotation_position="bottom right", row=1, col=1,
            )

    volume_colors = np.where(view["Close"] >= view["Open"], "seagreen", "indianred")
    fig.add_trace(go.Bar(x=view.index, y=view["Volume"], name="Volume", marker_color=volume_colors, opacity=0.7), row=2, col=1)
    average_volume = daily["Volume"].rolling(50).mean().reindex(view.index)
    fig.add_trace(go.Scatter(x=view.index, y=average_volume, name="50-bar avg volume", line=dict(color="gray", width=1)), row=2, col=1)

    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_layout(height=680, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified", showlegend=True)
    return fig


def build_intraday_chart(intraday, trigger_level):
    """5-minute candles for the evaluated session with VWAP and opening range."""

    session = intraday.session_bars
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=session.index, open=session["Open"], high=session["High"], low=session["Low"], close=session["Close"],
            name="5-min", increasing_line_color="seagreen", decreasing_line_color="indianred",
        )
    )
    if "VWAP" in session.columns:
        fig.add_trace(go.Scatter(x=session.index, y=session["VWAP"], name="VWAP", line=dict(color="purple", width=1.5)))
    for value, color, label in [
        (intraday.opening_range_high, "gray", "OR high"),
        (intraday.opening_range_low, "gray", "OR low"),
        (trigger_level, "orange", "Daily trigger"),
    ]:
        if value is not None:
            fig.add_hline(y=value, line_dash="dot", line_color=color, annotation_text=f"{label} {value:,.2f}")
    if intraday.entry_time is not None:
        fig.add_trace(
            go.Scatter(
                x=[intraday.entry_time], y=[intraday.entry_price], mode="markers", name="Entry trigger",
                marker=dict(symbol="star", size=16, color="gold", line=dict(color="black", width=1)),
            )
        )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
    return fig


def format_metric_value(value):
    if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
        return f"{value:,.3f}" if abs(value) < 10 else f"{value:,.1f}"
    return str(value)


def render_signal_analysis(ticker, end_date, signal_settings, risk_settings):
    """Pattern/trend signal, intraday entry, fundamentals, sizing and report."""

    end_timestamp = pd.Timestamp(end_date).normalize()
    end_str = end_timestamp.strftime("%Y-%m-%d")
    analysis_start = (end_timestamp - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    fundamentals_start = (end_timestamp - pd.DateOffset(years=7)).strftime("%Y-%m-%d")
    is_live = end_timestamp >= pd.Timestamp(date.today()) - pd.Timedelta(days=5)

    st.info(
        "**Validation note:** replayed on 52 large US stocks over 2016-2026, the breakout, pullback and reversal "
        "setups did not beat random entry days with the same stops and targets, and the confidence score did not "
        "rank trades better than chance. The moving-average and day-breakout rules were not part of that study. Treat signals as a structured checklist, and use the **Validate** tab on "
        "your own tickers before acting on them."
    )
    st.caption(
        "Daily/weekly breakout and trend analysis with volume confirmation, a 5-min/15-min entry "
        "trigger, fundamental confirmation and risk-based sizing. The analysis date is the end of the "
        "date range in the settings; it always uses up to 5 years of daily history."
    )

    with st.spinner(f"Running signal analysis for {ticker} as of {end_str}..."):
        daily = load_data(ticker, analysis_start, end_str, "Daily")
        earnings = load_earnings_data(ticker, fundamentals_start, end_str)
        fundamentals, _, fundamentals_source = load_quarterly_fundamentals(ticker, fundamentals_start, end_str)
        snapshot = load_fundamental_snapshot(ticker) if is_live else None
        bars_5m = load_intraday_data(ticker) if is_live else pd.DataFrame()

    if daily.empty:
        st.error(f"No daily data was returned for {ticker}.")
        return

    dropped_today = False
    if is_live:
        daily, dropped_today = drop_in_progress_daily_bar(daily)
    else:
        st.info(
            "Historical analysis date: intraday entry and the live valuation snapshot (forward P/E, Yahoo PEG) "
            "are skipped to avoid look-ahead bias. Fundamentals use only data reported up to that date."
        )
    if dropped_today:
        st.caption("Today's daily bar is still forming and was excluded - daily signals use completed bars only.")

    valuation_df, earnings_calculated = build_valuation_data(daily, earnings)
    fundamentals = fundamentals_by_release_date(fundamentals, earnings_calculated)
    report = analyze(
        ticker,
        daily,
        resample_weekly(daily),
        bars_5m=bars_5m,
        valuation_df=valuation_df,
        fundamentals_df=fundamentals,
        snapshot=snapshot,
        settings=signal_settings,
        risk_settings=risk_settings,
    )

    # Headline -------------------------------------------------------------
    if report.action in ("BUY", "SELL"):
        st.success(f"**{report.headline}**")
    elif report.action.endswith("SETUP") or report.action in ("WATCH", "EXIT"):
        st.warning(f"**{report.headline}**")
    elif report.action == "NO TRADE":
        st.error(f"**{report.headline}**")
    else:
        st.info(f"**{report.headline}**")

    trend_lookup = {trend.timeframe: trend for trend in report.trends}
    metric_columns = st.columns(6)
    metric_columns[0].metric("Confidence", "n/a" if report.confidence is None else f"{report.confidence:.0f}/100")
    for column, timeframe in zip(metric_columns[1:3], ["Daily", "Weekly"]):
        trend = trend_lookup.get(timeframe)
        if trend is not None and np.isfinite(trend.score):
            column.metric(f"{timeframe} trend", trend.label, f"{trend.score:+.2f}", delta_color="off")
        else:
            column.metric(f"{timeframe} trend", "n/a")
    if report.plan is not None:
        metric_columns[3].metric("Reward / risk", f"{report.plan.reward_risk:.2f}")
        metric_columns[4].metric(
            "Shares",
            f"{report.plan.shares:,}",
            f"{report.plan.size_multiplier:g}x size" if report.plan.shares > 0 else None,
            delta_color="off",
        )
        metric_columns[5].metric("Dollar risk", f"${report.plan.dollar_risk:,.0f}")

    for warning in report.warnings:
        if warning != report.headline:
            st.warning(warning)

    # Trade plan and reasoning ---------------------------------------------
    if report.plan is not None:
        plan = report.plan
        plan_column, reasons_column = st.columns([1, 1.3])
        exit_only = plan.direction < 0 and not signal_settings.allow_short
        with plan_column:
            st.subheader("Exit guidance" if exit_only else "Trade plan")
        if exit_only:
            invalidation = report.signal["invalidation"] if report.signal else plan.stop
            plan_column.info(
                f"Bearish signal in long-only mode: no new position. If you hold {ticker}, consider closing or "
                f"reducing it, or raise its stop to about {plan.stop:,.2f}. The signal is invalidated if price "
                f"closes back above {invalidation:,.2f}. The next support zone is near {plan.target:,.2f}."
            )
        else:
            with plan_column:
                st.table(
                    pd.DataFrame(
                        {
                            "Value": [
                                plan.side, f"{plan.entry:,.2f}", f"{plan.stop:,.2f}", f"{plan.target:,.2f}",
                                f"{plan.risk_per_share:,.2f}", f"{plan.reward_risk:.2f}", f"{plan.shares:,}",
                                f"${plan.position_value:,.0f}", f"${plan.dollar_risk:,.0f}",
                            ],
                            "Basis": [
                                "", plan.notes[0] if plan.notes else "", plan.stop_basis, plan.target_basis,
                                "", "", f"{risk_settings.risk_pct:g}% risk x {plan.size_multiplier:g} (confidence tier)",
                                f"cap {risk_settings.max_position_pct:g}% of equity", "",
                            ],
                        },
                        index=["Side", "Entry", "Stop", "Target", "Risk / share", "Reward / risk", "Shares", "Position value", "Dollar risk"],
                    )
                )
                for note in plan.notes[1:]:
                    st.caption(note)
        with reasons_column:
            st.subheader("Reasoning")
            st.markdown("**Supporting evidence**")
            st.markdown("\n".join(f"- {reason}" for reason in report.reasons_for) or "- None")
            st.markdown("**Risks and conflicts**")
            risks = report.reasons_against + [warning for warning in report.warnings if warning != report.headline]
            st.markdown("\n".join(f"- {reason}" for reason in risks) or "- None identified")

    show_chart(build_signal_chart(daily, report), width="stretch", config={"displaylogo": False})

    if report.components:
        st.subheader("Confidence breakdown")
        st.caption(
            "Each component scores from -1 (argues against the trade) to +1 (supports it). "
            "Confidence = 50 + 50 x weighted average; unavailable components are left out and the weights re-normalised. "
            "The weights are a starting point and have not been validated against outcomes yet - treat the score as a "
            "structured checklist, not a probability."
        )
        for component in report.components:
            weight = COMPONENT_WEIGHTS.get(component.name, 0) * 100
            score_text = f"{component.score:+.2f}" if component.available else "n/a"
            with st.expander(f"{component.name} - weight {weight:.0f}% - score {score_text}"):
                st.markdown("\n".join(f"- {detail}" for detail in component.details))
                if component.metrics:
                    st.dataframe(
                        pd.DataFrame(
                            {"Value": [format_metric_value(value) for value in component.metrics.values()]},
                            index=list(component.metrics.keys()),
                        ),
                        width="stretch",
                    )

    if report.intraday is not None and report.intraday.session_bars is not None and not report.intraday.session_bars.empty:
        st.subheader(f"Intraday entry - session {report.intraday.session_date}")
        st.caption(f"Status: {report.intraday.status}. 15-minute bars are resampled from the 5-minute data.")
        show_chart(
            build_intraday_chart(report.intraday, report.signal["trigger"]),
            width="stretch",
            config={"displaylogo": False},
        )

    if report.history_summary is not None and not report.history_summary.empty:
        st.subheader("Historical edge of this signal on this ticker")
        st.caption("Past volume-confirmed signals in the same direction, entering at the next bar's open.")
        summary = report.history_summary.copy()
        for column in ["Win rate", "Average return", "Median return", "Base win rate (any day)", "Base average return"]:
            summary[column] = summary[column].map(lambda value: "n/a" if pd.isna(value) else f"{value * 100:.1f}%")
        st.dataframe(summary, hide_index=True, width="stretch")
        if report.history_events is not None and not report.history_events.empty:
            with st.expander(f"{len(report.history_events)} past signals"):
                st.dataframe(report.history_events, hide_index=True, width="stretch")

    if fundamentals is not None and not fundamentals.empty:
        st.caption(f"Quarterly fundamentals source: {fundamentals_source}.")

    # Report export and notification ----------------------------------------
    st.subheader("Pre-trade report")
    markdown_report = report_to_markdown(report)
    export_columns = st.columns(3)
    export_columns[0].download_button(
        "Download report (Markdown)", markdown_report,
        file_name=f"{ticker}_{end_str}_signal_report.md", mime="text/markdown",
    )
    export_columns[1].download_button(
        "Download report (JSON)", json.dumps(report_to_dict(report), indent=2, default=str),
        file_name=f"{ticker}_{end_str}_signal_report.json", mime="application/json",
    )
    webhook_url = get_secret("ALERT_WEBHOOK_URL")
    if webhook_url:
        if export_columns[2].button("Send notification"):
            ok, message = send_webhook(webhook_url, short_summary(report))
            (st.success if ok else st.error)(f"Notification {'sent' if ok else 'failed'}: {message}")
    else:
        export_columns[2].caption("Add `ALERT_WEBHOOK_URL` (Slack/Discord webhook) to Streamlit secrets to send notifications.")
    with st.expander("Report preview"):
        st.markdown(markdown_report)


# ============================================================
# Validation: walk-forward replay of the signal engine
# ============================================================

DEFAULT_VALIDATION_TICKERS = "AAPL, MSFT, NVDA, AMZN, META, GOOGL, JPM, XOM, KO, COST, UNH, CAT"


@st.cache_data(ttl=21600, show_spinner=False)
def run_ticker_validation(ticker, start_str, end_str, settings_dict, risk_dict, max_hold, include_fundamentals):
    """Walk-forward replay for one ticker. Cached, so re-runs are instant."""

    data_start = (pd.Timestamp(start_str) - pd.DateOffset(years=6)).strftime("%Y-%m-%d")
    data_end = pd.Timestamp(date.today()).strftime("%Y-%m-%d")
    daily = load_data(ticker, data_start, data_end, "Daily")
    if daily.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    daily, _ = drop_in_progress_daily_bar(daily)

    valuation_df, fundamentals_df = None, None
    if include_fundamentals:
        earnings = load_earnings_data(ticker, data_start, data_end)
        valuation_df, earnings_calculated = build_valuation_data(daily, earnings)
        fundamentals, _, _ = load_quarterly_fundamentals(ticker, data_start, data_end)
        fundamentals_df = fundamentals_by_release_date(fundamentals, earnings_calculated)

    trades = walk_forward(
        ticker,
        daily,
        valuation_df,
        fundamentals_df,
        AnalysisSettings(**settings_dict),
        RiskSettings(**risk_dict),
        start_str,
        end_str,
        ValidationSettings(max_hold=int(max_hold)),
    )
    return trades, base_rates(daily, start_str, end_str, horizon=10)


def format_stats_table(frame):
    """Format R, percentage and count columns for display."""

    formatted = frame.copy()
    percent_columns = ["Win rate", "Average return", "Target hit", "Stopped out", "Time exit"]
    for column in formatted.columns:
        if column in percent_columns:
            formatted[column] = formatted[column].map(lambda v: "" if pd.isna(v) else f"{v * 100:.1f}%")
        elif column in ("Average R", "Median R", "Random-entry avg R", "Edge vs random (R)"):
            formatted[column] = formatted[column].map(lambda v: "" if pd.isna(v) else f"{v:+.2f}R")
        elif column in ("t-stat (avg R)", "t-stat (edge)", "Profit factor", "Avg bars held", "Rank correlation with R", "95% noise band"):
            formatted[column] = formatted[column].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    for column in formatted.columns:
        if formatted[column].dtype.name == "category" or column == "Confidence bucket":
            formatted[column] = formatted[column].astype(str)
    return formatted


def fill_similar_stocks(ticker):
    """Button callback: the stock plus up to 9 S&P 500 stocks in its sub-industry."""

    peers = sp500_competitors(ticker, 9)
    if peers:
        st.session_state["robustness_tickers"] = ", ".join([ticker] + peers)
    else:
        st.session_state["robustness_peers_error"] = (
            f"No S&P 500 sub-industry found for {ticker}; type the stocks to test instead.")


def yearly_returns(backtest):
    """Return of the backtest's equity in each calendar year (rebased at each year's first bar)."""

    equity = backtest["Equity"].astype(float)
    return equity.groupby(equity.index.year).apply(lambda year: year.iloc[-1] / year.iloc[0] - 1 if len(year) > 1 else np.nan)


def robustness_settings(s, space):
    """The settings to compare: the last Optimization's picks (when it was run on this rule) and the current ones.

    Returns (list of (label, params), the Optimization result or None).
    """

    current = current_param_values(s, space)
    result = st.session_state.get("optimizer_result")
    if result and "weights" in result and json.loads(result["signature"])[4] == s.setup:
        best, stable = pick_settings(result["trials"], space)
        if stable is not None:
            return [("Most stable (plateau)", params_of(stable, space)), ("Best single trial", params_of(best, space)),
                    ("Your current settings", current)], result
    return [("Your current settings", current)], None


def render_robustness(s):
    """Validate tab: run the Optimization's picks and the current settings on other stocks, before and after the
    tuning period's end, and year by year, against buy & hold."""

    space = SEARCH_SPACES.get(s.setup)
    result = st.session_state.get("optimizer_result")
    has_result = bool(result and "weights" in result and json.loads(result["signature"])[4] == s.setup)
    with st.expander("Test optimized settings on other stocks and periods", expanded=has_result):
        if space is None:
            st.caption("Works on the daily signal rules, not the first-bar day trade.")
            return
        candidates, result = robustness_settings(s, space)
        if result is None:
            st.info("No Optimization result for this rule yet: run one on the Backtest tab (Optimization) to compare "
                    "its picks here. Until then this tests your current settings.")
            tuned_ticker, tune_end, weights, label = s.ticker, None, {"sharpe": 1.0}, "Sharpe ratio"
        else:
            tuned_ticker = json.loads(result["signature"])[0]
            tune_end, weights, label = result["tune_end"], result["weights"], result["objective_label"]
        st.caption(
            f"Runs each set of settings with the Backtest's own engine and costs on every stock below, over the "
            f"Backtest date range ({s.start_date} to {s.end_date}, {s.timeframe.lower()} bars). The picks were tuned on "
            f"**{tuned_ticker}**"
            + (f" up to {tune_end:%Y-%m-%d}" if tune_end is not None else "")
            + ": every other stock, and every date after that, is data the search never saw. Score: "
            f"{label}.")
        st.latex(objective_latex(weights))
        if "robustness_tickers" not in st.session_state:
            st.session_state["robustness_tickers"] = tuned_ticker
        columns = st.columns([3, 1], vertical_alignment="bottom")
        tickers_text = columns[0].text_input("Stocks to test (comma-separated)", key="robustness_tickers")
        columns[1].button("Fill with similar stocks", on_click=fill_similar_stocks, args=(tuned_ticker,),
                          key="robustness_fill_btn",
                          help=f"{tuned_ticker} plus up to 9 S&P 500 stocks in its sub-industry, largest by trading.")
        error = st.session_state.pop("robustness_peers_error", None)
        if error:
            st.warning(error)
        run_clicked = st.button("Run test", type="primary", key="robustness_run_btn")

        tickers = list(dict.fromkeys(t.strip().upper() for t in tickers_text.replace(";", ",").split(",") if t.strip()))
        start_str = pd.Timestamp(s.start_date).strftime("%Y-%m-%d")
        end_str = pd.Timestamp(s.end_date).strftime("%Y-%m-%d")
        run_key = json.dumps([tickers, start_str, end_str, s.timeframe, s.setup, candidates, weights, str(tune_end)],
                             default=str)
        if run_clicked and tickers:
            periods = PERIODS_PER_YEAR.get(s.timeframe, 252)
            rows, yearly, failed = [], [], []
            progress = st.progress(0.0)
            for number, ticker in enumerate(tickers):
                progress.progress(number / len(tickers), text=f"Testing {ticker} ({number + 1}/{len(tickers)})...")
                prices = load_data(ticker, start_str, end_str, s.timeframe)
                if prices.empty or len(prices) < 50:
                    failed.append(ticker)
                    continue
                prices = prices[["Open", "High", "Low", "Close", "Volume"]].dropna()
                split = tune_end if tune_end is not None else split_date(prices.index, 0.7)
                windows = {"Tuning": (prices.index[0], split), "Forward": (split + pd.Timedelta(days=1), prices.index[-1])}
                buy_hold = prices[["Close"]].assign(Equity=prices["Close"] / prices["Close"].iloc[0], Buy_Signal=False)
                runs = [(name, rule_backtest(prices, s, params)) for name, params in candidates] + [("Buy & hold", buy_hold)]
                for name, backtest in runs:
                    row = {"Stock": ticker, "Settings": name}
                    for window, (start, end) in windows.items():
                        metrics = period_metrics(backtest, start, end, periods)
                        row[f"{window} score"] = objective_score(metrics, weights) if metrics["trades"] or name == "Buy & hold" else np.nan
                        row[f"{window} return %"] = metrics["return"] * 100
                        row[f"{window} max DD %"] = -metrics["max_dd_pct"]
                        row[f"{window} trades"] = metrics["trades"] if name != "Buy & hold" else None
                    rows.append(row)
                    yearly.append(yearly_returns(backtest).rename(name).to_frame().assign(Stock=ticker))
            progress.empty()
            st.session_state["robustness_result"] = {
                "key": run_key, "rows": pd.DataFrame(rows), "failed": failed, "tune_end": tune_end,
                "yearly": pd.concat(yearly) if yearly else pd.DataFrame(), "tickers": tickers,
                "tuned_ticker": tuned_ticker}

        stored = st.session_state.get("robustness_result")
        if not stored or stored["key"] != run_key:
            return
        render_robustness_results(stored, [name for name, _ in candidates])


def render_robustness_results(stored, names):
    rows, yearly = stored["rows"], stored["yearly"]
    if stored["failed"]:
        st.warning("No prices for: " + ", ".join(stored["failed"]))
    if rows.empty:
        return
    by_stock = rows.pivot(index="Stock", columns="Settings", values="Forward score")
    summary = []
    for name in names:
        group = rows.loc[rows["Settings"].eq(name)]
        others = group.loc[group["Stock"].ne(stored["tuned_ticker"])]
        forward = by_stock[name]
        summary.append({
            "Settings": name,
            "Median tuning score": group["Tuning score"].median(),
            "Median forward score": group["Forward score"].median(),
            "Median forward score, other stocks": others["Forward score"].median(),
            "Forward: beats buy & hold": (forward > by_stock["Buy & hold"]).mean() * 100,
            "Forward: beats current settings": ((forward > by_stock["Your current settings"]).mean() * 100
                                                if name != "Your current settings" else np.nan),
            "Median forward return %": group["Forward return %"].median(),
            "Median forward max DD %": group["Forward max DD %"].median(),
        })
    bh = rows.loc[rows["Settings"].eq("Buy & hold")]
    summary.append({"Settings": "Buy & hold", "Median tuning score": bh["Tuning score"].median(),
                    "Median forward score": bh["Forward score"].median(),
                    "Median forward score, other stocks": bh.loc[bh["Stock"].ne(stored["tuned_ticker"]), "Forward score"].median(),
                    "Median forward return %": bh["Forward return %"].median(),
                    "Median forward max DD %": bh["Forward max DD %"].median()})
    summary = pd.DataFrame(summary)
    st.markdown(f"**Across {rows['Stock'].nunique()} stocks**")
    st.dataframe(summary, hide_index=True, width="stretch", key="robustness_summary_table", column_config={
        **{column: st.column_config.NumberColumn(format="%.2f") for column in summary.columns if "score" in column or "%" in column},
        "Forward: beats buy & hold": st.column_config.NumberColumn(format="%.0f%% of stocks"),
        "Forward: beats current settings": st.column_config.NumberColumn(format="%.0f%% of stocks")})
    st.caption(
        "Scores are medians across the stocks (the middle stock, so one outlier can't carry it). \"Other stocks\" "
        f"leaves out {stored['tuned_ticker']}, the stock the picks were tuned on. Settings that only work on the stock "
        "they were tuned on, or only before the forward period, did not find a lasting edge.")

    if not yearly.empty:
        st.markdown("**Year by year**")
        columns = st.columns(2)
        shown = columns[0].selectbox("Settings", names, key="robustness_heat_settings")
        baselines = ["Buy & hold"] + [name for name in names if name != shown and name == "Your current settings"]
        baseline = columns[1].selectbox("Compared with", baselines, key="robustness_heat_baseline")
        show_chart(robustness_heatmap(yearly, shown, baseline, stored["tickers"], stored["tune_end"]),
                   width="stretch", key="robustness_heatmap")
        st.caption(
            f"Each cell is the year's return of {shown.lower()} minus {baseline.lower()}, in percentage points: green "
            "means it did better that year. Rows that stay green across stocks and across the forward years are "
            "what a lasting edge looks like; green only on one stock or only before the dashed line is not.")

    with st.expander("Every stock"):
        st.dataframe(rows, hide_index=True, width="stretch", key="robustness_rows_table", column_config={
            column: st.column_config.NumberColumn(format="%.2f") for column in rows.columns
            if "score" in column or "%" in column})


def robustness_heatmap(yearly, shown, baseline, tickers, tune_end):
    """Stocks x years: the year's return of ``shown`` minus ``baseline``, in percentage points."""

    table = yearly.reset_index(names="Year")
    pivot = {}
    for name in (shown, baseline):
        pivot[name] = table.dropna(subset=[name]).pivot(index="Stock", columns="Year", values=name)
    diff = ((pivot[shown] - pivot[baseline]) * 100).reindex([t for t in tickers if t in pivot[shown].index])
    years = [str(year) for year in diff.columns]
    # Colour scale to the 95th percentile, so one extreme year (a stock that tripled) doesn't wash out the rest.
    limit = float(np.nanpercentile(np.abs(diff.to_numpy()), 95)) if diff.notna().any().any() else 1.0
    limit = limit or 1.0
    text = [[f"{stock} {year}: {value:+.1f} pp" if np.isfinite(value) else f"{stock} {year}: no data"
             for year, value in zip(years, values)] for stock, values in zip(diff.index, diff.to_numpy())]
    fig = go.Figure(go.Heatmap(
        x=years, y=list(diff.index), z=diff.to_numpy(), text=text, hoverinfo="text", colorscale="RdYlGn",
        zmid=0, zmin=-limit, zmax=limit, xgap=1, ygap=1,
        colorbar=dict(title=dict(text="pp", side="right"), thickness=12)))
    if tune_end is not None:
        # Between the last tuning year and the first forward year (the split year itself is mixed).
        position = sum(1 for year in diff.columns if year < tune_end.year) - 0.5 + (tune_end.month - 0.5) / 12
        fig.add_vline(x=position, line_dash="dash", line_color="black",
                      annotation_text="forward →", annotation_position="top right")
    fig.update_layout(height=max(260, 34 * len(diff) + 120), margin=dict(t=60 if MOBILE else 30, b=40, l=60, r=20),
                      xaxis=dict(type="category", tickangle=-45 if MOBILE else 0), yaxis=dict(autorange="reversed"))
    return fig


def render_validation(signal_settings, risk_settings, default_end):
    """Validate tab: does the engine make money, and does confidence mean anything?"""

    st.caption(
        "Replays the Signal Analysis engine on every past day the selected setup fired, using only data available "
        "on that day, then simulates the trade plan: entry at the next open, exit at the stop or target (stop assumed "
        "first if both are hit in one bar; gaps fill at the open) or after the maximum holding period. Results are "
        "in R - profit divided by the amount risked. Setup, volume, risk and confidence settings come from the "
        "settings. The intraday trigger cannot be replayed (Yahoo keeps ~60 days of 5-minute data), so taken trades "
        "enter at the next open."
    )

    with settings_panel("Validation"):
        tickers_text = st.text_input("Tickers (comma-separated)", DEFAULT_VALIDATION_TICKERS, key="validation_tickers")
        date_columns = st.columns(2)
        test_start = date_columns[0].date_input(
            "Test from", value=(pd.Timestamp(default_end) - pd.DateOffset(years=2)).date(), max_value=date.today(), key="validation_start"
        )
        test_end = date_columns[1].date_input("Test to", value=default_end, max_value=date.today(), key="validation_end")
        max_hold = int(st.number_input("Max holding period (bars)", min_value=3, max_value=120, value=20, step=1,
                                       key="validation_max_hold"))
        include_fundamentals = st.checkbox(
            "Include fundamentals (slower: loads SEC and earnings history for each ticker)", value=True,
            key="validation_fundamentals",
        )
        run_clicked = st.button("Run validation", type="primary", key="validation_run_btn")

    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers_text.replace(";", ",").split(",") if t.strip()))
    settings_dict = dict(signal_settings.__dict__)
    risk_dict = dict(risk_settings.__dict__)
    start_str = pd.Timestamp(test_start).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(test_end).strftime("%Y-%m-%d")
    run_key = json.dumps([tickers, start_str, end_str, settings_dict, risk_dict, max_hold, include_fundamentals], default=str)

    if run_clicked and tickers and test_start < test_end:
        progress = st.progress(0.0)
        frames, bases, failed = [], [], []
        for index, ticker in enumerate(tickers):
            progress.progress(index / len(tickers), text=f"Replaying {ticker} ({index + 1}/{len(tickers)})...")
            try:
                trades, base = run_ticker_validation(
                    ticker, start_str, end_str, settings_dict, risk_dict, max_hold, include_fundamentals
                )
            except Exception as error:
                trades, base = pd.DataFrame(), pd.Series(dtype=float)
                failed.append(f"{ticker} ({type(error).__name__})")
            if trades.empty and base.empty:
                failed.append(ticker)
            if not trades.empty:
                frames.append(trades)
            if not base.empty:
                bases.append(base)
        progress.empty()
        st.session_state["validation_result"] = {
            "key": run_key,
            "trades": pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
            "base": pd.concat(bases) if bases else pd.Series(dtype=float),
            "failed": failed,
            "setup": signal_settings.setup,
        }

    result = st.session_state.get("validation_result")
    if result is None:
        st.info("Choose tickers and a test period, then press **Run validation**. Expect roughly 5-15 seconds per ticker.")
        return
    if result["key"] != run_key:
        st.warning("The settings changed since this run - press **Run validation** to refresh.")
    if result["failed"]:
        st.caption(f"No data for: {', '.join(result['failed'])}")

    trades = result["trades"]
    if trades.empty:
        st.warning("No signals fired for these tickers and dates.")
        return

    summary = summarize(trades)
    taken_stats = summary["taken"]
    message = verdict(taken_stats)
    if message.startswith("Positive"):
        st.success(f"**Trades the engine would take:** {message}")
    elif message.startswith("Negative"):
        st.error(f"**Trades the engine would take:** {message}")
    else:
        st.warning(f"**Trades the engine would take:** {message}")

    metric_columns = st.columns(6)
    metric_columns[0].metric("Signals replayed", f"{len(trades):,}")
    metric_columns[1].metric("Trades taken", f"{taken_stats.get('Trades', 0):,}")
    if taken_stats.get("Trades", 0):
        metric_columns[2].metric("Win rate", f"{taken_stats['Win rate'] * 100:.1f}%")
        metric_columns[3].metric("Average R", f"{taken_stats['Average R']:+.2f}R")
        random_average = taken_stats.get("Random-entry avg R", np.nan)
        metric_columns[4].metric("Random entries", "n/a" if pd.isna(random_average) else f"{random_average:+.2f}R")
        edge, edge_t = taken_stats.get("Edge vs random (R)", np.nan), taken_stats.get("t-stat (edge)", np.nan)
        metric_columns[5].metric(
            "Edge vs random" if pd.isna(edge_t) else f"Edge vs random (t = {edge_t:.1f})",
            "n/a" if pd.isna(edge) else f"{edge:+.2f}R",
        )
    st.caption(
        "**Random entries** copy each signal's stop and target distances (in ATR) onto random days of the same "
        "ticker and period. Positive R that random entries also earn comes from market drift and the exit rules - "
        "only the **edge vs random** shows whether the signal's timing adds value (|t| >= 2 to trust it)."
    )

    base = result["base"]
    if not base.empty:
        st.caption(
            f"Benchmark - buying these tickers on *any* day in the window and holding 10 bars: "
            f"{(base > 0).mean() * 100:.1f}% of holds were profitable, average {base.mean() * 100:+.2f}%. "
            f"Taken trades averaged {taken_stats.get('Average return', np.nan) * 100:+.2f}% per trade."
        )

    # Cumulative R: what following every taken signal (1R risk each) would have produced.
    curve = go.Figure()
    taken_trades = trades.loc[trades["Taken"]].dropna(subset=["R"]).sort_values("Exit_Date")
    for label, column, subset, color, dash in [
        ("Trades taken", "R", taken_trades, "seagreen", "solid"),
        ("Random entries, same stops/targets", "Random_R", taken_trades, "darkorange", "dot"),
        ("All signals (incl. WATCH / NO TRADE / EXIT)", "R", trades.dropna(subset=["R"]).sort_values("Exit_Date"), "gray", "solid"),
    ]:
        if column in subset and not subset.empty:
            curve.add_trace(
                go.Scatter(
                    x=subset["Exit_Date"], y=subset[column].fillna(0).cumsum(), name=label, mode="lines",
                    line=dict(color=color, width=2, dash=dash),
                )
            )
    curve.add_hline(y=0, line_dash="dot", line_color="gray")
    curve.update_layout(
        height=380, margin=dict(l=20, r=20, t=40, b=20), title="Cumulative R (each trade risks 1R)",
        yaxis_title="Cumulative R", hovermode="x unified",
    )
    show_chart(curve, width="stretch", config={"displaylogo": False})

    exits = exit_signal_stats(trades, base_forward=base)
    if exits:
        move = exits["Stock 10-bar return after signal"]
        any_day = exits.get("Any-day 10-bar return", np.nan)
        helped = exits.get("Exit helped vs holding")
        st.subheader("Were the bearish (EXIT) signals worth acting on?")
        (st.success if helped else st.warning)(
            f"After {exits['Bearish signals']:,} bearish signals the stock moved {move * 100:+.2f}% over the next "
            f"10 bars (rose {exits['Share of signals followed by a rise'] * 100:.0f}% of the time), versus "
            f"{any_day * 100:+.2f}% after any day. "
            + ("Exiting on them avoided below-average returns." if helped else
               "Exiting on them gave up above-average returns - treat EXIT signals with caution.")
        )

    st.subheader("Does the confidence score mean anything?")
    st.caption(
        "If confidence works, higher buckets should show a higher average R, and its rank correlation with R should "
        "sit outside the 95% noise band (the range a random score would produce)."
    )
    st.dataframe(format_stats_table(summary["by_confidence"]), hide_index=True, width="stretch")
    st.dataframe(format_stats_table(summary["components"]), hide_index=True, width="stretch")
    with st.expander("Results if only trades at or above a confidence threshold were taken"):
        st.dataframe(format_stats_table(summary["thresholds"]), hide_index=True, width="stretch")

    for title, key in [
        ("By action", "by_action"),
        ("By direction (1 = bullish, -1 = bearish)", "by_direction"),
        ("By year", "by_year"),
        ("By ticker", "by_ticker"),
    ]:
        with st.expander(title):
            st.dataframe(format_stats_table(summary[key]), hide_index=True, width="stretch")

    with st.expander(f"All {len(trades):,} replayed signals"):
        st.dataframe(trades, hide_index=True, width="stretch")
    st.download_button(
        "Download replayed signals (CSV)",
        trades.to_csv(index=False),
        file_name=f"validation_{result['setup']}_{start_str}_{end_str}.csv",
        mime="text/csv",
    )


# ============================================================
# Stock screener: data loaders
# ============================================================

def split_download(data, tickers):
    """Turn a multi-ticker ``yf.download`` frame into {ticker: OHLCV frame}."""

    frames = {}
    if data is None or data.empty:
        return frames
    fields = ["Open", "High", "Low", "Close", "Volume"]
    if not isinstance(data.columns, pd.MultiIndex):
        if len(tickers) == 1:
            frames[tickers[0]] = data
    else:
        level_0 = set(data.columns.get_level_values(0))
        for ticker in tickers:
            try:
                if ticker in level_0:
                    frame = data[ticker]
                else:
                    frame = data.xs(ticker, axis=1, level=1)
            except KeyError:
                continue
            frames[ticker] = frame
    cleaned = {}
    for ticker, frame in frames.items():
        if any(field not in frame.columns for field in fields):
            continue
        frame = frame[fields].copy()
        frame.index = pd.to_datetime(frame.index)
        if getattr(frame.index, "tz", None) is not None:
            frame.index = frame.index.tz_localize(None)
        frame = frame.sort_index()
        frame = frame.loc[~frame.index.duplicated(keep="last")].dropna(subset=["Close"])
        if not frame.empty:
            cleaned[ticker] = frame
    return cleaned


@st.cache_data(ttl=21600, show_spinner=False)
def load_price_chunk(tickers, start_str, end_str):
    """Daily OHLCV for a batch of tickers. Cached for 6 hours."""

    try:
        data = yf.download(
            list(tickers),
            start=start_str,
            end=end_str,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception:
        return {}
    return split_download(data, list(tickers))


def load_price_panel(tickers, chunk_size=50, progress=None, days=1100):
    """Daily OHLCV for many tickers (``days`` calendar days, about 3 years by default), downloaded in batches."""

    end = pd.Timestamp(date.today()) + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days)
    tickers = list(tickers)
    prices = {}
    for first in range(0, len(tickers), chunk_size):
        chunk = tuple(tickers[first:first + chunk_size])
        prices.update(load_price_chunk(chunk, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
        if progress is not None:
            progress(min(1.0, (first + chunk_size) / max(1, len(tickers))))
    return prices


@st.cache_data(ttl=86400, show_spinner=False)
def load_universe_members(market, universe):
    """Members of a screener universe. Cached for a day."""

    try:
        return load_us_index_members(universe), ""
    except Exception as error:
        return pd.DataFrame(columns=["Ticker", "Name", "Sector", "Industry"]), f"{type(error).__name__}: {error}"


# ============================================================
# Stock screener: page
# ============================================================

def price_lines_chart(frame, title, ma_periods=(), bars=TRADING_DAYS_PER_YEAR, height=380):
    """Last ``bars`` closes with SMAs (computed on the full history so they start filled)."""

    view = frame.iloc[-bars:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=view["Close"], name="Close", line=dict(color="gray", width=1.4)))
    for color, period in zip(["orange", "royalblue", "purple"], ma_periods):
        moving = frame["Close"].rolling(period, min_periods=period).mean().iloc[-bars:]
        fig.add_trace(go.Scatter(x=moving.index, y=moving, name=f"SMA {period}", line=dict(color=color, width=1.2)))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
    return fig


# Short sector names for the first column of the screener table (narrow on phones).
SHORT_SECTORS = {
    "Communication Services": "Comm.", "Consumer Discretionary": "Cons. disc.",
    "Consumer Staples": "Staples", "Information Technology": "Tech", "Health Care": "Health care",
    "Real Estate": "Real estate",
}
SCREENER_PAGE_SIZE = 10
SCREENER_CHART_BARS = 126  # about 6 months of trading days in view when the chart opens


def screener_candles_figure(frame, ticker, ma_periods=(), bars=SCREENER_CHART_BARS, height=460):
    """Daily OHLC candles over the whole loaded history, with SMAs and volume on a right-hand axis (like the
    Stock Fundamentals price panel). It opens on the last ``bars`` days, with the price scale fitted to
    them; zooming out shows the rest."""

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(x=frame.index, open=frame["Open"], high=frame["High"], low=frame["Low"],
                                 close=frame["Close"], name="OHLC", increasing_line_color=UP_COLOR,
                                 decreasing_line_color=DOWN_COLOR, line=dict(width=1)), row=1, col=1)
    lines = []
    for color, period in zip(["orange", "royalblue", "purple"], ma_periods):
        moving = frame["Close"].rolling(period, min_periods=period).mean()
        if moving.notna().any():
            lines.append(moving)
            fig.add_trace(go.Scatter(x=moving.index, y=moving, name=f"SMA {period}", mode="lines",
                                     line=dict(color=color, width=1.2)), row=1, col=1)
    if "Volume" in frame:
        add_volume_overlay(fig, frame, 1)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0))
    # Open on the last ``bars`` days, the price scale fitted to the candles and averages in that window.
    view = frame.iloc[-bars:]
    low = min([float(view["Low"].min())] + [float(line.loc[view.index].min()) for line in lines if line.loc[view.index].notna().any()])
    high = max([float(view["High"].max())] + [float(line.loc[view.index].max()) for line in lines if line.loc[view.index].notna().any()])
    pad = (high - low) * 0.05 or 1.0
    half_day = pd.Timedelta(hours=12)
    # No weekend gaps, and no range slider under the candles.
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])],
                     range=[view.index[0] - half_day, view.index[-1] + half_day])
    fig.update_yaxes(tickformat=",.2f", range=[low - pad, high + pad], row=1, col=1)
    return fig


@st.cache_resource(show_spinner=False)
def fundamentals_store():
    """Yahoo quote-summary rows shared across reruns: {ticker: (fetched at, row)}."""

    return {}


def fetch_fundamental_row(ticker):
    """Screener fundamentals for one ticker, or None when Yahoo fails (so the failure is not stored)."""

    try:
        return fundamental_row(yf.Ticker(ticker).info or {})
    except Exception:
        return None


def load_fundamental_table(tickers, progress=None, max_age_hours=12, retry_failed_minutes=15):
    """Fundamentals (market cap, P/E, growth, earnings dates ...) for many tickers, fetched 8 at a time."""

    store = fundamentals_store()
    now = time.time()
    def stale(ticker):
        fetched_at, row = store[ticker]
        return now - fetched_at > (max_age_hours * 3600 if row is not None else retry_failed_minutes * 60)

    missing = [ticker for ticker in tickers if ticker not in store or stale(ticker)]
    if missing:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for done, (ticker, row) in enumerate(zip(missing, pool.map(fetch_fundamental_row, missing)), start=1):
                # Failed lookups are stored too (as None) so every rerun does not refetch them; they retry sooner.
                store[ticker] = (now, row)
                if progress is not None:
                    progress(done / len(missing))
    rows = {ticker: store[ticker][1] for ticker in tickers if ticker in store and store[ticker][1]}
    return pd.DataFrame.from_dict(rows, orient="index")


def index_membership(market, tickers):
    """{ticker: [index names it belongs to]} over the market's built-in universes."""

    membership = {ticker: [] for ticker in tickers}
    for name in SCREENER_MARKETS[market]["universes"]:
        if name == "Custom list":
            continue
        members, _ = load_universe_members(market, name)
        for ticker in set(members["Ticker"]) & set(tickers):
            membership[ticker].append(name)
    return membership


# ============================================================
# Layout: desktop sidebar or mobile menus
# ============================================================

SUPPLY_CHAIN_TAB = "Semicon Supply Chain"
FUNDAMENTALS_TAB = "Stock Fundamentals"
ECONOMY_TAB = "Economic analysis"
JOURNAL_TAB = "Trade Journal"
FUNDAMENTALS_YEARS = [3, 5, 8, 10]
NEXT_BENCHMARK = "SPY"
TAB_NAMES = [ECONOMY_TAB, "Market Analysis", "Industry Analysis", FUNDAMENTALS_TAB, SUPPLY_CHAIN_TAB, "Stock Screener", "Backtest",
             "Signal Analysis & Trade Plan", "Validate", JOURNAL_TAB]
SIGNAL_TAB = "Signal Analysis & Trade Plan"
MOBILE_AGENT = re.compile(r"Mobi|Android|iPhone|iPad|iPod", re.IGNORECASE)
# Session keys that are actions or display state rather than settings; they are not carried across tab switches.
# A file uploader's value cannot be set; "_draft" keys are unsaved trade notes.
TRANSIENT_KEY_SUFFIXES = ("_btn", "_table", "_chart", "_upload", "_draft")
TRANSIENT_KEYS = {"main_tab", "password_correct"}
VIX_SYMBOL = "^VIX"
MAIN_INDEX_SYMBOLS = list(dict.fromkeys(
    list(SCREENER_MARKETS["US"]["indexes"].values()) + list(INDUSTRY_INDEXES.values()) + [VIX_SYMBOL]))

# Writing a widget's own value back into Session State is how its setting survives a tab switch (see
# keep_settings_across_tabs); Streamlit logs a warning for every such widget that also has a default.
logging.getLogger("streamlit.elements.lib.policies").setLevel(logging.ERROR)


def keep_settings_across_tabs():
    """Only the open tab runs, and Streamlit forgets the value of any widget that did not run.

    Re-storing every keyed setting at the start of each run keeps it, so a tab's settings are still
    there after a visit to another tab.
    """

    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key not in TRANSIENT_KEYS and not key.endswith(TRANSIENT_KEY_SUFFIXES):
            st.session_state[key] = st.session_state[key]


def detect_mobile():
    """Mobile layout on phones and tablets, told apart by the browser's user agent.

    ``layout_mode`` in Session State ("Desktop" or "Mobile") overrides it; the tests use it.

    Streamlit has no device check of its own; the user agent is the signal the server can see.
    """

    mode = st.session_state.get("layout_mode", "Auto")
    if mode != "Auto":
        return mode == "Mobile"
    try:
        agent = st.context.headers.get("User-Agent", "") or ""
    except Exception:
        agent = ""
    return bool(MOBILE_AGENT.search(agent))


MOBILE = False


def settings_panel(title, expanded=True):
    """Where a tab's settings go: the sidebar on desktop, a collapsible menu at the top of the tab on mobile."""

    if MOBILE:
        return st.expander(title, expanded=False)
    return st.sidebar.expander(title, expanded=expanded)


def _copy_widget_value(source, target):
    st.session_state[target] = st.session_state[source]


def slider_with_input(label, min_value, max_value, value, step=1, key=None, help=None, **slider_kwargs):
    """A slider with a number box beside it for typing an exact value; the two stay in sync.

    The slider keeps ``key``, so code that reads ``st.session_state[key]`` is unchanged.
    """

    box_key = f"{key}_input"
    if key not in st.session_state:
        st.session_state[key] = value
    if box_key not in st.session_state:
        st.session_state[box_key] = st.session_state[key]
    # Keep the box beside the slider on phones too, where Streamlit would otherwise stack the columns.
    st.markdown(
        "<style>[class*='st-key-slider-pair-'] [data-testid='stHorizontalBlock'] {flex-wrap: nowrap; gap: 0.75rem}"
        "[class*='st-key-slider-pair-'] [data-testid='stColumn'] {min-width: 0}"
        "[class*='st-key-slider-pair-'] [data-testid='stColumn']:last-child {flex: 0 0 4.5rem}</style>",
        unsafe_allow_html=True,
    )
    with st.container(key=f"slider-pair-{key}"):
        slider_column, box_column = st.columns([3, 1], vertical_alignment="bottom")
    with slider_column:
        result = st.slider(label, min_value, max_value, step=step, key=key, help=help,
                           on_change=_copy_widget_value, args=(key, box_key), **slider_kwargs)
    with box_column:
        st.number_input(f"{label} (type a value)", min_value=min_value, max_value=max_value, step=step, key=box_key,
                        label_visibility="collapsed", on_change=_copy_widget_value, args=(box_key, key))
    return result


def has_date_axis(fig):
    """True when a trace of ``fig`` is drawn against dates (a time chart rather than a scatter or a bar by label)."""

    for trace in fig.data:
        x = getattr(trace, "x", None)
        if x is None or len(x) == 0:
            continue
        first = next((value for value in x if value is not None), None)
        if isinstance(first, (pd.Timestamp, np.datetime64, date)):
            return True
        if isinstance(first, str) and re.match(r"^\d{4}-\d{2}-\d{2}", first):
            return True
    return False


def show_chart(fig, **kwargs):
    """``st.plotly_chart`` that, on mobile, moves the legend under the plot so the plot gets the full width.

    Charts with a unified x hover show the hovered values as labels on their axes (see CHART_AXIS_HOVER_JS).

    Charts zoom and pan along x only, and every y-axis fits the data in view (see analysis.chart_fit and
    CHART_FIT_Y_JS).

    On touch screens a swipe over a chart scrolls the page and two fingers zoom it (see CHART_TOUCH_JS),
    which needs Plotly's scroll zoom; it is only turned on in the mobile layout, so a mouse wheel on
    desktop still scrolls the page.
    """

    # Time charts (one hover line across the panels) get the TradingView-style value labels on the axes; a
    # chart over dates that doesn't pick its own hover gets them too, like Market Analysis.
    if fig.layout.hovermode is None and has_date_axis(fig):
        fig.update_layout(hovermode="x unified")
    if fig.layout.hovermode == "x unified" and fig.layout.meta is None:
        fig.update_layout(meta=dict(axis_hover=True))
    # Zoom the dates only; each y-axis fits the values in view (CHART_FIT_Y_JS).
    fit_y_axes(fig)
    if MOBILE:
        kwargs["config"] = {**kwargs.get("config", {}), "scrollZoom": True}
        compact = dict(font=dict(size=10), entrywidth=0.5, entrywidthmode="fraction")
        legends = [key for key in fig.layout.to_plotly_json() if key.startswith("legend") and key != "legend"]
        if legends:
            # A chart with a legend under each panel keeps them where they are, just smaller.
            for key in ["legend"] + legends:
                fig.layout[key].update(compact)
            fig.update_layout(margin=dict(l=5, r=5))
        else:
            fig.update_layout(
                legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0, **compact),
                margin=dict(l=5, r=5),
            )
    st.plotly_chart(fig, **kwargs)


@st.cache_data(ttl=21600, show_spinner=False)
def load_main_indexes():
    """Daily prices (about 5 years) for the main US and industry indexes, loaded when the app starts."""

    return load_price_panel(MAIN_INDEX_SYMBOLS, days=1900)


# ============================================================
# Economic analysis tab
# ============================================================

@st.cache_data(ttl=21600, show_spinner=False)
def _download_asset_closes(symbols):
    end = pd.Timestamp(date.today()) + pd.Timedelta(days=1)
    data = yf.download(list(symbols), start="1950-01-01", end=end.strftime("%Y-%m-%d"), auto_adjust=True,
                       progress=False, threads=True)
    if data is None or data.empty or "Close" not in data.columns.get_level_values(0):
        raise ValueError("no prices returned")  # raised, not returned, so a failure is not cached
    closes = data["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(symbols[0])
    closes = closes.reindex(columns=list(symbols))
    closes.index = pd.to_datetime(closes.index)
    if getattr(closes.index, "tz", None) is not None:
        closes.index = closes.index.tz_localize(None)
    return closes.sort_index()


def load_asset_closes(symbols):
    """Dividend-adjusted daily closes since 1950, a column per symbol (empty on failure). Cached for 6 hours."""

    try:
        return _download_asset_closes(symbols)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=43200, show_spinner=False)
def _download_fred(series_id):
    return fred_series(series_id)


def load_fred(series_id):
    """A FRED series and "", or an empty Series and the error. Successes are cached for 12 hours."""

    try:
        return _download_fred(series_id), ""
    except Exception as error:
        return pd.Series(dtype=float), f"{type(error).__name__}: {error}"


def render_economic_analysis():
    """The S&P 500 next to other asset classes, and next to inflation, unemployment and interest rates."""

    with settings_panel("Assets"):
        asset_names = st.multiselect("Assets", list(ASSETS), default=DEFAULT_ASSETS, key="economy_assets",
                                     help="The correlation panel and column need Stock (S&P 500) in the list.")
        asset_history = st.select_slider("Asset history", list(HISTORY_CHOICES), value="10y", key="economy_asset_years",
                                         help="All lines start at 100 on the first day every asset shown has a price, "
                                              "so an asset with a short history (bitcoin from 2014) moves the start.")
        asset_scale = st.radio("Scale", ["Start = 100", "Highest = 100%"], horizontal=True, key="economy_asset_scale",
                               help="Highest = 100% shows each asset as a share of its own highest value in the window, "
                                    "so assets with very different gains fit on one linear axis.")
    with settings_panel("Economy"):
        indicator_names = st.multiselect("Indicators", list(INDICATORS), default=DEFAULT_INDICATORS,
                                         key="economy_indicators",
                                         help="Each gets its own panel, with the S&P 500 on a right-hand log axis.")
        macro_history = st.select_slider("Economy history", list(HISTORY_CHOICES), value="30y", key="economy_macro_years")

    st.caption(
        "Context, not a timing signal. Over 1992-2026, switching from stocks to Treasuries on inflation, "
        "unemployment, the Fed rate or the yield curve did not beat holding the S&P 500; the better rules mostly "
        "cut the worst fall at a small cost in return. How much to hold elsewhere is mostly a question of how big "
        "a fall you can sit through: the table below shows each asset's volatility, worst fall and correlation."
    )

    st.subheader("Asset classes compared")
    names = [name for name in ASSETS if name in asset_names]
    if not names:
        st.info("Pick at least one asset in the settings.")
    else:
        symbols = tuple(ASSETS[name][0] for name in names)
        with st.spinner("Loading asset prices..."):
            closes = load_asset_closes(symbols)
        missing = [name for name in names
                   if closes.empty or ASSETS[name][0] not in closes or closes[ASSETS[name][0]].dropna().empty]
        if len(missing) == len(names):
            st.warning("No prices were returned. Yahoo Finance may be rate-limiting; try again in a few minutes.")
        elif missing:
            st.warning(f"No prices for {', '.join(missing)}; left out.")
    if names and len(missing) < len(names):
        shown = [name for name in names if name not in missing]
        frame = closes[[ASSETS[name][0] for name in shown]].set_axis(shown, axis=1)
        last_day = frame.dropna(how="all").index[-1]
        rebased, first_day = rebase(frame, window_start(last_day, HISTORY_CHOICES[asset_history]))
        if first_day is None or len(rebased) < 30:
            st.info("Not enough overlapping history to compare these assets.")
        else:
            starts = {name: frame[name].first_valid_index() for name in shown}
            limiter = max(starts, key=starts.get)
            note = (f" ({limiter} has prices only from {starts[limiter]:%b %Y})"
                    if starts[limiter] is not None and starts[limiter] >= first_day - pd.Timedelta(days=7) else "")
            st.caption(f"Rebased to 100 on {first_day:%d %b %Y}{note}. Stock (S&P 500) is SPY; it, the Treasury ETFs and "
                       "GLD are shown with dividends reinvested. Oil is the Brent front-month futures price, which an oil fund "
                       "does not earn (rolling futures costs money); bitcoin is BTC-USD.")
            show_chart(assets_figure(rebased, gap_px=163 if MOBILE else 108,
                                     scale="peak" if asset_scale == "Highest = 100%" else "start"), width="stretch")
            table = asset_table(rebased)
            formats = {column: "{:+,.0f}" for column in ["Total return %", "Below peak now %", "Worst fall %"]}
            formats |= {"Per year %": "{:+.1f}", "Volatility %/yr": "{:.0f}"}
            formats |= {column: "{:+.2f}" for column in table.columns if column.startswith("Correlation")}
            st.dataframe(table.style.format(formats, na_rep="-"),
                         width="stretch")
            st.caption("Volatility and correlation use weekly returns. A correlation near 0 or below means the asset "
                       "has tended to move on its own when the S&P 500 moves, which is what makes it a diversifier; it "
                       "changes over time (see the bottom panel), e.g. stocks and Treasuries fell together in 2022.")

    st.subheader("S&P 500 vs the US economy")
    if not indicator_names:
        st.info("Pick at least one indicator in the settings.")
        return
    with st.spinner("Loading FRED data..."):
        values, errors = {}, []
        for name in indicator_names:
            series_id, transform, _, _ = INDICATORS[name]
            raw, error = load_fred(series_id)
            if raw.empty:
                errors.append(f"{name} ({series_id}): {error or 'no data'}")
            else:
                values[name] = indicator_values(raw, transform)
        recession, _ = load_fred(RECESSION_SERIES)
        sp500 = load_asset_closes((SP500_INDEX,))
    if errors:
        st.warning("Could not load from FRED: " + "; ".join(errors))
    if not values:
        return
    sp500 = sp500[SP500_INDEX].dropna() if SP500_INDEX in sp500 else pd.Series(dtype=float)
    last_day = max(series.index[-1] for series in values.values())
    start = window_start(last_day, HISTORY_CHOICES[macro_history])
    fig = macro_figure(values, sp500, recession, start=start, gap_px=138 if MOBILE else 108)
    if fig is None:
        st.info("No data in the chosen window.")
        return
    st.caption("Monthly data from FRED (St. Louis Fed); the S&P 500 index level (^GSPC, no dividends) on a right-hand "
               "log axis in every panel. Grey bands are NBER US recessions. Where the lines cross depends only on "
               "how each axis is scaled, so a crossing is not a signal.")
    show_chart(fig, width="stretch")
    st.markdown("\n".join(f"- {latest_text(name, series)}" for name, series in values.items()))


# ============================================================
# Market Analysis tab
# ============================================================

# Index -> member list used for its breadth. Indexes without a member list have no breadth.
MARKET_BREADTH_UNIVERSES = {"S&P 500": "S&P 500", "Nasdaq-100": "Nasdaq-100", "Dow Jones": "Dow Jones 30"}


# VIX chart bands: (from, to, colour, label); the labels match vix_level.
VIX_BANDS = [(0, 16, "rgba(46,139,87,0.10)", "Calm"), (16, 20, "rgba(128,128,128,0.08)", "Normal"),
             (20, 30, "rgba(255,165,0,0.12)", "Nervous"), (30, 200, "rgba(205,92,92,0.14)", "Fear")]


def render_vix(frame, vix):
    """VIX tiles and a one-year chart with its 20-day average over calm / normal / nervous / fear bands."""

    st.subheader("Volatility (VIX)")
    if not vix:
        st.info("No VIX data was returned. Yahoo Finance may be rate-limiting; try again in a few minutes.")
        return
    tiles = st.columns(4)
    tiles[0].metric("VIX", f"{vix['Close']:.1f}", None if pd.isna(vix["Change"]) else f"{vix['Change']:+.1f}",
                    delta_color="inverse")
    tiles[1].metric("Level", vix["Level"])
    tiles[2].metric("20-day average", f"{vix['Average']:.1f}")
    tiles[3].metric("Higher than, past year", f"{vix['Percentile']:.0f}% of days")
    view = frame["Close"].dropna().iloc[-TRADING_DAYS_PER_YEAR:]
    average = frame["Close"].rolling(20, min_periods=20).mean().iloc[-TRADING_DAYS_PER_YEAR:]
    fig = go.Figure()
    top = max(35.0, float(view.max()) * 1.05)
    for low, high, color, label in VIX_BANDS:
        if low < top:
            fig.add_hrect(y0=low, y1=min(high, top), fillcolor=color, line_width=0, layer="below",
                          annotation_text=label, annotation_position="top left", annotation_font_size=10)
    fig.add_trace(go.Scatter(x=view.index, y=view, name="VIX", line=dict(color="gray", width=1.4)))
    fig.add_trace(go.Scatter(x=average.index, y=average, name="20-day average", line=dict(color="orange", width=1.2)))
    fig.update_layout(title="VIX, last year", height=340, margin=dict(l=10, r=10, t=40, b=10),
                      hovermode="x unified", yaxis=dict(range=[min(10.0, float(view.min()) * 0.95), top]))
    show_chart(fig, width="stretch")
    st.caption("VIX is the market's expected S&P 500 swing over the next 30 days, from option prices, as a yearly %. "
               "Divide by 16 for the expected daily move (VIX 16 is about 1% a day). Low VIX usually means calm, "
               "steady markets; high VIX means bigger, choppier swings.")


def render_market_analysis(index_prices):
    """US index trends, market breadth and the market filter. Stores the filter verdict for the screener."""

    indexes = SCREENER_MARKETS["US"]["indexes"]
    with settings_panel("Index"):
        chart_index = st.selectbox(
            "Index", list(indexes), key="market_index",
            help="Drives the index chart and the market breadth below. Breadth needs a member list, so it covers "
                 "S&P 500, Nasdaq-100 and Dow Jones only.",
        )
        universe = MARKET_BREADTH_UNIVERSES.get(chart_index)
        recent_days = int(st.number_input(
            "New-high window (days)", min_value=1, max_value=120, value=20, step=1, key="market_recent_days",
            help="Breadth counts stocks that made a new 52-week high (or low) at least once in this many trading days.",
        ))
        load_clicked = st.button("Load breadth data", type="primary", key="market_load_btn",
                                 help="Downloads every member of the universe; about a minute the first time, then cached.")
    with settings_panel("Moving averages"):
        ma_columns = st.columns(3)
        ma_fast = int(ma_columns[0].number_input("Short", min_value=2, max_value=400, value=20, step=1, key="market_ma_fast"))
        ma_mid = int(ma_columns[1].number_input("Medium", min_value=2, max_value=400, value=50, step=1, key="market_ma_mid"))
        ma_slow = int(ma_columns[2].number_input("Long", min_value=2, max_value=400, value=200, step=1, key="market_ma_slow"))
    ma_periods = tuple(dict.fromkeys([ma_fast, ma_mid, ma_slow]))
    with settings_panel("Market filter", expanded=False):
        gate_index = st.selectbox("Index", list(indexes), key="market_gate_index")
        use_index_ma = st.checkbox(f"Index close above its SMA {ma_slow}", value=True, key="market_gate_close")
        use_index_order = st.checkbox(f"Index SMA {ma_mid} above SMA {ma_slow}", value=False, key="market_gate_order")
        use_breadth = st.checkbox(f"Minimum % of stocks above SMA {ma_slow}", value=True, key="market_gate_breadth")
        breadth_threshold = st.number_input("% above", 0.0, 100.0, 50.0, 5.0, key="market_gate_breadth_pct", disabled=not use_breadth)
        use_highs = st.checkbox(f"Minimum % of stocks with a new 52w high in the last {recent_days} days", value=False, key="market_gate_highs")
        highs_threshold = st.number_input("% with new highs", 0.0, 100.0, 5.0, 1.0, key="market_gate_highs_pct", disabled=not use_highs)
        use_vix = st.checkbox("VIX below a level", value=False, key="market_gate_vix",
                              help="Passes while the latest VIX close is below this level. Calm markets are usually below 16-20.")
        vix_threshold = st.number_input("VIX below", 5.0, 80.0, 20.0, 1.0, key="market_gate_vix_level", disabled=not use_vix)

    st.caption(
        "The US market as a whole: index trends (loaded when the app starts) and the breadth of an index's members "
        "(share above their moving averages, share making new 52-week highs or lows). The market filter can gate "
        "the Stock Screener."
    )
    st.subheader("US indexes")
    index_rows = {}
    for name, symbol in indexes.items():
        if symbol in index_prices:
            row = stock_snapshot(index_prices[symbol], ma_periods=ma_periods, recent_days=recent_days)
            row["Trend"] = index_trend(index_prices[symbol], ma_mid, ma_slow)
            index_rows[name] = row
    if index_rows:
        index_table = pd.DataFrame.from_dict(index_rows, orient="index")
        index_columns = ["Trend", "Close", "Change %"] + [f"Close vs SMA {p} %" for p in ma_periods] + [
            "% from 52w high", "Return 1m %", "Return 3m %", "Return YTD %"]
        st.dataframe(index_table[index_columns].style.format(precision=2), width="stretch")
        if chart_index in index_rows:
            show_chart(price_lines_chart(index_prices[indexes[chart_index]], chart_index, ma_periods), width="stretch")
    else:
        st.warning("No index data was returned. Yahoo Finance may be rate-limiting; try again in a few minutes.")

    vix = vix_summary(index_prices[VIX_SYMBOL]) if VIX_SYMBOL in index_prices else {}
    render_vix(index_prices.get(VIX_SYMBOL), vix)

    # Prices are kept per universe, so switching index shows that index's breadth. Once any breadth
    # has been loaded this session, picking another index downloads its members without the button.
    loaded_universes = st.session_state.setdefault("market_data", {})
    breadth_latest = pd.Series(dtype=float)
    st.subheader(f"Market breadth: {chart_index}")
    members, members_error = (pd.DataFrame(), "") if universe is None else load_universe_members("US", universe)
    tickers = [] if members.empty else members["Ticker"].tolist()
    data_key = json.dumps([universe, tickers])
    loaded = loaded_universes.get(data_key)
    if tickers and (load_clicked or (loaded is None and loaded_universes)):
        progress = st.progress(0.0, text=f"Downloading {chart_index} prices...")
        prices = load_price_panel(tickers, progress=lambda value: progress.progress(value, text=f"Downloading {chart_index} prices..."))
        progress.empty()
        loaded = loaded_universes[data_key] = {"prices": prices}

    if universe is None:
        st.info(f"No member list for {chart_index}, so no breadth. Pick S&P 500, Nasdaq-100 or Dow Jones.")
    elif members.empty:
        st.error(f"Could not load the members of {universe}. {members_error}")
    elif loaded is None:
        st.info(f"Press **Load breadth data** in the settings to download the {len(tickers)} members of {universe}.")
    elif not loaded["prices"]:
        st.error("No price data was returned. Yahoo Finance may be rate-limiting; try again in a few minutes.")
    else:
        breadth = breadth_history(loaded["prices"], ma_periods=(ma_mid, ma_slow), recent_days=recent_days)
        if not breadth.empty:
            breadth_latest = breadth.dropna(how="all").iloc[-1]
            st.caption(f"{int(breadth_latest['Stocks'])} stocks with price data.")
            tiles = st.columns(4)
            tiles[0].metric(f"Above SMA {ma_mid}", f"{breadth_latest[f'% above SMA {ma_mid}']:.0f}%")
            tiles[1].metric(f"Above SMA {ma_slow}", f"{breadth_latest[f'% above SMA {ma_slow}']:.0f}%")
            tiles[2].metric(f"New 52w high, last {recent_days}d", f"{breadth_latest[f'% new 52w high in last {recent_days}d']:.1f}%")
            tiles[3].metric(f"New 52w low, last {recent_days}d", f"{breadth_latest[f'% new 52w low in last {recent_days}d']:.1f}%")
            recent_breadth = breadth.iloc[-TRADING_DAYS_PER_YEAR:]
            fig = go.Figure()
            for column, color in [
                (f"% above SMA {ma_mid}", "royalblue"), (f"% above SMA {ma_slow}", "purple"),
                (f"% new 52w high in last {recent_days}d", "seagreen"), (f"% new 52w low in last {recent_days}d", "indianred"),
            ]:
                fig.add_trace(go.Scatter(x=recent_breadth.index, y=recent_breadth[column], name=column, line=dict(color=color, width=1.4)))
            fig.update_layout(title="Market breadth (% of stocks)", height=340, margin=dict(l=10, r=10, t=40, b=10),
                              hovermode="x unified", yaxis=dict(range=[0, 100]))
            show_chart(fig, width="stretch")

    market_rules = {"index": gate_index}
    if use_index_ma:
        market_rules["index_ma"] = ma_slow
    if use_index_order:
        market_rules["index_ma_order"] = (ma_mid, ma_slow)
    if use_breadth:
        market_rules["min_pct_above_ma"] = (ma_slow, float(breadth_threshold))
    if use_highs:
        market_rules["min_pct_new_highs"] = (f"% new 52w high in last {recent_days}d", float(highs_threshold))
    if use_vix:
        market_rules["max_vix"] = (float(vix_threshold), vix.get("Close"))
    market_ok, market_checks = market_gate(index_rows, breadth_latest, market_rules)
    check_lines = "\n".join(
        f"- {'✅' if ok else ('⚠️ no data:' if ok is None else '❌')} {text}" for text, ok in market_checks
    )
    (st.success if market_ok else st.error)(f"**Market filter: {'PASS' if market_ok else 'FAIL'}**\n\n{check_lines}")
    st.session_state["market_filter_ok"] = market_ok


# ============================================================
# Industry Analysis tab
# ============================================================

INDUSTRY_PERIODS = ["1M", "3M", "6M", "YTD", "1Y", "2Y", "3Y", "5Y"]
# The industry list (chosen indexes and the ones the user added) is kept in a browser cookie, so it is
# still there on the next visit. Streamlit reads cookies when a session starts; the page writes them.
INDUSTRY_COOKIE = "industry_indexes"
INDUSTRY_MIN_BARS = 20  # days of history an added index needs to be worth a line on the chart


def industry_list_from_cookie():
    """(chosen labels, added indexes as {label: symbol}) as saved in the browser, else the defaults."""

    try:
        saved = json.loads(unquote(st.context.cookies.get(INDUSTRY_COOKIE) or "")) or {}
        added = {str(label): str(symbol) for label, symbol in dict(saved.get("added", {})).items()}
        chosen = [str(label) for label in saved["chosen"] if label in INDUSTRY_INDEXES or label in added]
        return chosen, {label: symbol for label, symbol in added.items() if label in chosen}
    except Exception:
        return list(DEFAULT_INDUSTRIES), {}


def save_industry_list(chosen, added):
    """Write the industry list to the browser cookie (for a year) when it differs from what is saved."""

    value = json.dumps({"chosen": chosen, "added": {label: added[label] for label in chosen if label in added}})
    if st.session_state.get("industry_saved") == value:
        return
    st.session_state["industry_saved"] = value
    components.html(
        "<script>window.parent.document.cookie = %s + '=' + encodeURIComponent(%s)"
        " + '; path=/; max-age=31536000; SameSite=Lax';</script>" % (json.dumps(INDUSTRY_COOKIE), json.dumps(value)),
        height=0,
    )


def add_industry_index():
    """Add the typed symbol (and optional name) to the industry list; a built-in index is just chosen."""

    symbol = normalize_ticker(st.session_state.get("industry_add_symbol", ""))
    name = st.session_state.get("industry_add_name", "").strip()
    st.session_state["industry_add_symbol"] = st.session_state["industry_add_name"] = ""
    if not symbol:
        return
    history = load_price_panel([symbol], days=60).get(symbol)
    if history is None or len(history) < INDUSTRY_MIN_BARS:
        st.session_state["industry_add_error"] = (
            f"Yahoo has no daily price history for {symbol}, so it can't be charted. Check the symbol, or add an ETF "
            "that tracks the index instead (some indexes, such as ^SPACE, only have today's quote on Yahoo)."
        )
        return
    builtin = next((label for label, known in INDUSTRY_INDEXES.items() if known == symbol), None)
    label = builtin or (f"{name} ({symbol})" if name else symbol)
    if not builtin:
        st.session_state["industry_added"][label] = symbol
    if label not in st.session_state["industry_indexes"]:
        st.session_state["industry_indexes"] = st.session_state["industry_indexes"] + [label]


# Yahoo lists holdings for funds, not indexes: the ETF that tracks each index stands in for it.
INDEX_FUNDS = {"^GSPC": "SPY", "^SOX": "SOXX", "^BKX": "KBWB", "^NBI": "IBB", "^DJT": "IYT", "^HUI": "GDX",
               "^IXIC": "QQQ", "^NDX": "QQQ", "^DJI": "DIA", "^RUT": "IWM"}
NO_LEADERS = "None"


@st.cache_data(ttl=86400, show_spinner=False)
def _yahoo_industry_leaders(symbol, count):
    fund = INDEX_FUNDS.get(symbol, symbol)
    try:
        holdings = yf.Ticker(fund).funds_data.top_holdings
    except Exception:
        holdings = None
    if holdings is not None and not holdings.empty:
        return [normalize_ticker(ticker) for ticker in holdings.index if str(ticker).strip()][:count]
    industry = (yf.Ticker(symbol).info or {}).get("industryKey")
    if industry:
        peers = [normalize_ticker(ticker) for ticker in yf.Industry(industry).top_companies.index]
        return [symbol] + [ticker for ticker in peers if ticker and ticker != symbol][:count - 1]
    raise LookupError(symbol)  # raised, not returned, so a Yahoo hiccup is not cached


def sp500_competitors(symbol, count):
    """The S&P 500 members in the stock's GICS sub-industry that trade the most dollars a day (a stand-in for size)."""

    members, _ = load_universe_members("US", "S&P 500")
    row = members[members["Ticker"].map(normalize_ticker) == symbol]
    if row.empty or not str(row["Industry"].iloc[0]).strip():
        return []
    peers = [normalize_ticker(ticker) for ticker in members.loc[members["Industry"] == row["Industry"].iloc[0], "Ticker"]]
    peers = [ticker for ticker in peers if ticker != symbol]
    dollar_volume = {ticker: (frame["Close"] * frame["Volume"]).tail(60).mean()
                     for ticker, frame in load_price_panel(peers, days=100).items() if not frame.empty}
    return sorted(dollar_volume, key=dollar_volume.get, reverse=True)[:count]


def industry_leaders(symbol, count=3):
    """The leading companies for one line of the industry chart.

    For an ETF (or the ETF that tracks an index) its largest holdings; for a stock, the stock and the largest
    companies in its industry. Yahoo answers first; without it, a built-in list for the built-in industries, and
    the stock's S&P 500 sub-industry for a stock. Empty when none of these has an answer.
    """

    try:
        return _yahoo_industry_leaders(symbol, count)
    except Exception:
        pass
    if symbol in INDUSTRY_TOP_HOLDINGS:
        return INDUSTRY_TOP_HOLDINGS[symbol][:count]
    competitors = sp500_competitors(symbol, count - 1)
    return [symbol] + competitors if competitors else []


def fill_industry_leaders():
    label = st.session_state.get("industry_leaders_of", NO_LEADERS)
    if label == NO_LEADERS:
        return
    symbol = INDUSTRY_INDEXES.get(label) or st.session_state.get("industry_added", {}).get(label, label)
    leaders = industry_leaders(symbol)
    if leaders:
        st.session_state["industry_leaders"] = ", ".join(leaders)
    else:
        st.session_state["industry_leaders_error"] = (
            f"Yahoo has no holdings or industry for {label}; type the tickers to compare in the settings.")


def reset_industry_list():
    st.session_state["industry_indexes"] = list(DEFAULT_INDUSTRIES)
    st.session_state["industry_added"] = {}


def render_industry_analysis(index_prices):
    """Overlay the main index of each chosen industry, plus an optional look at an industry's leading companies."""

    if "industry_indexes" not in st.session_state or "industry_added" not in st.session_state:
        st.session_state["industry_indexes"], st.session_state["industry_added"] = industry_list_from_cookie()
    added = st.session_state["industry_added"]
    with settings_panel("Industries"):
        chosen = st.multiselect(
            "Industry indexes", list(INDUSTRY_INDEXES) + [label for label in added if label not in INDUSTRY_INDEXES],
            key="industry_indexes",
            help="Tap × on an index to remove it, or pick one from the list. Your list is remembered on this device.",
        )
        st.text_input("Add an index: Yahoo symbol", key="industry_add_symbol", placeholder="e.g. SMH, ^RUT, ^N225")
        st.text_input("Name (optional)", key="industry_add_name", placeholder="e.g. Semiconductor ETF")
        add_column, reset_column = st.columns(2)
        add_column.button("Add index", key="industry_add_btn", on_click=add_industry_index)
        reset_column.button("Reset list", key="industry_reset_btn", on_click=reset_industry_list)
        if st.session_state.get("industry_add_error"):
            st.warning(st.session_state.pop("industry_add_error"))
        save_industry_list(chosen, added)
    with settings_panel("Chart"):
        period = st.select_slider("Period", INDUSTRY_PERIODS, value="1Y", key="industry_period")
        normalize = st.toggle("Normalize (start = 100)", value=True, key="industry_normalize")
        log_scale = st.checkbox("Log scale", value=False, key="industry_log")
    with settings_panel("Leading companies", expanded=False):
        if st.session_state.get("industry_leaders_of", NO_LEADERS) not in chosen:
            st.session_state["industry_leaders_of"] = NO_LEADERS
        leaders_of = st.selectbox(
            "Compare the leading companies of", [NO_LEADERS] + chosen, key="industry_leaders_of",
            on_change=fill_industry_leaders,
            help="Picks the top 3 holdings of an index or ETF, or a stock and its 2 largest competitors. "
            "You can change the tickers below.",
        )
        leaders_on = leaders_of != NO_LEADERS
        leaders_text = st.text_input("Industry leaders (up to 5 tickers)", value="NVDA, AMD, AVGO, TSM, INTC",
                                     key="industry_leaders", disabled=not leaders_on)
        if st.session_state.get("industry_leaders_error"):
            st.warning(st.session_state.pop("industry_leaders_error"))
        leaders_metric = st.radio("Quarterly chart", ["Revenue", "Net_Income"], horizontal=True, key="industry_fund_metric",
                                  format_func=lambda value: value.replace("_", " "), disabled=not leaders_on)

    st.caption(
        "Compare industries through their main index (or the ETF that tracks it, where Yahoo has no index history). "
        "With **Normalize** on, every line starts at 100 so moves are comparable."
    )
    symbols = {label: INDUSTRY_INDEXES.get(label) or added[label] for label in chosen}
    if not symbols:
        st.info("Pick at least one industry index in the settings.")
    else:
        raw_prices = {symbol: index_prices[symbol] for symbol in symbols.values() if symbol in index_prices}
        to_fetch = [symbol for symbol in dict.fromkeys(symbols.values()) if symbol not in raw_prices]
        if to_fetch:
            with st.spinner("Loading prices..."):
                raw_prices.update(load_price_panel(to_fetch, days=1900))
        prices = {label: raw_prices[symbol] for label, symbol in symbols.items()
                  if symbol in raw_prices and len(raw_prices[symbol]) >= INDUSTRY_MIN_BARS}
        missing = [label for label in symbols if label not in prices]
        if missing:
            st.warning(f"Not charted, because Yahoo has no daily price history for: {', '.join(missing)}. "
                       "Remove it and add an ETF that tracks the index instead.")
        if prices:
            last_date = max(frame.index[-1] for frame in prices.values())
            start = period_start(last_date, period)
            overlay = overlay_frame(prices, start=start, normalize=normalize)
            fig = go.Figure()
            for label in overlay.columns:
                benchmark = symbols[label] == "^GSPC"
                fig.add_trace(go.Scatter(x=overlay.index, y=overlay[label], name=label,
                                         line=dict(width=2.2 if benchmark else 1.5, dash="dash" if benchmark else None)))
            if normalize:
                fig.add_hline(y=100, line=dict(color="gray", width=1, dash="dot"))
            fig.update_layout(
                title=f"Industry indexes, {period} ({'rebased to 100' if normalize else 'price'})", height=460,
                margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                yaxis=dict(type="log" if log_scale else "linear"),
            )
            show_chart(fig, width="stretch")

            table = snapshot_table(prices, ma_periods=(50, 200))
            raw_overlay = overlay_frame(prices, start=start, normalize=True)
            period_column = f"Return {period} %"
            table[period_column] = (raw_overlay.ffill().iloc[-1] - 100).reindex(table.index)
            table.insert(0, "Trend", [index_trend(prices[label]) for label in table.index])
            table.insert(1, "Symbol", [symbols[label] for label in table.index])
            columns = ["Trend", "Symbol", "Close", "Change %", period_column, "Return 1m %", "Return 3m %", "Return YTD %",
                       "Return 1y %", "% from 52w high", "Close vs SMA 50 %", "Close vs SMA 200 %"]
            columns = [column for column in dict.fromkeys(columns)
                       if column == period_column or column.lower() != period_column.lower()]
            table.index.name = "Industry"
            st.dataframe(table.sort_values(period_column, ascending=False)[columns].style.format(precision=2), width="stretch")
            st.caption("Trend: Uptrend when the close is above its 50-day SMA and that is above the 200-day SMA; "
                       "Downtrend when both are reversed; otherwise Mixed.")

    if leaders_on:
        st.subheader(f"Leading companies: {leaders_of}")
        render_industry_leaders(leaders_text, leaders_metric)


def render_industry_leaders(leaders_text, metric):
    leaders = parse_ticker_list(leaders_text)
    if len(leaders) > 5:
        st.warning(f"Only the first 5 of {len(leaders)} tickers are compared.")
        leaders = leaders[:5]
    if not leaders:
        st.info("Type up to 5 tickers in the settings.")
        return
    ma_periods = (20, 50, 200)
    leader_prices = load_price_panel(leaders)
    leader_table = snapshot_table(leader_prices, ma_periods=ma_periods)
    if leader_table.empty:
        st.warning("No price data for these tickers.")
        return
    leader_columns = ["Close", "Prev close", "Change %"] + [f"SMA {p}" for p in ma_periods] + [
        f"Close vs SMA {p} %" for p in ma_periods] + ["Volume / avg", "% from 52w high", "Return 1m %", "Return 3m %", "Return 6m %"]
    st.dataframe(leader_table[leader_columns].style.format(precision=2), width="stretch")

    rebased = overlay_frame(leader_prices, start=period_start(max(f.index[-1] for f in leader_prices.values()), "1Y"))
    fig = go.Figure([go.Scatter(x=rebased.index, y=rebased[column], name=column) for column in rebased.columns])
    fig.update_layout(title="Price over the last year (rebased to 100)", height=340,
                      margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
    show_chart(fig, width="stretch")

    fundamentals_start = (pd.Timestamp(date.today()) - pd.DateOffset(years=4)).strftime("%Y-%m-%d")
    today_str = pd.Timestamp(date.today()).strftime("%Y-%m-%d")
    fundamental_rows, quarterly = {}, {}
    with st.spinner("Loading quarterly fundamentals..."):
        for ticker in leader_table.index:
            fundamentals, _, _ = load_quarterly_fundamentals(ticker, fundamentals_start, today_str)
            trend = fundamental_trend(fundamentals)
            if trend:
                fundamental_rows[ticker] = trend
                quarterly[ticker] = fundamentals
    if fundamental_rows:
        fundamental_table = pd.DataFrame.from_dict(fundamental_rows, orient="index")
        for column in fundamental_table.columns:
            if column.endswith("(last Q)") or column.endswith("TTM"):
                fundamental_table[column] = fundamental_table[column] / 1e6
        fundamental_table = fundamental_table.rename(columns=lambda c: f"{c} ($M)" if c.endswith("(last Q)") or c.endswith("TTM") else c)
        st.dataframe(fundamental_table.style.format(precision=1), width="stretch")
        st.caption("Amounts in millions of the reporting currency. YoY compares with the same quarter a year earlier.")
        fig = go.Figure()
        for ticker, frame in quarterly.items():
            if metric in frame:
                series = frame[metric].dropna().iloc[-8:] / 1e6
                fig.add_trace(go.Bar(x=series.index, y=series, name=ticker))
        fig.update_layout(title=f"{metric.replace('_', ' ')} by quarter (millions)", barmode="group", height=340,
                          margin=dict(l=10, r=10, t=40, b=10))
        show_chart(fig, width="stretch")
    else:
        st.caption("No quarterly fundamentals were available for these tickers.")

    health = leaders_health(leader_table, (50, 200))
    health_columns = st.columns(4)
    health_columns[0].metric("Leaders above SMA 50", f"{health.get('Above SMA 50', 0)} / {health['Leaders']}")
    health_columns[1].metric("Leaders above SMA 200", f"{health.get('Above SMA 200', 0)} / {health['Leaders']}")
    health_columns[2].metric("Avg % from 52w high", f"{health.get('Avg % from 52w high', float('nan')):.1f}%")
    health_columns[3].metric("Avg 3-month return", f"{health.get('Avg 3m return %', float('nan')):.1f}%")


# ============================================================
# Semicon Supply Chain tab
# ============================================================

NO_FOCUS = "(none)"


def supply_graph_key():
    return f"supply_graph_{st.session_state.get('supply_graph_round', 0)}_chart"


def focus_from_graph():
    """A tap or click on a company node in the supply-chain graph makes it the focused company.

    Every click also gives the chart a new key, so the next run draws it fresh instead of keeping Plotly's own
    selection, which would grey out every other point and hide the highlighted chain.
    """

    event = st.session_state.get(supply_graph_key())
    st.session_state["supply_graph_round"] = st.session_state.get("supply_graph_round", 0) + 1
    points = event.selection.points if event and event.selection else []
    for point in points:
        customdata = point.get("customdata")
        if customdata and customdata[0] in SUPPLY_COMPANIES:
            st.session_state["supply_focus"] = customdata[0]
            return


def wrap_text(text, width=48):
    words, lines, line = text.split(), [], ""
    for word in words:
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return "<br>".join(lines + [line])


def node_colors(values, metric):
    """Color range for the node metric: symmetric around 0 for returns, from the lowest value to 0 for % from the high."""

    finite = values[np.isfinite(values)]
    if metric == "% from 52w high":
        return dict(cmin=min(-1.0, float(finite.min())) if finite.size else -30.0, cmax=0.0)
    bound = max(1.0, float(np.nanpercentile(np.abs(finite), 90))) if finite.size else 20.0
    return dict(cmin=-bound, cmax=bound)


def supply_chain_figure(chain, table, metric, focus, highlight):
    positions = node_positions(chain)
    linked = linked_tickers(chain, focus) if focus and highlight else set(positions)
    fig = go.Figure()

    for active in (False, True):
        xs, ys = [], []
        for source, target, _ in SUPPLY_CHAINS[chain]["edges"]:
            if (source in linked and target in linked) == active:
                xs += [positions[source][0], positions[target][0], None]
                ys += [positions[source][1], positions[target][1], None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(width=1.6 if active and focus else 1.0,
                      color="rgba(65,105,225,0.75)" if active and focus else "rgba(128,128,128,0.35)"),
        ))

    # Invisible markers along each edge carry the edge's hover text. They sit 30% of the way from the customer,
    # which never lands on a company (companies sit on whole-number columns, edges span 1 to 3 columns).
    edges = SUPPLY_CHAINS[chain]["edges"]
    fig.add_trace(go.Scatter(
        x=[positions[t][0] + 0.3 * (positions[s][0] - positions[t][0]) for s, t, _ in edges],
        y=[positions[t][1] + 0.3 * (positions[s][1] - positions[t][1]) for s, t, _ in edges],
        mode="markers", marker=dict(size=10, color="rgba(0,0,0,0)"), showlegend=False,
        hovertext=[f"<b>{s} → {t}</b><br>{wrap_text(note)}" for s, t, note in edges], hoverinfo="text",
    ))

    tickers = list(positions)
    values = table.loc[tickers, metric].astype(float).to_numpy()
    labels = [f"<b>{t}</b><br>{v:+.1f}%" if np.isfinite(v) else f"<b>{t}</b><br>n/a" for t, v in zip(tickers, values)]
    customdata = [
        [t, SUPPLY_COMPANIES[t][0], table.at[t, "Level"], wrap_text(SUPPLY_COMPANIES[t][1]), table.at[t, "Trend"],
         *[f"{table.at[t, column]:+.1f}%" if pd.notna(table.at[t, column]) else "n/a"
           for column in ["Return 1m %", "Return 3m %", "Return YTD %", "% from 52w high"]],
         f"{table.at[t, 'Close']:.2f}" if pd.notna(table.at[t, "Close"]) else "n/a"]
        for t in tickers
    ]
    fig.add_trace(go.Scatter(
        x=[positions[t][0] for t in tickers], y=[positions[t][1] for t in tickers],
        mode="markers+text", text=labels, textposition="middle right",
        textfont=dict(size=10 if MOBILE else 12), customdata=customdata, showlegend=False,
        marker=dict(
            size=[26 if t == focus else 18 for t in tickers],
            color=np.where(np.isfinite(values), values, np.nan), colorscale="RdYlGn", **node_colors(values, metric),
            opacity=[1.0 if t in linked else 0.25 for t in tickers],
            line=dict(width=[3 if t == focus else 1 for t in tickers], color="gray"),
            showscale=not MOBILE, colorbar=dict(title=dict(text=metric, side="right"), thickness=10, len=0.6),
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b> · %{customdata[1]}<br><i>%{customdata[2]}</i><br>%{customdata[3]}<br><br>"
            "Close %{customdata[9]} · %{customdata[4]}<br>1m %{customdata[5]} · 3m %{customdata[6]} · "
            "YTD %{customdata[7]}<br>From 52w high %{customdata[8]}<extra></extra>"
        ),
    ))

    for x, level in enumerate(SUPPLY_LEVEL_SHORT_NAMES if MOBILE else SUPPLY_LEVELS):
        fig.add_annotation(x=x, y=1.09, text=f"<b>{level}</b>", showarrow=False, xanchor="left",
                           font=dict(size=11 if MOBILE else 12))
    hidden = dict(visible=False, fixedrange=True)
    fig.update_layout(
        height=max(640, 44 * max(len(level) for level in SUPPLY_CHAINS[chain]["levels"]) + 100),
        margin=dict(l=5, r=5, t=30, b=10), dragmode=False, clickmode="event+select",
        xaxis=dict(range=[-0.15, len(SUPPLY_LEVELS) - 0.35], **hidden), yaxis=dict(range=[-0.06, 1.14], **hidden),
        hoverlabel=dict(align="left"),
    )
    return fig


def render_fundamentals(ticker, years=VALUATION_HISTORY_YEARS, log_price=False, link_dates=False, next_note=True):
    """Growth vs price, revenue, EPS, trailing P/E and PEG for one ticker, with their captions.

    Returns False, after a warning, when there is no price history for the ticker. ``next_note`` shows the
    next earnings date above the chart (the supply chain panel shows it higher up itself).
    """

    today = pd.Timestamp(date.today()).strftime("%Y-%m-%d")
    history_start = (pd.Timestamp(date.today()) - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    with st.spinner("Loading quarterly revenue and P/E history..."):
        fundamentals, _, _ = load_quarterly_fundamentals(ticker, history_start, today)
        # Load two extra years so TTM EPS, and its year-on-year growth for PEG, exist from the first day shown.
        earnings_start = (pd.Timestamp(history_start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        earnings = load_earnings_data(ticker, earnings_start, today)
        prices = load_data(ticker, earnings_start, today, "Daily")
    if prices.empty or "Close" not in prices:
        st.warning(f"Yahoo Finance has no price history for {ticker}. Check the symbol: it must be the ticker "
                   "Yahoo uses, e.g. a US listing or an OTC ADR such as BESIY.")
        return False
    valuation, _ = build_valuation_data(prices, earnings)
    trend = fundamental_trend(fundamentals)
    forward = None
    past_releases, upcoming = next_earnings(ticker)
    quarter_ends = fundamentals.index if "Revenue" in fundamentals else None
    reactions = releases_with_reaction(past_releases, prices["Close"], quarter_ends)
    reactions["Text"] = [describe_release(row) for _, row in reactions.iterrows()]
    next_text = next_release_text(upcoming, today)
    if next_note:
        st.markdown(f"**{next_text}**" if upcoming else next_text)
    if trend and "Revenue" in fundamentals:
        revenue = fundamentals["Revenue"].dropna()
        yoy = trend.get("Revenue YoY %")
        margin = trend.get("Net margin %")
        st.caption(
            f"Revenue last quarter: {revenue.iloc[-1] / 1e6:,.0f}M in its reporting currency"
            + (f", {yoy:+.1f}% year on year" if yoy is not None and pd.notna(yoy) else "")
            + (f". Net margin {margin:.1f}%." if margin is not None and pd.notna(margin) else ".")
        )
        st.markdown(f"**{ticker} fundamentals: growth, revenue, EPS and valuation, last {years} years**")
        peg = trailing_peg(valuation).loc[history_start:]
        peg_gaps = peg_gap_reasons(valuation).loc[history_start:]
        # Yahoo's nextFiscalYearEnd is the end of the fiscal year in progress (its "0y").
        fiscal_year_end = load_fundamental_snapshot(ticker).get("nextFiscalYearEnd")
        fiscal_year_end = pd.to_datetime(fiscal_year_end, unit="s") if fiscal_year_end else None
        forward = forward_valuation(valuation, forward_eps(earnings, load_earnings_estimates(ticker),
                                                           fiscal_year_end=fiscal_year_end)).loc[history_start:]
        trailing_pe = valuation["Trailing_PE"].loc[history_start:]
        forward_pe = load_fundamental_snapshot(ticker).get("forwardPE")
        estimates = revenue_estimate_quarters(load_revenue_estimates(ticker), revenue.index[-1])
        all_eps = quarterly_eps(earnings, revenue.index)
        eps = all_eps.loc[history_start:]
        eps_estimates = eps_estimate_quarters(load_earnings_estimates(ticker), revenue.index[-1])
        price_change = price_yoy(valuation["Close"]).loc[history_start:]
        growth = {"Revenue": yoy_change(revenue), "EPS": yoy_change(all_eps)}
        # Captions follow the panel order: growth vs price above the chart, valuation below it.
        st.caption(correlation_text({name: growth_price_correlation(series.loc[history_start:], price_change)
                                     for name, series in growth.items()}))
        show_chart(revenue_pe_figure(revenue, trailing_pe, estimates, peg=peg, eps=eps, eps_estimates=eps_estimates,
                                     price_change=price_change, growth=growth,
                                     close=valuation["Close"].loc[history_start:], log_price=log_price,
                                     ohlcv=prices.loc[history_start:] if not prices.empty else None,
                                     peg_gaps=peg_gaps, forward=forward, link_dates=link_dates,
                                     releases=reactions.loc[reactions["Release"] >= pd.Timestamp(history_start)],
                                     next_release=(upcoming[0], next_text) if upcoming else None),
                   width="stretch")
        st.caption(pe_summary_text(pe_summary(trailing_pe), forward_pe))
        st.caption(peg_summary_text(peg, forward_pe, next_year_eps_growth(ticker), peg_gaps))
        st.caption(
            "YoY change: the price daily, revenue and EPS by quarter, left empty when the year-ago value is zero or "
            "negative. Revenue is in billions of the reporting currency. EPS is Yahoo's reported (adjusted) EPS per "
            "share, on the quarter it reports rather than the announcement date. The top panel shows daily candles "
            "(open, high, low, close), split-adjusted but not dividend-adjusted, the same price the P/E uses, with "
            "daily volume on its right-hand axis. The gray line on the right-hand axis of every other panel is that "
            "close, for comparison; where it crosses a panel's own line depends only on the axis scales. Trailing "
            "P/E is price / last 4 quarters' EPS, so it can differ from GAAP P/E; a spike usually means earnings "
            "fell, not that the price ran. PEG is that P/E / EPS growth %. It is blank, and shaded, while EPS is a "
            "loss, while the year-ago EPS was a loss (growth from a loss has no %), or while EPS is below a year ago. "
            "The purple lines are forward P/E (price / next four quarters' EPS) and forward PEG (that P/E / "
            "EPS growth % from the last four quarters to the next four). Past points use the EPS later reported, "
            "which nobody knew on the day, so they show how cheap the stock really was, not what it looked like then. "
            "The faded end uses analyst estimates for quarters not yet reported (Yahoo gives two quarters and next "
            "fiscal year; the quarters between are filled on a steady growth path). Hatched bars are analyst "
            "estimates, not results. Dotted vertical lines are earnings releases (tap the small triangle at the "
            "bottom of a panel for the quarter, EPS against the estimate and the price move); the red dashed line "
            "is the next one."
        )
    else:
        st.caption("No quarterly revenue was available for this company.")
    forward_pe = forward["Forward_PE"] if forward is not None and not forward.empty else None
    render_earnings_signals(ticker, with_valuation_changes(reactions, valuation["Trailing_PE"], forward_pe),
                            valuation, forward_pe, history_start, log_price)


def render_earnings_signals(ticker, releases, valuation, forward_pe, start, log_price=False, count=8):
    """Each earnings release on the price chart, with its EPS surprise and the P/E change from the release before."""

    st.markdown(f"**{ticker} earnings releases: surprise and P/E change**")
    if releases.empty:
        st.caption("No earnings release dates were available (Yahoo may be rate limiting; foreign filers have no "
                   "SEC 8-K to fall back on). Try again in a few minutes.")
        return
    summary = reaction_summary(releases, count)
    if summary:
        st.caption(summary)
    # The release before the window is kept for the first change, then dropped from the chart.
    start = pd.Timestamp(start)
    shown = releases[releases["Release"] >= start]
    days = valuation.loc[start:].index
    show_chart(earnings_signal_figure(
        valuation["Close"].loc[start:], shown,
        surprise=surprise_held(releases, days),
        pe_change=change_since_release(valuation["Trailing_PE"], releases).loc[start:],
        forward_change=change_since_release(forward_pe, releases).loc[start:] if forward_pe is not None else None,
        log_price=log_price, height=600 if MOBILE else None), width="stretch")
    st.caption(
        "Bold lines mark each earnings release, on the first session that could react (the next day for a release "
        "after the close). Top: the daily close, with a dot at each release (green = EPS beat the analysts' average "
        "estimate, red = missed; a bigger dot is a bigger surprise). Bottom, daily: the last release's EPS surprise %, "
        "held until the next release, and how far the trailing and forward P/E have moved since the close just before "
        "the last release (0 going into each release). On the release the trailing P/E jumps by the new quarter's "
        "EPS replacing the oldest one plus the price reaction; between releases both P/E lines move only with the "
        "price, so they run parallel. Forward P/E uses the EPS later reported (hindsight) where it is known."
    )
    with st.expander("Show as table"):
        render_earnings_table(releases, count)


def render_earnings_table(reactions, count=8):
    """The latest earnings releases with their EPS surprise, the price move after 1 and 5 sessions and the P/E change."""

    table = release_table(reactions, count)
    if MOBILE:
        # Keep the surprise and the price moves on screen; the tap box on the chart has the rest.
        table = table.drop(columns=["Quarter to", "EPS est.", "Source"])
        table["When"] = table["When"].str.replace(" close", "").str.replace(" open", "").str.replace(" market", "")
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "EPS est.": st.column_config.NumberColumn(format="%.2f"),
        "EPS": st.column_config.NumberColumn(format="%.2f"),
        "Surprise %": st.column_config.NumberColumn(format="%+.1f"),
        "1-day %": st.column_config.NumberColumn(format="%+.1f"),
        "5-day %": st.column_config.NumberColumn(format="%+.1f"),
        "P/E chg %": st.column_config.NumberColumn(format="%+.1f"),
        "Fwd P/E chg %": st.column_config.NumberColumn(format="%+.1f"),
    })
    st.caption(
        "Dates and EPS are Yahoo's (reported EPS is adjusted, as companies headline it); a release Yahoo is "
        "missing comes from the company's SEC 8-K filing, which has the time but no EPS. The price moves run from "
        "the close before the first session that could react: the next day for a release after the close, the "
        "same day for one before the open, so they include the overnight gap. 5-day is the close 5 sessions in."
    )



def render_what_happened_next(ticker, years, frequency, look_ahead, lookback_years, band):
    """After past dates with a P/E percentile like today's, how did the stock do against SPY?"""

    st.markdown(f"**What happened next: {ticker} after past dates with a P/E like today's**")
    today = pd.Timestamp(date.today())
    start = today - pd.DateOffset(years=years)
    # The first check needs its full look-back window, plus a year of EPS for the growth figure.
    load_start = (start - pd.DateOffset(years=lookback_years + 1)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    with st.spinner("Loading prices and earnings for the look-back..."):
        valuation, _ = build_valuation_data(load_data(ticker, load_start, end, "Daily"),
                                            load_earnings_data(ticker, load_start, end))
        total_return = load_total_return(ticker, load_start, end)
        benchmark = load_total_return(NEXT_BENCHMARK, load_start, end)
    obs = observations(valuation, total_return, benchmark, frequency, look_ahead, lookback_years, start=start)
    if obs.empty or not np.isfinite(obs["percentile"].iloc[-1]):
        st.caption(f"Not enough trailing P/E history for a {lookback_years}-year comparison of {ticker}.")
        return
    st.caption(today_text(obs, lookback_years))
    table = summary_table(obs, look_ahead, band).rename(columns=lambda c: c.replace("benchmark", NEXT_BENCHMARK))
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        column: st.column_config.NumberColumn(format="%+.1f") for column in table.columns if column.endswith(("%", "pts"))
    } | {f"Beat {NEXT_BENCHMARK} %": st.column_config.NumberColumn(format="%.0f")})

    past = obs.iloc[:-1].dropna(subset=["excess", "percentile"])
    similar = past[similar_mask(past, obs.iloc[-1], band)]
    choices = list(reversed(past.index))
    default = similar.index[-1] if len(similar) else choices[0]
    example = st.selectbox("Explain the dot from", choices, index=choices.index(default),
                           format_func=lambda when: f"{when:%d %b %Y}" + (" (similar to today)" if when in similar.index else ""),
                           key=f"next_example_{ticker}_{frequency}_{look_ahead}_{lookback_years}", help="Defaults to the latest check with a P/E percentile like today's.")
    show_chart(scatter_figure(obs, ticker, look_ahead, NEXT_BENCHMARK, band, lookback_years, example,
                              height=420 if MOBILE else None), width="stretch")
    show_chart(walkthrough_figure(valuation, total_return, benchmark, obs, example, ticker, look_ahead, NEXT_BENCHMARK,
                                  lookback_years, height=900 if MOBILE else None), width="stretch")
    st.caption(
        f"Each dot is one check date. x: the share of trading days in the {lookback_years} years before it with a lower "
        "trailing P/E, using only EPS reported by then. y: the return with dividends over the next "
        f"{look_ahead}, minus {NEXT_BENCHMARK}'s, in percentage points. Checks close together share most of their "
        "look-ahead, so they are not independent; 'Non-overlapping' counts how many do not overlap, a better guide to "
        "how much evidence there is. Past results describe this stock's history only; they are not a forecast."
    )


def render_stock_fundamentals():
    """Fundamentals of any ticker: growth vs price, revenue, EPS, trailing P/E and PEG."""

    with settings_panel("Stock"):
        ticker = st.text_input("Ticker", value="NVDA", key="fundamentals_ticker",
                               help="Any Yahoo symbol with quarterly results, e.g. NVDA, TSM, BESIY.").strip().upper()
        years = st.select_slider("History (years)", FUNDAMENTALS_YEARS, value=VALUATION_HISTORY_YEARS,
                                 key="fundamentals_years")
        log_price = st.checkbox("Log scale for the price", value=False, key="fundamentals_log_price",
                                help="Equal % moves look the same size, so early years aren't flattened.")
        link_dates = st.checkbox("Zoom all panels together", value=False, key="fundamentals_link_dates",
                                 help="Off: a zoom or pan on one panel leaves the others as they are. On: every "
                                      "panel follows, and a tap shows the values of all panels on the same date.")
    with settings_panel("What happened next"):
        frequency = st.radio("Check every", list(CHECK_FREQUENCIES), index=2, horizontal=True, key="next_frequency",
                             help="How often a dot is taken. Daily gives more dots, not more independent evidence.")
        look_ahead = st.select_slider("Look ahead", list(LOOK_AHEADS), value="12 months", key="next_look_ahead",
                                      help="The return measured after each check.")
        lookback_years = st.select_slider("Compare P/E with the past (years)", LOOKBACK_CHOICES,
                                          value=DEFAULT_LOOKBACK_YEARS, key="next_lookback")
        band = slider_with_input("Similar to today = P/E percentile within", 5, 25, DEFAULT_BAND, step=5,
                                 key="next_band")
    st.caption(
        "Is the price following the business? The top panel compares the year-over-year change in revenue and EPS "
        "with the price's own change; below it are revenue, EPS and the valuation those results imply."
    )
    if not ticker:
        st.info("Enter a ticker in the settings.")
        return
    name = load_fundamental_snapshot(ticker).get("shortName")
    st.subheader(f"{ticker} · {name}" if name else ticker)
    if render_fundamentals(ticker, years, log_price, link_dates) is False:
        return
    render_what_happened_next(ticker, years, frequency, look_ahead, lookback_years, band)


def quoted_next_earnings(symbol):
    """The next earnings date from Yahoo's quote (one cached call), or None when Yahoo has none in the future."""

    try:
        _, upcoming = past_and_next(yahoo_releases(None), pd.Timestamp.now(tz=EASTERN).tz_localize(None),
                                    load_fundamental_snapshot(symbol))
    except Exception:
        return None
    return upcoming[0] if upcoming else None


def render_company_detail(chain, ticker, table, prices, index_prices):
    name, role = SUPPLY_COMPANIES[ticker]
    st.subheader(f"{ticker} · {name}")
    st.caption(f"{table.at[ticker, 'Level']}. {role}")
    tiles = st.columns(4)
    close = table.at[ticker, "Close"]
    tiles[0].metric("Close", f"{close:.2f}" if pd.notna(close) else "n/a")
    for tile, column in zip(tiles[1:3], ["Return 3m %", "% from 52w high"]):
        value = table.at[ticker, column]
        tile.metric(column.replace(" %", ""), f"{value:+.1f}%" if pd.notna(value) else "n/a")
    tiles[3].metric("Trend", table.at[ticker, "Trend"])
    _, upcoming = next_earnings(ticker)
    st.markdown(f"**{next_release_text(upcoming, date.today())}**" if upcoming else next_release_text(upcoming, date.today()))

    suppliers, customers = neighbors(chain, ticker)
    # One at a time: Yahoo rejects parallel quote requests ("Invalid Crumb"). Each is cached for an hour.
    reports = {other: quoted_next_earnings(other) for other in sorted({other for other, _ in suppliers + customers})}
    columns = st.columns(2)
    for column, title, links in [(columns[0], "Buys from", suppliers), (columns[1], "Sells to", customers)]:
        with column:
            st.markdown(f"**{title}**")
            st.markdown("\n".join(f"- **{other}** ({SUPPLY_COMPANIES[other][0]}): {note}"
                                   + (f" · _reports {reports[other]:%d %b}_" if reports.get(other) is not None else "")
                                   for other, note in links)
                        if links else "_None in this chain._")
    if any(when is not None for when in reports.values()):
        st.caption("\"Reports\" is the next earnings date Yahoo lists for each supplier and customer, so you can see "
                   "who reports first in the chain.")

    if ticker in prices:
        lines = {ticker: prices[ticker]}
        if "^SOX" in index_prices:
            lines["^SOX"] = index_prices["^SOX"]
        rebased = overlay_frame(lines, start=period_start(prices[ticker].index[-1], "1Y"))
        fig = go.Figure([go.Scatter(x=rebased.index, y=rebased[column], name=column,
                                    line=dict(width=1.2, dash="dot") if column == "^SOX" else dict(width=1.8))
                         for column in rebased.columns])
        fig.add_hline(y=100, line=dict(color="gray", width=1, dash="dot"))
        fig.update_layout(title=f"{ticker} vs the semiconductor index, last year (rebased to 100)", height=320,
                          margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
        show_chart(fig, width="stretch")

    render_fundamentals(ticker, next_note=False)


def render_supply_chain(index_prices):
    """Bottom-up supply chain of one end market as a graph: hover a company for details, tap or click it to focus."""

    with settings_panel("Supply chain"):
        chain = st.selectbox("End market", list(SUPPLY_CHAINS), key="supply_chain")
        metric = st.selectbox("Node metric", SUPPLY_NODE_METRICS, index=1, key="supply_metric",
                              help="Shown on each company and used for its color.")
        highlight = st.checkbox("Highlight the focused company's chain", value=True, key="supply_highlight",
                                help="Dims everything that is not upstream or downstream of the focused company.")

    tickers = chain_tickers(chain)
    if st.session_state.get("supply_focus") not in [NO_FOCUS] + tickers:
        st.session_state["supply_focus"] = NO_FOCUS
    focus = st.session_state.get("supply_focus", NO_FOCUS)
    focus = None if focus == NO_FOCUS else focus

    st.caption(
        f"**{chain}**: traced back from the end market, US-listed companies only. A line means the "
        "left company buys from the right one. Hover a company or a line for details, or tap a company to focus it. "
        "Links are curated from public reporting and show the main relationships, not every one."
    )
    with st.spinner("Loading prices..."):
        prices = load_price_panel(tickers)
    if not prices:
        st.warning("No price data was returned. Yahoo Finance may be rate-limiting; try again in a few minutes.")
    table = node_table(chain, prices)

    show_chart(supply_chain_figure(chain, table, metric, focus, highlight), width="stretch", key=supply_graph_key(),
               on_select=focus_from_graph, selection_mode="points", config={"displayModeBar": False})
    st.selectbox("Focused company", [NO_FOCUS] + tickers, key="supply_focus",
                 format_func=lambda t: t if t == NO_FOCUS else f"{t} · {SUPPLY_COMPANIES[t][0]}")

    if focus:
        render_company_detail(chain, focus, table, prices, index_prices)

    with st.expander("All companies", expanded=not focus):
        columns = ["Company", "Level", "Trend", "Close"] + SUPPLY_NODE_METRICS
        st.dataframe(table[columns].style.format(precision=2), width="stretch")


# ============================================================
# Stock Screener tab
# ============================================================

def condition_table():
    """The screen's conditions, one row each (see analysis.condition_table)."""

    if "screen_table" not in st.session_state:
        st.session_state["screen_table"] = default_condition_table()
        st.session_state["screen_table_version"] = 0
    return st.session_state["screen_table"]


def store_condition_table(table):
    st.session_state["screen_table"] = table.reset_index(drop=True)
    # A new editor key each time, so the grid starts from the stored table rather than replaying old edits.
    st.session_state["screen_table_version"] = st.session_state.get("screen_table_version", 0) + 1


def add_condition_row():
    table = condition_table()
    store_condition_table(pd.concat([table, pd.DataFrame([condition_default_row(MOVING_AVERAGE)])], ignore_index=True))


def render_condition_editor(category_options):
    """The conditions as one editable table: a row of inputs per condition, a ✕ to tick for removing it,
    and a "Reads as" column that spells out each condition (or what it still needs). Returns the complete
    conditions."""

    table = condition_table()
    checks = [row_condition(row, category_options) for row in table.to_dict("records")]
    view = table.copy()
    view["Reads as"] = [condition_label(condition) if condition else f"⚠ {problem}" for condition, problem in checks]
    edited = st.data_editor(
        view, hide_index=True, width="stretch", num_rows="fixed",
        key=f"screen_conditions_{st.session_state.get('screen_table_version', 0)}_table",
        column_config={
            "Remove": st.column_config.CheckboxColumn("✕", width=36, help="Tick to remove this condition"),
            "Signal": st.column_config.SelectboxColumn("Signal", options=CONDITION_SIGNALS, required=True, width=120),
            "Line": st.column_config.TextColumn("Line", width=70,
                                                help="Close, SMA 50, EMA 20 (moving average, rate of change); "
                                                     "1W, 1M, 3M, 6M, YTD, 1Y (Perf %)"),
            "Op": st.column_config.SelectboxColumn("Op", options=CONDITION_OPERATORS, width=55),
            "Value": st.column_config.TextColumn("Value", width=80,
                                                 help="SMA 200, Close or a number; '0 to 30' with Between; a list "
                                                      "such as 'Tech, Energy' for sector, industry, index, rating"),
            "Days": st.column_config.NumberColumn("Days", min_value=1, max_value=365, step=1, format="%d", width=56,
                                                  help="Rate of change look-back, or the earnings-date window"),
            "Reads as": st.column_config.TextColumn("Reads as", disabled=True, width="large"),
        },
    )
    after = edited[CONDITION_COLUMNS].reset_index(drop=True)
    if not after.astype(str).equals(table[CONDITION_COLUMNS].astype(str)):
        store_condition_table(normalize_condition_table(table, after))
        st.rerun()
    st.button("+ Add condition", on_click=add_condition_row, key="screen_add_row_btn")
    return [condition for condition, _ in checks if condition]


def open_in_signal_analysis(ticker):
    """Send a screener result to the ticker box and switch to the Signal Analysis tab."""

    st.session_state["ticker_input"] = ticker
    st.session_state["main_tab"] = SIGNAL_TAB


def set_screener_page(page):
    st.session_state["screener_page"] = page
    st.session_state["screener_picked"] = None


def render_screener_pager(page, pages, count):
    """Previous / next buttons under the results table, with where you are."""

    st.html("<style>.st-key-screener_pager [data-testid='stHorizontalBlock'] {flex-wrap: nowrap}"
            ".st-key-screener_pager [data-testid='stColumn'] {min-width: 0}</style>")
    cells = st.container(key="screener_pager").columns([1, 2, 1], vertical_alignment="center")
    cells[0].button("‹ Prev", key="screener_prev_btn", disabled=page <= 1, on_click=set_screener_page, args=(page - 1,),
                    width="stretch")
    first = (page - 1) * SCREENER_PAGE_SIZE + 1
    cells[1].markdown(f"<div style='text-align:center'>{first}-{min(count, first + SCREENER_PAGE_SIZE - 1)} of {count} "
                      f"· page {page}/{pages}</div>", unsafe_allow_html=True)
    cells[2].button("Next ›", key="screener_next_btn", disabled=page >= pages, on_click=set_screener_page,
                    args=(page + 1,), width="stretch")


def render_screener_pick(picked, shown, prices, ma_periods):
    """Daily candle chart of the picked stock, shown right under its row in the results table."""

    # Keep the Analyze and close buttons on one line on phones too.
    st.html("<style>.st-key-screener_pick_box [data-testid='stHorizontalBlock'] {flex-wrap: nowrap}"
            ".st-key-screener_pick_box [data-testid='stColumn'] {min-width: 0}"
            ".st-key-screener_pick_box [data-testid='stColumn']:last-child {flex: 0 0 3rem}</style>")
    with st.container(border=True, key="screener_pick_box"):
        show_chart(screener_candles_figure(prices[picked], picked, ma_periods), width="stretch",
                   key="screener_candles_chart")
        buttons = st.columns([4, 1], vertical_alignment="center")
        buttons[0].button(f"Analyze {picked} in Signal Analysis", type="primary", key="screener_open_btn",
                          on_click=open_in_signal_analysis, args=(picked,))
        buttons[1].button("✕", key="screener_close_chart_btn", help="Close the chart",
                          on_click=lambda: st.session_state.update(screener_picked=None))
    if st.session_state.pop("screener_scroll_to_chart", False):
        # Bring the picked row and its chart into view (the old chart above may have moved them up).
        components.html(f"""<script>
            const doc = window.parent.document;
            const box = doc.querySelector('.st-key-screener_pick_box');
            // Leave the picked row (just above the chart) in view.
            if (box) {{ box.style.scrollMarginTop = "160px"; box.scrollIntoView({{block: "start"}}); }}
            // {picked} {time.time()}
            </script>""", height=0)


def render_stock_screener():
    """Stock screen over a US index's members, with conditions picked by signal group."""

    market = "US"
    with settings_panel("Stocks to scan"):
        universe = st.selectbox("Universe", SCREENER_MARKETS[market]["universes"], key="screener_universe")
        if universe == "Custom list":
            custom_text = st.text_area("Tickers (comma or space separated)", value="AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA",
                                       key="screener_custom")
            members = pd.DataFrame({"Ticker": parse_ticker_list(custom_text), "Name": "", "Sector": "", "Industry": ""})
            members_error = ""
        else:
            members, members_error = load_universe_members(market, universe)
        market_caps = st.multiselect("Market cap", list(MARKET_CAP_BUCKETS), key="screener_market_caps",
                                     placeholder="Any size",
                                     help="Only list stocks of these sizes (market cap from Yahoo Finance). Empty = any size.")
        load_clicked = st.button(f"Load data for {len(members)} stocks", type="primary", key="screener_load_btn",
                                 disabled=members.empty)

    tickers = members["Ticker"].tolist()
    data_key = json.dumps([market, universe, tickers])
    loaded = st.session_state.get("screener_data")
    if load_clicked:
        progress = st.progress(0.0, text="Downloading prices...")
        prices = load_price_panel(tickers, progress=lambda value: progress.progress(value, text="Downloading prices..."))
        progress.empty()
        loaded = {"key": data_key, "prices": prices}
        st.session_state["screener_data"] = loaded
    is_loaded = loaded is not None and loaded["key"] == data_key
    prices = loaded["prices"] if is_loaded else {}

    # Per-ticker columns the signal groups read, beyond the price snapshot.
    groups_in_use = set(condition_table()["Signal"])
    scanned = [ticker for ticker in tickers if ticker in prices] if is_loaded else tickers
    extra = members.drop_duplicates("Ticker").set_index("Ticker")[["Name", "Sector", "Industry"]].reindex(scanned)
    extra = extra.fillna("").replace("nan", "")
    fundamentals_note = fundamentals_warning = ""
    if is_loaded and prices:
        # The results table leads with each stock's sector; fill missing ones from Yahoo for short lists.
        sectors_missing = extra["Sector"].eq("").any() and len(scanned) <= 150
        wants_yahoo = bool(market_caps) or sectors_missing or any(SIGNAL_GROUPS.get(group, {}).get("fundamental") for group in groups_in_use) or (
            bool(groups_in_use & {"Sector", "Industry"}) and (extra["Sector"].eq("") | extra["Industry"].eq("")).any())
        if wants_yahoo:
            progress = st.progress(0.0, text="Loading fundamentals from Yahoo...")
            fundamentals = load_fundamental_table(scanned, progress=lambda value: progress.progress(value, text="Loading fundamentals from Yahoo..."))
            progress.empty()
            # Yahoo can rate-limit or block quote lookups; the columns must exist anyway so the conditions
            # simply fail for stocks without data instead of crashing the table.
            fundamentals = fundamentals.reindex(index=scanned, columns=FUNDAMENTAL_COLUMNS)
            extra = extra.join(fundamentals, how="left")
            extra["Sector"] = extra["Sector"].where(extra["Sector"] != "", extra["Yahoo sector"].fillna(""))
            extra["Industry"] = extra["Industry"].where(extra["Industry"] != "", extra["Yahoo industry"].fillna(""))
            found = int(fundamentals.notna().any(axis=1).sum())
            fundamentals_note = f"Fundamentals for {found} of {len(scanned)} stocks (Yahoo Finance, refreshed every 12 hours)."
            if found < len(scanned) / 2:
                fundamentals_warning = (f"Yahoo Finance returned fundamentals for only {found} of {len(scanned)} stocks, "
                                        "so conditions on them fail for the rest. Yahoo may be rate-limiting; it retries "
                                        "the missing ones after 15 minutes.")
        if "Index" in groups_in_use:
            membership = index_membership(market, scanned)
            extra["Index"] = [membership.get(ticker, []) for ticker in extra.index]
    category_options = {
        "Index": [name for name in SCREENER_MARKETS[market]["universes"] if name != "Custom list"],
        "Sector": sorted(set(GICS_SECTORS) | {value for value in extra["Sector"] if value}),
        "Industry": sorted({value for value in extra["Industry"] if value}),
        "Analyst rating": ANALYST_RATINGS,
    }

    with settings_panel("Conditions"):
        st.caption("A stock must meet every condition. Tap a cell to change it; tick ✕ to remove a row. "
                   "\"Reads as\" shows how each row is read.")
        conditions = render_condition_editor(category_options)
    with settings_panel("More filters", expanded=False):
        use_high = st.checkbox("Within % of 52w high", value=True, key="screener_use_high")
        max_from_high = st.number_input("Max % below 52w high", 0.0, 100.0, 25.0, 1.0,
                                        key="screener_max_from_high", disabled=not use_high)
        min_price = st.number_input("Min price", 0.0, value=5.0, step=1.0, key="screener_min_price")
        min_volume = st.number_input("Min avg daily volume (shares)", 0.0, value=0.0, step=100000.0, key="screener_min_volume")
        cross_window = int(st.number_input("Cross window (bars)", 1, 60, 5, 1, key="screener_cross_window",
                                           help="Cross Above / Below counts when the cross happened within this many bars and still holds."))
        recent_days = int(st.number_input("New-high window (days)", min_value=1, max_value=120, value=20, step=1,
                                          key="screener_recent_days",
                                          help="The results show whether each stock made a new 52-week high in this many trading days."))
        ma_columns = st.columns(3)
        ma_fast = int(ma_columns[0].number_input("Short MA", min_value=2, max_value=400, value=20, step=1, key="screener_ma_fast"))
        ma_mid = int(ma_columns[1].number_input("Medium MA", min_value=2, max_value=400, value=50, step=1, key="screener_ma_mid"))
        ma_slow = int(ma_columns[2].number_input("Long MA", min_value=2, max_value=400, value=200, step=1, key="screener_ma_slow"))
        require_market = st.checkbox("Only list stocks when the Market Analysis filter passes", value=False,
                                     key="screener_require_gates")
    ma_periods = tuple(dict.fromkeys([ma_fast, ma_mid, ma_slow]))

    st.caption(
        "Scan an index's members for stocks that meet every condition. Each condition starts with a **signal group** "
        "(moving averages, price, valuation, growth, sector, analyst rating, earnings dates ...). Prices come from "
        "Yahoo Finance (cached for 6 hours); fundamentals are fetched only when a condition needs them."
    )
    if members.empty:
        st.error(f"Could not load the members of {universe}. {members_error} Try **Custom list** instead.")
        return
    if not is_loaded:
        st.info(f"Press **Load data for {len(tickers)} stocks** in the settings to download prices. A full index takes "
                "about a minute the first time, then it is cached.")
        return
    missing = [ticker for ticker in tickers if ticker not in prices]
    if missing:
        st.caption(f"No price data for {len(missing)} ticker(s): {', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}")
    if not prices:
        st.error("No price data was returned. Yahoo Finance may be rate-limiting; try again in a few minutes.")
        return
    if fundamentals_warning:
        st.warning(fundamentals_warning)
    elif fundamentals_note:
        st.caption(fundamentals_note)

    screen_settings = dict(
        max_pct_from_high=float(max_from_high) if use_high else None,
        min_price=float(min_price) if min_price > 0 else None,
        min_avg_volume=float(min_volume) if min_volume > 0 else None,
        cross_window=cross_window,
        ma_periods=ma_periods,
        recent_days=recent_days,
        market_caps=market_caps,
    )
    # Screening a full index takes a couple of seconds; a tap on the table, a sort or a page turn reruns the
    # page with the same screen, so the last result is kept and reused until the data or a setting changes.
    screen_key = json.dumps([data_key, conditions, screen_settings, date.today().isoformat(),
                             int(pd.util.hash_pandas_object(extra.astype(str), index=True).sum())], default=str)
    cached = st.session_state.get("screener_results_cache")
    if cached and cached[0] == screen_key:
        results = cached[1]
    else:
        results = screen_stocks(prices, conditions, extra=extra, **screen_settings)
        st.session_state["screener_results_cache"] = (screen_key, results)
    if results.empty:
        st.warning("No stocks to screen.")
        return
    passing = results[results["Pass"]]
    if require_market:
        market_ok = st.session_state.get("market_filter_ok")
        if market_ok is None:
            st.info("Open the **Market Analysis** tab once to evaluate the market filter; it is ignored for now.")
        elif not market_ok:
            st.error(f"{len(passing)} stock(s) meet the conditions, but they are hidden because the market filter fails "
                     "(see Market Analysis).")
            return
    st.success(f"**{len(passing)} of {len(results)} stocks pass** "
               + ", ".join(condition_label(condition) for condition in conditions)
               + (f", within {max_from_high:g}% of the 52w high" if use_high else "")
               # "$" starts LaTeX in Markdown.
               + (f", market cap {', '.join(market_caps)}".replace("$", "\\$") if market_caps else ""))
    shown = passing.copy()
    if "Index" in shown:
        shown["Index"] = shown["Index"].apply(lambda value: ", ".join(value) if isinstance(value, list) else "")
    # A small chart of the last 3 months of daily closes takes the place of the Pass column.
    shown["Last 3 months"] = [close_sparkline(prices.get(ticker)) for ticker in shown.index]
    # The columns after Close / Change % follow the signal groups in use.
    group_columns = [condition_column(condition) for condition in conditions if condition["group"] not in (MOVING_AVERAGE, RATE_OF_CHANGE)]
    base_columns = ["Last 3 months", "Name", "Close", "Change %"] + (["Mkt cap ($B)"] if market_caps else [])
    display_columns = list(dict.fromkeys(base_columns + group_columns + [f"Close vs SMA {p} %" for p in ma_periods] + [
        "% from 52w high", f"New 52w high in last {recent_days}d", "Volume / avg", "Return 1m %", "Return 3m %"]))
    display_columns = [column for column in display_columns if column in shown.columns]
    st.caption("Tap a stock to see its daily chart right under it; tap it again to close the chart.")
    number_columns = [column for column in display_columns if pd.api.types.is_float_dtype(shown[column])]
    column_config = {column: st.column_config.NumberColumn(format="%.2f") for column in number_columns}
    column_config["Last 3 months"] = st.column_config.LineChartColumn("3 months", width="small", color="auto",
                                                                      help="Daily close over the last 3 months; green when "
                                                                           "it ended higher than it started, red when lower.")
    column_config["Sector"] = st.column_config.TextColumn("Sector", width="small")
    # The sector leads each row, where a row checkbox used to be; a tap on any cell picks the stock.
    sectors = shown["Sector"] if "Sector" in shown else pd.Series("", index=shown.index)
    table_rows = shown[[column for column in display_columns if column != "Sector"]].copy()
    table_rows.insert(0, "Ticker", table_rows.index)
    table_rows.insert(0, "Sector", sectors.fillna("").map(lambda value: SHORT_SECTORS.get(value, value)))
    all_shown = shown
    # Sorting happens here, not in the table's own headers: a header sort would only reorder one half of a
    # table split around an open chart.
    sort_options = ["Sector", "Ticker"] + [column for column in display_columns
                                           if column in number_columns or column in ("Name",)]
    st.html("<style>.st-key-screener_sort [data-testid='stHorizontalBlock'] {flex-wrap: nowrap}"
            ".st-key-screener_sort [data-testid='stColumn'] {min-width: 0}</style>")
    sort_cells = st.container(key="screener_sort").columns([1, 1], vertical_alignment="bottom")
    sort_by = sort_cells[0].selectbox("Sort by", sort_options,
                                      index=sort_options.index("% from 52w high") if "% from 52w high" in sort_options else 0,
                                      key="screener_sort_by", on_change=set_screener_page, args=(1,))
    direction = sort_cells[1].segmented_control("Order", ["↑ Low", "↓ High"],
                                                default="↑ Low" if sort_by in ("Sector", "Ticker", "Name") else "↓ High",
                                                key=f"screener_sort_order_{sort_by}", on_change=set_screener_page,
                                                args=(1,)) or "↓ High"
    ascending = direction.startswith("↑")
    # Ties (e.g. stocks in one sector) keep the screen's own order, nearest the 52-week high first.
    table_rows = table_rows.iloc[np.lexsort([np.arange(len(table_rows)),
                                             table_rows[sort_by].rank(method="dense", ascending=ascending,
                                                                      na_option="bottom").to_numpy()])]
    shown = shown.loc[table_rows.index]
    # 10 stocks a page keeps the page quick on phones.
    pages = max(1, -(-len(table_rows) // SCREENER_PAGE_SIZE))
    page = min(max(1, int(st.session_state.get("screener_page", 1))), pages)
    st.session_state["screener_page"] = page
    first = (page - 1) * SCREENER_PAGE_SIZE
    table_rows = table_rows.iloc[first:first + SCREENER_PAGE_SIZE]
    shown = shown.loc[table_rows.index]
    picked = st.session_state.get("screener_picked")
    if picked not in shown.index or picked not in prices:
        picked = None
    # The chart opens under the picked row: the table is split into the rows down to it and the rows after.
    # Tables are full height (no scrolling inside them), so the page scrolls and the chart stays next to its row.
    # Their keys follow the pick, so each new layout starts with nothing selected; the picked row is highlighted.
    if picked is None:
        parts = [("all", table_rows)]
    else:
        position = shown.index.get_loc(picked)
        parts = [("top", table_rows.iloc[:position + 1]), ("bottom", table_rows.iloc[position + 1:])]
    for part, rows in parts:
        if rows.empty:
            continue
        table = rows
        if picked is not None and part == "top":
            table = table.style.apply(
                lambda row: ["background-color: rgba(255, 196, 0, 0.25)" if row["Ticker"] == picked else "" for _ in row],
                axis=1)
        event = st.dataframe(table, width="stretch", height=38 + 35 * len(rows), column_config=column_config,
                             hide_index=True, on_select="rerun", selection_mode="single-cell",
                             key=f"screener_{part}_{picked}_table")
        selection = getattr(event, "selection", None) if event is not None else None
        cells = list(getattr(selection, "cells", None) or (selection or {}).get("cells", []) or [])
        row_number = cells[0][0] if cells else None
        if row_number is not None and row_number < len(rows):
            ticker = rows["Ticker"].iloc[row_number]
            st.session_state["screener_picked"] = None if ticker == picked else ticker
            st.session_state["screener_scroll_to_chart"] = ticker != picked
            st.rerun()
        if part != "bottom" and picked is not None:
            render_screener_pick(picked, shown, prices, ma_periods)
    if pages > 1:
        render_screener_pager(page, pages, len(passing))
    st.download_button(
        "Download results (CSV)",
        all_shown.drop(columns="Last 3 months").to_csv().encode("utf-8"),
        file_name=f"screener_{market}_{pd.Timestamp(date.today()):%Y%m%d}.csv",
        mime="text/csv",
    )


# ============================================================
# Backtest / Signal Analysis / Validate: strategy settings
# ============================================================

SETUP_OPTIONS = {
    "Moving-average rule": "ma_rule",
    "Breakout (new high/low on volume)": "breakout",
    "Pullback in trend (dip to the EMA, then reversal)": "pullback",
    "Reversal (buy a breakdown in an uptrend)": "reversal",
    "Day breakout (above the N-day high, hold on a top-third close)": "day_breakout",
    "MACD + Bollinger breakout (MACD turns up, bands widen, above a rising SMA)": "macd_bollinger",
    "First-bar day trade (the first five-minute bar decides the day)": "opening_bar",
}
OPENING_BAR_RULE = next(label for label, setup in SETUP_OPTIONS.items() if setup == "opening_bar")
INTRADAY_TIMEFRAMES = {"1 min": 1, "5 min": 5, "15 min": 15}


def strategy_settings(tab="Backtest"):
    """Ticker, dates, signal rule and risk settings shared by Backtest, Signal Analysis and Validate.

    Returns a namespace of the settings, or None (after saying why) when the ticker or dates are not usable.
    """

    # The first-bar day trade runs on the ~60 days of 5-minute bars Yahoo keeps, so timeframe and dates do not apply.
    intraday = st.session_state.get("signal_rule") == OPENING_BAR_RULE
    bar_minutes = 5
    timeframe = st.session_state.get("timeframe", "Daily")
    date_range = st.session_state.get("date_range", (date(2022, 1, 1), date.today()))
    with settings_panel("Ticker & dates"):
        ticker = normalize_symbol(st.text_input("Ticker Symbol", value="AAPL", key="ticker_input"))
        if intraday:
            date_range = (date.today() - timedelta(days=60), date.today())
            intraday_label = st.selectbox(
                "Timeframe", list(INTRADAY_TIMEFRAMES), index=1, key="intraday_timeframe",
                help="Bar length of the day trade. Yahoo keeps about 60 days of 5- and 15-minute bars and about "
                "30 days of 1-minute bars, so that is the period tested.")
            bar_minutes = INTRADAY_TIMEFRAMES[intraday_label]
        else:
            timeframe = st.selectbox("Timeframe", ["Daily", "Weekly", "Monthly"], index=0, key="timeframe")
            date_range = st.date_input(
                "Date Range",
                value=(date(2022, 1, 1), date.today()),
                max_value=date.today(),
                format="YYYY-MM-DD",
                key="date_range",
            )
        account_size = st.session_state.get("account_size", 100000.0)
        if tab != "Validate":  # Validate reports results in R, so the account size does not change them.
            account_size = st.number_input(
                "Account size ($)",
                min_value=1000.0,
                value=100000.0,
                step=5000.0,
                key="account_size",
                help="Starting equity for the Backtest tab and the equity used for position sizing. The first-bar "
                "day trade uses it as the capital allocated to the stock.",
            )

    # Moving-average rule defaults (also used for the Backtest chart's MA lines).
    fast_type, fast_period, slow_type, slow_period = "SMA", 50, "SMA", 200
    buy_left, buy_operator, buy_right = "Fast_MA", "Cross Above", "Slow_MA"
    sell_left, sell_operator, sell_right = "Fast_MA", "Cross Below", "Slow_MA"
    breakout_lookback, volume_multiple = 20, 1.5
    pullback_ema, trend_mid, trend_slow = 20, 50, 200
    swing_bars, pullback_bars, min_depth_atr = 20, 5, 1.5
    keep_fraction = 2 / 3
    exit_ma, max_hold_bars = 200, 0
    macd_ma, macd_stop_pct, macd_exit_on_ma_down, macd_cooldown = 20, 3.0, False, 1
    macd_widen_days, macd_widen_pct, macd_bb_std = 5, 10.0, 2.0
    opening_bar = None

    with settings_panel("Signal"):
        setup_label = st.selectbox(
            "Signal rule",
            list(SETUP_OPTIONS),
            index=0,
            key="signal_rule",
            help="Used by every tab: the Backtest buys and sells on it, Signal Analysis builds a trade plan from it, "
            "and Validate replays it.",
        )
        setup = SETUP_OPTIONS[setup_label]

        if setup == "ma_rule":
            ma_columns = st.columns(2)
            fast_type = ma_columns[0].selectbox("Fast MA Type", ["EMA", "SMA"], index=1, key="fast_type")
            fast_period = int(ma_columns[1].number_input("Fast MA Period", min_value=1, value=50, step=1, key="fast_period"))
            slow_type = ma_columns[0].selectbox("Slow MA Type", ["EMA", "SMA"], index=1, key="slow_type")
            slow_period = int(ma_columns[1].number_input("Slow MA Period", min_value=1, value=200, step=1, key="slow_period"))

            st.markdown("**Buy when**")
            buy_left = st.selectbox("Buy left signal", MA_RULE_SERIES, index=1, key="buy_left", label_visibility="collapsed")
            buy_operator = st.selectbox("Buy operator", MA_RULE_OPERATORS, index=4, key="buy_operator", label_visibility="collapsed")
            buy_right = st.selectbox("Buy right signal", MA_RULE_SERIES, index=2, key="buy_right", label_visibility="collapsed")

            st.markdown("**Sell / exit when**")
            sell_left = st.selectbox("Sell left signal", MA_RULE_SERIES, index=1, key="sell_left", label_visibility="collapsed")
            sell_operator = st.selectbox("Sell operator", MA_RULE_OPERATORS, index=5, key="sell_operator", label_visibility="collapsed")
            sell_right = st.selectbox("Sell right signal", MA_RULE_SERIES, index=2, key="sell_right", label_visibility="collapsed")
        elif setup == "breakout":
            breakout_lookback = int(st.number_input("Breakout lookback (bars)", min_value=2, max_value=250, value=20, step=1,
                                                    key="breakout_lookback"))
            volume_multiple = st.number_input("Breakout volume (x 50-bar average)", min_value=1.0, max_value=5.0, value=1.5, step=0.1,
                                              key="volume_multiple")
            exit_ma = int(st.number_input("Sell when the close is below this SMA (bars)", min_value=5, max_value=400,
                                          value=200, step=10, key="breakout_exit_ma"))
            st.caption(
                f"Buy: the first close above the prior {breakout_lookback}-bar high. Signal Analysis also needs "
                f"{volume_multiple:g}x average volume for a BUY (otherwise WATCH); the Backtest takes every breakout, "
                f"since the volume filter did not improve results in testing. Sell / exit (Backtest): the first close "
                f"below the {exit_ma}-bar SMA. A breakdown below the prior {breakout_lookback}-bar low is the bearish signal."
            )
        elif setup == "day_breakout":
            breakout_lookback = int(st.number_input("Breakout above the prior N-day high", min_value=1, max_value=250,
                                                    value=10, step=1, key="signal_day_breakout_lookback"))
            keep_fraction = exit_pct_slider("signal_day_breakout_exit_pct")
            st.caption(
                f"Buy: price trades above the prior {breakout_lookback}-day high (buy-stop at that level) and closes "
                f"within {(1 - keep_fraction) * 100:.0f}% of the day's range from its high. Sell / exit: the first close "
                "further below the high than that. Long only. The Day Breakout Rule tab runs it on many tickers with fixed $ size."
            )
        elif setup == "macd_bollinger":
            macd_ma = int(st.number_input("Moving average and Bollinger period (days)", min_value=5, max_value=100,
                                          value=20, step=1, key="macd_ma"))
            macd_bb_std = float(st.number_input(
                "Bollinger band width (standard deviations)", min_value=0.5, max_value=5.0, value=2.0, step=0.1,
                key="macd_bb_std",
                help="The bands sit this many standard deviations of the close above and below the SMA. 2 is the "
                "usual setting. Wider bands are drawn further out, but the widening test compares band width with "
                "itself N days before, so it gives the same buys at any setting."))
            widen_columns = st.columns(2)
            macd_widen_pct = float(widen_columns[0].number_input(
                "Bands at least this % wider", min_value=0.0, max_value=200.0, value=10.0, step=5.0,
                key="macd_widen_pct", help="Band width = (upper - lower) / SMA. 0% means just wider."))
            macd_widen_days = int(widen_columns[1].number_input(
                "than this many days before", min_value=1, max_value=20, value=5, step=1, key="macd_widen_days",
                help="1 day and 0% is the loosest reading: any uptick in width, even while the bands are still "
                "narrowing overall."))
            macd_stop_pct = float(st.number_input("Stop: sell when the close is this % below the entry (0 = off)",
                                                  min_value=0.0, max_value=30.0, value=3.0, step=0.5,
                                                  key="macd_stop_pct"))
            macd_exit_on_ma_down = st.checkbox(f"Also sell when the SMA {macd_ma} turns down", value=False,
                                               key="macd_exit_on_ma_down")
            macd_cooldown = int(st.number_input("Days to wait after a sell before buying again", min_value=0,
                                                max_value=20, value=1, step=1, key="macd_cooldown"))
            st.caption(
                f"Buy at the close when, on the same day, the MACD (12, 26, 9) histogram turns from negative to "
                f"positive, the Bollinger Bands ({macd_ma}, {macd_bb_std:g}) are at least {macd_widen_pct:g}% wider than {macd_widen_days} "
                f"day(s) before, and the close is above "
                f"the SMA {macd_ma}, which is rising. Sell / exit: the first close below the SMA {macd_ma}"
                + (f", or a close {macd_stop_pct:g}% below the entry" if macd_stop_pct else "")
                + (f", or the SMA {macd_ma} turning down" if macd_exit_on_ma_down else "")
                + f". Then wait {macd_cooldown} day(s) before the next buy. Long only, full position (the post's KD "
                "add-on and 20% sizing are not simulated here)."
            )
        elif setup == "opening_bar":
            opening_bar = opening_bar_settings(account_size, bar_minutes)
        elif setup == "pullback":
            trend_columns = st.columns(2)
            trend_mid = int(trend_columns[0].number_input("Trend fast MA (bars)", min_value=5, max_value=250, value=50, step=1, key="trend_mid"))
            trend_slow = int(trend_columns[1].number_input("Trend slow MA (bars)", min_value=10, max_value=400, value=200, step=1, key="trend_slow"))
            pullback_columns = st.columns(2)
            pullback_ema = int(pullback_columns[0].number_input("Pullback EMA (bars)", min_value=2, max_value=100, value=20, step=1, key="pullback_ema"))
            swing_bars = int(pullback_columns[1].number_input("Swing high lookback (bars)", min_value=2, max_value=120, value=20, step=1, key="swing_bars"))
            pullback_bars = int(pullback_columns[0].number_input("Pullback window (bars)", min_value=2, max_value=30, value=5, step=1, key="pullback_bars"))
            min_depth_atr = float(pullback_columns[1].number_input("Min dip depth (ATR)", min_value=0.0, max_value=10.0, value=1.5, step=0.1, key="min_depth_atr"))
            st.caption(
                f"Buy: {trend_mid}-bar MA rising and above the {trend_slow}-bar MA, a dip of at least {min_depth_atr:g} ATR "
                f"from the {swing_bars}-bar high reaching the {pullback_ema} EMA within the last {pullback_bars} bars, then a "
                "close above the prior high. Sell / exit: the mirror image in a downtrend."
            )
        else:
            breakout_lookback = int(st.number_input("Breakdown lookback (bars)", min_value=2, max_value=250, value=20, step=1,
                                                    key="breakdown_lookback"))
            trend_columns = st.columns(2)
            trend_mid = int(trend_columns[0].number_input("Trend fast MA (bars)", min_value=5, max_value=250, value=50, step=1, key="trend_mid"))
            trend_slow = int(trend_columns[1].number_input("Trend slow MA (bars)", min_value=10, max_value=400, value=200, step=1, key="trend_slow"))
            max_hold_bars = int(st.number_input("Sell after at most this many bars", min_value=1, max_value=60,
                                                value=10, step=1, key="reversal_max_hold"))
            st.caption(
                f"Buy: the first close below the prior {breakout_lookback}-bar low while the close is above the "
                f"{trend_slow}-bar MA and the {trend_mid}-bar MA is above it. Sell / exit (Backtest): the first close "
                f"above the prior bar's high, the bounce, or after {max_hold_bars} bars."
            )

        if setup == "ma_rule":
            preview = {"fast_type": fast_type, "fast_period": fast_period, "slow_type": slow_type,
                       "slow_period": slow_period, "buy": (buy_left, buy_operator, buy_right),
                       "sell": (sell_left, sell_operator, sell_right)}
            diagram = ma_rule_figure(f"{fast_type} {fast_period}", f"{slow_type} {slow_period}",
                                     describe_ma_rule(preview, "buy"), describe_ma_rule(preview, "sell"))
        elif setup == "breakout":
            diagram = breakout_figure(breakout_lookback, volume_multiple, exit_ma)
        elif setup == "pullback":
            diagram = pullback_figure(pullback_ema, trend_mid, trend_slow, min_depth_atr, swing_bars, pullback_bars)
        elif setup == "reversal":
            diagram = reversal_figure(breakout_lookback, trend_mid, trend_slow, max_hold_bars)
        elif setup == "macd_bollinger":
            diagram = macd_bollinger_figure(macd_ma, macd_stop_pct, macd_exit_on_ma_down, macd_widen_pct, macd_widen_days)
        elif setup == "opening_bar":
            diagram = opening_bar_figure(opening_bar.min_move, opening_bar.skip_bars, opening_bar.exit_mode,
                                         opening_bar.bar_minutes, opening_bar.custom_exit)
        else:
            diagram = day_breakout_figure(breakout_lookback, "top_range", keep_fraction)
        show_rule_diagram(diagram)

        # The Backtest is long only and the reversal and day-breakout rules have no short side. The first-bar day
        # trade has its own short setting.
        allow_short = bool(st.session_state.get("allow_short", False))
        if tab != "Backtest" and setup not in ("reversal", "day_breakout", "macd_bollinger", "opening_bar"):
            allow_short = st.checkbox("Allow short trades (otherwise sell signals mean exit)", value=False,
                                      key="allow_short")
        elif setup in ("reversal", "day_breakout", "macd_bollinger", "opening_bar"):
            allow_short = False

    commission_plan, slippage = "tiered", 0.01
    if tab == "Backtest":
        commission_plan, slippage = trading_cost_settings()
        if opening_bar is not None:
            opening_bar.commission_plan, opening_bar.slippage = commission_plan, slippage

    if setup == "opening_bar" and tab != "Backtest":
        st.info("The first-bar day trade trades on 5-minute bars, so it runs in the Backtest tab only.")
        return None

    # Sizing and confidence only shape the trade plan (Signal Analysis) and which replayed trades are taken
    # (Validate); the Backtest holds a full position on every signal. The intraday trigger cannot be replayed.
    risk_pct = float(st.session_state.get("risk_pct", 1.0))
    max_position_pct = float(st.session_state.get("max_position_pct", 25.0))
    full_size_confidence = st.session_state.get("full_size_confidence", 75)
    half_size_confidence = st.session_state.get("half_size_confidence", 60)
    opening_range_minutes = st.session_state.get("opening_range_minutes", 30)
    if tab != "Backtest":
        with settings_panel("Risk & sizing", expanded=False):
            if tab == SIGNAL_TAB:
                risk_pct = st.number_input("Risk per trade (% of equity)", min_value=0.1, max_value=5.0, value=1.0,
                                           step=0.1, key="risk_pct")
                max_position_pct = st.number_input("Max position (% of equity)", min_value=1.0, max_value=100.0,
                                                   value=25.0, step=1.0, key="max_position_pct")
            full_size_confidence = slider_with_input(
                "Full-size confidence threshold", 50, 95, 75, key="full_size_confidence",
                help="Confidence = 50 + 50 x the weighted score of signal quality, "
                "trend, volume, intraday, fundamentals and historical edge.")
            half_size_confidence = slider_with_input("Half-size confidence threshold", 40, 90, 60, key="half_size_confidence")
            if tab == SIGNAL_TAB:
                opening_range_minutes = st.selectbox(
                    "Intraday opening range (minutes)", [15, 30, 60], index=1, key="opening_range_minutes",
                    help="On the session after the daily signal, the entry is the first 5-minute close above this "
                    "opening range, the daily trigger and VWAP.")

    ma_rule = {
        "fast_type": fast_type,
        "fast_period": int(fast_period),
        "slow_type": slow_type,
        "slow_period": int(slow_period),
        "buy": (buy_left, buy_operator, buy_right),
        "sell": (sell_left, sell_operator, sell_right),
    }
    signal_settings = AnalysisSettings(
        setup=setup,
        allow_short=allow_short,
        breakout_lookback=breakout_lookback,
        volume_multiple=float(volume_multiple),
        opening_range_minutes=int(opening_range_minutes),
        ma_rule=ma_rule if setup == "ma_rule" else None,
        pullback_ema=pullback_ema,
        trend_mid=trend_mid,
        trend_slow=trend_slow,
        swing_bars=swing_bars,
        pullback_bars=pullback_bars,
        min_depth_atr=min_depth_atr,
        keep_fraction=keep_fraction,
        exit_ma=exit_ma,
        macd_ma=macd_ma,
        macd_stop_pct=macd_stop_pct,
        macd_exit_on_ma_down=macd_exit_on_ma_down,
        macd_widen_days=macd_widen_days,
        macd_widen_pct=macd_widen_pct,
        macd_bb_std=macd_bb_std,
    )
    risk_settings = RiskSettings(
        account_equity=float(account_size),
        risk_pct=float(risk_pct),
        max_position_pct=float(max_position_pct),
        full_size_confidence=float(max(full_size_confidence, half_size_confidence)),
        half_size_confidence=float(min(full_size_confidence, half_size_confidence)),
    )

    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info("Please select both a start date and an end date.")
        return None
    start_date, end_date = date_range
    if start_date > end_date:
        st.error("The start date must be earlier than the end date.")
        return None
    ticker = ticker.strip().upper()
    if not ticker:
        st.error("Please enter a ticker symbol.")
        return None

    return SimpleNamespace(
        ticker=ticker, timeframe=timeframe, start_date=start_date, end_date=end_date, account_size=account_size,
        setup=setup, setup_label=setup_label, fast_type=fast_type, fast_period=fast_period, slow_type=slow_type,
        slow_period=slow_period, buy_left=buy_left, buy_operator=buy_operator, buy_right=buy_right,
        sell_left=sell_left, sell_operator=sell_operator, sell_right=sell_right, breakout_lookback=breakout_lookback,
        volume_multiple=volume_multiple, ma_rule=ma_rule, signal_settings=signal_settings, risk_settings=risk_settings,
        max_hold_bars=max_hold_bars, opening_bar=opening_bar, macd_ma=macd_ma, macd_stop_pct=macd_stop_pct,
        macd_cooldown=macd_cooldown, macd_bb_std=macd_bb_std, commission_plan=commission_plan, slippage=slippage,
    )


# ============================================================
# Backtest tab
# ============================================================

# Seven stacked panels, each with its legend under it.
BACKTEST_CHART_HEIGHT = 2200
BACKTEST_PANEL_GAP_PX = 120


def rule_conditions(df, s):
    """Buy and sell conditions of the Backtest's signal rule (any rule but the first-bar day trade)."""

    if s.setup == "ma_rule":
        _, _, buy_condition, sell_condition = ma_rule_conditions(df, s.ma_rule)
        return buy_condition, sell_condition
    signals = detect_signals(df, s.setup, s.breakout_lookback, float(s.volume_multiple),
                             **s.signal_settings.detector_kwargs())
    buy_condition = signals["Signal_Up"].fillna(False).astype(bool)
    sell_column = "Exit_Long" if "Exit_Long" in signals else "Signal_Down"
    return buy_condition, signals[sell_column].fillna(False).astype(bool)


def run_rule_positions(df, buy_condition, sell_condition, s):
    """The position engine with the rule's own time exit, stop and wait after a sell."""

    macd_rule = s.setup == "macd_bollinger"
    return run_position_engine(df, buy_condition, sell_condition,
                               max_hold=s.max_hold_bars if s.setup == "reversal" else 0,
                               stop_pct=s.macd_stop_pct if macd_rule else 0.0,
                               cooldown=s.macd_cooldown if macd_rule else 0)


def strategy_with_params(s, params):
    """A copy of the strategy settings ``s`` with searched values (keyed by their Session State keys) in place."""

    s = SimpleNamespace(**vars(s))
    settings = {}
    for key, value in params.items():
        if key in ("fast_period", "slow_period", "fast_type", "slow_type"):
            setattr(s, key, value)
            s.ma_rule = {**s.ma_rule, key: value}
            settings["ma_rule"] = s.ma_rule
        elif key in ("breakout_lookback", "breakdown_lookback", "signal_day_breakout_lookback"):
            s.breakout_lookback = int(value)
            settings["breakout_lookback"] = int(value)
        elif key == "breakout_exit_ma":
            settings["exit_ma"] = int(value)
        elif key == "signal_day_breakout_exit_pct":
            settings["keep_fraction"] = 1 - value / 100
        elif key == "reversal_max_hold":
            s.max_hold_bars = int(value)
        elif key == "macd_cooldown":
            s.macd_cooldown = int(value)
        else:
            if key == "macd_stop_pct":
                s.macd_stop_pct = float(value)
            if key == "macd_ma":
                s.macd_ma = int(value)
            settings[key] = value
    s.signal_settings = dataclasses.replace(s.signal_settings, **settings)
    return s


def rule_backtest(price_df, s, params=None):
    """Equity (with costs) and buy markers of the signal rule over ``price_df``, optionally with searched values."""

    if params:
        s = strategy_with_params(s, params)
    buy_condition, sell_condition = rule_conditions(price_df, s)
    result = run_rule_positions(price_df, buy_condition, sell_condition, s)
    result["Equity"], _, _ = equity_with_costs(result, s.account_size, s.commission_plan, s.slippage)
    return result[["Close", "Equity", "Buy_Signal"]]


def describe_params(params, space):
    labels = {param.key: param.label for param in space.params}
    return ", ".join(f"{labels.get(key, key)} {value:g}" if isinstance(value, (int, float)) else f"{labels.get(key, key)} {value}"
                     for key, value in params.items())


def apply_params(params):
    """Button callback: put searched values into the rule's settings (and a slider's number box)."""

    for key, value in params.items():
        st.session_state[key] = value
        if f"{key}_input" in st.session_state:
            st.session_state[f"{key}_input"] = value


def optimizer_chart(trials, space, searched, picks):
    """Tuning score against each searched parameter: a broad band of high dots is a plateau, a lone dot a spike."""

    params = [param for param in space.params if param.key in searched]
    scored = trials.loc[trials["status"].eq("ok")]
    columns = 1 if MOBILE or len(params) == 1 else 2
    rows = -(-len(params) // columns)
    fig = make_subplots(rows=rows, cols=columns, subplot_titles=[param.label for param in params],
                        vertical_spacing=min(0.5 / max(rows, 1), 0.12), horizontal_spacing=0.08)
    for number, param in enumerate(params):
        row, col = number // columns + 1, number % columns + 1
        fig.add_trace(go.Scatter(
            x=scored[param.key], y=scored["score"], mode="markers", name="Trial", showlegend=number == 0,
            marker=dict(color=scored["trial"], colorscale="Blues", cmin=0, cmax=max(scored["trial"].max(), 1),
                        size=8, line=dict(color="gray", width=0.5)),
            customdata=scored["trial"],
            hovertemplate=f"{param.label}: %{{x}}<br>Tuning score: %{{y:.2f}}<br>Trial %{{customdata}}<extra></extra>",
        ), row=row, col=col)
        for label, pick, color, symbol in picks:
            if pick is None:
                continue
            fig.add_trace(go.Scatter(
                x=[pick[param.key]], y=[pick["score"]], mode="markers", name=label, showlegend=number == 0,
                marker=dict(color=color, size=19 if symbol == "diamond" else 14, symbol=symbol,
                            line=dict(color="black", width=1)),
                hovertemplate=f"{label}<br>{param.label}: %{{x}}<br>Tuning score: %{{y:.2f}}<extra></extra>",
            ), row=row, col=col)
        fig.update_yaxes(title_text="Tuning score" if col == 1 else None, row=row, col=col)
    # On phones the chart toolbar sits over the top, so leave room above the first panel's title.
    fig.update_layout(height=max(320, 260 * rows) + (40 if MOBILE else 0),
                      margin=dict(t=80 if MOBILE else 40, b=60, l=50, r=20), hovermode="closest",
                      legend=dict(orientation="h", y=-0.08 if rows > 1 else -0.2, x=0))
    return fig


def render_optimizer(price_df, s):
    """Bayesian search for the rule's settings on the first part of the dates, forward-tested on the rest."""

    space = SEARCH_SPACES.get(s.setup)
    if space is None:
        st.caption("The parameter search works on the daily signal rules, not the first-bar day trade.")
        return
    st.caption(
        "Optuna's TPE sampler (a form of Bayesian optimization) tries settings, learns which areas score well and "
        "tries more there. It sees only the **tuning period** (the first part of the date range). The **forward "
        "period** after it is scored once, after the search, as a test of the settings it picked. Costs are included.")
    labels = {param.label: param.key for param in space.params}
    with st.form("optimizer_form"):
        chosen = st.multiselect("Settings to search (the others stay as set)", list(labels), default=list(labels),
                                key=f"optimizer_params_{s.setup}")
        columns = st.columns(2)
        objective_label = columns[0].selectbox(
            "Score to maximise", list(OBJECTIVES), key="optimizer_objective",
            help="Sharpe: return per unit of volatility. Sortino: the same, counting only falls. Return / max "
            "drawdown: annual return divided by the worst fall from a peak. Custom: your own weighted mix, set under "
            "Custom score weights.")
        n_trials = int(columns[1].number_input("Trials", min_value=10, max_value=300, value=60, step=10,
                                               key="optimizer_trials",
                                               help="Each trial is one backtest. More trials search better but "
                                               "also make a lucky result more likely."))
        tune_pct = int(columns[0].number_input("Tuning period (% of the dates)", min_value=40, max_value=90, value=70,
                                               step=5, key="optimizer_tune_pct"))
        min_trades = int(columns[1].number_input("Min trades in the tuning period", min_value=1, max_value=100,
                                                 value=5, step=1, key="optimizer_min_trades",
                                                 help="Settings with fewer trades are skipped: a handful of trades "
                                                 "says little."))
        start_label = columns[0].selectbox(
            "Starting points", ["Latin hypercube", "Random"], key="optimizer_start",
            help=f"The first {N_STARTUP_TRIALS} trials map the ground before the model guides the search. Latin "
            "hypercube cuts each setting's range into equal slices and uses every slice once, so the start covers "
            "each range evenly; random picks can clump by chance.")
        max_dd_limit = float(columns[1].number_input(
            "Skip settings with drawdown over (%, 0 = off)", min_value=0.0, max_value=100.0, value=0.0, step=5.0,
            key="optimizer_max_dd", help="A hard limit: settings whose tuning-period drawdown is deeper are not "
            "scored, whatever their return."))
        with st.expander("Custom score weights (used when the score is Custom)"):
            st.caption("Score = the sum of weight x term over the tuning period; max drawdown is subtracted. "
                       "Ratios are around 0-3 and percentages around 0-50, so weight percentages lightly "
                       "(e.g. 0.02 x max drawdown % costs 0.4 for a 20% drawdown).")
            weight_columns = st.columns(2)
            custom_weights = {}
            defaults = {"sharpe": 1.0, "sortino": 0.0, "calmar": 0.0, "cagr_pct": 0.0, "max_dd_pct": 0.02}
            for number, (term, name) in enumerate(SCORE_TERMS.items()):
                custom_weights[term] = float(weight_columns[number % 2].number_input(
                    f"{name}{' (penalty)' if term in PENALTY_TERMS else ''}", min_value=0.0, max_value=100.0,
                    value=defaults[term], step=0.01 if term in ("cagr_pct", "max_dd_pct") else 0.1,
                    format="%.2f", key=f"optimizer_weight_{term}"))
        submitted = st.form_submit_button("Run search", type="primary", key="optimizer_run_btn")
    weights = OBJECTIVES[objective_label] or custom_weights
    st.latex(objective_latex(weights))

    searched = [labels[label] for label in chosen]
    signature = json.dumps([s.ticker, s.timeframe, str(s.start_date), str(s.end_date), s.setup], default=str)
    if submitted:
        if not searched:
            st.warning("Pick at least one setting to search.")
            return
        current = current_param_values(s, space)
        fixed = {key: value for key, value in current.items() if key not in searched}
        tune_end = split_date(price_df.index, tune_pct / 100)
        periods = PERIODS_PER_YEAR.get(s.timeframe, 252)
        progress = st.progress(0.0, text="Searching...")
        trials = run_search(
            lambda params: rule_backtest(price_df, s, params), space, price_df.index[0], tune_end,
            objective=weights, n_trials=n_trials, min_trades=min_trades, fixed=fixed, periods_per_year=periods,
            start="lhs" if start_label == "Latin hypercube" else "random", max_dd_limit=max_dd_limit or None,
            progress=lambda done, total: progress.progress(done / total, text=f"Trial {done} of {total}"))
        progress.empty()
        st.session_state["optimizer_result"] = {
            "signature": signature, "trials": trials, "searched": searched, "current": current,
            "weights": weights, "objective_label": objective_label, "tune_end": tune_end,
            "periods": periods, "fixed": fixed, "tune_pct": tune_pct, "min_trades": min_trades,
            "n_startup": N_STARTUP_TRIALS, "seed": 0, "start": start_label, "max_dd_limit": max_dd_limit,
        }

    result = st.session_state.get("optimizer_result")
    if not result or result["signature"] != signature or "weights" not in result:
        return
    trials, weights, tune_end, periods = result["trials"], result["weights"], result["tune_end"], result["periods"]
    forward = price_df.index[price_df.index > tune_end]
    if forward.empty:
        st.warning("The forward period is empty; use a longer date range or a smaller tuning share.")
        return
    tune_window = (price_df.index[0], tune_end)
    forward_window = (forward[0], price_df.index[-1])
    best, stable = pick_settings(trials, space)
    scored = int(trials["status"].eq("ok").sum())
    st.markdown(
        f"**{len(trials)} trials, {scored} scored** on {result['objective_label']}. Tuning period "
        f"{tune_window[0]:%Y-%m-%d} to {tune_window[1]:%Y-%m-%d}; forward period {forward_window[0]:%Y-%m-%d} to "
        f"{forward_window[1]:%Y-%m-%d}.")
    if stable is None:
        st.warning("No trial had enough trades to score. Lower the minimum trades or widen the dates.")
        return

    buy_hold = price_df[["Close"]].copy()
    buy_hold["Equity"] = buy_hold["Close"] / buy_hold["Close"].iloc[0]
    buy_hold["Buy_Signal"] = False
    candidates = [
        ("Most stable (plateau)", params_of(stable, space)),
        ("Best single trial", params_of(best, space)),
        ("Your current settings", result["current"]),
    ]
    rows = []
    for label, params in candidates:
        backtest = rule_backtest(price_df, s, params)
        rows.append((label, describe_params({key: params[key] for key in result["searched"]}, space), backtest))
    rows.append(("Buy & hold", "", buy_hold))
    table = []
    for label, text, backtest in rows:
        tune = period_metrics(backtest, *tune_window, periods)
        ahead = period_metrics(backtest, *forward_window, periods)
        # Tuning and forward side by side, so the drop from one to the other reads at a glance.
        table.append({
            "": label, "Tuning score": objective_score(tune, weights), "Forward score": objective_score(ahead, weights),
            "Tuning return %": tune["return"] * 100, "Forward return %": ahead["return"] * 100,
            "Tuning max DD %": tune["max_dd"] * 100, "Forward max DD %": ahead["max_dd"] * 100,
            "Tuning trades": tune["trades"], "Forward trades": ahead["trades"], "Settings": text,
        })
    table = pd.DataFrame(table)
    for column in ("Tuning trades", "Forward trades"):
        table.loc[table[""].eq("Buy & hold"), column] = None
    st.dataframe(table, hide_index=True, width="stretch", key="optimizer_results_table", column_config={
        column: st.column_config.NumberColumn(format="%.2f") for column in table.columns
        if column.endswith("score") or column.endswith("%")})
    st.caption(
        "**Most stable** is the trial whose own score and its 5 nearest neighbours' scores average highest: a "
        "setting on a plateau, where nearby values also work. **Best single trial** is the top score, often a lucky "
        "spike. Trust a setting only if its forward score holds up against your current settings and buy & hold. "
        f"With {len(trials)} trials, the best tuning score is flattered by luck, so expect the forward score to be lower.")
    buttons = st.columns(2)
    buttons[0].button("Use the most stable settings", on_click=apply_params, args=(params_of(stable, space),),
                      key="optimizer_apply_stable_btn")
    buttons[1].button("Use the best single trial", on_click=apply_params, args=(params_of(best, space),),
                      key="optimizer_apply_best_btn")
    # Best first and larger, so the star stays visible on top when both picks are the same trial.
    picks = [("Best single trial", best, "crimson", "diamond"), ("Most stable", stable, "gold", "star")]
    searched = result["searched"]
    if len(searched) >= 2:
        labels = {param.key: param.label for param in space.params}
        # Default axes: the two settings that mattered most, when that can be told.
        importance = param_importance(trials, space, searched)
        ranked = sorted(searched, key=lambda key: -importance.get(key, 0)) if importance else searched
        axis_columns = st.columns(2)
        x_key = axis_columns[0].selectbox("Heat map across", ranked, index=0, format_func=labels.get,
                                          key="optimizer_heat_x")
        y_options = [key for key in ranked if key != x_key]
        y_key = axis_columns[1].selectbox("and", y_options, index=0, format_func=labels.get, key="optimizer_heat_y")
        show_chart(optimizer_heatmap(trials, space, x_key, y_key, picks), width="stretch", key="optimizer_heatmap")
        st.caption(
            "Each cell is the average tuning score of the trials that landed in it (blank: none tried there). The "
            "dots are the trials; where they bunch is where the search spent its time, the area it found "
            "promising. Trust a broad block of good colour, not a lone bright cell. The other searched settings vary "
            "between the trials in a cell.")
    else:
        show_chart(optimizer_chart(trials, space, searched, picks), width="stretch", key="optimizer_chart")
    with st.expander("Advanced: how the search ran"):
        render_optimizer_details(result, space, stable, best, tune_window, forward_window, picks)


def _heatmap_bins(param, values):
    """Cell labels and the cell of each value along one heat-map axis (at most 10 cells for a numeric setting)."""

    if param.kind == "choice":
        labels = list(param.choices)
        return labels, np.array([labels.index(value) for value in values]), labels
    steps = int(round((param.high - param.low) / param.step)) + 1
    count = min(10, steps)
    edges = np.linspace(param.low - param.step / 2, param.high + param.step / 2, count + 1)
    cells = np.clip(np.searchsorted(edges, np.asarray(values, dtype=float), side="right") - 1, 0, count - 1)
    centers = (edges[:-1] + edges[1:]) / 2
    ranges = [f"{max(lo, param.low):g} to {min(hi, param.high):g}" for lo, hi in zip(edges[:-1], edges[1:])]
    return list(centers), cells, ranges


def optimizer_heatmap(trials, space, x_key, y_key, picks):
    """Average tuning score per cell of two settings, with every trial as a dot and the two picks marked."""

    params = {param.key: param for param in space.params}
    scored = unique_scored(trials)
    x_param, y_param = params[x_key], params[y_key]
    x_centers, x_cells, x_ranges = _heatmap_bins(x_param, scored[x_key])
    y_centers, y_cells, y_ranges = _heatmap_bins(y_param, scored[y_key])
    total = np.zeros((len(y_centers), len(x_centers)))
    count = np.zeros_like(total)
    for xi, yi, score in zip(x_cells, y_cells, scored["score"].to_numpy(dtype=float)):
        total[yi, xi] += score
        count[yi, xi] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    hover = [[f"{x_param.label}: {x_ranges[xi]}<br>{y_param.label}: {y_ranges[yi]}<br>"
              + (f"Average score {mean[yi, xi]:.2f} over {int(count[yi, xi])} trial(s)" if count[yi, xi] else
                 "No trials here")
              for xi in range(len(x_centers))] for yi in range(len(y_centers))]
    fig = go.Figure(go.Heatmap(
        x=x_centers, y=y_centers, z=mean, text=hover, hoverinfo="text", colorscale="Viridis", hoverongaps=True,
        colorbar=dict(title=dict(text="Score", side="right"), thickness=12)))
    fig.add_trace(go.Scatter(
        x=scored[x_key], y=scored[y_key], mode="markers", name="Trial",
        marker=dict(color="white", size=7, line=dict(color="black", width=1), opacity=0.85),
        customdata=np.column_stack([scored["trial"], scored["score"]]),
        hovertemplate=f"Trial %{{customdata[0]}}<br>{x_param.label}: %{{x}}<br>{y_param.label}: %{{y}}<br>"
                      "Score %{customdata[1]:.2f}<extra></extra>"))
    for label, pick, color, symbol in picks:
        if pick is None:
            continue
        fig.add_trace(go.Scatter(
            x=[pick[x_key]], y=[pick[y_key]], mode="markers", name=label,
            marker=dict(color=color, size=19 if symbol == "diamond" else 14, symbol=symbol,
                        line=dict(color="black", width=1)),
            hovertemplate=f"{label}<br>{x_param.label}: %{{x}}<br>{y_param.label}: %{{y}}<extra></extra>"))
    fig.update_layout(height=470 if MOBILE else 440, margin=dict(t=60 if MOBILE else 20, b=50, l=60, r=20),
                      hovermode="closest", yaxis=dict(title=y_param.label),
                      # Phones: show_chart puts the legend right under the axis, so the x title (the "Heat map
                      # across" choice above) is left off there.
                      xaxis=dict(title=None if MOBILE else x_param.label), legend=dict(orientation="h", y=-0.2, x=0))
    return fig


def render_optimizer_details(result, space, stable, best, tune_window, forward_window, picks):
    """How the score is defined, the search's setup, its progress trial by trial, which settings mattered, and why
    the stable pick won."""

    trials, searched, weights = result["trials"], result["searched"], result["weights"]
    st.markdown("**How the score is defined**")
    st.latex(objective_latex(weights))
    for term, weight in weights.items():
        if weight:
            st.latex(TERM_FORMULAS[term])
    limit = result.get("max_dd_limit")
    st.caption(
        f"All terms are measured on the strategy's equity over the tuning period only, after costs; r = its "
        f"{'daily' if result['periods'] == 252 else 'bar'} returns, N = {result['periods']} bars a year, E = equity. "
        f"A setting gets no score (and the search is steered away) if it trades fewer than {result['min_trades']} "
        "times" + (f" or falls more than {limit:g}% from a peak" if limit else "") + ". Change the score under "
        "Score to maximise; Custom lets you weight the terms yourself.")
    st.markdown("**Setup**")
    setup_rows = []
    for param in space.params:
        if param.kind == "choice":
            values = " / ".join(param.choices)
        else:
            values = f"{param.low:g} to {param.high:g}, step {param.step:g}"
        setup_rows.append({"Setting": param.label, "Searched": "Yes" if param.key in searched else "No (held)",
                           "Range tried": values if param.key in searched else "",
                           "Held at": "" if param.key in searched else str(result["fixed"].get(param.key, ""))})
    st.dataframe(pd.DataFrame(setup_rows), hide_index=True, width="stretch", key="optimizer_setup_table")
    repeats = int(trials["repeat_of"].notna().sum()) if "repeat_of" in trials else 0
    skipped = trials.loc[trials["status"].ne("ok"), "status"].value_counts()
    skipped_text = ("; skipped: " + ", ".join(f"{count} {reason}" for reason, count in skipped.items())
                    if not skipped.empty else "; none skipped")
    st.caption(
        f"{len(trials)} trials tried {len(trials) - repeats} distinct settings ({repeats} repeated an earlier trial; "
        "TPE re-proposes settings it likes, and a repeat is not backtested again or counted twice when picking). "
        f"Sampler: Optuna TPE, {result.get('start', 'Random')} start, seed {result['seed']}. Score: "
        f"{result['objective_label']} over the tuning period "
        f"({tune_window[0]:%Y-%m-%d} to {tune_window[1]:%Y-%m-%d}, {result['tune_pct']}% of the bars), costs "
        f"included. Trials need at least {result['min_trades']} trades to count{skipped_text}. Forward period "
        f"{forward_window[0]:%Y-%m-%d} to {forward_window[1]:%Y-%m-%d}, scored only after the search.")

    st.markdown("**How the search moved, trial by trial**")
    start = result.get("start", "Random")
    start_text = ("are a Latin hypercube: each setting's range is cut into equal slices and every slice is used "
                  "once, so the ground is mapped evenly" if start == "Latin hypercube" else
                  "pick values at random, to map the ground")
    st.caption(
        f"The first {result['n_startup']} trials {start_text}. After that TPE splits "
        "the trials so far into the best ~10% and the rest, and picks the next values where the best trials are "
        "common and the rest are rare. Dots bunching up near the top as the trials go on means it found an area "
        "worth exploring. A best-so-far line that stops rising early means more trials would add little.")
    show_chart(optimizer_history_chart(trials, result["n_startup"]), width="stretch", key="optimizer_history_chart")

    importance = param_importance(trials, space, searched)
    if importance:
        st.markdown("**Which settings mattered**")
        st.caption("Share of the score's variation explained by each setting (Optuna's PED-ANOVA importance, "
                   "measured on the better trials). A setting near 0 barely changed the result; you could hold it "
                   "fixed and search the others more closely.")
        labels = {param.key: param.label for param in space.params}
        ordered = sorted(importance.items(), key=lambda item: item[1])
        fig = go.Figure(go.Bar(x=[value for _, value in ordered], y=[labels[key] for key, _ in ordered],
                               orientation="h", marker_color="steelblue",
                               hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
        fig.update_layout(height=80 + 45 * len(ordered) + (40 if MOBILE else 0),
                          margin=dict(t=60 if MOBILE else 20, b=40, l=10, r=20),
                          xaxis=dict(tickformat=".0%", title="Importance"))
        show_chart(fig, width="stretch", key="optimizer_importance_chart")
    elif len(searched) >= 2:
        st.caption("Too few scored trials to say which settings mattered; run at least 10.")

    if len(searched) >= 2:
        st.markdown("**Score against each setting**")
        st.caption("Every trial's tuning score against one setting at a time. A broad band of high dots is a "
                   "plateau; a lone high dot is a spike.")
        show_chart(optimizer_chart(trials, space, searched, picks), width="stretch", key="optimizer_chart")

    st.markdown("**Why the most stable pick**")
    neighbours = nearest_trials(trials, space, stable)
    labels = {param.key: param.label for param in space.params}
    # The pick itself first (distance 0), then its neighbours.
    itself = stable.to_frame().T.assign(distance=0.0)
    table = pd.concat([itself, neighbours])[["trial", *searched, "score", "distance"]]
    table["trial"] = table["trial"].astype(int).astype(str)
    table.iloc[0, 0] = f"{table.iloc[0, 0]} (pick)"
    table = table.rename(
        columns={"trial": "Trial", "score": "Tuning score", "distance": "Distance", **labels})
    st.caption(
        f"Trial {int(stable['trial'])} scored {stable['score']:.2f}. Its {len(neighbours)} nearest trials (distance "
        "0 = the same values, 1 = one setting across its whole range) average with it to "
        f"{stable['neighbourhood']:.2f}, the highest of any trial. The best single trial, {int(best['trial'])}, "
        f"scored {best['score']:.2f} with a neighbourhood average of {best['neighbourhood']:.2f}.")
    st.dataframe(table, hide_index=True, width="stretch", key="optimizer_neighbours_table",
                 column_config={"Tuning score": st.column_config.NumberColumn(format="%.2f"),
                                "Distance": st.column_config.NumberColumn(format="%.2f")})

    st.markdown("**All trials**")
    shown = trials.copy()
    for column in ("return", "cagr", "max_dd"):
        if column in shown:
            shown[column] = shown[column] * 100
    shown = shown.rename(columns={"trial": "Trial", "phase": "Phase", "score": "Tuning score",
                                  "best_so_far": "Best so far", "return": "Return %", "cagr": "CAGR %",
                                  "max_dd": "Max DD %", "sharpe": "Sharpe", "calmar": "Return / max DD",
                                  "trades": "Trades", "status": "Status", "repeat_of": "Repeat of trial"})
    shown = shown.rename(columns=labels)
    two_decimals = ["Tuning score", "Best so far", "Return %", "CAGR %", "Max DD %", "Sharpe", "Return / max DD"]
    st.dataframe(shown, hide_index=True, width="stretch", key="optimizer_trials_table", column_config={
        **{column: st.column_config.NumberColumn(format="%.2f") for column in two_decimals},
        "Repeat of trial": st.column_config.NumberColumn(format="%d")})


def optimizer_history_chart(trials, n_startup):
    """Tuning score of each trial in order, with the best score so far as a step line."""

    scored = trials.loc[trials["status"].eq("ok")]
    fig = go.Figure()
    if n_startup:
        fig.add_vrect(x0=0.5, x1=min(n_startup, len(trials)) + 0.5, fillcolor="lightgray", opacity=0.3,
                      line_width=0, annotation_text="start", annotation_position="top left")
    repeat = scored["repeat_of"].notna() if "repeat_of" in scored else pd.Series(False, index=scored.index)
    for phase, color in (("Latin hypercube start", "gray"), ("random start", "gray"), ("guided", "steelblue")):
        points = scored.loc[scored["phase"].eq(phase) & ~repeat]
        if points.empty:
            continue
        fig.add_trace(go.Scatter(x=points["trial"], y=points["score"], mode="markers", name=phase.capitalize(),
                                 marker=dict(color=color, size=8),
                                 hovertemplate="Trial %{x}<br>Tuning score %{y:.2f}<extra></extra>"))
    repeats = scored.loc[repeat]
    if not repeats.empty:
        fig.add_trace(go.Scatter(x=repeats["trial"], y=repeats["score"], mode="markers", name="Repeat of an earlier trial",
                                 marker=dict(color="white", size=8, line=dict(color="steelblue", width=1.5)),
                                 customdata=repeats["repeat_of"],
                                 hovertemplate="Trial %{x} repeats trial %{customdata}<br>Tuning score %{y:.2f}"
                                 "<extra></extra>"))
    best = trials[["trial", "best_so_far"]].dropna()
    fig.add_trace(go.Scatter(x=best["trial"], y=best["best_so_far"], mode="lines", name="Best so far",
                             line=dict(color="crimson", width=2, shape="hv"),
                             hovertemplate="After trial %{x}: best %{y:.2f}<extra></extra>"))
    # Phones: room above for the chart toolbar and below for the legend, which wraps to two lines.
    # Phones: room above for the chart toolbar; show_chart puts the legend right under the axis there, so the
    # axis gets no title (the hover box says "Trial").
    fig.update_layout(height=430 if MOBILE else 360, margin=dict(t=60 if MOBILE else 30, b=50, l=50, r=20),
                      hovermode="closest", xaxis=dict(title=None if MOBILE else "Trial"),
                      yaxis=dict(title="Tuning score"), legend=dict(orientation="h", y=-0.25, x=0))
    return fig


def current_param_values(s, space):
    """The rule's present settings, keyed like the search space."""

    settings = s.signal_settings
    values = {
        "fast_period": s.fast_period, "slow_period": s.slow_period, "fast_type": s.fast_type, "slow_type": s.slow_type,
        "breakout_lookback": s.breakout_lookback, "breakdown_lookback": s.breakout_lookback,
        "signal_day_breakout_lookback": s.breakout_lookback, "breakout_exit_ma": settings.exit_ma,
        "signal_day_breakout_exit_pct": int(round((1 - settings.keep_fraction) * 100)),
        "reversal_max_hold": s.max_hold_bars, "trend_mid": settings.trend_mid, "trend_slow": settings.trend_slow,
        "pullback_ema": settings.pullback_ema, "swing_bars": settings.swing_bars,
        "pullback_bars": settings.pullback_bars, "min_depth_atr": settings.min_depth_atr,
        "macd_ma": s.macd_ma, "macd_widen_pct": settings.macd_widen_pct, "macd_widen_days": settings.macd_widen_days,
        "macd_stop_pct": s.macd_stop_pct, "macd_cooldown": s.macd_cooldown,
    }
    return {param.key: values[param.key] for param in space.params}


def render_backtest(s):
    """Moving-average / signal-rule backtest of the chosen ticker with EPS, fundamentals, P/E and benchmark panels."""

    ticker, timeframe, start_date, end_date = s.ticker, s.timeframe, s.start_date, s.end_date
    account_size, setup, setup_label, ma_rule = s.account_size, s.setup, s.setup_label, s.ma_rule
    fast_type, fast_period, slow_type, slow_period = s.fast_type, s.fast_period, s.slow_type, s.slow_period
    buy_left, buy_operator, buy_right = s.buy_left, s.buy_operator, s.buy_right
    sell_left, sell_operator, sell_right = s.sell_left, s.sell_operator, s.sell_right
    breakout_lookback, volume_multiple, signal_settings = s.breakout_lookback, s.volume_multiple, s.signal_settings
    intraday_bars = None
    if setup == "opening_bar":
        # The day trade runs on intraday bars; the daily data below only feeds the EPS, fundamentals and benchmark
        # panels over the same days.
        timeframe = "Daily"
        with st.spinner(f"Loading {ticker} {s.opening_bar.bar_minutes}-minute bars..."):
            intraday_bars = load_opening_bar_bars(ticker, s.opening_bar.bar_minutes)
        if intraday_bars.empty:
            st.error(f"No {s.opening_bar.bar_minutes}-minute bars were returned for {ticker}.")
            return
        start_date = intraday_bars.index.min().date()
        end_date = intraday_bars.index.max().date() + timedelta(days=1)

    start_date_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_date_str = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    earnings_lookback_start = (
        pd.Timestamp(start_date) - pd.DateOffset(years=2)
    ).strftime("%Y-%m-%d")

    with st.spinner(f"Loading {ticker} data from {start_date_str} to {end_date_str}..."):
        df = load_data(ticker, start_date_str, end_date_str, timeframe)
        earnings_history_df = load_earnings_data(ticker, earnings_lookback_start, end_date_str)
        fundamentals_df, fundamentals_sec_error, fundamentals_source = load_quarterly_fundamentals(
            ticker,
            start_date_str,
            end_date_str,
        )
        benchmark_df = load_benchmark_data(start_date_str, end_date_str, timeframe)

    if df.empty:
        st.error(f"No price data was returned for {ticker} between {start_date_str} and {end_date_str}.")
        return

    st.caption(
        f"Loaded {len(df):,} {timeframe.lower()} bars: "
        f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}"
    )


    if fundamentals_df.empty:
        st.warning(
            "No quarterly fundamentals were available for the selected period. "
            f"SEC diagnostic: {fundamentals_sec_error or 'No SEC error reported.'}"
        )
    else:
        st.caption(
            f"Quarterly fundamentals source: {fundamentals_source}. "
            f"Loaded {len(fundamentals_df)} quarter-end records from "
            f"{fundamentals_df.index.min():%Y-%m-%d} to "
            f"{fundamentals_df.index.max():%Y-%m-%d}."
        )
        if fundamentals_sec_error:
            st.caption(f"SEC fallback diagnostic: {fundamentals_sec_error}")

    # The plain bars, before the chart's columns are added: the parameter search backtests on these.
    price_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["Fast_MA"] = compute_ma(df["Close"], fast_type, fast_period)
    df["Slow_MA"] = compute_ma(df["Close"], slow_type, slow_period)

    if setup == "ma_rule" and len(df) < max(fast_period, slow_period):
        st.warning(
            f"The selected period contains only {len(df)} bars, but the longest "
            f"moving average requires {max(fast_period, slow_period)} bars."
        )

    valuation_df, earnings_calculated_df = build_valuation_data(df, earnings_history_df)

    if earnings_calculated_df.empty:
        earnings_display_df = pd.DataFrame()
    else:
        display_start = pd.Timestamp(start_date).normalize()
        display_end = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
        earnings_display_df = earnings_calculated_df.loc[
            (earnings_calculated_df.index >= display_start)
            & (earnings_calculated_df.index < display_end)
        ].copy()

    # --------------------------------------------------------
    # Align quarterly fundamentals with EPS announcement dates
    # --------------------------------------------------------
    fundamentals_plot_df = (
        align_fundamentals_to_earnings_dates(
            fundamentals_df=fundamentals_df,
            earnings_df=earnings_calculated_df,
            maximum_days=120,
        )
    )

    pe_error = valuation_df["PE_Reconstruction_Error"].dropna()
    if not pe_error.empty:
        st.caption(f"Maximum internal P/E calculation error: ${pe_error.max():.8f}")

    if setup == "opening_bar":
        trades, days = run_opening_bar({ticker: intraday_bars}, s.opening_bar)
        df, rule_text = opening_bar_equity(intraday_bars, trades, s.opening_bar, account_size), opening_bar_rule_text(s.opening_bar)
    elif setup == "ma_rule":
        buy_condition, sell_condition = rule_conditions(df, s)
        rule_text = (
            f"**BUY:** `{buy_left} {buy_operator} {buy_right}`\n\n"
            f"**SELL:** `{sell_left} {sell_operator} {sell_right}`"
        )
    else:
        buy_condition, sell_condition = rule_conditions(df, s)
        rule_text = (
            f"**BUY / SELL:** {setup_label} signals on {timeframe.lower()} bars. The position is held until "
            "its exit (see the rule's settings). Stops and targets are not simulated here; the **Validate** tab replays the "
            "full trade plan."
        )
        if setup == "macd_bollinger":
            rule_text = (
                f"**BUY / SELL:** {setup_label} on {timeframe.lower()} bars, buying and selling at the close. "
                f"Sells on the first close below the SMA {s.macd_ma}"
                + (f" or a close {s.macd_stop_pct:g}% below the entry" if s.macd_stop_pct else "")
                + f", then waits {s.macd_cooldown} day(s). Full position."
            )
    if setup != "opening_bar":
        df = run_rule_positions(df, buy_condition, sell_condition, s)

        df["Returns"] = df["Close"].pct_change().fillna(0.0)
        df["Equity"], total_costs, orders = equity_with_costs(df, account_size, s.commission_plan, s.slippage)
        df["Strat_Returns"] = df["Equity"].pct_change().fillna(0.0).where(df["Position"].eq(1) | df["Sell_Signal"]
                                                                           | df["Buy_Signal"], 0.0)
        rule_text += (f"\n\n**Costs:** ${total_costs:,.2f} over {orders} orders (broker fees + "
                      f"${s.slippage:.2f} a share slippage on each buy and sell).")
        df["Peak"] = df["Equity"].cummax()
        df["Drawdown"] = df["Equity"] / df["Peak"] - 1.0

    final_equity = df["Equity"].iloc[-1]
    total_return = (final_equity / float(account_size) - 1.0) * 100.0
    max_drawdown = df["Drawdown"].min() * 100.0
    active_returns = df.loc[df["Strat_Returns"] != 0, "Strat_Returns"]
    positive_active_bars = 0.0 if active_returns.empty else active_returns.gt(0).mean() * 100.0

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Final Equity", f"${final_equity:,.2f}", f"{total_return:.2f}%")
    metric_col2.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    if setup == "opening_bar":
        win_rate = (trades["P&L"] > 0).mean() * 100 if len(trades) else 0.0
        metric_col3.metric("Win rate", f"{win_rate:.0f}%", f"{len(trades)} trades", delta_color="off")
    else:
        metric_col3.metric("Positive Active Bars", f"{positive_active_bars:.1f}%")
    st.info(rule_text)
    if setup != "opening_bar":
        with st.expander("Optimization"):
            render_optimizer(price_df, s)

    fig = make_subplots(
        rows=7,
        cols=1,
        shared_xaxes=True,
        # Room under each panel for its legend and the next title; the day trade's price panel also has its
        # own time labels.
        vertical_spacing=BACKTEST_PANEL_GAP_PX / BACKTEST_CHART_HEIGHT,
        subplot_titles=(
            f"{ticker} Price, Indicators and Trade Signals",
            "Portfolio Equity Curve ($)",
            "Drawdown Profile (%)",
            f"{ticker} Quarterly and TTM EPS",
            f"{ticker} Quarterly Revenue, OpEx, Total Expense, CapEx and Net Income",
            f"{ticker} Trailing P/E and Estimate-Based P/E Proxy",
            "Market Benchmark Performance, Normalized to 100",
        ),
        row_heights=[0.27, 0.11, 0.07, 0.14, 0.16, 0.13, 0.12],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )

    # Row 1: price, moving averages and signals
    if setup == "opening_bar":
        # Every intraday bar as a candle, so the signals can be read against the bars they came from.
        fig.add_trace(go.Candlestick(
            x=bar_labels(df.index), open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name=f"{s.opening_bar.bar_minutes}-min bars", increasing_line_color="seagreen",
            decreasing_line_color="indianred", increasing_line_width=1, decreasing_line_width=1,
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="gray", width=1)), row=1, col=1)
    if setup == "macd_bollinger":
        rule_sma = df["Close"].rolling(s.macd_ma).mean()
        rule_band = s.macd_bb_std * df["Close"].rolling(s.macd_ma).std()
        fig.add_trace(go.Scatter(x=df.index, y=rule_sma, name=f"SMA {s.macd_ma}", line=dict(color="orange", width=1.5)), row=1, col=1)
        for band, label in [(rule_sma + rule_band, "Upper band"), (rule_sma - rule_band, "Lower band")]:
            fig.add_trace(go.Scatter(x=df.index, y=band, name=label, line=dict(color="royalblue", width=1, dash="dot")), row=1, col=1)
    if setup == "ma_rule":
        fig.add_trace(go.Scatter(x=df.index, y=df["Fast_MA"], name=f"{fast_type} {fast_period}", line=dict(color="orange", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Slow_MA"], name=f"{slow_type} {slow_period}", line=dict(color="blue", width=1.5)), row=1, col=1)

    if setup == "opening_bar":
        for trace in opening_bar_signal_traces(trades, s.opening_bar.bar_minutes):
            fig.add_trace(trace, row=1, col=1)
    else:
        buy_points = df.loc[df["Buy_Signal"]]
        sell_points = df.loc[df["Sell_Signal"]]
        fig.add_trace(
            go.Scatter(
                x=buy_points.index, y=buy_points["Close"], mode="markers", name="BUY signal",
                marker=dict(symbol="triangle-up", size=12, color="lime", line=dict(color="darkgreen", width=1)),
                hovertemplate="BUY signal<br>Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=sell_points.index, y=sell_points["Close"], mode="markers", name="SELL signal",
                marker=dict(symbol="triangle-down", size=12, color="red", line=dict(color="darkred", width=1)),
                hovertemplate="SELL signal<br>Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
            ), row=1, col=1
        )

    # Row 4: quarterly and TTM EPS
    eps_trace_added = False
    eps_trace_definitions = [
        ("Reported EPS", "Quarterly Reported EPS", "bar", "royalblue", False),
        ("EPS Estimate", "Quarterly EPS Estimate", "scatter", "orange", False),
        ("TTM_Reported_EPS", "TTM Reported EPS", "scatter", "limegreen", True),
        ("TTM_Estimated_EPS", "TTM EPS Estimate Proxy", "scatter", "cyan", True),
    ]

    for column, trace_name, trace_type, color, use_secondary in eps_trace_definitions:
        if not earnings_display_df.empty and column in earnings_display_df.columns:
            series = earnings_display_df[column].dropna()
            if series.empty:
                continue
            eps_trace_added = True
            if trace_type == "bar":
                trace = go.Bar(x=series.index, y=series, name=trace_name, marker_color=color, opacity=0.75)
            else:
                trace = go.Scatter(
                    x=series.index,
                    y=series,
                    mode="lines+markers",
                    name=trace_name,
                    line=dict(color=color, width=2, dash="dot" if "Estimate" in trace_name else "solid"),
                    marker=dict(size=7, color=color),
                )
            fig.add_trace(trace, row=4, col=1, secondary_y=use_secondary)

    if not eps_trace_added:
        fig.add_annotation(
            x=0.5, y=0.5, xref="x domain", yref="y domain", row=4, col=1,
            text="No quarterly EPS event falls within the selected display period",
            showarrow=False, font=dict(color="gray", size=12)
        )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray", row=4, col=1, secondary_y=False)

    # Row 5: quarterly fundamentals   
    fundamentals_trace_added = False

    scale_divisor, fundamentals_axis_title = (
        choose_fundamental_scale(
            fundamentals_plot_df
        )
    )

    fundamental_definitions = [
        (
            "Revenue",
            "Revenue",
            "royalblue",
        ),
        (
            "Total_Expense",
            "Total expenses",
            "crimson",
        ),
        (
            "Net_Income",
            "Net income",
            "limegreen",
        ),
        (
            "Operating_Expense",
            "Operating expenses",
            "orange",
        ),
        (
            "CapEx",
            "CapEx spending",
            "purple",
        ),
    ]


    for (
        column,
        trace_name,
        color,
    ) in fundamental_definitions:

        if (
            fundamentals_plot_df.empty
            or column not in fundamentals_plot_df.columns
        ):
            continue

        trace_data = fundamentals_plot_df[
            [
                column,
                "Fiscal_Quarter_End",
            ]
        ].dropna(
            subset=[column]
        )

        if trace_data.empty:
            continue

        fundamentals_trace_added = True

        scaled_values = (
            trace_data[column]
            / scale_divisor
        )

        fiscal_quarter_end_text = (
            trace_data["Fiscal_Quarter_End"]
            .dt.strftime("%Y-%m-%d")
        )

        fig.add_trace(
            go.Bar(
                x=trace_data.index,
                y=scaled_values,
                name=trace_name,
                marker_color=color,
                opacity=0.78,
                customdata=np.column_stack(
                    [
                        fiscal_quarter_end_text,
                    ]
                ),
                hovertemplate=(
                    f"{trace_name}<br>"
                    "Earnings date: "
                    "%{x|%Y-%m-%d}<br>"
                    "Fiscal quarter end: "
                    "%{customdata[0]}<br>"
                    "Value: "
                    f"%{{y:,.2f}} "
                    f"{fundamentals_axis_title}"
                    "<extra></extra>"
                ),
            ),
            row=5,
            col=1,
        )


    if not fundamentals_trace_added:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="x domain",
            yref="y domain",
            row=5,
            col=1,
            text=(
                "Quarterly revenue, total expenses, "
                "net income, operating expenses and "
                "CapEx data are unavailable"
            ),
            showarrow=False,
            font=dict(
                color="gray",
                size=12,
            ),
        )


    fig.add_hline(
        y=0,
        line_width=1,
        line_dash="dot",
        line_color="gray",
        row=5,
        col=1,
    )

    # Row 6: P/E
    trailing_pe = valuation_df["Trailing_PE"].dropna()
    estimate_based_pe = valuation_df["Estimate_Based_PE_Proxy"].dropna()

    if not trailing_pe.empty:
        fig.add_trace(
            go.Scatter(x=trailing_pe.index, y=trailing_pe, mode="lines", name="Trailing P/E", line=dict(color="gold", width=2)),
            row=6, col=1
        )
        median_value = trailing_pe.median()
        fig.add_hline(
            y=median_value, line_width=1, line_dash="dash", line_color="goldenrod",
            annotation_text=f"Median trailing P/E: {median_value:.1f}x",
            annotation_position="top left", row=6, col=1
        )

    if not estimate_based_pe.empty:
        fig.add_trace(
            go.Scatter(x=estimate_based_pe.index, y=estimate_based_pe, mode="lines", name="Estimate-Based P/E Proxy", line=dict(color="cyan", width=2, dash="dot")),
            row=6, col=1
        )
        median_value = estimate_based_pe.median()
        fig.add_hline(
            y=median_value, line_width=1, line_dash="dash", line_color="darkcyan",
            annotation_text=f"Median estimate-based P/E: {median_value:.1f}x",
            annotation_position="bottom left", row=6, col=1
        )

    if trailing_pe.empty and estimate_based_pe.empty:
        fig.add_annotation(
            x=0.5, y=0.5, xref="x domain", yref="y domain", row=6, col=1,
            text="Insufficient historical EPS data to calculate P/E",
            showarrow=False, font=dict(color="gray", size=12)
        )

    # Row 7: benchmarks
    benchmark_trace_added = False
    for column, color in [("S&P 500", "deepskyblue"), ("Nasdaq-100", "magenta")]:
        if not benchmark_df.empty and column in benchmark_df.columns:
            series = benchmark_df[column].dropna()
            if not series.empty:
                benchmark_trace_added = True
                fig.add_trace(
                    go.Scatter(x=series.index, y=series, name=column, line=dict(color=color, width=2)),
                    row=7, col=1
                )

    if not benchmark_trace_added:
        fig.add_annotation(
            x=0.5, y=0.5, xref="x domain", yref="y domain", row=7, col=1,
            text="Benchmark data is unavailable", showarrow=False,
            font=dict(color="gray", size=12)
        )
    fig.add_hline(y=100, line_width=1, line_dash="dot", line_color="gray", row=7, col=1)

    # Row 2: equity
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Equity"], name="Strategy Equity", line=dict(color="green", width=2),
            hovertemplate="Strategy Equity<br>Date: %{x|%Y-%m-%d}<br>Equity: $%{y:,.2f}<extra></extra>",
        ), row=2, col=1
    )

    # Row 3: drawdown
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Drawdown"] * 100, name="Drawdown %", fill="tozeroy",
            line=dict(color="red", width=1),
            hovertemplate="Drawdown<br>Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
        ), row=3, col=1
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Quarterly EPS", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="TTM EPS", row=4, col=1, secondary_y=True)
    fig.update_yaxes(title_text=fundamentals_axis_title, row=5, col=1)
    fig.update_yaxes(title_text="P/E (x)", row=6, col=1)
    fig.update_yaxes(title_text="Indexed", row=7, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
    fig.update_yaxes(title_text="DD (%)", row=3, col=1)

    # Apply the exact same time range to every subplot.
    common_x_start = pd.Timestamp(df.index.min())
    common_x_end = pd.Timestamp(df.index.max())
    for subplot_row in range(1, 8):
        fig.update_xaxes(
            range=[
                common_x_start,
                common_x_end,
            ],
            autorange=False,
            rangeslider_visible=False,
            row=subplot_row,
            col=1,
        )

    if setup == "opening_bar":
        # The price panel gets its own axis with one slot per bar, so nights and weekends take no room; it opens on
        # every bar of the period. Plotly's date rangebreaks did this before, but a pinch or drag on them snapped
        # back on phones.
        pad = (df["High"].max() - df["Low"].min()) * 0.05
        fig.update_xaxes(
            matches=None, type="category", showticklabels=True, autorange=False, nticks=4 if MOBILE else 8,
            tickangle=0, showgrid=False, range=[-0.5, len(df) - 0.5], row=1, col=1,
        )
        fig.update_yaxes(range=[df["Low"].min() - pad, df["High"].max() + pad], row=1, col=1)
        fig.update_layout(meta=dict(axis_hover=True))
        st.caption(f"The price panel shows every {s.opening_bar.bar_minutes}-minute bar. Pinch with two fingers "
                   "to zoom the time, drag with two fingers to move (double-tap resets); the price scale fits "
                   "the bars in view.")

    chart_state_text = "|".join(
        [
            ticker, start_date_str, end_date_str, timeframe, setup,
            fast_type, str(fast_period), slow_type, str(slow_period),
            buy_left, buy_operator, buy_right,
            sell_left, sell_operator, sell_right, str(account_size), str(s.opening_bar),
        ]
    )
    chart_revision = hashlib.sha256(chart_state_text.encode("utf-8")).hexdigest()[:16]

    fig.update_layout(
        height=BACKTEST_CHART_HEIGHT,
        margin=dict(l=20, r=70, t=50, b=20),
        showlegend=True,
        hovermode="x unified",
        barmode="group",
    )
    legends_under_panels(fig, 7, BACKTEST_CHART_HEIGHT, dated_rows=(1,) if setup == "opening_bar" else ())

    show_chart(
        fig,
        use_container_width=True,
        key=f"backtest_chart_{chart_revision}",
        config={"responsive": True, "displaylogo": False},
    )
    if setup == "opening_bar":
        render_opening_bar_details(trades, days, s.opening_bar)


# ============================================================
# Day Breakout Rule tab
# ============================================================

def show_rule_diagram(fig):
    """A rule's illustration under its settings, static so it reads like a picture on a phone."""

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "staticPlot": True})


def exit_pct_slider(key):
    """The day-breakout exit as the user reads it: sell when the close is more than X% of the range below the high.

    Returns the hold threshold the rule uses, (close - low) / (high - low) >= 1 - X/100.
    """

    exit_pct = slider_with_input("Exit when the close is more than this far below the day's high (% of the day's range)",
                                 0, 100, 33, 1, key=key, format="%d%%",
                                 help="33% is the top third: a close in the top third of the day's range is held overnight.")
    return 1 - exit_pct / 100


# ============================================================
# First-bar day trade (a Backtest signal rule)
# ============================================================

def opening_bar_settings(capital, bar_minutes=5):
    """Settings of the first-bar day trade, shown in the Signal panel; ``capital`` is the Account size."""

    st.caption(
        f"If the first {bar_minutes}-minute bar closes at least N points above its open, the day is a long day: buy "
        "at that bar's close with the stop at its low, sized so a stop-out loses at most the chosen % of the Account "
        "size (the capital allocated to the stock). Other days the rule stays out, or, with shorting on, shorts a "
        "first bar that closes at least N points below its open (the mirror image: stop at its high, exits on "
        "bounces and higher lows). After skipping the next M bars, sell everything on the chosen exit; anything still "
        "held is sold 5 minutes before the close."
    )
    risk_pct = float(st.number_input("Max loss per trade (% of capital)", min_value=0.1, max_value=100.0, value=10.0,
                                     step=0.5, key="opening_bar_risk"))
    cap_at_capital = st.checkbox("Never buy more than the capital", value=True, key="opening_bar_cap",
                                 help="Off lets the stop distance alone set the size, which can mean margin when the "
                                 "first bar is narrow.")
    min_move = float(st.number_input("Long day: first bar closes at least N points above its open", min_value=0.0,
                                     value=0.0, step=0.05, format="%.2f", key="opening_bar_min_move",
                                     help="In dollars. 0 means any first bar that closes above its open."))
    allow_short = st.checkbox("Short on short days", value=False, key="opening_bar_short",
                              help="Short when the first bar closes at least N points below its open: sell at its "
                              "close, stop at its high, and the exits turned upside down (a close above the last "
                              "bounce, or a bar whose low is above the previous bar's low, buys back). Borrow "
                              "costs are not included.")
    exit_label = st.selectbox("Exit", list(OPENING_BAR_EXITS), key="opening_bar_exit")
    exit_mode = OPENING_BAR_EXITS[exit_label]
    skip_bars = int(slider_with_input("Ignore the next M bars before the exit applies", 0, 30, 2,
                                      key="opening_bar_skip",
                                      help="The stop at the first bar's low still applies to these bars."))
    exit_fill = "close"
    custom_exit = CustomExit()
    if exit_mode == "custom":
        value_labels = list(OPENING_BAR_VALUES)
        left = st.selectbox("Sell when", value_labels, index=value_labels.index("Close"), key="opening_bar_custom_left",
                            help="Checked on every bar after the ignored ones; the sale is at that bar's close. "
                            "On a short day everything is turned upside down (below becomes above, a high becomes "
                            "a low).")
        operator = st.selectbox("is", OPENING_BAR_OPERATORS, key="opening_bar_custom_operator",
                                help="Crosses below: it was at or above on the previous bar and is below now.")
        right = st.selectbox("Compared with", value_labels, index=value_labels.index("Previous low"),
                             key="opening_bar_custom_right")
        offset = float(st.number_input("Plus ($, negative to subtract)", value=0.0, step=0.01, format="%.2f",
                                       key="opening_bar_custom_offset",
                                       help="Added to the right-hand value, e.g. -0.05 for 5 cents below the "
                                       "previous low."))
        ema_period = 9
        if "ema" in (OPENING_BAR_VALUES[left], OPENING_BAR_VALUES[right]):
            ema_period = int(st.number_input("EMA period (bars)", min_value=2, max_value=200, value=9, step=1,
                                             key="opening_bar_custom_ema"))
        custom_exit = CustomExit(OPENING_BAR_VALUES[left], operator, OPENING_BAR_VALUES[right], offset, ema_period)
        st.caption(f"Exit: **{custom_exit.describe()}**")
    if exit_mode == "pullback":
        fill_label = st.selectbox(
            "Pullback exit fill", list(OPENING_BAR_FILLS), key="opening_bar_fill",
            help="The close below the pullback is only known when its bar ends, so a market order fills near "
            "that close. Filling at the pullback price itself is the rule as written but is optimistic.")
        exit_fill = OPENING_BAR_FILLS[fill_label]
    # Commissions and slippage come from the Trading costs section (see trading_cost_settings).
    return OpeningBarSettings(min_move=min_move, skip_bars=skip_bars, exit_mode=exit_mode, risk_pct=risk_pct,
                              capital=float(capital), cap_at_capital=cap_at_capital, exit_fill=exit_fill,
                              allow_short=allow_short, bar_minutes=int(bar_minutes), custom_exit=custom_exit)


def load_opening_bar_bars(ticker, bar_minutes=5):
    """Intraday bars for the day trade: ~60 days of 5- or 15-minute bars, or ~30 days of 1-minute bars.

    Yahoo serves 1-minute bars 7 days per request, so they are fetched in weekly pieces. Today's session is left out
    until the market closes: its last bar so far is not the end-of-day exit.
    """

    if bar_minutes == 1:
        end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
        pieces = []
        for _ in range(4):
            start = end - pd.Timedelta(days=7)
            pieces.append(load_intraday_data(ticker, "1m", start=f"{start:%Y-%m-%d}", end=f"{end:%Y-%m-%d}"))
            end = start
        pieces = [piece for piece in pieces if not piece.empty]
        bars = pd.concat(pieces).sort_index() if pieces else pd.DataFrame()
        bars = bars.loc[~bars.index.duplicated(keep="last")] if len(bars) else bars
    else:
        bars = load_intraday_data(ticker, f"{bar_minutes}m", "60d")
    if bars.empty:
        return bars
    now = pd.Timestamp.now(tz="America/New_York")
    if now.hour < 16:
        bars = bars[bars.index.normalize() < now.tz_localize(None).normalize()]
    return bars


def opening_bar_equity(bars, trades, settings, account_size):
    """The intraday bars with the account's equity and drawdown, booking each trade's P&L on its exit bar."""

    frame = bars[["Open", "High", "Low", "Close"]].copy()
    equity = pd.Series(0.0, index=frame.index)
    if len(trades):
        bar_length = pd.Timedelta(minutes=settings.bar_minutes)
        exit_bar = trades["Exit time"].where(trades["Exit reason"] == "End of day", trades["Exit time"] - bar_length)
        booked = trades.groupby(exit_bar.to_numpy())["P&L"].sum()
        equity = booked.reindex(frame.index, fill_value=0.0).cumsum()
    frame["Equity"] = float(account_size) + equity
    frame["Peak"] = frame["Equity"].cummax()
    frame["Drawdown"] = frame["Equity"] / frame["Peak"] - 1.0
    frame["Strat_Returns"] = frame["Equity"].pct_change().fillna(0.0)
    return frame


def opening_bar_rule_text(settings):
    exit_text = {"pullback": "the next close below the last pullback close",
                 "lower_high": "the first bar whose high is below the previous bar's high",
                 "custom": f"the close of the first bar where {settings.custom_exit.describe()}"}[settings.exit_mode]
    side_text = " Short days are shorted, the mirror image." if settings.allow_short else ""
    return (f"**BUY / SELL:** on a long day (first {settings.bar_minutes}-minute bar closes at least "
            f"{settings.min_move:g} points above its open) buy at that bar's close, stop at its low, then after "
            f"{settings.skip_bars} bars sell on {exit_text}, or 5 minutes before the close.{side_text} Each trade risks "
            f"at most {settings.risk_pct:g}% of the Account size, after IBKR commissions and "
            f"${settings.slippage:.2f} a share of slippage.")


def bar_labels(times):
    """The price panel's bar names (one category per bar): "Oct 28 09:35"."""

    return pd.DatetimeIndex(times).strftime("%b %d %H:%M")


def opening_bar_signal_traces(trades, bar_minutes=5):
    """Entry and exit markers of the day trades for the price panel, each on the bar whose price it is."""

    marks = [
        ("BUY", trades[trades["Side"] == "Long"], "Entry", "triangle-up", "lime", "darkgreen"),
        ("SELL", trades[trades["Side"] == "Long"], "Exit", "triangle-down", "red", "darkred"),
        ("SHORT", trades[trades["Side"] == "Short"], "Entry", "triangle-down", "orange", "darkorange"),
        ("COVER", trades[trades["Side"] == "Short"], "Exit", "triangle-up", "deepskyblue", "steelblue"),
    ]
    traces = []
    for name, rows, column, symbol, color, edge in marks:
        if rows.empty:
            continue
        reason = rows["Exit reason"] if column == "Exit" else pd.Series("", index=rows.index)
        # Entries and exits are timed at the end of their bar; an end-of-day sale at the open of its bar.
        bar_start = rows[f"{column} time"].where(reason == "End of day",
                                                 rows[f"{column} time"] - pd.Timedelta(minutes=bar_minutes))
        traces.append(go.Scatter(
            x=bar_labels(bar_start), y=rows[column], mode="markers", name=name, customdata=reason,
            marker=dict(symbol=symbol, size=10, color=color, line=dict(color=edge, width=1)),
            hovertemplate=f"{name} %{{customdata}}<br>%{{x}}<br>Price: %{{y:.2f}}<extra></extra>",
        ))
    return traces


def render_opening_bar_details(trades, days, settings):
    """Trade statistics and the trade list under the Backtest chart for the first-bar day trade."""

    allow_short, exit_mode = settings.allow_short, settings.exit_mode
    if days.empty:
        st.info("No complete sessions to test.")
        return
    summary = summarize_opening_bar(trades, days)
    first_day, last_day = days["Date"].min(), days["Date"].max()
    st.markdown(f"**{first_day:%b %d} to {last_day:%b %d, %Y}**: {summary['Days']} days of "
                f"{settings.bar_minutes}-minute bars, {summary['Long days %']:.0f}% long days"
                + (f", {days['Short day'].mean() * 100:.0f}% short days." if allow_short else "."))
    if trades.empty:
        st.info("No long or short days in this period." if allow_short else "No long days in this period.")
        return

    tiles = st.columns(2 if MOBILE else 4)
    tiles[0].metric("Net P&L", f"${summary['Net P&L']:+,.0f}", f"{summary['Trades']} trades", delta_color="off")
    tiles[1].metric("Avg win / loss", f"${summary['Avg win']:,.0f} / ${-summary['Avg loss']:,.0f}")
    tiles[2 % len(tiles)].metric("Profit factor", f"{summary['Profit factor']:.2f}",
                                 f"avg {summary['Avg R']:+.2f} R", delta_color="off")
    tiles[3 % len(tiles)].metric("Costs", f"${summary['Costs']:,.0f}",
                                 f"${summary['Costs'] / summary['Trades']:.2f} a trade", delta_color="off")
    st.caption(f"Exits: {summary['Stopped out %']:.0f}% stopped out, "
               f"{summary['Pullback exit %'] + summary['Lower high exit %'] + summary['Custom exit %']:.0f}% on the "
               f"{exit_mode.replace('_', '-')} exit, "
               f"{summary['End of day %']:.0f}% sold before the close. Costs are commissions and fees; slippage is "
               "already in the fill prices.")
    if allow_short:
        sides = pd.DataFrame({side: summarize_opening_bar(trades[trades["Side"] == side])
                              for side in ("Long", "Short")}).T[["Trades", "Win rate %", "Net P&L", "Avg R",
                                                                 "Profit factor"]]
        st.dataframe(sides.style.format({"Trades": "{:.0f}", "Win rate %": "{:.0f}", "Net P&L": "${:+,.0f}",
                                         "Avg R": "{:+.2f}", "Profit factor": "{:.2f}"}, na_rep="-"),
                     width="stretch")
    with st.expander(f"All {len(trades):,} trades"):
        st.dataframe(
            trades.drop(columns=["Ticker"]).sort_values("Entry time", ascending=False).style.format(
                {"Entry": "{:.2f}", "Stop": "{:.2f}", "Exit": "{:.2f}", "Risk $": "${:,.0f}", "Costs $": "${:,.2f}",
                 "P&L": "${:+,.0f}", "Return %": "{:+.2f}", "R": "{:+.2f}", "Date": "{:%Y-%m-%d}",
                 "Entry time": "{:%H:%M}", "Exit time": "{:%H:%M}"}, na_rep="-"),
            hide_index=True, width="stretch",
        )
    st.download_button("Download trades (CSV)", trades.to_csv(index=False), file_name="first_bar_day_trades.csv",
                       mime="text/csv")
    st.caption(
        "Yahoo keeps about 60 days of 5- and 15-minute bars and 30 days of 1-minute bars, so this is a short sample: "
        "treat it as a first look, not proof. The stop is checked before the exit on every bar and fills at the stop "
        "(or the open on a gap past it). Return % is P&L over the capital."
    )


# ============================================================
# Trade Journal: import IBKR trades, see where the P&L comes from, replay them with other rules
# ============================================================

JOURNAL_BAR_CHOICES = {
    "Hourly bars (cover about two years)": "hourly",
    "5-minute bars where Yahoo has them (last ~60 days), hourly before": "fine",
}
JOURNAL_TABLE_COLUMNS = ["Trip", "Side", "Entry date", "Exit date", "Days held", "Shares", "Avg entry", "Avg exit",
                         "Cost $", "Net P&L", "Return %", "Last exit type", "Entry orders", "Added to loser"]
JOURNAL_WHATIF_COLUMNS = ["What-if P&L", "What-if exit", "What-if exit date", "Change", "Skipped buys"]
JOURNAL_NOTE_COLUMNS = ["Action", "Comment", "Estimate $"]
JOURNAL_FORMATS = {"Entry date": "{:%Y-%m-%d}", "Exit date": "{:%Y-%m-%d}", "What-if exit date": "{:%Y-%m-%d}",
                   "Days held": "{:.0f}", "Shares": "{:,.0f}", "Avg entry": "{:,.2f}", "Avg exit": "{:,.2f}",
                   "Cost $": "${:,.0f}", "Net P&L": "${:+,.0f}", "Return %": "{:+.2f}", "What-if P&L": "${:+,.0f}",
                   "Change": "${:+,.0f}", "Skipped buys": "{:.0f}", "Trades": "{:,.0f}", "Win rate %": "{:.0f}",
                   "Avg win": "${:,.0f}", "Avg loss": "${:,.0f}", "Profit factor": "{:.2f}",
                   "What-if net P&L": "${:+,.0f}", "Estimate $": "${:+,.0f}", "Your P&L": "${:+,.0f}",
                   "P&L with the action": "${:+,.0f}", "Improvement": "${:+,.0f}"}
JOURNAL_ACTUAL_COLOR, JOURNAL_WHATIF_COLOR = "gray", "royalblue"


def journal_symbol(symbol):
    """Yahoo symbol for an IBKR one: ``BRK B`` -> ``BRK-B``."""

    return normalize_symbol(str(symbol).strip().replace(" ", "."))


@st.cache_data(ttl=21600, show_spinner=False)
def load_journal_bars(symbols, start_str, end_str, interval):
    """Regular-session bars for the traded stocks as {IBKR symbol: frame}, in New York time. Cached for 6 hours.

    Yahoo keeps about two years of hourly ("60m") bars and about 60 days of 5-minute bars.
    """

    yahoo_to_ibkr = {journal_symbol(symbol): symbol for symbol in symbols}
    tickers = list(yahoo_to_ibkr)
    bars = {}
    for first in range(0, len(tickers), 40):
        chunk = tickers[first:first + 40]
        try:
            data = yf.download(chunk, start=start_str, end=end_str, interval=interval, auto_adjust=False,
                               prepost=False, progress=False, threads=True)
        except Exception:
            continue
        if data is None or data.empty:
            continue
        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_convert("America/New_York").tz_localize(None)
        for ticker, frame in split_download(data, chunk).items():
            bars[yahoo_to_ibkr[ticker]] = frame
    return bars


def journal_bars(trips, choice):
    """Price bars for every traded stock: hourly, or [5-minute, hourly] per stock for the "fine" choice."""

    symbols = tuple(sorted(trips["Symbol"].unique()))
    start = trips["Entry date"].min() - pd.Timedelta(days=3)
    end = trips["Exit date"].max() + pd.Timedelta(days=3)
    hourly = load_journal_bars(symbols, f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", "60m")
    if choice != "fine":
        return hourly
    fine_start = max(start, pd.Timestamp(date.today()) - pd.Timedelta(days=58))
    recent = tuple(sorted(trips.loc[trips["Exit date"] >= fine_start, "Symbol"].unique()))
    fine = load_journal_bars(recent, f"{fine_start:%Y-%m-%d}", f"{end:%Y-%m-%d}", "5m") if recent else {}
    return {symbol: [fine.get(symbol), hourly.get(symbol)] for symbol in symbols}


def journal_prepared(data, bar_choice):
    """Fills placed on price bars, once per file and bar choice (kept in Session State). Returns (prepared, bars)."""

    key = [data["id"], bar_choice]
    cached = st.session_state.get("journal_prepared")
    if cached and cached["key"] == key:
        return cached["value"], cached["bars"]
    closed = data["trips"][data["trips"]["Status"] == "Closed"]
    with st.spinner(f"Loading price bars for {closed['Symbol'].nunique()} stocks..."):
        bars = journal_bars(closed, JOURNAL_BAR_CHOICES[bar_choice])
        prepared = journal_prepare(data["fills"], closed, bars)
    if any(value is not None for value in prepared.values()):  # do not keep an empty result from a failed download
        st.session_state["journal_prepared"] = {"key": key, "value": prepared, "bars": bars}
    return prepared, bars


def journal_load_upload(upload, data):
    """Read a newly uploaded file into Session State and reset the filters for it. Returns the journal data."""

    file_id = f"{upload.name}:{upload.size}"
    if data is not None and data["id"] == file_id:
        return data
    try:
        with st.spinner(f"Reading {upload.name}..."):
            trades, notes = load_ibkr_trades(upload, name=upload.name)
            fills, trips = round_trips(trades)
    except Exception as error:
        st.error(f"Could not read {upload.name}: {error}")
        return data
    if trips.empty:
        st.error(f"{upload.name} has no US-dollar stock trades.")
        return data
    data = {"id": file_id, "name": upload.name, "notes": notes, "fills": fills, "trips": trips}
    st.session_state["journal_data"] = data
    st.session_state.pop("journal_prepared", None)
    st.session_state["journal_symbols"] = []
    st.session_state["journal_exclude"] = sorted(set(JOURNAL_CASH_LIKE) & set(trips["Symbol"]))
    st.session_state["journal_dates"] = (trips["Entry date"].min().date(), trips["Entry date"].max().date())
    return data


def journal_rules():
    """What-if rule settings. Returns a WhatIfRules and the price-bar choice."""

    with settings_panel("What-if rules"):
        st.caption("Your own buys and sells, replayed on Yahoo price bars with these changes.")
        use_stop = st.checkbox("Stop loss, placed with the first buy", key="journal_use_stop")
        stop_pct = st.number_input("Stop distance from the first buy (%)", 0.25, 50.0, 3.0, 0.25,
                                   key="journal_stop_pct", disabled=not use_stop)
        use_target = st.checkbox("Profit target", key="journal_use_target")
        target_pct = st.number_input("Target above the first buy (%)", 0.25, 100.0, 6.0, 0.25,
                                     key="journal_target_pct", disabled=not use_target)
        no_adds = st.checkbox("Don't add to a losing position", key="journal_no_adds",
                              help="Skips any buy more than 0.2% below the first buy (for a short, a sale above it). "
                              "Your later sells then close only the shares still held.")
        use_size = st.checkbox("Same dollar amount on every trade", key="journal_use_size",
                               help="Scales every fill of a trade so the first order is worth this amount.")
        size = st.number_input("Dollars per trade", 500.0, 1_000_000.0, 10_000.0, 500.0, key="journal_size",
                               disabled=not use_size)
        use_days = st.checkbox("Time limit", key="journal_use_days")
        max_days = st.number_input("Sell at the close after this many trading days (0 = the entry day)", 0, 250, 1, 1,
                                   key="journal_max_days", disabled=not use_days)
        slippage = st.number_input("Slippage on stop fills (basis points)", 0.0, 200.0, 5.0, 1.0,
                                   key="journal_slippage",
                                   help="1 bp = 0.01%. A triggered stop is a market order and usually fills a little "
                                   "worse than the stop price.")
        bar_choice = st.radio("Price bars", list(JOURNAL_BAR_CHOICES), key="journal_bars",
                              help="Finer bars place stops more precisely. Yahoo keeps about 60 days of 5-minute bars, "
                              "so older trades use hourly bars either way.")
    rules = WhatIfRules(
        stop_pct=float(stop_pct) if use_stop else None,
        target_pct=float(target_pct) if use_target else None,
        no_averaging_down=bool(no_adds),
        fixed_size=float(size) if use_size else None,
        max_days=int(max_days) if use_days else None,
        slippage_bps=float(slippage),
    )
    return rules, bar_choice


def journal_money(value, signed=True):
    """Dollars with the sign in front: -$1,234, +$56 (``signed=False``: $1,234)."""

    if value is None or pd.isna(value):
        return "-"
    if not signed:
        return f"${value:,.0f}"
    return f"{'-' if value < 0 else '+'}${abs(value):,.0f}"


def journal_tiles(stats, compare=None):
    """Four headline tiles; with ``compare`` (the actual stats), the what-if tiles show the change."""

    tiles = st.columns(2 if MOBILE else 4)

    def change(name, money=True):
        if compare is None or pd.isna(stats[name]) or pd.isna(compare[name]):
            return None
        delta = stats[name] - compare[name]
        return f"{journal_money(delta)} vs yours" if money else f"{delta:+.1f}"

    win = stats["Win rate %"]
    tiles[0].metric("Net P&L", journal_money(stats["Net P&L"]),
                    change("Net P&L") if compare else f"{stats['Trades']:,} trades",
                    delta_color="normal" if compare else "off")
    tiles[1].metric("Win rate", "-" if pd.isna(win) else f"{win:.0f}%",
                    f"break-even needs {stats['Breakeven win rate %']:.0f}%"
                    if pd.notna(stats["Breakeven win rate %"]) else None, delta_color="off")
    tiles[2 % len(tiles)].metric("Avg win / loss", f"{journal_money(stats['Avg win'], False)} / "
                                 f"{journal_money(-stats['Avg loss'] if pd.notna(stats['Avg loss']) else None, False)}",
                                 f"win is {stats['Payoff']:.2f}x the loss" if pd.notna(stats["Payoff"]) else None,
                                 delta_color="off")
    pf = stats["Profit factor"]
    tiles[3 % len(tiles)].metric("Profit factor", "-" if pd.isna(pf) else f"{pf:.2f}",
                                 f"fees {journal_money(stats['Fees'], False)}", delta_color="off")


def journal_equity_chart(curves):
    """Cumulative P&L lines: {name: (series, color)}."""

    fig = go.Figure()
    for name, (series, color) in curves.items():
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=name, mode="lines",
                                 line=dict(color=color, width=2 if name != "Your trades" else 1.5),
                                 hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}<br>$%{{y:+,.0f}}<extra></extra>"))
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      title="Cumulative P&L after fees", yaxis_title="$", legend=dict(orientation="h", y=1.12))
    show_chart(fig, width="stretch", key="journal_equity_chart")


def journal_breakdown_chart(table, has_whatif, title):
    fig = go.Figure()
    labels = [str(label) for label in table.index]
    fig.add_trace(go.Bar(x=labels, y=table["Net P&L"], name="Your trades", marker_color=JOURNAL_ACTUAL_COLOR,
                         hovertemplate="%{x}<br>$%{y:+,.0f}<extra>Your trades</extra>"))
    if has_whatif:
        fig.add_trace(go.Bar(x=labels, y=table["What-if net P&L"], name="What-if", marker_color=JOURNAL_WHATIF_COLOR,
                             hovertemplate="%{x}<br>$%{y:+,.0f}<extra>What-if</extra>"))
    fig.add_hline(y=0, line_width=1, line_color="gray")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), barmode="group", title=title,
                      yaxis_title="Net P&L ($)", legend=dict(orientation="h", y=1.12))
    show_chart(fig, width="stretch", key="journal_breakdown_chart")


def journal_chart_ticks(index, interval):
    """Short x-axis labels for a category axis: the date at the first bar of each day for intraday bars, about six
    dates for daily bars and six months for weekly bars."""

    index = pd.DatetimeIndex(index)
    if interval in INTRADAY_CHART_INTERVALS:
        labels = bar_labels(index)
        starts = np.flatnonzero(np.r_[True, index.normalize()[1:] != index.normalize()[:-1]])
        step = max(1, int(np.ceil(len(starts) / 8)))
        starts = starts[::step]
        return [labels[i] for i in starts], [f"{index[i]:%b %d}" for i in starts]
    labels = list(index.strftime("%Y-%m-%d"))
    picks = np.unique(np.linspace(0, len(index) - 1, min(6, len(index))).round().astype(int))
    text_format = "%b %d" if interval == "1d" else "%b %Y"
    return [labels[i] for i in picks], [f"{index[i]:{text_format}}" for i in picks]


def render_journal_trip(trip, fills, rules):
    """One round trip: its fills, and its price bars (a choice of bar lengths) with the buys, sells and what-if
    stop/target."""

    trip_fills = fills[fills["Trip"] == trip["Trip"]].reset_index(drop=True)
    line = (f"**{trip['Trip']}** - {trip['Side'].lower()}, {trip['Entry date']:%Y-%m-%d} to {trip['Exit date']:%Y-%m-%d}: "
            f"{journal_money(trip['Net P&L'])} ({trip['Return %']:+.2f}%)")
    if "What-if P&L" in trip and pd.notna(trip.get("What-if P&L")):
        line += f"; what-if {journal_money(trip['What-if P&L'])} ({trip['What-if exit']})"
    st.markdown(line)
    table = trip_fills[["Date", "Time", "Quantity", "Price", "OrderType", "Cost"]].rename(
        columns={"OrderType": "Order type", "Cost": "Fees"})
    table.insert(2, "Action", np.where(trip_fills["Quantity"] > 0, "BUY", "SELL"))
    if table["Time"].isna().all():
        table = table.drop(columns="Time")
    st.dataframe(table.style.format({"Date": "{:%Y-%m-%d}", "Quantity": "{:+,.0f}", "Price": "{:,.2f}",
                                     "Fees": "${:,.2f}"}, na_rep="-"), hide_index=True, width="stretch")

    # Only the bar lengths Yahoo still keeps for this trade's dates are offered; the last choice is remembered.
    options = chart_timeframes(trip["Entry date"], trip["Exit date"], date.today())
    preferred = st.session_state.get("journal_trip_timeframe_pref", "1 hour")
    if preferred not in options:
        preferred = "1 hour" if "1 hour" in options else "1 day"
    timeframe = st.radio("Bars", options, index=options.index(preferred), horizontal=True,
                         help="Yahoo keeps about 30 days of 1-minute bars, 60 days of 5- to 30-minute bars and two "
                         "years of hourly bars, so older trades offer fewer choices.")
    st.session_state["journal_trip_timeframe_pref"] = timeframe
    interval = CHART_TIMEFRAMES[timeframe][0]
    start, end = chart_request(trip["Entry date"], trip["Exit date"], interval)
    end = min(end, pd.Timestamp(date.today()) + pd.Timedelta(days=1))
    bars = load_journal_bars((trip["Symbol"],), f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", interval).get(trip["Symbol"])
    window = chart_window(bars, trip["Entry date"], trip["Exit date"], interval)
    if window is None or window.empty:
        st.caption(f"Yahoo has no {timeframe} bars for {trip['Symbol']} around this trade.")
        return

    intraday = interval in INTRADAY_CHART_INTERVALS
    labels = list(bar_labels(window.index)) if intraday else list(window.index.strftime("%Y-%m-%d"))
    positions = fill_bar_positions(trip_fills, window)
    fig = go.Figure(go.Candlestick(x=labels, open=window["Open"], high=window["High"], low=window["Low"],
                                   close=window["Close"], name=trip["Symbol"], showlegend=False))
    if positions is not None:
        buys = trip_fills["Quantity"] > 0
        for name, mask, symbol, color in [("BUY", buys, "triangle-up", "lime"), ("SELL", ~buys, "triangle-down", "red")]:
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=[labels[p] for p, keep in zip(positions, mask) if keep], y=trip_fills.loc[mask, "Price"],
                    mode="markers", name=name, marker=dict(symbol=symbol, size=11, color=color,
                                                           line=dict(color="black", width=1)),
                    hovertemplate=f"{name} %{{y:.2f}}<br>%{{x}}<extra></extra>"))
    first_price, sign = trip_fills["Price"].iloc[0], 1 if trip["Side"] == "Long" else -1
    if rules.stop_pct:
        fig.add_hline(y=first_price * (1 - sign * rules.stop_pct / 100), line_dash="dash", line_color="red",
                      annotation_text=f"stop {rules.stop_pct:g}%", annotation_position="bottom left")
    if rules.target_pct:
        fig.add_hline(y=first_price * (1 + sign * rules.target_pct / 100), line_dash="dash", line_color="green",
                      annotation_text=f"target {rules.target_pct:g}%", annotation_position="top left")
    tick_values, tick_text = journal_chart_ticks(window.index, interval)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False,
                      title=f"{trip['Symbol']}, {timeframe} bars", legend=dict(orientation="h", y=1.1),
                      xaxis=dict(type="category", tickmode="array", tickvals=tick_values, ticktext=tick_text,
                                 tickangle=0))
    show_chart(fig, width="stretch", key="journal_trip_chart")
    if trip_fills["Time"].isna().all() and intraday:
        st.caption("The export has no trade times, so each marker sits on the first bar of its day that traded at "
                   "its price.")
    elif not intraday:
        st.caption(f"Each marker sits on the {'day' if interval == '1d' else 'week'} of its fill.")


def journal_load_notes(upload):
    """Merge notes from an uploaded CSV into Session State (once per file)."""

    if upload is None:
        return
    file_id = f"{upload.name}:{upload.size}"
    if st.session_state.get("journal_notes_file") == file_id:
        return
    try:
        loaded = notes_from_csv(upload)
    except Exception as error:
        st.error(f"Could not read notes from {upload.name}: {error}")
        return
    notes = dict(st.session_state.get("journal_notes", {}))
    notes.update(loaded)
    st.session_state["journal_notes"] = notes
    st.session_state["journal_notes_file"] = file_id
    st.session_state["journal_notes_version"] = st.session_state.get("journal_notes_version", 0) + 1
    st.caption(f"Loaded {len(loaded)} notes from {upload.name}.")


def with_notes(trips, notes):
    """``trips`` with the Action, Comment and Estimate $ of each trade's note as its last columns."""

    out = trips.copy()
    out["Action"] = out["Key"].map(lambda key: notes.get(key, {}).get("Action", "")).replace(NO_ACTION, "")
    out["Comment"] = out["Key"].map(lambda key: notes.get(key, {}).get("Comment", ""))
    out["Estimate $"] = pd.to_numeric(out["Key"].map(lambda key: notes.get(key, {}).get("Estimate $")), errors="coerce")
    return out


def render_journal_note(trip):
    """The note form under a trade's chart: what would have improved it, a comment and, for actions the replay
    cannot test, an estimate of the gain."""

    notes = st.session_state.get("journal_notes", {})
    note = notes.get(trip["Key"], {})
    # Keys carry the trade and the notes version, so each trade (and freshly loaded notes) starts from its saved note.
    suffix = f"{trip['Key']}_{st.session_state.get('journal_notes_version', 0)}_draft"
    with st.form(f"journal_note_form_{suffix}", border=True):
        st.markdown(f"**Your note on {trip['Trip']}**")
        action = st.selectbox("What would have improved it?", IMPROVEMENT_ACTIONS,
                              index=IMPROVEMENT_ACTIONS.index(note.get("Action", NO_ACTION)),
                              key=f"journal_note_action_{suffix}",
                              help="Stops, targets, sizes and time limits are tested with the values under What-if "
                              "rules when the improvement summary is worked out.")
        comment = st.text_area("Comment", value=note.get("Comment", ""), key=f"journal_note_comment_{suffix}",
                               placeholder="What happened, and what you would do differently")
        saved_estimate = note.get("Estimate $")
        estimate = st.number_input(
            "Your estimate of the improvement ($), used for \"Hold the winner longer\" and \"Other\"",
            value=None if saved_estimate is None or pd.isna(saved_estimate) else float(saved_estimate),
            step=50.0, key=f"journal_note_estimate_{suffix}", placeholder="optional")
        saved = st.form_submit_button("Save note", type="primary")
    if saved:
        notes = dict(notes)
        if action == NO_ACTION and not comment.strip() and estimate is None:
            notes.pop(trip["Key"], None)
        else:
            notes[trip["Key"]] = {"Action": action, "Comment": comment.strip(),
                                  "Estimate $": np.nan if estimate is None else float(estimate)}
        st.session_state["journal_notes"] = notes
        st.rerun()
    if note:
        st.caption("Saved. Notes are kept while the app is open; use **Download notes** below to keep them.")


def render_journal_improvements(data, closed, fills, rules, bar_choice):
    """Summary of the notes: how much each kind of change would have improved the trades noted with it."""

    notes = st.session_state.get("journal_notes", {})
    if not notes:
        return
    st.subheader("Improvement notes")
    noted = closed[closed["Key"].isin(notes)]
    actions = {notes[key].get("Action") for key in noted["Key"]}
    prepared = None
    if any(action_rules(action, rules) is not None for action in actions):
        prepared, _ = journal_prepared(data, bar_choice)
    summary = improvement_summary(closed, notes, fills, prepared, rules)
    if summary.empty:
        st.caption("None of the trades shown has a note with an action yet.")
    else:
        st.dataframe(summary.style.format(JOURNAL_FORMATS, na_rep="-"), hide_index=True, width="stretch")
        st.caption(
            "Each action is tested only on the trades you noted with it: stops, targets, sizes and time limits are "
            "replayed with the values under What-if rules (even when the rule is not ticked there); \"Skip the trade\" "
            "removes the trade's P&L; \"Hold the winner longer\" and \"Other\" add up your own estimates.")
    table = notes_frame(notes, data["trips"])
    with st.expander(f"All {len(table)} notes"):
        st.dataframe(table.drop(columns="Key").style.format(JOURNAL_FORMATS, na_rep="-"), hide_index=True,
                     width="stretch")
    st.download_button("Download notes (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"trade_notes_{pd.Timestamp(date.today()):%Y%m%d}.csv", mime="text/csv")


def render_trade_journal():
    """Trade Journal tab: import IBKR trades (US dollars), see where the P&L comes from, replay with other rules."""

    data = st.session_state.get("journal_data")
    with settings_panel("Trade history"):
        upload = st.file_uploader(
            "IBKR trade export (.xlsx or .csv)", type=["xlsx", "xls", "csv"], key="journal_file_upload",
            help="An IBKR Flex Query trade export with Symbol, TradeDate, Buy/Sell, Quantity, Price, Commission, "
            "OrderType and CurrencyPrimary. Add Date/Time for exact fill times.")
        if upload is not None:
            data = journal_load_upload(upload, data)
        elif data is not None:
            st.caption(f"Using **{data['name']}**. Upload another file to replace it.")
        if data is not None:
            journal_load_notes(st.file_uploader(
                "Saved trade notes (.csv), optional", type=["csv"], key="journal_notes_upload",
                help="The trades or notes CSV downloaded from this tab. Notes are kept while the app is open; "
                "download them to keep them and load them here next time."))

    st.caption("Import your IBKR trades, see where the profit and loss comes from, then replay the same trades with "
               "different rules (a stop, a target, no adding to losers, a fixed size or a time limit). US-dollar trades "
               "only for now.")
    if data is None:
        st.info("Upload an IBKR trade export in the settings to start.")
        with st.expander("Which file to export from IBKR"):
            st.markdown(
                "In the IBKR Portal, **Performance & Reports > Flex Queries**: create an Activity Flex Query with the "
                "**Trades** section (executions), these fields: *CurrencyPrimary, Symbol, TradeID, OrderID, TradeDate, "
                "Date/Time, Buy/Sell, Quantity, TradePrice, IBCommission, OrderType*, and save it as CSV (an Excel "
                "copy works too). *Date/Time* is optional but lets the what-if replay place each fill in the right bar. "
                "Trades in other currencies and currency conversions are left out.")
        return

    trips_all, fills, notes = data["trips"], data["fills"], data["notes"]
    symbols_all = sorted(trips_all["Symbol"].unique())
    first_day, last_day = trips_all["Entry date"].min().date(), trips_all["Entry date"].max().date()
    with settings_panel("Filter trades"):
        dates = st.date_input("Entry dates", value=(first_day, last_day), key="journal_dates", format="YYYY-MM-DD")
        only = st.multiselect("Only these stocks (empty = all)", symbols_all, key="journal_symbols")
        exclude = st.multiselect("Leave out", symbols_all, key="journal_exclude",
                                 help="Cash-like funds such as SGOV are left out when a file is loaded.")
        side = st.radio("Side", ["All", "Long", "Short"], horizontal=True, key="journal_side")
    rules, bar_choice = journal_rules()

    start, end = (dates if isinstance(dates, (tuple, list)) and len(dates) == 2 else (first_day, last_day))
    trips = journal_filter(trips_all, start, end, only, exclude, side)
    closed = trips[trips["Status"] == "Closed"]
    still_open = trips_all[trips_all["Status"] == "Open"]
    dropped = []
    if notes.get("other_currency"):
        dropped.append(f"{notes['other_currency']:,} rows in {', '.join(notes['other_currencies'])}")
    if notes.get("forex"):
        dropped.append(f"{notes['forex']:,} currency conversions")
    st.caption(
        f"**{data['name']}**: {notes['kept']:,} US-dollar executions, {len(trips_all[trips_all['Status'] == 'Closed']):,} "
        f"round trips from {first_day:%d %b} to {trips_all['Exit date'].max():%d %b %Y}"
        + (f"; {len(still_open)} still open and not counted ({', '.join(still_open['Symbol'])})" if len(still_open) else "")
        + (f"; left out: {'; '.join(dropped)}" if dropped else "") + ".")
    if closed.empty:
        st.info("No closed trades match the filters.")
        return

    whatif, prepared = None, None
    if rules.active():
        prepared, _ = journal_prepared(data, bar_choice)
        whatif = journal_simulate(fills, closed, None, rules, prepared=prepared)

    st.subheader("Your results")
    actual = journal_summary(closed)
    journal_tiles(actual)
    st.caption(f"A round trip runs from the first buy (or short sale) to the sale that leaves no shares. Average "
               f"{journal_money(actual['Per trade'])} a trade; largest loss {journal_money(actual['Largest loss'])}. "
               "Fees are IBKR's commissions and fees.")

    if whatif is not None:
        valid = whatif.dropna(subset=["What-if P&L"])
        missing = len(whatif) - len(valid)
        st.subheader("What-if: " + rules.describe())
        if valid.empty:
            st.warning("No price bars could be loaded for these stocks, so the trades cannot be replayed. Try again later.")
        else:
            base = journal_summary(valid)
            tested = journal_summary(valid, "What-if P&L")
            journal_tiles(tested, compare=base)
            reasons = valid["What-if exit"].value_counts()
            st.caption(
                f"Compared on the same {len(valid):,} trades"
                + (f" ({missing} without Yahoo price bars are left out of both)" if missing else "")
                + f": your result {journal_money(base['Net P&L'])}, what-if {journal_money(tested['Net P&L'])}. Exits: "
                + ", ".join(f"{count:,} {reason.lower()}" for reason, count in reasons.items())
                + (f"; {int(valid['Skipped buys'].sum()):,} buys skipped" if valid["Skipped buys"].sum() else "")
                + ". Stops are checked before your own fills in the same bar; on the buy's own bar a stop counts only "
                "if the bar closes beyond it.")
            journal_equity_chart({"Your trades": (journal_equity(valid), JOURNAL_ACTUAL_COLOR),
                                  "What-if": (journal_equity(valid, "What-if P&L", "What-if exit date"),
                                              JOURNAL_WHATIF_COLOR)})
            sweep = stop_sweep(fills, closed.loc[valid.index], None, rules, prepared=prepared)
            fig = go.Figure(go.Bar(
                x=sweep["Stop %"], y=sweep["Net P&L"],
                marker_color=[JOURNAL_ACTUAL_COLOR if label == "None" else JOURNAL_WHATIF_COLOR for label in sweep["Stop %"]],
                customdata=np.column_stack([sweep["Win rate %"], sweep["Stopped %"]]),
                hovertemplate="Stop %{x}<br>Net $%{y:+,.0f}<br>Win rate %{customdata[0]:.0f}%"
                              "<br>Stopped out %{customdata[1]:.0f}%<extra></extra>"))
            fig.add_hline(y=0, line_width=1, line_color="gray")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Net P&L ($)",
                              xaxis_title="Stop distance from the first buy",
                              title="Net P&L for each stop distance (the other rules as set)")
            show_chart(fig, width="stretch", key="journal_sweep_chart")
            st.caption("A tight stop cuts the big losses but also sells trades that dip and recover. The best stop on "
                       "past trades is a guide, not a promise: check it again on trades made after you chose it.")
    else:
        journal_equity_chart({"Your trades": (journal_equity(closed), JOURNAL_ACTUAL_COLOR)})
        st.info("Turn on a rule under **What-if rules** to replay these trades with it.")

    st.subheader("Where the P&L comes from")
    by_label = st.selectbox("Break down by", list(JOURNAL_BREAKDOWNS), key="journal_breakdown")
    column = JOURNAL_BREAKDOWNS[by_label]
    source = valid if whatif is not None and not valid.empty else closed
    table = journal_breakdown(source, column)
    has_whatif = source is not closed
    if has_whatif:
        table["What-if net P&L"] = journal_breakdown(source, column, "What-if P&L")["Net P&L"]
    chart_table = table
    if column == "Symbol" and len(table) > 24:
        chart_table = pd.concat([table.head(12), table.tail(12)])
        title = "12 worst and 12 best stocks"
    else:
        title = by_label
    journal_breakdown_chart(chart_table, has_whatif, title)
    st.dataframe(table.style.format(JOURNAL_FORMATS, na_rep="-"), width="stretch",
                 height=min(420, 38 + 35 * len(table)))

    st.subheader("Trades")
    notes = st.session_state.get("journal_notes", {})
    shown = with_notes((valid if has_whatif else closed).sort_values("Entry date", ascending=False), notes)
    columns = JOURNAL_TABLE_COLUMNS + (JOURNAL_WHATIF_COLUMNS if has_whatif else []) + JOURNAL_NOTE_COLUMNS
    st.caption("Click a column header to sort; select a row to see that trade's fills and chart.")
    event = st.dataframe(shown[columns].style.format(JOURNAL_FORMATS, na_rep="-"), hide_index=True, width="stretch",
                         height=420, on_select="rerun", selection_mode="single-row", key="journal_trades_table")
    selected = event.selection.rows if event is not None and hasattr(event, "selection") else []
    st.download_button("Download trades (CSV)", shown[columns + ["Key"]].to_csv(index=False).encode("utf-8"),
                       file_name=f"round_trips_{pd.Timestamp(date.today()):%Y%m%d}.csv", mime="text/csv")
    if selected and selected[0] < len(shown):
        trip = shown.iloc[selected[0]]
        render_journal_trip(trip, fills, rules)
        render_journal_note(trip)

    render_journal_improvements(data, closed, fills, rules, bar_choice)


# ============================================================
# Main application
# ============================================================

if check_password():
    st.write("Welcome to the protected app!")
    keep_settings_across_tabs()
    MOBILE = detect_mobile()

    with st.spinner("Loading the main indexes..."):
        main_index_prices = load_main_indexes()

    # on_change="rerun" makes the tabs lazy: only the open tab runs, and its settings fill the sidebar.
    tabs = st.tabs(TAB_NAMES, key="main_tab", on_change="rerun")
    active_tab = st.session_state.get("main_tab") or TAB_NAMES[0]

    def tab_is_open(tab, label):
        return tab.open if tab.open is not None else label == active_tab

    if not MOBILE:
        st.sidebar.header(active_tab)
    for tab, label in zip(tabs, TAB_NAMES):
        with tab:
            if not tab_is_open(tab, label):
                continue
            if label == ECONOMY_TAB:
                render_economic_analysis()
            elif label == "Stock Screener":
                render_stock_screener()
            elif label == "Market Analysis":
                render_market_analysis(main_index_prices)
            elif label == "Industry Analysis":
                render_industry_analysis(main_index_prices)
            elif label == FUNDAMENTALS_TAB:
                render_stock_fundamentals()
            elif label == SUPPLY_CHAIN_TAB:
                render_supply_chain(main_index_prices)
            elif label == JOURNAL_TAB:
                render_trade_journal()
            else:
                strategy = strategy_settings(label)
                if strategy is None:
                    continue
                if label == "Backtest":
                    render_backtest(strategy)
                elif label == SIGNAL_TAB:
                    render_signal_analysis(strategy.ticker, strategy.end_date, strategy.signal_settings, strategy.risk_settings)
                else:
                    render_robustness(strategy)
                    render_validation(strategy.signal_settings, strategy.risk_settings, strategy.end_date)
