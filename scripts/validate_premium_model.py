"""Walk-forward check of the market premium model on many real stocks (run in CI, which has the network).

Loads each stock exactly as the app does (App.py's own loaders: Yahoo prices and EPS, SEC/Yahoo quarterly results),
runs the valuation model with its defaults (fundamental P/E, rules on, FRED policy rate, SPY for beta) and the
premium model's walk-forward test, and reports per stock and overall: the error of the next-quarter price forecast
against 'no change', the direction hit rate and how often the bear-to-bull band held.
"""

import ast
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

TICKERS = ["COST", "WMT", "KO", "PEP", "PG", "JNJ", "MCD", "HD", "NKE", "SBUX", "AAPL", "MSFT", "GOOGL", "AMZN",
           "META", "NVDA", "AVGO", "ADBE", "CRM", "ORCL", "INTC", "TXN", "AMAT", "KLAC", "LRCX", "TSLA", "UNH", "LLY",
           "MRK", "ABBV", "CAT", "DE", "HON", "NFLX", "ISRG", "TGT", "LOW", "QCOM", "CVX", "XOM"]


def app_functions():
    """App.py's imports, constants, functions and classes, without running its page."""

    source = open("App.py").read()
    tree = ast.parse(source)
    namespace = {"__name__": "app_functions", "__file__": "App.py"}
    kept = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
    for node in tree.body:
        if isinstance(node, kept):
            try:
                exec(compile(ast.Module([node], type_ignores=[]), "App.py", "exec"), namespace)
            except Exception as error:
                print(f"skipped a top-level statement at line {node.lineno}: {type(error).__name__}")
    return namespace


def main():
    app = app_functions()
    from analysis.premium_model import calibrate
    from analysis.valuation_model import model_params, run_model

    policy, _ = app["load_fred"]("DFF")
    market_yield, _ = app["load_fred"]("DGS10")
    rows, diagnostics = [], []
    for ticker in TICKERS:
        started = time.time()
        try:
            inputs = app["load_model_inputs"](ticker, 12, [], 5, "Trailing")
        except Exception as error:
            print(f"{ticker}: load failed {type(error).__name__}: {error}")
            continue
        if inputs is None or inputs.quarters.empty:
            print(f"{ticker}: no data")
            continue
        params = model_params({"anchor_source": "Fundamentals", "peer_weight": 1.0})
        result = run_model(inputs.quarters, inputs.close, None, params, None, start=inputs.start, policy_rate=policy,
                           market_yield=market_yield, market_close=getattr(inputs, "market_close", None))
        calibration = calibrate(result)
        if not calibration:
            print(f"{ticker}: too little history ({inputs.quarters['EPS'].notna().sum()} EPS quarters, "
                  f"{inputs.quarters.get('Revenue', pd.Series(dtype=float)).notna().sum()} revenue quarters)")
            continue
        quality, today = calibration["quality"], calibration["today"]
        tested = calibration["history"].dropna(subset=["forecast", "realised"])
        rows.append({
            "ticker": ticker, "months": quality["months"], "from": tested.index.min().strftime("%Y-%m"),
            "premium err %": quality["premium_error_pct"], "price err %": quality["price_error_pct"],
            "no-change err %": quality["naive_error_pct"],
            "beats": quality["price_error_pct"] < quality["naive_error_pct"],
            "direction": quality["direction_hit"], "band held": quality["band_coverage"],
            "today": today["side"].split(" ")[0], "gap %": today["gap_pct"],
            "next Q base %": (today["base"] / today["close"] - 1) * 100, "seconds": time.time() - started})
        full = calibration["history"]
        for months in (6, 12, 24, 36):
            full[f"return_{months}"] = np.log(full["close"].shift(-months) / full["close"])
        history = full.dropna(subset=["explained_premium", "realised", "premium", "ttm_eps"])
        history = history.assign(ticker=ticker)
        diagnostics.append(history)
        print(f"{ticker}: done in {time.time() - started:.0f}s")
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)
    print(f"\n=== Premium model, walk-forward, {len(table)} stocks ===")
    print(table.drop(columns=["seconds"]).round(2).to_string(index=False))
    if len(table):
        print(f"\nBeats 'no change' on {table['beats'].sum()} of {len(table)} stocks; "
              f"median price error {table['price err %'].median():.1f}% vs {table['no-change err %'].median():.1f}% "
              f"for 'no change'; direction right {table['direction'].median():.0%} (median stock); band held "
              f"{table['band held'].median():.0%} (60% expected); premium explained to within "
              f"{table['premium err %'].median():.0f}% (median).")
    table.to_csv("validation_premium.csv", index=False)
    if not diagnostics:
        return
    pooled = pd.concat(diagnostics)
    pooled = pooled[pooled["ttm_eps"] > 0]
    actual = np.log(pooled["realised"] / pooled["close"])
    gap = pooled["premium"] - pooled["explained_premium"]
    growth = pooled["growth"].fillna(0)
    print(f"\n=== Diagnostics, {len(pooled)} stock-months pooled ===")
    print("Median abs error of the next-quarter price (%), by how much of today's gap to the explained premium is "
          "kept and whether EPS growth is added:")
    for keep in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for with_growth in (True, False):
            eps_factor = np.log1p(growth / 100) * 0.25 if with_growth else 0.0
            predicted = eps_factor + (keep - 1) * gap   # log change from today's close
            error = (np.exp((predicted - actual).abs().median()) - 1) * 100
            hit = ((predicted > 0) == (actual > 0)).mean()
            print(f"  keep {keep:.2f}, EPS growth {'on ' if with_growth else 'off'}: error {error:5.2f}%, direction {hit:.0%}")
    print(f"Does the gap predict the next quarter's return? corr(-gap, return) = {np.corrcoef(-gap, actual)[0, 1]:.3f}; "
          f"cheapest fifth of months returned {actual[gap <= gap.quantile(0.2)].mean() * 100:+.1f}% a quarter, the "
          f"richest fifth {actual[gap >= gap.quantile(0.8)].mean() * 100:+.1f}% (all months {actual.mean() * 100:+.1f}%).")
    for months in (6, 12, 24, 36):
        column = f"return_{months}"
        part = pooled.dropna(subset=[column])
        part_gap = part["premium"] - part["explained_premium"]
        if len(part) < 100:
            continue
        cheap = part[column][part_gap <= part_gap.quantile(0.2)].mean()
        rich = part[column][part_gap >= part_gap.quantile(0.8)].mean()
        per = part.groupby("ticker").apply(
            lambda frame: np.corrcoef(-(frame["premium"] - frame["explained_premium"]), frame[column])[0, 1]
            if len(frame) > 12 else np.nan, include_groups=False)
        print(f"{months:2d} months on: corr(-gap, return) pooled {np.corrcoef(-part_gap, part[column])[0, 1]:+.3f}, "
              f"per stock median {per.median():+.3f} (positive on {(per > 0).sum()} of {per.notna().sum()}); cheapest fifth "
              f"{cheap * 100:+.1f}%, richest fifth {rich * 100:+.1f}%, all {part[column].mean() * 100:+.1f}% "
              f"(n={len(part)})")
    per_stock = pooled.groupby("ticker").apply(
        lambda part: np.corrcoef(-(part["premium"] - part["explained_premium"]), np.log(part["realised"] / part["close"]))[0, 1],
        include_groups=False)
    print(f"Per stock corr(-gap, next-quarter return): median {per_stock.median():.3f}, positive on "
          f"{(per_stock > 0).sum()} of {len(per_stock)} stocks.")


if __name__ == "__main__":
    main()
