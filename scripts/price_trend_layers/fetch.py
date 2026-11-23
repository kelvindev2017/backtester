import sys, pickle, time, os, traceback
exec(open("load_app.py").read())
import numpy as np, pandas as pd
g = G
TICKERS = ["COST","WMT","KO","PEP","PG","JNJ","MCD","HD","NKE","SBUX","AAPL","MSFT","GOOGL","AMZN","META","NVDA","AVGO",
 "ADBE","CRM","ORCL","INTC","TXN","AMAT","KLAC","LRCX","TSLA","UNH","LLY","MRK","ABBV","CAT","DE","HON","NFLX","ISRG",
 "TGT","LOW","QCOM","CVX","XOM"]
if len(sys.argv) > 1: TICKERS = sys.argv[1:]
os.makedirs("data", exist_ok=True)
for t in TICKERS:
    out = f"data/{t}.pkl"
    if os.path.exists(out): continue
    t0 = time.time()
    try:
        params = g["preset_params"](t)
        share, src = g["load_stock_pay_haircut"](t)
        if np.isfinite(share): params["eps_haircut_pct"] = round(share, 1)
        inputs = g["load_model_inputs"](t, 12, [], params["pe_lookback_years"], params["pe_basis"],
            (params["history_growth_follow"], params["history_growth_floor"], params["history_growth_cap"], params["history_recovery"]))
        policy, my, note = g["load_model_rates"](params["yield_years"])
        res = g["run_model"](inputs.quarters, inputs.close, g["preset_register"](t), params, inputs.peer_pe, start=inputs.start,
            own_pe=inputs.own_pe, policy_rate=policy, market_yield=my, market_close=inputs.market_close,
            analyst_growth=inputs.found["growth_check"][0])
        earn = g["load_earnings_data"](t, "2010-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
        pickle.dump({"daily": res["daily"], "quarterly": res["quarterly"], "quarters": inputs.quarters,
                     "close": inputs.close, "market_close": inputs.market_close, "earnings": earn,
                     "forward_from": res["forward_from"], "haircut": params["eps_haircut_pct"]}, open(out, "wb"))
        print(t, "ok", round(time.time()-t0), "s", res["daily"].index.min().date(), flush=True)
    except Exception as e:
        print(t, "FAIL", type(e).__name__, e, flush=True); traceback.print_exc()
