import numpy as np, pandas as pd, pickle, glob, os, warnings; warnings.filterwarnings("ignore")
P = pd.read_pickle("panel.pkl").set_index(["ticker","month"])
add=[]
for path in sorted(glob.glob("data/*.pkl")):
    t=os.path.basename(path)[:-4]; D=pickle.load(open(path,"rb"))["daily"]
    me=D.groupby(D.index.to_period("M")).tail(1)
    imp=me["implied_erp"].astype(float); rule=me["rule_erp"].astype(float)
    f=pd.DataFrame({"abs_dev": imp-rule,
                    "long_dev": imp - imp.shift(1).expanding(36).median(),
                    "dev5y": imp - imp.shift(1).rolling(60, min_periods=48).median()}, index=me.index)
    f["ticker"]=t; f["month"]=f.index.to_period("M"); add.append(f.set_index(["ticker","month"]))
P=P.join(pd.concat(add)).reset_index()
print("+ dev = cheaper than the reference premium. Rank IC per month, mean (share > 0), months:")
for h in ["3m","12m","36m"]:
    y=f"ret_{h}"
    for f in ["prem_dev","dev5y","long_dev","abs_dev"]:
        q=P.dropna(subset=[f,y])
        ic=q.groupby("month").apply(lambda x: x[f].rank().corr(x[y].rank()) if len(x)>10 else np.nan).dropna()
        cheap=q.groupby("month").apply(lambda x: x.loc[x[f].rank(pct=True)>0.8,y].mean()-x.loc[x[f].rank(pct=True)<=0.2,y].mean()).mean()
        print(f"{h:4s} {f:9s} IC {ic.mean():+.3f} ({(ic>0).mean()*100:.0f}%) n_months {len(ic)}  cheapest-minus-richest fifth {cheap:+.1f}% excess")
