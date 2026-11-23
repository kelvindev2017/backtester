import numpy as np, pandas as pd, pickle, glob, os, warnings; warnings.filterwarnings("ignore")
P = pd.read_pickle("panel.pkl")
# How the gap closes: price vs fair value, per horizon
rows=[]
for path in sorted(glob.glob("data/*.pkl")):
    t=os.path.basename(path)[:-4]; D=pickle.load(open(path,"rb"))["daily"]
    c=D["close"].astype(float); fv=D["model_price"].astype(float).where(lambda s:s>0)
    me=D.groupby(D.index.to_period("M")).tail(1).index
    for n,h in [(63,"3m"),(252,"12m"),(756,"36m")]:
        gap=np.log(c/fv); dp=np.log(c.shift(-n)/c); dfv=np.log(fv.shift(-n)/fv)
        rows.append(pd.DataFrame({"t":t,"h":h,"gap":gap,"dp":dp,"dfv":dfv}).loc[me].dropna())
X=pd.concat(rows)
X=X[X.gap.abs()<1.5]
print("Regression slope on today's gap (log). Gap closing fully through price => price slope -1.")
for h,g in X.groupby("h",sort=False):
    sp=np.polyfit(g.gap,g.dp,1)[0]; sf=np.polyfit(g.gap,g.dfv,1)[0]
    print(f"{h}: price change slope {sp:+.2f}, fair value change slope {sf:+.2f}, n={len(g)}")
print("\nUnivariate pooled rank IC per month (feature vs next excess return), mean over months, and share of months > 0:")
feats=["gap","prem_dev","rev_growth","growth_accel","margin_change","eps_growth","surprise","momentum"]
for h in ["3m","12m","36m"]:
    y=f"ret_{h}"; out={}
    for f in feats:
        ic=P.dropna(subset=[f,y]).groupby("month").apply(lambda x: x[f].rank().corr(x[y].rank()) if len(x)>10 else np.nan)
        early=ic[ic.index.year<2020].mean(); late=ic[ic.index.year>=2020].mean()
        out[f]=f"{ic.mean():+.3f} ({(ic>0).mean()*100:.0f}%)  pre-2020 {early:+.2f} / 2020+ {late:+.2f}"
    print(h); [print(f"   {k:14s} {v}") for k,v in out.items()]
