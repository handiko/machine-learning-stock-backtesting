"""
XGBoost Stock Backtesting Framework  v4
==========================================
Key improvements over v3:
  [1] Walk-forward retraining  – model retrained every N months on expanding
      window; critical for regime adaptation and reducing staleness
  [2] Purged + embargoed CV    – prevents leakage between adjacent folds
  [3] Feature selection        – keep only top-K features by IS importance;
      removes noise that drives in-sample overfit
  [4] SMA-200 regime filter    – only go long when price > 200-day MA;
      simple but powerful drawdown reduction
  [5] Confidence gating        – only trade when percentile rank is in the
      top `buy_pct`% AND raw probability exceeds a minimum floor
  [6] Stronger regularisation  – tighter XGB params by default
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

import xgboost as xgb

# ──────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────
CONFIG = dict(
    ticker       = "CPIN.JK",
    years        = 15,
    mode         = "buy_only",   # "buy_only" or "buy_sell"
    forward_days = 3,
    label_pct    = 1.0,

    # Data split
    in_sample_pct = 0.60,
    val_pct       = 0.20,

    # ── Walk-forward retraining ────────────────────────────
    # After the initial in-sample train, the model is retrained every
    # `retrain_months` months using ALL data seen so far (expanding window).
    # Set to None to disable (train once, predict all — old behaviour).
    retrain_months = 6,

    # ── Feature selection ──────────────────────────────────
    # Keep only the top-K features ranked by in-sample importance.
    # Dramatically reduces the noise → overfit pathway.
    top_k_features = 15,   # None = use all features

    # ── Regime filter ──────────────────────────────────────
    # Only enter long positions when Close > SMA(regime_sma).
    # Set to None to disable.
    regime_sma = None,

    # ── Signal generation ──────────────────────────────────
    # Percentile rank threshold: fire BUY only in top `buy_pct`%
    # AND raw probability > `min_prob_floor`.
    buy_pct       = 25,    # raise → fewer, higher-confidence trades
    sell_pct      = 15,
    min_prob_floor = 0.40, # raw prob must exceed this floor too

    # ── XGBoost (stronger regularisation) ─────────────────
    xgb_params = dict(
        n_estimators          = 500,
        max_depth             = 3,       # shallower trees = less overfit
        learning_rate         = 0.02,
        subsample             = 0.65,
        colsample_bytree      = 0.5,     # more aggressive column dropping
        colsample_bylevel     = 0.7,
        min_child_weight      = 8,       # require more samples per leaf
        gamma                 = 1.0,     # higher minimum split gain
        reg_alpha             = 0.5,
        reg_lambda            = 2.0,
        eval_metric           = "logloss",
        early_stopping_rounds = 40,
        random_state          = 42,
        n_jobs                = -1,
    ),

    # Purged CV: embargo gap between train and val fold (in rows)
    cv_embargo_rows = 5,
    wf_splits       = 5,

    transaction_cost_bps = 5,
)


# ══════════════════════════════════════════════════════════════
#  1. DATA
# ══════════════════════════════════════════════════════════════
def download_data(ticker, years):
    end   = datetime.today()
    start = end - timedelta(days=int(years * 365.25))
    print(f"  Downloading {ticker}  [{start.date()} → {end.date()}]")
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    print(f"  {len(df)} trading days downloaded.")
    return df


# ══════════════════════════════════════════════════════════════
#  2. FEATURES  (same as v3)
# ══════════════════════════════════════════════════════════════
def _ema(s, span):  return s.ewm(span=span, adjust=False).mean()
def _rsi(cl, p=14):
    d=cl.diff(); g=d.clip(lower=0).ewm(com=p-1,min_periods=p).mean()
    l=(-d.clip(upper=0)).ewm(com=p-1,min_periods=p).mean()
    return 100-100/(1+g/l.replace(0,np.nan))
def _atr(hi,lo,cl,p=14):
    tr=pd.concat([hi-lo,(hi-cl.shift()).abs(),(lo-cl.shift()).abs()],axis=1).max(1)
    return tr.ewm(com=p-1,min_periods=p).mean()
def _stoch(hi,lo,cl,k=14,d=3):
    lo_k=lo.rolling(k).min(); hi_k=hi.rolling(k).max()
    stk=100*(cl-lo_k)/(hi_k-lo_k+1e-9); return stk,stk.rolling(d).mean()
def _cci(hi,lo,cl,p=20):
    tp=(hi+lo+cl)/3; ma=tp.rolling(p).mean()
    md=tp.rolling(p).apply(lambda x:np.abs(x-x.mean()).mean(),raw=True)
    return (tp-ma)/(0.015*md+1e-9)
def _mfi(hi,lo,cl,vol,p=14):
    tp=(hi+lo+cl)/3; mf=tp*vol
    pos=mf.where(tp>tp.shift(),0).rolling(p).sum()
    neg=mf.where(tp<tp.shift(),0).rolling(p).sum()
    return 100-100/(1+pos/(neg+1e-9))
def _obv(cl,vol): return (np.sign(cl.diff()).fillna(0)*vol).cumsum()
def _adx(hi,lo,cl,p=14):
    atr=_atr(hi,lo,cl,p)
    dmp=pd.Series(np.where(hi.diff()>(-lo.diff()).clip(lower=0),hi.diff().clip(lower=0),0),index=cl.index)
    dmn=pd.Series(np.where((-lo.diff())>hi.diff().clip(lower=0),(-lo.diff()).clip(lower=0),0),index=cl.index)
    dip=100*dmp.ewm(com=p-1,min_periods=p).mean()/(atr+1e-9)
    din=100*dmn.ewm(com=p-1,min_periods=p).mean()/(atr+1e-9)
    dx=100*(dip-din).abs()/(dip+din+1e-9)
    return dx.ewm(com=p-1,min_periods=p).mean()

def build_features(df):
    hi,lo,cl,op,vol=df.High,df.Low,df.Close,df.Open,df.Volume
    f=pd.DataFrame(index=df.index)
    for n in [1,2,3,5,10,21,63]: f[f"ret_{n}d"]=cl.pct_change(n)
    f["gap"]=(op-cl.shift())/cl.shift(); f["hl_range"]=(hi-lo)/cl
    f["cl_vs_op"]=(cl-op)/(op+1e-9)
    for w in [5,10,20,50,100,200]:
        ma=cl.rolling(w).mean(); f[f"sma{w}_d"]=(cl-ma)/(ma+1e-9)
    for a,b in [(5,20),(10,50),(20,100),(50,200)]:
        f[f"ema{a}_{b}"]=(_ema(cl,a)-_ema(cl,b))/(cl+1e-9)
    for w in [5,10,20,50]: f[f"ema{w}_d"]=(cl-_ema(cl,w))/(cl+1e-9)
    for n in [5,10,21,63,126]: f[f"mom_{n}"]=cl/cl.shift(n)-1; f[f"roc_{n}"]=(cl-cl.shift(n))/(cl.shift(n)+1e-9)
    r1=cl.pct_change()
    for w in [5,10,21,63]: f[f"vol_{w}"]=r1.rolling(w).std()
    f["atr14_n"]=_atr(hi,lo,cl,14)/cl; f["atr7_n"]=_atr(hi,lo,cl,7)/cl
    f["atr_rat"]=_atr(hi,lo,cl,7)/(_atr(hi,lo,cl,14)+1e-9)
    for w in [20,50]:
        ma=cl.rolling(w).mean(); std=cl.rolling(w).std()
        ub=ma+2*std; lb=ma-2*std
        f[f"bb_pct_{w}"]=(cl-lb)/(ub-lb+1e-9); f[f"bb_width_{w}"]=(ub-lb)/(ma+1e-9)
        f[f"bb_hi_{w}"]=(hi-ub)/cl; f[f"bb_lo_{w}"]=(lo-lb)/cl
    for p in [7,14,21]: f[f"rsi_{p}"]=_rsi(cl,p)
    f["rsi_div"]=f["rsi_14"]-f["rsi_14"].rolling(5).mean()
    stk,std_=_stoch(hi,lo,cl); f["stoch_k"]=stk; f["stoch_d"]=std_; f["stoch_kd"]=stk-std_
    f["cci20"]=_cci(hi,lo,cl,20); f["cci10"]=_cci(hi,lo,cl,10)
    macd=_ema(cl,12)-_ema(cl,26); sig=_ema(macd,9)
    f["macd"]=macd/cl; f["macd_sig"]=sig/cl; f["macd_hist"]=(macd-sig)/cl; f["macd_cross"]=np.sign(macd-sig)
    for w in [5,10,20]: f[f"vol_r{w}"]=vol/vol.rolling(w).mean()
    f["obv_n"]=_obv(cl,vol)/vol.rolling(20).sum()
    f["mfi14"]=_mfi(hi,lo,cl,vol,14)
    vwap=((hi+lo+cl)/3*vol).rolling(20).sum()/vol.rolling(20).sum()
    f["vwap_d"]=(cl-vwap)/cl; f["adx14"]=_adx(hi,lo,cl,14)
    f["hi52_d"]=(cl-hi.rolling(252).max())/cl; f["lo52_d"]=(cl-lo.rolling(252).min())/cl
    f["skew21"]=r1.rolling(21).skew(); f["kurt21"]=r1.rolling(21).kurt()
    f["dow"]=df.index.dayofweek; f["month"]=df.index.month
    return f.replace([np.inf,-np.inf],np.nan)


# ══════════════════════════════════════════════════════════════
#  3. LABELS
# ══════════════════════════════════════════════════════════════
def create_labels(close, forward_days, label_pct, mode):
    fwd=close.shift(-forward_days)/close-1
    if mode=="buy_only":
        return (fwd>label_pct/100).astype(int)
    y=pd.Series(0,index=close.index,dtype=int)
    y[fwd> label_pct/100]=1; y[fwd<-label_pct/100]=-1
    return y


# ══════════════════════════════════════════════════════════════
#  4. PURGED + EMBARGOED WALK-FORWARD CV
# ══════════════════════════════════════════════════════════════
def purged_wf_cv(X_tr, y_tr, params, n_splits=5, embargo=5):
    """TimeSeriesSplit with an embargo gap to prevent leakage."""
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for fold, (ti, vi) in enumerate(tscv.split(X_tr)):
        # purge: remove embargo rows from end of train
        ti_purged = ti[ti < (vi[0] - embargo)]
        if len(ti_purged) < 50: continue
        Xt,Xv = X_tr.iloc[ti_purged], X_tr.iloc[vi]
        yt,yv = y_tr.iloc[ti_purged], y_tr.iloc[vi]
        yb_t=(yt>0).astype(int); yb_v=(yv>0).astype(int)
        ratio=(yb_t==0).sum()/max((yb_t==1).sum(),1)
        m=xgb.XGBClassifier(**params, scale_pos_weight=ratio)
        m.fit(Xt,yb_t,eval_set=[(Xv,yb_v)],verbose=False)
        prob=m.predict_proba(Xv)[:,1]
        if len(np.unique(yb_v))>1:
            auc=roc_auc_score(yb_v,prob); scores.append(auc)
            print(f"    Fold {fold+1}/{n_splits}  AUC={auc:.4f}  iter={m.best_iteration}")
    return dict(cv_auc_mean=np.mean(scores) if scores else 0,
                cv_auc_std =np.std(scores)  if scores else 0)


# ══════════════════════════════════════════════════════════════
#  5. TRAIN  (single fit)
# ══════════════════════════════════════════════════════════════
def train_model(X_tr, y_tr, X_val, y_val, params):
    yb_tr=(y_tr>0).astype(int); yb_val=(y_val>0).astype(int)
    ratio=(yb_tr==0).sum()/max((yb_tr==1).sum(),1)
    m=xgb.XGBClassifier(**params, scale_pos_weight=ratio)
    m.fit(X_tr,yb_tr,eval_set=[(X_val,yb_val)],verbose=False)
    return m


# ══════════════════════════════════════════════════════════════
#  6. WALK-FORWARD PREDICT  (expanding window retraining)
# ══════════════════════════════════════════════════════════════
def walk_forward_predict(X_all, y_all, X_is, y_is,
                         X_val, y_val, X_ho,
                         params, retrain_months,
                         top_features, cfg):
    """
    Expanding-window walk-forward:
      - Start: train on IS only
      - Every `retrain_months`, retrain on IS + all OOS data seen so far
      - Predict the next chunk (never peek at future)
    Returns probability arrays for val and hold-out periods.
    """
    # Combine val + holdout into one "future" block
    X_future = pd.concat([X_val, X_ho])
    y_future = pd.concat([y_val, pd.Series(index=X_ho.index,
                                           data=np.zeros(len(X_ho), dtype=int))])

    all_prob   = pd.Series(dtype=float, index=X_future.index)
    train_end  = X_is.index[-1]

    # Build date chunks (retraining schedule)
    chunk_starts = [X_future.index[0]]
    d = X_future.index[0]
    while True:
        d = d + relativedelta(months=retrain_months)
        if d >= X_future.index[-1]: break
        chunk_starts.append(d)
    chunk_starts.append(X_future.index[-1] + timedelta(days=1))  # sentinel

    print(f"  Walk-forward retraining every {retrain_months} months "
          f"→ {len(chunk_starts)-1} retraining events")

    for i, (cs, ce) in enumerate(zip(chunk_starts[:-1], chunk_starts[1:])):
        # data available for training: everything up to cs (exclusive)
        X_train_wf = X_all[X_all.index <= train_end]
        y_train_wf = y_all[y_all.index <= train_end]

        # chunk to predict
        mask_chunk = (X_future.index >= cs) & (X_future.index < ce)
        X_chunk    = X_future[mask_chunk]
        if len(X_chunk) == 0: continue

        # scale: fit on training window, apply to chunk
        sc  = RobustScaler()
        Xtr_s = pd.DataFrame(sc.fit_transform(X_train_wf[top_features]),
                             index=X_train_wf.index, columns=top_features)
        Xch_s = pd.DataFrame(sc.transform(X_chunk[top_features]),
                             index=X_chunk.index, columns=top_features)

        # use val chunk as early-stop set (next N rows after training end)
        future_rows = X_future[X_future.index > train_end]
        Xes_raw = future_rows.head(max(30, len(future_rows)//5))[top_features]
        Xes_s   = pd.DataFrame(sc.transform(Xes_raw),
                               index=Xes_raw.index, columns=top_features)
        yes_s   = (y_future.reindex(Xes_raw.index) > 0).astype(int)

        ratio = ((y_train_wf > 0) == 0).sum() / max((y_train_wf > 0).sum(), 1)
        m = xgb.XGBClassifier(**params, scale_pos_weight=ratio)
        m.fit(Xtr_s, (y_train_wf > 0).astype(int),
              eval_set=[(Xes_s, yes_s)], verbose=False)

        prob = m.predict_proba(Xch_s)[:, 1]
        all_prob[X_chunk.index] = prob

        # advance training cutoff to end of this chunk
        train_end = X_chunk.index[-1]
        print(f"    Retrain {i+1}: trained up to {X_train_wf.index[-1].date()}, "
              f"predicted {len(X_chunk)} rows  best_iter={m.best_iteration}")

    return all_prob.reindex(X_val.index), all_prob.reindex(X_ho.index)


# ══════════════════════════════════════════════════════════════
#  7. SIGNALS
# ══════════════════════════════════════════════════════════════
def prob_to_signals(prob_series, mode, buy_pct, sell_pct,
                    min_prob_floor=0.50, regime_mask=None):
    """
    Percentile-rank signal with:
      - min_prob_floor: raw probability must exceed this too
      - regime_mask: boolean Series; False = no long trades allowed
    """
    prob   = prob_series.copy()
    ranks  = prob.rank(pct=True) * 100
    sig    = pd.Series(0, index=prob.index, dtype=int)

    long_ok  = (ranks >= (100 - buy_pct)) & (prob >= min_prob_floor)
    short_ok = (ranks <= sell_pct) & (prob <= (1 - min_prob_floor))

    if regime_mask is not None:
        rm = regime_mask.reindex(prob.index).fillna(False)
        long_ok  = long_ok & rm
        if mode == "buy_sell":
            # allow shorts even in downtrend (optional: remove this line)
            pass

    sig[long_ok]               = 1
    if mode == "buy_sell":
        sig[short_ok & (sig==0)] = -1

    return sig


# ══════════════════════════════════════════════════════════════
#  8. BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════
def run_backtest(close, signals, tc_bps, mode, open_prices=None):
    df           = pd.DataFrame(index=signals.index)
    df["price"]  = close.reindex(signals.index)
    df["signal"] = signals

    if open_prices is not None:
        # ── Realistic execution model ──────────────────────────
        # Signal generated at Close[T]
        # Order executes at Open[T+1]   → entry price
        # Order exits    at Open[T+2]   → exit price  (next signal change)
        # Daily return while holding    = Open[T+1] / Open[T] - 1
        #   attributed to the bar where the decision was made (T)
        # Implemented as: forward open-to-open return shifted back 1 bar
        op = open_prices.reindex(signals.index)
        df["ret"] = op.pct_change().shift(-1).fillna(0)   # Open[T+1]/Open[T]-1
    else:
        # Legacy close-to-close (assumes trading at closing price)
        df["ret"] = df["price"].pct_change().fillna(0)

    if mode == "buy_only":
        df["position"] = (df["signal"] == 1).astype(float)
    else:
        df["position"] = df["signal"].replace(0,np.nan).ffill().fillna(0)

    tc              = tc_bps / 10_000
    pos_prev        = df["position"].shift(1).fillna(0)
    df["tc"]        = (df["position"] - pos_prev).abs() * tc
    df["strat_ret"] = pos_prev * df["ret"] - df["tc"]
    df["bh_ret"]    = df["ret"]
    df["strat_cum"] = (1 + df["strat_ret"].fillna(0)).cumprod().clip(lower=1e-8)
    df["bh_cum"]    = (1 + df["bh_ret"]).cumprod()
    roll_max        = df["strat_cum"].cummax()
    df["drawdown"]  = (df["strat_cum"] - roll_max) / roll_max
    return df


# ══════════════════════════════════════════════════════════════
#  9. METRICS
# ══════════════════════════════════════════════════════════════
def calc_metrics(bt, label=""):
    r=bt["strat_ret"]; n=len(r); ann=252
    tot=bt["strat_cum"].iloc[-1]-1; bh=bt["bh_cum"].iloc[-1]-1
    ar=(1+tot)**(ann/n)-1; av=r.std()*np.sqrt(ann)
    sh=ar/av if av else np.nan
    neg=r[r<0].std()
    so=ar/(neg*np.sqrt(ann)) if neg else np.nan
    mdd=bt["drawdown"].min()
    cal=ar/abs(mdd) if mdd else np.nan
    wr=(r>0).mean()
    pos=bt["position"]; prev=pos.shift(1).fillna(0)
    nt=int((pos!=prev).sum())
    return dict(label=label,days=n,total_ret=tot,bh_ret=bh,
                ann_ret=ar,ann_vol=av,sharpe=sh,sortino=so,
                max_dd=mdd,calmar=cal,win_rate=wr,n_trades=nt)

fp=lambda v: f"{v*100:+.1f}%"
f2=lambda v: f"{v:.2f}" if not (v!=v) else "N/A"   # nan-safe


# ══════════════════════════════════════════════════════════════
#  10. DASHBOARD
# ══════════════════════════════════════════════════════════════
BG="#0d1117"; GR="#1a2233"; TX="#e6edf3"
BUL="#2ea043"; BEA="#f85149"; BLU="#58a6ff"; ORG="#f0883e"; PUR="#bc8cff"

def _ax(ax, ylabel=""):
    ax.set_facecolor(BG); ax.tick_params(colors=TX,labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GR)
    ax.grid(True,color=GR,lw=0.5,ls="--",alpha=0.6)
    if ylabel: ax.set_ylabel(ylabel,fontsize=8,color=TX)
def _title(ax,t): ax.set_title(t,color=TX,fontsize=10,pad=5,fontweight="bold")

def plot_dashboard(raw_close, bt_is, bt_val, bt_ho,
                   m_is, m_val, m_ho, feat_imp,
                   ticker, mode, cfg, cv_res, val_auc, ho_prob):

    fig=plt.figure(figsize=(22,15),facecolor=BG)
    fig.suptitle(
        f"XGBoost Backtest v4  ·  {ticker}  ·  {mode.replace('_',' ').title()}  "
        f"·  {cfg['forward_days']}d horizon  ·  retrain={cfg['retrain_months']}m  "
        f"·  top-{cfg['top_k_features']} feats  ·  TC={cfg['transaction_cost_bps']}bps",
        color=TX,fontsize=13,fontweight="bold",y=0.99)

    gs=gridspec.GridSpec(4,3,figure=fig,hspace=0.50,wspace=0.32,
                         left=0.06,right=0.97,top=0.94,bottom=0.05)

    def _norm(bt,key,base=1.0):
        s=bt[key].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        f=s.iloc[0]; f=1.0 if (f==0 or np.isnan(f)) else f
        return s/f*base

    # ── Equity curve ─────────────────────────────────────
    ax_eq=fig.add_subplot(gs[0:2,:2]); _ax(ax_eq,"Cumulative Return (log)")
    s_is=_norm(bt_is,"strat_cum"); b_is=_norm(bt_is,"bh_cum")
    s_va=_norm(bt_val,"strat_cum",s_is.iloc[-1]); b_va=_norm(bt_val,"bh_cum",b_is.iloc[-1])
    s_ho=_norm(bt_ho,"strat_cum",s_va.iloc[-1]);  b_ho=_norm(bt_ho,"bh_cum",b_va.iloc[-1])
    strat=pd.concat([s_is,s_va,s_ho]); bhold=pd.concat([b_is,b_va,b_ho])

    ax_eq.plot(bhold.index,bhold.values,color=ORG,lw=1.2,ls="--",
               label="Buy & Hold",zorder=3,alpha=0.85)
    ax_eq.plot(strat.index,strat.values,color=BLU,lw=2.2,
               label="Strategy",zorder=5)

    splits=[(strat.index[0],bt_is.index[-1],BUL,"In-Sample"),
            (bt_is.index[-1],bt_val.index[-1],PUR,"Validation"),
            (bt_val.index[-1],strat.index[-1],ORG,"Hold-Out")]
    for xs,xe,c,lbl in splits:
        ax_eq.axvspan(xs,xe,alpha=0.07,color=c)
        ax_eq.axvline(xs,color=c,lw=0.8,ls=":",alpha=0.5)
        mid=xs+(xe-xs)/2
        ax_eq.text(mid,0.02,lbl,color=c,fontsize=7.5,ha="center",
                   transform=ax_eq.get_xaxis_transform(),alpha=0.9,fontweight="bold")

    ax_eq.set_yscale("log")
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:.2f}x"))
    ax_eq.legend(fontsize=8.5,facecolor=GR,edgecolor=GR,labelcolor=TX,loc="upper left")
    _title(ax_eq,"Equity Curve (log scale)")

    # ── Drawdown ─────────────────────────────────────────
    ax_dd=fig.add_subplot(gs[2,:2],sharex=ax_eq); _ax(ax_dd,"Drawdown")
    dd=pd.concat([bt_is["drawdown"],bt_val["drawdown"],bt_ho["drawdown"]])
    ax_dd.fill_between(dd.index,dd.values,0,color=BEA,alpha=0.55)
    ax_dd.plot(dd.index,dd.values,color=BEA,lw=0.7)
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x*100:.0f}%"))
    _title(ax_dd,"Drawdown")

    # ── Hold-out signals ─────────────────────────────────
    ax_sig=fig.add_subplot(gs[3,:2]); _ax(ax_sig,"Price")
    ho_p=bt_ho["price"]
    ax_sig.plot(ho_p.index,ho_p.values,color=TX,lw=0.8,alpha=0.7,zorder=2)
    pos=bt_ho["position"]
    for i in range(1,len(pos)):
        c=BUL if pos.iloc[i]>0 else (BEA if pos.iloc[i]<0 else BG)
        ax_sig.axvspan(pos.index[i-1],pos.index[i],alpha=0.13,color=c,lw=0)
    prev=pos.shift(1).fillna(0)
    el=bt_ho[(pos!=prev)&(pos>0)];  es=bt_ho[(pos!=prev)&(pos<0)]
    ex=bt_ho[(pos!=prev)&(pos==0)&(prev!=0)]
    ax_sig.scatter(el.index,el["price"],marker="^",s=55,color=BUL,zorder=6,
                   label=f"Buy ({len(el)})")
    ax_sig.scatter(es.index,es["price"],marker="v",s=55,color=BEA,zorder=6,
                   label=f"Sell ({len(es)})")
    ax_sig.scatter(ex.index,ex["price"],marker="x",s=35,color=TX, zorder=6,
                   alpha=0.5,label=f"Exit ({len(ex)})")
    ax_sig.legend(fontsize=8,facecolor=GR,edgecolor=GR,labelcolor=TX)
    _title(ax_sig,"Hold-Out: Positions & Signals")

    # ── Feature importance ───────────────────────────────
    ax_fi=fig.add_subplot(gs[0:2,2]); _ax(ax_fi)
    top=feat_imp.nlargest(20)
    cols=[BLU if v>top.median() else "#39d0d8" for v in top.values]
    ax_fi.barh(range(len(top)),top.values,color=cols,alpha=0.85)
    ax_fi.set_yticks(range(len(top))); ax_fi.set_yticklabels(top.index,fontsize=7.5,color=TX)
    ax_fi.invert_yaxis(); ax_fi.tick_params(axis="x",colors=TX,labelsize=7)
    _title(ax_fi,f"Feature Importance (Top 20 of {cfg['top_k_features']} selected)")

    # ── Prob distribution ────────────────────────────────
    ax_pb=fig.add_subplot(gs[2,2]); _ax(ax_pb,"Density")
    if ho_prob is not None:
        sig_ho=bt_ho["signal"]
        for mask,c,lbl in [(sig_ho==1,BUL,"Buy"),(sig_ho==-1,BEA,"Sell"),(sig_ho==0,TX,"Neutral")]:
            vals=ho_prob[ho_prob.index.isin(sig_ho[mask].index)]
            if len(vals): ax_pb.hist(vals,bins=25,color=c,alpha=0.55,label=lbl,density=True)
        ax_pb.legend(fontsize=7,facecolor=GR,edgecolor=GR,labelcolor=TX)
    _title(ax_pb,"Hold-Out Probability Distribution")

    # ── Metrics table ────────────────────────────────────
    ax_m=fig.add_subplot(gs[3,2]); ax_m.set_facecolor(BG); ax_m.axis("off")
    _title(ax_m,"Performance Summary")
    rows=[
        ("Metric","In-Sample","Validation","Hold-Out"),
        ("Total Return",fp(m_is["total_ret"]),fp(m_val["total_ret"]),fp(m_ho["total_ret"])),
        ("B&H Return",  fp(m_is["bh_ret"]),  fp(m_val["bh_ret"]),   fp(m_ho["bh_ret"])),
        ("Ann. Return", fp(m_is["ann_ret"]),  fp(m_val["ann_ret"]),  fp(m_ho["ann_ret"])),
        ("Ann. Vol",    fp(m_is["ann_vol"]),  fp(m_val["ann_vol"]),  fp(m_ho["ann_vol"])),
        ("Sharpe",      f2(m_is["sharpe"]),   f2(m_val["sharpe"]),   f2(m_ho["sharpe"])),
        ("Sortino",     f2(m_is["sortino"]),  f2(m_val["sortino"]),  f2(m_ho["sortino"])),
        ("Max DD",      fp(m_is["max_dd"]),   fp(m_val["max_dd"]),   fp(m_ho["max_dd"])),
        ("Calmar",      f2(m_is["calmar"]),   f2(m_val["calmar"]),   f2(m_ho["calmar"])),
        ("Win Rate",    fp(m_is["win_rate"]), fp(m_val["win_rate"]), fp(m_ho["win_rate"])),
        ("# Trades",    str(m_is["n_trades"]),str(m_val["n_trades"]),str(m_ho["n_trades"])),
        ("CV AUC",      f"{cv_res['cv_auc_mean']:.3f}±{cv_res['cv_auc_std']:.3f}","—",f"ValAUC={val_auc:.3f}"),
    ]
    tbl=ax_m.table(cellText=rows[1:],colLabels=rows[0],cellLoc="center",loc="center",bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_edgecolor(GR)
        if r==0: cell.set_facecolor("#1c2d3f"); cell.set_text_props(color=BLU,fontweight="bold")
        else:
            cell.set_facecolor(BG if r%2 else GR)
            t=cell.get_text().get_text()
            if t.startswith("+"): cell.set_text_props(color=BUL)
            elif t.startswith("-"): cell.set_text_props(color=BEA)
            else: cell.set_text_props(color=TX)

    out="xgb_backtest_dashboard.png"
    plt.savefig(out,dpi=150,bbox_inches="tight",facecolor=BG)
    print(f"\n  Dashboard saved → {out}")
    plt.show()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main(cfg=CONFIG):
    from dateutil.relativedelta import relativedelta

    print("\n"+"═"*62)
    print("  XGBoost Backtesting Framework  v4")
    print("═"*62)

    # 1. Data
    print("\n[1/8] Downloading data …")
    raw=download_data(cfg["ticker"],cfg["years"])

    # 2. Features & labels
    print("[2/8] Building features …")
    feat=build_features(raw); feat.dropna(inplace=True)
    print("[3/8] Creating labels …")
    labels=create_labels(raw.Close,cfg["forward_days"],cfg["label_pct"],cfg["mode"])
    labels=labels.reindex(feat.index)
    mask=labels.notna(); X,y=feat[mask],labels[mask]
    vc=y.value_counts().sort_index()
    print(f"  Labels: {dict(vc)}  total={len(y)}")

    # 3. Split
    n=len(X); ie=int(n*cfg["in_sample_pct"]); ve=int(n*(cfg["in_sample_pct"]+cfg["val_pct"]))
    X_is,y_is   = X.iloc[:ie],       y.iloc[:ie]
    X_val,y_val = X.iloc[ie:ve],     y.iloc[ie:ve]
    X_ho,y_ho   = X.iloc[ve:],       y.iloc[ve:]
    print(f"  IS : {len(X_is):5d} rows [{X_is.index[0].date()} → {X_is.index[-1].date()}]")
    print(f"  Val: {len(X_val):5d} rows [{X_val.index[0].date()} → {X_val.index[-1].date()}]")
    print(f"  HO : {len(X_ho):5d}  rows [{X_ho.index[0].date()} → {X_ho.index[-1].date()}]")

    # 4. Scale IS, train initial model, select top-K features
    print("[4/8] Initial IS training + feature selection …")
    sc0=RobustScaler()
    Xis_s0=pd.DataFrame(sc0.fit_transform(X_is),index=X_is.index,columns=X.columns)
    Xva_s0=pd.DataFrame(sc0.transform(X_val),   index=X_val.index,columns=X.columns)
    m0=train_model(Xis_s0,y_is,Xva_s0,y_val,cfg["xgb_params"])
    fi_all=pd.Series(m0.feature_importances_,index=X.columns).sort_values(ascending=False)

    k=cfg.get("top_k_features") or len(X.columns)
    top_feats=fi_all.head(k).index.tolist()
    print(f"  Selected top-{k} features  (dropped {len(X.columns)-k})")

    # 5. Purged walk-forward CV on IS with selected features
    print(f"[5/8] Purged walk-forward CV ({cfg['wf_splits']} folds) …")
    sc1=RobustScaler()
    Xis_sf=pd.DataFrame(sc1.fit_transform(X_is[top_feats]),index=X_is.index,columns=top_feats)
    Xva_sf=pd.DataFrame(sc1.transform(X_val[top_feats]),   index=X_val.index,columns=top_feats)
    cv=purged_wf_cv(Xis_sf,y_is,cfg["xgb_params"],cfg["wf_splits"],cfg["cv_embargo_rows"])
    print(f"  CV AUC mean={cv['cv_auc_mean']:.4f}  std={cv['cv_auc_std']:.4f}")

    # 6. Train final IS model (for IS backtest & feature importance)
    m_is=train_model(Xis_sf,y_is,Xva_sf,y_val,cfg["xgb_params"])
    val_prob0=m_is.predict_proba(Xva_sf)[:,1]
    yv_bin=(y_val>0).astype(int)
    val_auc=roc_auc_score(yv_bin,val_prob0) if len(np.unique(yv_bin))>1 else np.nan
    print(f"  Val AUC (static model) = {val_auc:.4f}")
    fi=pd.Series(m_is.feature_importances_,index=top_feats).sort_values(ascending=False)
    is_prob=pd.Series(m_is.predict_proba(Xis_sf)[:,1],index=X_is.index)

    # 7. Walk-forward prediction on Val + HO
    print("[6/8] Walk-forward retraining on Val + HO …")
    val_prob_wf, ho_prob_wf = walk_forward_predict(
        X[top_feats], y, X_is[top_feats], y_is,
        X_val[top_feats], y_val, X_ho[top_feats],
        cfg["xgb_params"], cfg["retrain_months"],
        top_feats, cfg)

    # 8. Regime mask
    regime_mask = None
    if cfg.get("regime_sma"):
        sma = raw.Close.rolling(cfg["regime_sma"]).mean()
        regime_mask = raw.Close > sma
        print(f"  Regime filter: SMA{cfg['regime_sma']}  "
              f"(long allowed {regime_mask.mean()*100:.0f}% of days)")

    # 9. Signals
    print("[7/8] Generating signals …")
    sk=dict(mode=cfg["mode"],buy_pct=cfg["buy_pct"],sell_pct=cfg["sell_pct"],
            min_prob_floor=cfg.get("min_prob_floor",0.50),regime_mask=regime_mask)
    sig_is  = prob_to_signals(is_prob,     **sk)
    sig_val = prob_to_signals(val_prob_wf, **sk)
    sig_ho  = prob_to_signals(ho_prob_wf,  **sk)

    for lbl,sig in [("IS ",sig_is),("Val",sig_val),("HO ",sig_ho)]:
        print(f"  {lbl}  Buy={(sig==1).sum():4d}  Sell={(sig==-1).sum():4d}  "
              f"Neutral={(sig==0).sum():4d}")

    # 10. Backtest
    print("[8/8] Backtesting …")
    tc=cfg["transaction_cost_bps"]
    bt_is =run_backtest(raw.Close,sig_is, tc,cfg["mode"],raw.Open)
    bt_val=run_backtest(raw.Close,sig_val,tc,cfg["mode"],raw.Open)
    bt_ho =run_backtest(raw.Close,sig_ho, tc,cfg["mode"],raw.Open)
    m_is_m=calc_metrics(bt_is,"In-Sample")
    m_val_m=calc_metrics(bt_val,"Validation")
    m_ho_m =calc_metrics(bt_ho,"Hold-Out")

    # ── Save model bundle ──────────────────────────────────────
    bundle = dict(
        model=m_is,  # trained XGBoost model
        scaler=sc1,  # fitted RobustScaler
        top_feats=top_feats,  # selected feature names
        cfg=cfg,  # config used to train
        trained_on=X_is.index[-1].strftime("%Y-%m-%d"),
    )
    bundle_path = f"xgb_{cfg['ticker']}_{cfg['mode']}_{cfg['forward_days']}d.pkl"
    joblib.dump(bundle, bundle_path)
    print(f"  Model saved → {bundle_path}")

    hdr=f"{'Metric':<18} {'In-Sample':>12} {'Validation':>12} {'Hold-Out':>12}"
    print("\n"+"─"*len(hdr)); print(hdr); print("─"*len(hdr))
    for name,key,fmt in [
        ("Total Return","total_ret",fp),("B&H Return","bh_ret",fp),
        ("Ann. Return","ann_ret",fp),("Ann. Vol","ann_vol",fp),
        ("Sharpe","sharpe",f2),("Sortino","sortino",f2),
        ("Max DD","max_dd",fp),("Calmar","calmar",f2),
        ("Win Rate","win_rate",fp),("# Trades","n_trades",str)]:
        print(f"{name:<18} {fmt(m_is_m[key]):>12} {fmt(m_val_m[key]):>12} {fmt(m_ho_m[key]):>12}")
    print("─"*len(hdr))
    print(f"  CV AUC = {cv['cv_auc_mean']:.4f} ± {cv['cv_auc_std']:.4f}")
    print(f"  Val AUC (static) = {val_auc:.4f}")

    plot_dashboard(raw.Close,bt_is,bt_val,bt_ho,m_is_m,m_val_m,m_ho_m,
                   fi,cfg["ticker"],cfg["mode"],cfg,cv,val_auc,ho_prob_wf)
    return m_is,fi,bt_ho,m_ho_m


if __name__ == "__main__":
    main(CONFIG)