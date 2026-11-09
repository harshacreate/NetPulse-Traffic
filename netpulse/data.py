"""Synthetic chronological series for NetPulse-Traffic — Telecommunications Traffic Forecasting; not sourced from operational data."""
from __future__ import annotations
import numpy as np
FEATURES=["hour_sin","hour_cos","week_sin","week_cos","trend","temperature","rainfall","event_flag","lag_1","lag_24","lag_168","rolling_mean_24"]
BASELINE_FEATURE="lag_24"
UNIT='relative synthetic throughput units'
def make_dataset(n=1200,seed=42):
    if n<240:raise ValueError("n must be at least 240")
    rng=np.random.default_rng(seed)
    t=np.arange(168,168+n);full=np.arange(n+168);hour=full%24;dow=(full//24)%7;temperature=18+9*np.sin(2*np.pi*(full/24-5)/365)+rng.normal(0,2,n+168);rain=np.clip(rng.gamma(1,1.3,n+168),0,12);event=(rng.random(n+168)<.02).astype(float);maintenance=(rng.random(n+168)<.01).astype(float);noise=rng.normal(0,8,n+168);burst=np.zeros(n+168);burst[maintenance>0]=-24;series=np.maximum(0,115+28*np.sin(2*np.pi*(hour-15)/24)+15*np.sin(2*np.pi*dow/7)+.4*temperature-1.1*rain+38*event+burst+noise);X=np.column_stack([np.sin(2*np.pi*hour[t]/24),np.cos(2*np.pi*hour[t]/24),np.sin(2*np.pi*dow[t]/7),np.cos(2*np.pi*dow[t]/7),t/(n+168),temperature[t],rain[t],event[t],series[t-1],series[t-24],series[t-168],np.array([series[k-24:k].mean() for k in t])]);y=series[t]
    if X.shape!=(n,len(FEATURES)) or y.shape!=(n,) or not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError("invalid generated time series")
    return X.astype(float),y.astype(float)
