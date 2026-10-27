"""Derived features. All functions take *already-clock-filtered* history frames."""
from __future__ import annotations
import numpy as np
import pandas as pd


def returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def realized_vol(close: pd.Series, window: int = 20, annualize: bool = True) -> float | None:
    r = returns(close).tail(window)
    if len(r) < max(5, window // 2):
        return None
    v = float(r.std())
    return round(v * np.sqrt(252) if annualize else v, 4)


def momentum(close: pd.Series, window: int) -> float | None:
    if len(close) <= window:
        return None
    return round(float(close.iloc[-1] / close.iloc[-1 - window] - 1), 4)


def rsi(close: pd.Series, window: int = 14) -> float | None:
    if len(close) <= window:
        return None
    d = close.diff().dropna()
    up = d.clip(lower=0).rolling(window).mean()
    dn = (-d.clip(upper=0)).rolling(window).mean()
    gain, loss = float(up.iloc[-1]), float(dn.iloc[-1])
    if not np.isfinite(gain) or not np.isfinite(loss):
        return None
    if loss == 0:
        return 100.0 if gain > 0 else 50.0
    return round(100 - 100 / (1 + gain / loss), 2)



def sma_ratio(close: pd.Series, window: int) -> float | None:
    if len(close) < window:
        return None
    return round(float(close.iloc[-1] / close.tail(window).mean()), 4)


def drawdown_from_high(close: pd.Series, window: int = 60) -> float | None:
    if len(close) < 5:
        return None
    c = close.tail(window)
    return round(float(c.iloc[-1] / c.max() - 1), 4)


def volume_ratio(volume: pd.Series, short: int = 5, long: int = 20) -> float | None:
    if len(volume) < long:
        return None
    return round(float(volume.tail(short).mean() / volume.tail(long).mean()), 3)


def relative_strength(asset_close: pd.Series, index_close: pd.Series, window: int = 20) -> float | None:
    a, b = momentum(asset_close, window), momentum(index_close, window)
    if a is None or b is None:
        return None
    return round(a - b, 4)
