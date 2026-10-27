"""Market data sources behind one interface.

Primary: yfinance (adjusted daily OHLCV + corporate actions).
Secondary: jugaad-data (NSE official, unadjusted) — for cross-checks.
Everything returns a DataFrame indexed by date with columns open, high, low, close, volume.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
import pandas as pd


@dataclass
class FetchResult:
    symbol: str
    bars: pd.DataFrame            # date-indexed OHLCV
    actions: pd.DataFrame | None  # dividends/splits if available
    source: str
    error: str | None = None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    return df.dropna(subset=["close"]).sort_index()


def fetch_yfinance(yahoo_symbol: str, start: date, end: date) -> FetchResult:
    try:
        import yfinance as yf
        t = yf.Ticker(yahoo_symbol)
        bars = t.history(start=start.isoformat(), end=end.isoformat(), interval="1d", auto_adjust=True)
        if bars is None or bars.empty:
            return FetchResult(yahoo_symbol, pd.DataFrame(), None, "yfinance", "empty")
        actions = None
        try:
            actions = t.actions
        except Exception:
            pass
        return FetchResult(yahoo_symbol, _normalize(bars), actions, "yfinance")
    except Exception as e:  # network, symbol change, etc.
        return FetchResult(yahoo_symbol, pd.DataFrame(), None, "yfinance", repr(e))


def fetch_jugaad(nse_symbol: str, start: date, end: date) -> FetchResult:
    try:
        from jugaad_data.nse import stock_df
        df = stock_df(symbol=nse_symbol, from_date=start, to_date=end, series="EQ")
        df = df.rename(columns={"DATE": "date", "OPEN": "open", "HIGH": "high", "LOW": "low",
                                "CLOSE": "close", "VOLUME": "volume"}).set_index("date")
        return FetchResult(nse_symbol, _normalize(df), None, "jugaad")
    except Exception as e:
        return FetchResult(nse_symbol, pd.DataFrame(), None, "jugaad", repr(e))


def to_yahoo(symbol: str) -> str:
    if symbol.startswith("^"):
        return symbol
    return f"{symbol}.NS"
