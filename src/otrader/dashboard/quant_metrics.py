"""Observer-only, strictly aligned performance statistics. No model calls."""
from __future__ import annotations
import numpy as np
import pandas as pd

SESSIONS = 252
MIN_RISK_OBSERVATIONS = 20


def aligned_values(equity, owners, first, last, starting_cash):
    """Include the opening mark; reject incomplete paths rather than filling gaps."""
    if first < 1 or last < first or starting_cash <= 0:
        raise ValueError('Invalid comparison window or initial capital')
    index = pd.Index(range(first - 1, last + 1), name='sim_day')
    paths, excluded = {}, {}
    for owner in dict.fromkeys(owners):
        g = equity[(equity.owner == owner) & equity.sim_day.between(first - 1, last)]
        if g.sim_day.duplicated().any():
            excluded[owner] = 'Duplicate daily marks'; continue
        series = g.set_index('sim_day').equity.astype(float).reindex(index)
        if first == 1 and pd.isna(series.loc[0]):
            series.loc[0] = starting_cash
        if not np.isfinite(series).all() or (series <= 0).any():
            excluded[owner] = 'Missing or nonpositive daily equity'; continue
        paths[owner] = series
    return pd.DataFrame(paths, index=index), excluded


def performance_metrics(values, benchmark, annual_rf=0.0):
    """Ratios use daily arithmetic returns; RF is an annual effective assumption."""
    if benchmark not in values or annual_rf <= -1:
        raise ValueError('Missing benchmark or invalid risk-free rate')
    returns = values.pct_change(fill_method=None).iloc[1:]
    daily_rf = (1 + annual_rf) ** (1 / SESSIONS) - 1
    bench = returns[benchmark]
    rows = []
    for owner in values:
        r = returns[owner]
        excess = r - daily_rf
        active = r - bench
        sd = r.std(ddof=1)
        downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
        enough = len(r) >= MIN_RISK_OBSERVATIONS
        def ratio(numerator, denominator):
            return float(numerator / denominator) if enough and np.isfinite(denominator) and denominator > 1e-12 else np.nan
        rows.append(dict(owner=owner, observations=len(r),
            total_return=values[owner].iloc[-1] / values[owner].iloc[0] - 1,
            max_drawdown=(values[owner] / values[owner].cummax() - 1).min(),
            volatility=sd * np.sqrt(SESSIONS) if enough else np.nan,
            sharpe=ratio(excess.mean() * np.sqrt(SESSIONS), sd),
            sortino=ratio(excess.mean() * np.sqrt(SESSIONS), downside),
            information_ratio=ratio(active.mean() * np.sqrt(SESSIONS), active.std(ddof=1)),
            beta=ratio(r.cov(bench), bench.var(ddof=1)) if enough else np.nan))
    result = pd.DataFrame(rows).set_index('owner')
    result['advantage'] = result.total_return - result.loc[benchmark, 'total_return']
    return result


def trading_metrics(equity, fills, owners, first, last):
    """Fees are explanatory: they have already been deducted from saved equity."""
    rows = []
    for owner in owners:
        g = equity[(equity.owner == owner) & equity.sim_day.between(first, last)]
        f = fills[(fills.owner == owner) & fills.sim_day.between(first, last)]
        rows.append(dict(owner=owner,
            invested=((g.equity-g.cash)/g.equity).mean() if len(g) else np.nan,
            fees=f.costs.sum() if len(g) else np.nan,
            turnover=f.gross.sum()/g.equity.mean() if len(g) else np.nan))
    return pd.DataFrame(rows).set_index('owner')
