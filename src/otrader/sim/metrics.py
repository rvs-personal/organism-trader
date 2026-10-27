"""Performance metrics over an equity curve + fills."""
from __future__ import annotations
import numpy as np
import pandas as pd


def summarize(equity: pd.Series, fills: list[dict], starting_cash: float, bench: pd.Series | None = None) -> dict:
    eq = equity.dropna()
    if len(eq) == 0:
        return {}
    with_start = pd.Series([starting_cash, *eq.tolist()])
    r = with_start.pct_change().dropna()
    dd = with_start / with_start.cummax() - 1
    total_ret = float(eq.iloc[-1] / starting_cash - 1)
    out = {
        "final_equity": round(float(eq.iloc[-1]), 2),
        "total_return": round(total_ret, 4),
        "max_drawdown": round(float(dd.min()), 4),
        "volatility_ann": round(float(r.std() * np.sqrt(252)), 4) if len(r) > 2 else None,
        "sharpe_ann": round(float(r.mean() / r.std() * np.sqrt(252)), 3) if len(r) > 2 and r.std() > 0 else None,
        "n_days": int(len(eq)),
    }
    if fills:
        f = pd.DataFrame(fills)
        sells = f[f["side"] == "SELL"]
        out.update({
            "n_fills": int(len(f)),
            "n_round_trips": int(len(sells)),
            "turnover": round(float(f["gross"].sum() / starting_cash), 3),
            "total_costs": round(float(f["costs"].sum()), 2),
            "win_rate": round(float((sells["realized_pnl"] > 0).mean()), 3) if len(sells) else None,
            "avg_win": round(float(sells.loc[sells["realized_pnl"] > 0, "realized_pnl"].mean()), 2) if (sells["realized_pnl"] > 0).any() else None,
            "avg_loss": round(float(sells.loc[sells["realized_pnl"] <= 0, "realized_pnl"].mean()), 2) if (sells["realized_pnl"] <= 0).any() else None,
        })
    if bench is not None and len(bench) > 0:
        b = bench.reindex(eq.index).ffill()
        out["benchmark_return"] = round(float(b.iloc[-1] / starting_cash - 1), 4)
        out["excess_return"] = round(total_ret - out["benchmark_return"], 4)
    return out
