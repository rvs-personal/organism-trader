"""Download raw (real) data for the universe into data/raw/<exp>/ and write a fetch report.

Run:  python -m otrader.data.fetch
"""
from __future__ import annotations
import json
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
from ..config import DATA_DIR, load_experiment, load_universe
from .sources import fetch_yfinance, to_yahoo


def fetch_all(exp: dict | None = None, uni: dict | None = None, out_dir: Path | None = None) -> dict:
    exp = exp or load_experiment()
    uni = uni or load_universe()
    out_dir = out_dir or DATA_DIR / "raw" / exp["experiment_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2x lookback in calendar days comfortably covers lookback trading days
    start = date.fromisoformat(exp["start_date"]) - timedelta(days=int(exp["lookback_days"]) * 2)
    end = date.fromisoformat(exp["end_date"]) + timedelta(days=1)

    report: dict = {"ok": [], "failed": {}, "start": str(start), "end": str(end)}
    targets = [(s["symbol"], s["sector"]) for s in uni["stocks"]]
    targets += [(uni["index"], "INDEX"), (uni["vix"], "VIX")]

    frames = {}
    for sym, sector in targets:
        r = fetch_yfinance(to_yahoo(sym), start, end)
        if r.error or r.bars.empty or len(r.bars) < 50:
            report["failed"][sym] = r.error or f"too few rows ({len(r.bars)})"
            continue
        b = r.bars.copy()
        b["symbol"] = sym
        b["sector"] = sector
        frames[sym] = b
        if r.actions is not None and not r.actions.empty:
            safe = sym.replace("^", "IDX_").replace("&", "AND")
            r.actions.to_csv(out_dir / f"actions_{safe}.csv")
        report["ok"].append(sym)

    if frames:
        allbars = pd.concat(frames.values()).reset_index()
        allbars.to_parquet(out_dir / "bars.parquet", index=False)
    (out_dir / "fetch_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    rep = fetch_all()
    print(json.dumps(rep, indent=2))
