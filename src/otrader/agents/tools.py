"""Perception tools. Each is a plain Python function over the ClockedStore, so look-ahead is
impossible by construction. `TOOL_SPECS` is the provider-neutral JSON-schema description; both
the Cursor backend (custom_tools) and the OpenAI-compatible backend consume it.
"""
from __future__ import annotations
from typing import Any, Callable
import numpy as np
from ..data.store import ClockedStore
from ..data import features as F
from ..sim.portfolio import Portfolio


class ToolBox:
    def __init__(self, store: ClockedStore, portfolio: Portfolio, memory_search: Callable[[str], list[dict]],
                 marks: dict[str, float], preopen: bool = False):
        self.preopen = preopen
        self.store = store
        self.pf = portfolio
        self.memory_search = memory_search
        self.marks = marks
        self.trace: list[dict[str, Any]] = []

    # ---- helpers ------------------------------------------------------------------------
    def _record(self, name: str, args: dict, result: Any) -> Any:
        self.trace.append({"tool": name, "args": args, "result_size": len(str(result))})
        return result

    def _asset_ok(self, asset: str) -> bool:
        return asset in self.store.assets() or asset in ("MKT", "VOLX")

    # ---- tools ----------------------------------------------------------------------------
    def inspect_market(self, args: dict) -> dict:
        idx = self.store.close_series("MKT", 60)
        vix = self.store.close_series("VOLX", 20)
        sectors = self.store.sectors()
        table = []
        for a in self.store.assets():
            c = self.store.close_series(a, 25)
            if len(c) < 6:
                continue
            table.append({"asset": a, "sector": sectors.get(a), "ret_1d": F.momentum(c, 1),
                          "ret_5d": F.momentum(c, 5), "ret_20d": F.momentum(c, 20)})
        table.sort(key=lambda r: (r["ret_5d"] or 0), reverse=True)
        out = {
            "sim_day": self.store.clock.today,
            "index": {"ret_1d": F.momentum(idx, 1), "ret_5d": F.momentum(idx, 5), "ret_20d": F.momentum(idx, 20),
                      "vol_20d": F.realized_vol(idx, 20), "drawdown_60d": F.drawdown_from_high(idx, 60)},
            "vol_index_level": round(float(vix.iloc[-1]), 2) if len(vix) else None,
            "vol_index_change_5d": F.momentum(vix, 5) if len(vix) > 5 else None,
            "top_5d": table[:5], "bottom_5d": table[-5:],
            "n_assets": len(table),
        }
        return self._record("inspect_market", args, out)

    def inspect_stock(self, args: dict) -> dict:
        a = args["asset"]
        if not self._asset_ok(a):
            return self._record("inspect_stock", args, {"error": f"unknown asset {a}"})
        h = self.store.history(a, 90)
        c, v = h["close"], h["volume"]
        idx = self.store.close_series("MKT", 90)
        out = {
            "asset": a, "sector": self.store.sectors().get(a),
            "last_close": round(float(c.iloc[-1]), 2), "today_open": None if self.preopen else self.store.today_open(a),
            "ret_1d": F.momentum(c, 1), "ret_5d": F.momentum(c, 5), "ret_20d": F.momentum(c, 20), "ret_60d": F.momentum(c, 60),
            "vol_20d_ann": F.realized_vol(c, 20), "rsi_14": F.rsi(c, 14),
            "px_over_sma20": F.sma_ratio(c, 20), "px_over_sma50": F.sma_ratio(c, 50),
            "drawdown_60d": F.drawdown_from_high(c, 60), "volume_ratio_5_20": F.volume_ratio(v),
            "rel_strength_20d_vs_index": F.relative_strength(c, idx, 20),
        }
        return self._record("inspect_stock", args, out)

    def screen_universe(self):
        """Compact deterministic scan, using only decision-time-visible data."""
        rows = []
        keys = ['asset','sector','ret_5d','ret_20d','rsi_14','px_over_sma20','px_over_sma50',
                'volume_ratio_5_20','rel_strength_20d_vs_index','vol_20d_ann']
        for asset in self.store.assets():
            full = self.inspect_stock({'asset': asset})
            rows.append({key: full.get(key) for key in keys})
        vix = self.store.history('VOLX', 60)
        closes = self.store.close_series('MKT', 30)
        market = self.inspect_market({})
        market['previous_decision_mkt_ret_5d'] = F.momentum(closes.iloc[:-1], 5)
        market['volx_last_observed_day'] = int(vix.sim_day.iloc[-1]) if len(vix) else None
        market['volx_age_in_sessions'] = self.store.clock.today - 1 - int(vix.sim_day.iloc[-1]) if len(vix) else None
        market['volx_trailing_percentile'] = round(float((vix.close <= vix.close.iloc[-1]).mean()*100), 1) if len(vix) else None
        market['volx_percentile_observations'] = len(vix)
        market['volx_5_observation_span_sessions'] = int(vix.sim_day.iloc[-1] - vix.sim_day.iloc[-6]) if len(vix)>5 else None
        return {'as_of_day': self.store.clock.today-1, 'market':market, 'candidates':rows,
                'note':'Supplied by deterministic screen; no future closes. VOLX changes span available observations, not necessarily consecutive market sessions. Do not treat stale evidence as fresh.'}

    def get_history(self, args: dict) -> dict:
        a, lb = args["asset"], int(args.get("lookback", 20))
        if not self._asset_ok(a):
            return self._record("get_history", args, {"error": f"unknown asset {a}"})
        h = self.store.history(a, min(lb, 120))
        rows = [{"day": int(r.sim_day), "o": r.open, "h": r.high, "l": r.low, "c": r.close, "v": int(r.volume)}
                for r in h.itertuples()]
        return self._record("get_history", args, {"asset": a, "bars": rows})

    def get_volume(self, args: dict) -> dict:
        a = args["asset"]
        if not self._asset_ok(a):
            return self._record("get_volume", args, {"error": f"unknown asset {a}"})
        h = self.store.history(a, 40)
        v = h["volume"]
        out = {"asset": a, "avg_vol_20d": int(v.tail(20).mean()) if len(v) else None,
               "last_5_vol": [int(x) for x in v.tail(5)], "volume_ratio_5_20": F.volume_ratio(v)}
        return self._record("get_volume", args, out)

    def compare_to_index(self, args: dict) -> dict:
        a = args["asset"]
        if not self._asset_ok(a):
            return self._record("compare_to_index", args, {"error": f"unknown asset {a}"})
        c = self.store.close_series(a, 90); idx = self.store.close_series("MKT", 90)
        out = {"asset": a}
        for w in (5, 20, 60):
            out[f"rel_{w}d"] = F.relative_strength(c, idx, w)
        r_a, r_i = F.returns(c).tail(60), F.returns(idx).tail(60)
        n = min(len(r_a), len(r_i))
        if n > 10:
            beta = float(np.cov(r_a.tail(n), r_i.tail(n))[0, 1] / np.var(r_i.tail(n)))
            out["beta_60d"] = round(beta, 3)
        return self._record("compare_to_index", args, out)

    def get_volatility(self, args: dict) -> dict:
        a = args["asset"]
        if not self._asset_ok(a):
            return self._record("get_volatility", args, {"error": f"unknown asset {a}"})
        c = self.store.close_series(a, 90)
        out = {"asset": a, "vol_10d_ann": F.realized_vol(c, 10), "vol_20d_ann": F.realized_vol(c, 20),
               "vol_60d_ann": F.realized_vol(c, 60)}
        return self._record("get_volatility", args, out)

    def inspect_portfolio(self, args: dict) -> dict:
        from ..governance import portfolio_feedback
        snap = portfolio_feedback(self.pf, self.marks) if self.preopen else self.pf.snapshot(self.marks)
        for p in snap["positions"]:
            pos = self.pf.positions[p["asset"]]
            p["held_days"] = self.store.clock.today - pos.opened_day
        recent = [f for f in self.pf.fills if f.sim_day >= self.store.clock.today - 10]
        snap["recent_fills"] = [{"day": f.sim_day, "asset": f.asset, "side": f.side, "qty": f.qty,
                                 "price": f.price, "realized_pnl": f.realized_pnl} for f in recent]
        if self.preopen:
            for fill in snap["recent_fills"]:
                fill["legacy_sale_side_pnl_excludes_entry_fees"] = fill.pop("realized_pnl")
        return self._record("inspect_portfolio", args, snap)

    def search_memory(self, args: dict) -> dict:
        q = args.get("query", "")
        return self._record("search_memory", args, {"query": q, "results": self.memory_search(q)})

    def get_company_events(self, args: dict) -> dict:
        # v1: no event feed. Returning an honest empty answer keeps the tool surface stable for v2.
        return self._record("get_company_events", args, {"asset": args.get("asset"), "events": [],
                                                          "note": "no event data available in this environment"})

    def registry(self) -> dict[str, Callable[[dict], dict]]:
        return {n: getattr(self, n) for n in TOOL_SPECS}

    def inspect_stocks(self, args: dict) -> dict:
        """Batch the same individual calculation; never mix portfolios or memories."""
        assets = args.get('assets')
        if not isinstance(assets, list) or not 1 <= len(assets) <= 31 or not all(isinstance(a, str) for a in assets):
            raise ValueError('assets must contain 1 to 31 asset IDs')
        rows = [self.inspect_stock({'asset': asset}) for asset in dict.fromkeys(assets)]
        columns = list(rows[0])
        if all(set(row) == set(columns) for row in rows):
            return {'columns': columns, 'rows': [[row[k] for k in columns] for row in rows],
                    'format': 'Values follow columns order. Same calculations as inspect_stock.'}
        return {'results': rows}


def _s(desc: str, props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"description": desc,
            "input_schema": {"type": "object", "properties": props or {}, "required": required or []}}


TOOL_SPECS: dict[str, dict] = {
    "inspect_market": _s("Market overview: index momentum/vol/drawdown, volatility index, top and bottom movers over 5 days."),
    "inspect_stock": _s("Key stats for one asset: returns, vol, RSI, SMA ratios, drawdown, volume ratio, relative strength.",
                        {"asset": {"type": "string"}}, ["asset"]),
    "get_history": _s("Raw daily OHLCV bars for an asset.",
                      {"asset": {"type": "string"}, "lookback": {"type": "integer", "description": "bars, max 120"}}, ["asset"]),
    "get_volume": _s("Volume profile for an asset.", {"asset": {"type": "string"}}, ["asset"]),
    "compare_to_index": _s("Relative strength vs the market index and 60-day beta.", {"asset": {"type": "string"}}, ["asset"]),
    "get_volatility": _s("Realized volatility at 10/20/60 days.", {"asset": {"type": "string"}}, ["asset"]),
    "inspect_portfolio": _s("Your cash, positions, weights, unrealized P&L, held days, recent fills."),
    "search_memory": _s("Search your own episodic and long-term memory.", {"query": {"type": "string"}}, ["query"]),
    "get_company_events": _s("Corporate events for an asset (results, dividends, splits). May be empty.",
                             {"asset": {"type": "string"}}, ["asset"]),
}
