"""Clock-enforced access to anonymized bars.

THE LOOK-AHEAD GUARD LIVES HERE, IN CODE. Every read goes through `ClockedStore`, which is
constructed with a `Clock` and refuses to return any row with sim_day > clock.today.
Agents never touch the DataFrame directly; tools call this store.

Intraday convention for v1 (daily bars): during a decision on sim_day D the agent sees bars up to
and including D-1's close, plus D's open. Orders placed on D fill at D's open (see sim/execution).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from ..sim.clock import Clock


class LookAheadError(RuntimeError):
    pass


@dataclass
class ClockedStore:
    bars: pd.DataFrame       # agent-facing anonymized bars
    clock: Clock

    @classmethod
    def load(cls, anon_dir: Path, clock: Clock) -> "ClockedStore":
        bars = pd.read_parquet(anon_dir / "bars.parquet")
        return cls(bars=bars, clock=clock)

    # ---- guards -------------------------------------------------------------------------
    def _assert_visible(self, sim_day: int) -> None:
        if sim_day > self.clock.today:
            raise LookAheadError(f"requested sim_day {sim_day} > today {self.clock.today}")

    def _visible(self) -> pd.DataFrame:
        """Rows fully visible: strictly before today (closes), plus today's open only."""
        past = self.bars[self.bars["sim_day"] < self.clock.today]
        return past

    # ---- public reads --------------------------------------------------------------------
    def assets(self) -> list[str]:
        return sorted(a for a in self.bars.loc[self.bars["sim_day"] <= self.clock.today, "asset"].unique() if a not in ("MKT", "VOLX"))

    def sectors(self) -> dict[str, str]:
        d = self.bars[self.bars["sim_day"] <= self.clock.today].sort_values("sim_day").drop_duplicates("asset", keep="last").set_index("asset")["sector"].to_dict()
        return {a: s for a, s in d.items() if a not in ("MKT", "VOLX")}

    def history(self, asset: str, lookback: int) -> pd.DataFrame:
        """Last `lookback` completed daily bars for asset (excludes today)."""
        v = self._visible()
        h = v[v["asset"] == asset].sort_values("sim_day").tail(int(lookback))
        return h[["sim_day", "weekday", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    def today_open(self, asset: str) -> float | None:
        row = self.bars[(self.bars["asset"] == asset) & (self.bars["sim_day"] == self.clock.today)]
        return float(row["open"].iloc[0]) if len(row) else None

    def last_close(self, asset: str) -> float | None:
        h = self.history(asset, 1)
        return float(h["close"].iloc[-1]) if len(h) else None

    def close_series(self, asset: str, lookback: int) -> pd.Series:
        h = self.history(asset, lookback)
        return h.set_index("sim_day")["close"]

    # ---- accounting-side reads (NOT exposed to agents) ------------------------------------
    def bar_on(self, asset: str, sim_day: int) -> pd.Series | None:
        """Full bar for a day. Guarded: only <= today. Used by execution/marking, not by tools."""
        self._assert_visible(sim_day)
        row = self.bars[(self.bars["asset"] == asset) & (self.bars["sim_day"] == sim_day)]
        return row.iloc[0] if len(row) else None

    def max_day(self) -> int:
        return int(self.bars["sim_day"].max())
