"""Portfolio accounting. Long-only, cash + positions, marked at last visible close (or today's open)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Position:
    asset: str
    qty: int
    avg_cost: float
    opened_day: int


@dataclass
class Fill:
    sim_day: int
    asset: str
    side: str          # BUY / SELL
    qty: int
    price: float
    gross: float
    costs: float
    realized_pnl: float
    reason: str


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    total_costs: float = 0.0
    realized_pnl: float = 0.0

    def market_value(self, prices: dict[str, float]) -> float:
        return sum(p.qty * prices.get(a, p.avg_cost) for a, p in self.positions.items())

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def gross_exposure_frac(self, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        return self.market_value(prices) / eq if eq > 0 else 0.0

    def apply_fill(self, f: Fill) -> None:
        if f.side == "BUY":
            self.cash -= f.gross + f.costs
            pos = self.positions.get(f.asset)
            if pos:
                new_qty = pos.qty + f.qty
                pos.avg_cost = (pos.avg_cost * pos.qty + f.price * f.qty) / new_qty
                pos.qty = new_qty
            else:
                self.positions[f.asset] = Position(f.asset, f.qty, f.price, f.sim_day)
        else:
            pos = self.positions[f.asset]
            self.cash += f.gross - f.costs
            pos.qty -= f.qty
            if pos.qty == 0:
                del self.positions[f.asset]
        self.total_costs += f.costs
        self.realized_pnl += f.realized_pnl
        self.fills.append(f)

    def snapshot(self, prices: dict[str, float]) -> dict[str, Any]:
        eq = self.equity(prices)
        return {
            "cash": round(self.cash, 2),
            "equity": round(eq, 2),
            "gross_exposure_frac": round(self.gross_exposure_frac(prices), 4),
            "positions": [
                {"asset": a, "qty": p.qty, "avg_cost": round(p.avg_cost, 2),
                 "last": round(prices.get(a, p.avg_cost), 2),
                 "unrealized_pnl": round((prices.get(a, p.avg_cost) - p.avg_cost) * p.qty, 2),
                 "weight": round(p.qty * prices.get(a, p.avg_cost) / eq, 4) if eq else 0.0,
                 "held_days": None}
                for a, p in sorted(self.positions.items())
            ],
            "total_costs": round(self.total_costs, 2),
            "realized_pnl": round(self.realized_pnl, 2),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"cash": self.cash, "positions": {a: asdict(p) for a, p in self.positions.items()},
                "fills": [asdict(f) for f in self.fills],
                "total_costs": self.total_costs, "realized_pnl": self.realized_pnl}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Portfolio":
        p = cls(cash=d["cash"], total_costs=d.get("total_costs", 0.0), realized_pnl=d.get("realized_pnl", 0.0))
        p.positions = {a: Position(**v) for a, v in d.get("positions", {}).items()}
        p.fills = [Fill(**f) for f in d.get("fills", [])]
        return p
