"""Order validation + fills. Orders decided on day D fill at D's open with slippage and Indian
delivery transaction costs. Rejections are returned to the agent as feedback (they are information)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .portfolio import Portfolio, Fill
from .risk import RiskRules


@dataclass
class Order:
    asset: str
    side: str            # BUY / SELL
    qty: int | None      # explicit shares, or
    notional: float | None  # INR notional (engine converts to shares at open)
    reason: str = ""


@dataclass
class CostModel:
    brokerage_per_order: float
    stt_delivery_frac: float
    exchange_txn_frac: float
    gst_on_brokerage_frac: float
    stamp_duty_buy_frac: float
    slippage_frac: float

    def fill_price(self, open_px: float, side: str) -> float:
        return open_px * (1 + self.slippage_frac) if side == "BUY" else open_px * (1 - self.slippage_frac)

    def costs(self, gross: float, side: str) -> float:
        c = self.brokerage_per_order * (1 + self.gst_on_brokerage_frac)
        c += gross * (self.stt_delivery_frac + self.exchange_txn_frac)
        if side == "BUY":
            c += gross * self.stamp_duty_buy_frac
        return round(c, 2)


def execute_orders(orders: list[Order], pf: Portfolio, opens: dict[str, float], marks: dict[str, float],
                   rules: RiskRules, costs: CostModel, sim_day: int) -> list[dict[str, Any]]:
    """Apply orders in sequence. Returns per-order results (filled/rejected + reason)."""
    results: list[dict[str, Any]] = []
    orders = orders[: rules.max_orders_per_day]
    for o in orders:
        res: dict[str, Any] = {"asset": o.asset, "side": o.side, "status": "rejected", "detail": ""}
        px_open = opens.get(o.asset)
        if px_open is None:
            res["detail"] = "no open price today (asset unknown or not trading)"
            results.append(res); continue
        if o.side not in {"BUY", "SELL"} or (o.qty is not None and o.qty <= 0):
            res["detail"] = "invalid side or non-positive quantity"
            results.append(res); continue
        px = round(costs.fill_price(px_open, o.side), 2)
        equity = pf.equity(marks)

        if o.side == "SELL":
            pos = pf.positions.get(o.asset)
            if not pos:
                res["detail"] = "no position to sell (shorting disabled)" if rules.no_shorting else "no position"
                results.append(res); continue
            qty = min(o.qty if o.qty is not None else pos.qty, pos.qty)
            gross = qty * px
            c = costs.costs(gross, "SELL")
            realized = (px - pos.avg_cost) * qty - c
            pf.apply_fill(Fill(sim_day, o.asset, "SELL", qty, round(px, 2), round(gross, 2), c, round(realized, 2), o.reason))
            res.update(status="filled", qty=qty, price=round(px, 2), costs=c, realized_pnl=round(realized, 2))
            results.append(res); continue

        # BUY
        qty = o.qty if o.qty else int((o.notional or 0) // px)
        if qty <= 0:
            res["detail"] = "quantity resolves to 0 shares"; results.append(res); continue
        gross = qty * px
        c = costs.costs(gross, "BUY")
        if gross + c > pf.cash:
            qty = int((pf.cash - costs.brokerage_per_order * 2) // (px * (1 + costs.stt_delivery_frac + costs.stamp_duty_buy_frac + 0.001)))
            if qty <= 0:
                res["detail"] = "insufficient cash"; results.append(res); continue
            gross = qty * px; c = costs.costs(gross, "BUY")
            res["detail"] = "qty reduced to available cash; "
        existing = pf.positions.get(o.asset)
        if not existing and len(pf.positions) >= rules.max_positions:
            res["detail"] += f"max positions ({rules.max_positions}) reached"; results.append(res); continue
        new_val = gross + (existing.qty * marks.get(o.asset, px) if existing else 0)
        if new_val / equity > rules.max_position_frac + 1e-9:
            cap = equity * rules.max_position_frac - (existing.qty * marks.get(o.asset, px) if existing else 0)
            qty = int(cap // px)
            if qty <= 0:
                res["detail"] += f"position cap {rules.max_position_frac:.0%} reached"; results.append(res); continue
            gross = qty * px; c = costs.costs(gross, "BUY")
            res["detail"] += "qty reduced to position cap; "
        if (pf.market_value(marks) + gross) / equity > rules.max_gross_exposure_frac + 1e-9:
            res["detail"] += "gross exposure cap reached"; results.append(res); continue
        pf.apply_fill(Fill(sim_day, o.asset, "BUY", qty, round(px, 2), round(gross, 2), c, 0.0, o.reason))
        res.update(status="filled", qty=qty, price=round(px, 2), costs=c)
        results.append(res)
    return results
