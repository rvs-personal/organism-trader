"""Non-LLM baselines. Each returns a list of Orders for today given the store + portfolio.
Same execution engine, same costs, same risk rules as the agents."""
from __future__ import annotations
from ..data.store import ClockedStore
from ..data import features as F
from ..sim.portfolio import Portfolio
from ..sim.execution import Order


def cash(store: ClockedStore, pf: Portfolio, marks: dict, day: int) -> list[Order]:
    return []


def index_buy_hold(store, pf, marks, day) -> list[Order]:
    """Equal-weight all assets on day 1 as a proxy for index buy-and-hold (MKT itself isn't tradable)."""
    if day == 1 and not pf.positions:
        per = pf.cash * 0.99 / len(store.assets())
        return [Order(a, "BUY", None, per, "index proxy") for a in store.assets()]
    return []


def equal_weight_rebalanced(store, pf, marks, day) -> list[Order]:
    """Rebalance to equal weight every 20 days (cost-aware: only if drift > 3% of target allocation)."""
    assets = store.assets()
    if day != 1 and day % 20 != 0:
        return []
    if not assets:
        return []
    eq = pf.equity(marks); target = eq * 0.98 / len(assets)
    orders = []
    for a in assets:
        cur = pf.positions[a].qty * marks[a] if a in pf.positions else 0.0
        diff = target - cur
        if abs(diff) > target * 0.03:
            orders.append(Order(a, "BUY" if diff > 0 else "SELL",
                                None if diff > 0 else int(abs(diff) // marks[a]), diff if diff > 0 else None, "rebalance"))
    return sorted(orders, key=lambda o: o.side != "SELL")


def momentum_20d(store, pf, marks, day) -> list[Order]:
    """Weekly: hold top-5 by 20d return, equal weight."""
    if day % 5 != 1:
        return []
    scores = {a: F.momentum(store.close_series(a, 30), 20) for a in store.assets()}
    ranked = sorted((a for a in scores if scores[a] is not None), key=lambda a: scores[a], reverse=True)[:5]
    orders = [Order(a, "SELL", None, None, "momentum exit") for a in list(pf.positions) if a not in ranked]
    eq = pf.equity(marks)
    for a in ranked:
        if a not in pf.positions:
            orders.append(Order(a, "BUY", None, eq * 0.18, "momentum entry"))
    return orders


def reversal_5d(store, pf, marks, day) -> list[Order]:
    """Weekly: buy bottom-5 by 5d return, hold one week."""
    if day % 5 != 1:
        return []
    scores = {a: F.momentum(store.close_series(a, 10), 5) for a in store.assets()}
    ranked = sorted((a for a in scores if scores[a] is not None), key=lambda a: scores[a])[:5]
    orders = [Order(a, "SELL", None, None, "reversal exit") for a in list(pf.positions)]
    eq = pf.equity(marks)
    orders += [Order(a, "BUY", None, eq * 0.18, "reversal entry") for a in ranked]
    return orders


BASELINES = {
    "B_CASH": cash,
    "B_INDEX": index_buy_hold,
    "B_EQUAL": equal_weight_rebalanced,
    "B_MOM20": momentum_20d,
    "B_REV5": reversal_5d,
}
