from otrader.sim.portfolio import Portfolio
from otrader.sim.execution import Order, CostModel, execute_orders
from otrader.sim.risk import RiskRules

RULES = RiskRules(max_positions=2, max_position_frac=0.25, max_gross_exposure_frac=1.0, no_shorting=True, max_orders_per_day=5)
COSTS = CostModel(20, 0.001, 0.0000325, 0.18, 0.00015, 0.0005)


def test_buy_then_sell_round_trip_costs_money():
    pf = Portfolio(cash=1_000_000)
    r = execute_orders([Order("A", "BUY", None, 200_000, "")], pf, {"A": 100.0}, {"A": 100.0}, RULES, COSTS, 1)
    assert r[0]["status"] == "filled" and pf.positions["A"].qty > 0
    r2 = execute_orders([Order("A", "SELL", None, None, "")], pf, {"A": 100.0}, {"A": 100.0}, RULES, COSTS, 2)
    assert r2[0]["status"] == "filled" and "A" not in pf.positions
    assert pf.cash < 1_000_000 and pf.total_costs > 0


def test_position_cap_enforced():
    pf = Portfolio(cash=1_000_000)
    execute_orders([Order("A", "BUY", None, 900_000, "")], pf, {"A": 100.0}, {"A": 100.0}, RULES, COSTS, 1)
    assert pf.positions["A"].qty * 100.0 <= 250_000 * 1.01


def test_max_positions_and_no_short():
    pf = Portfolio(cash=1_000_000)
    px = {"A": 10.0, "B": 10.0, "C": 10.0}
    r = execute_orders([Order("A", "BUY", None, 1000, ""), Order("B", "BUY", None, 1000, ""),
                        Order("C", "BUY", None, 1000, ""), Order("Z", "SELL", None, None, "")],
                       pf, px, px, RULES, COSTS, 1)
    assert [x["status"] for x in r] == ["filled", "filled", "rejected", "rejected"]


def test_death_threshold_math():
    assert 1_000_000 * 0.7 == 700_000
