"""Whole pipeline on synthetic data with the stub backend, including kill/reload."""
import json
from pathlib import Path
from otrader.sim.clock import Clock
from otrader.data.store import ClockedStore
from otrader.agents.snapshot import AgentSnapshot, StateDB
from otrader.agents.loop import run_agent_day, run_reflection
from otrader.sim.risk import RiskRules
from otrader.sim.execution import CostModel
from otrader.inference.stub_backend import StubBackend
from conftest import synthetic_bars

CFG = {"capital": {"starting_cash": 1_000_000, "death_threshold_frac": 0.7},
       "risk": {"max_positions": 8, "max_position_frac": 0.25, "max_gross_exposure_frac": 1.0, "no_shorting": True, "max_orders_per_day": 5},
       "inference": {"daily_model": "stub", "reflection_model": "stub", "max_tool_turns": 8},
       "evolution": {"reflection_every_days": 10, "review_every_days": 50, "lock_strategy_from_day": 161}}
COSTS = CostModel(20, 0.001, 0.0000325, 0.18, 0.00015, 0.0005)


def test_pipeline_persists_and_resumes(tmp_path: Path):
    db = StateDB(tmp_path / "state.sqlite")
    clock = Clock(1)
    store = ClockedStore(bars=synthetic_bars(days=30), clock=clock)
    rules = RiskRules.from_config(CFG["risk"])
    snap = AgentSnapshot("A1", "test", "stub", strategy="s1")
    db.save_agent(snap, 0)
    prev = None
    for _ in range(10):
        snap = db.load_agent("A1")          # reload from disk every day = restart-safe
        prev = run_agent_day(snap, store, db, StubBackend(), CFG, rules, COSTS, prev)
        if clock.today % 10 == 0:
            run_reflection(snap, db, StubBackend(), CFG, clock.today, "reflection")
        clock.advance()
    snap2 = db.load_agent("A1")
    assert snap2.age_days == 10 and len(snap2.equity_curve) == 10 and len(snap2.episodic) == 10
    assert snap2.portfolio["cash"] < 1_000_000  # it traded
    n_dec = db.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    assert n_dec == 10
    assert db.conn.execute("SELECT COUNT(*) FROM reflections").fetchone()[0] == 1
