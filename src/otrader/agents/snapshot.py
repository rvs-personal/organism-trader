"""The organism. Everything the agent IS lives here, outside any LLM conversation, in SQLite.

Mutable by the agent (through structured outputs): strategy, beliefs, hypotheses, memories,
perception_policy, working_memory. Everything else is set by the engine.
"""
from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from ..sim.portfolio import Portfolio


@dataclass
class AgentSnapshot:
    agent_id: str
    persona: str
    model_id: str
    display_name: str = ""
    generation: int = 1
    lineage: list[str] = field(default_factory=list)
    status: str = "ALIVE"                      # ALIVE / DEAD
    born_day: int = 1
    age_days: int = 0
    death_day: int | None = None

    strategy: str = ""
    strategy_version: int = 1
    beliefs: dict[str, str] = field(default_factory=dict)
    hypotheses: list[str] = field(default_factory=list)
    perception_policy: str = ("Start broad: inspect_market, then inspect 3-6 assets that stand out, "
                              "check portfolio, then decide.")
    working_memory: str = ""                   # note_to_self from yesterday
    episodic: list[dict[str, Any]] = field(default_factory=list)   # recent entries (kept short)
    long_term: list[str] = field(default_factory=list)
    calibration: list[dict[str, Any]] = field(default_factory=list)  # {day, confidence, pnl_5d}

    portfolio: dict[str, Any] = field(default_factory=dict)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)  # {day, equity}

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "AgentSnapshot":
        d = json.loads(s)
        return cls(**d)

    # ---- agent-facing view (what goes into the prompt) --------------------------------------
    def mind_view(self, recent_episodic: int = 8) -> dict[str, Any]:
        return {
            "identity": {"id": self.agent_id, "name": self.display_name or self.agent_id, "persona": self.persona, "age_days": self.age_days,
                         "strategy_version": self.strategy_version},
            "strategy": self.strategy,
            "beliefs": self.beliefs,
            "hypotheses": self.hypotheses,
            "perception_policy": self.perception_policy,
            "working_memory": self.working_memory,
            "recent_episodic_memory": self.episodic[-recent_episodic:],
            "long_term_memory": self.long_term[-10:],
        }


# ------------------------------------------------------------------------------------------
# Persistence (SQLite). One DB per experiment run: runs/<exp>/state.sqlite
# ------------------------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
  agent_id TEXT PRIMARY KEY, status TEXT, snapshot_json TEXT, updated_day INTEGER
);
CREATE TABLE IF NOT EXISTS strategy_versions (
  agent_id TEXT, version INTEGER, sim_day INTEGER, strategy TEXT, trigger TEXT,
  PRIMARY KEY (agent_id, version)
);
CREATE TABLE IF NOT EXISTS decisions (
  agent_id TEXT, sim_day INTEGER, decision_json TEXT, exec_json TEXT, tool_trace_json TEXT,
  input_tokens INTEGER, output_tokens INTEGER, cost_inr REAL, wall_s REAL, model_id TEXT,
  PRIMARY KEY (agent_id, sim_day)
);
CREATE TABLE IF NOT EXISTS reflections (
  agent_id TEXT, sim_day INTEGER, kind TEXT, reflection_json TEXT, model_id TEXT,
  input_tokens INTEGER, output_tokens INTEGER, cost_inr REAL,
  PRIMARY KEY (agent_id, sim_day, kind)
);
CREATE TABLE IF NOT EXISTS equity (
  owner TEXT, sim_day INTEGER, equity REAL, cash REAL, exposure REAL, PRIMARY KEY (owner, sim_day)
);
CREATE TABLE IF NOT EXISTS fills (
  owner TEXT, sim_day INTEGER, asset TEXT, side TEXT, qty INTEGER, price REAL, gross REAL,
  costs REAL, realized_pnl REAL, reason TEXT
);
CREATE TABLE IF NOT EXISTS graveyard (
  agent_id TEXT PRIMARY KEY, death_day INTEGER, snapshot_json TEXT
);
CREATE TABLE IF NOT EXISTS research_context (
  agent_id TEXT, sim_day INTEGER, context_json TEXT, PRIMARY KEY(agent_id,sim_day)
);
CREATE TABLE IF NOT EXISTS inference_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT, sim_day INTEGER, phase TEXT, event_json TEXT
);
CREATE TABLE IF NOT EXISTS run_state (
  key TEXT PRIMARY KEY, value TEXT
);
"""


class StateDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self._transaction_depth = 0

    def _commit(self):
        if not self._transaction_depth:
            self.conn.commit()

    @contextmanager
    def transaction(self):
        """Commit each portfolio/snapshot and its ledger together, or roll back all."""
        outer = self._transaction_depth == 0
        if outer:
            self.conn.execute("BEGIN IMMEDIATE")
        self._transaction_depth += 1
        try:
            yield
        except BaseException:
            if outer:
                self.conn.rollback()
            raise
        else:
            if outer:
                self.conn.commit()
        finally:
            self._transaction_depth -= 1

    # agents
    def save_agent(self, snap: AgentSnapshot, day: int) -> None:
        self.conn.execute("INSERT OR REPLACE INTO agents VALUES (?,?,?,?)",
                          (snap.agent_id, snap.status, snap.to_json(), day))
        self._commit()

    def load_agent(self, agent_id: str) -> AgentSnapshot | None:
        row = self.conn.execute("SELECT snapshot_json FROM agents WHERE agent_id=?", (agent_id,)).fetchone()
        return AgentSnapshot.from_json(row[0]) if row else None

    def list_agents(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT agent_id FROM agents ORDER BY agent_id")]

    def save_strategy_version(self, agent_id: str, version: int, day: int, strategy: str, trigger: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO strategy_versions VALUES (?,?,?,?,?)",
                          (agent_id, version, day, strategy, trigger))
        self._commit()

    def save_decision(self, agent_id: str, day: int, decision: dict, exec_results: list, trace: list,
                      usage: dict, wall_s: float, model_id: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (agent_id, day, json.dumps(decision), json.dumps(exec_results), json.dumps(trace),
                           usage.get("input_tokens", 0), usage.get("output_tokens", 0), usage.get("cost_inr", 0.0),
                           wall_s, model_id))
        self._commit()

    def save_reflection(self, agent_id: str, day: int, kind: str, refl: dict, usage: dict, model_id: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO reflections VALUES (?,?,?,?,?,?,?,?)",
                          (agent_id, day, kind, json.dumps(refl), model_id,
                           usage.get("input_tokens", 0), usage.get("output_tokens", 0), usage.get("cost_inr", 0.0)))
        self._commit()

    def save_equity(self, owner: str, day: int, equity: float, cash: float, exposure: float) -> None:
        self.conn.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?,?)", (owner, day, equity, cash, exposure))
        self._commit()

    def save_fills(self, owner: str, fills: list[dict], day: int | None = None) -> None:
        days = {f["sim_day"] for f in fills} | ({day} if day is not None else set())
        self.conn.executemany("DELETE FROM fills WHERE owner=? AND sim_day=?",
                              [(owner, d) for d in days])
        self.conn.executemany("INSERT INTO fills VALUES (?,?,?,?,?,?,?,?,?,?)",
                              [(owner, f["sim_day"], f["asset"], f["side"], f["qty"], f["price"], f["gross"],
                                f["costs"], f["realized_pnl"], f["reason"]) for f in fills])
        self._commit()

    def bury(self, snap: AgentSnapshot) -> None:
        self.conn.execute("INSERT OR REPLACE INTO graveyard VALUES (?,?,?)",
                          (snap.agent_id, snap.death_day, snap.to_json()))
        self._commit()

    def get_state(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM run_state WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_state(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO run_state VALUES (?,?)", (key, value))
        self._commit()
