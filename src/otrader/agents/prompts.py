"""Prompt builders. The mind (strategy/beliefs/memory) is injected from the snapshot each day.
The physics (rules, objective, tool list) is fixed text the agent cannot edit."""
from __future__ import annotations
import json
from .snapshot import AgentSnapshot
from .context_format import dumps as compact_dumps


OBJECTIVE = (
    "Objective (immutable): grow simulated equity over the experiment while avoiding a 30% drawdown from "
    "starting capital, which is death. You are judged on return, drawdown, turnover, costs and calibration, "
    "not on activity. Doing nothing is a valid decision."
)

WORLD = (
    "World: a simulated equity market with ~30 assets named STK_xx, a market index MKT and a volatility index VOLX. "
    "Assets and dates are anonymized on purpose; do not try to guess real companies or periods. "
    "Daily bars only. You decide once per day before the open; orders fill at today's open with slippage and "
    "realistic transaction costs. Positions are typically held 3-10 days. Long-only."
)


def system_prompt(snap: AgentSnapshot, risk_text: str) -> str:
    return "\n\n".join([
        f"You are trading organism {snap.agent_id}. Persona: {snap.persona}",
        OBJECTIVE, WORLD,
        f"Risk rules enforced by the engine (you cannot change them): {risk_text}",
        "Research procedure: use the tools to look at what YOU think matters. You have a budget of about 8 tool "
        "calls. Then reply with ONLY a JSON object matching the Decision schema — no prose outside the JSON.",
    ])


def decision_user_prompt(snap: AgentSnapshot, sim_day: int, yesterday_exec: list[dict] | None,
                         decision_schema: dict, mind_override: dict | None = None, compact: bool = False) -> str:
    dump = compact_dumps if compact else json.dumps
    parts = [f"Simulation day {sim_day}.",
             "YOUR MIND (this is you; it persists across days):",
             dump(mind_override if mind_override is not None else snap.mind_view())]
    if yesterday_exec:
        parts += ["Yesterday's order results (rejections are information):", dump(yesterday_exec)]
    parts += ["Decision JSON schema:", dump(decision_schema),
              "Research, then output the Decision JSON."]
    return "\n\n".join(parts)


def episodic_prompt(snap: AgentSnapshot, sim_day: int, decision: dict, exec_results: list[dict],
                    pf_snapshot: dict, day_pnl: float, schema: dict, compact: bool = False) -> str:
    dump = compact_dumps if compact else json.dumps
    return "\n\n".join([
        f"Day {sim_day} is over for organism {snap.agent_id}. Write ONE episodic memory entry as JSON.",
        "Your decision today:", dump(decision),
        "Execution results:", dump(exec_results),
        f"End-of-day portfolio (day P&L {day_pnl:+.0f}):", dump(pf_snapshot),
        "Be concrete and short. A lesson is optional; do not invent one.",
        "Schema:", dump(schema), "Output ONLY the JSON."])


def reflection_prompt(snap: AgentSnapshot, sim_day: int, kind: str, perf: dict, episodic: list[dict],
                      calibration: list[dict], schema: dict, strategy_locked: bool, compact: bool = False) -> str:
    dump = compact_dumps if compact else json.dumps
    lock = ("STRATEGY IS LOCKED for the out-of-sample phase: set strategy_change to null. You may still update "
            "beliefs, hypotheses and perception_policy." if strategy_locked else
            "You MAY rewrite your strategy (full text) if the evidence warrants; otherwise set it to null.")
    return "\n\n".join([
        f"{kind.upper()} for organism {snap.agent_id} on day {sim_day}.",
        "Current mind:", dump(snap.mind_view(recent_episodic=0)),
        "Performance since last reflection:", dump(perf),
        "Episodic memory since last reflection:", dump(episodic),
        "Confidence calibration samples (stated confidence vs 5-day P&L after the decision):", dump(calibration),
        lock,
        "Update beliefs only with evidence from your own record. Prefer fewer, sharper beliefs.",
        "Schema:", dump(schema), "Output ONLY the JSON."])
