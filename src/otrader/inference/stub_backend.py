"""No-LLM backend so the whole pipeline runs without any key. Calls a few tools and emits a
schema-valid Decision built from a trivial rule. Use it to test plumbing, dashboard and baselines."""
from __future__ import annotations
import json
import random
from .base import Backend, EpisodeResult, Usage, ToolFn


class StubBackend(Backend):
    name = "stub"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def run_episode(self, system, user, tools: dict[str, ToolFn], tool_specs, model_id="stub", max_tool_turns=8):
        calls = []
        if "Decision JSON schema" in user:
            mkt = tools["inspect_market"]({}); calls.append({"tool": "inspect_market"})
            pf = tools["inspect_portfolio"]({}); calls.append({"tool": "inspect_portfolio"})
            orders, theses = [], {}
            # sell anything held > 5 days, buy the top 5d mover if room
            for p in pf["positions"]:
                if (p.get("held_days") or 0) >= 5:
                    orders.append({"asset": p["asset"], "side": "SELL", "reason": "stub: time exit"})
            if len(pf["positions"]) < 3 and mkt["top_5d"]:
                pick = mkt["top_5d"][0]["asset"]
                tools["inspect_stock"]({"asset": pick}); calls.append({"tool": "inspect_stock"})
                orders.append({"asset": pick, "side": "BUY", "notional_inr": pf["equity"] * 0.15,
                               "reason": "stub: 5d momentum"})
                theses[pick] = "stub momentum"
            out = {"market_view": "stub view", "orders": orders, "theses": theses,
                   "confidence": round(self.rng.uniform(0.3, 0.7), 2), "watchlist": [], "note_to_self": "stub"}
            if 'candidate_assessments' in user:
                out['candidate_assessments'] = []
            if 'preview_allocation' in tools:
                preview = tools['preview_allocation']({'orders': orders})
                calls.append({'tool': 'preview_allocation'})
                if not preview['affordable_in_order']:
                    orders[:] = [order for order in orders if order['side'] == 'SELL']
                    preview = tools['preview_allocation']({'orders': orders})
                    calls.append({'tool': 'preview_allocation'})
                out['candidate_assessments'] = [
                    {'asset': order['asset'], 'disposition': 'act', 'reason': order['reason']}
                    for order in orders[:5]]
                out['allocation_plan'] = {
                    'cash_reserve_inr': preview['estimated_cash_reserve_inr'],
                    'reserve_reason': 'The stub keeps cash after its fixed test orders.',
                    'reserve_release_condition': 'The next stub decision checks the same test rule.',
                    'sizing_rationale': 'Fixed test amounts verify execution. This is not model reasoning.'}
        elif "episodic memory entry" in user:
            out = {"sim_day": 0, "summary": "stub day", "lesson": None, "tags": ["stub"]}
        else:
            out = {"what_worked": [], "what_failed": [], "belief_updates": {}, "beliefs_to_drop": [],
                   "hypotheses": [], "perception_policy": "stub", "strategy_change": None,
                   "confidence_calibration_note": "", "long_term_memory_additions": []}
        return EpisodeResult(text=json.dumps(out), usage=Usage(500, 100, 0, 0.0), tool_calls=calls, model_id="stub")
