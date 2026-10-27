"""One simulated day for one organism, plus the periodic reflection/review.

Order of a day D:
  1. marks = last closes (D-1); intervention 2 hides D's open until orders commit
  2. LLM episode: research via tools -> Decision JSON (validated; one retry)
  3. execute orders at D open under immutable risk rules
  4. mark portfolio at D close (engine-side; the agent only sees it tomorrow)
  5. LLM: write one episodic memory entry
  6. death check; age++
"""
from __future__ import annotations
import json
import time
import copy
from typing import Any
from pydantic import ValidationError
from ..data.store import ClockedStore
from ..sim.portfolio import Portfolio
from ..sim.execution import Order, CostModel, execute_orders
from ..sim.risk import RiskRules
from ..inference.base import Backend, extract_json, Usage
from ..inference.retry import EpisodeRetriesExhausted, retry_reason
from .snapshot import AgentSnapshot, StateDB
from .schemas import AllocationDecision, InterventionDecision, Decision, EpisodicEntry, Reflection, Review
from .allocation import preview_allocation, PREVIEW_SPEC
from .tools import ToolBox, TOOL_SPECS
from . import prompts as P
from ..intervention import active, instructions, reflection_feedback
from .. import governance as G


def _memory_search(snap: AgentSnapshot):
    def search(q: str) -> list[dict]:
        ql = q.lower().split()
        hits = []
        for e in snap.episodic:
            s = json.dumps(e).lower()
            if any(w in s for w in ql):
                hits.append(e)
        for lt in snap.long_term:
            if any(w in lt.lower() for w in ql):
                hits.append({"long_term": lt})
        return hits[-8:]
    return search


from .context_format import research_json, observer


def _run_structured(backend: Backend, system: str, user: str, tools, specs, model_id: str, schema_cls,
                    max_tool_turns: int, validate=None, observe=None):
    """Run an episode and validate its JSON; retry once with the validation error appended."""
    usage = Usage()
    trace: list[dict] = []
    last_err = None
    recovery_delays = (5, 15)
    recovery_count = 0  # shared across validation repair: at most TWO additional calls
    budget_retried = False
    for attempt in range(2):
        extra = {'output_schema': schema_cls.model_json_schema()} if getattr(backend, 'supports_json_schema', False) else {}
        while True:
            started = time.monotonic()
            if observe:
                observe(dict(event='start', attempt=attempt+1, recovery_attempt=recovery_count,
                             prompt_characters=len(system)+len(user), model=model_id))
            try:
                res = backend.run_episode(system, user, tools, specs, model_id, max_tool_turns, **extra)
                break
            except Exception as error:
                reason = retry_reason(backend, error)
                if observe:
                    observe(dict(event='failed', attempt=attempt+1, seconds=round(time.monotonic()-started,2),
                                 error_type=type(error).__name__, retry_reason=reason))
                if reason is None:
                    raise
                if recovery_count == len(recovery_delays) or (reason == 'tool_budget_exceeded' and budget_retried):
                    if observe:
                        observe(dict(event='episode_retry_exhausted', reason=reason, retries=recovery_count))
                    raise EpisodeRetriesExhausted('Episode recovery exhausted; saved progress preserved') from error
                if reason == 'tool_budget_exceeded':
                    budget_retried = True  # one fresh attempt, same eight-tool limit
                delay = recovery_delays[recovery_count]
                recovery_count += 1
                if observe:
                    observe(dict(event='episode_retry', reason=reason, retry_number=recovery_count,
                                 delay_seconds=delay, scope='same phase and unchanged prompt',
                                 failed_call_usage='unknown'))
                time.sleep(delay)
        if observe:
            observe(dict(event='response', attempt=attempt+1, seconds=round(time.monotonic()-started,2), tool_calls=len(res.tool_calls), **res.usage.as_dict()))
        usage.add(res.usage); trace += res.tool_calls
        try:
            obj = schema_cls.model_validate(extract_json(res.text))
            if validate:
                validate(obj)
            return obj, usage, trace, res.model_id
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            if isinstance(e, ValidationError):
                details = [dict(field='.'.join(str(p) for p in err['loc']),
                                error=err['msg'], type=err['type']) for err in e.errors()]
                error_text = json.dumps(details, ensure_ascii=False)
            else:
                details = [dict(error=str(e)[:1200])]
                error_text = str(e)[:1200]
            if observe:
                observe(dict(event="validation_retry" if attempt == 0 else "validation_exhausted",
                             attempt=attempt+1, error_type=type(e).__name__, errors=details))
            last_err = e
            # Every backend gets the actual failed response. Repeating research from
            # scratch does not reliably repair missing nested fields.
            user = user + "\n\nVALIDATION REPAIR — exact errors:\n" + error_text + (
                "\nPrevious response to correct:\n" + res.text +
                "\nCorrect every listed error and include every required field from the schema. "
                "Preserve valid decisions and evidence; do not start a different plan or repeat completed research. "
                "Do not fabricate missing facts. Return the complete corrected JSON object, not a patch. "
                "Output ONLY valid JSON for the schema.")
    raise RuntimeError(f"structured output failed twice: {last_err}")


def run_agent_day(snap: AgentSnapshot, store: ClockedStore, db: StateDB, backend: Backend, cfg: dict,
                  rules: RiskRules, costs: CostModel, yesterday_exec: list[dict] | None) -> list[dict]:
    day = store.clock.today
    compact = cfg["inference"].get("compact_context", False)
    tracing = cfg["inference"].get("trace_inference", False)
    if snap.equity_curve and snap.equity_curve[-1]["day"] >= day:
        row = db.conn.execute("SELECT exec_json FROM decisions WHERE agent_id=? AND sim_day=?",
                              (snap.agent_id, day)).fetchone()
        return json.loads(row[0]) if row else []
    pf = Portfolio.from_dict(snap.portfolio) if snap.portfolio else Portfolio(cash=float(cfg["capital"]["starting_cash"]))
    assets = store.assets()
    marks = {a: store.last_close(a) for a in assets}
    marks = {a: v for a, v in marks.items() if v is not None}
    opens = {a: store.today_open(a) for a in assets}
    opens = {a: v for a, v in opens.items() if v is not None}

    # --- 2. decide ---------------------------------------------------------------------------
    governed = G.enabled(cfg, day)
    factual_search = lambda _: G.trade_outcomes(pf.fills)[-8:]
    tool_store = ClockedStore(store.bars[store.bars.sim_day < day], store.clock) if governed else store
    tb = ToolBox(tool_store, pf, factual_search if governed else _memory_search(snap), marks, preopen=governed)
    system = P.system_prompt(snap, rules.describe())
    flexible = governed and G.discretionary(cfg)
    schema = AllocationDecision if flexible else (InterventionDecision if active(cfg, day) else Decision)
    if flexible:
        system = system.replace(P.OBJECTIVE, 'Objective: grow net simulated equity and outperform the buy-and-hold stock basket by day500. Evaluate return, drawdown, concentration, costs and evidence quality. Cash and position sizes are daily choices; no fixed allocation percentage is required.')
        system = system.replace('Positions are typically held 3-10 days. ', 'You choose holding horizons and exit criteria. ')
    mind = G.decision_mind(snap, cfg) if governed else None
    if flexible:
        rows = db.conn.execute('SELECT sim_day,reflection_json FROM reflections WHERE agent_id=? AND sim_day<? ORDER BY sim_day DESC LIMIT 2', (snap.agent_id, day)).fetchall()
        mind['recent_reflection_evidence'] = [dict(day=r[0], **{k:v for k,v in json.loads(r[1]).items() if k in ('what_worked','what_failed','hypotheses')}) for r in rows]
    user = P.decision_user_prompt(snap, day, yesterday_exec, schema.model_json_schema(),
                                  mind_override=mind, compact=compact)
    research = None
    if active(cfg, day):
        system += '\n\n' + instructions(cfg, snap.agent_id)
        research = tb.screen_universe()
        if governed:
            research['operational_policy'] = G.policy_context(research, pf, marks, snap.agent_id, cfg)
            system = system.replace('and calibration', 'and evidence accuracy')
            user += '\nPortfolio and net trade outcomes:\n' + json.dumps(G.portfolio_feedback(pf, marks))
        user += '\n\nFull-universe screen (engine supplied):\n' + research_json(research, compact)

    def validate_allocation(dec):
        if not dec.orders and abs(dec.allocation_plan.cash_reserve_inr - pf.cash) > 1:
            raise ValueError(f'Empty orders means no trades: cash stays {pf.cash:.2f} INR. Either put your intended trades into orders, or choose to hold and set planned cash to current cash. Do not claim deployment only in prose.')
        for order in dec.orders:
            if order.side == 'BUY' and (order.notional_inr is None or order.notional_inr <= 0 or order.qty is not None):
                raise ValueError('Each BUY needs a positive notional_inr and qty null; the engine determines shares at the unseen open.')
        planned_cash = preview_allocation([o.model_dump() for o in dec.orders], pf, marks, costs)['estimated_cash_reserve_inr']
        # Allow estimates for fees/whole shares; actual unseen opening fills may differ.
        tolerance = max(100, pf.equity(marks) * .005)
        if planned_cash < -tolerance or abs(planned_cash - dec.allocation_plan.cash_reserve_inr) > tolerance:
            raise ValueError(f'Allocation arithmetic mismatch: current cash {pf.cash:.2f}; your orders imply about {planned_cash:.2f} cash after estimated fees, but claimed reserve is {dec.allocation_plan.cash_reserve_inr:.2f}. Revise your own order amounts or reserve so they reconcile. Sum BUY amounts, subtract from cash plus estimated SELL proceeds. No required investment level.')

    if flexible:
        user += '\nEXECUTION CONTRACT: Put every intended transaction in the required orders array. Sizing prose does not execute trades. If you choose no transactions, orders=[] and cash_reserve_inr must equal current cash. BUY uses positive notional_inr and qty=null. SELL uses qty or null to close. Keep every explanation concise.'
        user += '\nUse preview_allocation to calculate estimated cash after proposed orders. It is a calculator, not an investment rule; you choose the amounts. Your final cash_reserve_inr should agree with its estimate.'
    tools, specs = tb.registry(), dict(TOOL_SPECS)
    if compact:
        tools['inspect_stocks'] = tb.inspect_stocks
        specs['inspect_stocks'] = {'description': 'Inspect several chosen assets in one call; same complete stock metrics as inspect_stock. Column names appear once and rows follow their order.',
            'input_schema': {'type': 'object', 'properties': {'assets': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 31}}, 'required': ['assets']}}
        user += '\nFor multiple independent stock inspections, prefer inspect_stocks with their IDs in one call. Ask for additional evidence whenever needed. Keep justifications concise and avoid copying input tables into your answer.'
    if flexible:
        tools['preview_allocation'] = lambda args: preview_allocation(args['orders'], pf, marks, costs)
        specs['preview_allocation'] = PREVIEW_SPEC
    t0 = time.time()
    decision, usage, trace, model_used = _run_structured(
        backend, system, user, tools, specs, cfg["inference"]["daily_model"], schema,
        int(cfg["inference"]["max_tool_turns"]), validate_allocation if flexible else None,
        observe=observer(db, snap.agent_id, day, "decision", tracing))
    wall = time.time() - t0
    if governed:
        research['policy_audit'] = G.audit_decision(decision, research['operational_policy'])

    # --- 3. execute at open -------------------------------------------------------------------
    orders = [Order(o.asset, o.side, o.qty, o.notional_inr, o.reason) for o in decision.orders]
    if governed:
        context = research['operational_policy']
        exec_results = []
        for order in orders[:rules.max_orders_per_day]:
            blocked = None
            if order.side == 'BUY':
                if order.asset not in context.get('eligible_buys', context['eligible_new']) or (order.asset in pf.positions and not flexible):
                    blocked = 'Intervention 2: not an eligible new position; no averaging down'
                elif order.asset not in pf.positions and len(pf.positions) >= context['max_positions']:
                    blocked = 'Intervention 2: strategy sleeve capacity reached'
                elif order.qty is not None:
                    blocked = 'Intervention 2: BUY must use notional_inr with hidden open'
                elif not flexible:
                    order.notional = min(order.notional or 0, context['target_entry_inr'])
            if blocked:
                exec_results.append(dict(asset=order.asset, side=order.side, status='rejected', detail=blocked))
            else:
                exec_results.extend(execute_orders([order], pf, opens, marks, rules, costs, day))
    else:
        exec_results = execute_orders(orders, pf, opens, marks, rules, costs, day)
    new_fills = [f for f in pf.fills if f.sim_day == day]
    if governed:
        outcomes = iter(x for x in G.trade_outcomes(pf.fills) if x['sell_day'] == day)
        for result in exec_results:
            if result['status'] == 'filled' and result['side'] == 'SELL':
                result['legacy_sale_side_pnl_excludes_entry_fees'] = result.pop('realized_pnl')
                result['net_realized_fifo_pnl_inr'] = next(outcomes)['net_pnl_inr']

    # --- 4. mark at today's close (engine-side) -----------------------------------------------
    closes = {}
    for a in assets:
        bar = store.bar_on(a, day)
        if bar is not None:
            closes[a] = float(bar["close"])
    eq_prev = snap.equity_curve[-1]["equity"] if snap.equity_curve else float(cfg["capital"]["starting_cash"])
    eq_now = pf.equity(closes)
    day_pnl = eq_now - eq_prev
    pf_snap = G.portfolio_feedback(pf, closes) if governed else pf.snapshot(closes)
    if flexible:
        research['allocation_outcome'] = dict(planned_cash_reserve_inr=decision.allocation_plan.cash_reserve_inr,
            actual_cash_inr=round(pf.cash, 2), actual_invested_fraction=pf.gross_exposure_frac(closes),
            note='Cash is chosen by the agent; actual fills and closing prices can differ from the plan.')

    # --- 5. episodic memory -------------------------------------------------------------------
    ep_user = P.episodic_prompt(snap, day, decision.model_dump(), exec_results, pf_snap, day_pnl,
                                EpisodicEntry.model_json_schema(), compact=compact)
    entry, ep_usage, _, _ = _run_structured(backend, system, ep_user, {}, {}, cfg["inference"]["daily_model"],
                                            EpisodicEntry, 0, observe=observer(db, snap.agent_id, day, "memory", tracing))
    entry.sim_day = day
    snap.episodic.append(entry.model_dump())
    snap.episodic = snap.episodic[-60:]           # keep bounded; older lessons migrate to long_term at review
    usage.add(ep_usage)

    # --- 6. update organism -------------------------------------------------------------------
    snap.working_memory = decision.note_to_self
    snap.portfolio = pf.to_dict()
    snap.equity_curve.append({"day": day, "equity": round(eq_now, 2)})
    snap.calibration.append({"day": day, "confidence": decision.confidence, "equity": round(eq_now, 2),
                             **({'metric_kind':'uncalibrated_conviction'} if active(cfg, day) else {})})
    for c in snap.calibration:
        if not active(cfg, day) and c.get("pnl_5d") is None and day - c["day"] >= 5:
            c["pnl_5d"] = round(eq_now - c["equity"], 2)
    snap.age_days += 1
    threshold = float(cfg["capital"]["starting_cash"]) * float(cfg["capital"]["death_threshold_frac"])
    if eq_now < threshold and snap.status == "ALIVE":
        snap.status, snap.death_day = "DEAD", day


    # --- persist ---------------------------------------------------------------------------
    with db.transaction():
        if research is not None:
            db.conn.execute('INSERT OR REPLACE INTO research_context VALUES (?,?,?)', (snap.agent_id,day,json.dumps(research)))
        db.save_decision(snap.agent_id, day, decision.model_dump(), exec_results, trace, usage.as_dict(), wall, model_used)
        db.save_equity(snap.agent_id, day, eq_now, pf.cash, pf.gross_exposure_frac(closes))
        db.save_fills(snap.agent_id, [f.__dict__ for f in new_fills], day)
        db.save_agent(snap, day)
        if snap.status == "DEAD":
            db.bury(snap)
    return exec_results


def run_reflection(snap: AgentSnapshot, db: StateDB, backend: Backend, cfg: dict, day: int, kind: str) -> None:
    """kind in {'reflection', 'review'}. Applies belief/strategy updates; preserves strategy versions."""
    if db.conn.execute("SELECT 1 FROM reflections WHERE agent_id=? AND sim_day=? AND kind=?",
                       (snap.agent_id, day, kind)).fetchone():
        return
    every = cfg["evolution"]["review_every_days" if kind == "review" else "reflection_every_days"]
    since = day - every
    if active(cfg, day):
        since = max(since, cfg['_intervention']['effective_day']-1)
    eps = [e for e in snap.episodic if e["sim_day"] > since]
    curve = [p for p in snap.equity_curve if p["day"] > since]
    perf = {}
    if len(curve) >= 2:
        perf = {"start_equity": curve[0]["equity"], "end_equity": curve[-1]["equity"],
                "return": round(curve[-1]["equity"] / curve[0]["equity"] - 1, 4),
                "worst_day": min(curve, key=lambda p: p["equity"])["equity"]}
    calib = [c for c in snap.calibration if c.get("pnl_5d") is not None and c["day"] > since][-15:]
    locked = day >= int(cfg["evolution"]["lock_strategy_from_day"])
    governed = G.enabled(cfg, day)
    schema_cls = Review if kind == "review" else Reflection
    system = P.system_prompt(snap, "(reflection mode; no trading)")
    if G.discretionary(cfg):
        system = system.replace(P.OBJECTIVE, 'Objective: grow net simulated equity and outperform the buy-and-hold stock basket by day500. Evaluate return, drawdown, concentration, costs and evidence quality. There is no early performance-based termination or required allocation.')
        system = system.replace('Positions are typically held 3-10 days. ', 'You choose holding horizons and exit criteria. ')
    if active(cfg, day):
        perf = reflection_feedback(db, snap.agent_id, since, day, float(cfg['capital']['starting_cash']))
        calib = []
        system += '\n\n' + instructions(cfg, snap.agent_id)
    reflection_snap = snap
    if governed:
        reflection_snap = copy.deepcopy(snap)
        reflection_snap.beliefs = {}; reflection_snap.hypotheses = []; reflection_snap.long_term = []
        reflection_snap.working_memory = ''
        eps = [dict(sim_day=e['sim_day'],summary=e['summary']) for e in eps]
        pf = Portfolio.from_dict(snap.portfolio) if snap.portfolio else Portfolio(float(cfg['capital']['starting_cash']))
        perf['closed_trades_net_of_all_fees'] = [x for x in G.trade_outcomes(pf.fills) if since < x['sell_day'] <= day]
        system = system.replace('and calibration', 'and evidence accuracy')
    user = P.reflection_prompt(reflection_snap, day, kind, perf, eps, calib, schema_cls.model_json_schema(), locked, compact=cfg["inference"].get("compact_context", False))
    if governed:
        user += '\nPROPOSAL-ONLY REVIEW: proposed strategy, beliefs, hypotheses and perception changes are archived for the next independent review, never activated automatically. Evaluate net FIFO outcomes; no permanent rule from untested skips.'
    if active(cfg, day):
        user = user.replace('Confidence calibration samples (stated confidence vs 5-day P&L after the decision):',
                            'Legacy confidence calibration is disabled; this empty list is intentional:')
        user += '\nDo not score confidence against later portfolio P&L. Use confidence_calibration_note only to describe uncertainty. In any belief/strategy revision record evidence count, counter-evidence, falsification and next review day. Prior unsupported beliefs are tentative, not validated.' 
    refl, usage, _, model_used = _run_structured(backend, system, user, {}, {}, cfg["inference"]["reflection_model"],
                                                 schema_cls, 0, observe=observer(db, snap.agent_id, day, kind, cfg["inference"].get("trace_inference", False)))

    with db.transaction():
        if governed:
            db.save_reflection(snap.agent_id, day, kind, refl.model_dump(), usage.as_dict(), model_used)
            db.save_agent(snap, day)
            return
        for k, v in refl.belief_updates.items():
            snap.beliefs[k] = v
        for k in refl.beliefs_to_drop:
            snap.beliefs.pop(k, None)
        snap.hypotheses = refl.hypotheses or snap.hypotheses
        snap.perception_policy = refl.perception_policy or snap.perception_policy
        if isinstance(refl, Review):
            snap.long_term.extend(refl.long_term_memory_additions)
        if refl.strategy_change and not locked:
            snap.strategy_version += 1
            snap.strategy = refl.strategy_change
            db.save_strategy_version(snap.agent_id, snap.strategy_version, day, snap.strategy, kind)
        db.save_reflection(snap.agent_id, day, kind, refl.model_dump(), usage.as_dict(), model_used)
        db.save_agent(snap, day)
