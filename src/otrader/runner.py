"""Resumable experiment runner. State lives in runs/<exp>/state.sqlite; re-running continues.

  python -m otrader.runner --days 20            # run 20 more sim days (agents + baselines)
  python -m otrader.runner --baselines-only
  python -m otrader.runner --backend stub
"""
from __future__ import annotations
import argparse
import json
import os
import fcntl
import re
import hashlib
import sys
import time
from pathlib import Path
from .config import DATA_DIR, RUNS_DIR, load_experiment
from .data.store import ClockedStore
from .sim.clock import Clock
from .sim.portfolio import Portfolio
from .sim.execution import CostModel, execute_orders
from .sim.risk import RiskRules
from .agents.snapshot import AgentSnapshot, StateDB
from .agents.loop import run_agent_day, run_reflection
from .baselines.strategies import BASELINES
from .inference.factory import make_backend


class RateLimitPause(Exception):
    pass


def _is_rate_limit(e: Exception) -> bool:
    s = repr(e).lower()
    return any(word in s for word in ("ratelimit", "rate_limit", "rate limit", "429", "usage limit", "quota exceeded", "resource_exhausted"))


def bootstrap_agents(db: StateDB, cfg: dict) -> None:
    for a in cfg["agents"]:
        if db.load_agent(a["id"]) is None:
            snap = AgentSnapshot(agent_id=a["id"], persona=a["persona"], model_id=cfg["inference"]["daily_model"],
                                 display_name=a.get("name", a["id"]),
                                 strategy=a.get("initial_strategy") or
                                 f"Initial strategy (v1): act according to persona: {a['persona']} "
                                 "Keep positions to 3-6 names, size 10-20% each, review after 5 days.")
            db.save_agent(snap, 0)
            db.save_strategy_version(snap.agent_id, 1, 0, snap.strategy, "birth")


def run_baseline_day(name: str, store: ClockedStore, db: StateDB, cfg: dict, rules: RiskRules, costs: CostModel) -> None:
    day = store.clock.today
    completed = db.get_state(f"baseline_day:{name}")
    if completed is not None and int(completed) >= day:
        return
    # Legacy runs committed their portfolio last; equity confirms the completed day.
    if completed is None and db.get_state(f"baseline:{name}") and db.conn.execute(
            "SELECT 1 FROM equity WHERE owner=? AND sim_day=?", (name, day)).fetchone():
        return
    raw = db.get_state(f"baseline:{name}")
    pf = Portfolio.from_dict(json.loads(raw)) if raw else Portfolio(cash=float(cfg["capital"]["starting_cash"]))
    assets = store.assets()
    marks = {a: store.last_close(a) for a in assets}; marks = {a: v for a, v in marks.items() if v is not None}
    opens = {a: store.today_open(a) for a in assets}; opens = {a: v for a, v in opens.items() if v is not None}
    orders = BASELINES[name](store, pf, marks, day)
    # baselines are not bound by the agent order cap / position caps beyond a sane maximum
    loose = RiskRules(max_positions=40, max_position_frac=1.0, max_gross_exposure_frac=1.0, no_shorting=True,
                      max_orders_per_day=40)
    execute_orders(orders, pf, opens, marks, loose, costs, day)
    closes = {a: float(store.bar_on(a, day)["close"]) for a in assets if store.bar_on(a, day) is not None}
    with db.transaction():
        db.save_equity(name, day, pf.equity(closes), pf.cash, pf.gross_exposure_frac(closes))
        db.save_fills(name, [f.__dict__ for f in pf.fills if f.sim_day == day], day)
        db.set_state(f"baseline:{name}", json.dumps(pf.to_dict()))
        db.set_state(f"baseline_day:{name}", str(day))



def _main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5, help="how many more sim days to run")
    ap.add_argument("--backend", default=None)
    ap.add_argument('--config', type=Path, help='Separate experiment configuration; never overwrite another run design')
    ap.add_argument('--prepare-only', action='store_true', help='Initialize a fresh run at day zero without inference or trading')
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--agents", default=None, help="comma list of agent ids to run (default all alive)")
    ap.add_argument("--run-id", help="separate run folder; preserves existing histories")
    ap.add_argument("--wait-for-reset", action="store_true", help="retry limits at the configured interval")
    ap.add_argument("--retry-seconds", type=int, help="poll interval, not a predicted reset time")
    ap.add_argument("--adaptive-parallel", action="store_true", help="adapt between 1 and 4 concurrent agents")
    args = ap.parse_args(argv)
    if args.days < 1:
        ap.error("--days must be positive")
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        ap.error("--run-id may contain letters, numbers, hyphens and underscores")

    import copy
    cfg = copy.deepcopy(load_experiment(args.config) if args.config else load_experiment())
    if cfg.get('initial_policy'):
        cfg['_intervention'] = copy.deepcopy(cfg['initial_policy'])
    exp = cfg["experiment_id"]
    anon_dir = DATA_DIR / "anon" / exp
    if not (anon_dir / "bars.parquet").exists():
        print(f"no anonymized data at {anon_dir}. Run: python -m otrader.data.fetch && python -m otrader.data.anonymize")
        return 2
    run_id = args.run_id or exp
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Held through this invocation, including waits; another runner cannot replay a day.
    global _run_lock, _run_db
    _run_lock = open(run_dir / "runner.lock", "a")
    try:
        fcntl.flock(_run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another runner owns this run. Choose a different --run-id or stop it first.")
        return 2
    db = _run_db = StateDB(run_dir / "state.sqlite")
    clock = Clock(int(db.get_state("clock", "1")))
    if args.prepare_only and (clock.today != 1 or db.conn.execute('SELECT 1 FROM decisions LIMIT 1').fetchone()):
        raise ValueError('Prepare-only requires a fresh run; it cannot reset saved history')
    store = ClockedStore.load(anon_dir, clock)
    rules = RiskRules.from_config(cfg["risk"]); costs = CostModel(**cfg["costs"])
    inference = cfg["inference"]
    backend_name = args.backend or os.environ.get("OTRADER_BACKEND") or inference.get("backend", "stub")
    if args.agents:
        unknown = set(args.agents.split(',')) - {a['id'] for a in cfg['agents']}
        if unknown:
            ap.error(f"Unknown agents: {sorted(unknown)}")
    design = {k: cfg[k] for k in ('capital', 'risk', 'costs', 'evolution', 'agents', 'experiment_id', 'n_sim_days')}
    design['inference'] = {k: inference.get(k) for k in ('daily_model', 'reflection_model', 'daily_effort', 'max_tool_turns')}
    if cfg.get('initial_policy'):
        design['initial_policy'] = cfg['initial_policy']
    if backend_name == 'ollama':
        design['local_options'] = inference.get('local_options', {})
    fingerprint = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    data_hash = hashlib.sha256((anon_dir / "bars.parquet").read_bytes()).hexdigest()
    mode = {"data_hash": data_hash, "design_hash": fingerprint, "backend": backend_name, "agents": sorted(args.agents.split(",")) if args.agents else "all",
            "baselines_only": args.baselines_only, "daily_model": inference["daily_model"],
            "reflection_model": inference["reflection_model"]}
    recorded = db.get_state("run_mode")
    if recorded and json.loads(recorded) != mode:
        print("Run configuration differs from its history; choose a new --run-id.")
        return 2
    if not recorded and backend_name != "stub" and db.conn.execute(
            "SELECT 1 FROM decisions WHERE model_id='stub' LIMIT 1").fetchone():
        print("Existing run contains stub decisions. Use --run-id cursor001 for a clean run.")
        return 2
    # Validate the original run identity first; an explicit extension may only append bars.
    from .data.extension import load_extension
    extension = load_extension(run_dir, store.bars, data_hash)
    if extension is not None:
        recorded_extension = db.get_state('data_extension')
        if recorded_extension and json.loads(recorded_extension) != json.loads((run_dir / 'data_extension.json').read_text()):
            raise ValueError('Active extension changed; preserve the run history')
        store.bars, cfg['n_sim_days'] = extension
        db.set_state('data_extension', (run_dir / 'data_extension.json').read_text())
        db.set_state('run_config', json.dumps(cfg))
    from .intervention import load_intervention
    load_intervention(run_dir, db, cfg, clock.today)
    backend = None
    if not args.baselines_only:
        kwargs = {}
        if backend_name == "cursor":
            kwargs = {"effort": inference.get("daily_effort"),
                      "included_only": inference.get("included_only", True),
                      "on_demand_disabled_confirmed": inference.get("on_demand_disabled_confirmed", False)}
        elif backend_name == 'ollama':
            kwargs = inference.get('local_options', {})
        backend = make_backend(backend_name, **kwargs)
        if backend_name == "cursor":
            backend.validate_models([inference["daily_model"], inference["reflection_model"]])
    db.set_state("run_mode", json.dumps(mode))
    if db.get_state("run_config") is None:
        db.set_state("run_config", json.dumps(cfg))
    db.set_state("experiment_id", exp)
    if args.config:
        db.set_state('config_path', str(args.config.resolve()))
    retry_seconds = args.retry_seconds if args.retry_seconds is not None else inference.get("retry_seconds", 3600)
    if retry_seconds < 60:
        ap.error("retry interval must be at least 60 seconds")
    bootstrap_agents(db, cfg)
    if args.prepare_only:
        db.set_state('clock', '1')
        db.set_state('runner_status', 'prepared: waiting to start')
        print(f'Prepared {run_id}: {len(cfg["agents"])} fresh agents, zero completed days. No inference performed.')
        return 0
    last_day = min(store.max_day(), int(cfg["n_sim_days"]))
    ev = cfg["evolution"]

    from .parallel import AdaptiveConcurrency, run_batch, transient, quota, host_pressure
    settings = json.loads(db.get_state('parallel_control', '{}'))
    maximum = max(1, min(4, int(inference.get('local_parallel_max', 2)))) if backend_name == 'ollama' else max(1, min(4, int(inference.get('parallel_max', 4))))
    control = AdaptiveConcurrency(workers=max(1, min(maximum, int(settings.get('workers', 2)))),
                                  stable_days=int(settings.get('stable_days', 0)), maximum=maximum)

    def parallel_agent(aid):
        # Connections and Cursor workspaces are owned by the worker, never shared.
        worker_db = StateDB(run_dir / 'state.sqlite')
        started = time.monotonic()
        outcome = 'ok'
        worker_backend = None
        try:
            snap = worker_db.load_agent(aid)
            if snap is None or (snap.status != 'ALIVE' and (not snap.equity_curve or snap.equity_curve[-1]['day'] != clock.today)):
                return
            worker_backend = backend
            if backend_name == 'cursor':
                worker_backend = make_backend(backend_name, **kwargs)
                worker_backend._catalog = backend._catalog
            elif backend_name == 'ollama':
                worker_backend = make_backend(backend_name, **kwargs)
            prev = worker_db.get_state(f'exec:{aid}')
            print(f'[day {clock.today}] {aid}: parallel decision/memory/reflection', flush=True)
            res = run_agent_day(snap, store, worker_db, worker_backend, cfg, rules, costs, json.loads(prev) if prev else None)
            worker_db.set_state(f'exec:{aid}', json.dumps(res))
            if snap.status == 'ALIVE':
                if clock.today % ev['review_every_days'] == 0:
                    run_reflection(snap, worker_db, worker_backend, cfg, clock.today, 'review')
                elif clock.today % ev['reflection_every_days'] == 0:
                    run_reflection(snap, worker_db, worker_backend, cfg, clock.today, 'reflection')
        except Exception as error:
            outcome = 'quota' if quota(error) else 'transient' if transient(error) else 'error'
            raise
        finally:
            if backend_name == 'ollama' and worker_backend is not None and worker_backend is not backend:
                worker_backend.client.close()
            worker_db.set_state(f'parallel_agent:{aid}', json.dumps(dict(day=clock.today,
                seconds=round(time.monotonic()-started, 1), outcome=outcome)))
            worker_db.conn.close()

    end_day = min(clock.today + args.days - 1, last_day)
    while clock.today <= end_day:
        if backend_name == 'ollama':
            from .host_health import snapshot, should_pause, should_reduce
            health = snapshot()
            db.set_state('host_health', json.dumps(health))
            if should_reduce(health):
                control.failure()
            if should_pause(health):
                db.set_state('runner_status', 'paused: laptop resource pressure')
                print('Stopped before the next day because of laptop resource pressure. Saved progress is intact.', flush=True)
                return 3
        checkpoint = run_dir / 'pause_after_day.json'
        if checkpoint.exists():
            pause_day = int(json.loads(checkpoint.read_text())['day'])
            if clock.today > pause_day:
                db.set_state('runner_status', f'paused: review after day {pause_day}')
                print(f'Review checkpoint reached after day {pause_day}; saved and stopped before day {clock.today}.', flush=True)
                return 0
        day = clock.today
        if day > last_day:
            print("experiment complete at day", last_day); break
        t0 = time.time()
        paused = False
        db.set_state("runner_status", "running")
        for b in BASELINES:
            run_baseline_day(b, store, db, cfg, rules, costs)
        if not args.baselines_only:
            ids = args.agents.split(",") if args.agents else db.list_agents()
            if args.adaptive_parallel:
                pending = ids
                failures = False
                for attempt in range(4):
                    if host_pressure(backend_name):
                        control.failure()
                    db.set_state('parallel_control', json.dumps(control.__dict__))
                    errors = run_batch(pending, parallel_agent, control.workers)
                    if not errors:
                        break
                    failures = True
                    control.failure()
                    db.set_state('parallel_control', json.dumps(control.__dict__))
                    fatal = [e for e in errors.values() if not transient(e) and not quota(e)]
                    if fatal:
                        raise fatal[0]
                    pending = list(errors)
                    if any(quota(e) for e in errors.values()) or attempt == 3:
                        db.set_state('runner_status', 'paused: usage/rate limit')
                        if not args.wait_for_reset:
                            return 3
                        db.set_state('retry_at', str(time.time() + retry_seconds))
                        print(f'[day {day}] repeated limit; waiting {retry_seconds}s at {control.workers} worker(s)', flush=True)
                        time.sleep(retry_seconds)
                        paused = True
                        break
                    delay = 60 * 2 ** attempt
                    db.set_state('runner_status', 'paused: concurrency backoff')
                    db.set_state('retry_at', str(time.time() + delay))
                    print(f'[day {day}] transient failure; reducing to {control.workers}, retry in {delay}s', flush=True)
                    time.sleep(delay)
                    db.set_state('runner_status', 'running')
                if not failures:
                    control.success(host_pressure(backend_name))
                db.set_state('parallel_control', json.dumps(control.__dict__))
                ids = []  # sequential path below is disabled for this day
            for aid in ids:
                snap = db.load_agent(aid)
                if snap is None or (snap.status != "ALIVE" and (not snap.equity_curve or snap.equity_curve[-1]["day"] != day)):
                    continue
                prev = db.get_state(f"exec:{aid}")
                try:
                    print(f"[day {day}] {snap.display_name or aid} ({aid}): decision/memory", flush=True)
                    res = run_agent_day(snap, store, db, backend, cfg, rules, costs, json.loads(prev) if prev else None)
                    db.set_state(f"exec:{aid}", json.dumps(res))
                    if snap.status != "ALIVE":
                        continue
                    if day % ev["review_every_days"] == 0:
                        run_reflection(snap, db, backend, cfg, day, "review")
                    elif day % ev["reflection_every_days"] == 0:
                        run_reflection(snap, db, backend, cfg, day, "reflection")
                except Exception as e:
                    if _is_rate_limit(e):
                        print(f"[day {day}] rate/usage limit for {aid}: {e}. Pausing; re-run to resume.")
                        db.set_state("clock", str(day))
                        db.set_state("runner_status", "paused: usage/rate limit")
                        if not args.wait_for_reset:
                            return 3
                        db.set_state("retry_at", str(time.time() + retry_seconds))
                        time.sleep(retry_seconds)
                        paused = True
                        break
                    raise
        if paused:
            continue
        db.set_state("clock", str(day + 1))
        clock.advance()
        if args.adaptive_parallel:
            import resource
            sample = dict(day=day, seconds=round(time.time()-t0, 1), next_workers=control.workers,
                          load_1m=os.getloadavg()[0], runner_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            with (run_dir / 'parallel_metrics.jsonl').open('a') as log:
                log.write(json.dumps(sample) + '\n')
        alive = [a for a in db.list_agents() if (db.load_agent(a) or AgentSnapshot('x','','')).status == "ALIVE"]
        print(f"day {day} done in {time.time()-t0:.1f}s; alive={alive}")
    db.set_state("runner_status", "complete" if clock.today > last_day else "idle")
    return 0


_run_lock = None
_run_db = None


def main(argv=None):
    global _run_lock, _run_db
    try:
        return _main(argv)
    except KeyboardInterrupt:
        if _run_db is not None:
            _run_db.set_state("runner_status", "stopped")
        return 130
    except Exception:
        if _run_db is not None:
            _run_db.set_state("runner_status", "error: see runner log")
        raise
    finally:
        if _run_db is not None:
            _run_db.conn.close()
            _run_db = None
        if _run_lock is not None:
            _run_lock.close()
            _run_lock = None


if __name__ == "__main__":
    sys.exit(main())
