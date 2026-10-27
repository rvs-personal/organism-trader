"""Explicit, immutable intervention applied once at an approved day boundary."""
import json
from pathlib import Path


def load_intervention(run_dir, db, cfg, today):
    paths = sorted(Path(run_dir).glob('intervention-*.json'), key=lambda p: int(p.stem.split('-')[-1]))
    recorded_keys = {r[0] for r in db.conn.execute("SELECT key FROM run_state WHERE key LIKE 'intervention:%'")}
    file_keys = {'intervention:'+p.stem.split('-')[-1] for p in paths}
    if recorded_keys - file_keys:
        raise ValueError('Recorded intervention file is missing; refusing policy rollback')
    for path in paths:
        number = int(path.stem.split('-')[-1])
        spec = json.loads(path.read_text())
        if spec.get('approved') is not True:
            raise ValueError('Intervention is not approved')
        recorded = db.get_state(f'intervention:{number}')
        if recorded and json.loads(recorded) != spec:
            raise ValueError('Intervention changed after activation; create a new version instead')
        if today < spec['effective_day']:
            continue
        if not recorded:
            if today != spec['effective_day']:
                raise ValueError('Intervention must first activate at its declared boundary')
            boundary = 50 if number == 1 else spec['effective_day']-1
            checkpoint = Path(run_dir) / f'checkpoints/day-{boundary:03d}/state.sqlite'
            if not checkpoint.exists():
                raise ValueError('Original checkpoint must be preserved before intervention')
            with db.transaction():
                for aid, change in spec['agents'].items():
                    snap = db.load_agent(aid)
                    if not snap or snap.age_days != spec['effective_day']-1:
                        raise ValueError(f'{aid} is not at the intervention boundary')
                    snap.strategy_version += 1
                    snap.strategy = change['strategy']
                    snap.perception_policy = 'Use the supplied full-universe screen and operational policy. Inspect holdings and ranked candidates; explain exclusions.'
                    snap.working_memory = f'Intervention {number} begins. Prior lessons are historical observations, not binding restrictions.'
                    if spec.get('policy_version') == 2:
                        snap.beliefs = {}; snap.hypotheses = []
                        # Historical memories remain in the saved snapshot/archive; decision context is factual-only.
                    db.save_strategy_version(aid, snap.strategy_version, spec['effective_day'], snap.strategy, f'intervention_{number}')
                    db.save_agent(snap, spec['effective_day'])
                db.set_state(f'intervention:{number}', json.dumps(spec, sort_keys=True))
        cfg['_intervention'] = spec
    if paths:
        db.set_state('run_config', json.dumps(cfg))


def active(cfg, day):
    return bool(cfg.get('_intervention') and day >= cfg['_intervention']['effective_day'])


def instructions(cfg, aid):
    spec = cfg['_intervention']
    return '\n'.join([f"APPROVED {spec.get('id', 'intervention_1')}: applies from day {spec['effective_day']}. This protocol overrides contradictory historical beliefs, episodic lessons and long-term memories.",
        spec['protocol'], 'Agent-specific intervention:', spec['agents'][aid]['protocol']])


def reflection_feedback(db, aid, since, day, starting_cash):
    def window(owner):
        start = starting_cash if since == 0 else db.conn.execute(
            'SELECT equity FROM equity WHERE owner=? AND sim_day=?', (owner, since)).fetchone()
        if start is None:
            return None
        start = float(start if since == 0 else start[0])
        rows = db.conn.execute('SELECT sim_day,equity,exposure FROM equity WHERE owner=? AND sim_day>? AND sim_day<=? ORDER BY sim_day', (owner,since,day)).fetchall()
        if not rows or rows[-1][0] != day:
            return None
        prev = peak = start; worst = (0, None); drawdown = 0.
        for d, equity, exposure in rows:
            ret = equity / prev - 1
            if worst[1] is None or ret < worst[0]: worst = (ret,d)
            peak = max(peak,equity); drawdown = min(drawdown,equity/peak-1);prev=equity
        return dict(start_equity=start,end_equity=prev,return_frac=prev/start-1,
                    worst_daily_return_frac=worst[0],worst_day=worst[1],max_drawdown_frac=drawdown,
                    mean_exposure_frac=sum(r[2] for r in rows)/len(rows))
    own = window(aid)
    return dict(start_boundary=since,end_day=day,portfolio=own,
                comparisons={owner:window(owner) for owner in ['B_CASH','B_INDEX','B_EQUAL','B_MOM20','B_REV5']},
                interpretation='Returns include trading fees. Benchmarks have different exposures and looser position/order limits. Cash preservation is not proof that skipped trades would lose. Confidence is self-reported conviction, not a calibrated probability. Later portfolio P&L is not causal attribution to one decision.')
