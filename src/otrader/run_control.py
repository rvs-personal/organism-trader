"""Local dashboard controls. A review resume is an explicit user action."""
from pathlib import Path
import fcntl
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
from .config import ROOT, RUNS_DIR


def running(run_dir):
    with (run_dir / 'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def start_run(run_id='cursor001', resume_review=False):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid run id')
    run = RUNS_DIR / run_id
    if not (run/'state.sqlite').exists():
        raise ValueError('No saved simulation to resume')
    if (run/'cloud_source.json').exists():
        raise ValueError('This is a Colab mirror. Resume the experiment on Colab, never on this Mac.')
    with (run/'launch.lock').open('a') as launch:
        fcntl.flock(launch, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if running(run):
            raise ValueError('The simulation is already running')
        if (run/'process.pid').exists():
            try:
                os.kill(int((run/'process.pid').read_text()), 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError('A runner is starting or still exiting; try again shortly')
        with sqlite3.connect(f'file:{run}/state.sqlite?mode=ro', uri=True) as db:
            state = dict(db.execute('SELECT key,value FROM run_state'))
        backend = json.loads(state['run_mode'])['backend']
        if backend not in ('cursor', 'ollama'):
            raise ValueError('This control resumes existing Cursor or local Ollama runs only')
        clock = int(state['clock'])
        if state.get('runner_status', '').startswith('prepared:') and state.get('launch_readiness') != 'ready':
            raise ValueError('Local validation is not complete; this fresh run is not ready to start')
        config = json.loads(state['run_config'])
        target = int(config['n_sim_days'])
        if (run/'data_extension.json').exists():
            target=max(target,int(json.loads((run/'data_extension.json').read_text())['n_sim_days']))
        if clock > target:
            raise ValueError('The full experiment is complete')
        pause = run/'pause_after_day.json'
        checkpoint = json.loads(pause.read_text()) if pause.exists() else None
        if checkpoint and clock > checkpoint['day'] and not resume_review:
            raise ValueError('Review checkpoint requires an explicit Resume action')
        if resume_review:
            if not checkpoint or clock != checkpoint['day']+1 or not state['runner_status'].startswith('paused: review'):
                raise ValueError('Resume is available after a fully saved review checkpoint')
            if not (run/'reviews'/f"day-{checkpoint['day']:03d}-review.md").exists():
                raise ValueError('Complete the milestone review before resuming')
        env=os.environ.copy()
        for line in ((ROOT/'.env').read_text().splitlines() if backend == 'cursor' else []):
            line=line.strip()
            if not line or line.startswith('#') or '=' not in line: continue
            key,value=line.removeprefix('export ').split('=',1)
            if key.strip()=='CURSOR_API_KEY': env[key.strip()]=' '.join(shlex.split(value,comments=True))
        if backend == 'cursor' and (not env.get('CURSOR_API_KEY') or env['CURSOR_API_KEY']=='crsr_xxx'):
            raise ValueError('Save the Cursor key locally before resuming')
        old=pause.read_bytes() if pause.exists() else None
        next_checkpoint=min(checkpoint['day']+50,target) if resume_review else (checkpoint['day'] if checkpoint else target)
        if resume_review:
            temp=run/'pause_after_day.json.tmp'
            temp.write_text(json.dumps({'day':next_checkpoint,'reason':'Stop for review. Explicit resume required.'},indent=2));temp.replace(pause)
        try:
            with (run/'runner.log').open('a') as log:
                command = [sys.executable,'-u','-m','otrader.runner','--backend',backend,'--run-id',run_id,
                    '--days',str(target),'--wait-for-reset','--adaptive-parallel']
                if state.get('config_path'): command += ['--config', state['config_path']]
                proc=subprocess.Popen(command,cwd=ROOT,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        except Exception:
            if old is not None: pause.write_bytes(old)
            raise
        (run/'process.pid').write_text(str(proc.pid))
        with (run/'control_events.jsonl').open('a') as events:
            import time
            events.write(json.dumps(dict(time=time.time(),action='resume_review' if resume_review else 'start',from_day=clock,stop_after_day=next_checkpoint,pid=proc.pid))+'\n')
        return dict(pid=proc.pid,from_day=clock,stop_after_day=next_checkpoint)
