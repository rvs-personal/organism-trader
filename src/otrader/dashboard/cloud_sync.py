"""Authenticated, write-only checkpoint receiver; never runs simulation code."""
from pathlib import Path
import hashlib
import hmac
import json
import os
import sqlite3
import tempfile
import time
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from otrader.cloud_checkpoint import restore, allowed_metadata

MAX_UPLOAD = 20 * 1024 * 1024
MAX_EXPANDED = 100 * 1024 * 1024


def import_snapshot(archive, run_dir, bundle_hash, remote):
    run_dir = Path(run_dir)
    if run_dir.exists() and not (run_dir / 'cloud_source.json').exists():
        raise ValueError('Refusing to replace a local experiment')
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist()) > MAX_EXPANDED:
            raise ValueError('Checkpoint is too large')
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=run_dir.parent, prefix='.cloud-import-') as temp:
        staged = Path(temp) / 'verified'
        restore(archive, staged)
        with sqlite3.connect(f'file:{staged}/state.sqlite?mode=ro', uri=True) as db:
            state = dict(db.execute('SELECT key,value FROM run_state'))
            if state.get('cloud_bundle_hash') != bundle_hash:
                raise ValueError('Unexpected experiment identity')
            clock = int(state['clock'])
            decisions = db.execute('SELECT COUNT(*) FROM decisions').fetchone()[0]
            if {r[0] for r in db.execute('SELECT agent_id FROM agents')} != {f'A{i}' for i in range(1, 9)}:
                raise ValueError('Unexpected agent cohort')
        if (run_dir / 'state.sqlite').exists():
            with sqlite3.connect(f'file:{run_dir}/state.sqlite?mode=ro', uri=True) as db:
                prior = dict(db.execute('SELECT key,value FROM run_state'))
                count = db.execute('SELECT COUNT(*) FROM decisions').fetchone()[0]
                if clock < int(prior['clock']) or decisions < count:
                    raise ValueError('Refusing an older checkpoint')
        source = dict(host='Colab', run_id=run_dir.name, received_at=time.time(),
                      sampled_at=remote['sampled_at'], runner_alive=remote['runner_alive'],
                      checkpoint_age_seconds=remote['checkpoint_age_seconds'],
                      completed_days=clock-1, bundle_hash=bundle_hash)
        (staged / 'cloud_source.json').write_text(json.dumps(source))
        if not run_dir.exists():
            staged.rename(run_dir)
        else:
            # Read-only dashboard connections keep their existing inode/snapshot.
            for name in ('state.sqlite', 'pause_after_day.json', 'parallel_metrics.jsonl', 'control_events.jsonl', 'cloud_source.json'):
                if (staged / name).exists():
                    os.replace(staged / name, run_dir / name)
            for path in staged.rglob('*'):
                name = path.relative_to(staged).as_posix()
                if path.is_file() and allowed_metadata(name):
                    target = run_dir / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(path, target)
        return source


def serve(config_path):
    cfg = json.loads(Path(config_path).read_text())
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Never log request headers, keys or contents.
        def do_GET(self):
            self.send_error(405)
        def do_POST(self):
            self.connection.settimeout(20)
            if self.path != '/checkpoint' or not hmac.compare_digest(
                self.headers.get('Authorization', ''), 'Bearer ' + cfg['token']
            ):
                self.send_error(403); return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= MAX_UPLOAD:
                    self.send_error(413); return
                remote = json.loads(self.headers['X-Cloud-Status'])
                now = time.time()
                if not isinstance(remote['runner_alive'], bool) or abs(now-float(remote['sampled_at'])) > 300:
                    raise ValueError('Invalid status')
                if not 0 <= float(remote['checkpoint_age_seconds']) < 86400:
                    raise ValueError('Invalid checkpoint age')
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError('Incomplete upload')
                with tempfile.NamedTemporaryFile(suffix='.zip') as file:
                    file.write(body); file.flush()
                    result = import_snapshot(file.name, cfg['run_dir'], cfg['bundle_hash'], remote)
                response = json.dumps({'ok': True, 'completed_days': result['completed_days']}).encode()
                self.send_response(200); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response))); self.end_headers(); self.wfile.write(response)
            except Exception:
                self.send_error(400, 'Checkpoint rejected; previous results preserved')
    HTTPServer(('127.0.0.1', cfg.get('port', 8503)), Handler).serve_forever()

if __name__ == '__main__':
    import sys
    serve(sys.argv[1])
