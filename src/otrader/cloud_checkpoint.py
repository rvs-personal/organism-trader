"""Portable snapshots for ephemeral notebook runtimes. No keys or SDK stores."""
from pathlib import Path
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import zipfile


def allowed_metadata(name):
    return bool(re.fullmatch(r'intervention-[1-9][0-9]*\.json', name) or
                re.fullmatch(r'reviews/[A-Za-z0-9_-]+\.(?:md|json)', name))


def checkpoint(run_dir: Path, destination: Path) -> Path:
    """Back up committed SQLite/WAL state, then atomically replace one archive.

    Safe during inference; an in-flight, uncommitted agent is intentionally absent.
    Keep the active DB on the VM disk, never on a Google Drive/FUSE mount.
    """
    run_dir, destination = Path(run_dir), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='otrader-checkpoint-') as tmp:
        db_copy = Path(tmp) / 'state.sqlite'
        with sqlite3.connect(f'file:{run_dir / "state.sqlite"}?mode=ro', uri=True) as source:
            with sqlite3.connect(db_copy) as target:
                source.backup(target)
                if target.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise ValueError('Checkpoint integrity check failed')
        payload = {'state.sqlite': db_copy.read_bytes()}
        for name in ('pause_after_day.json', 'parallel_metrics.jsonl', 'control_events.jsonl'):
            if (run_dir / name).is_file():
                payload[name] = (run_dir / name).read_bytes()
        for path in sorted(run_dir.rglob('*')):
            name = path.relative_to(run_dir).as_posix()
            if path.is_file() and not path.is_symlink() and allowed_metadata(name):
                payload[name] = path.read_bytes()
        manifest = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
        payload['checkpoint-manifest.json'] = json.dumps(manifest, sort_keys=True).encode()
        fd, temp_name = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
                    for name, data in payload.items():
                        archive.writestr(name, data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, destination)
        finally:
            Path(temp_name).unlink(missing_ok=True)
    return destination


def restore(archive_path: Path, run_dir: Path) -> None:
    """Restore only into an absent run, with an allowlist and digest checks."""
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise ValueError('Refusing to overwrite an existing run; choose an absent run directory')
    allowed = {'state.sqlite', 'pause_after_day.json', 'parallel_metrics.jsonl', 'control_events.jsonl'}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(name not in allowed | {'checkpoint-manifest.json'} and not allowed_metadata(name) for name in names):
            raise ValueError('Unexpected checkpoint members')
        manifest = json.loads(archive.read('checkpoint-manifest.json'))
        if set(manifest) != set(names) - {'checkpoint-manifest.json'} or 'state.sqlite' not in manifest:
            raise ValueError('Incomplete checkpoint manifest')
        payload = {name: archive.read(name) for name in manifest}
        if any(hashlib.sha256(data).hexdigest() != manifest[name] for name, data in payload.items()):
            raise ValueError('Checkpoint checksum mismatch')
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.restore-', dir=run_dir.parent) as tmp:
        stage = Path(tmp) / 'run'
        stage.mkdir()
        for name, data in payload.items():
            (stage / name).parent.mkdir(parents=True, exist_ok=True)
            (stage / name).write_bytes(data)
        with sqlite3.connect(stage / 'state.sqlite') as db:
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Restored database is corrupt')
        stage.rename(run_dir)
