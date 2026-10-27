import json
import sqlite3
import time
import zipfile
import pytest
from otrader.cloud_checkpoint import checkpoint
from otrader.dashboard.cloud_sync import import_snapshot


def make_checkpoint(tmp_path, clock=3, identity='expected'):
    source=tmp_path/f'source{clock}';source.mkdir()
    with sqlite3.connect(source/'state.sqlite') as db:
        db.execute('CREATE TABLE run_state (key TEXT,value TEXT)')
        db.executemany('INSERT INTO run_state VALUES (?,?)',[('clock',str(clock)),('cloud_bundle_hash',identity)])
        db.execute('CREATE TABLE decisions (agent_id TEXT)')
        db.executemany('INSERT INTO decisions VALUES (?)',[('A1',)]*clock)
        db.execute('CREATE TABLE agents (agent_id TEXT)')
        db.executemany('INSERT INTO agents VALUES (?)',[(f'A{i}',) for i in range(1,9)])
    return checkpoint(source,tmp_path/f'checkpoint{clock}.zip')


def test_import_update_preserves_last_good_and_blocks_regression(tmp_path):
    run=tmp_path/'mirror'; remote=dict(sampled_at=time.time(),runner_alive=True,checkpoint_age_seconds=5)
    archive=make_checkpoint(tmp_path)
    import_snapshot(archive,run,'expected',remote)
    assert json.loads((run/'cloud_source.json').read_text())['completed_days']==2
    newer=make_checkpoint(tmp_path,4)
    import_snapshot(newer,run,'expected',remote)
    original=(run/'state.sqlite').read_bytes()
    with pytest.raises(ValueError,match='older'): import_snapshot(archive,run,'expected',remote)
    with pytest.raises(ValueError,match='identity'): import_snapshot(newer,run,'other',remote)
    corrupt=tmp_path/'corrupt.zip'
    with zipfile.ZipFile(newer) as src,zipfile.ZipFile(corrupt,'w') as dst:
        for name in src.namelist(): dst.writestr(name, b'broken' if name=='state.sqlite' else src.read(name))
    with pytest.raises(ValueError,match='checksum'): import_snapshot(corrupt,run,'expected',remote)
    assert (run/'state.sqlite').read_bytes()==original


def test_cannot_replace_local_or_launch_mirror(tmp_path,monkeypatch):
    from otrader import run_control
    local=tmp_path/'local';local.mkdir()
    with pytest.raises(ValueError,match='local experiment'): import_snapshot(None,local,'',{})
    (local/'state.sqlite').touch();(local/'cloud_source.json').write_text('{}')
    monkeypatch.setattr(run_control,'RUNS_DIR',tmp_path)
    with pytest.raises(ValueError,match='Colab mirror'): run_control.start_run('local')
