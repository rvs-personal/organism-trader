import json,sqlite3
from types import SimpleNamespace
import pytest
from otrader import run_control as control


def test_resume_moves_checkpoint_once_and_guards_running(tmp_path,monkeypatch):
    monkeypatch.setattr(control,'ROOT',tmp_path);monkeypatch.setattr(control,'RUNS_DIR',tmp_path/'runs')
    run=tmp_path/'runs/test';run.mkdir(parents=True)
    db=sqlite3.connect(run/'state.sqlite');db.execute('CREATE TABLE run_state(key TEXT PRIMARY KEY,value TEXT)')
    db.executemany('INSERT INTO run_state VALUES (?,?)',[('clock','51'),('runner_status','paused: review after day 50'),('run_mode',json.dumps({'backend':'cursor'})),('run_config',json.dumps({'n_sim_days':500}))]);db.commit();db.close()
    (run/'pause_after_day.json').write_text(json.dumps({'day':50}));(tmp_path/'.env').write_text('CURSOR_API_KEY=test_local_key')
    monkeypatch.setattr(control,'running',lambda _:False)
    with pytest.raises(ValueError,match='Complete the milestone review'):control.start_run('test',True)
    (run/'reviews').mkdir();(run/'reviews/day-050-review.md').write_text('Completed review')
    monkeypatch.setattr(control.subprocess,'Popen',lambda *a,**k:SimpleNamespace(pid=12345))
    result=control.start_run('test',True)
    assert result==dict(pid=12345,from_day=51,stop_after_day=100)
    assert json.loads((run/'pause_after_day.json').read_text())['day']==100
    monkeypatch.setattr(control,'running',lambda _:True)
    with pytest.raises(ValueError,match='already running'):control.start_run('test',True)
