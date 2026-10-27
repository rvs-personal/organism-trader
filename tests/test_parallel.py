import threading
import json
from otrader.parallel import AdaptiveConcurrency, run_batch
from otrader import runner
from otrader.agents.snapshot import StateDB
from otrader.inference.stub_backend import StubBackend
from test_resume_and_dashboard import setup_runner


def test_batch_actually_overlaps_and_drains_errors():
    barrier=threading.Barrier(2)
    completed=[]
    def work(aid):
        barrier.wait(timeout=3)
        if aid=='A': raise RuntimeError('429')
        completed.append(aid)
    errors=run_batch(['A','B'],work,2)
    assert list(errors)==['A']
    assert completed==['B']


def test_growth_and_backoff():
    c=AdaptiveConcurrency()
    for _ in range(3): c.success()
    assert c.workers==3
    c.failure()
    assert c.workers==1 and c.stable_days==0
    for _ in range(12): c.success()
    assert c.workers==4
    c.success(pressure=True)
    assert c.workers==2


def test_parallel_retries_only_failed_agent(tmp_path,monkeypatch):
    path=setup_runner(tmp_path,monkeypatch,StubBackend())
    from otrader import parallel
    monkeypatch.setattr(parallel,'host_pressure',lambda *args:False)
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    original=runner.run_agent_day
    counts={}
    def fail_once(snap,*args,**kw):
        counts[snap.agent_id]=counts.get(snap.agent_id,0)+1
        if snap.agent_id=='A1' and counts['A1']==1: raise RuntimeError('429 rate limit')
        return original(snap,*args,**kw)
    monkeypatch.setattr(runner,'run_agent_day',fail_once)
    assert runner.main(['--days','1','--backend','stub','--adaptive-parallel'])==0
    assert counts=={'A1':2,'A2':1}
    db=StateDB(path)
    assert db.get_state('clock')=='2'
    assert db.conn.execute('SELECT COUNT(*) FROM decisions').fetchone()[0]==2
    assert json.loads(db.get_state('parallel_control'))['workers']==1
    assert db.conn.execute('SELECT COUNT(*) FROM reflections').fetchone()[0]==2
    db.conn.close()


def test_review_checkpoint_finishes_day_and_blocks_resume(tmp_path,monkeypatch):
    path=setup_runner(tmp_path,monkeypatch,StubBackend())
    path.parent.mkdir(parents=True,exist_ok=True)
    (path.parent/'pause_after_day.json').write_text(json.dumps({'day':1}))
    assert runner.main(['--days','12','--backend','stub','--adaptive-parallel'])==0
    db=StateDB(path)
    assert db.get_state('clock')=='2'
    assert db.get_state('runner_status')=='paused: review after day 1'
    assert db.conn.execute('SELECT COUNT(*) FROM decisions WHERE sim_day=1').fetchone()[0]==2
    assert db.conn.execute('SELECT COUNT(*) FROM reflections WHERE sim_day=1').fetchone()[0]==2
    before={t:sorted(db.conn.execute(f'SELECT * FROM {t}').fetchall()) for t in ['agents','decisions','reflections','equity','fills','run_state']};db.conn.close()
    assert runner.main(['--days','12','--backend','stub','--adaptive-parallel'])==0
    db=StateDB(path)
    assert {t:sorted(db.conn.execute(f'SELECT * FROM {t}').fetchall()) for t in before}==before
    db.conn.close()


def test_remote_inference_does_not_throttle_on_local_cpu_load(monkeypatch):
    from otrader import parallel
    monkeypatch.setattr(parallel.os, 'getloadavg', lambda: (100, 100, 100))
    monkeypatch.setattr(parallel.os, 'cpu_count', lambda: 2)
    assert parallel.host_pressure('cursor') is False
    assert parallel.host_pressure('ollama') is True
    control=AdaptiveConcurrency(workers=4, maximum=4)
    for _ in range(5):
        if parallel.host_pressure('cursor'): control.failure()
        control.success(parallel.host_pressure('cursor'))
    assert control.workers == 4
    assert parallel.transient(RuntimeError('429 rate limit'))
    assert parallel.transient(TimeoutError('request timed out'))
    control.failure()
    assert control.workers == 2


def test_cursor_runner_ignores_cpu_pressure_but_retries_failed_agent(tmp_path,monkeypatch):
    class FakeCursor(StubBackend):
        _catalog = {}
        def validate_models(self, models): pass
    path=setup_runner(tmp_path,monkeypatch,FakeCursor())
    from otrader import parallel
    monkeypatch.setattr(parallel.os,'getloadavg',lambda:(100,100,100))
    monkeypatch.setattr(parallel.os,'cpu_count',lambda:2)
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    db=StateDB(path)
    db.set_state('parallel_control',json.dumps(dict(workers=4,stable_days=0,maximum=4)))
    db.conn.close()
    observed=[]; original_batch=parallel.run_batch
    def batch(ids,work,workers):
        observed.append(workers)
        return original_batch(ids,work,workers)
    monkeypatch.setattr(parallel,'run_batch',batch)
    original=runner.run_agent_day; counts={}
    def fail_once(snap,*args,**kw):
        counts[snap.agent_id]=counts.get(snap.agent_id,0)+1
        if snap.agent_id=='A1' and counts[snap.agent_id]==1: raise RuntimeError('429 rate limit')
        return original(snap,*args,**kw)
    monkeypatch.setattr(runner,'run_agent_day',fail_once)
    assert runner.main(['--days','1','--backend','cursor','--adaptive-parallel']) == 0
    assert observed == [4,2]
    assert counts == {'A1':2,'A2':1}
    db=StateDB(path)
    assert db.get_state('clock') == '2'
    assert db.conn.execute('SELECT COUNT(*) FROM decisions').fetchone()[0] == 2
    db.conn.close()
