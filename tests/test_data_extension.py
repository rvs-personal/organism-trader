import hashlib
import json
import pandas as pd
import pytest
from otrader.data.extension import load_extension
from otrader import runner
from otrader.agents.snapshot import StateDB
from otrader.inference.stub_backend import StubBackend
from test_resume_and_dashboard import setup_runner


def manifest(run, base, extended, target):
    path = run / 'extended.parquet'
    extended.to_parquet(path, index=False)
    spec = dict(bars_path=str(path), base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                extension_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), n_sim_days=target)
    (run/'data_extension.json').write_text(json.dumps(spec))
    return spec


def test_extension_resumes_without_replaying(tmp_path, monkeypatch):
    dbpath = setup_runner(tmp_path, monkeypatch, StubBackend())
    assert runner.main(['--days','12','--backend','stub']) == 0
    base = tmp_path/'data/anon/test/bars.parquet'
    old = pd.read_parquet(base)
    future = old[old.sim_day==12].copy()
    future['sim_day']=13
    spec=manifest(dbpath.parent, base, pd.concat([old,future],ignore_index=True),13)
    db=StateDB(dbpath)
    previous=db.conn.execute('SELECT * FROM decisions').fetchall();db.conn.close()
    assert runner.main(['--days','10','--backend','stub']) == 0
    db=StateDB(dbpath)
    assert db.get_state('clock')=='14'
    assert db.conn.execute('SELECT * FROM decisions WHERE sim_day<=12').fetchall()==previous
    assert json.loads(db.get_state('run_config'))['n_sim_days']==13
    db.conn.close()
    # Even a newly signed manifest cannot replace an existing historical price.
    altered=pd.read_parquet(spec['bars_path']); altered.loc[0,'close']+=1
    manifest(dbpath.parent,base,altered,13)
    with pytest.raises(AssertionError):
        runner.main(['--days','1','--backend','stub'])


def test_extension_rejects_gaps_and_checksum_changes(tmp_path):
    old=pd.DataFrame([dict(sim_day=1,asset='MKT',open=1.,high=1.,low=1.,close=1.,volume=1.)])
    base=tmp_path/'base.parquet';old.to_parquet(base,index=False)
    new=old.copy();new.sim_day=3
    spec=manifest(tmp_path,base,pd.concat([old,new],ignore_index=True),3)
    with pytest.raises(ValueError,match='consecutive'):
        load_extension(tmp_path,old,spec['base_sha256'])
    with open(spec['bars_path'],'ab') as f: f.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        load_extension(tmp_path,old,spec['base_sha256'])
