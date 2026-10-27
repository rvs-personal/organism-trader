import copy
import json
from pathlib import Path
import pandas as pd
import pytest
from otrader.intervention import reflection_feedback, load_intervention
from otrader.agents.snapshot import StateDB, AgentSnapshot
from otrader.agents.schemas import InterventionDecision
from otrader.agents.tools import ToolBox
from otrader.sim.portfolio import Portfolio
from otrader.data.features import rsi
from otrader.data.store import ClockedStore
from otrader.sim.clock import Clock
from conftest import synthetic_bars


def test_feedback_includes_first_day_and_excludes_future(tmp_path):
    db=StateDB(tmp_path/'state.sqlite')
    for d,e in [(10,120),(11,110),(12,130),(13,9999)]:
        db.save_equity('A1',d,e,e/2,.5)
    feedback=reflection_feedback(db,'A1',10,12,100)['portfolio']
    assert feedback['start_equity']==120
    assert feedback['return_frac']==pytest.approx(130/120-1)
    assert feedback['worst_day']==11
    assert feedback['worst_daily_return_frac']==pytest.approx(110/120-1)
    assert feedback['max_drawdown_frac']==pytest.approx(110/120-1)
    db.conn.close()


def test_rsi_extremes_are_current():
    assert rsi(pd.Series(range(30)))==100
    assert rsi(pd.Series(range(30,0,-1)))==0
    assert rsi(pd.Series([5.]*30))==50


def test_screen_covers_universe_without_future_data():
    bars=synthetic_bars(days=30)
    store=ClockedStore(bars,Clock(20));tb=ToolBox(store,Portfolio(1e6),lambda _:[],{})
    before=tb.screen_universe()
    assert len(before['candidates'])==6
    assert before['as_of_day']==19
    assert before['market']['volx_age_in_sessions']==0
    store.bars.loc[store.bars.sim_day>=20,['close','volume']]*=10
    assert before==tb.screen_universe()
    store.bars=store.bars[~((store.bars.asset=='VOLX')&(store.bars.sim_day==19))]
    assert tb.screen_universe()['market']['volx_age_in_sessions']==1


def test_intervention_is_once_only_and_keeps_historical_ledger(tmp_path):
    db=StateDB(tmp_path/'state.sqlite');snap=AgentSnapshot('A1','persona','stub',age_days=50,strategy='old')
    db.save_agent(snap,50);db.save_strategy_version('A1',1,0,'old','birth');db.save_equity('A1',50,1e6,1e6,0)
    spec={'approved':True,'effective_day':51,'agents':{'A1':{'strategy':'new'}}}
    path=tmp_path/'intervention-1.json';path.write_text(json.dumps(spec))
    checkpoint=tmp_path/'checkpoints/day-050';checkpoint.mkdir(parents=True);(checkpoint/'state.sqlite').touch()
    cfg={};load_intervention(tmp_path,db,cfg,51)
    assert db.load_agent('A1').strategy_version==2
    load_intervention(tmp_path,db,cfg,52)
    assert db.load_agent('A1').strategy_version==2
    assert db.conn.execute('SELECT equity FROM equity WHERE sim_day=50').fetchone()[0]==1e6
    assert db.conn.execute('SELECT strategy FROM strategy_versions WHERE sim_day=0').fetchone()[0]=='old'
    spec['agents']['A1']['strategy']='tampered';path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='changed'):load_intervention(tmp_path,db,cfg,52)
    db.conn.close()


def test_intervention_requires_candidate_reason():
    with pytest.raises(ValueError):
        InterventionDecision(market_view='x',confidence=.4)
    d=InterventionDecision(market_view='x',confidence=.4,candidate_assessments=[dict(asset='STK_01',disposition='reject',reason='RSI too high')])
    assert d.candidate_assessments[0].asset=='STK_01'


def test_intervention_continuation_records_screen_and_feedback(tmp_path,monkeypatch):
    from otrader import runner
    from otrader.inference.stub_backend import StubBackend
    from test_resume_and_dashboard import setup_runner
    class Backend(StubBackend):
        saw_feedback=False
        def run_episode(self,system,user,*args,**kwargs):
            out=super().run_episode(system,user,*args,**kwargs)
            if 'candidate_assessments' in user and 'Decision JSON schema' in user:
                body=json.loads(out.text);body['candidate_assessments']=[];out.text=json.dumps(body)
            if 'Legacy confidence calibration is disabled' in user:
                self.saw_feedback=True
                assert 'mean_exposure_frac' in user
            return out
    backend=Backend();path=setup_runner(tmp_path,monkeypatch,backend)
    assert runner.main(['--days','1','--backend','stub'])==0
    db=StateDB(path);before=db.conn.execute('SELECT * FROM decisions WHERE sim_day=1').fetchall();db.conn.close()
    checkpoint=path.parent/'checkpoints/day-050';checkpoint.mkdir(parents=True);(checkpoint/'state.sqlite').touch()
    spec=dict(approved=True,effective_day=2,protocol='Test intervention',agents={a:dict(strategy='Test strategy',protocol='Test instructions') for a in ['A1','A2']})
    (path.parent/'intervention-1.json').write_text(json.dumps(spec))
    assert runner.main(['--days','1','--backend','stub','--adaptive-parallel'])==0
    db=StateDB(path)
    assert db.conn.execute('SELECT * FROM decisions WHERE sim_day=1').fetchall()==before
    assert db.conn.execute('SELECT COUNT(*) FROM research_context WHERE sim_day=2').fetchone()[0]==2
    assert backend.saw_feedback
    db.conn.close()
