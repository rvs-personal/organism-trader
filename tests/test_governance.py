import copy
import json
from pathlib import Path
import pytest
from otrader import governance as G
from otrader.agents.snapshot import StateDB, AgentSnapshot
from otrader.agents.loop import run_agent_day, run_reflection
from otrader.agents.tools import ToolBox
from otrader.sim.portfolio import Portfolio, Fill
from otrader.sim.risk import RiskRules
from otrader.data.store import ClockedStore
from otrader.sim.clock import Clock
from otrader.inference.stub_backend import StubBackend
from otrader.intervention import load_intervention
from conftest import synthetic_bars
from test_end_to_end_stub import CFG, COSTS


def spec():
    return dict(id='intervention_2', approved=True, policy_version=2,effective_day=1,review_day=50,
                protocol='Test fixed policy',agents={'A1':dict(strategy='Fixed',protocol='Fixed',policy=dict(
                    entry=[['rsi_14','le',100]],rank='rsi_14',descending=False,max_positions=2,entry_fraction=.1,exit_style='trend'))})


def test_fifo_feedback_partial_lots_includes_all_fees():
    pf=Portfolio(10000)
    for f in [Fill(1,'X','BUY',10,100,1000,10,0,''),Fill(2,'X','BUY',10,120,1200,10,0,''),
              Fill(3,'X','SELL',15,130,1950,15,0,''),Fill(4,'X','SELL',5,100,500,5,0,'')]:pf.apply_fill(f)
    result=G.portfolio_feedback(pf,{})
    assert [x['net_pnl_inr'] for x in result['closed_trades_net_of_all_fees']]==[320,-110]
    assert result['net_realized_fifo_pnl_inr']==210
    assert result['equity']-10000==210


def test_preopen_stock_tool_hides_open_and_future_history():
    store=ClockedStore(synthetic_bars(days=30),Clock(20))
    tb=ToolBox(store,Portfolio(1e6),lambda _:[],{},preopen=True)
    before=tb.inspect_stock({'asset':'STK_01'})
    assert before['today_open'] is None
    store.bars.loc[store.bars.sim_day>=20,['open','close','high','low','volume']]*=100
    assert tb.inspect_stock({'asset':'STK_01'})==before
    assert max(x['day'] for x in tb.get_history({'asset':'STK_01'})['bars'])==19


def test_reflection_proposals_do_not_mutate_policy_memory(tmp_path):
    class Changing(StubBackend):
        def run_episode(self,*a,**k):
            out=super().run_episode(*a,**k);v=json.loads(out.text)
            v.update(strategy_change='NEVER BUY AGAIN',belief_updates={'ban':'all stocks'},hypotheses=['ban'],perception_policy='only cash')
            out.text=json.dumps(v);return out
    cfg=copy.deepcopy(CFG);cfg['_intervention']=spec()
    db=StateDB(tmp_path/'state.sqlite');snap=AgentSnapshot('A1','p','stub',strategy='Fixed',beliefs={'keep':'fact'})
    db.save_agent(snap,0);run_reflection(snap,db,Changing(),cfg,10,'reflection')
    after=db.load_agent('A1');assert after.strategy=='Fixed' and after.beliefs=={'keep':'fact'}
    assert after.hypotheses==[] and after.strategy_version==1
    assert 'NEVER BUY AGAIN' in db.conn.execute('select reflection_json from reflections').fetchone()[0]
    db.conn.close()


def test_governed_day_uses_facts_and_preserves_resume(tmp_path):
    class Inspecting(StubBackend):
        def run_episode(self,system,user,tools,specs,*args,**kw):
            if 'Decision JSON schema' in user:
                assert 'OLD_HIDDEN_POLICY' not in user
                s=json.loads(user.split('Full-universe screen (engine supplied):\n')[1]);ctx=s['operational_policy']
                name=ctx['eligible_new'][0]
                assert tools['inspect_stock']({'asset':name})['today_open'] is None
                assert tools['search_memory']({'query':'OLD_HIDDEN_POLICY'})['results']==[]
                out=super().run_episode(system,user,tools,specs,*args,**kw);v=json.loads(out.text)
                v['orders']=[dict(asset=name,side='BUY',notional_inr=ctx['target_entry_inr'],reason='qualified')]
                v['candidate_assessments']=[dict(asset=name,disposition='act',reason='qualified')]
                out.text=json.dumps(v);return out
            return super().run_episode(system,user,tools,specs,*args,**kw)
    cfg=copy.deepcopy(CFG);cfg['_intervention']=spec();store=ClockedStore(synthetic_bars(days=30),Clock(20))
    db=StateDB(tmp_path/'state.sqlite');snap=AgentSnapshot('A1','p','stub',strategy='Fixed',age_days=19,
        beliefs={'ban':'OLD_HIDDEN_POLICY'},working_memory='OLD_HIDDEN_POLICY',long_term=['OLD_HIDDEN_POLICY'],episodic=[dict(sim_day=19,summary='OLD_HIDDEN_POLICY',lesson='ban')])
    result=run_agent_day(snap,store,db,Inspecting(),cfg,RiskRules.from_config(CFG['risk']),COSTS,None)
    assert result[0]['status']=='filled'
    assert len(snap.portfolio['positions'])==1
    before=list(db.conn.iterdump())
    run_agent_day(snap,store,db,Inspecting(),cfg,RiskRules.from_config(CFG['risk']),COSTS,None)
    assert list(db.conn.iterdump())==before
    r=json.loads(db.conn.execute('select context_json from research_context').fetchone()[0])
    assert not r['policy_audit']['eligible_idle']
    db.conn.close()


def test_second_intervention_preserves_first_and_refuses_rollback(tmp_path):
    db=StateDB(tmp_path/'state.sqlite');snap=AgentSnapshot('A1','p','stub',strategy='old',age_days=50);db.save_agent(snap,50)
    (tmp_path/'checkpoints/day-050').mkdir(parents=True);(tmp_path/'checkpoints/day-050/state.sqlite').touch()
    one=dict(approved=True,effective_day=51,agents={'A1':dict(strategy='one')});(tmp_path/'intervention-1.json').write_text(json.dumps(one))
    cfg={};load_intervention(tmp_path,db,cfg,51);snap=db.load_agent('A1');snap.age_days=100;db.save_agent(snap,100)
    (tmp_path/'checkpoints/day-100').mkdir();(tmp_path/'checkpoints/day-100/state.sqlite').touch()
    two=spec();two['effective_day']=101;(tmp_path/'intervention-2.json').write_text(json.dumps(two))
    load_intervention(tmp_path,db,cfg,101);assert cfg['_intervention']['id']=='intervention_2'
    assert db.load_agent('A1').strategy=='Fixed';assert json.loads(db.get_state('intervention:1'))==one
    count=db.load_agent('A1').strategy_version;load_intervention(tmp_path,db,cfg,102);assert db.load_agent('A1').strategy_version==count
    (tmp_path/'intervention-2.json').unlink()
    with pytest.raises(ValueError,match='missing'):load_intervention(tmp_path,db,cfg,102)
    db.conn.close()


def test_bounded_trade_feedback_keeps_all_time_accounting():
    pf = Portfolio(100000)
    for day in range(1,51,2):
        pf.apply_fill(Fill(day,'X','BUY',1,100,100,1,0,''))
        pf.apply_fill(Fill(day+1,'X','SELL',1,110,110,1,8,''))
    before=pf.to_dict()
    result=G.portfolio_feedback(pf,{})
    assert len(result['closed_trades_net_of_all_fees'])==20
    assert result['closed_trade_count_all_time']==25
    assert result['net_realized_fifo_pnl_inr']==200
    assert result['closed_trades_net_of_all_fees'][0]['sell_day']==12
    assert pf.to_dict()==before
