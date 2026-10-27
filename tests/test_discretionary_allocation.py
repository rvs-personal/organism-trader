import copy,json
import pytest
import httpx
from otrader import governance as G
from otrader.agents.schemas import AllocationDecision
from otrader.agents.snapshot import StateDB,AgentSnapshot
from otrader.agents.loop import run_agent_day
from otrader.data.store import ClockedStore
from otrader.sim.clock import Clock
from otrader.sim.risk import RiskRules
from otrader.inference.stub_backend import StubBackend
from otrader.inference.ollama_backend import OllamaBackend
from conftest import synthetic_bars
from test_end_to_end_stub import CFG,COSTS


def config():
 c=copy.deepcopy(CFG);c['_intervention']=dict(policy_version=3,effective_day=1,id='discretion',allocation_mode='agent_discretion',protocol='Choose sizes',agents={'A1':dict(protocol='Choose sizes',policy=dict(entry=[],rank='ret_20d',descending=True,exit_style='discretionary'))});return c


def test_agent_can_choose_large_allocations_keep_cash_and_add_to_existing(tmp_path):
 class Allocator(StubBackend):
  def run_episode(self,system,user,tools,specs,*args,**kw):
   out=super().run_episode(system,user,tools,specs,*args,**kw)
   if 'Decision JSON schema' not in user:return out
   ctx=json.JSONDecoder().raw_decode(user.split('Full-universe screen (engine supplied):\n')[1])[0]['operational_policy']
   assert 'target_entry_inr' not in ctx and ctx['max_positions']==8
   assert 'OLD_POLICY' not in user
   names=ctx['eligible_buys'];pf=tools['inspect_portfolio']({})
   v=json.loads(out.text)
   if pf['positions']:
    name=pf['positions'][0]['asset'];v['orders']=[dict(asset=name,side='BUY',notional_inr=30000,reason='Increase evidence-backed position within25% cap')]
   else:
    v['orders']=[dict(asset=n,side='BUY',notional_inr=200000,reason='Choose20% independently') for n in names[:3]]
   v['candidate_assessments']=[]
   v['allocation_plan']=dict(cash_reserve_inr=400000,reserve_reason='Keep liquidity for candidates after confirmation',reserve_release_condition='Price and volume confirm the thesis',sizing_rationale='Three20% allocations; remaining cash conditional, not a rule')
   if pf['positions']: v['allocation_plan']['cash_reserve_inr']=pf['cash']-30000
   out.text=json.dumps(v);return out
 c=config();store=ClockedStore(synthetic_bars(days=30),Clock(20));db=StateDB(tmp_path/'state.sqlite');snap=AgentSnapshot('A1','p','stub',strategy='Research style',age_days=19,beliefs={'old':'OLD_POLICY'})
 res=run_agent_day(snap,store,db,Allocator(),c,RiskRules.from_config(c['risk']),COSTS,None)
 assert all(x['status']=='filled' for x in res) and len(snap.portfolio['positions'])==3
 assert snap.portfolio['cash']<450000
 original=next(iter(snap.portfolio['positions'].values()))['qty'];store.clock.set(21)
 res=run_agent_day(snap,store,db,Allocator(),c,RiskRules.from_config(c['risk']),COSTS,None)
 assert res[0]['status']=='filled'
 assert next(iter(snap.portfolio['positions'].values()))['qty']>original
 assert json.loads(db.conn.execute('select context_json from research_context where sim_day=20').fetchone()[0])['allocation_outcome']['planned_cash_reserve_inr']==400000
 db.conn.close()


def test_local_native_schema_and_no_paid_fallback():
 with pytest.raises(ValueError):OllamaBackend(base_url='https://example.com')
 b=OllamaBackend()
 with pytest.raises(ValueError):b.run_episode('','',{}, {},'model:cloud',0)
 def handler(request):
  body=json.loads(request.content);assert body['think'] is False and body['format']=={'type':'object'}
  assert 'Authorization' not in request.headers
  return httpx.Response(200,json={'message':{'role':'assistant','content':'{"ok":true}'},'prompt_eval_count':100,'eval_count':10,'done_reason':'stop'})
 b.client=httpx.Client(transport=httpx.MockTransport(handler))
 r=b.run_episode('system','user',{}, {},'test',0,output_schema={'type':'object'})
 assert json.loads(r.text)=={'ok':True} and r.usage.cost_inr==0


def test_eight_agent_pairs_have_equal_expertise_and_free_allocation():
 from otrader.config import ROOT,load_experiment
 c=load_experiment(ROOT/'config/experiment.yaml')
 assert len(c['agents'])==8 and c['risk']['max_position_frac']==1
 assert c['risk']['max_positions']==29 and c['capital']['death_threshold_frac']==0
 for normal,pressure in zip(c['agents'][:4],c['agents'][4:]):
  assert normal['persona']==pressure['persona']
  assert normal['initial_strategy']==pressure['initial_strategy']
  assert pressure['paired_with']==normal['id']
  policies=c['initial_policy']['agents']
  assert policies[normal['id']]['policy']==policies[pressure['id']]['policy']
  assert 'PERFORMANCE-PRESSURE' not in policies[normal['id']]['protocol']
  assert 'PERFORMANCE-PRESSURE' in policies[pressure['id']]['protocol']


def test_local_episodes_do_not_share_messages_and_tools_follow_up():
 b=OllamaBackend();requests=[]
 def handler(request):
  body=json.loads(request.content);requests.append(body)
  if len(requests)==1:
   return httpx.Response(200,json={'message':{'role':'assistant','content':'','tool_calls':[{'function':{'name':'lookup','arguments':{'asset':'STK_01'}}}]},'done_reason':'stop'})
  return httpx.Response(200,json={'message':{'role':'assistant','content':'{"ok":true}'},'done_reason':'stop'})
 b.client.close();b.client=httpx.Client(transport=httpx.MockTransport(handler))
 spec={'lookup':{'description':'Read own data','input_schema':{'type':'object'}}}
 r=b.run_episode('identity A1','private A1',{'lookup':lambda args:{'own':'A1'}},spec,'test',1,{'type':'object'})
 b.run_episode('identity A5','private A5',{}, {},'test',0,{'type':'object'})
 assert r.tool_calls[0]['tool']=='lookup'
 assert requests[1]['messages'][-1]['role']=='tool'
 assert len(requests[2]['messages'])==2
 assert 'A1' not in json.dumps(requests[2]['messages'])
 b.client.close()


def test_false_deployment_without_orders_is_retried_and_never_saved(tmp_path):
 class EmptyPlan(StubBackend):
  attempts=0
  def run_episode(self,system,user,tools,specs,*args,**kw):
   self.attempts+=1
   out=super().run_episode(system,user,tools,specs,*args,**kw)
   out.text=json.dumps(dict(market_view='Buy in prose only',orders=[],confidence=.5,candidate_assessments=[],allocation_plan=dict(cash_reserve_inr=100000,reserve_reason='Claim deployed',reserve_release_condition='Wait',sizing_rationale='Claim 90% invested')))
   return out
 c=config();db=StateDB(tmp_path/'state.sqlite');backend=EmptyPlan()
 with pytest.raises(RuntimeError,match='Empty orders means no trades'):
  run_agent_day(AgentSnapshot('A1','p','stub'),ClockedStore(synthetic_bars(days=30),Clock(20)),db,backend,c,RiskRules.from_config(c['risk']),COSTS,None)
 assert backend.attempts==2
 assert db.conn.execute('select count(*) from decisions').fetchone()[0]==0
 db.conn.close()


def test_preview_uses_cash_and_known_marks_without_mutating_portfolio():
 from otrader.agents.allocation import preview_allocation
 from otrader.sim.portfolio import Portfolio,Position
 pf=Portfolio(cash=100000);pf.positions['STK_01']=Position('STK_01',100,100,1)
 before=copy.deepcopy(pf.to_dict())
 p=preview_allocation([dict(asset='STK_01',side='SELL',qty=50),dict(asset='STK_02',side='BUY',notional_inr=20000)],pf,{'STK_01':110,'STK_02':200},COSTS)
 expected=100000+5500-COSTS.costs(5500,'SELL')-20000-COSTS.costs(20000,'BUY')
 assert p['estimated_cash_reserve_inr']==round(expected,2)
 assert pf.to_dict()==before and p['affordable_in_order']
 with pytest.raises(ValueError,match='hold'):
  preview_allocation([dict(asset='STK_01',side='SELL',qty=101)],pf,{'STK_01':110},COSTS)


def test_json_research_mode_preserves_tools_for_completion_only_models():
 b=OllamaBackend(tool_mode='json');requests=[]
 def handler(request):
  body=json.loads(request.content);requests.append(body);assert 'tools' not in body
  content='{"tool_requests":[{"name":"lookup","arguments":{"asset":"STK_01"}}]}' if len(requests)==1 else '{"ok":true}'
  return httpx.Response(200,json={'message':{'role':'assistant','content':content},'done_reason':'stop'})
 b.client.close();b.client=httpx.Client(transport=httpx.MockTransport(handler))
 spec={'lookup':{'description':'Look up own causal data','input_schema':{'type':'object','properties':{'asset':{'type':'string'}}}}}
 r=b.run_episode('agent A1','private context',{'lookup':lambda args:{'last_close':100}},spec,'test',1,{'type':'object'})
 assert r.tool_calls==[{'tool':'lookup','args':{'asset':'STK_01'}}]
 assert 'Python tool result for lookup' in requests[1]['messages'][-1]['content']
 assert requests[1]['format']=={'type':'object'}
 b.client.close()


def test_stub_supports_governed_allocation_without_model_calls(tmp_path):
 c=config(); store=ClockedStore(synthetic_bars(days=30),Clock(20))
 db=StateDB(tmp_path/'state.sqlite'); snap=AgentSnapshot('A1','demo','stub',age_days=19)
 run_agent_day(snap,store,db,StubBackend(),c,RiskRules.from_config(c['risk']),COSTS,None)
 decision=json.loads(db.conn.execute('select decision_json from decisions').fetchone()[0])
 assert decision['orders']
 assert decision['allocation_plan']['cash_reserve_inr'] >= 0
 assert decision['candidate_assessments']
 assert snap.age_days==20
 db.close() if hasattr(db,'close') else db.conn.close()
