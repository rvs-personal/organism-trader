import copy
import json
import pytest
from otrader.agents.loop import _run_structured, run_agent_day
from otrader.inference.retry import EpisodeRetriesExhausted
from otrader.inference.base import EpisodeResult, Usage
from otrader.inference.stub_backend import StubBackend
from otrader.agents.snapshot import AgentSnapshot, StateDB
from otrader.sim.clock import Clock
from otrader.data.store import ClockedStore
from otrader.sim.risk import RiskRules
from conftest import synthetic_bars
from test_end_to_end_stub import CFG, COSTS
from test_structured_repair import Schema, BAD, GOOD

class Flaky:
    name='cursor'
    def __init__(self, responses): self.responses=iter(responses);self.calls=[]
    def run_episode(self,*a,**kw):
        self.calls.append((a,kw));value=next(self.responses)
        if isinstance(value,Exception):raise value
        return EpisodeResult(json.dumps(value),Usage(10,5),model_id='test')

def call(backend,events):
    return _run_structured(backend,'system','user',{}, {},'test',Schema,8,observe=events.append)

@pytest.mark.parametrize('error',[RuntimeError('Cursor run error: '),TimeoutError('timed out'),RuntimeError('503 service unavailable')])
def test_retries_exact_phase_prompt_with_recorded_backoff(monkeypatch,error):
    delays=[];monkeypatch.setattr('otrader.agents.loop.time.sleep',delays.append)
    b=Flaky([error,GOOD]);events=[]
    _,usage,_,_=call(b,events)
    assert b.calls[0]==b.calls[1] and delays==[5]
    assert usage.input_tokens==10  # failed request usage unknown, never invented
    assert next(e for e in events if e['event']=='episode_retry')['failed_call_usage']=='unknown'

@pytest.mark.parametrize('message',['429 rate limit','429 usage limit','quota exceeded','Cursor reported a paid charge','401 unauthorized timeout','invalid request 503','context length timeout','Cursor run error: unsupported model'])
def test_terminal_and_rate_errors_are_not_retried(monkeypatch,message):
    monkeypatch.setattr('otrader.agents.loop.time.sleep',lambda _:pytest.fail('must not wait'))
    b=Flaky([RuntimeError(message)]);
    with pytest.raises(RuntimeError,match='.'):
        call(b,[])
    assert len(b.calls)==1

def test_retry_exhaustion_does_not_trigger_outer_day_retry(monkeypatch):
    from otrader.parallel import transient,quota
    delays=[];monkeypatch.setattr('otrader.agents.loop.time.sleep',delays.append)
    b=Flaky([TimeoutError()]*3);events=[]
    with pytest.raises(EpisodeRetriesExhausted) as e:call(b,events)
    assert not transient(e.value) and not quota(e.value)
    assert len(b.calls)==3 and delays==[5,15]
    assert events[-1]['event']=='episode_retry_exhausted'

def test_validation_repair_shares_bounded_recovery_budget(monkeypatch):
    monkeypatch.setattr('otrader.agents.loop.time.sleep',lambda _:None)
    b=Flaky([TimeoutError(),BAD,TimeoutError(),TimeoutError()])
    with pytest.raises(EpisodeRetriesExhausted):call(b,[])
    assert len(b.calls)==4 and b.calls[2]==b.calls[3]
    assert 'VALIDATION REPAIR' in b.calls[2][0][1]

def test_memory_retry_does_not_repeat_decision_or_duplicate_trade(tmp_path,monkeypatch):
    monkeypatch.setattr('otrader.agents.loop.time.sleep',lambda _:None)
    class MemoryFlaky(StubBackend):
        name='cursor'
        decisions=0;memories=0
        def run_episode(self,system,user,*a,**kw):
            if 'Decision JSON schema' in user:self.decisions+=1
            elif 'episodic memory entry' in user:
                self.memories+=1
                if self.memories==1:raise RuntimeError('Cursor run error: ')
            return super().run_episode(system,user,*a,**kw)
    store=ClockedStore(synthetic_bars(),Clock(1));rules=RiskRules.from_config(CFG['risk'])
    db=StateDB(tmp_path/'retry.sqlite');normal=StateDB(tmp_path/'normal.sqlite')
    b=MemoryFlaky();snap=AgentSnapshot('A1','test','stub')
    run_agent_day(snap,store,db,b,copy.deepcopy(CFG),rules,COSTS,None)
    run_agent_day(AgentSnapshot('A1','test','stub'),store,normal,StubBackend(),copy.deepcopy(CFG),rules,COSTS,None)
    assert b.decisions==1 and b.memories==2
    assert db.conn.execute('select * from fills').fetchall()==normal.conn.execute('select * from fills').fetchall()
    assert db.load_agent('A1').portfolio==normal.load_agent('A1').portfolio
    assert len(db.load_agent('A1').episodic)==1
    assert db.conn.execute('select count(*) from decisions').fetchone()[0]==1
    db.conn.close();normal.conn.close()


def test_tool_budget_gets_one_retry_without_relaxing_limit(monkeypatch):
    monkeypatch.setattr('otrader.agents.loop.time.sleep',lambda _:None)
    error=RuntimeError('Cursor episode exceeded the configured tool-call budget.')
    b=Flaky([error,GOOD]);call(b,[])
    assert len(b.calls)==2 and b.calls[0]==b.calls[1] and b.calls[1][0][5]==8
    b=Flaky([error,error])
    with pytest.raises(EpisodeRetriesExhausted):call(b,[])
    assert len(b.calls)==2
