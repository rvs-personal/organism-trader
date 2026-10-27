import json
import pytest
from pydantic import BaseModel
from otrader.agents.schemas import AllocationPlan
from otrader.agents.loop import _run_structured
from otrader.inference.base import EpisodeResult,Usage

class Schema(BaseModel):
    allocation_plan: AllocationPlan

BAD={'allocation_plan':{'cash_reserve_inr':2907,'reserve_reason':'Rounding buffer'}}
GOOD={'allocation_plan':{**BAD['allocation_plan'],'reserve_release_condition':'Deploy when enough cash covers fees and one share','sizing_rationale':'Existing positions reflect the current evidence'}}

@pytest.mark.parametrize('native_schema',[False,True])
def test_all_backends_receive_original_response_and_each_missing_field(native_schema):
    class Backend:
        supports_json_schema=native_schema
        calls=0
        def run_episode(self,system,user,tools,specs,model,turns,**extra):
            self.calls+=1
            if self.calls==2:
                assert json.dumps(BAD) in user
                assert 'allocation_plan.reserve_release_condition' in user
                assert 'allocation_plan.sizing_rationale' in user
                assert 'complete corrected JSON object' in user
            return EpisodeResult(json.dumps(BAD if self.calls==1 else GOOD),Usage(10,5),model_id=model)
    events=[]
    result,usage,_,_= _run_structured(Backend(),'system','prompt',{}, {},'test',Schema,8,observe=events.append)
    assert result.allocation_plan.cash_reserve_inr==2907
    assert usage.input_tokens==20 and usage.output_tokens==10
    errors=next(e['errors'] for e in events if e['event']=='validation_retry')
    assert [e['field'] for e in errors]==['allocation_plan.reserve_release_condition','allocation_plan.sizing_rationale']


def test_invalid_responses_remain_rejected_and_retries_are_bounded():
    class Backend:
        calls=0
        def run_episode(self,*a,**kw):
            self.calls+=1
            return EpisodeResult(json.dumps(BAD),Usage(),model_id='test')
    backend=Backend();events=[]
    with pytest.raises(RuntimeError,match='failed twice'):
        _run_structured(backend,'s','u',{}, {},'test',Schema,8,observe=events.append)
    assert backend.calls==2
    assert events[-1]['event']=='validation_exhausted'
