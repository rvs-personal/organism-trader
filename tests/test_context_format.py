import json
import pytest
from conftest import synthetic_bars
from otrader.agents.context_format import research_json, observer
from otrader.agents.tools import ToolBox
from otrader.agents.snapshot import StateDB
from otrader.data.store import ClockedStore
from otrader.sim.clock import Clock
from otrader.sim.portfolio import Portfolio


def test_compact_screen_retains_every_value_and_asof():
    tb = ToolBox(ClockedStore(synthetic_bars(days=30), Clock(20)), Portfolio(1e6), lambda _: [], {}, preopen=True)
    original = tb.screen_universe()
    compact = json.loads(research_json(original, True))
    table = compact['candidates']
    compact['candidates'] = [dict(zip(table['columns'], row)) for row in table['rows']]
    assert compact == original


def test_batch_matches_individuals_and_cannot_see_future():
    store = ClockedStore(synthetic_bars(days=30), Clock(20))
    tb = ToolBox(store, Portfolio(1e6), lambda _: [], {}, preopen=True)
    ids = ['STK_01', 'STK_02']
    result = tb.inspect_stocks({'assets': ids})
    assert [dict(zip(result['columns'], row)) for row in result['rows']] == [tb.inspect_stock({'asset': a}) for a in ids]
    store.bars.loc[store.bars.sim_day >= 20, ['open','high','low','close','volume']] *= 100
    assert tb.inspect_stocks({'assets': ids}) == result
    assert tb.inspect_stocks({'assets':['STK_01','UNKNOWN']})['results'][1] == {'error':'unknown asset UNKNOWN'}
    with pytest.raises(ValueError):
        tb.inspect_stocks({'assets': []})


def test_usage_events_preserve_attempts_and_agent_identity(tmp_path):
    db = StateDB(tmp_path/'test.sqlite')
    for agent in ['A1','A5']:
        write = observer(db, agent, 1, 'decision', True)
        write({'event':'response','attempt':1,'input_tokens':100,'cache_read_tokens':40,'cost_inr':None})
        write({'event':'validation_retry','attempt':1})
        write({'event':'response','attempt':2,'input_tokens':120,'cache_read_tokens':0,'cost_inr':None})
    rows = db.conn.execute('SELECT agent_id,event_json FROM inference_events').fetchall()
    assert len(rows) == 6
    assert sum(json.loads(v).get('input_tokens',0) for a,v in rows if a=='A1') == 220
    assert all(json.loads(v).get('cost_inr') is None for a,v in rows)
    db.conn.close()
