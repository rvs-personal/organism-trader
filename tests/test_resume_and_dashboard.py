import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
from conftest import synthetic_bars
from test_end_to_end_stub import CFG, COSTS
from otrader import runner
from otrader.agents.snapshot import StateDB, AgentSnapshot
from otrader.agents.loop import run_agent_day, run_reflection
from otrader.inference.stub_backend import StubBackend
from otrader.inference.cursor_backend import CursorBackend
from otrader.data.store import ClockedStore
from otrader.sim.clock import Clock
from otrader.sim.risk import RiskRules
from otrader.sim.metrics import summarize
from otrader.dashboard.analytics import fifo_matches, sector_pnl


def dump(db):
    return '\n'.join(db.conn.iterdump())


def test_replay_is_identical_and_transaction_rolls_back(tmp_path):
    db = StateDB(tmp_path/'state.sqlite')
    store = ClockedStore(synthetic_bars(n_assets=40), Clock(1))
    rules = RiskRules.from_config(CFG['risk'])
    snap = AgentSnapshot('A1', 'test', 'stub')
    run_agent_day(snap, store, db, StubBackend(), CFG, rules, COSTS, None)
    runner.run_baseline_day('B_EQUAL', store, db, CFG, rules, COSTS)
    before = dump(db)
    run_agent_day(db.load_agent('A1'), store, db, StubBackend(), CFG, rules, COSTS, None)
    runner.run_baseline_day('B_EQUAL', store, db, CFG, rules, COSTS)
    assert dump(db) == before
    assert db.conn.execute("SELECT COUNT(*) FROM fills WHERE owner='B_EQUAL'").fetchone()[0] == 40
    with pytest.raises(RuntimeError), db.transaction():
        db.set_state('test', 'partial')
        db.save_equity('test', 1, 0, 0, 0)
        raise RuntimeError('crash')
    assert dump(db) == before


def setup_runner(tmp_path, monkeypatch, backend):
    cfg = copy.deepcopy(CFG)
    cfg.update(experiment_id='test', n_sim_days=12, costs=COSTS.__dict__,
               agents=[{'id': 'A1', 'persona': 'test'}, {'id': 'A2', 'persona': 'test'}])
    cfg['evolution']['reflection_every_days'] = 1
    anon = tmp_path/'data'/'anon'/'test'
    anon.mkdir(parents=True)
    synthetic_bars(days=12).to_parquet(anon/'bars.parquet')
    monkeypatch.setattr(runner, 'DATA_DIR', tmp_path/'data')
    monkeypatch.setattr(runner, 'RUNS_DIR', tmp_path/'runs')
    monkeypatch.setattr(runner, 'load_experiment', lambda: cfg)
    monkeypatch.setattr(runner, 'make_backend', lambda *a, **k: backend)
    return tmp_path/'runs'/'test'/'state.sqlite'


def test_resume_reflection_without_retrading(tmp_path, monkeypatch):
    class Limited(StubBackend):
        failed = False
        def run_episode(self, system, user, *args, **kwargs):
            if 'Decision JSON schema' not in user and 'episodic memory entry' not in user and not self.failed:
                self.failed = True
                raise RuntimeError('429 usage limit')
            return super().run_episode(system, user, *args, **kwargs)
    path = setup_runner(tmp_path, monkeypatch, Limited())
    assert runner.main(['--days', '2', '--backend', 'stub']) == 3
    db = StateDB(path)
    assert db.load_agent('A1').age_days == 1
    fills = db.conn.execute('SELECT * FROM fills WHERE sim_day=1').fetchall()
    db.conn.close()
    assert runner.main(['--days', '2', '--backend', 'stub']) == 0
    db = StateDB(path)
    assert db.load_agent('A1').age_days == 2
    assert db.load_agent('A2').age_days == 2
    assert db.get_state('clock') == '3'
    assert db.conn.execute('SELECT COUNT(*) FROM reflections').fetchone()[0] == 4
    assert db.conn.execute("SELECT * FROM fills WHERE sim_day=1 AND owner != 'A2'").fetchall() == fills
    db.conn.close()


def test_wait_retries_same_target_days(tmp_path, monkeypatch):
    class Limited(StubBackend):
        failed = False
        def run_episode(self, *a, **k):
            if not self.failed:
                self.failed = True
                raise RuntimeError('rate_limit')
            return super().run_episode(*a, **k)
    path = setup_runner(tmp_path, monkeypatch, Limited())
    sleeps = []
    monkeypatch.setattr(runner.time, 'sleep', sleeps.append)
    assert runner.main(['--days','2','--backend','stub','--wait-for-reset','--retry-seconds','60']) == 0
    db = StateDB(path)
    assert sleeps == [60]
    assert db.get_state('clock') == '3'
    assert db.load_agent('A1').age_days == 2
    db.conn.close()


def test_day_one_metrics_include_initial_loss():
    m = summarize(pd.Series([90.], index=[1]), [], 100, pd.Series([95.], index=[1]))
    assert m['max_drawdown'] == -.1
    assert m['total_return'] == -.1
    assert m['benchmark_return'] == -.05
    assert m['excess_return'] == -.05


def test_fifo_partial_sales_and_sector_fees():
    fills = pd.DataFrame([
        dict(owner='A', asset='STK_01', sim_day=1, side='BUY', qty=10, price=100., gross=1000., costs=10.),
        dict(owner='A', asset='STK_01', sim_day=2, side='BUY', qty=10, price=120., gross=1200., costs=10.),
        dict(owner='A', asset='STK_01', sim_day=5, side='SELL', qty=15, price=130., gross=1950., costs=15.)])
    matches = fifo_matches(fills)
    assert matches.holding_days.tolist() == [4,3]
    assert matches.qty.tolist() == [10,5]
    assert matches.net_pnl.sum() == 320
    bars = pd.DataFrame([dict(asset='STK_01', sector='SECTOR_1', sim_day=5, close=130),
                         dict(asset='STK_01', sector='SECTOR_1', sim_day=6, close=999)])
    assert sector_pnl(fills, bars, 5).net_pnl.sum() == 365


def test_cursor_rejects_unconfirmed_billing(monkeypatch):
    monkeypatch.setenv('CURSOR_API_KEY', 'test-key')
    with pytest.raises(ValueError, match='on-demand'):
        CursorBackend()


def test_cursor_catalog_and_model_policy(monkeypatch):
    monkeypatch.setenv('CURSOR_API_KEY', 'test-key')
    backend = CursorBackend(on_demand_disabled_confirmed=True, effort='medium')
    from cursor_sdk.types import SDKModel, ModelParameterDefinition, ModelParameterDefinitionValue
    backend._catalog = {'grok-4.6': SDKModel(id='grok-4.6', display_name='test', parameters=[
        ModelParameterDefinition('effort', values=[ModelParameterDefinitionValue('medium')])])}
    selection = backend._model_selection('grok-4.6')
    assert selection.params[0].value == 'medium'
    with pytest.raises(ValueError, match='allowlist'):
        backend._model_selection('claude-sonnet-5')
    with pytest.raises(ValueError, match='fixed'):
        backend._model_selection('auto')


def test_dashboard_empty_and_day_one(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import otrader.config as config
    path = setup_runner(tmp_path, monkeypatch, StubBackend())
    monkeypatch.setattr(config, 'RUNS_DIR', tmp_path/'runs')
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path/'data')
    monkeypatch.setattr(config, 'load_experiment', runner.load_experiment)
    app = Path(__file__).parents[1]/'src'/'otrader'/'dashboard'/'app.py'
    test = AppTest.from_file(str(app)).run(timeout=20)
    assert not test.exception
    StateDB(path).conn.close()
    test = AppTest.from_file(str(app)).run(timeout=20)
    assert not test.exception
    assert runner.main(['--days','1','--backend','stub']) == 0
    test = AppTest.from_file(str(app)).run(timeout=20)
    assert not test.exception


def test_zero_sell_cannot_liquidate_portfolio():
    from otrader.sim.portfolio import Portfolio, Position
    from otrader.sim.execution import execute_orders, Order
    pf = Portfolio(cash=1000, positions={'STK_01': Position('STK_01', 10, 100, 1)})
    result = execute_orders([Order('STK_01', 'SELL', 0, None)], pf, {'STK_01':100}, {'STK_01':100},
                            RiskRules.from_config(CFG['risk']), COSTS, 2)
    assert result[0]['status'] == 'rejected'
    assert pf.positions['STK_01'].qty == 10


def test_future_assets_and_sectors_are_hidden():
    bars = synthetic_bars(days=12)
    future = bars[bars.sim_day == 11].iloc[[0]].copy()
    future['asset'], future['sector'] = 'STK_99', 'SECTOR_99'
    store = ClockedStore(pd.concat([bars, future]), Clock(10))
    assert 'STK_99' not in store.assets()
    assert 'STK_99' not in store.sectors()


def test_resume_rejects_replaced_data(tmp_path, monkeypatch):
    path = setup_runner(tmp_path, monkeypatch, StubBackend())
    assert runner.main(['--days','1','--backend','stub']) == 0
    synthetic_bars(days=12, seed=42).to_parquet(tmp_path/'data'/'anon'/'test'/'bars.parquet')
    assert runner.main(['--days','1','--backend','stub']) == 2
    db = StateDB(path)
    assert db.get_state('clock') == '2'
    assert json.loads(db.get_state('run_config'))['capital']['starting_cash'] == 1_000_000
    db.conn.close()


def test_strategy_changes_lock_and_reflection_idempotency(tmp_path):
    class Changing(StubBackend):
        def run_episode(self, *a, **k):
            res = super().run_episode(*a, **k)
            body = json.loads(res.text)
            body['strategy_change'] = 'New test strategy'
            res.text = json.dumps(body)
            return res
    db = StateDB(tmp_path/'state.sqlite')
    snap = AgentSnapshot('A1','test','stub',strategy='Original')
    db.save_agent(snap,0)
    run_reflection(snap,db,Changing(),CFG,10,'reflection')
    assert db.load_agent('A1').strategy_version == 2
    before = dump(db)
    run_reflection(db.load_agent('A1'),db,Changing(),CFG,10,'reflection')
    assert dump(db) == before
    run_reflection(db.load_agent('A1'),db,Changing(),CFG,170,'reflection')
    assert db.load_agent('A1').strategy_version == 2
    db.conn.close()


def test_personas_have_distinct_starting_strategies(tmp_path):
    from otrader.config import load_experiment
    cfg = load_experiment()
    db = StateDB(tmp_path/'state.sqlite')
    runner.bootstrap_agents(db, cfg)
    snapshots = [db.load_agent(a['id']) for a in cfg['agents']]
    assert len({s.strategy for s in snapshots}) == 4
    assert all(s.strategy == a['initial_strategy'] for s, a in zip(snapshots, cfg['agents']))
    before = dump(db)
    runner.bootstrap_agents(db, cfg)
    assert dump(db) == before
    db.conn.close()


def test_cursor_exposes_only_custom_tools_and_normalizes_trace(monkeypatch):
    import cursor_sdk
    monkeypatch.setenv('CURSOR_API_KEY', 'test-key')
    backend = CursorBackend(on_demand_disabled_confirmed=True)
    monkeypatch.setattr(backend, '_model_selection', lambda mid: mid)
    captured = {}
    class Run:
        def messages(self):
            yield SimpleNamespace(type='tool_call', status='running', name='mcp',
                                  args={'providerIdentifier':'custom-user-tools','toolName':'ping','args':{'n':41}})
        def wait(self):
            return SimpleNamespace(status='finished',result='{"answer":42}',usage=None)
    class Agent:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def send(self, prompt): return Run()
        def get_usage(self): return SimpleNamespace(cost=None)
    def create(options):
        captured['options'] = options
        return Agent()
    monkeypatch.setattr(cursor_sdk.Agent, 'create', create)
    res = backend.run_episode('system','user',{'ping':lambda a:{'pong':42}},
                              {'ping':{'description':'ping','input_schema':{'type':'object'}}},'grok-4.6',2)
    opts = captured['options']
    assert opts.tools == ['mcp']
    assert opts.mcp_servers == {} and opts.local.setting_sources == []
    assert opts.local.store.type == 'jsonl'
    assert res.tool_calls == [{'tool':'ping','args':{'n':41}}]
    assert res.usage.cost_inr is None
