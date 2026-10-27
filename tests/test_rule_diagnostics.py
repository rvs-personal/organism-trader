from copy import deepcopy
import pytest
from otrader.rule_diagnostics import failures, exits, outcome_probe, replay
from otrader.sim.execution import CostModel
from otrader.sim.portfolio import Portfolio

COST=dict(brokerage_per_order=20,stt_delivery_frac=.001,exchange_txn_frac=.0000325,gst_on_brokerage_frac=.18,stamp_duty_buy_frac=.00015,slippage_frac=.0005)

def test_outcome_probe_requires_mature_horizon_and_accounts_for_both_fees():
    bars={('S',151):{'open':100},('S',156):{'open':110}}
    cost=CostModel(**COST)
    assert outcome_probe(151,'S',5,bars,cost,10000,155) is None
    x=outcome_probe(151,'S',5,bars,cost,10000,156)
    assert x['qty']==99
    basis=99*100.05+cost.costs(99*100.05,'BUY')
    assert x['purchase_outlay']==pytest.approx(basis)
    assert x['pnl']==pytest.approx(round(99*109.95-cost.costs(99*109.95,'SELL')-basis,2))

def test_independent_and_overlapping_predicates():
    p={'entry':[['rsi_14','lt',40],['volume_ratio_5_20','gt',1.3]]}
    assert len(failures({'rsi_14':50,'volume_ratio_5_20':1},p))==2
    assert len(failures({'rsi_14':None,'volume_ratio_5_20':1.4},p))==1
    r=dict(px_over_sma50=1.1,rel_strength_20d_vs_index=.1,ret_5d=-.01)
    assert exits(r,10,'experiment')==['10-session horizon','5-session negative momentum']

def test_replay_uses_fresh_boundary_and_separate_state_without_future_prices():
    p=dict(entry=[['ret_20d','gt',0]],entry_fraction=.1,max_positions=1,rank='ret_20d',descending=True,exit_style='trend')
    row=dict(asset='S',sector='sector',ret_20d=-.01,px_over_sma20=1.1,rel_strength_20d_vs_index=.1,rsi_14=50,ret_5d=.01)
    screens={('A1',151):dict(as_of_day=150,candidates=[row],market={'index':{}})}
    bars={('S',150):dict(open=100,close=100),('S',151):dict(open=100,close=110),('S',152):dict(open=10000,close=10000)}
    cfg=dict(risk=dict(max_positions=8,max_position_frac=.25,max_gross_exposure_frac=1,no_shorting=True,max_orders_per_day=5),capital=dict(starting_cash=100000,death_threshold_frac=.7),costs=COST)
    initial=Portfolio(100000);before=deepcopy(p)
    control=replay('A1',p,{'id':'control','kind':'control'},initial,screens,bars,cfg,151,start_day=151)
    alternative=replay('A1',p,{'id':'remove','kind':'entry','predicate':['ret_20d','gt',0]},initial,screens,bars,cfg,151,start_day=151)
    assert control['periods']['151-151']['return_pct']==0
    assert alternative['periods']['151-151']['return_pct']>0
    assert len(alternative['fills'])==1 and alternative['fills'][0]['sim_day']==151
    assert [r['day'] for r in alternative['curve']]==[150,151]
    assert initial.cash==100000 and not initial.positions and p==before
    assert alternative['curve'][-1]['equity']<101000  # no use of day152's large future rise
