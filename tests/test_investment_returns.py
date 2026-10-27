import pandas as pd
import pytest
from otrader.dashboard.analytics import investment_returns
from otrader.dashboard.charts import investment_ranking


def test_investment_return_partial_sales_open_losses_and_future_cutoff():
    f=pd.DataFrame([
        dict(owner='A1',sim_day=1,asset='X',side='BUY',qty=10,price=100.,gross=1000.,costs=10.),
        dict(owner='A1',sim_day=2,asset='X',side='BUY',qty=10,price=120.,gross=1200.,costs=10.),
        dict(owner='A1',sim_day=3,asset='X',side='SELL',qty=15,price=130.,gross=1950.,costs=15.),
        dict(owner='A1',sim_day=4,asset='X',side='SELL',qty=5,price=999.,gross=4995.,costs=5.)])
    e=pd.DataFrame([dict(owner='A1',sim_day=3,equity=10165.,cash=9715.),
                    dict(owner='A1',sim_day=4,equity=14705.,cash=14705.)])
    r=investment_returns(f,e,3,10000,['A1']).iloc[0]
    assert r.deployed==2220
    assert r.closed_basis==1615
    assert r.closed_net==320
    assert r.open_basis==605
    assert r.open_net==-155
    assert r.net_pnl==165
    assert r.deployed_return==pytest.approx(165/2220)
    assert r.closed_return==pytest.approx(320/1615)
    assert r.open_positions==1 and r.sell_orders==1
    fig=investment_ranking(pd.DataFrame([r]))
    assert list(fig.data[0].y)==['Atlas']
    assert fig.data[0].x[0]==pytest.approx(165/2220)
    assert fig.data[0].customdata[0][0]==2220


def test_cash_has_no_investment_return_and_buy_hold_is_included():
    f=pd.DataFrame([dict(owner='B_INDEX',sim_day=1,asset='X',side='BUY',qty=10,price=100.,gross=1000.,costs=10.)])
    e=pd.DataFrame([dict(owner='B_INDEX',sim_day=1,equity=10090.,cash=8990.),dict(owner='B_CASH',sim_day=1,equity=10000.,cash=10000.)])
    r=investment_returns(f,e,1,10000,['B_INDEX','B_CASH']).set_index('owner')
    assert r.loc['B_INDEX','deployed_return']==pytest.approx(90/1010)
    assert pd.isna(r.loc['B_INDEX','closed_return'])
    assert pd.isna(r.loc['B_CASH','deployed_return'])
    assert len(investment_ranking(r.reset_index(),True).data[0].x)==0


def test_recycled_cash_counts_each_purchase_and_invalid_ledger_fails():
    f=pd.DataFrame([dict(owner='A1',sim_day=d,asset='X',side=s,qty=10,price=p,gross=10*p,costs=0.)
                    for d,s,p in [(1,'BUY',100),(2,'SELL',110),(3,'BUY',100),(4,'SELL',110)]])
    e=pd.DataFrame([dict(owner='A1',sim_day=4,equity=10200.,cash=10200.)])
    r=investment_returns(f,e,4,10000,['A1']).iloc[0]
    assert r.deployed==2000 and r.net_pnl==200
    assert r.deployed_return==pytest.approx(.1)
    assert r.portfolio_return==pytest.approx(.02)
    e.loc[0,'equity']=99999
    with pytest.raises(ValueError,match='reconcile'):investment_returns(f,e,4,10000,['A1'])
    with pytest.raises(ValueError,match='Incomplete FIFO'):investment_returns(f.iloc[1:],e,4,10000,['A1'])
