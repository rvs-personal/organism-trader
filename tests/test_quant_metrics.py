import numpy as np
import pandas as pd
import pytest
from otrader.dashboard.quant_metrics import aligned_values, performance_metrics, trading_metrics


def test_alignment_rebases_subwindow_and_rejects_missing_days_and_duplicates():
    eq = pd.DataFrame({'owner':['A']*4+['B']*2+['C']*3,
        'sim_day':[1,2,3,4,1,3,1,2,2], 'equity':[100,110,99,999,100,105,100,100,100]})
    v, excluded = aligned_values(eq,['A','B','C'],2,3,100)
    assert list(v.index)==[1,2,3]
    assert list(v)==['A']
    assert set(excluded)=={'B','C'}
    assert performance_metrics(v,'A').loc['A','total_return']==pytest.approx(-.01)


def test_initial_cash_and_partial_day_exclusion():
    eq = pd.DataFrame({'owner':['A','A'],'sim_day':[1,2],'equity':[102,900]})
    v,_=aligned_values(eq,['A'],1,1,100)
    assert list(v.A)==[100,102]
    m=performance_metrics(v,'A').loc['A']
    assert m.total_return==pytest.approx(.02)
    assert pd.isna(m.sharpe)


def test_known_ratios_drawdown_and_cash_denominators():
    r=np.array([.01,-.02,.03,-.005]*10)
    b=r/2
    values=pd.DataFrame({'A':np.r_[100,100*np.cumprod(1+r)],
        'B':np.r_[100,100*np.cumprod(1+b)],'Cash':np.full(41,100.)})
    m=performance_metrics(values,'B')
    assert m.loc['A','sharpe']==pytest.approx(r.mean()/r.std(ddof=1)*np.sqrt(252))
    assert m.loc['A','sortino']==pytest.approx(r.mean()/np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(252))
    assert m.loc['A','beta']==pytest.approx(2)
    assert m.loc['A','max_drawdown']==pytest.approx(-.02)
    assert pd.isna(m.loc['Cash','sharpe'])
    assert pd.isna(m.loc['Cash','sortino'])
    assert pd.isna(m.loc['B','information_ratio'])
    assert performance_metrics(values,'B',.08).loc['A','sharpe'] < m.loc['A','sharpe']
    assert pd.isna(performance_metrics(values,'Cash').loc['A','beta'])


def test_fees_exposure_and_turnover_only_selected_window():
    eq=pd.DataFrame({'owner':['A']*2,'sim_day':[1,2],'equity':[100,110],'cash':[50,22]})
    f=pd.DataFrame({'owner':['A']*3,'sim_day':[1,2,3],'gross':[50,30,900],'costs':[1,2,99]})
    m=trading_metrics(eq,f,['A','NIFTY_REF'],2,2)
    assert m.loc['A','invested']==pytest.approx(.8)
    assert m.loc['A','fees']==2
    assert m.loc['A','turnover']==pytest.approx(30/110)
    assert pd.isna(m.loc['NIFTY_REF','fees'])
