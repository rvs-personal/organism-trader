import pandas as pd
from otrader.dashboard.charts import token_chart, history_chart, comparison_frame


def test_token_bars_are_vertical_with_sqlite_object_columns():
    usage=pd.DataFrame({'agent_id':['A1','A2','A1'],'sim_day':['1','1','2'],
                        'input_tokens':['40000','45000','41000'],'output_tokens':['2000','3000','2500']})
    fig=token_chart(usage)
    assert all(trace.orientation=='v' for trace in fig.data)
    assert list(fig.data[0].x)==['Day 1','Day 2']
    assert list(fig.data[0].y)==[42000,43500]
    assert fig.data[0].name=='Atlas'
    assert fig.layout.xaxis.type=='category'


def test_comparison_excludes_partial_day_and_chart_starts_at_zero():
    eq=pd.DataFrame({'owner':['A1','A1','A2'],'sim_day':[1,2,1],
                     'equity':[101,150,102],'exposure':[.5,.9,.5]})
    result=comparison_frame(eq,1,100,['A1','A2'])
    assert result.set_index('owner').loc['A1','equity']==101
    fig=history_chart(eq[eq.sim_day<=1],['A1','A2'],100)
    assert list(fig.data[0].x)==[0,1]
    assert fig.data[0].y[0]==0
