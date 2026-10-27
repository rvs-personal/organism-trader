"""Explicitly typed, consistently labelled charts for the read-only dashboard."""
import pandas as pd
import plotly.graph_objects as go

COLORS = {'A1':'#4C9AFF','A2':'#F5A65B','A3':'#43C6AC','A4':'#B695F4','A5':'#80C7FF','A6':'#FFD08A','A7':'#94E8BD','A8':'#E4AEF5'}
NAMES = {'A1':'Atlas','A2':'Phoenix','A3':'Sentinel','A4':'Scout',
         'A5':'Atlas-P','A6':'Phoenix-P','A7':'Sentinel-P','A8':'Scout-P',
         'B_INDEX':'Buy & hold','B_CASH':'Cash only','B_EQUAL':'Equal weight',
         'B_MOM20':'Momentum rules','B_REV5':'Reversal rules'}
STYLES = {'A1':'Patient trend follower','A2':'Mean-reversion contrarian',
          'A3':'Risk-aware investor','A4':'Curious generalist',
          'A5':'Trend investor · pressure','A6':'Contrarian · pressure',
          'A7':'Risk-aware · pressure','A8':'Generalist · pressure'}


def finish(fig, ytitle, height=370, percent=False):
    fig.update_layout(height=height, margin=dict(l=15,r=20,t=25,b=35),
                      xaxis_title='Simulation day', yaxis_title=ytitle,
                      legend=dict(orientation='h', y=1.16, x=0, title=None),
                      hovermode='x unified', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', font=dict(size=13))
    fig.update_xaxes(dtick=1, showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.16)', zerolinecolor='rgba(128,128,128,0.4)')
    if percent: fig.update_yaxes(tickformat='.1%')
    return fig


def history_chart(equity, owners, starting_cash, metric='return'):
    fig=go.Figure()
    for owner in owners:
        g=equity[equity.owner==owner].sort_values('sim_day')
        if g.empty: continue
        days=[0]+pd.to_numeric(g.sim_day).tolist()
        eq=pd.Series([starting_cash]+pd.to_numeric(g.equity).tolist())
        if metric=='return': values=eq/starting_cash-1
        elif metric=='drawdown': values=eq/eq.cummax()-1
        else: values=[0]+pd.to_numeric(g.exposure).tolist()
        fig.add_trace(go.Scatter(x=days,y=values,name=NAMES.get(owner,owner),mode='lines+markers',
                                line=dict(color=COLORS.get(owner,'#97A3B3'),width=3 if owner in COLORS else 2,
                                          dash='solid' if owner in COLORS else 'dash'),marker=dict(size=5),
                                hovertemplate='%{y:.2%}<extra>%{fullData.name}</extra>'))
    labels={'return':'Return from starting capital','drawdown':'Fall from previous peak','exposure':'Portfolio invested in stocks'}
    finish(fig,labels[metric],percent=True)
    fig.add_hline(y=0,line_width=1,line_color='rgba(128,128,128,0.4)')
    maxday=int(pd.to_numeric(equity.sim_day).max()) if len(equity) else 0
    fig.update_xaxes(dtick=max(1,maxday//10),range=[0,max(1,maxday)])
    if metric=='exposure': fig.update_yaxes(range=[0,1.05])
    return fig


def token_chart(usage):
    """Vertical grouped bars, including when SQLite returns object-typed columns."""
    frame=usage.copy()
    for col in ['sim_day','input_tokens','output_tokens']:
        frame[col]=pd.to_numeric(frame[col],errors='coerce').fillna(0)
    frame['tokens']=frame.input_tokens+frame.output_tokens
    grouped=frame.groupby(['sim_day','agent_id'],as_index=False).tokens.sum()
    fig=go.Figure()
    for aid in COLORS:
        g=grouped[grouped.agent_id==aid]
        if g.empty: continue
        fig.add_trace(go.Bar(x=['Day '+str(int(d)) for d in g.sim_day],y=g.tokens,
                             orientation='v',name=NAMES[aid],marker_color=COLORS[aid],
                             hovertemplate='%{x}: %{y:,.0f} tokens<extra>%{fullData.name}</extra>'))
    finish(fig,'Tokens used',height=360)
    fig.update_layout(barmode='group',bargap=.25,hovermode='closest')
    fig.update_xaxes(type='category',dtick=None)
    fig.update_yaxes(rangemode='tozero',tickformat='~s')
    return fig


def comparison_frame(equity, completed_day, starting_cash, owners):
    """Compare at a shared completed day. Never carry future partial-day results backwards."""
    rows=[]
    for owner in owners:
        g=equity[(equity.owner==owner)&(equity.sim_day<=completed_day)].sort_values('sim_day')
        if g.empty: continue
        last=g.iloc[-1]
        curve=pd.Series([starting_cash]+g.equity.tolist())
        rows.append(dict(owner=owner,name=NAMES.get(owner,owner),equity=float(last.equity),
                         profit=float(last.equity)-starting_cash,returns=float(last.equity)/starting_cash-1,
                         drawdown=float((curve/curve.cummax()-1).min()),day=int(last.sim_day)))
    return pd.DataFrame(rows,columns=['owner','name','equity','profit','returns','drawdown','day'])


def investment_ranking(frame, closed_only=False):
    """Clearly labelled purchase-outlay return, never portfolio return / exposure."""
    metric='closed_return' if closed_only else 'deployed_return'
    basis='closed_basis' if closed_only else 'deployed'
    profit='closed_net' if closed_only else 'net_pnl'
    g=frame[frame[metric].notna()].sort_values(metric)
    fig=go.Figure(go.Bar(x=g[metric],y=[NAMES.get(x,x) for x in g.owner],orientation='h',
        marker_color=[COLORS.get(x,'#97A3B3') for x in g.owner],
        text=[f'{v:+.2%}' for v in g[metric]],textposition='outside',cliponaxis=False,
        customdata=g[[basis,profit,'buy_orders','sell_orders','average_invested']].values,
        hovertemplate='%{y}<br>Return on purchase outlay: %{x:+.2%}<br>Outlay counted: ₹%{customdata[0]:,.0f}<br>Net gain/loss: ₹%{customdata[1]:+,.0f}<br>Buy orders: %{customdata[2]:.0f} · sell orders: %{customdata[3]:.0f}<br>Average portfolio invested: %{customdata[4]:.1%}<extra></extra>'))
    finish(fig,'',height=max(300,45*len(g)+70))
    fig.update_layout(xaxis_title='Net gain / purchase outlay (including entry fees)',hovermode='closest',showlegend=False)
    lo=min(0,float(g[metric].min())) if len(g) else 0;hi=max(0,float(g[metric].max())) if len(g) else 0
    spread=max(hi-lo,.01)
    fig.update_xaxes(dtick=None,tickformat='.1%',zeroline=True,showgrid=True,
                     gridcolor='rgba(128,128,128,0.12)',range=[lo-spread*.2,hi+spread*.25])
    fig.update_yaxes(type='category',showgrid=False)
    return fig
