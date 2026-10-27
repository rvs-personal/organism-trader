"""Quant benchmark view. Reads saved results only; never invokes inference."""
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from otrader.dashboard.charts import COLORS, NAMES
from otrader.dashboard.quant_metrics import aligned_values, performance_metrics, trading_metrics

LABELS = {**NAMES, 'B_INDEX': 'Buy & hold · 29-stock basket', 'NIFTY_REF': 'Nifty 50 · price index'}
BENCHMARKS = {
    'B_INDEX': 'Buy 29 stocks in approximately equal amounts on day 1, deploying about 99% of initial cash; then hold.',
    'B_EQUAL': 'Target 98% invested across 29 stocks. Rebalance every 20 days when a position differs from its target by more than 3% of that target.',
    'B_MOM20': 'Every 5 days, select the five highest trailing 20-day returns. Sell names that leave the list and buy new names at 18% of equity each. Retained positions are not resized.',
    'B_REV5': 'Every 5 days, sell holdings and buy the five lowest trailing 5-day returns at 18% of equity each.',
    'B_CASH': 'Keep all money in cash, earning zero interest.',
    'NIFTY_REF': 'Untraded Nifty 50 price-index reference from the saved MKT series. No fees or dividend reinvestment; not a total-return investment.'}


def chart(fig, key, ytitle=None):
    fig.update_layout(height=340, margin=dict(l=10,r=10,t=20,b=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#dce4ed'), legend_title_text='', yaxis_title=ytitle)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='rgba(150,170,190,.12)')
    st.plotly_chart(fig, use_container_width=True, key=key, config={'displaylogo':False})


def render_quant(equity, fills, agent_ids, completed, starting_cash, dataset_dir):
    st.subheader('Quant benchmarks')
    st.caption('Return is the outcome. Risk, exposure and costs explain how it was achieved. Historical development results—not a forecast.')
    if completed < 1:
        st.info('Comparisons start after the first fully completed day.'); return
    c1,c2,c3 = st.columns([2,2,1])
    blocks = {'Since start': (1,completed)}
    blocks.update({f'Days {d}–{min(d+49,completed)}': (d,min(d+49,completed)) for d in range(1,completed+1,50)})
    window = c1.selectbox('Comparison window', list(blocks), key='quant_window')
    first,last = blocks[window]
    all_equity = equity.copy()
    owners = list(agent_ids) + [k for k in BENCHMARKS if k.startswith('B_') and k in set(equity.owner)]
    index_note = None
    try:
        mapping = json.loads((dataset_dir/'mapping.json').read_text())
        if mapping.get('symbols',{}).get('^NSEI') == 'MKT':
            market = pd.read_parquet(dataset_dir/'bars.parquet', columns=['asset','sim_day','close'],
                filters=[('asset','==','MKT'),('sim_day','>=',first-1),('sim_day','<=',last)])
            # Relative index units are enough: statistics rebase at the window opening mark.
            index_equity = market.rename(columns={'close':'equity'}).assign(owner='NIFTY_REF')
            # Day 0 must be real for this reference, not synthetic initial cash.
            if first-1 in set(index_equity.sim_day):
                all_equity = pd.concat([all_equity,index_equity], ignore_index=True)
                owners.append('NIFTY_REF')
            else:
                index_note = 'Nifty reference unavailable: the opening index mark is missing.'
    except (OSError, ValueError, KeyError) as exc:
        index_note = f'Nifty reference unavailable ({type(exc).__name__}); saved strategy comparisons remain available.'
    values, excluded = aligned_values(all_equity, owners, first, last, starting_cash)
    choices = [b for b in BENCHMARKS if b in values]
    if not choices:
        st.warning('No benchmark has a complete daily history for this window.'); return
    benchmark = c2.selectbox('Benchmark to beat',choices,format_func=lambda k:LABELS[k],key='quant_benchmark')
    rf = c3.number_input('Annual risk-free rate (%)',min_value=0.0,max_value=20.0,value=0.0,step=0.5,key='quant_rf') / 100
    stats = performance_metrics(values,benchmark,rf).join(trading_metrics(equity,fills,values.columns,first,last))
    st.caption(f'Days {first}–{last} · {last-first+1} daily returns · same opening and closing marks for every participant · partial day {completed+1} excluded. Cash flows are internal buys/sells; no deposits or withdrawals.')
    if index_note: st.caption(index_note)
    if excluded: st.warning('Excluded incomplete histories: ' + '; '.join(f'{LABELS.get(k,k)}: {v}' for k,v in excluded.items()))
    selected_options = [a for a in agent_ids if a in stats.index]
    if not selected_options:
        st.info('No agent has a complete history in this window.'); return
    selected = st.selectbox('Inspect agent',selected_options,format_func=lambda a:LABELS.get(a,a),key='quant_agent')
    row = stats.loc[selected]
    k1,k2,k3,k4 = st.columns(4)
    k1.metric('Portfolio return',f'{row.total_return:+.2%}',f'{row.advantage*100:+.2f} pp vs benchmark')
    k2.metric('Worst peak-to-trough fall',f'{row.max_drawdown:.2%}')
    k3.metric('Sharpe · return / volatility','N/A' if pd.isna(row.sharpe) else f'{row.sharpe:.2f}')
    k4.metric('Average capital invested',f'{row.invested:.1%}')
    st.caption('Sharpe and Sortino are estimates, not scores with a guaranteed pass mark. Risk ratios require at least 20 daily returns; N/A also means the denominator is zero.')
    left,right = st.columns(2)
    rank = stats.loc[selected_options].sort_values('advantage').reset_index()
    rank['Participant'] = rank.owner.map(LABELS)
    rank['Advantage (pp)'] = rank.advantage*100
    with left:
        st.markdown('#### Who beats the selected benchmark?')
        fig = px.bar(rank,x='Advantage (pp)',y='Participant',orientation='h',color='owner',
            color_discrete_map={a:COLORS.get(a,'#789') if a==selected else '#70859e' for a in selected_options})
        fig.update_layout(showlegend=False); fig.add_vline(x=0,line_color='#ccd5df',line_width=1)
        chart(fig,'quant_advantage')
    with right:
        st.markdown('#### Return versus risk')
        scatter = stats.reset_index(); scatter['Participant'] = scatter.owner.map(LABELS)
        scatter['Annualized volatility (%)'] = scatter.volatility*100
        scatter['Window return (%)'] = scatter.total_return*100
        scatter['Type'] = scatter.owner.map(lambda a:'AI agent' if a in agent_ids else 'Benchmark')
        if last-first+1 >= 20:
            fig = px.scatter(scatter,x='Annualized volatility (%)',y='Window return (%)',hover_name='Participant',
                symbol='Type',color='Type',color_discrete_map={'AI agent':'#60a5fa','Benchmark':'#aab7c7'})
            fig.update_traces(marker_size=12)
            chart(fig,'quant_risk')
        else: st.info('Risk comparison appears after 20 complete sessions in the selected window.')
    st.markdown('#### Same money, same dates')
    st.caption('Both lines start at 100. For example, 108 means an 8% gain from the opening mark of this window.')
    fig = go.Figure()
    for owner in [selected,benchmark]:
        fig.add_scatter(x=values.index,y=100*values[owner]/values[owner].iloc[0],mode='lines',
            name=LABELS.get(owner,owner),line=dict(color=COLORS.get(owner,'#abb9cc'),width=3))
    fig.update_layout(xaxis_title='Simulation day',hovermode='x unified')
    chart(fig,'quant_wealth','Growth of 100')
    st.markdown('#### Benchmark matrix')
    st.caption('All percentages refer to the selected window unless labelled annualized. Higher return, Sharpe, Sortino and information ratio are preferable; lower volatility and smaller drawdowns mean less measured risk. Beta measures market sensitivity, not quality.')
    columns = {'total_return':'Return (%)','advantage':'Ahead of benchmark (pp)', 'max_drawdown':'Worst fall (%)',
        'volatility':'Annualized volatility (%)','sharpe':'Sharpe','sortino':'Sortino',
        'information_ratio':'Information ratio','beta':'Beta vs benchmark','invested':'Average invested (%)',
        'fees':'Trading fees (₹)','turnover':'Two-way turnover (×)'}
    table = stats[list(columns)].copy()
    for key in ['total_return','advantage','max_drawdown','volatility','invested']: table[key] *= 100
    table.index = table.index.map(lambda a:LABELS.get(a,a)); table.index.name='Participant'
    st.dataframe(table.rename(columns=columns).style.format('{:,.2f}',na_rep='N/A'),use_container_width=True)
    with st.expander('How to read the metrics and judge the experiment'):
        st.markdown('''- **Sharpe:** average daily return above the assumed risk-free rate, divided by daily return volatility, multiplied by √252.
- **Sortino:** the same excess return, divided by downside deviation. Downside deviation uses all sessions, with positive excess returns counted as zero downside.
- **Information ratio:** average daily return minus the chosen benchmark’s daily return, divided by the volatility of that difference, multiplied by √252. It measures consistency of relative performance.
- **Beta:** covariance with benchmark daily returns divided by benchmark variance. Near 1 means similar sensitivity; near 0 means little measured sensitivity.
- **Worst fall:** largest decline from an earlier portfolio peak, including the opening mark of this window.
- **Average invested:** mean daily closing stock value divided by total portfolio value. It explains cash drag; it is not a return on deployed capital.
- **Two-way turnover:** gross buys plus gross sells divided by average daily closing equity. Trading fees already reduce portfolio returns; they are not subtracted again here. Model/API costs are separate in Model usage.

Risk ratios use 252 sessions/year, sample standard deviation, and no adjustment for serial correlation. Short samples and repeated strategy changes make estimates unstable. Zero risk-free rate matches the cash-only baseline; changing the assumption does not credit interest to any portfolio.

**What this can establish:** which recorded agent or rule-based strategy performed better on identical observed dates. These are development results, including reviewed interventions; a before/after difference is not proof the intervention caused it. No external AI models were run in this comparison.

**Before claiming superiority:** freeze the chosen strategy and memory-update policy, evaluate on untouched later data, include trading costs and execution assumptions, and report risk and uncertainty across multiple periods. A holdout cannot remain untouched if its outcomes are repeatedly used to tune the strategy.

**Nifty caveat:** the index reference is a price index, not Nifty TRI. It excludes reinvested dividends and trading costs; stock data uses the experiment’s adjusted prices. This is a contextual market comparison, not a dividend-matched investable benchmark.

Metric references: [Sharpe’s definition](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm), [QuantConnect backtest reporting](https://www.quantconnect.com/docs/v2/cloud-platform/backtesting/report), [Nifty total-return index](https://www.niftyindices.com/resources/index-concepts/total-return-index).''')
        for key in choices: st.markdown(f'**{LABELS[key]}:** {BENCHMARKS[key]}')
