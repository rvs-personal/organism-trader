"""Manual notebook dashboard. Reads cloud SQLite; never invokes an inference backend."""
import html
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from .charts import NAMES, COLORS, history_chart, token_chart, comparison_frame, investment_ranking, finish
from .analytics import investment_returns, fifo_matches


def render(run_dir):
    run_dir = Path(run_dir)
    with sqlite3.connect(f'file:{run_dir / "state.sqlite"}?mode=ro', uri=True) as db:
        db.execute('BEGIN')
        state = dict(db.execute('SELECT key,value FROM run_state'))
        tables = {t: pd.read_sql_query(f'SELECT * FROM {t}', db) for t in
                  ('agents','equity','fills','decisions','reflections')}
        tables['inference_events'] = pd.read_sql_query('SELECT * FROM inference_events',db) if db.execute("SELECT 1 FROM sqlite_master WHERE name='inference_events'").fetchone() else pd.DataFrame()
    cfg = json.loads(state['run_config'])
    start = float(cfg['capital']['starting_cash'])
    day = int(state.get('clock', '1'))-1
    equity = tables['equity']; equity = equity[equity.sim_day <= day].copy()
    fills = tables['fills']; fills = fills[fills.sim_day <= day].copy()
    owners = [a['id'] for a in cfg['agents']] + [n for n in NAMES if n.startswith('B_')]
    esc = lambda x: html.escape(str(x))
    parts = ['<style>body{font:15px system-ui;color:#202630;background:#fff;margin:24px}h1{font-size:27px}h2{margin-top:30px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef2f6}details{margin:12px 0;padding:12px;border:1px solid #ddd;border-radius:8px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere}p{line-height:1.55}.scroll{overflow:auto} .muted{color:#607080}</style>',
             f'<h1>Eight independent agents · {esc(run_dir.name)}</h1>',
             f'<p><b>{day} / {cfg["n_sim_days"]} days complete</b> · {esc(state.get("runner_status"))} · First review: day50</p>',
             f'<p class="muted">Manual snapshot: {datetime.now():%Y-%m-%d %H:%M:%S} UTC runtime time. Refresh reads saved results; it makes no model calls. Charts and rankings use fully completed market days.</p>']
    loaded_js = False
    def plot(fig):
        nonlocal loaded_js
        parts.append(fig.to_html(full_html=False, include_plotlyjs=True if not loaded_js else False,
                                 config={'displaylogo':False,'responsive':True}))
        loaded_js = True
    def table(frame):
        parts.append('<div class="scroll">'+frame.to_html(index=False, escape=True, border=0)+'</div>')
    ranking = comparison_frame(equity, day, start, owners)
    if ranking.empty:
        ranking = pd.DataFrame([dict(owner=o,name=NAMES.get(o,o),equity=start,profit=0.,returns=0.,drawdown=0.,day=0) for o in owners])
    ranking = ranking.sort_values('returns',ascending=False)
    shown = pd.DataFrame({'Participant':ranking.name, 'Through day':ranking.day,
                          'Portfolio value':ranking.equity.map(lambda x:f'₹{x:,.0f}'),
                          'Profit / loss':ranking.profit.map(lambda x:f'₹{x:+,.0f}'),
                          'Return':ranking.returns.map(lambda x:f'{x:+.2%}'),
                          'Worst fall':ranking.drawdown.map(lambda x:f'{x:.2%}')})
    parts.append('<h2>Leaderboard</h2>'); table(shown)
    if day == 0:
        parts.append('<p>Starting balances shown. No complete trading day is available for performance comparison yet.</p>')
    if not equity.empty:
        fig = go.Figure()
        for owner in owners:
            g = equity[equity.owner==owner].sort_values('sim_day')
            if len(g): fig.add_trace(go.Scatter(x=[0]+g.sim_day.tolist(),y=[start]+g.equity.tolist(),name=NAMES.get(owner,owner),line={'color':COLORS.get(owner,'#8b949e')}))
        parts.append('<h2>Portfolio value</h2>');plot(finish(fig,'Portfolio value (INR)'))
        for metric,label in [('return','Returns'),('drawdown','Drawdown'),('exposure','Capital invested over time')]:
            parts.append(f'<h2>{label}</h2>');plot(history_chart(equity,owners,start,metric))
        invested = investment_returns(fills,equity,day,start,owners)
        if len(invested) and invested.deployed_return.notna().any():
            parts.append('<h2>Net gain per rupee deployed</h2><p>Purchase outlay counts recycled capital again. This is not a risk-adjusted return.</p>')
            plot(investment_ranking(invested))
    parts.append('<h2>All eight agents · portfolios, decisions and memory</h2>')
    for a in cfg['agents']:
        rows=tables['agents'][tables['agents'].agent_id==a['id']]
        if rows.empty: continue
        snap=json.loads(rows.iloc[0].snapshot_json)
        parts.append(f'<details><summary>{esc(a["name"])} · {esc(a.get("condition","normal"))} · saved through agent day {snap["age_days"]}</summary>')
        parts.append(f'<p>{esc(a["persona"])}</p><p>{esc(snap["strategy"])}</p><p>{esc(cfg.get("initial_policy",{}).get("agents",{}).get(a["id"],{}).get("protocol",""))}</p>')
        parts.append('<h3>Latest saved portfolio</h3><pre>'+esc(json.dumps(snap.get('portfolio',{}),indent=2))+'</pre>')
        decisions=tables['decisions'][tables['decisions'].agent_id==a['id']].sort_values('sim_day')
        if len(decisions): parts.append('<h3>Latest decision</h3><pre>'+esc(json.dumps(json.loads(decisions.iloc[-1].decision_json),indent=2))+'</pre>')
        parts.append('<h3>Recent memory</h3><pre>'+esc(json.dumps(snap.get('episodic',[])[-5:],indent=2))+'</pre></details>')
    parts.append('<h2>Model usage</h2>')
    events=[dict(agent_id=r.agent_id,phase=r.phase,**json.loads(r.event_json)) for r in tables['inference_events'].itertuples()]
    responses=pd.DataFrame([e for e in events if e.get('event')=='response'])
    if len(responses):
        usage=responses.groupby(['agent_id','phase'],as_index=False)[['input_tokens','output_tokens','cache_read_tokens','seconds']].sum()
        table(usage.rename(columns={'agent_id':'Agent','phase':'Stage','input_tokens':'Input tokens','output_tokens':'Output tokens','cache_read_tokens':'Reported cache reads','seconds':'Response seconds'}).round(2))
        parts.append('<p>Attempt totals include validation retries. Provider failures may have unreported usage. Cache reads do not mean agent decisions are reused.</p>')
    else: parts.append('<p>No completed model response recorded yet.</p>')
    if len(tables['decisions']): plot(token_chart(tables['decisions']))
    if len(fills):
        parts.append('<details><summary>Executed trades</summary>');table(fills);parts.append('</details>')
        matched=fifo_matches(fills)
        if len(matched): parts.append('<details><summary>FIFO sold lots</summary>');table(matched);parts.append('</details>')
    parts.append('<h2>Review checkpoints</h2><p>Reflection every10 days; independent review and pause at50. Strategy proposals are recorded, not automatically imposed. Resuming the next block requires the review and your instruction.</p>')
    for path in sorted((run_dir/'reviews').glob('day-*-review.md')):
        parts.append('<details><summary>'+esc(path.stem)+'</summary><pre>'+esc(path.read_text())+'</pre></details>')
    return '\n'.join(parts)
