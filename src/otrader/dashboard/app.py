"""Read-only experiment dashboard: outcomes first, detailed investigation on demand."""
from __future__ import annotations
import difflib
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from otrader.config import RUNS_DIR, DATA_DIR, load_experiment
from otrader.run_control import start_run, running
from otrader.governance import enabled as governed_policy
from otrader.dashboard.analytics import fifo_matches, sector_pnl, investment_returns
from otrader.dashboard.quant import render_quant
from otrader.dashboard.charts import COLORS, NAMES, STYLES, history_chart, token_chart, comparison_frame, finish, investment_ranking

st.set_page_config(page_title='Organism Trader · Live experiment',page_icon='◈',layout='wide')
st.markdown('''<style>
.block-container {padding-top:3.5rem; max-width:1500px}
[data-testid="stMetricValue"] {font-size:1.9rem; font-weight:650}
[data-testid="stMetricLabel"] {opacity:.8}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px}
[data-testid="stTabs"] button {font-weight:600}
h1 {letter-spacing:-.04em} h3 {letter-spacing:-.02em}
</style>''',unsafe_allow_html=True)

runs=sorted(p.parent.name for p in RUNS_DIR.glob('*/state.sqlite') if not (p.parent/'.dashboard-hidden').exists())
latest=max(runs,key=lambda r:(RUNS_DIR/r/'state.sqlite').stat().st_mtime) if runs else None
st.sidebar.title('Organism Trader')
st.sidebar.caption('Independent minds. One market.')
selected_run=st.sidebar.selectbox('Experiment',runs,index=runs.index(latest)) if runs else None
st.sidebar.caption('Manual updates · use Refresh dashboard to load the latest saved results.')
if st.sidebar.button('Refresh dashboard',use_container_width=True): st.cache_data.clear()
st.sidebar.divider()
st.sidebar.divider()
st.sidebar.caption('Historical market replay · simulated money')
if not selected_run:
    st.title('Ready for day one')
    st.metric('Completed days',0)
    st.info('Atlas, Phoenix, Sentinel and Scout are ready. No new agent trades have been recorded.')
    st.stop()

@st.cache_data
def read_snapshot(path, generation=None):
    conn=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    try:
        conn.execute('BEGIN')
        tables={name:pd.read_sql_query(f'SELECT * FROM {name}',conn) for name in
                ['run_state','equity','fills','agents','decisions','reflections','strategy_versions','graveyard']}
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_context'").fetchone():
            tables['research_context']=pd.read_sql_query('SELECT * FROM research_context ORDER BY sim_day,agent_id',conn)
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='inference_events'").fetchone():
            tables['recovery_events'] = pd.read_sql_query("SELECT agent_id,sim_day,phase,event_json FROM inference_events WHERE json_extract(event_json,'$.event') IN ('episode_retry','episode_retry_exhausted') ORDER BY id DESC LIMIT 50", conn)
        for name in ['equity','fills','decisions','reflections','strategy_versions']:
            df=tables[name]
            for col in ['sim_day','equity','cash','exposure','input_tokens','output_tokens','cost_inr','qty','gross','costs','price','version']:
                if col in df: df[col]=pd.to_numeric(df[col],errors='coerce')
        tables['equity']=tables['equity'].sort_values('sim_day',kind='stable')
        tables['agents']=tables['agents'].sort_values('agent_id',kind='stable')
        return tables
    finally: conn.close()


def plot(fig,key):
    st.plotly_chart(fig,use_container_width=True,key=key,config={'displaylogo':False,'scrollZoom':False})


def render_agent_detail(agents, fills, decisions, reflections, strat, cfg):
    agent_names = {r.agent_id: json.loads(r.snapshot_json).get('display_name') or r.agent_id
                   for r in agents.itertuples()}
    aid = st.selectbox("Agent", agents["agent_id"].tolist(),
                       format_func=lambda id: f"{agent_names[id]} ({id})") if len(agents) else None
    if aid:
        st.subheader(agent_names[aid])
        snap = json.loads(agents.set_index("agent_id").loc[aid, "snapshot_json"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", snap["status"]); c2.metric("Age (days)", snap["age_days"])
        c3.metric("Strategy version", snap["strategy_version"]); c4.metric("Generation", snap["generation"])

        tabs = st.tabs(["Mind", "Trades", "Decisions", "Evolution", "Perception", "Confidence (descriptive)", "Memory"])
        with tabs[0]:
            st.subheader("Strategy"); st.write(snap["strategy"])
            governed = governed_policy(cfg, int(snap["age_days"]))
            st.subheader("Reviewed memory used in decisions" if governed else "Long-term memory")
            if snap.get("long_term"):
                st.caption(f"{len(snap['long_term'])} saved entries for {agent_names[aid]}. Reviewed observations guide reasoning; they are not guaranteed trading rules.")
                for entry in snap["long_term"]:
                    st.markdown(f"- {entry}")
            else:
                st.info("No reviewed long-term entries have been saved yet.")
            if governed:
                st.caption("These are separate concepts: a belief is a claim the agent thinks may be true; a hypothesis is an idea to test. Self-reflection proposes both. Reviewed memory contains accepted observations and lessons supplied to later decisions; it is not a rename of either field.")
                st.caption("Beliefs and hypotheses proposed during self-reflection are archived for independent review. They are not automatically copied into the active policy. Empty legacy fields do not mean the agent has no memory.")
                own_reflections = reflections[reflections["agent_id"] == aid].sort_values("sim_day", ascending=False)
                st.subheader("Latest self-reflection proposals")
                if own_reflections.empty:
                    st.info("No reflection saved yet. Self-reflections are scheduled every 10 days.")
                else:
                    latest = own_reflections.iloc[0]
                    proposal = json.loads(latest["reflection_json"])
                    st.caption(f"Day {int(latest['sim_day'])} · {latest['kind']} · original proposals, not an approval list. Accepted or corrected lessons appear in reviewed memory above.")
                    st.markdown("**Proposed beliefs**")
                    if proposal.get("belief_updates"):
                        for title, value in proposal["belief_updates"].items():
                            st.markdown(f"**{title.replace('_', ' ')}**")
                            st.write(value)
                    else:
                        st.write("No new beliefs proposed in this reflection.")
                    st.markdown("**Proposed hypotheses to test**")
                    if proposal.get("hypotheses"):
                        for hypothesis in proposal["hypotheses"]:
                            st.write(hypothesis)
                    else:
                        st.write("No new hypotheses proposed in this reflection.")
                    st.caption("Earlier proposals and their original evidence remain in the Evolution tab.")
                with st.expander("Legacy active belief and hypothesis fields"):
                    st.json({"beliefs": snap["beliefs"], "hypotheses": snap["hypotheses"]})
            else:
                st.subheader("Beliefs")
                st.json(snap["beliefs"]) if snap["beliefs"] else st.info("No active beliefs saved yet.")
                st.subheader("Hypotheses")
                st.write(snap["hypotheses"]) if snap["hypotheses"] else st.info("No active hypotheses saved yet.")
            st.subheader("Perception policy"); st.write(snap["perception_policy"])
            st.subheader("Working memory"); st.write(snap["working_memory"])
        with tabs[1]:
            f = fills[fills["owner"] == aid].sort_values("sim_day", ascending=False)
            st.dataframe(f, use_container_width=True, hide_index=True)
            pos = pd.DataFrame(snap["portfolio"].get("positions", {})).T if snap.get("portfolio") else pd.DataFrame()
            st.subheader("Open positions"); st.dataframe(pos, use_container_width=True)
        with tabs[2]:
            d = decisions[decisions["agent_id"] == aid].sort_values("sim_day", ascending=False)
            for _, r in d.head(30).iterrows():
                dec = json.loads(r["decision_json"])
                with st.expander(f"Day {r['sim_day']} · conf {dec.get('confidence')} · {len(dec.get('orders', []))} orders · "
                                 f"{r['input_tokens']+r['output_tokens']} tok"):
                    st.write(dec.get("market_view"))
                    st.json({"orders": dec.get("orders"), "theses": dec.get("theses"), "exec": json.loads(r["exec_json"]),
                             "allocation_plan": dec.get("allocation_plan"), "candidate_assessments": dec.get("candidate_assessments"), "tools": [t["tool"] for t in json.loads(r["tool_trace_json"])]})
        with tabs[3]:
            sv = strat[strat["agent_id"] == aid]
            st.subheader("Strategy versions")
            previous = None
            for _, r in sv.iterrows():
                with st.expander(f"v{r['version']} · day {r['sim_day']} · {r['trigger']}"):
                    st.write(r["strategy"])
                    if previous is not None:
                        diff = '\n'.join(difflib.unified_diff(previous.splitlines(), r['strategy'].splitlines(),
                                                           fromfile='previous', tofile='current', lineterm=''))
                        st.code(diff or 'No text change', language='diff')
                previous = r['strategy']
            st.subheader("Reflections")
            for _, r in reflections[reflections["agent_id"] == aid].sort_values("sim_day", ascending=False).iterrows():
                with st.expander(f"Day {r['sim_day']} · {r['kind']} · {r['model_id']}"):
                    st.json(json.loads(r["reflection_json"]))
        with tabs[4]:
            d = decisions[decisions["agent_id"] == aid]
            if len(d):
                recs = []
                for _, r in d.iterrows():
                    for t in json.loads(r["tool_trace_json"]):
                        recs.append({"sim_day": r["sim_day"], "tool": t["tool"]})
                tp = pd.DataFrame(recs)
                if len(tp):
                    tp["bucket"] = (tp["sim_day"] - 1) // 10 * 10 + 1
                    pivot = tp.groupby(["bucket", "tool"]).size().reset_index(name="calls")
                    st.plotly_chart(px.bar(pivot, x="bucket", y="calls", color="tool", title="Research tools used per 10-day period", orientation="v"),
                                    use_container_width=True)
        with tabs[5]:
            st.caption('Days 1–50: legacy association only; later trades affect this P&L. From day 51, conviction is recorded without a misleading future-P&L score.')
            cal = pd.DataFrame([c for c in snap["calibration"] if c.get("pnl_5d") is not None])
            if len(cal):
                st.plotly_chart(px.scatter(cal, x="confidence", y="pnl_5d", trendline=None,
                                           title="Historical conviction vs subsequent portfolio P&L (not calibration)"), use_container_width=True)
            else:
                st.info("No legacy confidence/P&L observations to display.")
        with tabs[6]:
            st.subheader("Long-term memory"); st.write(snap["long_term"])
            st.subheader("Episodic memory (recent)")
            st.dataframe(pd.DataFrame(snap["episodic"]), use_container_width=True, hide_index=True)


@st.fragment
def render():
    snapshot_path = RUNS_DIR/selected_run/'state.sqlite'
    data=read_snapshot(str(snapshot_path), snapshot_path.stat().st_mtime_ns)
    state=dict(data['run_state'].itertuples(index=False,name=None))
    cfg=json.loads(state['run_config']) if state.get('run_config') else load_experiment()
    equity,fills,agents,decisions,reflections=(data[k] for k in ['equity','fills','agents','decisions','reflections'])
    start=float(cfg['capital']['starting_cash'])
    end=int(cfg['n_sim_days'])
    extension_path = RUNS_DIR / selected_run / 'data_extension.json'
    if extension_path.exists():
        extension_info = json.loads(extension_path.read_text())
        segment_end = end
        end = max(end, int(extension_info['n_sim_days']))
        extension_status = 'active' if state.get('data_extension') else f'queued after day {segment_end}'
        st.info(f"Daily history: {extension_info['n_sim_days']} trading days prepared · extension {extension_status}.")
        gaps = sum(len(dates) for dates in extension_info.get('missing_dates', {}).values())
        if gaps:
            st.caption(f'Stock and market-index coverage is complete in the extension. VOLX has {gaps} missing observations; its history uses available observations only.')
    completed=max(0,min(end,int(state.get('clock',1))-1))
    status=state.get('runner_status','Not started')
    mode=json.loads(state.get('run_mode','{}'))
    stub=mode.get('backend')=='stub' or (len(decisions)>0 and decisions.model_id.eq('stub').all())
    st.caption(f'LIVE EXPERIMENT / {selected_run.upper()}')
    st.title(f'{len(agents)} agents. A documented experiment.')
    st.caption(f'{"Rule-based demo" if stub else "AI decisions · "+cfg["inference"]["daily_model"]} · '
               f'₹{start:,.0f} starting capital per participant · updated {datetime.now():%H:%M:%S}')
    run_dir = RUNS_DIR / selected_run
    intervention_specs = [json.loads(p.read_text()) for p in sorted(run_dir.glob('intervention-*.json'), key=lambda p:int(p.stem.split('-')[-1]))]
    intervention_spec = intervention_specs[-1] if intervention_specs else None
    # Multiple approved revisions at one boundary describe one performance period.
    effective_specs = list({sp['effective_day']: sp for sp in intervention_specs}.values())
    pause_path = run_dir / 'pause_after_day.json'
    pause_day = int(json.loads(pause_path.read_text())['day']) if pause_path.exists() else end
    cloud_path = run_dir / 'cloud_source.json'
    cloud = json.loads(cloud_path.read_text()) if cloud_path.exists() else None
    cloud_fresh = bool(cloud and time.time()-cloud['received_at'] < 180 and (not cloud['runner_alive'] or cloud['checkpoint_age_seconds'] < 180))
    is_running = bool(cloud_fresh and cloud['runner_alive']) if cloud else running(run_dir)
    if cloud:
        stamp = datetime.fromtimestamp(cloud['received_at']).strftime('%H:%M:%S')
        st.info(f'Colab → local dashboard · checkpoint received at {stamp} · Refresh dashboard loads the latest received results. Cloud exports arrive approximately once a minute; no model calls are made by this dashboard.')
        if not cloud_fresh:
            st.warning('Cloud connection is stale. These are saved results; the current runner status is unknown.')
        elif not cloud['runner_alive']:
            st.warning(f'Colab runner is stopped. Saved status: {status}.')
        if cfg.get('dashboard', {}).get('cloud_notebook_url'):
            st.link_button('Open cloud run', cfg['dashboard']['cloud_notebook_url'])
    next_resume = min(pause_day + 50, end)
    review_ready = (run_dir / 'reviews' / f'day-{pause_day:03d}-review.md').exists()
    can_resume = not cloud and not is_running and status.startswith('paused: review') and completed == pause_day and completed < end and review_ready
    can_start = not cloud and not is_running and status.startswith('prepared:') and state.get('launch_readiness') == 'ready'
    with st.container(border=True):
        controls, action = st.columns([3,1])
        at_review = status.startswith('paused: review') and completed == pause_day
        controls.markdown(f'**Day {pause_day} review complete · paused before continuation**' if at_review and review_ready else f'**Next stop: day {pause_day} review**')
        if intervention_spec:
            phases = ['Days 1–50: original'] + [f"Days {sp['effective_day']}–{sp.get('review_day', sp['effective_day']+49)}: {sp.get('name', sp['id'])}" + (' (prepared)' if completed < sp['effective_day'] else '') for sp in effective_specs]
            controls.caption(' · '.join(phases))
        elif cfg.get('initial_policy', {}).get('allocation_mode') == 'agent_discretion':
            controls.caption('Fresh portfolios and memories · agents choose allocations and cash reserves · up to 100% invested · no fixed starter sizes or inherited stock filters')
            controls.caption('Four matched pairs: normal versus performance pressure. Reflection every 10 days; independent review and pause every 50 days.')
        action.caption(('Running on Colab' if is_running else 'Cloud status unknown' if not cloud_fresh else 'Stopped on Colab') if cloud else ('Running' if is_running else 'Paused / stopped'))
        resume_label = f'Running to day {pause_day}' if is_running else f'Start days 1–{pause_day}' if status.startswith('prepared:') else f'Resume to day {next_resume}'
        if cloud: resume_label = 'Controlled in Colab'
        if action.button(resume_label, disabled=not (can_resume or can_start), type='primary', key='resume_run',
                         help='Starts the saved portfolios and memories; stops at the next review. A completed review is required.'):
            try:
                result = start_run(selected_run, resume_review=can_resume)
                st.success(f"Resuming day {result['from_day']}; stop after day {result['stop_after_day']}.")
                st.cache_data.clear()
                st.rerun()
            except (ValueError, OSError) as error:
                st.error(str(error))
        if not is_running and status.startswith('paused: review') and not review_ready:
            controls.caption('Resume becomes available after the milestone review is saved.')
        if status.startswith('prepared:'):
            controls.caption('Day 0 · ' + state.get('launch_readiness', 'Waiting to start'))
    a,b,c=st.columns([2,1,1])
    a.metric('Days fully completed',f'{completed} / {end}',help='Progress toward the full planned replay, including the queued data extension. A day counts only after every scheduled participant finishes.')
    b.metric('Agents active',f'{int(agents.status.eq("ALIVE").sum())} / {len(agents)}')
    c.metric('Runner', ('Colab running' if is_running else 'Unknown / stale' if not cloud_fresh else 'Colab stopped') if cloud else status.split(':')[0].capitalize())
    block_start=max(0,pause_day-50)
    st.progress(min(1,max(0,(completed-block_start)/50)),text=f'Current block: {max(0,completed-block_start)} / 50 days · stop at day {pause_day}')
    if stub: st.warning('Offline demo — these decisions use fixed rules, not an AI model.')
    if status.startswith('prepared:') and state.get('model_preflight'):
        st.warning('Prepared at day 0. The local model has not passed the full readiness check; the simulation has not started.')
        with st.expander('Why the local run has not started', expanded=True):
            for label,value in json.loads(state['model_preflight']).items():
                st.markdown(f'**{label}:** {value}')
    if state.get('local_capacity'):
        capacity=json.loads(state['local_capacity'])
        st.caption(f"Local capacity: {capacity['actual_inference_slots']} inference at a time · maximum {capacity['model_copies']} loaded model · eight isolated agent memories. Requests run through a bounded queue.")
        with st.expander('Laptop health and execution limits'):
            st.write(capacity['reason'])
            health=json.loads(state.get('host_health','{}'))
            st.write({'Last resource sample':health})
            st.caption('Memory warning reduces concurrency. Critical memory pressure or detected thermal throttling pauses new work. No smaller context or lower precision is applied to gain speed.')
    if status.startswith('paused: review'):
        st.info('Stopped at the review checkpoint. All completed work is saved; waiting for your decision before continuing.')
    elif status.startswith('paused'):
        st.warning(f'{status}. Saved progress is safe.')
    elif status.startswith('error'): st.error('The runner stopped on an error. Check the runner log before restarting.')
    if state.get('parallel_control'):
        control = json.loads(state['parallel_control'])
        st.caption(f"Automatic concurrency: {control['workers']} agent(s) · ceiling {control['maximum']} · backs off on overload and rate limits")
        timing_rows = []
        for key, value in state.items():
            if key.startswith('parallel_agent:'):
                timing_rows.append({'Agent': key.split(':', 1)[1], **json.loads(value)})
        with st.expander('Execution monitoring'):
            st.caption('Last finished attempt per agent, not a live failure alert. An old error remains here while its retry is running. Use the saved day and current runner status to check recovery. Timings include decisions, memory and any scheduled reflection.')
            st.dataframe(pd.DataFrame(timing_rows), hide_index=True, use_container_width=True)
            recovery_events = data.get('recovery_events', pd.DataFrame())
            st.markdown('**Automatic call recovery**')
            if state.get('operational_retry_revision'):
                st.caption('Enabled: temporary call errors retry after 5 and 15 seconds; tool-budget overrun gets one retry at the same limit. Quota and billing checks still stop or pause the run.')
            if recovery_events.empty:
                st.caption('No call-retry events recorded in this saved checkpoint. Rate-limit backoff is reported separately above.')
            else:
                recovery_rows = []
                for row in recovery_events.itertuples():
                    event = json.loads(row.event_json)
                    recovery_rows.append({'Agent':NAMES.get(row.agent_id,row.agent_id), 'Day':row.sim_day,
                        'Phase':row.phase, 'Event':event['event'], 'Reason':event.get('reason'),
                        'Retry':event.get('retry_number',event.get('retries')), 'Wait (seconds)':event.get('delay_seconds')})
                st.dataframe(pd.DataFrame(recovery_rows), hide_index=True, use_container_width=True)
                st.caption('Historical recovery attempts. Failed calls may consume allowance without returning usage; a retry is not free or proof of success.')
    snapshots={r.agent_id:json.loads(r.snapshot_json) for r in agents.itertuples()}
    for idx,aid in enumerate(agents.agent_id):
        if idx % 4 == 0: cards=st.columns(4)
        col=cards[idx % 4]
        snap=snapshots.get(aid,{})
        g=equity[equity.owner==aid]
        latest_day=int(g.sim_day.max()) if len(g) else 0
        value=float(g.iloc[-1].equity) if len(g) else start
        with col.container(border=True):
            st.markdown(f'### {NAMES[aid]}')
            st.caption(STYLES[aid])
            st.metric('Latest portfolio',f'₹{value/100000:.2f} lakh',f'{(value/start-1):+.2%} since start')
            exposure=float(g.iloc[-1].exposure) if len(g) else 0
            st.caption(f'Day {latest_day} saved · {exposure:.0%} invested · {snap.get("status","Waiting").lower()}')
    st.caption('Cards show the latest saved result for each agent. Comparisons below stop at the last fully completed day.')
    overview,quant,risk,lab,usage,reviews=st.tabs(['Performance','Quant benchmarks','Risk & trades','Inside the agents','Model usage','Reviews'])
    with quant:
        render_quant(equity, fills, list(agents.agent_id), completed, start,
                     DATA_DIR/'anon'/state.get('experiment_id',cfg['experiment_id']))
    known=list(agents.agent_id)+[k for k in NAMES if k.startswith('B_')]
    common=equity[equity.sim_day<=completed].copy()
    # Preserve a stopped agent's final portfolio as a flat terminal result for comparison only.
    for aid,snap in snapshots.items():
        if snap.get('status')=='DEAD':
            g=common[common.owner==aid]
            if len(g) and int(g.sim_day.max())<completed:
                last=g.iloc[-1].to_dict()
                carry=[dict(last,sim_day=d) for d in range(int(g.sim_day.max())+1,completed+1)]
                common=pd.concat([common,pd.DataFrame(carry)],ignore_index=True)
    common=common.sort_values('sim_day',kind='stable')
    table=comparison_frame(common,completed,start,known)
    with overview:
        st.subheader('Who is ahead?')
        if completed==0 or table.empty:
            st.info('The first fair comparison appears when all participants finish day 1. Live progress is shown above.')
        else:
            st.caption(f'All results through day {completed}. Gains include simulated trading fees.')
            baseline_options=[k for k in NAMES if k.startswith('B_')]
            chosen=st.multiselect('Compare with rule-based strategies',baseline_options,default=['B_INDEX'],
                                  format_func=lambda k:NAMES[k],key='baseline_compare')
            owners=list(agents.agent_id)+chosen
            return_figure = history_chart(common,owners,start)
            if intervention_spec:
                return_figure.add_vline(x=50, line_dash='dot', line_color='#F5A65B', annotation_text='Day 50 review · changes from day 51')
                if len(intervention_specs)>1:
                    return_figure.add_vline(x=100,line_dash='dot',line_color='#8ADBB2',annotation_text='Day 100 review · changes from day 101')
            plot(return_figure,'returns')
            ranked=table[table.owner.isin(owners)].sort_values('returns')
            fig=go.Figure(go.Bar(x=ranked.returns,y=ranked.name,orientation='h',
                                marker_color=[COLORS.get(o,'#97A3B3') for o in ranked.owner],
                                text=[f'{v:+.2%}' for v in ranked.returns],textposition='outside',cliponaxis=False,
                                hovertemplate='%{y}: %{x:+.2%}<extra></extra>'))
            finish(fig,'',height=260,percent=False)
            fig.update_layout(xaxis_title='Return since day 1',hovermode='closest',showlegend=False)
            spread=max(float(ranked.returns.max())-float(ranked.returns.min()),.005)
            fig.update_xaxes(tickformat='.1%',dtick=None,zeroline=True,
                             range=[min(0,float(ranked.returns.min()))-spread*.22,
                                    max(0,float(ranked.returns.max()))+spread*.22])
            st.subheader(f'Return ranking · day {completed}')
            plot(fig,'ranking')
            display=table.sort_values('returns',ascending=False).copy()
            bench=display[display.owner=='B_INDEX']
            br=float(bench.iloc[0].returns) if len(bench) else 0
            display['Type']=display.owner.map(lambda o:'AI agent' if o in COLORS and not stub else ('Demo agent' if o in COLORS else 'Fixed rules'))
            display['Return']=display.returns.map(lambda x:f'{x:+.2%}')
            display['Profit / loss']=display.profit.map(lambda x:f'₹{x:+,.0f}')
            display['Portfolio value']=display.equity.map(lambda x:f'₹{x:,.0f}')
            display['Worst fall']=display.drawdown.map(lambda x:f'{x:.2%}')
            display['vs buy & hold']=(display.returns-br).map(lambda x:f'{x*100:+.2f} pp')
            display=display.rename(columns={'name':'Participant','day':'Through day'})
            st.dataframe(display[['Participant','Type','Through day','Portfolio value','Profit / loss','Return','vs buy & hold','Worst fall']],
                         hide_index=True,use_container_width=True)
            with st.expander('How to read these numbers'):
                st.markdown('**Return** is the gain or loss from starting capital. **+2%** on ₹10 lakh is a ₹20,000 gain. '
                            '**vs buy & hold** measures the gap in percentage points. **Worst fall** is the largest drop '
                            'from a previous portfolio peak; closer to zero means a smaller drawdown.')
                st.markdown('**Fixed-rule comparisons:** Buy & hold keeps a stock basket; Cash only makes no trades; '
                            'Equal weight rebalances across stocks; Momentum rules buy recent winners; Reversal rules buy recent losers.')
            st.download_button('Download performance data',display.to_csv(index=False),'performance.csv','text/csv')
            st.divider()
            st.subheader('How did the money put into stocks perform?')
            st.caption('A separate view of trading capital: idle cash is excluded from the denominator. Portfolio return above still measures the outcome for your whole account.')
            cutoff_options=['Latest completed day']+[f'Day {d}' for d in range(50,completed+1,50)]
            cutoff_label=st.selectbox('Investment comparison cutoff',cutoff_options,key='investment_cutoff')
            cutoff=completed if cutoff_label=='Latest completed day' else int(cutoff_label.split()[-1])
            closed_only=st.radio('Count these investments', ['All purchases · include open holdings','Sold shares only · FIFO'],horizontal=True,key='investment_basis').startswith('Sold')
            with st.expander('Formula, examples and comparison limits'):
                st.markdown('**All purchases:** (profit on sold shares + gain/loss on open shares) ÷ total purchase outlay. Purchase outlay includes buy fees; sold-share profit also deducts sell fees. Open shares use saved market values; future selling fees are not estimated.')
                st.markdown('**Sold shares only:** net FIFO profit on shares already sold ÷ their allocated purchase cost, including buy fees. Open holdings are excluded, so this view can hide unfinished losses. A portfolio with no sales has no closed-trade return yet.')
                st.markdown('**Illustrative examples:** invest ₹1 lakh and earn ₹5,000 net → **+5% on deployed money**, but only **+0.5% on a ₹10 lakh account**. Invest ₹9 lakh and earn ₹27,000 → **+3% deployed**, but **+2.7% on the account**. The first has a higher trade-capital percentage; the second made more money.')
                st.markdown('Reinvesting the same ₹1 lakh in a second trade counts ₹2 lakh of cumulative deployment. These are weighted trade-capital results, not annualized returns, equal-risk comparisons or projected returns at 100% exposure. Holding periods, turnover and open losses still matter. All figures run from day 1 to the chosen cutoff, across any interventions.')
            try:
                capital=investment_returns(fills,common,cutoff,start,known)
            except ValueError as error:
                st.error(f'Investment comparison unavailable: {error}')
                capital=pd.DataFrame()
            if not capital.empty:
                rate='closed_return' if closed_only else 'deployed_return'
                amount='closed_basis' if closed_only else 'deployed'
                pnl='closed_net' if closed_only else 'net_pnl'
                eligible=capital[capital[rate].notna()].sort_values(rate,ascending=False)
                st.caption(f'Days 1–{cutoff} · all agents and fixed-rule baselines · '+('sold shares only' if closed_only else 'sold shares plus marked open holdings'))
                if eligible.empty: st.info('No eligible invested capital yet for this view.')
                else: plot(investment_ranking(capital,closed_only),'investment_ranking')
                missing=[NAMES.get(o,o) for o in capital.loc[capital[rate].isna(),'owner']]
                if missing: st.caption('No denominator; excluded from ranking: '+', '.join(missing)+'. This means not applicable, not a 0% investment return.')
                examples=capital[capital.owner.isin(list(agents.agent_id))].sort_values(rate,ascending=False,na_position='last')
                selected_example=st.selectbox('Explain one participant using actual numbers',capital.owner.tolist(),format_func=lambda o:NAMES.get(o,o),index=capital.owner.tolist().index(examples.iloc[0].owner) if len(examples) else 0,key='investment_example')
                x=capital[capital.owner==selected_example].iloc[0]
                a,b,c=st.columns(3)
                a.metric('Purchase cost counted',f'₹{x[amount]:,.0f}')
                b.metric('Net gain / loss counted',f'₹{x[pnl]:+,.0f}')
                c.metric('Gain per ₹100 deployed',f'₹{x[rate]*100:+.2f}' if pd.notna(x[rate]) else 'Not applicable')
                st.write(f"{NAMES.get(selected_example,selected_example)} placed {int(x.buy_orders)} buys and {int(x.sell_orders)} sells, averaging {x.average_invested:.1%} of its portfolio invested. Its whole-account return was {x.portfolio_return:+.2%}. Open positions: {int(x.open_positions)}, with marked gain/loss of ₹{x.open_net:+,.0f} after allocated entry fees.")
                detail=capital.sort_values(rate,ascending=False,na_position='last').copy()
                detail['Participant']=detail.owner.map(lambda o:NAMES.get(o,o))
                detail['Investment return']=detail[rate].map(lambda v:f'{v:+.2%}' if pd.notna(v) else '—')
                for title,col in [('Purchase outlay',amount),('Net gain / loss',pnl),('Open-position gain / loss','open_net')]:detail[title]=detail[col].map(lambda v:f'₹{v:,.0f}')
                detail['Average invested']=detail.average_invested.map(lambda v:f'{v:.1%}')
                detail['Account return']=detail.portfolio_return.map(lambda v:f'{v:+.2%}')
                detail=detail.rename(columns={'buy_orders':'Buys','sell_orders':'Sells','open_positions':'Open positions'})
                st.dataframe(detail[['Participant','Investment return','Purchase outlay','Net gain / loss','Account return','Average invested','Buys','Sells','Open positions','Open-position gain / loss']],hide_index=True,use_container_width=True)
                st.caption('Counts are orders, not independent experiments. Partial sells use only the corresponding FIFO purchase cost. Open positions can still change the outcome; an agent that stopped is valued at its recorded terminal portfolio.')
                st.download_button('Download investment comparison',capital.to_csv(index=False),f'investment-comparison-day-{cutoff}.csv','text/csv',key='investment_download')

    with risk:
        st.subheader('How much risk are they taking?')
        st.caption(f'Compared through completed day {completed}. Stopped agents are held flat at their last recorded value.')
        if common.empty: st.info('Risk charts appear after the first completed day.')
        else:
            left,right=st.columns(2)
            with left:
                st.markdown('**Fall from each portfolio’s peak**')
                plot(history_chart(common,list(agents.agent_id),start,'drawdown'),'drawdown')
                st.caption('A line at −5% means the portfolio is 5% below its previous high.')
            with right:
                st.markdown('**Money invested versus cash**')
                plot(history_chart(common,list(agents.agent_id),start,'exposure'),'exposure')
                st.caption('100% means fully invested in stocks; 0% means all cash.')
        selected=st.selectbox('Explore an agent’s trades',list(agents.agent_id),format_func=lambda a:NAMES.get(a,a),key='risk_agent') if len(agents) else None
        if selected:
            own=fills[(fills.owner==selected)&(fills.sim_day<=completed)]
            matches=fifo_matches(own)
            left,right=st.columns(2)
            with left:
                st.markdown('**How long were sold shares held?**')
                if matches.empty: st.info('No closed holdings yet. This chart appears after the first sale.')
                else:
                    grouped=matches.groupby('holding_days',as_index=False).qty.sum()
                    fig=go.Figure(go.Bar(x=grouped.holding_days,y=grouped.qty,orientation='v',marker_color=COLORS.get(selected)))
                    finish(fig,'Shares sold',height=310)
                    fig.update_xaxes(title='Days held')
                    plot(fig,'holding')
                st.caption('Buys are matched to sells oldest-first (FIFO). Open positions are excluded.')
            with right:
                st.markdown('**Which sectors contributed to profit?**')
                path=DATA_DIR/'anon'/state.get('experiment_id',cfg['experiment_id'])/'bars.parquet'
                if own.empty or not path.exists(): st.info('Sector results appear after trades have been recorded.')
                else:
                    bars=pd.read_parquet(path,columns=['sim_day','asset','sector','close'])
                    sectors=sector_pnl(own,bars,completed).sort_values('net_pnl')
                    fig=go.Figure(go.Bar(x=sectors.net_pnl,y=sectors.sector.str.replace('SECTOR_','Sector '),orientation='h',
                                        marker_color=COLORS.get(selected),hovertemplate='%{y}: ₹%{x:,.0f}<extra></extra>'))
                    finish(fig,'',height=310)
                    fig.update_xaxes(title='Profit / loss (₹)',dtick=None)
                    plot(fig,'sectors')
                st.caption('Includes realized and unrealized profit, after trading fees. Sector names are anonymized.')
            with st.expander('Trade ledger'):
                st.dataframe(own.rename(columns={'sim_day':'Day','asset':'Stock','side':'Action','qty':'Shares','price':'Price','costs':'Fees'}),
                             hide_index=True,use_container_width=True)
    with lab:
        render_agent_detail(agents,fills,decisions,reflections,data['strategy_versions'].sort_values(['agent_id','version']), cfg)
    with usage:
        st.subheader('How much model usage is this taking?')
        all_usage=pd.concat([decisions,reflections],ignore_index=True)
        if all_usage.empty: st.info('Usage appears after the first saved agent episode.')
        else:
            for col in ['input_tokens','output_tokens','cost_inr','sim_day']:
                all_usage[col]=pd.to_numeric(all_usage[col],errors='coerce')
            tokens=all_usage.input_tokens.sum()+all_usage.output_tokens.sum()
            settled=all_usage.cost_inr.notna()
            x,y,z=st.columns(3)
            x.metric('Recorded tokens',f'{tokens:,.0f}')
            y.metric('Saved daily decisions',len(decisions))
            z.metric('Reported charges',f'₹{all_usage.cost_inr.sum():.2f}' if settled.any() else 'Pending')
            st.caption('Latest saved episodes, including the current partial day. This is model usage, not portfolio performance.')
            plot(token_chart(all_usage),'tokens')
            st.caption('Each vertical bar is one agent’s recorded input + output tokens for a simulation day.')
            if not settled.all(): st.info('Some billing records have not settled. Pending charges are unknown, not zero.')
            st.caption('Includes saved decisions, memory and reflections; failed attempts are excluded. ' + ('Local Ollama inference: no provider token charges. Electricity and hardware costs are not measured.' if mode.get('backend')=='ollama' else 'Currency conversion uses ₹88/USD. Cursor Spending is authoritative for billing and allowance.'))
    with reviews:
        if cfg.get('research_design'):
            st.info('Eight-agent experiment: four styles × two conditions. Each agent sees only its own memory and market history available at that decision. The -P agents receive the final-performance pressure prompt.')
            st.caption('Every 10 days: evidence-based self-reflection is saved and fed to that agent’s later decisions. Proposed strategy rewrites await independent review. Every 50 days: stop for review. Previously examined historical data is development data, not an untouched test.')
            st.dataframe(pd.DataFrame([{'Agent':a['name'], 'Style':STYLES.get(a['id'],''), 'Condition':a.get('condition','normal'), 'Paired with':NAMES.get(a.get('paired_with'), '—')} for a in cfg['agents']]),hide_index=True,use_container_width=True)
        st.subheader('Review and intervention timeline')
        timeline = []
        for milestone in range(50, min(end, max(100, pause_day, int(intervention_spec.get('review_day', 100)) if intervention_spec else 100)) + 1, 50):
            exists = (run_dir / 'reviews' / f'day-{milestone:03d}-review.md').exists()
            timeline.append({'Checkpoint':f'Day {milestone}', 'Status':'Reviewed' if exists else 'Ready for review' if completed>=milestone else 'Pending',
                'Evidence window':f'Days {milestone-49}–{milestone} plus cumulative results',
                'Action':next((f"{sp['id']} approved; effective day {sp['effective_day']}" for sp in reversed(intervention_specs) if sp['effective_day']==milestone+1),'Discuss before next resume')})
        st.dataframe(pd.DataFrame(timeline),hide_index=True,use_container_width=True)
        for displayed_spec in intervention_specs:
            st.markdown('**'+displayed_spec['id'].replace('_',' ').title()+' · '+displayed_spec.get('name','Strategy clarification')+'**')
            st.caption(f"Effective day {displayed_spec['effective_day']}; review day {displayed_spec.get('review_day', 100)}. Combined intervention across different periods; before/after results do not isolate causality.")
            changes=[{'Scope':'All agents','Change':x['change'],'Why':x['why']} for x in displayed_spec['global_changes']]
            changes += [{'Scope':x['name'],'Change':x['change'],'Why':x['why']} for x in displayed_spec['agents'].values()]
            st.dataframe(pd.DataFrame(changes),hide_index=True,use_container_width=True)
            with st.expander('Exact approved instructions and preserved history'):
                st.caption(f"Preserved checkpoint: runs/{selected_run}/checkpoints/day-{displayed_spec['effective_day']-1:03d}. Existing trades and returns are preserved.")
                st.write(displayed_spec['protocol'])
                for change in displayed_spec['agents'].values():
                    st.markdown('**'+change['name']+'**'); st.write(change['strategy']);st.write(change['protocol'])
        if intervention_spec:
            phase_rows=[]
            for owner in [a['id'] for a in cfg['agents']] + ['B_INDEX','B_MOM20']:
                g=equity[equity.owner==owner].set_index('sim_day').sort_index()
                for lo,hi,label in [(1,50,'Original')]+[(sp['effective_day'],sp.get('review_day',100),sp['id'].replace('_',' ').title()) for sp in effective_specs]:
                    through=min(hi,completed)
                    window=g[(g.index>=lo)&(g.index<=through)]
                    if window.empty: continue
                    initial=start if lo==1 else float(g.loc[lo-1,'equity'])
                    phase_rows.append({'Participant':NAMES[owner],'Phase':label,'Days':f'{lo}–{through}',
                        'Phase return':f"{float(window.equity.iloc[-1])/initial-1:+.2%}",
                        'Mean invested':f'{window.exposure.mean():.1%}'})
            st.markdown('**Compare phases using each phase’s own starting capital**')
            st.dataframe(pd.DataFrame(phase_rows),hide_index=True,use_container_width=True)
            if intervention_spec.get('policy_version')==2:
                st.caption('Intervention 2: current-day opens hidden; trade feedback includes both fees; reflections saved as proposals; narrative policy memory excluded. No daily trade quota.')
                audit_rows=[]
                for row in data['research_context'].itertuples() if 'research_context' in data else []:
                    record=json.loads(row.context_json)
                    if 'policy_audit' in record:
                        audit=record['policy_audit'];audit_rows.append({'Agent':NAMES.get(row.agent_id,row.agent_id),'Day':row.sim_day,'Eligible but idle':audit['eligible_idle'],'Ineligible buy proposals':len(audit['ineligible_buys']),'Exit candidates not sold':len(audit['exit_candidates_not_sold'])})
                if audit_rows:
                    st.markdown('**Participation checks · eligible idle days need explanation**')
                    st.dataframe(pd.DataFrame(audit_rows).tail(40),hide_index=True,use_container_width=True)
        review_dir = RUNS_DIR / selected_run / 'reviews'
        diagnostics_index = review_dir / 'rule-diagnostics-latest.json'
        if diagnostics_index.exists():
            diagnostic = json.loads(diagnostics_index.read_text())
            with st.expander(f"Rule-by-rule diagnosis · days {diagnostic.get('start_day', 101)}–{diagnostic['cutoff']}"):
                st.caption('Separate offline replays, using completed days only. No diagnostic rule changes are applied to the live agents.')
                findings = review_dir / diagnostic['report'].replace('.md', '-findings.md')
                if findings.exists():
                    st.markdown(findings.read_text())
                diagnosis_text = (review_dir / diagnostic['report']).read_text()
                st.download_button('Download rule-by-rule analysis', diagnosis_text, file_name=diagnostic['report'], mime='text/markdown')
                st.markdown(diagnosis_text)
        reports = sorted(review_dir.glob('day-*-review.md'))
        if reports:
            report_path = st.selectbox('Independent milestone review', reports, index=len(reports)-1, format_func=lambda p: p.stem.replace('-', ' '))
            prefix = report_path.name.removesuffix('-review.md')
            evidence_path = review_dir / f'{prefix}-evidence.json'
            if evidence_path.exists():
                evidence = json.loads(evidence_path.read_text())
                if 'coverage' in evidence:
                    st.caption('Stock opportunity audit · days 51–100 · historical outcomes, not future buy recommendations.')
                    coverage = pd.DataFrame(evidence['coverage'])
                    top_assets = [r['asset'] for r in evidence['rankings'][:5]]
                    weights = coverage[coverage['asset'].isin(top_assets)].pivot(index='name', columns='asset', values='mean_portfolio_weight_pct')
                    st.write('Average portfolio weight in the five strongest stocks over this period (%)')
                    st.dataframe(weights.round(2), use_container_width=True)
                    st.caption('Zero means no end-of-day position. The full report below separates missed gains, avoided losses, sizing and recorded reasons.')
                elif 'summary' in evidence and 'milestone' in evidence:
                    st.caption(f"Frozen evidence through day {evidence['milestone']}. This review does not change the agents.")
                    metric = st.radio('Compare ten-day periods', ['Return', 'Average invested'], horizontal=True, key='review_metric')
                    field = 'return_pct' if metric == 'Return' else 'mean_exposure_pct'
                    owners = ['A1','A2','A3','A4','B_INDEX','B_MOM20']
                    values = [[w[field] for w in evidence['summary'][aid]['windows']] for aid in owners]
                    fig = go.Figure(go.Heatmap(z=values, x=[w['days'] for w in evidence['summary']['A1']['windows']],
                        y=[NAMES[aid] for aid in owners], colorscale='RdBu' if metric == 'Return' else 'Blues',
                        zmid=0 if metric == 'Return' else None, text=[[f'{v:+.2f}%' if metric=='Return' else f'{v:.1f}%' for v in row] for row in values],
                        texttemplate='%{text}', colorbar={'title':'%'}, hovertemplate='%{y} · days %{x}: %{z:.2f}%<extra></extra>'))
                    finish(fig, f'{metric} by ten-day period', height=330)
                    fig.update_xaxes(title='Simulation days');fig.update_yaxes(title='Participant', autorange='reversed')
                    plot(fig, 'review_periods')
                    st.caption(f"Source: saved day-{evidence['milestone']} ledger. Return uses each period’s opening capital; invested is mean end-of-day stock exposure.")
            st.download_button('Download full review', report_path.read_text(), file_name=report_path.name, mime='text/markdown')
            st.markdown(report_path.read_text())
        else:
            st.caption('Independent reviews appear here after a milestone has been analyzed.')
    if len(data['graveyard']):
        with st.expander('Stopped agents'):
            st.dataframe(data['graveyard'][['agent_id','death_day']].rename(columns={'agent_id':'Agent','death_day':'Stopped on day'}),hide_index=True)

render()
