"""Observer-only rule diagnostics. Never imported by the trading loop; no model calls."""
from collections import Counter
from copy import deepcopy
from math import isfinite
from .sim.execution import Order, CostModel, execute_orders
from .sim.portfolio import Portfolio
from .sim.risk import RiskRules

OPS={'gt':lambda a,b:a>b,'ge':lambda a,b:a>=b,'lt':lambda a,b:a<b,'le':lambda a,b:a<=b}

def rule_id(predicate):
    return ' '.join(map(str,predicate))

def failures(row,policy):
    return [rule_id(p) for p in policy['entry'] if row.get(p[0]) is None or not isfinite(row[p[0]]) or not OPS[p[1]](row[p[0]],p[2])]

def exits(row,held,style):
    out=[]
    if style in ('trend','risk') and row['px_over_sma20']<1:out.append('SMA20 break')
    if style=='trend' and row['rel_strength_20d_vs_index']<=0 and (row['rsi_14']<50 or row['ret_5d']<0):out.append('weak relative strength')
    if style=='reversal':
        if row['px_over_sma50']<1:out.append('SMA50 break')
        if row['px_over_sma20']>=1:out.append('SMA20 recovery')
        if held>=10:out.append('10-session horizon')
    if style=='experiment':
        if row['px_over_sma50']<1:out.append('SMA50 break')
        if row['rel_strength_20d_vs_index']<=0 and row['ret_5d']<0:out.append('weak relative strength')
        if held>=10:out.append('10-session horizon')
        if held>=5 and row['ret_5d']<0:out.append('5-session negative momentum')
    return out

def variants(policy):
    """Fixed finite menu: one change per arm, never optimized against results."""
    v=[dict(id='control',kind='control')]
    v += [dict(id='remove entry: '+rule_id(p),kind='entry',predicate=p) for p in policy['entry']]
    if 'market_min_5d' in policy:v.append(dict(id='remove market gate',kind='market'))
    labels={'trend':['SMA20 break','weak relative strength'],'risk':['SMA20 break'],
            'reversal':['SMA50 break','SMA20 recovery','10-session horizon'],
            'experiment':['SMA50 break','weak relative strength','10-session horizon','5-session negative momentum']}
    v += [dict(id='remove exit: '+r,kind='exit',clause=r) for r in labels[policy['exit_style']]]
    v += [dict(id='one extra sleeve, same size',kind='capacity'),dict(id='reverse ranking',kind='rank')]
    if policy['exit_style'] in ('trend','risk'):
        v.append(dict(id='proposed SMA20 cushion 1.01',kind='cushion'))
    return v

def outcome_probe(day,asset,horizon,bars,cost,notional,cutoff):
    """Isolated BUY D open / SELL D+h open probe. Not an executable portfolio claim."""
    if day+horizon>cutoff or (asset,day) not in bars or (asset,day+horizon) not in bars:return None
    buy=round(cost.fill_price(bars[asset,day]['open'],'BUY'),2)
    sell=round(cost.fill_price(bars[asset,day+horizon]['open'],'SELL'),2)
    qty=int(notional//buy)
    if qty<=0:return None
    basis=round(qty*buy,2)+cost.costs(qty*buy,'BUY')
    pnl=round(qty*sell,2)-cost.costs(qty*sell,'SELL')-basis
    return dict(horizon=horizon,exit_day=day+horizon,qty=qty,purchase_outlay=basis,pnl=round(pnl,2),return_pct=pnl/basis*100)

def replay(aid,policy,variant,initial,screens,bars,cfg,cutoff,start_day=101):
    """Deterministic proxy: new cash/inventory path per arm, D-1 decisions, D-open fills."""
    p=deepcopy(policy);pf=deepcopy(initial);risk=RiskRules.from_config(cfg['risk']);cost=CostModel(**cfg['costs'])
    kind=variant['kind']
    if kind=='entry':p['entry'].remove(variant['predicate'])
    if kind=='market':p.pop('market_min_5d',None)
    if kind=='capacity':p['max_positions']=min(p['max_positions']+1,risk.max_positions)
    if kind=='rank':p['descending']=not p.get('descending',False)
    if kind=='cushion':p['entry'].append(['px_over_sma20','ge',1.01])
    marks={a:r['close'] for (a,d),r in bars.items() if d==start_day-1}
    curve=[dict(day=start_day-1,equity=pf.equity(marks),exposure=pf.gross_exposure_frac(marks))];events=[];dead=None
    for day in range(start_day,cutoff+1):
        screen=screens[aid,day];assert screen['as_of_day']==day-1
        rows={r['asset']:r for r in screen['candidates']};marks={a:bars[a,day-1]['close'] for a in rows}
        opens={a:bars[a,day]['open'] for a in rows};closes={a:bars[a,day]['close'] for a in rows}
        if dead is None:
            target=pf.equity(marks)*p['entry_fraction'];sold=set();used=0
            for a,pos in list(pf.positions.items()):
                clauses=exits(rows[a],day-pos.opened_day,p['exit_style'])
                if kind=='exit':clauses=[x for x in clauses if x!=variant['clause']]
                if clauses and used<risk.max_orders_per_day:
                    events+=execute_orders([Order(a,'SELL',None,None,'; '.join(clauses))],pf,opens,marks,risk,cost,day)
                    used+=1;sold.add(a)
            gate=p.get('market_min_5d');m=screen['market']['index'].get('ret_5d')
            allowed=gate is None or (m is not None and m>=gate)
            eligible=[r for a,r in rows.items() if not failures(r,p) and a not in pf.positions and a not in sold] if allowed else []
            eligible.sort(key=lambda r:(r[p['rank']]*(-1 if p.get('descending') else 1),r['asset']))
            while eligible and len(pf.positions)<p['max_positions'] and used<risk.max_orders_per_day:
                # Explicit tie policy: risk agent prefers a different sector if one is eligible;
                # all other agents use exact numeric rank. This is a proxy, not recreated reasoning.
                if aid=='A3':
                    heldsectors={rows[a].get('sector') for a in pf.positions}
                    idx=next((i for i,r in enumerate(eligible) if r.get('sector') not in heldsectors),0)
                else:idx=0
                r=eligible.pop(idx)
                events+=execute_orders([Order(r['asset'],'BUY',None,target,'diagnostic ranked policy')],pf,opens,marks,risk,cost,day);used+=1
            eq=pf.equity(closes)
            if eq<cfg['capital']['starting_cash']*cfg['capital']['death_threshold_frac']:dead=day
            curve.append(dict(day=day,equity=eq,exposure=pf.gross_exposure_frac(closes)))
        else:
            curve.append(dict(day=day,equity=curve[-1]['equity'],exposure=curve[-1]['exposure']))
    periods={}
    for start in sorted({start_day-1,150}):
        if cutoff<=start:continue
        part=[r for r in curve if start<=r['day']<=cutoff];peak=part[0]['equity'];dd=0
        for r in part:peak=max(peak,r['equity']);dd=min(dd,r['equity']/peak-1)
        fills=[f for f in pf.fills if start<f.sim_day<=cutoff]
        periods[f'{start+1}-{cutoff}']=dict(return_pct=(part[-1]['equity']/part[0]['equity']-1)*100,drawdown_pct=dd*100,mean_exposure_pct=sum(r['exposure'] for r in part[1:])/len(part[1:])*100,fees=sum(f.costs for f in fills),buys=sum(f.side=='BUY' for f in fills),sells=sum(f.side=='SELL' for f in fills),turnover=sum(f.gross for f in fills)/part[0]['equity'])
    return dict(variant=variant,periods=periods,death_day=dead,rejections=sum(e['status']!='filled' for e in events),curve=curve,fills=[vars(f) for f in pf.fills if f.sim_day>=start_day])
