"""Intervention-2 policy: factual memory, explicit eligibility and proposal-only learning.

The fixed policy is developmental, not an untouched performance test. No automatic buys.
"""
from collections import defaultdict, deque


def enabled(cfg, day):
    spec = cfg.get('_intervention', {})
    return spec.get('policy_version') in (2, 3) and day >= spec['effective_day']


def discretionary(cfg):
    return cfg.get('_intervention', {}).get('allocation_mode') == 'agent_discretion'


def trade_outcomes(fills):
    """FIFO sale events net of both sides' fees, including partial lots and prior phases."""
    lots = defaultdict(deque)
    results = []
    for f in sorted(fills, key=lambda f: f.sim_day):
        if f.side == 'BUY':
            lots[f.asset].append([f.qty, f.price, f.costs/f.qty, f.sim_day])
            continue
        left = f.qty; net = 0.; entry_fees = 0.; first = f.sim_day
        while left:
            if not lots[f.asset]:
                raise ValueError('Incomplete FIFO history for trade feedback')
            lot = lots[f.asset][0]; n = min(left, lot[0]); first = min(first, lot[3])
            entry_fees += n*lot[2]
            net += n*(f.price-lot[1]-lot[2])-f.costs*n/f.qty
            left -= n; lot[0] -= n
            if lot[0] == 0: lots[f.asset].popleft()
        results.append(dict(asset=f.asset, buy_day=first, sell_day=f.sim_day, qty=f.qty,
                            net_pnl_inr=round(net,2), entry_fees_inr=round(entry_fees,2),
                            exit_fees_inr=f.costs))
    return results


def portfolio_feedback(pf, marks):
    result = pf.snapshot(marks)
    result['legacy_sale_side_pnl_excludes_entry_fees'] = result.pop('realized_pnl')
    outcomes = trade_outcomes(pf.fills)
    result['closed_trades_net_of_all_fees'] = outcomes[-20:]
    result['closed_trade_count_all_time'] = len(outcomes)
    result['closed_trade_history_scope'] = 'Most recent 20 sale events; aggregate net P&L includes ALL sales. Full ledger is preserved.'
    result['net_realized_fifo_pnl_inr'] = round(sum(x['net_pnl_inr'] for x in outcomes),2)
    result['note'] = 'FIFO outcomes include both entry and exit fees. Unrealized position P&L excludes entry fees. Full portfolio equity includes all fees.'
    return result


def factual_mind(snap):
    """Exclude policy-bearing old lessons, notes and reflections from decision context."""
    return dict(identity=dict(id=snap.agent_id,name=snap.display_name,persona=snap.persona,age_days=snap.age_days),
                strategy=snap.strategy, policy_status='Fixed operational policy through this review checkpoint. Reflections are proposals only.',
                recent_equity=snap.equity_curve[-10:])


def decision_mind(snap, cfg):
    result = factual_mind(snap)
    if discretionary(cfg):
        result['policy_status'] = 'Strategy style stays fixed for this comparison. Position sizes and cash reserves are your daily decisions, within shared engine limits.'
        result['recent_episodes'] = snap.episodic[-4:]
        result['previous_allocation_note'] = snap.working_memory
        result['memory_instruction'] = 'These are past observations, not permanent sizing limits. Reassess opportunity and cash needs today.'
    return result


def policy_context(screen, pf, marks, aid, cfg):
    spec = cfg['_intervention']['agents'][aid]['policy']
    rows = screen['candidates']; market = screen['market']['index']; qualified=[]
    comparisons={'gt':lambda x,y:x>y,'ge':lambda x,y:x>=y,'lt':lambda x,y:x<y,'le':lambda x,y:x<=y}
    for row in rows:
        failures=[f'{key} {op} {value}' for key,op,value in spec['entry']
                  if row.get(key) is None or not comparisons[op](row[key],value)]
        if not failures:qualified.append(row)
    gate = spec.get('market_min_5d')
    market_open = gate is None or (market.get('ret_5d') is not None and market['ret_5d']>=gate)
    ranked=sorted(qualified,key=lambda x:(x.get(spec['rank'],0)*( -1 if spec.get('descending') else 1),x['asset']))
    available=[r['asset'] for r in ranked if r['asset'] not in pf.positions] if market_open else []
    exits=[]
    byname={r['asset']:r for r in rows}
    for asset,pos in pf.positions.items():
        r=byname.get(asset)
        if r is None:continue
        held=screen['as_of_day']+1-pos.opened_day
        style=spec['exit_style'];reasons=[]
        if style in ['trend','risk'] and r['px_over_sma20']<1:reasons.append('close below SMA20')
        if style=='trend' and r['rel_strength_20d_vs_index']<=0 and (r['rsi_14']<50 or r['ret_5d']<0):reasons.append('relative strength invalidation')
        if style=='reversal':
            if r['px_over_sma50']<1:reasons.append('close below SMA50')
            if r['px_over_sma20']>=1:reasons.append('SMA20 recovery target')
            if held>=10:reasons.append('10-session horizon')
        if style=='experiment':
            if r['px_over_sma50']<1 or (r['rel_strength_20d_vs_index']<=0 and r['ret_5d']<0):reasons.append('experiment invalidated')
            if held>=10 or (held>=5 and r['ret_5d']<0):reasons.append('experiment horizon/negative momentum')
        if reasons:exits.append(dict(asset=asset,reasons=reasons))
    if discretionary(cfg):
        hard = cfg['risk']; eq = pf.equity(marks)
        additions = [r['asset'] for r in ranked if r['asset'] in pf.positions] if market_open else []
        return dict(policy_id=cfg['_intervention']['id'], allocation_mode='agent_discretion',
            market_entry_allowed=market_open, stock_qualifiers=[r['asset'] for r in ranked],
            eligible_new=available, eligible_buys=available+additions,
            slots_before_orders=max(0,hard['max_positions']-len(pf.positions)),
            max_positions=hard['max_positions'], max_position_fraction=hard['max_position_frac'],
            max_gross_exposure_fraction=hard['max_gross_exposure_frac'],
            equity_inr=round(eq,2), cash_inr=round(pf.cash,2), exit_candidates=exits,
            allocation_instruction='Choose BUY notionals, SELL quantities and remaining cash yourself. No fixed starter size, no one/two-position ceiling, no minimum cash reserve and no forced full investment. Existing positions may be increased or reduced under the shared caps. Explain sizing and what observable opportunity would justify spending reserved cash. Numeric rankings are information, not mandatory selections. The engine retains cash, position, gross-exposure and order-count limits.')
    return dict(policy_id=cfg['_intervention']['id'],market_entry_allowed=market_open,
                stock_qualifiers=[r['asset'] for r in ranked],eligible_new=available,
                slots_before_orders=max(0,spec['max_positions']-len(pf.positions)),
                target_entry_inr=round(pf.equity(marks)*spec['entry_fraction'],2),
                max_positions=spec['max_positions'],exit_candidates=exits,
                allocation_instruction='Assess ranked eligible names and deploy the declared sleeve when a sound candidate and capacity exist. No daily trade quota. If abstaining despite eligibility, provide a specific measurable reason. Old market/VOLX/ticker vetoes are retired unless listed in this policy.')


def audit_decision(decision, context):
    buys={x.asset for x in decision.orders if x.side=='BUY'}
    sells={x.asset for x in decision.orders if x.side=='SELL'}
    return dict(buy_assets=sorted(buys),ineligible_buys=sorted(buys-set(context.get('eligible_buys', context['eligible_new']))),
                eligible_idle=bool(context['eligible_new']) and context['slots_before_orders']>0 and not buys,
                exit_candidates_not_sold=[x['asset'] for x in context['exit_candidates'] if x['asset'] not in sells])
