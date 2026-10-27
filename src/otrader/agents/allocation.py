"""Read-only allocation arithmetic at known closes; never changes agent order choices."""
from math import isfinite


def preview_allocation(orders, portfolio, marks, costs):
    cash = portfolio.cash
    held = {asset: pos.qty for asset, pos in portfolio.positions.items()}
    rows = []
    for order in orders:
        asset, side = order['asset'], order['side']
        if asset not in marks:
            raise ValueError(f'{asset}: no known price; unknown or unavailable asset')
        if side == 'BUY':
            amount = float(order.get('notional_inr') or 0)
            if not isfinite(amount) or amount <= 0 or order.get('qty') is not None:
                raise ValueError('BUY needs positive notional_inr and qty null')
            cash -= amount + costs.costs(amount, 'BUY')
        elif side == 'SELL':
            qty = order.get('qty')
            qty = held.get(asset, 0) if qty is None else qty
            if not isinstance(qty, int) or qty <= 0 or qty > held.get(asset, 0):
                raise ValueError(f'{asset}: cannot sell {qty} shares; hold {held.get(asset,0)}')
            held[asset] -= qty
            gross = qty * marks[asset]
            cash += gross - costs.costs(gross, 'SELL')
        else:
            raise ValueError('Order side must be BUY or SELL')
        rows.append(dict(asset=asset,side=side,estimated_cash_after_inr=round(cash,2)))
    return dict(starting_cash_inr=round(portfolio.cash,2),estimated_cash_reserve_inr=round(cash,2),
                estimated_invested_fraction=round(1-cash/portfolio.equity(marks),4),
                affordable_in_order=all(r['estimated_cash_after_inr'] >= 0 for r in rows),orders=rows,
                note='Estimate using known previous closes and requested BUY amounts plus fees. Real fills use unseen opens and whole shares. No orders are placed by this calculator.')


PREVIEW_SPEC = dict(description='Calculator: check your proposed order amounts and the cash reserve before submitting a decision. Uses current cash, known closes and estimated fees. It never places trades.', input_schema=dict(type='object',required=['orders'],properties={'orders':dict(type='array',items=dict(type='object',required=['asset','side'],properties={'asset':{'type':'string'},'side':{'enum':['BUY','SELL']},'notional_inr':{'type':['number','null']},'qty':{'type':['integer','null']}}))}))
