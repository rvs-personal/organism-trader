"""Read-only analytics on the observer's ledger (never exposed as agent tools)."""
from collections import defaultdict, deque
import pandas as pd


def fifo_matches(fills: pd.DataFrame) -> pd.DataFrame:
    lots = defaultdict(deque)
    matches = []
    for f in fills.sort_values('sim_day', kind='stable').to_dict('records'):
        key = (f['owner'], f['asset'])
        qty = f['qty']
        if f['side'] == 'BUY':
            lots[key].append([f['sim_day'], qty, f['price'], f['costs'] / qty])
        else:
            while qty > 0 and lots[key]:
                lot = lots[key][0]
                n = min(qty, lot[1])
                matches.append(dict(owner=key[0], asset=key[1], buy_day=lot[0], sell_day=f['sim_day'],
                                    holding_days=f['sim_day']-lot[0], qty=n,
                                    net_pnl=n*(f['price']-lot[2]-lot[3]-f['costs']/f['qty'])))
                qty -= n
                lot[1] -= n
                if lot[1] == 0:
                    lots[key].popleft()
    return pd.DataFrame(matches, columns=['owner', 'asset', 'buy_day', 'sell_day', 'holding_days', 'qty', 'net_pnl'])


def sector_pnl(fills: pd.DataFrame, bars: pd.DataFrame, day: int) -> pd.DataFrame:
    """Total marked P&L = net trade cashflow + remaining inventory value, net of all fees."""
    f = fills[fills.sim_day <= day].copy()
    if f.empty:
        return pd.DataFrame(columns=['owner', 'sector', 'net_pnl'])
    latest = bars[bars.sim_day <= day].sort_values('sim_day').drop_duplicates('asset', keep='last').set_index('asset')
    f['signed_qty'] = f.qty.where(f.side == 'BUY', -f.qty)
    f['cashflow'] = f.gross.where(f.side == 'SELL', -f.gross) - f.costs
    g = f.groupby(['owner', 'asset'])[['signed_qty', 'cashflow']].sum().reset_index()
    g['mark'] = g.asset.map(latest.close)
    if g['mark'].isna().any():
        raise ValueError('Some traded assets have no visible mark; sector P&L unavailable.')
    g['sector'] = g.asset.map(latest.sector).fillna('Unknown')
    g['net_pnl'] = g.cashflow + g.signed_qty * g.mark
    return g.groupby(['owner', 'sector'], as_index=False).net_pnl.sum()


def investment_returns(fills, equity, completed_day, starting_cash, owners):
    """Since-start P&L per purchase outlay, including fees and partial FIFO lots.

    Recycled capital counts on each purchase. This is not annualized, time weighted,
    or a same-risk benchmark. Open holdings use the saved portfolio mark at cutoff.
    """
    rows=[]
    visible=fills[fills.sim_day<=completed_day].sort_values('sim_day',kind='stable')
    for owner in dict.fromkeys(owners):
        g=equity[(equity.owner==owner)&(equity.sim_day<=completed_day)].sort_values('sim_day')
        if g.empty: continue
        last=g.iloc[-1];f=visible[visible.owner==owner]
        lots=defaultdict(deque);deployed=closed_basis=closed_net=sale_proceeds=0.;buys=sells=wins=0
        for z in f.itertuples():
            if z.qty<=0:raise ValueError('Nonpositive filled quantity in investment return ledger')
            if z.side=='BUY':
                outlay=float(z.gross+z.costs);deployed+=outlay;buys+=1
                lots[z.asset].append([float(z.qty),outlay/z.qty])
            elif z.side=='SELL':
                qty=float(z.qty);basis=0.
                while qty>0:
                    if not lots[z.asset]:raise ValueError('Incomplete FIFO history for investment returns')
                    lot=lots[z.asset][0];n=min(qty,lot[0]);basis+=n*lot[1];qty-=n;lot[0]-=n
                    if lot[0]==0:lots[z.asset].popleft()
                proceeds=float(z.gross-z.costs);sale_proceeds+=proceeds
                net=proceeds-basis;closed_basis+=basis;closed_net+=net;sells+=1;wins+=net>0
            else:raise ValueError('Unknown side in investment return ledger')
        open_basis=sum(q*cost for queue in lots.values() for q,cost in queue)
        marked_stocks=float(last.equity-last.cash)
        if abs(starting_cash-deployed+sale_proceeds-float(last.cash))>.10 or (open_basis==0 and abs(marked_stocks)>.10):
            raise ValueError('Investment cash/holdings do not reconcile to the saved ledger')
        open_net=marked_stocks-open_basis;total_net=closed_net+open_net
        if abs(total_net-(float(last.equity)-starting_cash))>.10:
            raise ValueError('Investment P&L does not reconcile to portfolio; check flows/marks before comparing')
        rows.append(dict(owner=owner,day=int(last.sim_day),deployed=deployed,net_pnl=total_net,
                         deployed_return=total_net/deployed if deployed>0 else None,
                         closed_basis=closed_basis,closed_net=closed_net,
                         closed_return=closed_net/closed_basis if closed_basis>0 else None,
                         open_basis=open_basis,open_net=open_net,marked_stocks=marked_stocks,
                         average_invested=float(((g.equity-g.cash)/g.equity).mean()),
                         portfolio_return=float(last.equity)/starting_cash-1,buy_orders=buys,
                         sell_orders=sells,winning_sells=wins,open_positions=sum(bool(v) for v in lots.values())))
    return pd.DataFrame(rows)
