import pytest
from otrader.data.store import LookAheadError


def test_history_excludes_today_and_future(store):
    h = store.history("STK_01", 200)
    assert h["sim_day"].max() == store.clock.today - 1


def test_today_open_visible_close_not(store):
    assert store.today_open("STK_01") is not None
    assert store.last_close("STK_01") == store.history("STK_01", 1)["close"].iloc[0]


def test_bar_on_future_raises(store):
    with pytest.raises(LookAheadError):
        store.bar_on("STK_01", store.clock.today + 1)


def test_advancing_clock_reveals_exactly_one_more_bar(store):
    n0 = len(store.history("STK_01", 500))
    store.clock.advance()
    assert len(store.history("STK_01", 500)) == n0 + 1


def test_tools_cannot_see_future(store):
    from otrader.agents.tools import ToolBox
    from otrader.sim.portfolio import Portfolio
    tb = ToolBox(store, Portfolio(cash=1e6), lambda q: [], {})
    bars = tb.get_history({"asset": "STK_02", "lookback": 120})["bars"]
    assert max(b["day"] for b in bars) < store.clock.today
