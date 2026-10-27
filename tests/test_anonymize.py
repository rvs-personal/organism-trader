import pandas as pd, numpy as np
from otrader.data.anonymize import build_mapping, anonymize_frame


def test_returns_preserved_and_names_hidden():
    dates = pd.bdate_range("2024-01-01", periods=30)
    px = 100 * (1 + np.random.default_rng(0).normal(0, 0.01, 30)).cumprod()
    bars = pd.DataFrame({"date": dates, "open": px, "high": px, "low": px, "close": px, "volume": 1e6,
                         "symbol": "RELIANCE", "sector": "Energy"})
    m = build_mapping(["RELIANCE"], {"RELIANCE": "Energy"}, "^NSEI", "^INDIAVIX", 7, (0.5, 2.0), (0.5, 2.0))
    day_index = {pd.Timestamp(d): i + 1 for i, d in enumerate(dates)}
    out = anonymize_frame(bars, m, day_index)
    assert set(out["asset"]) == {"STK_01"} and "RELIANCE" not in out.to_string()
    r_real = pd.Series(px).pct_change().dropna().round(6).values
    r_anon = out.sort_values("sim_day")["close"].pct_change().dropna().round(6).values
    assert np.allclose(r_real, r_anon, atol=1e-3)
