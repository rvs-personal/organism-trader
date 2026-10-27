import numpy as np, pandas as pd, pytest
from otrader.sim.clock import Clock
from otrader.data.store import ClockedStore


def synthetic_bars(n_assets=6, days=80, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    names = [f"STK_{i+1:02d}" for i in range(n_assets)] + ["MKT", "VOLX"]
    for a in names:
        px = 100 * (1 + rng.normal(0, 0.01, days + 20)).cumprod()
        for d in range(-19, days + 1):
            i = d + 19
            o = px[i] * (1 + rng.normal(0, 0.003)); c = px[i]
            rows.append({"sim_day": d, "weekday": (d % 5), "asset": a, "sector": "SECTOR_1",
                         "open": round(o, 2), "high": round(max(o, c) * 1.005, 2), "low": round(min(o, c) * 0.995, 2),
                         "close": round(c, 2), "volume": float(rng.integers(1e5, 1e6))})
    return pd.DataFrame(rows)


@pytest.fixture
def store():
    return ClockedStore(bars=synthetic_bars(), clock=Clock(10))
