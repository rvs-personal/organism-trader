"""Write synthetic 'raw' bars in the exact format fetch.py produces, for offline testing.
    python scripts/make_synthetic.py && python -m otrader.data.anonymize
"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from otrader.config import load_experiment, load_universe, DATA_DIR
exp, uni = load_experiment(), load_universe()
rng = np.random.default_rng(3)
dates = pd.bdate_range(pd.Timestamp(exp["start_date"]) - pd.Timedelta(days=exp["lookback_days"] * 2), exp["end_date"])
frames = []
for s in uni["stocks"] + [{"symbol": uni["index"], "sector": "INDEX"}, {"symbol": uni["vix"], "sector": "VIX"}]:
    base = 15.0 if s["symbol"] == uni["vix"] else rng.uniform(200, 4000)
    px = base * (1 + rng.normal(0.0003, 0.012, len(dates))).cumprod()
    o = px * (1 + rng.normal(0, 0.003, len(dates)))
    frames.append(pd.DataFrame({"date": dates, "open": o, "high": np.maximum(o, px) * 1.006, "low": np.minimum(o, px) * 0.994,
                                "close": px, "volume": rng.integers(1e5, 5e6, len(dates)).astype(float),
                                "symbol": s["symbol"], "sector": s["sector"]}))
out = DATA_DIR / "raw" / exp["experiment_id"]; out.mkdir(parents=True, exist_ok=True)
pd.concat(frames).to_parquet(out / "bars.parquet", index=False)
print("synthetic raw written to", out)
