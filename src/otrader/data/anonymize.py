"""Anonymize real market data so the LLM cannot recognize the period or the companies.

Symbols  -> STK_xx (salted shuffle)
Sectors  -> SECTOR_n
Index    -> MKT, VIX -> VOLX
Dates    -> sim_day 1..N (weekday preserved); real dates dropped from agent-facing tables
Prices   -> multiplied by a per-symbol constant (returns preserved exactly)
Volumes  -> multiplied by a per-symbol constant

The mapping is written to mapping.json. The agent-facing store never contains it.

Run:  python -m otrader.data.anonymize
"""
from __future__ import annotations
import json
import random
from pathlib import Path
import pandas as pd
from ..config import DATA_DIR, load_experiment, load_universe


def build_mapping(symbols: list[str], sectors: dict[str, str], index_sym: str, vix_sym: str,
                  seed: int, price_range, volume_range) -> dict:
    rng = random.Random(seed)
    shuffled = sorted(symbols)
    rng.shuffle(shuffled)
    sym_map = {s: f"STK_{i+1:02d}" for i, s in enumerate(shuffled)}
    sym_map[index_sym] = "MKT"
    sym_map[vix_sym] = "VOLX"
    uniq_sectors = sorted(set(sectors.values()))
    rng.shuffle(uniq_sectors)
    sector_map = {s: f"SECTOR_{i+1}" for i, s in enumerate(uniq_sectors)}
    price_scale = {s: round(rng.uniform(*price_range), 6) for s in sym_map}
    price_scale[vix_sym] = 1.0  # VIX level stays readable; it is a volatility index
    volume_scale = {s: round(rng.uniform(*volume_range), 6) for s in sym_map}
    return {"seed": seed, "symbols": sym_map, "sectors": sector_map,
            "price_scale": price_scale, "volume_scale": volume_scale}


def anonymize_frame(bars: pd.DataFrame, mapping: dict, day_index: dict) -> pd.DataFrame:
    out = bars[bars["date"].isin(day_index.keys())].copy()
    out["sim_day"] = out["date"].map(day_index).astype(int)
    out["weekday"] = out["date"].dt.weekday
    out["asset"] = out["symbol"].map(mapping["symbols"])
    out["sector_anon"] = out["sector"].map(lambda s: mapping["sectors"].get(s, s))
    ps = out["symbol"].map(mapping["price_scale"]).astype(float)
    vs = out["symbol"].map(mapping["volume_scale"]).astype(float)
    for c in ["open", "high", "low", "close"]:
        out[c] = (out[c] * ps).round(2)
    out["volume"] = (out["volume"] * vs).round(0)
    return (out[["sim_day", "weekday", "asset", "sector_anon", "open", "high", "low", "close", "volume"]]
            .rename(columns={"sector_anon": "sector"})
            .sort_values(["sim_day", "asset"]).reset_index(drop=True))


def anonymize(exp: dict | None = None, uni: dict | None = None,
              raw_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    exp = exp or load_experiment()
    uni = uni or load_universe()
    raw_dir = raw_dir or DATA_DIR / "raw" / exp["experiment_id"]
    out_dir = out_dir or DATA_DIR / "anon" / exp["experiment_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = pd.read_parquet(raw_dir / "bars.parquet")
    bars["date"] = pd.to_datetime(bars["date"])
    symbols = sorted(s for s in bars["symbol"].unique() if not str(s).startswith("^"))
    sectors = {s["symbol"]: s["sector"] for s in uni["stocks"]}
    a = exp["anonymize"]
    mapping = build_mapping(symbols, sectors, uni["index"], uni["vix"], int(a["seed"]),
                            a["price_scale_range"], a["volume_scale_range"])

    # Simulation calendar = index trading days in [start, end]; warm-up days get sim_day <= 0
    idx_days = sorted(bars.loc[bars["symbol"] == uni["index"], "date"].unique())
    start, end = pd.Timestamp(exp["start_date"]), pd.Timestamp(exp["end_date"])
    sim_days = [d for d in idx_days if start <= d <= end][: int(exp["n_sim_days"])]
    warm = [d for d in idx_days if d < start][-int(exp["lookback_days"]):]
    calendar = list(warm) + list(sim_days)
    day_index = {pd.Timestamp(d): i - len(warm) + 1 for i, d in enumerate(calendar)}

    agent_facing = anonymize_frame(bars, mapping, day_index)
    agent_facing.to_parquet(out_dir / "bars.parquet", index=False)

    mapping["calendar"] = {str(k.date()): v for k, v in day_index.items()}
    mapping["n_sim_days"] = len(sim_days)
    (out_dir / "mapping.json").write_text(json.dumps(mapping, indent=2))
    (out_dir / "README.txt").write_text(
        "bars.parquet is agent-facing. mapping.json de-anonymizes; NEVER expose it to agents.\n")
    return out_dir


if __name__ == "__main__":
    print(anonymize())
