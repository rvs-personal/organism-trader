"""Validate an explicit, append-only daily-data extension for an existing run."""
import hashlib
import json
from pathlib import Path
import pandas as pd


def load_extension(run_dir: Path, original: pd.DataFrame, base_hash: str):
    manifest = run_dir / 'data_extension.json'
    if not manifest.exists():
        return None
    spec = json.loads(manifest.read_text())
    path = Path(spec['bars_path'])
    if spec['base_sha256'] != base_hash:
        raise ValueError('Extension does not match the original dataset')
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['extension_sha256']:
        raise ValueError('Extension checksum changed')
    bars = pd.read_parquet(path)
    last = int(original.sim_day.max())
    prefix = bars[bars.sim_day <= last].reset_index(drop=True)
    pd.testing.assert_frame_equal(original.reset_index(drop=True), prefix)
    future = bars[bars.sim_day > last]
    target = int(spec['n_sim_days'])
    if target <= last or sorted(future.sim_day.unique()) != list(range(last + 1, target + 1)):
        raise ValueError('Extension must append consecutive trading days')
    assets = set(original.asset.unique())
    if future.duplicated(['sim_day', 'asset']).any() or any(
            not (assets - {"VOLX"}).issubset(set(group.asset)) or not set(group.asset).issubset(assets) for _, group in future.groupby('sim_day')):
        raise ValueError('Extension must cover the unchanged asset universe each day')
    values = future[['open', 'high', 'low', 'close', 'volume']]
    import numpy as np
    if not np.isfinite(values.to_numpy()).all() or (values[['open','high','low','close']] <= 0).any().any() or (values.volume < 0).any():
        raise ValueError('Extension contains invalid market values')
    return bars, target
