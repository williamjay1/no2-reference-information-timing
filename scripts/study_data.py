"""Data access boundary and deterministic hourly deployment preparation."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_data():
    # Keep sensor availability independent of reference availability.
    raw = pd.read_parquet(ROOT/'datasets/no2_hourly_oob.parquet')
    raw.loc[raw.sensor_flag.eq('Error'), 'x'] = np.nan
    raw.loc[~raw.reference_version.eq('Ratified'), 'y'] = np.nan
    raw.loc[(raw['T'] < -40) | (raw['T'] > 65) | raw.T_flag.eq('Error'), 'T'] = np.nan
    raw.loc[(raw.RH < 0) | (raw.RH > 100) | raw.RH_flag.eq('Error'), 'RH'] = np.nan
    return raw


def deployment_arrays(group, stop=None, start=None, weather=True):
    group = group.sort_values('timestamp')
    a = pd.Timestamp(start) if start is not None else group.timestamp.min()
    b = pd.Timestamp(stop) if stop is not None else group.timestamp.max()
    b = min(b, group.timestamp.max())
    idx = pd.date_range(a, b, freq='h')
    g = group.set_index('timestamp').reindex(idx)
    X = g[['x','T','RH'] if weather else ['x']].to_numpy(dtype=float)
    y = g.y.to_numpy(dtype=float)
    warning = g.sensor_flag.eq('Warning').to_numpy()
    return dict(X=X, y=y, times=idx, warning=warning,
                site=group.site.iloc[0], device=group.device.iloc[0],
                deployment=group.deployment.iloc[0], family=group.family.iloc[0])


def eligible(arr, init_end=336, eval_start=None, minimum=672):
    t = arr['times']
    train = np.isfinite(arr['X'][:init_end,0]) & np.isfinite(arr['y'][:init_end])
    score = np.arange(len(t)) >= init_end
    if eval_start:
        score &= t >= pd.Timestamp(eval_start)
    score &= np.isfinite(arr['X'][:,0]) & np.isfinite(arr['y'])
    return int(train.sum()) >= 72 and int(score.sum()) >= minimum, score
