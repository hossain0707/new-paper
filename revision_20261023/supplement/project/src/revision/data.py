"""Observed-only 15-minute forecasting data with explicit provenance and exclusions."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

STEP = 15
HORIZONS = (30, 60, 90)
CHANNELS = ['glucose_mgdl', 'hr_bpm', 'tod_sin', 'tod_cos']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def strict_csv(path, audit):
    """Quarantine malformed records instead of silently skipping shifted columns."""
    records, bad_lines = [], []
    with Path(path).open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                bad_lines.append(line)
            else:
                records.append(row)
    audit[str(Path(path).name)] = {'sha256': sha256(path), 'valid_records': len(records),
                                 'quarantined_line_numbers': bad_lines}
    return pd.DataFrame(records, columns=header).replace('', np.nan)


def _timestamps(s):
    # Offset-free observations during the repeated DST hour cannot be assigned UTC.
    return pd.to_datetime(s.where(s.str.contains(r'(?:[+-]\d{4}|Z)$', na=False)),
                          format='ISO8601', utc=True, errors='coerce')


def load_ddatshr(root):
    root = Path(root)
    audit = {'dataset': 'DDATSHR', 'timezone': 'UTC; offset-required',
             'dst_policy': 'Exclude 27 October; Medtronic until first recorded clock adjustment plus two hours; exclude +/-2h around other adjustments.'}
    abb = strict_csv(root / 'Abbott.csv', audit)
    med = strict_csv(root / 'Medtronic.csv', audit)
    records = []
    for df, col, sensor in [(abb, 'Historic Glucose [mmol/l]', 'Abbott'),
                            (med, 'Sensor Glucose [mmol/l]', 'Medtronic')]:
        ts = _timestamps(df['Local datetime [ISO8601]'])
        g = pd.DataFrame({'PtID': df['Subject code number'].astype(str), 'ts': ts,
                          'glucose_mgdl': pd.to_numeric(df[col], errors='coerce') * 18.0182,
                          'sensor': sensor})
        invalid = ts.isna() | df['Local date [yyyy-mm-dd]'].eq('2019-10-27')
        if sensor == 'Medtronic':
            transitions = df[df['New Device Time'].notna()].copy()
            transitions['ts'] = ts.loc[transitions.index]
            for pt, adj in transitions.groupby('Subject code number'):
                dates = adj.ts.dropna().sort_values()
                after = dates[dates >= pd.Timestamp('2019-10-27', tz='UTC')]
                if len(after):
                    end = after.iloc[0] + pd.Timedelta(hours=2)
                    invalid |= g.PtID.eq(str(pt)) & g.ts.between(pd.Timestamp('2019-10-27', tz='UTC'), end)
                for t in dates:
                    invalid |= g.PtID.eq(str(pt)) & g.ts.between(t-pd.Timedelta(hours=2), t+pd.Timedelta(hours=2))
        audit[sensor + '_excluded_glucose_clock'] = int((invalid & g.glucose_mgdl.notna()).sum())
        g = g[~invalid].dropna(subset=['ts', 'glucose_mgdl'])
        records.append(g)
    glucose = pd.concat(records, ignore_index=True)
    hpath = root / 'Fitbit' / 'Fitbit-heart-rate.csv'
    hr = pd.read_csv(hpath, usecols=['Subject code number', 'Local datetime [ISO8601]', 'heart rate [#/min]'])
    hr = pd.DataFrame({'PtID': hr['Subject code number'].astype(str),
                       'ts': _timestamps(hr['Local datetime [ISO8601]']),
                       'hr_bpm': pd.to_numeric(hr['heart rate [#/min]'], errors='coerce')})
    hr = hr.dropna(subset=['ts', 'hr_bpm'])
    hr = hr[hr.ts.dt.date.astype(str) != '2019-10-27']
    audit['heart_rate_sha256'] = sha256(hpath)
    audit['population_sha256'] = sha256(root / 'population.csv')
    return _grid(glucose, hr, 'DDATSHR'), audit


def load_diadata(root):
    path = Path(root) / 'SDBII_15min_sampling_raw.csv'
    if not path.exists():
        raise FileNotFoundError('Acquire RAW DiaData with scripts/acquire_revision_data.py; preprocessed data are forbidden here.')
    df = pd.read_csv(path)
    audit = {'dataset': 'DiaData', 'sha256': sha256(path),
             'excluded_DDATSHR_subjects': int(df.loc[df.Database.eq('DDATSHR'), 'PtID'].nunique()),
             'timezone': 'as supplied by source; no inferred timezone conversion',
             'upstream': 'raw 15-minute export; no local imputation'}
    df = df[~df.Database.eq('DDATSHR')].copy()
    df['ts'] = pd.to_datetime(df.ts)
    df['PtID'] = df.PtID.astype(str)
    g = df.rename(columns={'GlucoseCGM': 'glucose_mgdl', 'HR': 'hr_bpm'})
    grids = []
    for source, d in g.groupby('Database'):
        glucose = d[['PtID', 'ts', 'glucose_mgdl']].dropna().copy()
        glucose['sensor'] = source
        hr = d[['PtID', 'ts', 'hr_bpm']].dropna()
        grids.append(_grid(glucose, hr, source))
    return pd.concat(grids, ignore_index=True), audit


def _grid(glucose, hr, source):
    parts = []
    for pt, g in glucose.groupby('PtID'):
        g = g.sort_values(['ts', 'sensor']).drop_duplicates('ts', keep='last').set_index('ts')
        # last observation in (t-15min,t] is available at t; no future filling.
        grid = g[['glucose_mgdl', 'sensor']].resample('15min', closed='right', label='right').last()
        h = hr[hr.PtID.eq(pt)].sort_values('ts').drop_duplicates('ts').set_index('ts')
        grid['hr_bpm'] = h.hr_bpm.resample('15min', closed='right', label='right').mean()
        grid['PtID'] = str(pt)
        grid['source'] = source
        parts.append(grid.reset_index())
    return pd.concat(parts, ignore_index=True)


def build_samples(grid, history_minutes=120, min_samples=50, input_lag_steps=0):
    """Identical observed histories and targets for all models; no learned preprocessing."""
    length = history_minutes // STEP + 1
    xs, ys, metadata, retention = [], [], [], []
    for pt, d in grid.groupby('PtID', sort=True):
        d = d.sort_values('ts').reset_index(drop=True)
        clock = d.ts.dt.hour + d.ts.dt.minute / 60
        d['tod_sin'] = np.sin(clock * 2*np.pi/24)
        d['tod_cos'] = np.cos(clock * 2*np.pi/24)
        features = d[CHANNELS].shift(input_lag_steps)
        x = features.to_numpy(dtype=np.float32)
        targets = np.stack([d.glucose_mgdl.shift(-h//STEP) for h in HORIZONS], axis=1)
        complete = np.isfinite(x).all(axis=1)
        history_ok = pd.Series(complete).rolling(length).sum().eq(length).to_numpy()
        valid = history_ok & np.isfinite(targets).all(axis=1)
        ends = np.flatnonzero(valid)
        retained = len(ends) >= min_samples
        source = str(d.source.iloc[0])
        retention.append({'PtID': str(pt), 'source': source, 'grid_rows': len(d),
                          'observed_glucose': int(d.glucose_mgdl.notna().sum()),
                          'observed_hr': int(d.hr_bpm.notna().sum()),
                          'span_days': (d.ts.max()-d.ts.min()).total_seconds()/86400,
                          'eligible_samples': len(ends), 'retained': retained,
                          'sensors': '|'.join(sorted(d.sensor.dropna().unique()))})
        if not retained:
            continue
        xs.append(np.stack([x[e-length+1:e+1] for e in ends]))
        ys.append(targets[ends].astype(np.float32))
        m = d.iloc[ends][['PtID', 'ts', 'source', 'sensor', 'glucose_mgdl', 'hr_bpm']].copy()
        m['glucose_mgdl'] = features.iloc[ends].glucose_mgdl.to_numpy()
        m['hr_bpm'] = features.iloc[ends].hr_bpm.to_numpy()
        m['input_lag_minutes'] = input_lag_steps * STEP
        m['sample_id'] = [f'{pt}|{t.isoformat()}' for t in m.ts]
        metadata.append(m)
    if not xs:
        raise ValueError('No eligible patients')
    return np.concatenate(xs), np.concatenate(ys), pd.concat(metadata, ignore_index=True), pd.DataFrame(retention)


def save_dataset(grid, audit, out, name, history_minutes=120):
    out = Path(out);out.mkdir(parents=True, exist_ok=True)
    x, y, meta, retention = build_samples(grid, history_minutes, input_lag_steps=1 if name=='diadata' else 0)
    np.savez_compressed(out / f'{name}.npz', X=x, y=y)
    meta.to_parquet(out / f'{name}_samples.parquet', index=False)
    retention.to_csv(out / f'{name}_retention.csv', index=False)
    audit.update({'history_minutes': history_minutes, 'channels': CHANNELS, 'samples':len(y),
                  'input_lag_minutes': 15 if name=='diadata' else 0,
                  'patients': int(meta.PtID.nunique()), 'array_sha256': sha256(out / f'{name}.npz')})
    (out / f'{name}_audit.json').write_text(json.dumps(audit, indent=2))
    return audit
