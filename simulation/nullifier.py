"""
nullifier.py

Applies simulated data-transmission loss to the scadafied telemetry tables
(scadafier.scadafy_telemetry / scadafy_batch_sensors output -- long tag/timestamp/
value format). NOT applied to the 6 business-record ledgers (orders, pallet
history, compressor/humidifier/door events): those are system-of-record data
where "the record either got logged or it didn't" isn't the same failure mode
as an IoT sensor dropping a reading.

Three independent mechanisms, matching real instrumentation failure patterns:
  1. Background dropout   -- per-reading Bernoulli loss, independent per tag.
     Rate is expressed per hour (not per reading), so it's comparable across
     tags with different polling intervals -- a tag polling every 10s and a
     tag polling every 5min should have the same *rate* of loss over time,
     not the same probability per individual reading.
  2. Per-channel outages   -- one tag goes dark for a contiguous stretch while
     every other tag keeps reporting (a single transmitter/probe/load-cell
     fails and stays down). Independent Poisson-arrival process per tag.
  3. Facility-wide outages -- every tag drops together for a contiguous
     stretch (comms/power loss). One Poisson-arrival process shared across
     all tags in the table.

All three are applied by nulling `value` (setting it to NaN), not by deleting
rows -- this matches how a real historian represents "we know a reading was
due, but it never arrived" (a gap in an otherwise-regular series) versus
scadafier's already-correct decision to drop rows for polls that never
happened engine-side. Downstream (dbt), you decide whether NaN rows get
filtered or kept as an explicit "missing" signal.
"""

import numpy as np
import pandas as pd


def apply_background_dropout(
    df: pd.DataFrame,
    rng: np.random.Generator,
    rate_per_hour: float = 0.5,
    tag_col: str = "tag_name",
    timestamp_col: str = "timestamp_ms",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    rate_per_hour: expected number of dropped readings per hour for a tag,
    given ITS OWN polling interval -- converted internally to a per-reading
    probability using each tag's actual median inter-poll gap, so a fast-
    polling tag and a slow-polling tag lose data at the same time-normalized
    rate rather than the same per-row probability.
    """
    df = df.copy()
    for tag, group in df.groupby(tag_col):
        idx = group.index
        if len(idx) < 2:
            continue
        gaps_hr = np.diff(np.sort(group[timestamp_col].values)) / 3_600_000.0
        median_gap_hr = np.median(gaps_hr) if len(gaps_hr) else 1.0
        # expected drops per hour * hours between readings = per-reading probability
        p_drop = min(1.0, rate_per_hour * max(median_gap_hr, 1e-6))
        drop_mask = rng.random(len(idx)) < p_drop
        df.loc[idx[drop_mask], value_col] = np.nan
    return df


def apply_channel_outages(
    df: pd.DataFrame,
    rng: np.random.Generator,
    outages_per_week: float = 0.3,
    mean_duration_min: float = 120.0,
    tag_col: str = "tag_name",
    timestamp_col: str = "timestamp_ms",
    value_col: str = "value",
) -> pd.DataFrame:
    """One tag goes dark for a window; every other tag unaffected. Independent per tag."""
    df = df.copy()
    for tag, group in df.groupby(tag_col):
        ts = group[timestamp_col].values
        if len(ts) == 0:
            continue
        span_hr = (ts.max() - ts.min()) / 3_600_000.0
        n_weeks = max(span_hr / (24 * 7), 1e-6)
        n_outages = rng.poisson(outages_per_week * n_weeks)
        for _ in range(n_outages):
            start_ms = ts.min() + rng.random() * (ts.max() - ts.min())
            duration_ms = rng.exponential(mean_duration_min) * 60_000
            in_window = group.index[(group[timestamp_col] >= start_ms) & (group[timestamp_col] < start_ms + duration_ms)]
            df.loc[in_window, value_col] = np.nan
    return df


def apply_facility_outages(
    df: pd.DataFrame,
    rng: np.random.Generator,
    outages_per_week: float = 0.1,
    mean_duration_min: float = 30.0,
    timestamp_col: str = "timestamp_ms",
    value_col: str = "value",
) -> pd.DataFrame:
    """Every tag drops together for a window -- comms/power loss."""
    df = df.copy()
    ts_all = df[timestamp_col].values
    if len(ts_all) == 0:
        return df
    span_hr = (ts_all.max() - ts_all.min()) / 3_600_000.0
    n_weeks = max(span_hr / (24 * 7), 1e-6)
    n_outages = rng.poisson(outages_per_week * n_weeks)
    for _ in range(n_outages):
        start_ms = ts_all.min() + rng.random() * (ts_all.max() - ts_all.min())
        duration_ms = rng.exponential(mean_duration_min) * 60_000
        in_window = (df[timestamp_col] >= start_ms) & (df[timestamp_col] < start_ms + duration_ms)
        df.loc[in_window, value_col] = np.nan
    return df


def nullify_telemetry(
    df: pd.DataFrame,
    rng: np.random.Generator,
    background_rate_per_hour: float = 0.5,
    channel_outages_per_week: float = 0.3,
    channel_outage_mean_duration_min: float = 120.0,
    facility_outages_per_week: float = 0.1,
    facility_outage_mean_duration_min: float = 30.0,
    tag_col: str = "tag_name",
    timestamp_col: str = "timestamp_ms",
    value_col: str = "value",
) -> pd.DataFrame:
    """Applies all three mechanisms in order: background -> per-channel -> facility-wide."""
    df = apply_background_dropout(df, rng, background_rate_per_hour, tag_col, timestamp_col, value_col)
    df = apply_channel_outages(df, rng, channel_outages_per_week, channel_outage_mean_duration_min, tag_col, timestamp_col, value_col)
    df = apply_facility_outages(df, rng, facility_outages_per_week, facility_outage_mean_duration_min, timestamp_col, value_col)
    return df
