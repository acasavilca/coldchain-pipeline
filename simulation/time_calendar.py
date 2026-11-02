"""
time_calendar.py

Epoch <-> local-calendar conversions and calendar/seasonal weighting: timezone lookup, epoch<->datetime string conversion, schedule timestamp cleanup, day-of-week/seasonal arrival weighting.
"""

import math
import struct
import base64
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from jit import njit
from timezonefinder import TimezoneFinder

from matrix_index_maps import *
from constants import *
from utils import sort_both_axes
from fruit_biology import draw_fruit_quality_grade

DELETE_MAP = str.maketrans("", "", "-T:Z")

def get_iana_timezone(latitude, longitude, default_tz="UTC"):
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lng=longitude, lat=latitude)
    return tz_name if tz_name is not None else default_tz

def encode_coordinates(lat: float, lon: float) -> str:
    """Converts lat/lon floats into a short URL-safe string key."""
    # "!dd" packs two 64-bit double precision floats (big-endian)
    byte_data = struct.pack("!dd", lat, lon)
    
    # Encode to base64 and remove the trailing "=" padding for a cleaner string
    encoded_str = base64.urlsafe_b64encode(byte_data).decode("utf-8").rstrip("=")
    return encoded_str


@njit
def calculate_time_dependent_weights(
    year_local,
    month_local,
    day_of_week_local,
    historical_yield_matrix,
    day_of_week_weights,
    baseline_year=2022,
):
    # Get year index code relative to baseline year
    year_row_idx = int(year_local - int(baseline_year))

    # Get month idx
    month_idx = int(month_local) - 1

    # If an out-of-bounds historical year drops in, lock to maximum boundary capacity
    if year_row_idx < 0:
        year_row_idx = 0
    elif year_row_idx >= historical_yield_matrix.shape[1]:
        year_row_idx = historical_yield_matrix.shape[1] - 1

    # Extract the base climate yield modifier
    climate_yield_weight = historical_yield_matrix[year_row_idx, month_idx]

    # Extract the weekly labor capacity modifier
    weekly_labor_weight = day_of_week_weights[day_of_week_local]

    # Calculate throughput weight
    throughput_weight = climate_yield_weight * weekly_labor_weight

    return climate_yield_weight, weekly_labor_weight, throughput_weight

@njit
def clean_schedule_timestamps(
    schedule_timestamps,
    start_simulation_timestamp,
    end_simulation_timestamp,
    grain,
    minimum_overlap=0,
    sentinel=SENTINEL_FLOAT64,
):
    if len(schedule_timestamps.ravel()) > 2:
        sorted_schedule_timestamps = sort_both_axes(schedule_timestamps)
        grain = int(grain)
        minimum_overlap = int(minimum_overlap)
        flattened_schedule_timestamps = (np.round(sorted_schedule_timestamps.ravel() / grain) * grain).astype(np.int64)
        valid_flat_indices_arr = np.full(len(flattened_schedule_timestamps), sentinel, dtype=np.int64)

        flat_idx = 0
        valid_flat_indices_arr[0] = 0
        valid_flat_count = 1
        while flat_idx <= len(flattened_schedule_timestamps) - 4:
            if (
                flattened_schedule_timestamps[flat_idx] <= flattened_schedule_timestamps[flat_idx+2]
                and flattened_schedule_timestamps[flat_idx+3] <= flattened_schedule_timestamps[flat_idx+1]
            ):
                flat_idx += 4
                if flat_idx == len(flattened_schedule_timestamps):
                    valid_flat_indices_arr[valid_flat_count] = flat_idx - 3
                    valid_flat_count += 1
                    break
                else:
                    continue
            elif not (
                flattened_schedule_timestamps[flat_idx] <= flattened_schedule_timestamps[flat_idx+3]
                and flattened_schedule_timestamps[flat_idx+2] <= flattened_schedule_timestamps[flat_idx+1] + minimum_overlap
            ):
                valid_flat_indices_arr[valid_flat_count] = flat_idx + 1
                valid_flat_indices_arr[valid_flat_count + 1] = flat_idx + 2
                valid_flat_count += 2
            flat_idx += 2
            if flat_idx == len(flattened_schedule_timestamps) - 2:
                valid_flat_indices_arr[valid_flat_count] = flat_idx + 1
                valid_flat_count += 1

    else:
        flattened_schedule_timestamps = np.sort(schedule_timestamps.flatten())
        valid_flat_indices_arr = np.array([0, 1], dtype=np.int64)
        valid_flat_count = 2

    schedule_in_simulated_timeframe = False

    final_count = 0
    for i in range(0, valid_flat_count - 1, 2):
        start_idx = valid_flat_indices_arr[i]
        end_idx = valid_flat_indices_arr[i + 1]
        if not (
            start_simulation_timestamp <= flattened_schedule_timestamps[end_idx]
            and flattened_schedule_timestamps[start_idx] <= end_simulation_timestamp
        ):
            valid_flat_indices_arr[i] = sentinel
            valid_flat_indices_arr[i + 1] = sentinel
        else:
            if not schedule_in_simulated_timeframe:
                schedule_in_simulated_timeframe = True
            final_count += 2

    # Initialize index for downtime schedule
    final_schedule_timestamps = np.empty(final_count, dtype=np.int64)
    count = 0
    for idx in valid_flat_indices_arr:
        if idx == sentinel:
            continue
        final_schedule_timestamps[count] = flattened_schedule_timestamps[idx]
        count += 1

    return final_schedule_timestamps, schedule_in_simulated_timeframe


def _to_epoch_ms(dt_str, tz_obj):
     # 1. If it has a "+00:00" suffix, split it to isolate the numeric clock time
    if "+" in dt_str:
        dt_str, offset_part = dt_str.split("+", 1)

    # 2. Strip structural symbols instantly using the fast translation map
    cleaned_str = dt_str.translate(DELETE_MAP)

    # 3. Add a decimal pad if it was completely omitted
    if "." not in cleaned_str:
        cleaned_str += ".0"

    # 4. Parse as pure UTC time since it was saved as a UTC timestamp
    dt_local = pd.to_datetime(cleaned_str, format="%Y%m%d%H%M%S.%f").tz_localize(tz_obj) 
    
    # Return the timestamp in milliseconds
    return int(dt_local.timestamp() * 1000)


def _to_datetime_str(ts_ms):
    # utc_dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    # utc_str = utc_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    # Convert milliseconds back to a Pandas Timestamp
    dt = pd.to_datetime(ts_ms, unit="ms", utc=True)

    raw_str = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

    base_time, decimals = raw_str.split(".")

    clean_decimals = decimals.rstrip("0")

    if clean_decimals:
        return f"{base_time}.{clean_decimals}+00:00"
    else:
        return f"{base_time}+00:00"


@njit
def compute_local_calendar_components(epoch_timestamp_ms, local_utc_offset_ms):
    # Shift timestamp to local timezone (e.g. -5h for Chao, Peru)
    local_epoch_ms = epoch_timestamp_ms + local_utc_offset_ms

    # Hour and days since epoch
    d_1970, ms_of_day = divmod(local_epoch_ms, 86_400_000)
    hour = ms_of_day // 3_600_000

    # Day of the week
    day_of_week = (d_1970 + 4) % 7 # Jan 1 1970 was a Thursday, and Thurday is 4 days away from Sunday (day 0)

    # Date and Month
    # Move leap day to the end of the year
    # Get absolute day count since birth of Christ (year 0)
    d_absolute = d_1970 + 719_468 # 719_468 days between birth of Christ and Mar 1, 1970

    # Find current 400-year Gregorian cycle
    cycle, d_cycle = divmod(d_absolute, 146_097) # 146_097 days in Gregorian cycle

    # Calculate year within the Gregorian cycle
    y_internal = (d_cycle - d_cycle // 1460 + d_cycle // 36524 - d_cycle // 146097) // 365
    year = y_internal + (cycle * 400)

    # Calculate day within that specific year
    d_year = d_absolute - (year * 365 + year // 4 - year // 100 + year // 400)

    # Calculate internal month block
    m_internal, remainder_month = divmod(5 * d_year + 2, 153) # 53 days in 5-month block starting on March 1

    # Calculate current day of the month
    # d_zero = d_year - (153 * m_internal + 2) // 5
    d_zero = remainder_month // 5
    weekday_idx = d_zero // 7
    day = d_zero + 1

    # Map internal shifted months back to regular calendar months
    m_div_10, _ = divmod(m_internal, 10)
    month = m_internal + 3 - 12 * m_div_10
    year = year + m_div_10

    # Calculate if it's last weekday
    days_in_month = ((153 * (m_internal + 1) + 2) // 5) - ((153 * m_internal + 2) // 5)
    is_feb = m_internal // 11
    is_leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    days_in_month += is_feb * (is_leap - 3)
    is_last_weekday = ((day + 7) > days_in_month) * 1

    return year, month, day, hour, day_of_week, weekday_idx, is_last_weekday
