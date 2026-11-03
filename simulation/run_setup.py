"""
run_setup.py

One-time setup that runs before the step loop: satellite weather retrieval, forcing-array resampling, downtime/rotation schedule loading, first-run state initialization.
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from jit import njit

from constants import *
from matrix_index_maps import *
from psychrometrics import p_sat_magnus, partial_pressure_from_w, w_from_partial_pressure
from refrigeration_cycle import derive_fixed_hardware_parameters
from fruit_biology import compute_market_scarcity_matrix
from time_calendar import _to_epoch_ms, clean_schedule_timestamps, get_iana_timezone
from utils import evaluate_antoine_fit, numba_clip, fast_vstack_numba, get_persistent_session, bounded_normal_draw

_SESSION = get_persistent_session()

def retrieve_satellite_data(
    rng,
    start_epoch_ms: int,
    end_epoch_ms: int,
    measurements: str,
    timezone_param=None,
    facility_cfg: Union[Dict[str, Any], str, Path] = "configs/facility_runtime_config.json",
    table_schema: Union[Dict[str, Any], str, Path] = "schemas/table_schemas.json",
    active_location_id: int = None,
    frequency: str = "hourly",
    is_backfill: bool = False,
    earliest_date_with_available_data: str = "2021-03-23",
):
    if isinstance(facility_cfg, (str, Path)):
        cfg_path = Path(facility_cfg)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Runtime configuration file not found at: {cfg_path}")
        with cfg_path.open("r") as f:  
            facility_cfg = json.load(f)
    if not isinstance(facility_cfg, dict):
        raise TypeError("facility_cfg must resolve to a valid dictionary source.")

    if isinstance(table_schema, (str, Path)):
        schema_path = Path(table_schema)
        if not schema_path.exists():
            raise FileNotFoundError(f"Runtime configuration file not found at: {schema_path}")
        with schema_path.open("r") as f:  
            table_schema = json.load(f)
    if not isinstance(table_schema, dict):
        raise TypeError("table_schema must resolve to a valid dictionary source.")
    
    location_cfg = facility_cfg["facility_location"]
    if active_location_id is None:
        raise ValueError("retrieve_satellite_data: active_location_id is required")

    location = location_cfg["locations"][str(active_location_id)]

    latitude, longitude = location["latitude"] , location["longitude"]

    soil_temperature_54cm_fallback = location["soil_temperature_54cm_fallback"]

    # datetime_cols = weather_schema["datetime_cols"]

    historical = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    forecast ="https://api.open-meteo.com/v1/forecast"

    start_hour_epoch_ms = (start_epoch_ms // MS_PER_HOUR) * MS_PER_HOUR
    end_hour_epoch_ms   = ((end_epoch_ms - 1) // MS_PER_HOUR) * MS_PER_HOUR + MS_PER_HOUR # 100 msec grain

    start_hour_epoch = start_hour_epoch_ms / 1000
    end_hour_epoch   = end_hour_epoch_ms / 1000

    earliest_epoch_with_available_data = pd.to_datetime(earliest_date_with_available_data).timestamp()

    today_epoch_midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    start_epoch_midnight = (start_hour_epoch // 86400) * 86400
    end_epoch_midnight = (end_hour_epoch // 86400) * 86400

    days_since_start_sim = (today_epoch_midnight - start_epoch_midnight) // 86400
    days_since_end_sim = (today_epoch_midnight - end_epoch_midnight) // 86400
    
    if days_since_start_sim <= 60 and days_since_end_sim <= 60:
        base_url = forecast
    elif days_since_start_sim > 60 and days_since_end_sim > 60:
        base_url = historical
    else:
        base_url = historical
    
    if days_since_end_sim < 4 and base_url == historical:
        print("Please select an end date at least 4 days in the past to ensure historical data availability, or select more recent dates to use forecast data.")
        sys.exit("Error: Exiting program.")

    if start_epoch_midnight < earliest_epoch_with_available_data or end_epoch_midnight < earliest_epoch_with_available_data:
        print(f"Please select start and end dates from {earliest_date_with_available_data} onward to ensure data availability, or select more recent dates to use forecast data.")
        sys.exit("Error: Exiting program.")

    print(f"Using {base_url} endpoint based on selected date range and current date.")
    
    start_hour = datetime.fromtimestamp(start_hour_epoch, tz=timezone.utc)
    end_hour = datetime.fromtimestamp(end_hour_epoch, tz=timezone.utc)

    start_hour_str, end_hour_str = start_hour.strftime("%Y-%m-%dT%H:%M"), end_hour.strftime("%Y-%m-%dT%H:%M")

    params = {
        "start_hour": start_hour_str,
        "end_hour": end_hour_str,
        "latitude": str(latitude),
        "longitude": str(longitude),
        frequency: measurements,
        "wind_speed_unit": "ms",
        "timezone": timezone_param,
    }

    needs_clipping = False
    try:
        response = _SESSION.get(base_url, params=params, timeout=15)
        # Force an exception if the status code is 5xx (includes 504)
        response.raise_for_status() # turns any 4xx or 5xx status code into a requests.exceptions.HTTPError
    except (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.HTTPError
    ) as e:
        base_url = forecast
        print(f"The request failed ({type(e).__name__}): {e}.\nFalling back to {base_url} endpoint.")
        params.pop("start_hour", None)
        params.pop("end_hour", None)
        if is_backfill and days_since_start_sim > 73:
            print(f"Your start date is not available in {base_url} endpoint. Please select a more recent date if the Kestra retries keep failing.")
            raise
        elif is_backfill and days_since_start_sim <= 73:
            params["past_days"] = 73
        else:
            params["past_days"] = 32
        try:
            needs_clipping = True
            response = _SESSION.get(base_url, params=params, timeout=15)
        except Exception as e:
            print(f"An error occurred: {e}. Kestra will retry the data retrieval 5 times. Exception type: {type(e)}")
            raise
    except requests.exceptions.RequestException as e:
        print(f"An error occurred {e}. Kestra will retry the data retrieval 5 times. Exception type: {type(e)}")
        raise

    data = response.json()
    df = pd.DataFrame(data["hourly"])

    # df = df.astype(dtypes)
    datetime_cols = ["time"]
    df[datetime_cols] = df[datetime_cols].apply(
        pd.to_datetime
    )

    # Clip the dataframe if needed
    if needs_clipping:
        df = df[df["time"].between(start_hour_str, end_hour_str)]
        df = df.reset_index(drop=True)

    if df.drop(columns=["soil_temperature_54cm"]).isna().any().any():
        base_url = historical if base_url == forecast else forecast
        print(f"Missing values detected.\nFalling back to {base_url} endpoint.")
        try:
            response = _SESSION.get(base_url, params=params, timeout=15)
            # Force an exception if the status code is 5xx (includes 504)
            response.raise_for_status() # turns any 4xx or 5xx status code into a requests.exceptions.HTTPError
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as e:
            base_url = forecast
            print(f"The request failed ({type(e).__name__}): {e}.\nFalling back to {base_url} endpoint.")
            params.pop("start_hour", None)
            params.pop("end_hour", None)
            if is_backfill and days_since_start_sim > 73:
                print(f"Your start date is not available in {base_url} endpoint. Please select a more recent date if the Kestra retries keep failing.")
                raise
            elif is_backfill and days_since_start_sim <= 73:
                params["past_days"] = 73
            else:
                params["past_days"] = 32
            try:
                needs_clipping = True
                response = _SESSION.get(base_url, params=params, timeout=15)
            except Exception as e:
                print(f"An error occurred: {e}. Kestra will retry the data retrieval 5 times. Exception type: {type(e)}")
                raise
        except requests.exceptions.RequestException as e:
            print(f"An error occurred {e}. Kestra will retry the data retrieval 5 times. Exception type: {type(e)}")
            raise
        data = response.json()
        df = pd.DataFrame(data["hourly"])
        # datetime_cols = ["time"]
        df[datetime_cols] = df[datetime_cols].apply(
            pd.to_datetime
        )
        # df = df.astype(dtypes)

        # Clip the dataframe if needed
        if needs_clipping:
            df = df[df["time"].between(start_hour_str, end_hour_str)]
            df = df.reset_index(drop=True)

    if df.drop(columns=["soil_temperature_54cm"]).isna().any().any():
        raise ValueError("Function failed: Retrieved weather data contains NaN values. Simulation is seeded on weather data, therefore there can't be NaN values.")

    # Rename columns for unit clarity
    df = df.rename( # type: ignore[reportCallIssue]
        columns={
            "time": "timestamp",
            "temperature_2m": "temperature_2m_c",
            "relative_humidity_2m": "relative_humidity_2m_pct",
            "dewpoint_2m": "dewpoint_2m_c",
            "shortwave_radiation": "shortwave_radiation_wm2",
            "wind_speed_10m": "wind_speed_10m_ms",
            "surface_pressure": "surface_pressure_hpa",
            "soil_temperature_54cm": "soil_temperature_54cm_c",
        }
    )

    df["location_id"] = active_location_id
    df["latitude"], df["longitude"] = latitude, longitude

    processed_df = df.copy()

    processed_df["soil_temperature_54cm_c"] = processed_df["soil_temperature_54cm_c"].astype(float)
    # 1. Create a boolean mask for the missing rows
    missing_mask = processed_df["soil_temperature_54cm_c"].isna()
    num_missing = missing_mask.sum() 
    # 2. Generate the noise vector matching that count
    noise = rng.normal(0, 0.1, size=num_missing)
    # 3. Add the fallback scalar to the noise vector and assign it directly using raw values
    processed_df.loc[missing_mask, "soil_temperature_54cm_c"] = soil_temperature_54cm_fallback + noise

    weather_schema = table_schema["weather_dataset"]
    dtypes = weather_schema["dtypes"]

    export_df = enforce_dtypes(df, dtypes, weather_schema["datetime_cols"])
    # Drop unnecessary columns in dataframe for export
    export_df = export_df.drop(columns=["latitude", "longitude"])

    return processed_df.astype(dtypes), export_df


def prepare_forcing_arrays(sim_start_ms, sim_end_ms, weather_data: Union[pd.DataFrame, str, Path] = "weather_data.parquet", facility_cfg: Union[Dict[str, Any], str, Path] = "configs/facility_runtime_config.json"):
    if isinstance(weather_data, (str, Path)):
        data_path = Path(weather_data)
        if not data_path.exists():
            raise FileNotFoundError(f"Runtime configuration file not found at: {data_path}")
        weather_data = pd.read_parquet(data_path, engine="pyarrow")
    if not isinstance(weather_data, pd.DataFrame):
        raise TypeError("weather_data must resolve to a valid pd.DataFrame source.")

    if isinstance(facility_cfg, (str, Path)):
        cfg_path = Path(facility_cfg)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Runtime configuration file not found at: {cfg_path}")
        with cfg_path.open("r") as f:  
            facility_cfg = json.load(f)
    if not isinstance(facility_cfg, dict):
        raise TypeError("facility_cfg must resolve to a valid dictionary source.")

    latitude  = weather_data["latitude"][0]
    longitude = weather_data["longitude"][0]
    df = weather_data.drop(columns=["latitude", "longitude", "location_id"])

    resample_interval_sec = facility_cfg["simulation_runtime_settings"]["forcing_arrays_resample_interval_sec"]

    time_delta = pd.to_timedelta(resample_interval_sec, unit="s")

    df_resampled = (
        df.set_index("timestamp")
          .resample(time_delta)
          .asfreq()
    )

    # 2. SEPARATE: Isolate numeric columns (the ones that CAN be interpolated)
    numeric_cols = df_resampled.select_dtypes(include=["number"]).columns

    # 3 INTERPOLATE: Only the numbers
    df_resampled[numeric_cols] = df_resampled[numeric_cols].interpolate(method="time")

    # 4. STITCH: Reset index to bring "datetime" back as a column
    df_resampled = df_resampled.reset_index()

    # # clip the last row, which corresponds to midnight of day after end date and was used for interpolation. # DEPRECATED
    # df_resampled = df_resampled[:-1]

    # 2. Convert raw epochs into strict timestamps and align to minute grid
    time_delta = pd.to_timedelta(resample_interval_sec, unit="s")
    start_bucket = pd.Timestamp(sim_start_ms, unit="ms").floor(time_delta)
    end_bucket = pd.Timestamp(sim_end_ms, unit="ms").ceil(time_delta)

    # 3. Safe Coverage Check: Compare against your aligned bucket needs
    available_start = df_resampled["timestamp"].iloc[0]
    available_end   = df_resampled["timestamp"].iloc[-1]
    # resampled_start_ms = int(df_resampled["time"].iloc[0].timestamp() * 1000)
    # resampled_end_ms = int(df_resampled["time"].iloc[-1].timestamp() * 1000)

    if available_start > start_bucket or available_end < end_bucket:
        raise ValueError(
            f"Weather data window [{available_start}, {available_end}] does not fully cover "
            f"requested simulation window [{available_start}, {available_end}]."
        )

    
    mask = (df_resampled["timestamp"] >= start_bucket) & \
           (df_resampled["timestamp"] < end_bucket)
    df_resampled = df_resampled[mask].reset_index(drop=True)

    # # Explicit clip to the simulation's actual bounds, not just "whatever the API gave back"
    # mask = (df_resampled["time"] >= pd.Timestamp(sim_start_ms, unit="ms", tz="UTC").tz_localize(None)) & \
    #        (df_resampled["time"] < pd.Timestamp(sim_end_ms, unit="ms", tz="UTC").tz_localize(None))
    # df_resampled = df_resampled[mask].reset_index(drop=True)

    iana_tz_str = get_iana_timezone(latitude, longitude)
    tz = ZoneInfo(iana_tz_str)

    # DST-correct offset per weather-bucket row
    local_dt_series = df_resampled["timestamp"].dt.tz_localize("UTC").dt.tz_convert(tz)
    local_utc_offset_ms_vector = (
        local_dt_series.apply(lambda x: x.utcoffset().total_seconds()) * 1000
    ).to_numpy(np.int64)

    # t_ambient_vec = df_resampled["temperature_2m"].to_numpy(np.float64)
    forcing_matrix = np.zeros((len(local_utc_offset_ms_vector), N_FORCING_FIELDS), dtype=np.float64)

    raw_p_ambient_pa = df_resampled["surface_pressure_hpa"].to_numpy(np.float64) * 100.0 # hPa to Pa

    # Fill forcing_matrix
    forcing_matrix[..., IDX_FORCING_AMBIENT_AIR_TEMP_C]       = df_resampled["temperature_2m_c"].to_numpy(np.float64)
    forcing_matrix[..., IDX_FORCING_AMBIENT_RH_FRAC]          = df_resampled["relative_humidity_2m_pct"].to_numpy(np.float64) / 100.0
    forcing_matrix[..., IDX_FORCING_AMBIENT_DEWPOINT_TEMP_C]  = df_resampled["dewpoint_2m_c"].to_numpy(np.float64)
    forcing_matrix[..., IDX_FORCING_AMBIENT_IRRADIANCE_W_M2]  = df_resampled["shortwave_radiation_wm2"].to_numpy(np.float64)
    forcing_matrix[..., IDX_FORCING_AMBIENT_WIND_SPEED_M_S]   = df_resampled["wind_speed_10m_ms"].to_numpy(np.float64)
    forcing_matrix[..., IDX_FORCING_AMBIENT_SOIL_TEMP_C]      = df_resampled["soil_temperature_54cm_c"].to_numpy(np.float64)

    valid_pressure_mask = raw_p_ambient_pa > SAFE_PRESSURE_THRESHOLD

    p_ambient_vec     = np.full_like(raw_p_ambient_pa, P_ATM_STANDARD)
    inv_p_ambient_vec = np.full_like(raw_p_ambient_pa, 1.0 / P_ATM_STANDARD)
    p_ambient_vec[valid_pressure_mask] = raw_p_ambient_pa[valid_pressure_mask]
    inv_p_ambient_vec[valid_pressure_mask] = 1.0 / raw_p_ambient_pa[valid_pressure_mask]

    forcing_matrix[..., IDX_FORCING_AMBIENT_SURF_PRESSURE_PA] = p_ambient_vec
    forcing_matrix[..., IDX_FORCING_AMBIENT_INV_SURF_PRESSURE_PA] = inv_p_ambient_vec
    
    forcing_start_time_ms = int(df_resampled["timestamp"].iloc[0].timestamp() * 1000)
    
    forcing_data  = dict(
        forcing_matrix     = forcing_matrix,
        local_utc_offset_ms_vector = local_utc_offset_ms_vector,
        forcing_start_time_ms = forcing_start_time_ms,
    )
    return forcing_data, df_resampled
    # return (
    #     forcing_matrix,
    #     local_utc_offset_ms_vector,
    #     df_resampled, # NOTE: For testing, remove later
    # )


def join_downtimes(facility_cfg: Union[Dict[str, Any], str, Path] = "configs/facility_runtime_config.json", location_id: Union[int, str] = None, input_schedule: Union[Dict[str, Any], str, Path] = "configs/downtime_schedule.json", state_file_path: Optional[Union[str, Path]] = "state.npz"):
    if isinstance(facility_cfg, (str, Path)):
        cfg_path = Path(facility_cfg)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Runtime configuration file not found at: {cfg_path}")
        with cfg_path.open("r") as f:  
            facility_cfg = json.load(f)
    if not isinstance(facility_cfg, dict):
        raise TypeError("facility_cfg must resolve to a valid dictionary source.")

    if isinstance(input_schedule, (str, Path)):
        sched_path = Path(input_schedule)
        if not sched_path.exists():
            raise FileNotFoundError(f"Runtime configuration file not found at: {sched_path}")
        with sched_path.open("r") as f:  
            input_schedule = json.load(f)
    if not isinstance(input_schedule, dict):
        raise TypeError("input_schedule must resolve to a valid dictionary source.")

    if location_id is None:
        raise ValueError("join_downtimes: location_id is required")
    location = facility_cfg["facility_location"]["locations"][str(location_id)]

    iana_tz_str = get_iana_timezone(location["latitude"], location["longitude"])
    tz = ZoneInfo(iana_tz_str)

    downtime_rows = [
        [_to_epoch_ms(w["start"], tz), _to_epoch_ms(w["end"], tz)]
        for w in input_schedule["downtime_schedule"]
    ]
    input_downtime_timestamps_ms_matrix = (
        np.array(downtime_rows, dtype=np.int64) if downtime_rows
        else np.empty((0, 2), dtype=np.int64)
    )

    input_crop_rotation_timestamps_ms_vector = np.array(
        [_to_epoch_ms(t, tz) for t in input_schedule.get("crop_rotation_dates", [])], dtype=np.int64
    )
    input_refrigerant_rotation_timestamps_ms_vector = np.array(
        [_to_epoch_ms(t, tz) for t in input_schedule.get("refrigerant_rotation_dates", [])], dtype=np.int64
    )

    if state_file_path is not None:
        state_path = Path(state_file_path)

        if state_path.exists():
            with np.load(state_path) as state:
                input_downtime_timestamps_ms_matrix = np.vstack((
                    input_downtime_timestamps_ms_matrix,
                    state["downtime_timestamps_ms_vector"].reshape(-1, 2),
                ))
                input_crop_rotation_timestamps_ms_vector = np.concatenate((
                    input_crop_rotation_timestamps_ms_vector,
                    state["valid_crop_rotation_timestamps_ms_vector"],
                ))
                input_refrigerant_rotation_timestamps_ms_vector = np.concatenate((
                    input_refrigerant_rotation_timestamps_ms_vector,
                    state["valid_refrigerant_rotation_timestamps_ms_vector"],
                ))

    return dict (
        input_downtime_timestamps_ms_matrix             = input_downtime_timestamps_ms_matrix,
        input_crop_rotation_timestamps_ms_vector        = input_crop_rotation_timestamps_ms_vector,
        input_refrigerant_rotation_timestamps_ms_vector = input_refrigerant_rotation_timestamps_ms_vector,
    )


def initialize_first_run_state(
    rng,
    sim_start_ms,
    forcing_matrix,
    room_configurations_float64_vector,
    equipment_configurations_float64_vector,
    equipment_configurations_int64_vector,
    fruit_parameters_float64_matrix,
    fruit_parameters_int64_matrix,
    refrigerant_properties_matrix,
    refrigerant_surfaces_matrix,
    historical_yield_matrix,
    base_quality_grade_probs_matrix,
    fruit_respiration_curves_matrix,
    room_ephemeral_int64_vector,
    crop_rotation_event_ledger,
    refrigerant_rotation_event_ledger,
    starting_crop_row=0,
    starting_refrigerant_row=0,
    magnus_const_set=ALDUCHOV_LIQ_CONST_SET,
):
    room_state_float64_vector     = np.zeros(N_ROOM_STATE_FLOAT64_FIELDS, dtype=np.float64)
    room_state_int64_vector       = np.zeros(N_ROOM_STATE_INT64_FIELDS, dtype=np.int64)
    pallet_metadata_int64_matrix  = np.zeros((MAX_ACTIVE_PALLETS, N_PALLET_METADATA_FIELDS), dtype=np.int64)
    pallet_timeline_int64_matrix  = np.full((MAX_ACTIVE_PALLETS, N_TIMELINE_FIELDS), SENTINEL_INT64, dtype=np.int64)
    pallet_physics_float64_matrix = np.full((MAX_ACTIVE_PALLETS, N_PALLET_PHYSICS_FIELDS), SENTINEL_FLOAT64, dtype=np.float64)
    inbound_wave_queue_matrix     = np.zeros((MAX_ACTIVE_PALLETS, N_WAVE_QUEUE_FIELDS), dtype=np.int64) # Swapped MAX_QUEUE_SIZE for MAX_ACTIVE_PALLETS
    outbound_wave_queue_matrix    = np.zeros((MAX_QUEUE_SIZE, N_WAVE_QUEUE_FIELDS), dtype=np.int64)
    box_masses_ledger = np.full((MAX_NUM_BOXES, MAX_ACTIVE_PALLETS, N_BOX_LEDGER_FIELDS, N_BOX_STATUS_FIELDS), SENTINEL_FLOAT64, dtype=np.float64)
    box_flags_ledger  = np.zeros((MAX_NUM_BOXES, MAX_ACTIVE_PALLETS, N_BOX_FLAG_FIELDS), dtype=np.int8)
    partial_dispatch_box_temperatures_ledger = np.full((MAX_NUM_BOXES, MAX_ACTIVE_PALLETS), SENTINEL_FLOAT64, dtype=np.float64)
    door_states_int64_matrix   = np.zeros((NUM_DOORS, N_DOOR_STATE_INT64_FIELDS), dtype=np.int64)
    door_states_float64_matrix = np.zeros((NUM_DOORS, N_DOOR_STATE_FLOAT64_FIELDS), dtype=np.float64)
    airflow_pallets_compact_vector          = np.zeros(MAX_ACTIVE_PALLETS, dtype=np.int64)
    inverse_airflow_pallets_lookup_vector   = np.full(MAX_ACTIVE_PALLETS, -1, dtype=np.int64)
    crossing_blobs_compact_vector           = np.zeros(MAX_ACTIVE_PALLETS, dtype=np.int64)
    inverse_crossing_blobs_lookup_vector    = np.full(MAX_ACTIVE_PALLETS, -1, dtype=np.int64)


    room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS] = int(sim_start_ms)
    room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE] = 0  # normal operation
    room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS] = int(sim_start_ms)
    room_state_int64_vector[IDX_ROOM_STATE_NEXT_DEFROST_MS] = (
        int(sim_start_ms) + equipment_configurations_int64_vector[IDX_EQ_DEFROST_INTERVAL_MS]
    )
    room_state_int64_vector[IDX_ROOM_STATE_DEFROST_HEATER_HEALTHY] = 1
    room_state_int64_vector[IDX_ROOM_STATE_EVAP_AIRFLOW_PROOF_GRACE_END_MS] = 0
    room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PROBE_ABOVE_SINCE_MS] = -1
    room_state_int64_vector[IDX_ROOM_STATE_DEFROST_CYCLE_START_MS] = -1
    room_state_int64_vector[IDX_ROOM_STATE_DEFROST_TERMINATION_REASON] = 0
    room_state_float64_vector[IDX_ROOM_STATE_Q_DEFROST_HEATER_W] = 0.0
    room_state_float64_vector[IDX_ROOM_STATE_DEFROST_MELTED_MASS_ACCUM_KG] = 0.0
    room_state_float64_vector[IDX_ROOM_STATE_DEFROST_COIL_MELTED_MASS_ACCUM_KG] = 0.0
    room_state_float64_vector[IDX_ROOM_STATE_DEFROST_PAN_MELTED_MASS_ACCUM_KG] = 0.0
    room_state_float64_vector[IDX_ROOM_STATE_DEFROST_FACE_SHED_MASS_ACCUM_KG] = 0.0
    room_state_float64_vector[IDX_ROOM_STATE_EVAP_AIR_VELOCITY_SENSED_M_S] = 0.0
    room_state_float64_vector[IDX_ROOM_STATE_FACE_BOND_MELTED_KG] = 0.0
    room_state_float64_vector[IDX_ROOM_STATE_M_PAN_ICE_KG] = 0.0
    room_state_float64_vector[IDX_ROOM_STATE_E_PAN_ICE_J] = 0.0

    T_ambient_0  = forcing_matrix[0, IDX_FORCING_AMBIENT_AIR_TEMP_C]
    RH_ambient_0 = forcing_matrix[0, IDX_FORCING_AMBIENT_RH_FRAC]
    P_ambient_0  = forcing_matrix[0, IDX_FORCING_AMBIENT_SURF_PRESSURE_PA]
    room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C] = T_ambient_0
    room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_FILTERED_VAL] = T_ambient_0
    room_state_float64_vector[IDX_ROOM_STATE_CO2_PPM] = room_configurations_float64_vector[IDX_ROOM_OUTDOORS_CO2_PPM]
    room_state_float64_vector[IDX_ROOM_STATE_O2_PCT] = room_configurations_float64_vector[IDX_ROOM_OUTDOORS_O2_PCT]
    room_state_float64_vector[IDX_ROOM_STATE_CO2_PPM_FILTERED_VAL] = room_configurations_float64_vector[IDX_ROOM_OUTDOORS_CO2_PPM]
    room_state_float64_vector[IDX_ROOM_STATE_O2_PCT_FILTERED_VAL] = room_configurations_float64_vector[IDX_ROOM_OUTDOORS_O2_PCT]

    P_sat_room = p_sat_magnus(T_ambient_0, magnus_const_set)
    P_w_room = RH_ambient_0 * P_sat_room
    W_room = w_from_partial_pressure(P_w_room, P_ambient_0)
    room_state_float64_vector[IDX_ROOM_STATE_AIR_HUMIDITY_RATIO_W] = W_room
    room_state_float64_vector[IDX_ROOM_STATE_P_W_ROOM] = partial_pressure_from_w(W_room, P_ambient_0)
    room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM] = numba_clip(P_w_room / P_sat_room, 0.0, 1.0)
    room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM_FILTERED_VAL] = room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM]
    room_state_float64_vector[IDX_ROOM_STATE_W_COIL_EVAP_SAT] = w_from_partial_pressure(P_sat_room, P_ambient_0)

    room_state_float64_vector[IDX_ROOM_STATE_T_EVAP] = T_ambient_0
    room_state_float64_vector[IDX_ROOM_STATE_T_COND] = T_ambient_0
    room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAPORATOR] = T_ambient_0
    room_state_float64_vector[IDX_ROOM_STATE_T_COIL_CONDENSER] = T_ambient_0
    room_state_float64_vector[IDX_ROOM_STATE_T_LIQUID_OUTLET] = T_ambient_0

    room_state_float64_vector[IDX_ROOM_STATE_T_EVAP_AIR] = T_ambient_0
    room_state_float64_vector[IDX_ROOM_STATE_RH_EVAP_AIR] = room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM]

    active_refrigerant_surfaces_matrix = refrigerant_surfaces_matrix[starting_refrigerant_row, ...]
    log_p_evap = evaluate_antoine_fit(T_ambient_0, active_refrigerant_surfaces_matrix[SURFACE_ANTOINE_EVAP, COEFFS, :])
    log_p_cond = evaluate_antoine_fit(T_ambient_0, active_refrigerant_surfaces_matrix[SURFACE_ANTOINE_COND, COEFFS, :])
    room_state_float64_vector[IDX_ROOM_STATE_P_SUCTION_LINE_FILTERED_VAL] = math.exp(log_p_evap)
    room_state_float64_vector[IDX_ROOM_STATE_T_SUCTION_LINE_FILTERED_VAL] = T_ambient_0 + equipment_configurations_float64_vector[IDX_EQ_EEV_CLOSED_VALVE_SH_K]
    room_state_float64_vector[IDX_ROOM_STATE_P_LIQUID_LINE_FILTERED_VAL] = math.exp(log_p_cond)
    room_state_float64_vector[IDX_ROOM_STATE_T_LIQUID_LINE_FILTERED_VAL] = T_ambient_0
    room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_AIR_RETURN_FILTERED_VAL] = T_ambient_0
    room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_AIR_SUPPLY_FILTERED_VAL] = T_ambient_0
    room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAP_FILTERED_VAL] = T_ambient_0
    room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM_AIR_SUPPLY_FILTERED_VAL] = room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM]
    room_state_float64_vector[IDX_ROOM_STATE_ACTIVE_COIL_EFFICIENCY] = 1.0

    # room_state_float64_vector[IDX_ROOM_STATE_V_DOT_AIR_EVAP_ACTUAL] = equipment_configurations_float64_vector[IDX_EQ_EVAP_FLOOR_AIRFLOW_M3_S]
    # room_state_float64_vector[IDX_ROOM_STATE_V_DOT_AIR_COND_ACTUAL] = equipment_configurations_float64_vector[IDX_EQ_COND_FLOOR_AIRFLOW_M3_S]

    T_plant_0 = (
        room_configurations_float64_vector[IDX_ROOM_PLANT_TEMP_COEFF_A] +
        room_configurations_float64_vector[IDX_ROOM_PLANT_TEMP_COEFF_B] * T_ambient_0
    ) 
    RH_plant_0 = (
        room_configurations_float64_vector[IDX_ROOM_PLANT_RH_COEFF_A] + 
        room_configurations_float64_vector[IDX_ROOM_PLANT_RH_COEFF_B] * RH_ambient_0
    )

    # Plant psychrometrics 
    P_sat_plant = p_sat_magnus(T_plant_0, magnus_const_set)
    P_w_plant = RH_plant_0 * P_sat_plant
    W_plant_0 = w_from_partial_pressure(P_w_plant, P_ambient_0)
    R_moist_plant = R_DRY_AIR * (1 + MOISTURE_BUOYANCY_FACTOR * W_plant_0)
    rho_air_plant_0 = P_ambient_0 / (R_moist_plant * (T_plant_0 + 273.15))

    room_state_float64_vector[IDX_ROOM_STATE_ADJACENT_PLANT_TEMP_C] = T_plant_0
    room_state_float64_vector[IDX_ROOM_STATE_ADJACENT_PLANT_RH] = RH_plant_0
    room_state_float64_vector[IDX_ROOM_STATE_ADJACENT_PLANT_AIR_HUMIDITY_RATIO_W] = W_plant_0
    room_state_float64_vector[IDX_ROOM_STATE_ADJACENT_PLANT_RHO_AIR] = rho_air_plant_0

    room_state_int64_vector[IDX_ROOM_STATE_IS_FIRST_RUN_FLAG] = 1
    room_state_int64_vector[IDX_ROOM_STATE_LAST_COMPRESSOR_SWITCH_OFF_MS] = MIN_EPOCH_TIMESTAMP_MS
    room_state_int64_vector[IDX_ROOM_STATE_LAST_COMPRESSOR_SWITCH_ON_MS] = MIN_EPOCH_TIMESTAMP_MS
    room_state_int64_vector[IDX_ROOM_STATE_NEXT_BATCH_ID] = 1
    room_state_int64_vector[IDX_ROOM_STATE_NEXT_INBOUND_DELIVERY_ID] = 1
    room_state_int64_vector[IDX_ROOM_STATE_NEXT_ORDER_ID] = 1
    room_state_int64_vector[IDX_ROOM_STATE_CROP_ROTATION_COUNTER] = starting_crop_row
    room_state_int64_vector[IDX_ROOM_STATE_REFRIGERANT_ROTATION_COUNTER] = starting_refrigerant_row

    room_state_int64_vector[IDX_ROOM_STATE_ACTIVE_FRUIT_TYPE_CODE]  = fruit_parameters_int64_matrix[starting_crop_row, IDX_FRUIT_PARAM_FRUIT_TYPE_CODE]
    room_state_int64_vector[IDX_ROOM_STATE_ACTIVE_REFRIGERANT_CODE] = starting_refrigerant_row 

    room_state_float64_vector[IDX_ROOM_STATE_BULK_PULP_TEMP_C] = T_ambient_0

    room_state_float64_vector[IDX_ROOM_STATE_LEAK_NOISE_FACTOR] = bounded_normal_draw(
        rng   = rng,
        mean  = 1.0,
        scale = 0.1,
        low   = 0.85,
        high  = 1.15,
    )

    active_crop_row = starting_crop_row
    active_refrigerant_row = starting_refrigerant_row
    # Slice fruit parameter arrays
    active_crop_params_float64_vector = fruit_parameters_float64_matrix[active_crop_row, ...]
    active_crop_params_int64_vector = fruit_parameters_int64_matrix[active_crop_row, ...]
    room_state_int64_vector[IDX_ROOM_STATE_ACTIVE_FRUIT_TYPE_CODE] = active_crop_params_int64_vector[IDX_FRUIT_PARAM_FRUIT_TYPE_CODE]
    active_crop_historical_yield_matrix = historical_yield_matrix[active_crop_row, ...]
    # Compute market scarcity matrix
    active_crop_market_scarcity_matrix = compute_market_scarcity_matrix(
        rng,
        active_crop_historical_yield_matrix,
        active_crop_params_float64_vector[IDX_FRUIT_PARAM_MARKET_CORRELATION],
        active_crop_params_float64_vector[IDX_FRUIT_PARAM_MARKET_VOLATILITY_SCALE],
    )

    active_crop_base_grades_probs_matrix = base_quality_grade_probs_matrix[active_crop_row, ...]
    active_crop_respiration_curves_matrix = fruit_respiration_curves_matrix[active_crop_row, ...]

    # crop rotation block
    crop_rotation_row = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_CROP_ROTATION_EVENT_WRITE_IDX]
    crop_rotation_event_ledger[crop_rotation_row, IDX_CROP_ROTATION_EVENT_TIMESTAMP_MS] = sim_start_ms
    crop_rotation_event_ledger[crop_rotation_row, IDX_CROP_ROTATION_EVENT_FRUIT_TYPE_CODE] = active_crop_params_int64_vector[IDX_FRUIT_PARAM_FRUIT_TYPE_CODE]
    room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_CROP_ROTATION_EVENT_WRITE_IDX] = crop_rotation_row + 1

    # Slice refrigerant arrays
    active_refrigerant_properties_vector = refrigerant_properties_matrix[active_refrigerant_row, ...]
    active_refrigerant_surfaces_matrix   = refrigerant_surfaces_matrix[active_refrigerant_row, ...]

    # refrigerant rotation block
    refrigerant_rotation_row = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_REFRIGERANT_ROTATION_EVENT_WRITE_IDX]
    refrigerant_rotation_event_ledger[refrigerant_rotation_row, IDX_REFRIGERANT_ROTATION_EVENT_TIMESTAMP_MS] = sim_start_ms
    refrigerant_rotation_event_ledger[refrigerant_rotation_row, IDX_REFRIGERANT_ROTATION_EVENT_REFRIGERANT_CODE] = starting_refrigerant_row
    room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_REFRIGERANT_ROTATION_EVENT_WRITE_IDX] = refrigerant_rotation_row + 1

    (
        room_state_float64_vector[IDX_ROOM_STATE_RATED_COMPRESSOR_Q_COOLING], 
        room_state_float64_vector[IDX_ROOM_STATE_RATED_COMPRESSOR_W_ELECT],
        room_state_float64_vector[IDX_ROOM_STATE_RATED_COMPRESSOR_REFRIG_M_DOT],
        room_state_float64_vector[IDX_ROOM_STATE_V_DOT_EVAP_FULLY_ON],
    ) = derive_fixed_hardware_parameters(
        room_state_float64_vector,
        equipment_configurations_float64_vector,
        active_crop_params_float64_vector,
        # active_refrigerant_properties_vector,
        active_refrigerant_surfaces_matrix,
    )

    state = dict(
        room_state_float64_vector                   = room_state_float64_vector,
        room_state_int64_vector                     = room_state_int64_vector,
        pallet_metadata_int64_matrix                = pallet_metadata_int64_matrix,
        pallet_timeline_int64_matrix                = pallet_timeline_int64_matrix,
        pallet_physics_float64_matrix               = pallet_physics_float64_matrix,
        inbound_wave_queue_matrix                   = inbound_wave_queue_matrix,
        outbound_wave_queue_matrix                  = outbound_wave_queue_matrix,
        box_masses_ledger                           = box_masses_ledger,
        box_flags_ledger                            = box_flags_ledger,
        partial_dispatch_box_temperatures_ledger    = partial_dispatch_box_temperatures_ledger,
        door_states_int64_matrix                    = door_states_int64_matrix,
        door_states_float64_matrix                  = door_states_float64_matrix,
        active_crop_params_float64_vector           = active_crop_params_float64_vector,
        active_crop_params_int64_vector             = active_crop_params_int64_vector,
        active_crop_historical_yield_matrix         = active_crop_historical_yield_matrix,
        active_crop_market_scarcity_matrix          = active_crop_market_scarcity_matrix,
        active_crop_base_grades_probs_matrix        = active_crop_base_grades_probs_matrix,
        active_crop_respiration_curves_matrix       = active_crop_respiration_curves_matrix,
        active_refrigerant_properties_vector        = active_refrigerant_properties_vector,
        active_refrigerant_surfaces_matrix          = active_refrigerant_surfaces_matrix,
        airflow_pallets_compact_vector              = airflow_pallets_compact_vector, 
        inverse_airflow_pallets_lookup_vector       = inverse_airflow_pallets_lookup_vector,
        crossing_blobs_compact_vector               = crossing_blobs_compact_vector, 
        inverse_crossing_blobs_lookup_vector        = inverse_crossing_blobs_lookup_vector,
    )
    return state

@njit
def _filter_and_expand(
    raw_vec,
    duration_ms,
    start_ms,
    end_ms,
    dt_ms,
    sentinel=MIN_EPOCH_TIMESTAMP_MS,
):
    quantized = np.unique(np.round(raw_vec / dt_ms) * dt_ms).astype(np.int64)
    valid_count = 0
    for i, ts in enumerate(quantized):
        if start_ms <= ts < end_ms:
            valid_count += 1
        else:
            quantized[i] = sentinel
    valid_vec = np.empty(valid_count, dtype=np.int64)
    windows = np.empty((valid_count, 2), dtype=np.int64)
    if valid_count > 0:
        count = 0
        for ts in quantized:
            if ts != sentinel:
                valid_vec[count] = ts
                windows[count, 0] = ts
                windows[count, 1] = ts + duration_ms
                count += 1
    return valid_vec, windows, valid_count

@njit
def process_schedules(
    input_downtime_timestamps_ms_matrix,
    input_crop_rotation_timestamps_ms_vector,
    input_refrigerant_rotation_timestamps_ms_vector,
    room_state_int64_vector,
    room_ephemeral_int64_vector,
    room_configurations_int64_vector,
    sim_start_ms,
    sim_end_ms,
    dt_engine_ms,
    sentinel=MIN_EPOCH_TIMESTAMP_MS,
):
    valid_crop_rotation_timestamps_ms_vector, crop_rotation_windows, _ = _filter_and_expand(
        input_crop_rotation_timestamps_ms_vector,
        room_configurations_int64_vector[IDX_ROOM_LOGISTICS_CROP_ROTATION_DOWNTIME_DURATION_MS],
        sim_start_ms,
        sim_end_ms,
        dt_engine_ms,
        sentinel,
    )
    valid_refrigerant_rotation_timestamps_ms_vector, refrigerant_rotation_windows, _ = _filter_and_expand(
        input_refrigerant_rotation_timestamps_ms_vector,
        room_configurations_int64_vector[IDX_ROOM_LOGISTICS_REFRIG_ROTATION_DOWNTIME_DURATION_MS],
        sim_start_ms,
        sim_end_ms,
        dt_engine_ms,
        sentinel,
    )

    input_downtime_timestamps_ms_matrix = fast_vstack_numba((crop_rotation_windows, refrigerant_rotation_windows, input_downtime_timestamps_ms_matrix))
    # downtime_timestamps_ms_matrix = fast_vstack_numba((refrigerant_rotation_windows, downtime_timestamps_ms_matrix))

    if room_state_int64_vector[IDX_ROOM_STATE_IS_FIRST_RUN_FLAG]:
        # Trigger ramp up by tricking the engine into thinking first epoch is last epoch of a downtime window
        ramp_up_trigger_matrix = np.full((1, 2), MIN_EPOCH_TIMESTAMP_MS, dtype=np.int64)
        ramp_up_trigger_matrix[0, 1] = sim_start_ms + MS_PER_10_MINUTES # let room equilibrate a bit after initialization before turning the refrigeration on
        input_downtime_timestamps_ms_matrix = fast_vstack_numba((ramp_up_trigger_matrix, input_downtime_timestamps_ms_matrix))

    if input_downtime_timestamps_ms_matrix.shape[0] == 0:
        downtime_timestamps_ms_vector = np.empty(0, dtype=np.int64)
        is_downtime_in_simulated_timeframe = False
    else:
        downtime_timestamps_ms_vector, is_downtime_in_simulated_timeframe = clean_schedule_timestamps(
            input_downtime_timestamps_ms_matrix,
            sim_start_ms,
            sim_end_ms,
            dt_engine_ms,
            room_configurations_int64_vector[IDX_ROOM_LOGISTICS_MINIMUM_OPERATION_PERIOD_VIABLE_MS],
            sentinel,
        )

    ### DEBUG ###
    print("valid crop rotation timestamps:", valid_crop_rotation_timestamps_ms_vector)
    print("downtime timestamps:", downtime_timestamps_ms_vector)
    ### DEBUG ###

    room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_IS_DOWNTIME_IN_TIMEFRAME] = is_downtime_in_simulated_timeframe

    return (
        downtime_timestamps_ms_vector,
        valid_crop_rotation_timestamps_ms_vector,
        valid_refrigerant_rotation_timestamps_ms_vector,
    )

# --- Raw daily parquet export, for the weather-ingestion side of the ---
# --- historical backfill pipeline (chunked retrieval, run separately ---
# --- from the daily cadence run). The chunked-retrieval orchestration ---
# --- itself isn't written yet -- these three just do the "split into ---
# --- year=/month=/day= parquet packets for the datalake" part once   ---
# --- a weather dataframe exists.                                      ---

def fix_timestamps(df):
    for col in df.select_dtypes(include="datetime64[ns]").columns:
        df[col] = df[col].astype("datetime64[us]")
    return df


def split_df_into_daily_parquets(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime", drop=False)

    current = df["datetime"].iloc[0]
    end_dt = df["datetime"].iloc[-1]

    days_processed = 0

    while current < end_dt:
        midnight = current + pd.Timedelta(days=1)

        daily_df = df.loc[current:midnight].copy()

        daily_df["loaded_at"] = pd.Timestamp.now()
        daily_df = daily_df.reset_index(drop=True)

        year_str = current.strftime("%Y")
        month_str = current.strftime("%m")
        day_str = current.strftime("%d")

        dir_path = f"satellite/year={year_str}/month={month_str}/day={day_str}"
        os.makedirs(dir_path, exist_ok=True)

        file_name = f"satellite_data.parquet"
        full_path = os.path.join(dir_path, file_name)

        daily_df = fix_timestamps(daily_df)
        daily_df.to_parquet(full_path, engine="pyarrow")

        days_processed += 1
        current += pd.Timedelta(days=1)

    return f"Successfully generated {days_processed} daily Parquet packets."


def enforce_dtypes(df, dtypes, datetime_cols):
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col]).astype("datetime64[us]")
    return df.astype(dtypes)
