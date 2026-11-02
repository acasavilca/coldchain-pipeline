"""
run.py

Final, non-debug entry point for a simulation run. This is new_run.py with
the debugging cruft (matplotlib, the stop() cell-killer, the post-run
downtime-spike inspection loop) stripped out -- everything below is setup
that feeds into a single run_simulator() call.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from constants import *
from matrix_index_maps import *
from time_calendar import get_iana_timezone, _to_epoch_ms, _to_datetime_str
from array_builders import (
    load_json_files,
    build_all_parameter_arrays,
    build_ephemeral_arrays,
)
from run_setup import (
    retrieve_satellite_data,
    prepare_forcing_arrays,
    join_downtimes,
    initialize_first_run_state,
    process_schedules,
    enforce_dtypes,
    fix_timestamps,
)
from engine import run_simulator
from state_persistence import save_run_state, load_run_state
from run_outputs_io import save_run_outputs, trim_run_outputs, _find_chunks
from scadafier import scadafy_all_outputs
from nullifier import nullify_telemetry

RUN_DUMMY = 0

NUM_SIMUL_HOURS = 24 * 3 # 24*7 + 4

STARTING_FRUIT_TYPE = FRUIT_TYPE_BLUEBERRY # FRUIT_TYPE_BLUEBERRY FRUIT_TYPE_AVOCADO
STARTING_REFRIGERANT = REFRIGERANT_R404A # REFRIGERANT_R404A REFRIGERANT_R134A

SEND_REQUEST = 0

if RUN_DUMMY:
    SEND_REQUEST = 0

# Load configs and databases
configs = load_json_files("./configs")
facility_cfg      = configs["facility_runtime_config"]
equipment_catalog = configs["catalog_equipment_config"]
respiration_db    = configs["fruit_respiration_curves"]

schemas = load_json_files("./schemas")
table_schemas = schemas["table_schemas"]

active_location_id = facility_cfg["facility_location"]["active_location_id"]
location = facility_cfg["facility_location"]["locations"][str(active_location_id)]
runtime_settings = facility_cfg["simulation_runtime_settings"]

# Resume from a previous run's saved state if it exists, else cold-start.
resumed = load_run_state()
if resumed is not None:
    state_arrays, rng = resumed
    saved_float_count = state_arrays["room_state_float64_vector"].size
    saved_int_count = state_arrays["room_state_int64_vector"].size
    if saved_float_count != N_ROOM_STATE_FLOAT64_FIELDS or saved_int_count != N_ROOM_STATE_INT64_FIELDS:
        raise ValueError(
            "Saved room-state vectors have obsolete lengths "
            f"({saved_float_count} float, {saved_int_count} int); current layout needs "
            f"({N_ROOM_STATE_FLOAT64_FIELDS} float, {N_ROOM_STATE_INT64_FIELDS} int). "
            "Use a fresh state or explicitly migrate this checkpoint."
        )
    print("Resuming from saved state.npz / rng_state.json.")
else:
    rng = np.random.default_rng(runtime_settings["rng_seed"])
    state_arrays = None  # built below, after parameter_arrays exist

iana_tz_str = get_iana_timezone(location["latitude"] , location["longitude"])
tz_obj = ZoneInfo(iana_tz_str)

# START_LOCAL_DATE only matters for the very first run ever (Kestra's
# "start of operations" parameter, cold-start only) -- every run after that
# must continue from exactly where the previous run's state left off, not
# from a re-typed date. Local dates/timezone only get used internally
# (fatigue factors, supervisor schedules); everything else runs on epoch ms
# and stays timezone-agnostic all the way to the marts.
START_LOCAL_DATE = "2025-05-01T03:00:00" # "2025-09-29T03:00:00" # "2025-05-01T03:00:00" # "2024-12-30T00:00:00"

if resumed is not None:
    start_ms = int(state_arrays["room_state_int64_vector"][IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS])
else:
    start_ms = _to_epoch_ms(str(START_LOCAL_DATE), tz_obj)
end_ms = int(start_ms + MS_PER_HOUR * NUM_SIMUL_HOURS)

WEATHER_FILE_PATH = "weather_data.parquet"

print("start:", _to_datetime_str(start_ms), "\n", "end:", _to_datetime_str(end_ms))
if SEND_REQUEST:
    weather_data_df, raw_weather_data_df = retrieve_satellite_data(    
        rng                = rng,    
        start_epoch_ms     = start_ms,    
        end_epoch_ms       = end_ms,    
        measurements       = runtime_settings["required_weather_measurements"],
        active_location_id = active_location_id,
        facility_cfg       = facility_cfg,
        table_schema       = table_schemas,
      )

    weather_data_df.to_parquet(WEATHER_FILE_PATH, engine="pyarrow")
    raw_weather_data_df.to_parquet("raw_weather_data.parquet", engine="pyarrow")
else:
    weather_data_df = pd.read_parquet(WEATHER_FILE_PATH, engine="pyarrow")

if RUN_DUMMY:
    end_ms = start_ms + int(4 * 60e3)

forcing_data, weather_resampled_df = prepare_forcing_arrays(
    sim_start_ms = start_ms,
    sim_end_ms   = end_ms,
    facility_cfg = facility_cfg,
    weather_data = weather_data_df, # type: ignore
)

print(weather_resampled_df)
# print(forcing_data["forcing_matrix"][:, IDX_FORCING_AMBIENT_RH_FRAC])
print("start:", _to_datetime_str(start_ms), "\n", "end:", _to_datetime_str(end_ms))
parameter_arrays = build_all_parameter_arrays(
    facility_cfg      = facility_cfg,
    equipment_catalog = equipment_catalog,
    respiration_db    = respiration_db,
)
print(parameter_arrays["equipment_configurations_int64_vector"])
ephemeral_arrays = build_ephemeral_arrays(
    start_ms,
    end_ms,
    parameter_arrays["base_quality_grade_probs_matrix"],
    facility_cfg,
)
if state_arrays is None:
    state_arrays = initialize_first_run_state(
        rng,
        start_ms,
        forcing_data["forcing_matrix"],
        parameter_arrays["room_configurations_float64_vector"],
        parameter_arrays["equipment_configurations_float64_vector"],
        parameter_arrays["equipment_configurations_int64_vector"],
        parameter_arrays["fruit_parameters_float64_matrix"],
        parameter_arrays["fruit_parameters_int64_matrix"],
        parameter_arrays["refrigerant_properties_matrix"],
        parameter_arrays["refrigerant_surfaces_matrix"],
        parameter_arrays["historical_yield_matrix"],
        parameter_arrays["base_quality_grade_probs_matrix"],
        parameter_arrays["fruit_respiration_curves_matrix"],
        ephemeral_arrays["room_ephemeral_int64_vector"],
        ephemeral_arrays["crop_rotation_event_ledger"],
        ephemeral_arrays["refrigerant_rotation_event_ledger"],
        starting_crop_row=STARTING_FRUIT_TYPE,
        starting_refrigerant_row=STARTING_REFRIGERANT,
        magnus_const_set=ALDUCHOV_LIQ_CONST_SET,
    )

# join_downtimes() already folds any leftover rows saved in state.npz
# (downtime_timestamps_ms_vec / valid_crop_rotation_timestamps_ms_vec /
# valid_refrigerant_rotation_timestamps_ms_vec, written by save_run_state()
# at the end of a prior run) in with today's config-declared schedule.
input_schedule_arrays = join_downtimes(facility_cfg)

# process_schedules() used to run inside run_simulator(), which meant its
# three outputs never left the njit function's local scope. Calling it here
# instead lets us assign its outputs straight into state_arrays -- so
# run_simulator gets them via **state_arrays like everything else (no
# separate kwargs, no return statement needed: run_simulator mutates these
# in place same as every other state array), and save_run_state() below
# just persists the one dict.
(
    state_arrays["downtime_timestamps_ms_vector"],
    state_arrays["valid_crop_rotation_timestamps_ms_vector"],
    state_arrays["valid_refrigerant_rotation_timestamps_ms_vector"],
) = process_schedules(
    input_schedule_arrays["input_downtime_timestamps_ms_matrix"],
    input_schedule_arrays["input_crop_rotation_timestamps_ms_vector"],
    input_schedule_arrays["input_refrigerant_rotation_timestamps_ms_vector"],
    state_arrays["room_state_int64_vector"],
    ephemeral_arrays["room_ephemeral_int64_vector"],
    parameter_arrays["room_configurations_int64_vector"],
    start_ms,
    end_ms,
    parameter_arrays["room_configurations_int64_vector"][IDX_ROOM_CFG_DT_ENGINE_MS],
)

start_time = time.perf_counter()
run_simulator(
    **state_arrays,
    **forcing_data,
    **parameter_arrays,
    **ephemeral_arrays,
    sim_end_epoch_ms=end_ms,
    rng=rng,
)
end_time = time.perf_counter()
execution_time = end_time - start_time

print(f"Execution time: {execution_time:.6f} seconds. Simulated hours: {NUM_SIMUL_HOURS}")
if RUN_DUMMY:
    print(f"Dummy load was run to save cache. Exiting now...")
    sys.exit()

output_int64_matrix = ephemeral_arrays["output_int64_matrix"]
output_float64_matrix = ephemeral_arrays["output_float64_matrix"]

# Persist state for tomorrow's run.
save_run_state(state_arrays, rng)

# Test/inspection only (see run_outputs_io.py docstring) -- save this chunk's
# ephemeral output arrays so a separate script can plot across chunks without
# re-running the simulation. Chunk index = however many chunk files already
# exist, so consecutive `python run.py` calls append in order.
this_chunk_index = len(_find_chunks("run_outputs"))
saved_path = save_run_outputs(ephemeral_arrays, this_chunk_index)
print(f"Saved chunk outputs to {saved_path}")

# scadafier + nullifier: the "raw data" layer for the rest of the pipeline.
# telemetry/batch_sensors get nullified (genuine sensor transmission-loss
# failure mode); controller-event and business ledgers don't (see nullifier.py).
#
# Two different rate profiles, not one -- telemetry's channels are fixed-position
# instruments realistically hardwired to a local PLC/BMS (HART/Modbus-class
# reliability), batch_sensors' load cell + pulp probe are realistically wireless/
# battery-powered and mounted on pallets that physically move. Defaults below are
# informed by published MTBF/PDR figures for each category (see conversation --
# not exact for your specific hardware/site, override via facility_cfg once you
# have real numbers or vendor spec sheets):
nullifier_cfg = facility_cfg.get("nullifier_rates", {})
hardwired_rates = nullifier_cfg.get("hardwired", {
    "background_rate_per_hour": 0.001,       # <0.1% packet loss -- HART/Modbus-class
    "channel_outages_per_week": 0.03,        # ~once per 8 months per channel
    "channel_outage_mean_duration_min": 240, # hours, until a tech swaps the transmitter
})
wireless_rates = nullifier_cfg.get("wireless", {
    "background_rate_per_hour": 0.15,        # ~2-4% per-reading loss at typical polling cadence
    "channel_outages_per_week": 0.35,        # ~once every 2-3 weeks per sensor
    "channel_outage_mean_duration_min": 720, # battery swap / pallet repositioning turnaround
})
facility_rates = nullifier_cfg.get("facility_wide", {
    "facility_outages_per_week": 0.15,       # placeholder -- not literature-backed, tune to your site
    "facility_outage_mean_duration_min": 45,
})

scadafied = scadafy_all_outputs(trim_run_outputs(ephemeral_arrays))
scadafied["telemetry"] = nullify_telemetry(
    scadafied["telemetry"], rng, **hardwired_rates, **facility_rates,
)
scadafied["batch_sensors"] = nullify_telemetry(
    scadafied["batch_sensors"], rng, **wireless_rates, **facility_rates,
)

# dtype/datetime enforcement, same convention as the weather schema
# (table_schemas.json). Every raw table in SCHEMA_KEY_MAP below needs an entry
# there, otherwise it is written without location_id/event_timestamp.
SCHEMA_KEY_MAP = {
    "telemetry": "telemetry", "batch_sensors": "batch_sensors",
    "compressor_events": "compressor_events", "humidifier_events": "humidifier_events", "door_events": "door_events", 
    "crop_rotation_events": "crop_rotation_events", "refrigerant_rotation_events": "refrigerant_rotation_events",
    "downtime_sequence_events": "downtime_sequence_events", "lp_trip_events": "lp_trip_events",
    "defrost_events": "defrost_events",
    "inbound_orders": "orders", "outbound_orders": "orders",
    "inbound_pallet_history": "pallet_history", "outbound_pallet_history": "pallet_history",
}

raw_dir = Path("raw_parquets") / f"chunk_{this_chunk_index:04d}"
raw_dir.mkdir(parents=True, exist_ok=True)
for name, frame in scadafied.items():
    schema = table_schemas.get(SCHEMA_KEY_MAP[name])
    if schema is not None:
        frame["location_id"] = int(active_location_id) # add location_id col
        frame["event_timestamp"] = pd.to_datetime(frame["timestamp_ms"], unit="ms")
        frame = enforce_dtypes(frame, schema["dtypes"], schema["datetime_cols"])
    frame = fix_timestamps(frame)
    frame.to_parquet(raw_dir / f"{name}.parquet", engine="pyarrow")
print(f"Wrote {len(scadafied)} raw parquets to {raw_dir}")
