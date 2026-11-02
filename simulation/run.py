#!/usr/bin/env python3
"""
run.py -- command-line entry point for one chunk of simulated time.

Each call simulates ``--hours`` of plant operation, continuing from the state
saved by the previous call (state.npz + rng_state.json in --run-dir), or
cold-starting at ``--start-date`` when no state exists. It then writes:

    <run-dir>/state.npz, rng_state.json         resume point for the next call
    <run-dir>/run_outputs/run_outputs_chunk_NNNN.npz   raw simulator arrays
    <run-dir>/raw_parquets/chunk_NNNN/*.parquet        the raw tables loaded into BigQuery

Examples
    python run.py                               # 72 h, continue or cold-start
    python run.py --hours 168 --fruit avocado --refrigerant r134a   # cold start only
    python run.py --smoke-test                  # 4 simulated minutes, writes nothing
    python run.py --no-cache                    # ignore the numba compilation cache
    python run.py --clear-cache --smoke-test    # wipe the cache, recompile, exit

Numba caching: compiled functions are cached in ``.numba_cache/<hash>``, where
the hash covers every simulator source and config file, so editing any of them
recompiles automatically. --no-cache / --clear-cache exist as escape hatches.
"""
import argparse
import os
import sys
import time
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent

DEFAULT_HOURS = 72
DEFAULT_START_DATE = "2025-05-01T03:00:00"   # local time at the facility; cold start only
SMOKE_TEST_DEFAULT_MINUTES = 4

# raw table name -> key in schemas/table_schemas.json
SCHEMA_KEY_MAP = {
    "telemetry": "telemetry", "batch_sensors": "batch_sensors",
    "compressor_events": "compressor_events", "humidifier_events": "humidifier_events",
    "door_events": "door_events",
    "crop_rotation_events": "crop_rotation_events",
    "refrigerant_rotation_events": "refrigerant_rotation_events",
    "downtime_sequence_events": "downtime_sequence_events", "lp_trip_events": "lp_trip_events",
    "defrost_events": "defrost_events",
    "inbound_orders": "orders", "outbound_orders": "orders",
    "inbound_pallet_history": "pallet_history", "outbound_pallet_history": "pallet_history",
}

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulate one chunk of cold-storage operation and write the raw tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hours", type=float, default=DEFAULT_HOURS, help="simulated hours in this chunk")
    p.add_argument("--start-date", default=DEFAULT_START_DATE,
                   help="cold start only: facility-local start, ISO format (ignored when resuming)")
    p.add_argument("--fruit", choices=["blueberry", "avocado"], default="blueberry",
                   help="cold start only: initial crop")
    p.add_argument("--refrigerant", choices=["r404a", "r134a"], default="r404a",
                   help="cold start only: initial refrigerant")
    p.add_argument("--run-dir", type=Path, default=SIM_DIR,
                   help="where state, run_outputs/ and raw_parquets/ live")
    p.add_argument("--refresh-weather", action="store_true",
                   help="download fresh satellite weather instead of reading weather_data.parquet")
    p.add_argument("--no-run-outputs", action="store_true",
                   help="skip the (multi-GB) run_outputs_chunk_NNNN.npz diagnostic dump")
    p.add_argument("--smoke-test", nargs="?", type=float, const=SMOKE_TEST_DEFAULT_MINUTES, metavar="MINUTES",
                   help="simulate only MINUTES (default %(const)s) to warm the numba cache; writes nothing")
    p.add_argument("--no-cache", action="store_true", help="compile without reading or writing the numba cache")
    p.add_argument("--clear-cache", action="store_true", help="delete the numba cache before running")
    p.add_argument("--fastmath", choices=["safe", "full", "off"], default="safe",
                   help="numba fast-math level (full may optimise away NaN/Inf checks)")
    return p.parse_args(argv)


def configure_numba(args: argparse.Namespace) -> None:
    """Must run before any simulator module (and therefore numba) is imported."""
    os.environ["COLDCHAIN_NUMBA_CACHE"] = "0" if args.no_cache else "1"
    os.environ["COLDCHAIN_NUMBA_FASTMATH"] = args.fastmath
    sys.path.insert(0, str(SIM_DIR))
    if args.clear_cache:
        import jit
        print(f"Cleared numba cache ({jit.clear_cache()} entries).")


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    from zoneinfo import ZoneInfo

    import constants as C
    import matrix_index_maps as M
    from time_calendar import get_iana_timezone, _to_epoch_ms, _to_datetime_str
    from array_builders import load_json_files, build_all_parameter_arrays, build_ephemeral_arrays
    from run_setup import (
        retrieve_satellite_data, prepare_forcing_arrays, join_downtimes,
        initialize_first_run_state, process_schedules, enforce_dtypes, fix_timestamps,
    )
    from engine import run_simulator
    from state_persistence import save_run_state, load_run_state
    from run_outputs_io import save_run_outputs, trim_run_outputs, _find_chunks
    from scadafier import scadafy_all_outputs
    from nullifier import nullify_telemetry

    smoke = args.smoke_test is not None
    run_dir = args.run_dir.resolve()
    state_npz, rng_json = run_dir / "state.npz", run_dir / "rng_state.json"
    outputs_dir, parquet_root = run_dir / "run_outputs", run_dir / "raw_parquets"
    weather_file, raw_weather_file = SIM_DIR / "weather_data.parquet", SIM_DIR / "raw_weather_data.parquet"
    os.chdir(SIM_DIR)   # the config loaders use paths relative to the simulation folder

    configs = load_json_files("./configs")
    facility_cfg = configs["facility_runtime_config"]
    equipment_catalog = configs["catalog_equipment_config"]
    respiration_db = configs["fruit_respiration_curves"]
    table_schemas = load_json_files("./schemas")["table_schemas"]

    active_location_id = facility_cfg["facility_location"]["active_location_id"]
    location = facility_cfg["facility_location"]["locations"][str(active_location_id)]
    runtime_settings = facility_cfg["simulation_runtime_settings"]

    # Resume from saved state if present, else cold start (rng is built first either way).
    resumed = load_run_state(str(state_npz), str(rng_json))
    if resumed is not None:
        state_arrays, rng = resumed
        saved_float = state_arrays["room_state_float64_vector"].size
        saved_int = state_arrays["room_state_int64_vector"].size
        if saved_float != M.N_ROOM_STATE_FLOAT64_FIELDS or saved_int != M.N_ROOM_STATE_INT64_FIELDS:
            raise ValueError(
                f"Saved room-state vectors have obsolete lengths ({saved_float} float, {saved_int} int); "
                f"current layout needs ({M.N_ROOM_STATE_FLOAT64_FIELDS} float, {M.N_ROOM_STATE_INT64_FIELDS} int). "
                "Delete state.npz / rng_state.json to cold-start, or migrate the checkpoint."
            )
        print(f"Resuming from {state_npz.name} / {rng_json.name} in {run_dir}")
    else:
        rng = np.random.default_rng(runtime_settings["rng_seed"])
        state_arrays = None

    tz_obj = ZoneInfo(get_iana_timezone(location["latitude"], location["longitude"]))
    if resumed is not None:
        start_ms = int(state_arrays["room_state_int64_vector"][M.IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS])
    else:
        start_ms = _to_epoch_ms(str(args.start_date), tz_obj)

    if smoke:
        end_ms = start_ms + int(args.smoke_test * 60e3)
    else:
        end_ms = int(start_ms + C.MS_PER_HOUR * args.hours)
    sim_hours = (end_ms - start_ms) / C.MS_PER_HOUR
    print(f"start: {_to_datetime_str(start_ms)}\nend:   {_to_datetime_str(end_ms)}")

    if args.refresh_weather and not smoke:
        weather_data_df, raw_weather_data_df = retrieve_satellite_data(
            rng=rng, start_epoch_ms=start_ms, end_epoch_ms=end_ms,
            measurements=runtime_settings["required_weather_measurements"],
            active_location_id=active_location_id, facility_cfg=facility_cfg, table_schema=table_schemas,
        )
        weather_data_df.to_parquet(weather_file, engine="pyarrow")
        raw_weather_data_df.to_parquet(raw_weather_file, engine="pyarrow")
    else:
        weather_data_df = pd.read_parquet(weather_file, engine="pyarrow")

    forcing_data, _ = prepare_forcing_arrays(
        sim_start_ms=start_ms, sim_end_ms=end_ms, facility_cfg=facility_cfg, weather_data=weather_data_df,
    )
    parameter_arrays = build_all_parameter_arrays(
        facility_cfg=facility_cfg, equipment_catalog=equipment_catalog, respiration_db=respiration_db,
    )
    ephemeral_arrays = build_ephemeral_arrays(
        start_ms, end_ms, parameter_arrays["base_quality_grade_probs_matrix"], facility_cfg,
    )

    if state_arrays is None:
        fruit = {"blueberry": M.FRUIT_TYPE_BLUEBERRY, "avocado": M.FRUIT_TYPE_AVOCADO}[args.fruit]
        refrigerant = {"r404a": M.REFRIGERANT_R404A, "r134a": M.REFRIGERANT_R134A}[args.refrigerant]
        state_arrays = initialize_first_run_state(
            rng, start_ms, forcing_data["forcing_matrix"],
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
            starting_crop_row=fruit,
            starting_refrigerant_row=refrigerant,
            magnus_const_set=C.ALDUCHOV_LIQ_CONST_SET,
        )

    # Folds leftover schedule rows saved in the state with the config-declared schedule.
    input_schedule_arrays = join_downtimes(facility_cfg, state_file_path=str(state_npz))
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
        start_ms, end_ms,
        parameter_arrays["room_configurations_int64_vector"][M.IDX_ROOM_CFG_DT_ENGINE_MS],
    )

    t0 = time.perf_counter()
    run_simulator(
        **state_arrays, **forcing_data, **parameter_arrays, **ephemeral_arrays,
        sim_end_epoch_ms=end_ms, rng=rng,
    )
    print(f"Execution time: {time.perf_counter() - t0:.2f} s for {sim_hours:.4g} simulated hours "
          f"(includes numba compilation on a cold cache).")

    if smoke:
        print("Smoke test finished: numba cache is warm; no state or outputs were written.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_state(state_arrays, rng, str(state_npz), str(rng_json))

    chunk_index = len(_find_chunks(str(outputs_dir)))
    if not args.no_run_outputs:
        print(f"Saved chunk outputs to {save_run_outputs(ephemeral_arrays, chunk_index, str(outputs_dir))}")

    # scadafier + nullifier: the "raw data" layer. Telemetry and batch sensors are
    # nullified (sensor transmission loss); event and business ledgers are not.
    # Telemetry = hardwired instruments; batch sensors = wireless, on moving pallets.
    # Rates live in facility_runtime_config.json ("nullifier_rates").
    rates = facility_cfg["nullifier_rates"]
    facility_rates = rates["facility_wide"]
    scadafied = scadafy_all_outputs(trim_run_outputs(ephemeral_arrays))
    scadafied["telemetry"] = nullify_telemetry(scadafied["telemetry"], rng, **rates["hardwired"], **facility_rates)
    scadafied["batch_sensors"] = nullify_telemetry(scadafied["batch_sensors"], rng, **rates["wireless"], **facility_rates)

    raw_dir = parquet_root / f"chunk_{chunk_index:04d}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in scadafied.items():
        schema = table_schemas.get(SCHEMA_KEY_MAP[name])
        if schema is not None:
            frame["location_id"] = int(active_location_id)
            frame["event_timestamp"] = pd.to_datetime(frame["timestamp_ms"], unit="ms")
            frame = enforce_dtypes(frame, schema["dtypes"], schema["datetime_cols"])
        frame = fix_timestamps(frame)
        frame.to_parquet(raw_dir / f"{name}.parquet", engine="pyarrow")
    print(f"Wrote {len(scadafied)} raw parquets to {raw_dir}")


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_numba(args)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
