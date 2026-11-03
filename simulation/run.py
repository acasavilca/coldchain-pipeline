#!/usr/bin/env python3
"""
run.py -- command-line entry point for one chunk of simulated time.

Each call simulates ``--hours`` (or ``--days``) of plant operation for ONE
location, continuing from the state saved by the previous call, or
cold-starting at ``--start-date`` when the run directory has no state. The run
directory defaults to ``runs/location_<id>/`` and is bound to its location and
seed by ``run_meta.json``, written at the cold start. Each call writes:

    <run-dir>/state.npz, rng_state.json, run_meta.json   resume point and identity
    <run-dir>/run_outputs/run_outputs_chunk_NNNN.npz     raw simulator arrays
    <run-dir>/raw_parquets/chunk_NNNN/*.parquet          the raw tables loaded into BigQuery
    <run-dir>/raw_parquets/chunk_NNNN/run_manifest.json  row counts, checksums, code fingerprint

Weather is cached per location in ``weather/location_<id>/``.

Examples
    python run.py                               # 72 h, location from config default, continue or cold-start
    python run.py --location 2 --days 1 --no-run-outputs     # one UTC day for location 2
    python run.py --location 2 --seed 7 --hours 168 --fruit avocado --refrigerant r134a   # cold start only
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
MS_PER_DAY = 86_400_000

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
    p.add_argument("--location", type=int, default=None,
                   help="location id from facility_runtime_config.json. Cold start: defaults to "
                        "facility_location.default_location_id (a message is printed). Resume: must match the run directory")
    p.add_argument("--seed", type=int, default=None,
                   help="cold start only: RNG seed (default: the location's rng_seed in the config)")
    length = p.add_mutually_exclusive_group()
    length.add_argument("--hours", type=float, default=None,
                        help=f"simulated hours in this chunk (default {DEFAULT_HOURS} if neither --hours nor --days is given)")
    length.add_argument("--days", type=int, default=None,
                        help="simulated whole UTC days in this chunk; the chunk must start at 00:00 UTC (daily mode)")
    p.add_argument("--align-first", action="store_true",
                   help="with --days on a cold start not at 00:00 UTC: run only up to the next 00:00 UTC")
    p.add_argument("--weather-only", action="store_true",
                   help="download the weather for the period (needs --refresh-weather's network access), write the "
                        "location's weather cache and exit without simulating")
    p.add_argument("--start-date", default=DEFAULT_START_DATE,
                   help="cold start only: facility-local start, ISO format (ignored when resuming)")
    p.add_argument("--fruit", choices=["blueberry", "avocado"], default="blueberry",
                   help="cold start only: initial crop")
    p.add_argument("--refrigerant", choices=["r404a", "r134a"], default="r404a",
                   help="cold start only: initial refrigerant")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="where state, run_outputs/ and raw_parquets/ live (default: runs/location_<id>/ next to this script)")
    p.add_argument("--refresh-weather", action="store_true",
                   help="download weather for this chunk and overwrite the location's cache (weather/location_<id>/) "
                        "instead of reading it; use --weather-only to fetch a long period at once")
    p.add_argument("--no-run-outputs", action="store_true",
                   help="skip the (multi-GB) run_outputs_chunk_NNNN.npz diagnostic dump")
    p.add_argument("--smoke-test", nargs="?", type=float, const=SMOKE_TEST_DEFAULT_MINUTES, metavar="MINUTES",
                   help="simulate only MINUTES (default %(const)s) to warm the numba cache; writes nothing")
    p.add_argument("--no-cache", action="store_true", help="compile without reading or writing the numba cache")
    p.add_argument("--clear-cache", action="store_true", help="delete the numba cache before running")
    p.add_argument("--fastmath", choices=["safe", "full", "off"], default="safe",
                   help="numba fast-math level (full may optimise away NaN/Inf checks)")
    args = p.parse_args(argv)
    if args.weather_only:
        args.refresh_weather = True
    return args


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
    from run_outputs_io import save_run_outputs, trim_run_outputs
    from scadafier import scadafy_all_outputs
    from nullifier import nullify_telemetry
    import jit
    import hashlib, json
    from datetime import datetime, timezone

    smoke = args.smoke_test is not None
    os.chdir(SIM_DIR)   # the config loaders use paths relative to the simulation folder

    configs = load_json_files("./configs")
    facility_cfg = configs["facility_runtime_config"]
    equipment_catalog = configs["catalog_equipment_config"]
    respiration_db = configs["fruit_respiration_curves"]
    table_schemas = load_json_files("./schemas")["table_schemas"]
    locations = facility_cfg["facility_location"]["locations"]
    runtime_settings = facility_cfg["simulation_runtime_settings"]

    # ---- which location and run directory --------------------------------------------------
    run_dir_arg = args.run_dir.resolve() if args.run_dir is not None else None
    location_id = args.location
    meta = None
    if run_dir_arg is not None and (run_dir_arg / "run_meta.json").exists():
        meta = json.loads((run_dir_arg / "run_meta.json").read_text())
        if location_id is not None and location_id != meta["location_id"]:
            raise SystemExit(f"error: {run_dir_arg} belongs to location {meta['location_id']}, "
                             f"but --location {location_id} was given.")
        location_id = meta["location_id"]
    if location_id is None:
        location_id = int(facility_cfg["facility_location"]["default_location_id"])
        print(f"No --location given: using the default location {location_id} "
              f"({locations[str(location_id)]['name']}). Pass --location to choose another.")
    if str(location_id) not in locations:
        raise SystemExit(f"error: location {location_id} is not in the config (known: {', '.join(locations)}).")
    location = locations[str(location_id)]
    run_dir = run_dir_arg if run_dir_arg is not None else SIM_DIR / "runs" / f"location_{location_id}"
    if meta is None and (run_dir / "run_meta.json").exists():
        meta = json.loads((run_dir / "run_meta.json").read_text())
        if meta["location_id"] != location_id:
            raise SystemExit(f"error: {run_dir} belongs to location {meta['location_id']}, not {location_id}.")

    state_npz, rng_json = run_dir / "state.npz", run_dir / "rng_state.json"
    outputs_dir, parquet_root = run_dir / "run_outputs", run_dir / "raw_parquets"
    weather_dir = SIM_DIR / "weather" / f"location_{location_id}"
    weather_file, raw_weather_file = weather_dir / "weather_data.parquet", weather_dir / "raw_weather_data.parquet"

    if state_npz.exists() and meta is None:
        raise SystemExit(f"error: {state_npz} exists without run_meta.json (a state from before multi-location "
                         "support). It cannot be tied to a location or seed: delete the run directory's state "
                         "files (state.npz, rng_state.json, run_outputs/, raw_parquets/) and cold-start again.")

    print(f"Location {location_id}: {location['name']}   run dir: {run_dir}")

    # Resume from saved state if present, else cold start (rng is built first either way).
    resumed = load_run_state(str(state_npz), str(rng_json))
    if resumed is not None:
        if args.seed is not None:
            raise SystemExit("error: --seed is only valid on a cold start; this run directory already has state "
                             f"(seed {meta['rng_seed']}).")
        state_arrays, rng = resumed
        seed = int(meta["rng_seed"])
        saved_float = state_arrays["room_state_float64_vector"].size
        saved_int = state_arrays["room_state_int64_vector"].size
        if saved_float != M.N_ROOM_STATE_FLOAT64_FIELDS or saved_int != M.N_ROOM_STATE_INT64_FIELDS:
            raise ValueError(
                f"Saved room-state vectors have obsolete lengths ({saved_float} float, {saved_int} int); "
                f"current layout needs ({M.N_ROOM_STATE_FLOAT64_FIELDS} float, {M.N_ROOM_STATE_INT64_FIELDS} int). "
                "Delete state.npz / rng_state.json to cold-start, or migrate the checkpoint."
            )
        print(f"Resuming from {state_npz.name} / {rng_json.name} (seed {seed})")
    else:
        seed = int(args.seed) if args.seed is not None else int(location["rng_seed"])
        rng = np.random.default_rng(seed)
        state_arrays = None
        print(f"Cold start with seed {seed}")
        for other in sorted(run_dir.parent.glob("*/run_meta.json")):
            if other.parent.resolve() == run_dir:
                continue
            try:
                if json.loads(other.read_text()).get("rng_seed") == seed:
                    print(f"WARNING: {other.parent.name} uses the same seed ({seed}); two runs with the same seed "
                          "draw the same random sequence. Use --seed to make this run independent.")
            except (OSError, ValueError):
                pass

    tz_obj = ZoneInfo(get_iana_timezone(location["latitude"], location["longitude"]))
    if resumed is not None:
        start_ms = int(state_arrays["room_state_int64_vector"][M.IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS])
    else:
        start_ms = _to_epoch_ms(str(args.start_date), tz_obj)

    if smoke:
        end_ms = start_ms + int(args.smoke_test * 60e3)
    elif args.days is not None:
        # daily mode: every chunk is whole UTC days, so a chunk maps onto BigQuery's day partitions
        if start_ms % MS_PER_DAY != 0:
            next_midnight = (start_ms // MS_PER_DAY + 1) * MS_PER_DAY
            if resumed is None and args.align_first:
                end_ms = next_midnight
                print(f"--align-first: this chunk runs {(next_midnight - start_ms) / C.MS_PER_HOUR:.4g} h, "
                      "up to the next 00:00 UTC.")
            elif resumed is None:
                raise SystemExit(
                    f"error: --days needs a start at 00:00 UTC, but {_to_datetime_str(start_ms)} (UTC) "
                    f"is not. Choose a --start-date that is 00:00 UTC, or add --align-first to run only up to "
                    "the next 00:00 UTC.")
            else:
                raise SystemExit(
                    f"error: the saved state is at {_to_datetime_str(start_ms)} (UTC), not at 00:00 UTC. "
                    f"Run `--hours {(next_midnight - start_ms) / C.MS_PER_HOUR:.6g}` to reach the next midnight, "
                    "then continue with --days.")
        else:
            end_ms = start_ms + args.days * MS_PER_DAY
    else:
        end_ms = int(start_ms + C.MS_PER_HOUR * (args.hours if args.hours is not None else DEFAULT_HOURS))
    sim_hours = (end_ms - start_ms) / C.MS_PER_HOUR
    print(f"start: {_to_datetime_str(start_ms)}\nend:   {_to_datetime_str(end_ms)}")

    if args.refresh_weather and not smoke:
        weather_data_df, raw_weather_data_df = retrieve_satellite_data(
            rng=rng, start_epoch_ms=start_ms, end_epoch_ms=end_ms,
            measurements=runtime_settings["required_weather_measurements"],
            active_location_id=location_id, facility_cfg=facility_cfg, table_schema=table_schemas,
        )
        weather_dir.mkdir(parents=True, exist_ok=True)
        weather_data_df.to_parquet(weather_file, engine="pyarrow")
        raw_weather_data_df.to_parquet(raw_weather_file, engine="pyarrow")
        print(f"Wrote weather cache for location {location_id} to {weather_dir}")
        if args.weather_only:
            return
    else:
        if not weather_file.exists():
            raise SystemExit(f"error: no weather cache for location {location_id} ({weather_file}). Download it with "
                             f"`python run.py --location {location_id} --weather-only --days N` (needs network access).")
        weather_data_df = pd.read_parquet(weather_file, engine="pyarrow")

    cached_ids = set(weather_data_df["location_id"].unique())
    if cached_ids != {location_id}:
        raise SystemExit(f"error: {weather_file} holds weather for location(s) {sorted(int(i) for i in cached_ids)}, not {location_id}.")
    w_first = int(pd.Timestamp(weather_data_df["timestamp"].min()).value // 10**6)
    w_last = int(pd.Timestamp(weather_data_df["timestamp"].max()).value // 10**6)
    if w_first > start_ms or w_last < end_ms:
        raise SystemExit(
            f"error: the weather cache for location {location_id} covers {weather_data_df['timestamp'].min()} to "
            f"{weather_data_df['timestamp'].max()} (UTC), which does not include this chunk "
            f"({_to_datetime_str(start_ms)} to {_to_datetime_str(end_ms)} UTC). Re-download it with "
            "--weather-only over a longer period.")

    forcing_data, _ = prepare_forcing_arrays(
        sim_start_ms=start_ms, sim_end_ms=end_ms, facility_cfg=facility_cfg, weather_data=weather_data_df,
    )
    parameter_arrays = build_all_parameter_arrays(
        facility_cfg=facility_cfg, equipment_catalog=equipment_catalog, respiration_db=respiration_db,
        location_id=location_id,
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
    input_schedule_arrays = join_downtimes(facility_cfg, location_id=location_id, state_file_path=str(state_npz))
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
    if resumed is None:
        (run_dir / "run_meta.json").write_text(json.dumps({
            "location_id": location_id, "rng_seed": seed, "start_utc": _to_datetime_str(start_ms),
            "start_epoch_ms": int(start_ms), "fruit": args.fruit, "refrigerant": args.refrigerant,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2) + "\n")

    # next chunk number = one past the highest chunk_NNNN already written (run_outputs/ may be skipped)
    existing = [int(d.name.split("_")[1]) for d in parquet_root.glob("chunk_[0-9][0-9][0-9][0-9]")] if parquet_root.exists() else []
    chunk_index = max(existing) + 1 if existing else 0
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
    row_counts = {}
    for name, frame in scadafied.items():
        schema = table_schemas.get(SCHEMA_KEY_MAP[name])
        if schema is not None:
            frame["location_id"] = int(location_id)
            frame["event_timestamp"] = pd.to_datetime(frame["timestamp_ms"], unit="ms")
            frame = enforce_dtypes(frame, schema["dtypes"], schema["datetime_cols"])
        frame = fix_timestamps(frame)
        frame.to_parquet(raw_dir / f"{name}.parquet", engine="pyarrow")
        row_counts[name] = int(len(frame))
    # Manifest: lets a downstream gate check completeness and provenance before loading.
    checksums = {}
    for name in scadafied:
        h = hashlib.sha256()
        with open(raw_dir / f"{name}.parquet", "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        checksums[name] = h.hexdigest()
    (raw_dir / "run_manifest.json").write_text(json.dumps({
        "location_id": location_id, "rng_seed": seed, "chunk_index": chunk_index,
        "start_epoch_ms": int(start_ms), "end_epoch_ms": int(end_ms),
        "start_utc": _to_datetime_str(start_ms), "end_utc": _to_datetime_str(end_ms),
        "simulated_hours": sim_hours, "row_counts": row_counts, "sha256": checksums,
        "code_fingerprint": jit.source_fingerprint(),
        "facility_config_sha256": hashlib.sha256((SIM_DIR / "configs" / "facility_runtime_config.json").read_bytes()).hexdigest(),
        "fastmath": args.fastmath, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    print(f"Wrote {len(scadafied)} raw parquets and run_manifest.json to {raw_dir}")


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_numba(args)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
