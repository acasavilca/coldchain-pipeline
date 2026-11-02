"""
state_persistence.py

Save/load the two things a resumed run needs beyond configs and weather:

1. The full room "state" dict as returned by initialize_first_run_state()
   (room/pallet/door/box arrays) -- these are mutated in place by
   run_simulator, so whatever run.py holds a reference to after the call
   is already the final state; this module just serializes it.
2. The RNG's bit-generator state, so draws continue rather than restart.

The three post-processed schedule vectors (downtime_timestamps_ms_vector,
valid_crop_rotation_timestamps_ms_vector, valid_refrigerant_rotation_timestamps_ms_vector)
are saved into the SAME npz file, under the key names join_downtimes() in
simulator_engine.py already reads on load (downtime_timestamps_ms_vec,
valid_crop_rotation_timestamps_ms_vec, valid_refrigerant_rotation_timestamps_ms_vec)
-- that load path already existed; this module supplies the save side.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from utils import clean_rng_state, NumpyEncoder

# Keys of the dict returned by simulator_engine.initialize_first_run_state().
# Kept explicit (rather than "whatever's in the dict") so a load with a stale
# or hand-edited npz fails loudly instead of silently handing run_simulator
# a state dict that's missing a field.
STATE_ARRAY_KEYS = (
    "room_state_float64_vector",
    "room_state_int64_vector",
    "pallet_metadata_int64_matrix",
    "pallet_timeline_int64_matrix",
    "pallet_physics_float64_matrix",
    "inbound_wave_queue_matrix",
    "outbound_wave_queue_matrix",
    "box_masses_ledger",
    "box_flags_ledger",
    "partial_dispatch_box_temperatures_ledger",
    "door_states_int64_matrix",
    "door_states_float64_matrix",
    "active_crop_params_float64_vector",
    "active_crop_params_int64_vector",
    "active_crop_historical_yield_matrix",
    "active_crop_market_scarcity_matrix",
    "active_crop_base_grades_probs_matrix",
    "active_crop_respiration_curves_matrix",
    "active_refrigerant_properties_vector",
    "active_refrigerant_surfaces_matrix",
    "airflow_pallets_compact_vector",
    "inverse_airflow_pallets_lookup_vector",
    "crossing_blobs_compact_vector",
    "inverse_crossing_blobs_lookup_vector",
    # The three process_schedules() outputs -- folded in here rather than
    # threaded through save_run_state/load_run_state as separate params,
    # since run.py already assigns them into state_arrays before calling
    # run_simulator(**state_arrays, ...). No run_simulator return statement
    # needed either way: numpy arrays passed into an njit function and
    # mutated in place stay visible to the caller through the same dict
    # entry, exactly like every other state array here.
    "downtime_timestamps_ms_vector",
    "valid_crop_rotation_timestamps_ms_vector",
    "valid_refrigerant_rotation_timestamps_ms_vector",
)


def save_run_state(
    state_arrays: Dict[str, np.ndarray],
    rng: np.random.Generator,
    npz_path: str = "state.npz",
    rng_path: str = "rng_state.json",
) -> None:
    missing = [k for k in STATE_ARRAY_KEYS if k not in state_arrays]
    if missing:
        raise KeyError(f"save_run_state: state_arrays is missing expected keys: {missing}")

    np.savez(
        npz_path,
        **{k: state_arrays[k] for k in STATE_ARRAY_KEYS},
    )

    with open(rng_path, "w") as f:
        json.dump(clean_rng_state({"rng_state": rng.bit_generator.state}), f, cls=NumpyEncoder)


def load_run_state(
    npz_path: str = "state.npz",
    rng_path: str = "rng_state.json",
) -> Optional[Tuple[Dict[str, np.ndarray], np.random.Generator]]:
    """
    Returns (state_arrays, rng) if both files exist, else None -- in which
    case run.py should cold-start via initialize_first_run_state() and a
    fresh np.random.default_rng(seed) instead.
    """
    npz_file = Path(npz_path)
    rng_file = Path(rng_path)
    if not npz_file.exists() or not rng_file.exists():
        return None

    with np.load(npz_file) as data:
        state_arrays = {k: data[k] for k in STATE_ARRAY_KEYS}

    with open(rng_file, "r") as f:
        rng_payload = json.load(f)
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_payload["rng_state"]

    return state_arrays, rng

