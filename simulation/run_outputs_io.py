"""
run_outputs_io.py

Test/inspection tooling, not production. Saves the full ephemeral_arrays dict
(all 18 keys -- output matrices + all 8 ledgers + row-counters) to one npz per
run_simulator() call, so a multi-week local test can go chunk-by-chunk without
holding every week's output in RAM at once. load_run_outputs() then finds every
chunk file in a directory, trims each ledger to its real row-counter, and
concatenates across chunks in call order.

In production this isn't needed -- output goes straight to parquet + upload
right after each run_simulator() call, nothing sits on disk waiting to be
reloaded. This exists so you can run several week-chunks locally, then load
everything into a separate plotting script without re-running the simulation.
"""

import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from matrix_index_maps import (
    IDX_ROOM_EPHEMERAL_INBOUND_ORDER_ROW_IDX,
    IDX_ROOM_EPHEMERAL_OUTBOUND_ORDER_ROW_IDX,
    IDX_ROOM_EPHEMERAL_INBOUND_HISTORY_ROW_IDX,
    IDX_ROOM_EPHEMERAL_OUTBOUND_HISTORY_ROW_IDX,
    IDX_ROOM_EPHEMERAL_BATCH_SENSOR_WRITE_IDX,
    IDX_ROOM_EPHEMERAL_COMPRESSOR_EVENT_WRITE_IDX,
    IDX_ROOM_EPHEMERAL_HUMIDIFIER_EVENT_WRITE_IDX,
    IDX_ROOM_EPHEMERAL_DOOR_EVENT_WRITE_IDX,
    IDX_ROOM_EPHEMERAL_CROP_ROTATION_EVENT_WRITE_IDX,
    IDX_ROOM_EPHEMERAL_REFRIGERANT_ROTATION_EVENT_WRITE_IDX,
    IDX_ROOM_EPHEMERAL_DOWNTIME_SEQUENCE_EVENT_WRITE_IDX,
    IDX_ROOM_EPHEMERAL_LP_TRIP_EVENT_WRITE_IDX,
    IDX_ROOM_EPHEMERAL_DEFROST_EVENT_WRITE_IDX,
    IDX_ROOM_EPHEMERAL_TELEMETRY_EXPORT_WRITE_IDX,
    N_DEFROST_EVENT_FIELDS,
)

# ledger key -> ephemeral counter index. output_int64_matrix and output_float64_matrix are
# NOT here -- they're sized exactly to total_steps, every row is a real step, nothing to trim.
# room_telemetry_matrix is trimmed separately (see _trim_telemetry): one row per sensor
# polling tick, counted by IDX_ROOM_EPHEMERAL_TELEMETRY_EXPORT_WRITE_IDX.
LEDGER_COUNTERS = {
    "inbound_orders_ledger":             IDX_ROOM_EPHEMERAL_INBOUND_ORDER_ROW_IDX,
    "outbound_orders_ledger":            IDX_ROOM_EPHEMERAL_OUTBOUND_ORDER_ROW_IDX,
    "inbound_pallet_history_ledger":     IDX_ROOM_EPHEMERAL_INBOUND_HISTORY_ROW_IDX,
    "outbound_pallet_history_ledger":    IDX_ROOM_EPHEMERAL_OUTBOUND_HISTORY_ROW_IDX,
    "batch_fruit_sensor_ledger":         IDX_ROOM_EPHEMERAL_BATCH_SENSOR_WRITE_IDX,
    "compressor_event_ledger":           IDX_ROOM_EPHEMERAL_COMPRESSOR_EVENT_WRITE_IDX,
    "humidifier_event_ledger":           IDX_ROOM_EPHEMERAL_HUMIDIFIER_EVENT_WRITE_IDX,
    "door_event_ledger":                 IDX_ROOM_EPHEMERAL_DOOR_EVENT_WRITE_IDX,
    "crop_rotation_event_ledger":        IDX_ROOM_EPHEMERAL_CROP_ROTATION_EVENT_WRITE_IDX,
    "refrigerant_rotation_event_ledger": IDX_ROOM_EPHEMERAL_REFRIGERANT_ROTATION_EVENT_WRITE_IDX,
    "downtime_sequence_event_ledger":    IDX_ROOM_EPHEMERAL_DOWNTIME_SEQUENCE_EVENT_WRITE_IDX,
    "lp_trip_event_ledger":              IDX_ROOM_EPHEMERAL_LP_TRIP_EVENT_WRITE_IDX,
    "defrost_event_ledger":              IDX_ROOM_EPHEMERAL_DEFROST_EVENT_WRITE_IDX,
}

UNTRIMMED_KEYS = ("output_int64_matrix", "output_float64_matrix")


def _trim_telemetry(telemetry_matrix: np.ndarray, room_ephemeral_int64_vector: np.ndarray) -> np.ndarray:
    """Keep only the rows actually written (one per sensor polling tick)."""
    if len(room_ephemeral_int64_vector) <= IDX_ROOM_EPHEMERAL_TELEMETRY_EXPORT_WRITE_IDX:
        # Chunk saved before telemetry rows were compacted: one row per step, nothing to trim.
        return telemetry_matrix
    return telemetry_matrix[:int(room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_TELEMETRY_EXPORT_WRITE_IDX])]

FILENAME_RE = re.compile(r"^run_outputs_chunk_(\d+)\.npz$")


def trim_run_outputs(ephemeral_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Trims a single (not-yet-saved) ephemeral_arrays dict to real row counts,
    same logic as load_run_outputs() but for one in-memory run rather than
    reloading saved chunk files. Use this right after run_simulator() returns,
    before handing the data to scadafier.scadafy_all_outputs().
    """
    room_ephemeral_int64_vector = ephemeral_arrays["room_ephemeral_int64_vector"]
    trimmed = {}
    for key, counter_idx in LEDGER_COUNTERS.items():
        n_rows = int(room_ephemeral_int64_vector[counter_idx])
        trimmed[key] = ephemeral_arrays[key][:n_rows]
    trimmed["room_telemetry_matrix"] = _trim_telemetry(ephemeral_arrays["room_telemetry_matrix"], room_ephemeral_int64_vector)
    for key in UNTRIMMED_KEYS:
        trimmed[key] = ephemeral_arrays[key]
    return trimmed


def save_run_outputs(ephemeral_arrays: Dict[str, np.ndarray], chunk_index: int, out_dir: str = "run_outputs") -> Path:
    """Call once per run_simulator() call, in order, with an incrementing chunk_index (0, 1, 2, ...)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"run_outputs_chunk_{chunk_index:04d}.npz"
    np.savez(file_path, **ephemeral_arrays)
    return file_path


def _find_chunks(out_dir: str) -> List[Path]:
    out_path = Path(out_dir)
    chunks = []
    for f in out_path.glob("run_outputs_chunk_*.npz"):
        m = FILENAME_RE.match(f.name)
        if m:
            chunks.append((int(m.group(1)), f))
    chunks.sort(key=lambda t: t[0])
    return [f for _, f in chunks]


def load_run_outputs(out_dir: str = "run_outputs") -> Dict[str, np.ndarray]:
    """
    Loads every run_outputs_chunk_*.npz in out_dir, in chunk order, trims each
    ledger to its real row-counter from that chunk, and concatenates across
    chunks. room_telemetry_matrix is trimmed to its polling-tick row counter;
    output_int64_matrix / output_float64_matrix are concatenated whole (no trim
    needed -- see UNTRIMMED_KEYS above).
    """
    chunk_paths = _find_chunks(out_dir)
    if not chunk_paths:
        raise FileNotFoundError(f"No run_outputs_chunk_*.npz files found in {out_dir}/")

    collected: Dict[str, List[np.ndarray]] = {}

    for path in chunk_paths:
        with np.load(path) as data:
            room_ephemeral_int64_vector = data["room_ephemeral_int64_vector"]

            for key, counter_idx in LEDGER_COUNTERS.items():
                if key == "defrost_event_ledger" and key not in data.files:
                    # Saved chunks from before this ledger existed remain inspectable.
                    collected.setdefault(key, []).append(np.empty((0, N_DEFROST_EVENT_FIELDS), dtype=np.int64))
                    continue
                n_rows = int(room_ephemeral_int64_vector[counter_idx])
                collected.setdefault(key, []).append(data[key][:n_rows])

            collected.setdefault("room_telemetry_matrix", []).append(
                _trim_telemetry(data["room_telemetry_matrix"], room_ephemeral_int64_vector)
            )
            for key in UNTRIMMED_KEYS:
                collected.setdefault(key, []).append(data[key])

    return {key: np.concatenate(arrays, axis=0) for key, arrays in collected.items()}
