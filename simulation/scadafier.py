"""
scadafier.py

Converts the internal numpy arrays (already trimmed by run_outputs_io.load_run_outputs
or equivalent) into what a real facility's historian/WMS would actually hand off:
long/tag-value format for sensor telemetry (the real-world historian convention --
PI, Ignition, etc. store (timestamp, tag_name, value) rows, not one wide row per
timestamp, precisely because different instruments poll at different, uncorrelated
rates), and named/typed tidy DataFrames for the business ledgers.

T_ambient / RH_ambient are deliberately excluded from the telemetry output: they're
written every step with no polling gate and no noise/filtering applied in the engine,
i.e. they were never modeled as a real on-site instrument -- they're a weather-cache
passthrough. The real weather parquet already carries that data to the datalake.

Two tag-value tables, not one per instrument, not one per instrument type: this
matches the dbt convention of one staging model per source table.
"""

from typing import Dict, Any

import numpy as np
import pandas as pd

from constants import SENTINEL_FLOAT64, MAX_EPOCH_TIMESTAMP_MS
from matrix_index_maps import (
    IDX_EXPORT_TIMESTAMP_MS, IDX_EXPORT_T_ROOM, IDX_EXPORT_RH_ROOM,
    IDX_EXPORT_T_SUCTION_LINE, IDX_EXPORT_P_SUCTION_LINE, IDX_EXPORT_T_LIQUID_LINE,
    IDX_EXPORT_P_LIQUID_LINE, IDX_EXPORT_COMPRESSOR_POWER, IDX_EXPORT_T_ROOM_AIR_RETURN,
    IDX_EXPORT_T_ROOM_AIR_SUPPLY, IDX_EXPORT_RH_ROOM_AIR_SUPPLY, IDX_EXPORT_CO2_PPM,
    IDX_EXPORT_O2_PCT,
    IDX_EXPORT_T_COIL_EVAP,
    IDX_EXPORT_EVAP_AIR_VELOCITY,
    IDX_BATCH_SENSOR_TIMESTAMP_MS, IDX_BATCH_SENSOR_BATCH_ID, IDX_BATCH_SENSOR_PALLET_SLOT,
    IDX_BATCH_SENSOR_T_PULP, IDX_BATCH_SENSOR_MASS,
    IDX_COMPRESSOR_EVENT_TIMESTAMP_MS, IDX_COMPRESSOR_EVENT_IS_ACTIVE,
    IDX_HUMIDIFIER_EVENT_TIMESTAMP_MS, IDX_HUMIDIFIER_EVENT_NUM_DISCS_ON,
    IDX_DOOR_EVENT_TIMESTAMP_MS, IDX_DOOR_EVENT_DOOR_CODE, IDX_DOOR_EVENT_IS_OPEN,
    IDX_CROP_ROTATION_EVENT_TIMESTAMP_MS, IDX_CROP_ROTATION_EVENT_FRUIT_TYPE_CODE,
    IDX_REFRIGERANT_ROTATION_EVENT_TIMESTAMP_MS, IDX_REFRIGERANT_ROTATION_EVENT_REFRIGERANT_CODE,
    IDX_DOWNTIME_SEQUENCE_EVENT_TIMESTAMP_MS,
    IDX_DOWNTIME_SEQUENCE_EVENT_SCHEDULED_START_TIMESTAMP_MS,
    IDX_DOWNTIME_SEQUENCE_EVENT_SHUTDOWN_TIMESTAMP_MS,
    IDX_DOWNTIME_SEQUENCE_EVENT_RAMP_UP_START_TIMESTAMP_MS,
    IDX_DOWNTIME_SEQUENCE_EVENT_RAMP_UP_END_TIMESTAMP_MS,
    IDX_LP_TRIP_EVENT_TIMESTAMP_MS, IDX_LP_TRIP_EVENT_TRIPPED,
    IDX_DEFROST_EVENT_START_MS, IDX_DEFROST_EVENT_END_MS,
    IDX_DEFROST_EVENT_TERMINATION_REASON, IDX_DEFROST_EVENT_FROST_MELTED_G,
    IDX_DEFROST_EVENT_COIL_ICE_MELTED_G, IDX_DEFROST_EVENT_PAN_ICE_MELTED_G,
    IDX_DEFROST_EVENT_FACE_ICE_SHED_G,
    IDX_ORDER_LEDGER_ORDER_ID, IDX_ORDER_LEDGER_TIMESTAMP_MS, IDX_ORDER_LEDGER_TARGET_FRUIT_MASS,
    IDX_ORDER_LEDGER_TOTAL_BOX_COUNT, IDX_ORDER_LEDGER_MINIMUM_GRADE, IDX_ORDER_LEDGER_TOTAL_PALLET_COUNT,
    IDX_ORDER_LEDGER_CROP_TYPE, IDX_ORDER_LEDGER_ASSIGNED_DOOR_CODE, IDX_ORDER_LEDGER_ASSIGNED_DRIVER, IDX_ORDER_LEDGER_DISPATCH_TYPE,
    IDX_HISTORICAL_PALLET_BATCH_ID, IDX_HISTORICAL_PALLET_PARENT_ORDER_ID, IDX_HISTORICAL_PALLET_TIMESTAMP_MS,
    IDX_HISTORICAL_PALLET_QUALITY_GRADE, IDX_HISTORICAL_PALLET_PULP_TEMP, IDX_HISTORICAL_PALLET_GROSS_START,
    IDX_HISTORICAL_PALLET_BOX_START, IDX_HISTORICAL_PALLET_GROSS_MOVED, IDX_HISTORICAL_PALLET_BOX_MOVED,
    IDX_HISTORICAL_PALLET_SUPERVISOR_INTERVENTION_MS,
)

# tag_name -> (column index in room_telemetry_matrix, rounding decimals, unit)
# Rounding matches typical real-instrument resolution -- a real transmitter doesn't
# report float64 full precision.
TELEMETRY_TAGS = {
    "t_room_c":              (IDX_EXPORT_T_ROOM, 2, "C"),
    "rh_room_frac":          (IDX_EXPORT_RH_ROOM, 3, "frac"),
    "t_suction_line_c":      (IDX_EXPORT_T_SUCTION_LINE, 2, "C"),
    "p_suction_line_bar":    (IDX_EXPORT_P_SUCTION_LINE, 3, "bar"),
    "t_liquid_line_c":       (IDX_EXPORT_T_LIQUID_LINE, 2, "C"),
    "p_liquid_line_bar":     (IDX_EXPORT_P_LIQUID_LINE, 3, "bar"),
    "compressor_power_w":    (IDX_EXPORT_COMPRESSOR_POWER, 0, "W"),
    "t_room_air_return_c":   (IDX_EXPORT_T_ROOM_AIR_RETURN, 2, "C"),
    "t_room_air_supply_c":   (IDX_EXPORT_T_ROOM_AIR_SUPPLY, 2, "C"),
    "rh_room_air_supply_frac": (IDX_EXPORT_RH_ROOM_AIR_SUPPLY, 3, "frac"),
    "co2_ppm":               (IDX_EXPORT_CO2_PPM, 0, "ppm"),
    "o2_pct":                (IDX_EXPORT_O2_PCT, 2, "pct"),
    "t_coil_evap_c":         (IDX_EXPORT_T_COIL_EVAP, 2, "C"),
    "evap_air_velocity_m_s": (IDX_EXPORT_EVAP_AIR_VELOCITY, 2, "m/s"),
    # T_ambient / RH_ambient deliberately omitted -- see module docstring.
}

BATCH_SENSOR_TAGS = {
    "t_pulp_c": (IDX_BATCH_SENSOR_T_PULP, 2, "C"),
    "mass_kg":  (IDX_BATCH_SENSOR_MASS, 3, "kg"),
}

# 1. Centralized SCADA Log Configuration Registry
SCADA_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "compressor": {
        "mapping": {
            "timestamp_ms": IDX_COMPRESSOR_EVENT_TIMESTAMP_MS,
            "is_active": IDX_COMPRESSOR_EVENT_IS_ACTIVE,
        }
    },
    "humidifier": {
        "mapping": {
            "timestamp_ms": IDX_HUMIDIFIER_EVENT_TIMESTAMP_MS,
            "num_discs_on": IDX_HUMIDIFIER_EVENT_NUM_DISCS_ON,
        }
    },
    "door": {
        "mapping": {
            "timestamp_ms": IDX_DOOR_EVENT_TIMESTAMP_MS,
            "door_code": IDX_DOOR_EVENT_DOOR_CODE,
            "is_open": IDX_DOOR_EVENT_IS_OPEN,
        }
    },
    "crop_rotation": {
        "mapping": {
            "timestamp_ms": IDX_CROP_ROTATION_EVENT_TIMESTAMP_MS,
            "fruit_type_code": IDX_CROP_ROTATION_EVENT_FRUIT_TYPE_CODE,
        }
    },
    "refrigerant_rotation": {
        "mapping": {
            "timestamp_ms": IDX_REFRIGERANT_ROTATION_EVENT_TIMESTAMP_MS,
            "refrigerant_code": IDX_REFRIGERANT_ROTATION_EVENT_REFRIGERANT_CODE,
        }
    },
    "lp_trip": {
        "mapping": {
            "timestamp_ms": IDX_LP_TRIP_EVENT_TIMESTAMP_MS,
            "switch_on": IDX_LP_TRIP_EVENT_TRIPPED,
        }
    },
    "defrost": {
        "mapping": {
            "timestamp_ms": IDX_DEFROST_EVENT_END_MS,
            "start_timestamp_ms": IDX_DEFROST_EVENT_START_MS,
            "termination_reason": IDX_DEFROST_EVENT_TERMINATION_REASON,
            "ice_melted_g": IDX_DEFROST_EVENT_FROST_MELTED_G,
            "coil_ice_melted_g": IDX_DEFROST_EVENT_COIL_ICE_MELTED_G,
            "pan_ice_melted_g": IDX_DEFROST_EVENT_PAN_ICE_MELTED_G,
            "face_ice_shed_g": IDX_DEFROST_EVENT_FACE_ICE_SHED_G,
        }
    },
    "downtime_sequence": {
        "mapping": {
            "timestamp_ms": IDX_DOWNTIME_SEQUENCE_EVENT_TIMESTAMP_MS,
            "scheduled_start_timestamp_ms": IDX_DOWNTIME_SEQUENCE_EVENT_SCHEDULED_START_TIMESTAMP_MS,
            "shutdown_timestamp_ms": IDX_DOWNTIME_SEQUENCE_EVENT_SHUTDOWN_TIMESTAMP_MS,
            "ramp_up_start_timestamp_ms": IDX_DOWNTIME_SEQUENCE_EVENT_RAMP_UP_START_TIMESTAMP_MS,
            "ramp_up_end_timestamp_ms": IDX_DOWNTIME_SEQUENCE_EVENT_RAMP_UP_END_TIMESTAMP_MS,
        },
        "nullifiable_timestamps": ["scheduled_start_timestamp_ms", "shutdown_timestamp_ms", "ramp_up_start_timestamp_ms", "ramp_up_end_timestamp_ms"]
    },
    "orders": {
        "mapping": {
            "order_id": IDX_ORDER_LEDGER_ORDER_ID,
            "timestamp_ms": IDX_ORDER_LEDGER_TIMESTAMP_MS,
            "target_fruit_mass_kg": IDX_ORDER_LEDGER_TARGET_FRUIT_MASS,
            "total_box_count": IDX_ORDER_LEDGER_TOTAL_BOX_COUNT,
            "minimum_grade": IDX_ORDER_LEDGER_MINIMUM_GRADE,
            "total_pallet_count": IDX_ORDER_LEDGER_TOTAL_PALLET_COUNT,
            "fruit_type_code": IDX_ORDER_LEDGER_CROP_TYPE,
            "assigned_door_code": IDX_ORDER_LEDGER_ASSIGNED_DOOR_CODE,
            "assigned_driver_code": IDX_ORDER_LEDGER_ASSIGNED_DRIVER,
            "dispatch_type": IDX_ORDER_LEDGER_DISPATCH_TYPE,
        },
        "rounding": {"target_fruit_mass_kg": 1}
    },
    "pallet_history": {
        "mapping": {
            "batch_id": IDX_HISTORICAL_PALLET_BATCH_ID,
            "parent_order_id": IDX_HISTORICAL_PALLET_PARENT_ORDER_ID,
            "timestamp_ms": IDX_HISTORICAL_PALLET_TIMESTAMP_MS,
            "quality_grade": IDX_HISTORICAL_PALLET_QUALITY_GRADE,
            "pulp_temp_c": IDX_HISTORICAL_PALLET_PULP_TEMP,
            "gross_start_kg": IDX_HISTORICAL_PALLET_GROSS_START,
            "box_start_count": IDX_HISTORICAL_PALLET_BOX_START,
            "gross_moved_kg": IDX_HISTORICAL_PALLET_GROSS_MOVED,
            "box_moved_count": IDX_HISTORICAL_PALLET_BOX_MOVED,
            "supervisor_intervention_ms": IDX_HISTORICAL_PALLET_SUPERVISOR_INTERVENTION_MS,
        },
        "rounding": {
            "pulp_temp_c": 2, # "pulp_temp_filtered_c": 2,
            "gross_start_kg": 2, "gross_moved_kg": 2, # "gross_start_filtered_kg": 2,
        },
        "nullifiable_timestamps": ["supervisor_intervention_ms"]
    }
}

def scadafy_telemetry(room_telemetry_matrix: np.ndarray, sentinel=SENTINEL_FLOAT64) -> pd.DataFrame:
    """
    room_telemetry_matrix (wide, one row per step, most cells SENTINEL_FLOAT64
    except on that channel's own polling step) -> long tag/timestamp/value table.
    Rows where the channel didn't actually poll that step are dropped entirely
    (a real historian has no row for a reading that never happened).
    """
    ts = room_telemetry_matrix[:, IDX_EXPORT_TIMESTAMP_MS].astype(np.int64)
    frames = []
    for tag_name, (col_idx, decimals, unit) in TELEMETRY_TAGS.items():
        values = room_telemetry_matrix[:, col_idx]
        polled = values != sentinel
        if not polled.any():
            continue
        frames.append(pd.DataFrame({
            "timestamp_ms": ts[polled],
            "tag_name": tag_name,
            "value": np.round(values[polled], decimals),
            "unit": unit,
        }))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["timestamp_ms", "tag_name", "value", "unit"]
    )
    return out.sort_values(["timestamp_ms", "tag_name"]).reset_index(drop=True)


def scadafy_batch_sensors(batch_fruit_sensor_ledger: np.ndarray) -> pd.DataFrame:
    """
    batch_fruit_sensor_ledger (already trimmed) -> long tag/timestamp/value table,
    dimensioned by batch_id + pallet_slot. Pulp temp and mass are two separate
    physical instruments (probe vs. load cell) even though the engine writes both
    on the same tick, so they're unpivoted into separate tag rows here to match
    how they'd actually arrive from two independent transmitters.
    """
    if len(batch_fruit_sensor_ledger) == 0:
        return pd.DataFrame(columns=["timestamp_ms", "timestamp", "batch_id", "pallet_slot", "tag_name", "value", "unit"])

    ts = batch_fruit_sensor_ledger[:, IDX_BATCH_SENSOR_TIMESTAMP_MS].astype(np.int64)
    batch_id = batch_fruit_sensor_ledger[:, IDX_BATCH_SENSOR_BATCH_ID].astype(np.int64)
    pallet_slot = batch_fruit_sensor_ledger[:, IDX_BATCH_SENSOR_PALLET_SLOT].astype(np.int64)
    # fruit_type_code = batch_fruit_sensor_ledger[:, IDX_BATCH_SENSOR_FRUIT_TYPE_CODE].astype(np.int64)

    frames = []
    for tag_name, (col_idx, decimals, unit) in BATCH_SENSOR_TAGS.items(): # [
    #     ("t_pulp_filtered_c", IDX_BATCH_SENSOR_T_PULP_FILTERED, 2, "C"),
    #     ("mass_filtered_kg", IDX_BATCH_SENSOR_MASS_FILTERED, 3, "kg"),
    # ]:
        frames.append(pd.DataFrame({
            "timestamp_ms": ts,
            "batch_id": batch_id,
            "pallet_slot": pallet_slot,
            "tag_name": tag_name,
            "value": np.round(batch_fruit_sensor_ledger[:, col_idx], decimals),
            "unit": unit,
        }))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["timestamp_ms", "batch_id", "tag_name"]).reset_index(drop=True)


def _named_frame(arr: np.ndarray, columns: Dict[str, int], epoch_sentinel_cols: list = None, sentinel=SENTINEL_FLOAT64, epoch_sentinel=MAX_EPOCH_TIMESTAMP_MS) -> pd.DataFrame:
    """
    Generic int64/float64 ledger -> named-column DataFrame. columns: name -> col_idx.
    SENTINEL_FLOAT64 values are converted to real NaN here -- this is a structural
    "field not applicable" marker (e.g. minimum_grade on an inbound order,
    supervisor_intervention_ms when there was no intervention), not simulated
    data loss, so it belongs in the scadafier's cleanup, not the nullifier.
    """
    if len(arr) == 0:
        return pd.DataFrame(columns=list(columns.keys()))
    data = {name: arr[:, idx] for name, idx in columns.items()}
    out = pd.DataFrame(data)
    for col_name in columns:
        if col_name in (epoch_sentinel_cols or []):
            out[col_name] = out[col_name].replace(epoch_sentinel, np.nan)
        else:
            out[col_name] = out[col_name].replace(sentinel, np.nan)
    # out = out.replace(sentinel, np.nan)
    return out


def scadafy_compressor_events(ledger: np.ndarray) -> pd.DataFrame:
    return _named_frame(ledger, {
        "timestamp_ms": IDX_COMPRESSOR_EVENT_TIMESTAMP_MS,
        "is_active": IDX_COMPRESSOR_EVENT_IS_ACTIVE,
    })


def scadafy_humidifier_events(ledger: np.ndarray) -> pd.DataFrame:
    return _named_frame(ledger, {
        "timestamp_ms": IDX_HUMIDIFIER_EVENT_TIMESTAMP_MS,
        "num_discs_on": IDX_HUMIDIFIER_EVENT_NUM_DISCS_ON,
    })


def scadafy_door_events(ledger: np.ndarray) -> pd.DataFrame:
    return _named_frame(ledger, {
        "timestamp_ms": IDX_DOOR_EVENT_TIMESTAMP_MS,
        "door_code": IDX_DOOR_EVENT_DOOR_CODE,
        "is_open": IDX_DOOR_EVENT_IS_OPEN,
    })


def scadafy_orders(ledger: np.ndarray) -> pd.DataFrame:
    df = _named_frame(ledger, {
        "order_id": IDX_ORDER_LEDGER_ORDER_ID,
        "timestamp_ms": IDX_ORDER_LEDGER_TIMESTAMP_MS,
        "target_fruit_mass_kg": IDX_ORDER_LEDGER_TARGET_FRUIT_MASS,
        "total_box_count": IDX_ORDER_LEDGER_TOTAL_BOX_COUNT,
        "minimum_grade": IDX_ORDER_LEDGER_MINIMUM_GRADE,
        "total_pallet_count": IDX_ORDER_LEDGER_TOTAL_PALLET_COUNT,
        "fruit_type_code": IDX_ORDER_LEDGER_CROP_TYPE,
        "assigned_door_code": IDX_ORDER_LEDGER_ASSIGNED_DOOR_CODE,
        "assigned_driver_code": IDX_ORDER_LEDGER_ASSIGNED_DRIVER,
        "dispatch_type": IDX_ORDER_LEDGER_DISPATCH_TYPE,
    })
    round_map = {
        "target_fruit_mass_kg": 1,
    }
    for col, decimals in round_map.items():
        df[col] = df[col].round(decimals)
    return df


def scadafy_pallet_history(ledger: np.ndarray) -> pd.DataFrame:
    df = _named_frame(ledger, {
        "batch_id": IDX_HISTORICAL_PALLET_BATCH_ID,
        "parent_order_id": IDX_HISTORICAL_PALLET_PARENT_ORDER_ID,
        "timestamp_ms": IDX_HISTORICAL_PALLET_TIMESTAMP_MS,
        "quality_grade": IDX_HISTORICAL_PALLET_QUALITY_GRADE,
        "pulp_temp_c": IDX_HISTORICAL_PALLET_PULP_TEMP,
        "gross_start_kg": IDX_HISTORICAL_PALLET_GROSS_START,
        "box_start_count": IDX_HISTORICAL_PALLET_BOX_START,
        "gross_moved_kg": IDX_HISTORICAL_PALLET_GROSS_MOVED,
        "box_moved_count": IDX_HISTORICAL_PALLET_BOX_MOVED,
        "supervisor_intervention_ms": IDX_HISTORICAL_PALLET_SUPERVISOR_INTERVENTION_MS,
    }, epoch_sentinel_cols=["supervisor_intervention_ms"])
    round_map = {
        "pulp_temp_c": 2, # "pulp_temp_filtered_c": 2,
        "gross_start_kg": 2, "gross_moved_kg": 2, # "gross_start_filtered_kg": 2,
    }
    for col, decimals in round_map.items():
        df[col] = df[col].round(decimals)
    return df

def scadafy_ledger(ledger_type:str, ledger: np.ndarray) -> pd.DataFrame:
    if ledger_type not in SCADA_SCHEMAS:
        raise ValueError(f"Unknown ledger type: '{ledger_type}'. Add it to SCADA_SCHEMAS.")
    cfg = SCADA_SCHEMAS[ledger_type]

    mapping = cfg["mapping"]
    rounding = cfg.get("rounding", {})
    epoch_sentinels = cfg.get("nullifiable_timestamps", None)

    df = _named_frame(ledger, mapping, epoch_sentinel_cols=epoch_sentinels)

    # Apply floating point rounding if defined in configuration schema
    for col, decimals in rounding.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").round(decimals)
        
    return df


def scadafy_all_outputs(out: Dict[str, np.ndarray]) -> Dict[str, pd.DataFrame]:
    """
    out: the dict returned by run_outputs_io.load_run_outputs() (or equivalent,
    already trimmed to real row counts). Returns one tidy DataFrame per structure,
    ready for the nullifier and then parquet.
    """
    return {
        "telemetry": scadafy_telemetry(out["room_telemetry_matrix"]),
        "batch_sensors": scadafy_batch_sensors(out["batch_fruit_sensor_ledger"]),
        "compressor_events":       scadafy_ledger("compressor", out["compressor_event_ledger"]),
        "humidifier_events":       scadafy_ledger("humidifier", out["humidifier_event_ledger"]),
        "door_events":             scadafy_ledger("door", out["door_event_ledger"]),
        "crop_rotation_events":        scadafy_ledger("crop_rotation", out["crop_rotation_event_ledger"]),
        "refrigerant_rotation_events": scadafy_ledger("refrigerant_rotation", out["refrigerant_rotation_event_ledger"]),
        "downtime_sequence_events": scadafy_ledger("downtime_sequence", out["downtime_sequence_event_ledger"]),
        "lp_trip_events": scadafy_ledger("lp_trip", out["lp_trip_event_ledger"]),
        "defrost_events": scadafy_ledger("defrost", out["defrost_event_ledger"]),
        "inbound_orders":          scadafy_ledger("orders", out["inbound_orders_ledger"]),
        "outbound_orders":         scadafy_ledger("orders", out["outbound_orders_ledger"]),
        "inbound_pallet_history":  scadafy_ledger("pallet_history", out["inbound_pallet_history_ledger"]),
        "outbound_pallet_history": scadafy_ledger("pallet_history", out["outbound_pallet_history_ledger"]),
    }
