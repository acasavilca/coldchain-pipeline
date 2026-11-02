"""
inspect_outputs.py

Standalone, not part of the run_simulator import chain. Loads whatever chunk
files run.py's save_run_outputs() has written to run_outputs/, and produces a
handful of physics-validation PNGs.

Uses output_float64_matrix / output_int64_matrix (IDX_OUTPUT_*), not
room_telemetry_matrix -- this is for checking the physics is doing what you
expect, and the raw debug matrix has the full-precision, ungapped values
(room_telemetry_matrix's per-sensor polling gaps and filtering are exactly
the SCADA-realism you don't want when you're trying to see the true signal).

Usage: python inspect_outputs.py [--out-dir run_outputs] [--plots-dir plots]
       python inspect_outputs.py --latest-chunk [--out-dir run_outputs]
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from constants import *

from matrix_index_maps import (
    DOOR_CHANNEL_INTERNAL,
    DOOR_CHANNEL_EXTERNAL,
    IDX_OUTPUT_EPOCH_TIMESTAMP_MS,
    IDX_OUTPUT_T_ROOM_C,
    IDX_OUTPUT_T_ROOM_FILTERED_C,
    IDX_OUTPUT_RH_ROOM,
    IDX_OUTPUT_RH_ROOM_FILTERED,
    IDX_OUTPUT_T_PULP_BULK_C,
    IDX_OUTPUT_T_FLOOR_C,
    IDX_OUTPUT_T_AMBIENT_C,
    IDX_OUTPUT_T_EVAP_C,
    IDX_OUTPUT_T_COND_C,
    IDX_OUTPUT_COP,
    IDX_OUTPUT_W_COMPRESSOR_W,
    IDX_OUTPUT_REFRIGERATION_IS_ACTIVE,
    IDX_OUTPUT_INT_DOOR_IS_OPEN,
    IDX_OUTPUT_EXT_DOOR_IS_OPEN,
    IDX_OUTPUT_COOLING_FRACTION,
    IDX_OUTPUT_SOLVER_N_ITERS,
    IDX_OUTPUT_SOLVER_ABS_ERROR,
    IDX_OUTPUT_TOTAL_BATCHES_IN_ROOM,
    IDX_OUTPUT_ACTIVE_FRUIT_TYPE_CODE,
    IDX_OUTPUT_Q_SENSIBLE_FORCED,
    IDX_OUTPUT_Q_SENSIBLE_FACE,
    IDX_OUTPUT_Q_LATENT,
    IDX_OUTPUT_Q_FANS,
    IDX_OUTPUT_Q_REFRIG_EVAP_W,
    IDX_OUTPUT_T_COIL_EVAP_C,
    IDX_OUTPUT_T_DEW_ROOM,
    IDX_OUTPUT_W_ROOM_HUMIDITY_RATIO,
    IDX_OUTPUT_W_COIL_EVAP_SAT,
    IDX_OUTPUT_Q_MELT,
    IDX_OUTPUT_M_FROST_KG,
    IDX_OUTPUT_V_FROST_M3,
    IDX_OUTPUT_M_PASSAGE_ICE_KG,
    IDX_OUTPUT_M_FACE_ICE_KG,
    IDX_OUTPUT_V_PASSAGE_ICE_M3,
    IDX_OUTPUT_V_FACE_ICE_M3,
    IDX_OUTPUT_T_PASSAGE_ICE_C,
    IDX_OUTPUT_T_FACE_INNER_ICE_C,
    IDX_OUTPUT_T_FACE_OUTER_ICE_C,
    IDX_OUTPUT_E_LIQUID_EXCESS_J,
    IDX_OUTPUT_M_DOT_FROST,
    IDX_OUTPUT_M_DOT_SUBLIMATED,
    IDX_OUTPUT_M_DOT_CONDENSED,
    IDX_OUTPUT_V_DOT_AIR_EVAP_ACTUAL,
    IDX_OUTPUT_EVAP_FAN_MODULATION_FRAC,
    IDX_OUTPUT_COND_FAN_MODULATION_FRAC,
    IDX_OUTPUT_EVAP_AIRFLOW_PROVED,
    IDX_OUTPUT_EVAP_AIR_VELOCITY_SENSED_M_S,
    IDX_OUTPUT_DEFROST_PHASE,
    IDX_OUTPUT_Q_DEFROST_HEATER_W,
    IDX_OUTPUT_T_COIL_EVAP_FILTERED_C,
    IDX_OUTPUT_M_DOT_FACE_ICE_SHED,
    IDX_OUTPUT_M_PAN_ICE_KG,
    IDX_OUTPUT_M_DOT_PAN_ICE_MELTED,
    IDX_OUTPUT_Q_COIL_DEFROST_HEATER_W,
    IDX_OUTPUT_Q_PAN_DEFROST_HEATER_W,
    IDX_OUTPUT_UA_COIL_EVAP_AIR,
    IDX_OUTPUT_P_EVAP_BAR,
    IDX_OUTPUT_P_SUCTION_LINE_FILTERED,
    IDX_OUTPUT_LP_TRIPPED,
    IDX_OUTPUT_THERMOSTAT_CALL,
    IDX_COMPRESSOR_EVENT_TIMESTAMP_MS,
    IDX_COMPRESSOR_EVENT_IS_ACTIVE,
    IDX_LP_TRIP_EVENT_TIMESTAMP_MS,
    IDX_LP_TRIP_EVENT_TRIPPED,
    IDX_DEFROST_EVENT_START_MS,
    IDX_DEFROST_EVENT_END_MS,
    IDX_DEFROST_EVENT_TERMINATION_REASON,
    IDX_DEFROST_EVENT_COIL_ICE_MELTED_G,
    IDX_DEFROST_EVENT_PAN_ICE_MELTED_G,
    IDX_DEFROST_EVENT_FACE_ICE_SHED_G,
    IDX_DOOR_EVENT_TIMESTAMP_MS,
    IDX_DOOR_EVENT_DOOR_CODE,
    IDX_DOOR_EVENT_IS_OPEN,
    IDX_BATCH_SENSOR_TIMESTAMP_MS,
    IDX_BATCH_SENSOR_BATCH_ID,
    IDX_BATCH_SENSOR_T_PULP,
    IDX_HISTORICAL_PALLET_QUALITY_GRADE,
    IDX_HISTORICAL_PALLET_BATCH_ID,
    IDX_ORDER_LEDGER_MINIMUM_GRADE,
    IDX_ORDER_LEDGER_ORDER_ID,
    IDX_EXPORT_TIMESTAMP_MS,
    IDX_EXPORT_T_ROOM,
    IDX_EXPORT_RH_ROOM,
)
# from constants import SENTINEL_FLOAT64
from run_outputs_io import load_run_outputs, trim_run_outputs, _find_chunks

# SENTINEL_FLOAT64 = -999.9 # NOTE: old value written in by mistake in arrays

def _hours(epoch_ms, t0):
    return (epoch_ms - t0) / MS_PER_HOUR


def _overview_slice(n_rows, max_points=20_000):
    """Keep full-run plots responsive without changing the saved data."""
    return slice(None, None, max(1, (n_rows + max_points - 1) // max_points))

def plot_room_and_refrigeration(out, plots_dir):
    m = out["output_float64_matrix"]
    i64 = out["output_int64_matrix"]
    t0 = i64[0, IDX_OUTPUT_EPOCH_TIMESTAMP_MS]
    hrs = _hours(i64[:, IDX_OUTPUT_EPOCH_TIMESTAMP_MS], t0)
    cop_vec = m[:, IDX_OUTPUT_COP].copy()
    cooling_frac_vec = m[:, IDX_OUTPUT_COOLING_FRACTION]
    mask = (cooling_frac_vec <= 0.99)
    cop_vec[(cop_vec == SENTINEL_FLOAT64) | mask] = np.nan

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(hrs, m[:, IDX_OUTPUT_T_ROOM_C], label="T_room (raw)", alpha=0.4)
    axes[0].plot(hrs, m[:, IDX_OUTPUT_T_ROOM_FILTERED_C], label="T_room (filtered)")
    axes[0].plot(hrs, m[:, IDX_OUTPUT_T_AMBIENT_C], label="T_ambient", alpha=0.5, linestyle="--")
    axes[0].plot(hrs, m[:, IDX_OUTPUT_T_FLOOR_C], label="T_floor", alpha=0.5)
    axes[0].plot(hrs, m[:, IDX_OUTPUT_T_PULP_BULK_C], label="T_pulp_bulk", alpha=0.7)
    axes[0].set_ylabel("Temp (C)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Room / ambient / fruit temperatures")

    axes[1].plot(hrs, m[:, IDX_OUTPUT_RH_ROOM], label="RH_room (raw)", alpha=0.4)
    axes[1].plot(hrs, m[:, IDX_OUTPUT_RH_ROOM_FILTERED], label="RH_room (filtered)")
    axes[1].set_ylabel("RH (frac)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("Room relative humidity")

    axes[2].plot(hrs, m[:, IDX_OUTPUT_T_EVAP_C], label="T_evap")
    axes[2].plot(hrs, m[:, IDX_OUTPUT_T_COND_C], label="T_cond")
    ax2b = axes[2].twinx()
    ax2b.plot(hrs, cop_vec, label="COP", color="green", alpha=0.5)
    ax2b.set_ylabel("COP")
    axes[2].set_ylabel("Temp (C)")
    axes[2].legend(loc="upper left", fontsize=8)
    ax2b.legend(loc="upper right", fontsize=8)
    axes[2].set_title("Refrigeration cycle: evap/cond temps + COP")

    is_active = out["output_int64_matrix"][:, IDX_OUTPUT_REFRIGERATION_IS_ACTIVE]
    axes[3].fill_between(hrs, 0, is_active, step="post", alpha=0.3, label="compressor active")
    axes[3].plot(hrs, m[:, IDX_OUTPUT_W_COMPRESSOR_W] / max(m[:, IDX_OUTPUT_W_COMPRESSOR_W].max(), 1e-9),
                 label="W_compressor (normalized)", alpha=0.6)
    axes[3].set_ylabel("on/off, normalized W")
    axes[3].set_xlabel("Hours since start")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].set_title("Compressor state and power")

    fig.tight_layout()
    fig.savefig(plots_dir / "room_and_refrigeration.png", dpi=120)
    plt.close(fig)


def plot_solver_health(out, plots_dir):
    m = out["output_float64_matrix"]
    i = out["output_int64_matrix"]
    t0 = i[0, IDX_OUTPUT_EPOCH_TIMESTAMP_MS]
    hrs = _hours(i[:, IDX_OUTPUT_EPOCH_TIMESTAMP_MS], t0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(hrs, i[:, IDX_OUTPUT_SOLVER_N_ITERS])
    axes[0].set_ylabel("solver n_iters")
    axes[0].set_title("Refrigeration cycle solver: iteration count (watch for pinning at max)")

    axes[1].plot(hrs, m[:, IDX_OUTPUT_SOLVER_ABS_ERROR])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("abs error (log)")
    axes[1].set_xlabel("Hours since start")
    axes[1].set_title("Refrigeration cycle solver: absolute error")

    fig.tight_layout()
    fig.savefig(plots_dir / "solver_health.png", dpi=120)
    plt.close(fig)


def plot_inventory_and_events(out, plots_dir):
    i = out["output_int64_matrix"]
    t0 = i[0, IDX_OUTPUT_EPOCH_TIMESTAMP_MS]
    hrs = _hours(i[:, IDX_OUTPUT_EPOCH_TIMESTAMP_MS], t0)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    axes[0].plot(hrs, i[:, IDX_OUTPUT_TOTAL_BATCHES_IN_ROOM], label="total batches in room")
    ax0b = axes[0].twinx()
    ax0b.step(hrs, i[:, IDX_OUTPUT_ACTIVE_FRUIT_TYPE_CODE], color="orange", alpha=0.5, label="active fruit type code")
    axes[0].set_ylabel("batch count")
    ax0b.set_ylabel("fruit type code")
    axes[0].set_title("Inventory and crop rotation")

    comp = out["compressor_event_ledger"]
    if len(comp):
        comp_hrs = _hours(comp[:, IDX_COMPRESSOR_EVENT_TIMESTAMP_MS], t0)
        axes[1].step(comp_hrs, comp[:, IDX_COMPRESSOR_EVENT_IS_ACTIVE], where="post")
    axes[1].set_ylabel("compressor on/off")
    axes[1].set_title(f"Compressor event ledger ({len(comp)} events)")

    door = out["door_event_ledger"]
    for code, label, state_idx in (
        (DOOR_CHANNEL_INTERNAL, "internal door", IDX_OUTPUT_INT_DOOR_IS_OPEN),
        (DOOR_CHANNEL_EXTERNAL, "external door", IDX_OUTPUT_EXT_DOOR_IS_OPEN),
    ):
        events = door[door[:, IDX_DOOR_EVENT_DOOR_CODE] == code]
        event_hrs = _hours(events[:, IDX_DOOR_EVENT_TIMESTAMP_MS], t0)
        states = events[:, IDX_DOOR_EVENT_IS_OPEN]
        # The ledger records transitions only. Include the observed initial and
        # final states so each door line spans the full run.
        trace_hrs = np.concatenate(([hrs[0]], event_hrs, [hrs[-1]]))
        trace_states = np.concatenate(([i[0, state_idx]], states, [i[-1, state_idx]]))
        axes[2].step(trace_hrs, trace_states, where="post", label=label)
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_ylabel("door open/closed")
    axes[2].legend(fontsize=8)
    axes[2].set_title(f"Door event ledger ({len(door)} events)")

    sample = _overview_slice(len(i))
    axes[3].plot(hrs[sample], out["output_float64_matrix"][sample, IDX_OUTPUT_M_FROST_KG])
    axes[3].set_ylabel("frost mass (kg)")
    axes[3].set_xlabel("Hours since start")
    axes[3].set_title("Frost mass aligned with door events")
    axes[3].set_xlim(hrs[0], hrs[-1])

    fig.tight_layout()
    fig.savefig(plots_dir / "inventory_and_events.png", dpi=120)
    plt.close(fig)


def plot_batch_sensors(out, plots_dir):
    b = out["batch_fruit_sensor_ledger"]
    if len(b) == 0:
        print("batch_fruit_sensor_ledger is empty, skipping batch sensor plot.")
        return
    t0 = b[0, IDX_BATCH_SENSOR_TIMESTAMP_MS]
    hrs = _hours(b[:, IDX_BATCH_SENSOR_TIMESTAMP_MS], t0)

    fig, ax = plt.subplots(figsize=(14, 6))
    batch_ids = np.unique(b[:, IDX_BATCH_SENSOR_BATCH_ID])
    # avoid an unreadable legend if there are hundreds of batches
    sample_ids = batch_ids if len(batch_ids) <= 20 else np.random.choice(batch_ids, 20, replace=False)
    for bid in sample_ids:
        mask = b[:, IDX_BATCH_SENSOR_BATCH_ID] == bid
        ax.plot(hrs[mask], b[mask, IDX_BATCH_SENSOR_T_PULP], alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Hours since start")
    ax.set_ylabel("T_pulp_filtered (C)")
    ax.set_title(f"Batch pulp temperature ({len(batch_ids)} total batches, {len(sample_ids)} shown)")

    fig.tight_layout()
    fig.savefig(plots_dir / "batch_pulp_temperatures.png", dpi=120)
    plt.close(fig)


def plot_quality_grades(out, plots_dir):
    inbound_hist = out["inbound_pallet_history_ledger"]
    outbound_orders = out["outbound_orders_ledger"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    valid_in = inbound_hist[:, IDX_HISTORICAL_PALLET_BATCH_ID] > 0.0
    grades_in = (inbound_hist[valid_in, IDX_HISTORICAL_PALLET_QUALITY_GRADE]).astype(np.int8)
    if len(grades_in):
        codes, counts = np.unique(grades_in, return_counts=True)
        axes[0].bar(codes, counts)
    axes[0].set_xlabel("Quality grade code (0 = best)")
    axes[0].set_ylabel("Pallet count")
    axes[0].set_title(f"What reaches inventory (n={len(grades_in)})")

    valid_out = outbound_orders[:, IDX_ORDER_LEDGER_ORDER_ID] > 0.0
    min_grades = outbound_orders[valid_out, IDX_ORDER_LEDGER_MINIMUM_GRADE]
    # sentinel-filtered: not every outbound order specifies a minimum grade
    min_grades = (min_grades[min_grades != SENTINEL_FLOAT64]).astype(np.int8)
    if len(min_grades):
        codes, counts = np.unique(min_grades, return_counts=True)
        axes[1].bar(codes, counts, color="orange")
    axes[1].set_xlabel("Minimum grade requested (0 = best)")
    axes[1].set_ylabel("Order count")
    axes[1].set_title(f"What's demanded by clients (n={len(min_grades)})")

    fig.tight_layout()
    fig.savefig(plots_dir / "quality_grades.png", dpi=120)
    plt.close(fig)


def plot_telemetry_sample(out, plots_dir):
    """
    Sanity-check the actual SCADA-bound export (room_telemetry_matrix), not
    the internal debug matrix -- confirms the per-sensor polling-cadence gaps
    look like real sparse SCADA data before you build the scadafier around it.
    """
    tel = out["room_telemetry_matrix"]
    t0 = tel[0, IDX_EXPORT_TIMESTAMP_MS]
    hrs = _hours(tel[:, IDX_EXPORT_TIMESTAMP_MS], t0)

    t_room = tel[:, IDX_EXPORT_T_ROOM].copy()
    rh_room = tel[:, IDX_EXPORT_RH_ROOM].copy()
    t_room[t_room == SENTINEL_FLOAT64] = np.nan
    rh_room[rh_room == SENTINEL_FLOAT64] = np.nan

    # zoom to the first 48h -- the polling gaps are the point, and they're
    # invisible if you plot weeks of data at once
    zoom_mask = hrs <= 48

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(hrs[zoom_mask], t_room[zoom_mask], marker=".", markersize=2, linewidth=0.5)
    axes[0].set_ylabel("T_room (C)")
    axes[0].set_title("room_telemetry_matrix T_room -- gaps are real (unpolled steps), not missing data")

    axes[1].plot(hrs[zoom_mask], rh_room[zoom_mask], marker=".", markersize=2, linewidth=0.5, color="orange")
    axes[1].set_ylabel("RH_room (frac)")
    axes[1].set_xlabel("Hours since start")
    axes[1].set_title("room_telemetry_matrix RH_room")

    fig.tight_layout()
    fig.savefig(plots_dir / "telemetry_sample.png", dpi=120)
    plt.close(fig)

def plot_coil_heat_flows(out, plots_dir):
    m = out["output_float64_matrix"]; i64 = out["output_int64_matrix"]
    hrs = _hours(i64[:, IDX_OUTPUT_EPOCH_TIMESTAMP_MS], i64[0, IDX_OUTPUT_EPOCH_TIMESTAMP_MS])
    zoom = hrs <= 48
    fig, ax = plt.subplots(figsize=(14, 5))
    for idx, label in [(IDX_OUTPUT_Q_SENSIBLE_FORCED, "Q_sensible_forced"), (IDX_OUTPUT_Q_SENSIBLE_FACE, "Q_sensible_face"), (IDX_OUTPUT_Q_LATENT, "Q_latent"),
                       (IDX_OUTPUT_Q_REFRIG_EVAP_W, "Q_refrig_evap"), (IDX_OUTPUT_Q_FANS, "Q_fans")]:
        ax.plot(hrs[zoom], m[zoom, idx], label=label, linewidth=0.7)
    ax.set_ylabel("W"); ax.set_xlabel("Hours since start"); ax.legend(fontsize=8)
    ax.set_title("Evaporator heat flows (first 48 h)")
    fig.tight_layout(); fig.savefig(plots_dir / "coil_heat_flows.png", dpi=120); plt.close(fig)

def plot_frost_diagnostics(out, plots_dir, zoom_hours=None):
    m = out["output_float64_matrix"]; i64 = out["output_int64_matrix"]
    t_ms = i64[:, IDX_OUTPUT_EPOCH_TIMESTAMP_MS]
    hrs = _hours(t_ms, t_ms[0])
    dt_s = np.median(np.diff(t_ms)) / 1000.0
    if zoom_hours is None:
        zoom_hours = (24.0, 30.0) if hrs[-1] > 30.0 else (max(0.0, hrs[-1] - 6.0), hrs[-1])

    m_frost = m[:, IDX_OUTPUT_M_FROST_KG]
    dep, sub, cond = m[:, IDX_OUTPUT_M_DOT_FROST], m[:, IDX_OUTPUT_M_DOT_SUBLIMATED], m[:, IDX_OUTPUT_M_DOT_CONDENSED]
    q_melt = m[:, IDX_OUTPUT_Q_MELT]

    has_pan = m.shape[1] > IDX_OUTPUT_M_DOT_PAN_ICE_MELTED
    shed = m[:, IDX_OUTPUT_M_DOT_FACE_ICE_SHED] if has_pan else np.zeros_like(dep)
    pan_melt = m[:, IDX_OUTPUT_M_DOT_PAN_ICE_MELTED] if has_pan else np.zeros_like(dep)
    # Coil ice loses mass through melting, sublimation, and shedding.
    dep_kg, sub_kg = dep[1:].sum() * dt_s, sub[1:].sum() * dt_s
    melt_kg = q_melt[1:].sum() * dt_s / H_FUSION
    shed_kg = shed[1:].sum() * dt_s
    resid = (m_frost[-1] - m_frost[0]) - (dep_kg - sub_kg - melt_kg - shed_kg)
    print(f"[frost] final M_frost={m_frost[-1]:.3f} kg | deposited={dep_kg:.3f} | sublimated={sub_kg:.3f} "
          f"| melted={melt_kg:.3f} | shed={shed_kg:.3f} | closure residual={resid:.2e} kg | peak M_frost={m_frost.max():.3f} kg")
    if has_pan:
        pan_mass = m[:, IDX_OUTPUT_M_PAN_ICE_KG]
        pan_melt_kg = pan_melt[1:].sum() * dt_s
        pan_resid = (pan_mass[-1] - pan_mass[0]) - (shed_kg - pan_melt_kg)
        print(f"[pan] final ice={pan_mass[-1]:.3f} kg | melted/drained={pan_melt_kg:.3f} kg | closure residual={pan_resid:.2e} kg")
    if m.shape[1] > IDX_OUTPUT_E_LIQUID_EXCESS_J:
        print(f"[frost] sensible energy exported with melted ice={m[:, IDX_OUTPUT_E_LIQUID_EXCESS_J].sum():.3f} J")

    # Overview: frost mass and cumulative flows over the whole run
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(hrs, m_frost); axes[0].set_ylabel("M_frost (kg)"); axes[0].set_title("Frost mass on the coil")
    axes[1].plot(hrs, np.cumsum(dep) * dt_s, label="deposited")
    axes[1].plot(hrs, np.cumsum(sub) * dt_s, label="sublimated")
    axes[1].plot(hrs, np.cumsum(q_melt) * dt_s / H_FUSION, label="melted")
    if has_pan:
        axes[1].plot(hrs, np.cumsum(shed) * dt_s, label="shed to pan")
        axes[1].plot(hrs, np.cumsum(pan_melt) * dt_s, label="pan ice melted")
    axes[1].plot(hrs, np.cumsum(cond) * dt_s, label="liquid condensate", linestyle="--")
    axes[1].set_ylabel("cumulative (kg)"); axes[1].legend(fontsize=8)
    axes[2].plot(hrs, m[:, IDX_OUTPUT_V_DOT_AIR_EVAP_ACTUAL]); axes[2].set_ylabel("evaporator airflow (m3/s)"); axes[2].set_xlabel("Hours since start")
    fig.tight_layout(); fig.savefig(plots_dir / "frost_overview.png", dpi=120); plt.close(fig)

    # Zoom: regime view over a few cycles
    z = (hrs >= zoom_hours[0]) & (hrs <= zoom_hours[1])
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(hrs[z], m[z, IDX_OUTPUT_T_COIL_EVAP_C], label="T_coil_evap")
    axes[0].plot(hrs[z], m[z, IDX_OUTPUT_T_DEW_ROOM], label="T_dew_room", alpha=0.6)
    axes[0].axhline(0.0, color="k", linewidth=0.5); axes[0].set_ylabel("C"); axes[0].legend(fontsize=8)
    axes[0].set_title("Coil temperature vs room dew point")
    axes[1].plot(hrs[z], m[z, IDX_OUTPUT_W_ROOM_HUMIDITY_RATIO] * 1e3, label="W_room")
    axes[1].plot(hrs[z], m[z, IDX_OUTPUT_W_COIL_EVAP_SAT] * 1e3, label="W_sat at coil surface")
    axes[1].set_ylabel("g/kg"); axes[1].legend(fontsize=8); axes[1].set_title("Room vs coil-surface humidity ratio (sign sets deposition vs sublimation)")
    axes[2].plot(hrs[z], dep[z] * 1e3, label="frost deposition")
    axes[2].plot(hrs[z], sub[z] * 1e3, label="sublimation")
    axes[2].plot(hrs[z], cond[z] * 1e3, label="liquid condensate")
    axes[2].set_ylabel("g/s"); axes[2].legend(fontsize=8); axes[2].set_title("Water flows at the coil")
    axes[3].plot(hrs[z], m[z, IDX_OUTPUT_Q_SENSIBLE_FORCED] / 1e3, label="Q_sensible_forced")
    axes[3].plot(hrs[z], m[z, IDX_OUTPUT_Q_SENSIBLE_FACE] / 1e3, label="Q_sensible_face")
    axes[3].plot(hrs[z], m[z, IDX_OUTPUT_Q_LATENT] / 1e3, label="Q_latent")
    axes[3].plot(hrs[z], m[z, IDX_OUTPUT_Q_MELT] / 1e3, label="Q_melt")
    axes[3].plot(hrs[z], m[z, IDX_OUTPUT_Q_REFRIG_EVAP_W] / 1e3, label="Q_refrig_evap", alpha=0.6)
    axes[3].set_ylabel("kW"); axes[3].set_xlabel("Hours since start"); axes[3].legend(fontsize=8)
    axes[3].set_title("Coil heat flows (Q_latent < 0 while frost sublimates)")
    fig.tight_layout(); fig.savefig(plots_dir / "frost_zoom.png", dpi=120); plt.close(fig)


def plot_frost_feedback(out, plots_dir):
    """Separate frost blockage from deliberate fan shutdown over the whole run."""
    m = out["output_float64_matrix"]
    if m.shape[1] <= IDX_OUTPUT_UA_COIL_EVAP_AIR:
        print("Skipping frost_feedback.png: this run predates the Phase 3 output columns.")
        return

    i64 = out["output_int64_matrix"]
    sample = _overview_slice(len(m))
    hrs = _hours(i64[sample, IDX_OUTPUT_EPOCH_TIMESTAMP_MS], i64[0, IDX_OUTPUT_EPOCH_TIMESTAMP_MS])
    mass = m[sample, IDX_OUTPUT_M_FROST_KG]
    volume = m[sample, IDX_OUTPUT_V_FROST_M3]
    density = np.full(mass.shape, np.nan)
    present = (mass > MASS_KG_FLOOR) & (volume > 0.0)
    density[present] = mass[present] / volume[present]

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    mass_line, = axes[0].plot(hrs, mass, color="tab:blue", label="Frost mass")
    axes[0].set_ylabel("Mass (kg)")
    volume_axis = axes[0].twinx()
    volume_line, = volume_axis.plot(hrs, volume * 1e3, color="tab:cyan",
                                    linestyle="--", label="Frost volume")
    volume_axis.set_ylabel("Volume (L)")
    axes[0].legend(handles=[mass_line, volume_line], loc="upper left", fontsize=8)
    axes[0].set_title("Frost inventory: temperature alone should not change mass or volume")

    axes[1].plot(hrs, density, color="tab:purple")
    axes[1].set_ylabel("Bulk density (kg/m³)")
    axes[1].set_title("Bulk frost density = mass / volume")

    flow_line, = axes[2].plot(hrs, m[sample, IDX_OUTPUT_V_DOT_AIR_EVAP_ACTUAL],
                             color="tab:blue", label="Through-coil airflow")
    axes[2].set_ylabel("Airflow (m³/s)")
    fan_axis = axes[2].twinx()
    evap_line, = fan_axis.plot(hrs, m[sample, IDX_OUTPUT_EVAP_FAN_MODULATION_FRAC],
                               color="tab:orange", label="Evaporator fan modulation", alpha=0.8)
    cond_line, = fan_axis.plot(hrs, m[sample, IDX_OUTPUT_COND_FAN_MODULATION_FRAC],
                               color="tab:green", label="Condenser fan modulation", alpha=0.6)
    fan_lines = [flow_line, evap_line, cond_line]
    if i64.shape[1] > IDX_OUTPUT_EVAP_AIRFLOW_PROVED:
        proof_line, = fan_axis.step(
            hrs, i64[sample, IDX_OUTPUT_EVAP_AIRFLOW_PROVED], where="post",
            color="tab:red", label="Evaporator airflow proved", alpha=0.8,
        )
        fan_lines.append(proof_line)
    fan_axis.set_ylabel("Fan modulation (0–1)")
    fan_axis.set_ylim(-0.05, 1.05)
    axes[2].legend(handles=fan_lines, loc="upper right", fontsize=8)
    axes[2].set_title("Falling airflow with a running fan indicates coil blockage")

    axes[3].plot(hrs, m[sample, IDX_OUTPUT_UA_COIL_EVAP_AIR] / 1e3, color="tab:red")
    axes[3].set_ylabel("Effective UA (kW/K)")
    axes[3].set_xlabel("Hours since start")
    axes[3].set_title("Evaporator air-side UA after frost resistance")

    fig.tight_layout()
    fig.savefig(plots_dir / "frost_feedback.png", dpi=120)
    plt.close(fig)


def plot_airflow_proof_sensor(out, plots_dir):
    """Compare the flow switch's sampled velocity with its delayed relay contact."""
    m = out["output_float64_matrix"]
    i64 = out["output_int64_matrix"]
    if m.shape[1] <= IDX_OUTPUT_EVAP_AIR_VELOCITY_SENSED_M_S:
        print("Skipping airflow_proof_sensor.png: this run predates the sampled velocity output.")
        return

    sample = _overview_slice(len(m))
    hours = _hours(i64[sample, IDX_OUTPUT_EPOCH_TIMESTAMP_MS], i64[0, IDX_OUTPUT_EPOCH_TIMESTAMP_MS])
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(hours, m[sample, IDX_OUTPUT_EVAP_AIR_VELOCITY_SENSED_M_S],
                 label="Flow switch sensed velocity")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title("Sampled and filtered airflow measurement")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].step(hours, i64[sample, IDX_OUTPUT_EVAP_AIRFLOW_PROVED],
                 where="post", label="Flow switch contact")
    axes[1].step(hours, i64[sample, IDX_OUTPUT_REFRIGERATION_IS_ACTIVE],
                 where="post", label="Compressor", alpha=0.7)
    axes[1].plot(hours, m[sample, IDX_OUTPUT_EVAP_FAN_MODULATION_FRAC],
                 label="Evaporator fan", alpha=0.7)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("State / fraction")
    axes[1].set_xlabel("Hours since start")
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(plots_dir / "airflow_proof_sensor.png", dpi=120)
    plt.close(fig)


def plot_frost_regions(out, plots_dir, sentinel=SENTINEL_FLOAT64):
    """Show the two inventories and the three ice temperatures on one time axis."""
    m = out["output_float64_matrix"]
    if m.shape[1] <= IDX_OUTPUT_T_FACE_OUTER_ICE_C:
        print("Skipping frost_regions.png: this run predates the two-region outputs.")
        return
    i64 = out["output_int64_matrix"]
    sample = _overview_slice(len(m))
    hrs = _hours(
        i64[sample, IDX_OUTPUT_EPOCH_TIMESTAMP_MS],
        i64[0, IDX_OUTPUT_EPOCH_TIMESTAMP_MS],
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(hrs, m[sample, IDX_OUTPUT_M_PASSAGE_ICE_KG], label="Passage ice")
    axes[0].plot(hrs, m[sample, IDX_OUTPUT_M_FACE_ICE_KG], label="Face ice")
    axes[0].set_ylabel("Mass (kg)")
    axes[0].legend(fontsize=8)

    axes[1].plot(hrs, m[sample, IDX_OUTPUT_V_PASSAGE_ICE_M3] * 1e3, label="Passage ice")
    axes[1].plot(hrs, m[sample, IDX_OUTPUT_V_FACE_ICE_M3] * 1e3, label="Face ice")
    axes[1].set_ylabel("Volume (L)")
    axes[1].legend(fontsize=8)

    axes[2].plot(hrs, m[sample, IDX_OUTPUT_T_COIL_EVAP_C], label="Metal")

    t_passage_vec = m[sample, IDX_OUTPUT_T_PASSAGE_ICE_C]
    t_inner_vec = m[sample, IDX_OUTPUT_T_FACE_INNER_ICE_C]
    t_outer_vec = m[sample, IDX_OUTPUT_T_FACE_OUTER_ICE_C]
    axes[2].plot(hrs, np.place(t_passage_vec, t_passage_vec==sentinel, np.nan), label="Passage ice")
    axes[2].plot(hrs, np.place(t_inner_vec, t_inner_vec==sentinel, np.nan), label="Face inner")
    axes[2].plot(hrs, np.place(t_outer_vec, t_outer_vec==sentinel, np.nan), label="Face outer")
    axes[2].axhline(0.0, color="black", linewidth=0.5)
    axes[2].set_ylabel("Temperature (C)")
    axes[2].set_xlabel("Hours since start")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(plots_dir / "frost_regions.png", dpi=120)
    plt.close(fig)


def plot_defrost_diagnostics(out, plots_dir):
    """Show every defrost and a detailed view of the first cycle."""
    m = out["output_float64_matrix"]
    i64 = out["output_int64_matrix"]
    if (
        i64.shape[1] <= IDX_OUTPUT_DEFROST_PHASE
        or m.shape[1] <= IDX_OUTPUT_T_COIL_EVAP_FILTERED_C
    ):
        print("Skipping defrost diagnostics: this run predates the defrost output columns.")
        return
    if len(i64) == 0:
        return

    t_ms = i64[:, IDX_OUTPUT_EPOCH_TIMESTAMP_MS]
    t0 = t_ms[0]
    phase = i64[:, IDX_OUTPUT_DEFROST_PHASE]
    phase_names = ("NORMAL", "WAIT", "SETTLE", "HEAT", "DRIP", "FAN_DELAY")
    sample = _overview_slice(len(i64))
    hours = _hours(t_ms[sample], t0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].step(hours, phase[sample], where="post", color="tab:purple")
    defrost_events = out.get("defrost_event_ledger")
    if defrost_events is not None:
        for event in defrost_events:
            reason = event[IDX_DEFROST_EVENT_TERMINATION_REASON]
            reason_name = (
                "probe" if reason == DF_TERMINATION_PROBE else
                "timeout" if reason == DF_TERMINATION_TIMEOUT else "unknown"
            )
            color = {"probe": "tab:green", "timeout": "tab:orange", "unknown": "tab:gray"}[reason_name]
            event_hour = (event[IDX_DEFROST_EVENT_END_MS] - t0) / 3_600_000.0
            axes[0].axvline(event_hour, color=color, alpha=0.5, linewidth=1.0)
            print(
                f"[defrost] {event[IDX_DEFROST_EVENT_START_MS]} -> {event[IDX_DEFROST_EVENT_END_MS]} "
                f"reason={reason_name} coil_melted={event[IDX_DEFROST_EVENT_COIL_ICE_MELTED_G]} g "
                f"pan_melted={event[IDX_DEFROST_EVENT_PAN_ICE_MELTED_G]} g "
                f"face_shed={event[IDX_DEFROST_EVENT_FACE_ICE_SHED_G]} g"
            )
    axes[0].set_yticks(range(6), phase_names)
    axes[0].set_ylabel("Defrost phase")
    axes[0].set_title("Defrost schedule and heater operation (green: probe, orange: timeout)")
    heater_axis = axes[0].twinx()
    heater_axis.plot(hours, m[sample, IDX_OUTPUT_Q_DEFROST_HEATER_W] / 1e3,
                     color="tab:red", alpha=0.7, label="Heater")
    heater_axis.set_ylabel("Heater (kW)")

    axes[1].plot(hours, m[sample, IDX_OUTPUT_T_COIL_EVAP_C], label="Metal")
    axes[1].plot(hours, m[sample, IDX_OUTPUT_T_COIL_EVAP_FILTERED_C],
                 label="Termination probe", alpha=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].legend(fontsize=8)

    axes[2].plot(hours, m[sample, IDX_OUTPUT_M_PASSAGE_ICE_KG], label="Passage ice")
    axes[2].plot(hours, m[sample, IDX_OUTPUT_M_FACE_ICE_KG], label="Face ice")
    if m.shape[1] > IDX_OUTPUT_M_PAN_ICE_KG:
        axes[2].plot(hours, m[sample, IDX_OUTPUT_M_PAN_ICE_KG], label="Pan ice")
    axes[2].set_ylabel("Ice mass (kg)")
    axes[2].set_xlabel("Hours since start")
    axes[2].legend(loc="upper left", fontsize=8)
    flow_axis = axes[2].twinx()
    flow_axis.plot(hours, m[sample, IDX_OUTPUT_V_DOT_AIR_EVAP_ACTUAL],
                   color="tab:gray", alpha=0.65, label="Evaporator airflow")
    flow_axis.set_ylabel("Airflow (m³/s)")

    fig.tight_layout()
    fig.savefig(plots_dir / "defrost_overview.png", dpi=120)
    plt.close(fig)

    active_indices = np.flatnonzero(phase != DF_NORMAL)
    if len(active_indices) == 0:
        print("[defrost] No defrost phase occurred in this run.")
        return

    first_start = int(active_indices[0])
    normal_after = np.flatnonzero(phase[first_start + 1:] == DF_NORMAL)
    first_end = (
        first_start + 1 + int(normal_after[0])
        if len(normal_after) else len(phase) - 1
    )
    margin_ms = 15 * 60 * 1000
    window_start_ms = max(t0, t_ms[first_start] - margin_ms)
    window_end_ms = min(t_ms[-1], t_ms[first_end] + margin_ms)
    left = int(np.searchsorted(t_ms, window_start_ms, side="left"))
    right = int(np.searchsorted(t_ms, window_end_ms, side="right"))
    stride = max(1, (right - left + 20_000 - 1) // 20_000)
    z = slice(left, right, stride)
    zoom_hours = _hours(t_ms[z], t0)

    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    axes[0].step(zoom_hours, phase[z], where="post", color="tab:purple")
    axes[0].set_yticks(range(6), phase_names)
    axes[0].set_ylabel("Phase")
    axes[0].set_title("First defrost cycle")

    for column, label in (
        (IDX_OUTPUT_T_COIL_EVAP_C, "Metal"),
        (IDX_OUTPUT_T_COIL_EVAP_FILTERED_C, "Termination probe"),
        (IDX_OUTPUT_T_PASSAGE_ICE_C, "Passage ice"),
        (IDX_OUTPUT_T_FACE_INNER_ICE_C, "Face inner ice"),
        (IDX_OUTPUT_T_FACE_OUTER_ICE_C, "Face outer ice"),
    ):
        axes[1].plot(zoom_hours, m[z, column], label=label)
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].legend(loc="upper right", fontsize=8)

    for column, label in (
        (IDX_OUTPUT_Q_DEFROST_HEATER_W, "Heater"),
        (IDX_OUTPUT_Q_REFRIG_EVAP_W, "Refrigerant extraction"),
        (IDX_OUTPUT_Q_MELT, "Fusion heat"),
    ):
        axes[2].plot(zoom_hours, m[z, column] / 1e3, label=label)
    if m.shape[1] > IDX_OUTPUT_Q_PAN_DEFROST_HEATER_W:
        axes[2].plot(zoom_hours, m[z, IDX_OUTPUT_Q_COIL_DEFROST_HEATER_W] / 1e3, label="Coil heater", linestyle="--")
        axes[2].plot(zoom_hours, m[z, IDX_OUTPUT_Q_PAN_DEFROST_HEATER_W] / 1e3, label="Pan heater", linestyle="--")
    axes[2].set_ylabel("Heat rate (kW)")
    axes[2].legend(loc="upper right", fontsize=8)

    axes[3].plot(zoom_hours, m[z, IDX_OUTPUT_M_PASSAGE_ICE_KG], label="Passage ice")
    axes[3].plot(zoom_hours, m[z, IDX_OUTPUT_M_FACE_ICE_KG], label="Face ice")
    if m.shape[1] > IDX_OUTPUT_M_PAN_ICE_KG:
        axes[3].plot(zoom_hours, m[z, IDX_OUTPUT_M_PAN_ICE_KG], label="Pan ice")
    axes[3].set_ylabel("Ice mass (kg)")
    axes[3].legend(loc="upper left", fontsize=8)
    flow_axis = axes[3].twinx()
    flow_axis.plot(zoom_hours, m[z, IDX_OUTPUT_V_DOT_AIR_EVAP_ACTUAL],
                   color="tab:gray", alpha=0.7)
    flow_axis.set_ylabel("Airflow (m³/s)")

    axes[4].step(zoom_hours, i64[z, IDX_OUTPUT_REFRIGERATION_IS_ACTIVE],
                 where="post", label="Compressor")
    axes[4].step(zoom_hours, i64[z, IDX_OUTPUT_EVAP_AIRFLOW_PROVED],
                 where="post", label="Raw airflow proof")
    axes[4].plot(zoom_hours, m[z, IDX_OUTPUT_EVAP_FAN_MODULATION_FRAC],
                 label="Evaporator fan")
    axes[4].plot(zoom_hours, m[z, IDX_OUTPUT_COND_FAN_MODULATION_FRAC],
                 label="Condenser fan")
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].set_ylabel("State / fraction")
    axes[4].set_xlabel("Hours since start")
    axes[4].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(plots_dir / "defrost_first_cycle.png", dpi=120)
    plt.close(fig)


def plot_lp_control(out, plots_dir):
    """Show the measured pressure that drives the LP switch and its outcomes."""
    m = out["output_float64_matrix"]
    i64 = out["output_int64_matrix"]
    if i64.shape[1] <= IDX_OUTPUT_THERMOSTAT_CALL:
        print("Skipping lp_control.png: this run predates the Phase 3 control columns.")
        return

    sample = _overview_slice(len(m))
    t0 = i64[0, IDX_OUTPUT_EPOCH_TIMESTAMP_MS]
    hrs = _hours(i64[sample, IDX_OUTPUT_EPOCH_TIMESTAMP_MS], t0)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(hrs, m[sample, IDX_OUTPUT_P_EVAP_BAR], label="Evaporating pressure (raw)", alpha=0.75)
    axes[0].plot(hrs, m[sample, IDX_OUTPUT_P_SUCTION_LINE_FILTERED],
                 label="Suction pressure sensor (filtered)", alpha=0.8)
    axes[0].set_ylabel("Pressure (bar)")
    axes[0].set_title("Low-pressure control input")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].step(hrs, i64[sample, IDX_OUTPUT_THERMOSTAT_CALL], where="post",
                 linestyle="--", label="Thermostat call")
    axes[1].step(hrs, i64[sample, IDX_OUTPUT_REFRIGERATION_IS_ACTIVE],
                 where="post", label="Compressor on", alpha=0.85)
    axes[1].set_ylim(-0.15, 1.15)
    axes[1].set_yticks((0, 1))
    axes[1].set_ylabel("Call / compressor")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("Thermostat demand and compressor response")

    axes[2].step(hrs, i64[sample, IDX_OUTPUT_LP_TRIPPED],
                 where="post", color="tab:red", label="LP tripped")
    lp_events = out.get("lp_trip_event_ledger")
    if lp_events is not None and len(lp_events):
        event_hrs = _hours(lp_events[:, IDX_LP_TRIP_EVENT_TIMESTAMP_MS], t0)
        axes[2].scatter(event_hrs, lp_events[:, IDX_LP_TRIP_EVENT_TRIPPED],
                        marker="x", s=28, color="black", label="LP transitions", zorder=4)
    axes[2].set_ylim(-0.15, 1.15)
    axes[2].set_yticks((0, 1))
    axes[2].set_ylabel("LP switch")
    axes[2].set_xlabel("Hours since start")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("Low-pressure trip and reset")

    fig.tight_layout()
    fig.savefig(plots_dir / "lp_control.png", dpi=120)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="run_outputs")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument(
        "--latest-chunk", action="store_true",
        help="Inspect only the newest run chunk (useful when older chunks have a different output schema).",
    )
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading chunks from {args.out_dir}/ ...")
    if args.latest_chunk:
        chunks = _find_chunks(args.out_dir)
        if not chunks:
            raise FileNotFoundError(f"No run_outputs_chunk_*.npz files found in {args.out_dir}/")
        with np.load(chunks[-1]) as data:
            out = trim_run_outputs({key: data[key] for key in data.files})
        print(f"Inspecting latest chunk: {chunks[-1].name}")
    else:
        out = load_run_outputs(args.out_dir)
    print(f"Loaded {out['output_float64_matrix'].shape[0]} total simulated steps.")
    for k, v in out.items():
        if v.ndim == 2:
            print(f"  {k}: {v.shape[0]} rows")

    plot_room_and_refrigeration(out, plots_dir)
    plot_solver_health(out, plots_dir)
    plot_inventory_and_events(out, plots_dir)
    plot_batch_sensors(out, plots_dir)
    plot_quality_grades(out, plots_dir)
    plot_telemetry_sample(out, plots_dir)
    plot_coil_heat_flows(out, plots_dir)
    plot_frost_diagnostics(out, plots_dir)
    plot_frost_feedback(out, plots_dir)
    plot_airflow_proof_sensor(out, plots_dir)
    plot_frost_regions(out, plots_dir)
    plot_defrost_diagnostics(out, plots_dir)
    plot_lp_control(out, plots_dir)

    print(f"Plots written to {plots_dir}/")


if __name__ == "__main__":
    main()
