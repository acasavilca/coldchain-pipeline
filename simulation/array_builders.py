"""
Parameter input array builders for the Chao Cold Storage Analytics Pipeline simulator.

This module builds every static input array `execute_solver_loop` / `initialize_first_run_state`
/ `update_downtime_and_ramp_up_schedule` / logistics functions expect:

    room_configurations_vector           (float64, 1D)
    equipment_configurations_float64_vector      (float64, 1D)
    fruit_parameters_matrix              (float64, 2D: [crop_row, field])
    fruit_respiration_curves_matrix                  (float64, 3D: [curve_row, crop_row, T/R])
    base_quality_grade_probs_matrix             (float64, 3D: [crop_row, grade, dist/demanded])
    historical_yield_matrix              (float64, 3D: [crop_row, month, ...])  # monthly x yearly-trend
    employee_parameters_matrix           (float64, 2D: [operator_row, field])
    door_specs_matrix                    (float64, 2D: [door_row, field])
    refrigerant_properties_matrix        (float64, 3D: [refrig_row, scalar_field])
    refrigerant_surfaces_matrix          (float64, 4D-ish: [refrig_row, surface, coeffs/bounds, :])

IMPORTANT: `matrix_index_maps.py` as it currently exists has structural bugs (duplicate
indices, undefined placeholders, a stale employee-matrix column layout) that make it
unsafe to import directly for this job. See reconciliation_notes.md for the full list.
This file defines its own *corrected* index scheme (the `*_IDX` dicts / SimpleNamespaces
below) that supersedes the broken bits. Diff against your real matrix_index_maps.py and
merge deliberately -- do not just import this blind.

Every place where the JSON config is missing a field that the simulator code needs is
marked `# TODO-CONFIG` with a fallback value and a note. Search this file for
"TODO-CONFIG" before trusting the numbers in a real run.
"""

import json
from pathlib import Path
from typing import Any, Dict, Union, Iterable, Collection
import numpy as np
import os
import sys
import math
from constants import *
from matrix_index_maps import *

from fruit_biology import get_respiration_curve, get_fruit_respiration_parameters
from utils import evaluate_poly1d_fit, evaluate_poly2d_cubic_fit, evaluate_antoine_fit, least_squares_regression, least_squares_regression_through_origin
from psychrometrics import calculate_ua_coil_rated

# Add the tools directory to path
tools_path = Path(__file__).resolve().parent / "tools"
if str(tools_path) not in sys.path:
    sys.path.insert(0, str(tools_path))

try:
    import fit_refrigerant_surfaces
except ModuleNotFoundError as e:
    # If a dependency is missing, let the user know explicitly
    print(f"\n[ERROR] fit_refrigerant_surfaces.py is missing a dependency: {e}", file=sys.stderr)
    print("Please install it in your virtual environment.\n", file=sys.stderr)
    raise
except ImportError:
    fit_refrigerant_surfaces = None

try:
    import fit_fan_curves
except ModuleNotFoundError as e:
    print(f"\n[ERROR] fit_fan_curves.py is missing a dependency: {e}", file=sys.stderr)
    print("Please install it in your virtual environment.\n", file=sys.stderr)
    raise
except ImportError:
    fit_refrigerant_surfaces = None

def _load_json_source(src, label=None):
    if isinstance(src, (str, Path)):
        p = Path(src)
        if not p.exists():
            raise FileNotFoundError(f"{label} not found at: {p}")
        with p.open("r") as f:
            src = json.load(f)
    if not isinstance(src, dict):
        raise TypeError(f"{label} must resolve to a valid dictionary source.")
    return src


def resolve_active_equipment(facility_cfg, equipment_catalog):
    sel = facility_cfg["active_equipment_id_selection"]
    out = []
    for section, key in (("compressors", "compressor"), ("evaporators", "evaporator"), ("condensers", "condenser")):
        if section not in equipment_catalog:
            raise KeyError(f"Equipment catalog has no top-level '{section}' section "
                           f"(top-level keys: {list(equipment_catalog.keys())})")
        eq_id = str(sel[key])
        if eq_id not in equipment_catalog[section]:
            raise KeyError(f"'{section}' has no id '{eq_id}' (available: {list(equipment_catalog[section].keys())})")
        out.append(equipment_catalog[section][eq_id])
    return tuple(out)

def _rating_for_fluid(unit, fluid_name, unit_label):
    ratings = unit["ratings"]
    if fluid_name in ratings:
        return ratings[fluid_name]
    fallback_name = next(iter(ratings))
    print(f"[WARN] {unit_label} '{unit['metadata']['model']}' has no rating for {fluid_name}; "
          f"falling back to {fallback_name}.")
    return ratings[fallback_name]

def load_fan_fit(
    evaporator_id: int = 1,
    catalog_path: Union[str, Path] = "configs/catalog_equipment_config.json",
    output_dir: Union[str, Path] = "polynomial_surfaces",
):
    file_path = Path(output_dir) / f"fan_fit_evaporator_{evaporator_id}.json"
    if not file_path.exists():
        if fit_refrigerant_surfaces is None:
            raise RuntimeError(
                    f"{file_path} doesn't exist and fit_refrigerant_surfaces.py isn't "
                    f"importable -- put it on PYTHONPATH so missing fan fit can "
                    f"be auto-generated the way you built it to work."
            )
        sys.argv = [
            "fit_fan_curves.py",
            "--catalog-config-path", str(catalog_path),
            "--evaporator-id", str(evaporator_id),
            "--output-folder", str(output_dir)
        ]
        fit_fan_curves.main()
        if not file_path.exists():
            raise FileNotFoundError(f"Generator run failed to produce expected file: {file_path}")

    with file_path.open("r") as f:
        return json.load(f)

def calibrate_fan_clean_pressure(coeffs, bounds, q_rated_m3_s):
    """Pressure at which the fan curve delivers the catalog airflow (clean coil operating point)."""
    # Here, q is airflow, which in other places if referred as V_dot
    q_at_p_min = evaluate_poly1d_fit(bounds[0], coeffs, bounds)
    q_at_p_max = evaluate_poly1d_fit(bounds[1], coeffs, bounds)

    if not (q_at_p_max <= q_rated_m3_s <= q_at_p_min):
        raise ValueError(f"Catalog airflow {q_rated_m3_s} m3/s is outside the fan curve range "
                         f"[{q_at_p_max:.2f}, {q_at_p_min:.2f}] m3/s")
    p_lo, p_hi = bounds[0], bounds[1]
    for _ in range(100):
        p_mid = 0.5 * (p_lo + p_hi)
        if evaluate_poly1d_fit(p_mid, coeffs, bounds) > q_rated_m3_s:
            p_lo = p_mid
        else:
            p_hi = p_mid
    return 0.5 * (p_lo + p_hi)

# ==============================================================================
# ROOM CONFIGURATIONS
# ==============================================================================
def build_room_configurations_vectors(facility_cfg: Union[Dict[str, Any], str, Path] = "configs/facility_runtime_config.json"):
    facility_cfg = _load_json_source(facility_cfg, "Runtime configuration file")

    v_float64 = np.zeros(N_ROOM_FLOAT64_FIELDS, dtype=np.float64)
    v_int64 = np.zeros(N_ROOM_INT64_FIELDS, dtype=np.int64)
    geo = facility_cfg["room_structural_geometry"]
    mat = facility_cfg["building_insulation_layer_materials"]
    plant = facility_cfg["adjacent_plant_thermal_model"]
    logi = facility_cfg["general_logistics_and_cargo_properties"]
    sched = facility_cfg["macro_scheduling_environment"]
    runtime = facility_cfg["simulation_runtime_settings"]
    static_w = facility_cfg["static_weather_data"]

    dt_engine_sec = runtime["dt_engine_seconds"]

    length, width, height = geo["length_m"], geo["width_m"], geo["height_m"]

    v_float64[IDX_ROOM_GEOM_LENGTH] = length
    v_float64[IDX_ROOM_GEOM_WIDTH] = width
    v_float64[IDX_ROOM_GEOM_HEIGHT] = height
    v_float64[IDX_ROOM_GEOM_VOLUME] = length * width * height

    # --- Derived geometry / thermal mass ---
    floor_area = length * width
    roof_area = floor_area
    internal_walls_area = width * height
    external_walls_area = (2 * length * height) + (width * height)
    v_float64[IDX_ROOM_FLOOR_AREA_M2] = floor_area
    v_float64[IDX_ROOM_ROOF_AREA_M2] = roof_area
    v_float64[IDX_ROOM_INTERNAL_WALL_AREA_M2] = internal_walls_area
    v_float64[IDX_ROOM_EXTERNAL_WALL_AREA_M2] = external_walls_area 

    # Floor thermal mass: concrete slab, rho*cp*thickness*area
    C_floor = (
        mat["concrete_density_kg_m3"]
        * mat["concrete_specific_heat_j_kg_k"]
        * mat["concrete_slab_thickness_m"]
        * floor_area
    )
    v_float64[IDX_ROOM_FLOOR_THERMAL_MASS] = C_floor
    v_float64[IDX_ROOM_DT_OVER_FLOOR_THERMAL_MASS] = dt_engine_sec / max(C_floor, 1e-4)

    # Panel (wall+roof insulation) thermal mass, scaled by f_structure per your own
    # documented pattern of scaling internal-structure mass.
    panel_area = internal_walls_area + external_walls_area + roof_area
    C_panels = (
        mat["insulation_panel_density_kg_m3"]
        * mat["insulation_panel_specific_heat_j_kg_k"]
        * mat["insulation_panel_thickness_m"]
        * panel_area
    )
    v_float64[IDX_ROOM_PANELS_THERMAL_MASS] = C_panels

    U_wall = geo["U_wall_w_m2_k"]
    U_floor = geo["U_floor_w_m2_k"]
    U_ground = geo["U_ground_w_m2_k"]
    h_i_walls = geo["h_i_walls_w_m2_k"]
    h_i_roof = geo["h_i_roof_w_m2_k"]
    h_i_floor = geo["h_i_floor_w_m2_k"]

    inv_U_wall = 1.0 / max(U_wall, SAFE_FLOOR)
    inv_h_i_walls = 1.0 / max(h_i_walls, SAFE_FLOOR)
    inv_h_i_roof = 1.0 / max(h_i_roof, SAFE_FLOOR)

    v_float64[IDX_ROOM_THERMAL_U_WALL] = U_wall
    v_float64[IDX_ROOM_THERMAL_U_FLOOR] = U_floor
    v_float64[IDX_ROOM_THERMAL_U_GROUND] = U_ground

    v_float64[IDX_ROOM_THERMAL_ALPHA_ROOF] = geo["alpha_roof_solar_absorp"]
    v_float64[IDX_ROOM_THERMAL_H_I_WALLS] = h_i_walls
    v_float64[IDX_ROOM_THERMAL_H_I_ROOF] = h_i_roof
    v_float64[IDX_ROOM_THERMAL_H_I_FLOOR] = h_i_floor

    v_float64[IDX_ROOM_THERMAL_EXT_WALL_ENV_CONSTANT] = inv_U_wall + inv_h_i_walls
    v_float64[IDX_ROOM_THERMAL_ROOF_ENV_CONSTANT] = inv_U_wall + inv_h_i_roof
    v_float64[IDX_ROOM_THERMAL_UA_WALLS_INT] = 1.0 / (2.0 * inv_h_i_walls + inv_U_wall) * internal_walls_area

    v_float64[IDX_ROOM_THERMAL_UA_ROOM_TO_FLOOR] = h_i_floor * floor_area
    v_float64[IDX_ROOM_THERMAL_UA_FLOOR_TO_GROUND] = U_ground * floor_area

    v_float64[IDX_ROOM_H_FUSION_OVER_DT_SEC] = H_FUSION / max(v_float64[IDX_ROOM_PLANT_DT_STEP_SEC], SAFE_FLOOR)

    v_float64[IDX_ROOM_NOMINAL_ACH_PER_SEC] = geo["n_ach_nominal_per_day"] / 86400.0

    v_float64[IDX_ROOM_PLANT_DT_STEP_SEC] = plant["dt_update_sec"]
    v_int64[IDX_ROOM_PLANT_DT_STEP_MS] = int(plant["dt_update_sec"] * 1e3)

    v_float64[IDX_ROOM_PLANT_TEMP_COEFF_A] = plant["T_plant_coeff_a"]
    v_float64[IDX_ROOM_PLANT_TEMP_COEFF_B] = plant["T_plant_coeff_b"]
    v_float64[IDX_ROOM_PLANT_TAU_THERMAL]  = plant["tau_plant_thermal_lag_sec"]
    v_float64[IDX_ROOM_PLANT_DT_OVER_TAU_THERMAL] = v_float64[IDX_ROOM_PLANT_DT_STEP_SEC] / max(v_float64[IDX_ROOM_PLANT_TAU_THERMAL], SAFE_FLOOR)

    v_float64[IDX_ROOM_PLANT_RH_COEFF_A] = plant["RH_plant_coeff_a"]
    v_float64[IDX_ROOM_PLANT_RH_COEFF_B] = plant["RH_plant_coeff_b"]
    v_float64[IDX_ROOM_PLANT_TAU_VAPOR]  = plant["tau_plant_vapor_lag_sec"]
    v_float64[IDX_ROOM_PLANT_DT_OVER_TAU_VAPOR] = v_float64[IDX_ROOM_PLANT_DT_STEP_SEC] / max(v_float64[IDX_ROOM_PLANT_TAU_VAPOR], SAFE_FLOOR)

    v_float64[IDX_ROOM_OUTDOORS_CO2_PPM] = static_w["co2_outdoor_ppm"]
    v_float64[IDX_ROOM_OUTDOORS_O2_PCT] = static_w["o2_outdoor_pct"]

    v_float64[IDX_ROOM_LOGISTICS_TD_COEFF] = logi["td_coeff_cross_flow_penalty"]
    v_int64[IDX_ROOM_LOGISTICS_PLANNER_INTERVAL_MS] = int(logi["logistics_planner_evaluation_interval_sec"] * 1e3)
    v_float64[IDX_ROOM_LOGISTICS_PLANNER_INTERVAL_HR] = logi["logistics_planner_evaluation_interval_sec"] * INV_3600

    v_int64[IDX_ROOM_LOGISTICS_SPOIL_CHECK_INTERVAL_MS] = int(logi["spoilage_check_interval_sec"] * 1e3)

    v_float64[IDX_ROOM_LOGISTICS_P_BASE_SPOILAGE_OVERRIDE] = logi["baseline_probability_spoilage_override"]
    v_float64[IDX_ROOM_LOGISTICS_P_BASE_GATE_OVERRIDE] = logi["baseline_probability_gate_override"]
    v_float64[IDX_ROOM_LOGISTICS_P_BASE_FORGET_DOOR] = logi["baseline_probability_forget_door"]
    v_float64[IDX_ROOM_LOGISTICS_P_BASE_FORGET_DOOR_REMEDY] = logi["baseline_probability_forget_door_remedy"]
    v_float64[IDX_ROOM_LOGISTICS_P_BASE_SUPERVISOR_FORGET_DOOR] = logi["baseline_probability_supervisor_forget_door"]

    v_float64[IDX_ROOM_LOGISTICS_MACRO_ARR_DELAY_MS] = logi["macro_arrival_order_buffer_delay_mean_min"] * 60e3
    v_float64[IDX_ROOM_LOGISTICS_MACRO_DISP_DELAY_MS] = logi["macro_dispatch_order_buffer_delay_mean_min"] * 60e3
    v_float64[IDX_ROOM_LOGISTICS_MACRO_DUMP_DELAY_MS] = logi["macro_emergency_dump_buffer_delay_mean_min"] * 60e3
    v_float64[IDX_ROOM_LOGISTICS_MACRO_DELAY_SHAPE] = logi["macro_order_buffer_delay_distribution_shape"]

    v_float64[IDX_ROOM_LOGISTICS_STAGING_DELAY_MEAN_MS] = logi["staging_delay_mean_sec"] * 1e3
    v_float64[IDX_ROOM_LOGISTICS_STAGING_DELAY_SHAPE] = logi["staging_delay_shape"]
    v_float64[IDX_ROOM_LOGISTICS_OUTBOUND_LEG_MEAN_MS] = logi["outbound_leg_mean_sec"] * 1e3
    v_float64[IDX_ROOM_LOGISTICS_OUTBOUND_LEG_SHAPE] = logi["outbound_leg_shape"]
    v_float64[IDX_ROOM_LOGISTICS_RETURN_LEG_MEAN_MS] = logi["return_leg_mean_sec"] * 1e3
    v_float64[IDX_ROOM_LOGISTICS_RETURN_LEG_SHAPE] = logi["return_leg_shape"]

    v_int64[IDX_ROOM_LOGISTICS_BASELINE_YEAR] = int(sched["simulation_baseline_year"])

    v_int64[IDX_ROOM_LOGISTICS_DRIVER_Q_SAMPLING_INTERVAL_MS] = int(logi["driver_convective_q_resample_interval_sec"] * 1e3)

    v_int64[IDX_ROOM_LOGISTICS_CROP_ROTATION_DOWNTIME_DURATION_MS]   = int(logi["crop_rotation_downtime_duration_hrs"] * MS_PER_HOUR)
    v_int64[IDX_ROOM_LOGISTICS_REFRIG_ROTATION_DOWNTIME_DURATION_MS] = int(logi["refrigerant_rotation_downtime_duration_hrs"] * MS_PER_HOUR)
    v_int64[IDX_ROOM_LOGISTICS_MINIMUM_OPERATION_PERIOD_VIABLE_MS]   = int(logi["minimum_operation_period_viable_hrs"] * MS_PER_HOUR)

    v_float64[IDX_ROOM_CFG_DT_ENGINE_SEC] = runtime["dt_engine_seconds"]
    v_float64[IDX_ROOM_CFG_INV_DT_ENGINE_SEC] = 1.0 / max(v_float64[IDX_ROOM_CFG_DT_ENGINE_SEC], SAFE_FLOOR)
    v_int64[IDX_ROOM_CFG_DT_ENGINE_MS]  = int(runtime["dt_engine_seconds"] * 1e3)
    v_int64[IDX_ROOM_CFG_MAX_NUMERICAL_SOLVER_ITERS] = int(runtime["max_numerical_solver_iterations"])
    v_float64[IDX_ROOM_CFG_NUMERICAL_SOLVER_TOLERANCE] = runtime["numerical_solver_tolerance"]

    v_int64[IDX_ROOM_CFG_FORCING_RESAMPLE_INTERVAL_MS] = int(runtime["forcing_arrays_resample_interval_sec"] * 1e3)

    v_int64[IDX_ROOM_CFG_LEAK_NOISE_FACTOR_RESAMPLE_INTERVAL_MS] = int(runtime["leak_noise_factor_resample_interval_sec"] * 1e3)

    return v_float64, v_int64,


# ==============================================================================
# EQUIPMENT CONFIGURATIONS
# ==============================================================================
def build_evaporator_fan_curve_matrix(
    facility_cfg: Union[Dict[str, Any], str, Path] = "configs/facility_runtime_config.json",
    equipment_catalog_path: Union[str, Path] = "configs/catalog_equipment_config.json",
    fits_output_dir: Union[str, Path] = "polynomial_surfaces",
):
    facility_cfg = _load_json_source(facility_cfg, "Runtime configuration file")

    evaporator_id = facility_cfg["active_equipment_id_selection"]["evaporator"]
    fit = load_fan_fit(evaporator_id, equipment_catalog_path, fits_output_dir)["fit"]

    c = np.asarray(fit["coefficients"], dtype=np.float64)
    b = np.asarray(fit["bounds"], dtype=np.float64)
    m = np.zeros((2, MAX_N_COEFFS), dtype=np.float64)
    m[COEFFS, :len(c)] = c
    m[BOUNDS, :len(b)] = b
    return m

def build_equipment_configurations_vectors(
    fan_curve_matrix,
    facility_cfg: Union[Dict[str, Any], str, Path] = "configs/facility_runtime_config.json",
    equipment_catalog: Union[Dict[str, Any], str, Path] = "configs/catalog_equipment_config.json",
    location_id: Union[str, int] = None,
):
    if location_id is None:
        raise ValueError("build_equipment_configurations_vectors: location_id is required")
    facility_cfg = _load_json_source(facility_cfg, "Runtime configuration file")
    equipment_catalog = _load_json_source(equipment_catalog, "Equipment catalog")
    comp, evap, cond = resolve_active_equipment(facility_cfg, equipment_catalog)

    v_float64 = np.zeros(N_EQ_FLOAT64_FIELDS, dtype=np.float64)
    v_int64 = np.zeros(N_EQ_INT64_FIELDS, dtype=np.int64)

    location_id = str(location_id)
    loc = facility_cfg["facility_location"]["locations"][location_id]
    hw = facility_cfg["refrigeration_and_humidifier_hardware"]
    sens = facility_cfg["digital_sensor_monitoring_profiles"]
    aux = facility_cfg["auxiliary_environmental_hardware"]
    runtime = facility_cfg["simulation_runtime_settings"]
    dt_engine_sec = runtime["dt_engine_seconds"]
    cp_metal = hw["heat_exchanger_metal_specific_heat_j_kg_k"]

    # --- Compressor (equipment-only) ---
    v_float64[IDX_EQ_COMPRESSOR_V_DISPLACEMENT_M3] = comp["displacement_cm3_rev"] * 1e-6
    v_float64[IDX_EQ_COMPRESSOR_SPEED_HZ] = comp["nominal_speed_rpm"] / 60.0
    v_float64[IDX_EQ_COMPRESSOR_IDLING_FRICTION] = comp["idling_friction_factor"]
    v_float64[IDX_EQ_COMPRESSOR_TAU_COOL_SEC] = comp["cooling_inertia_time_constant_sec"]
    v_float64[IDX_EQ_DT_OVER_COMPRESSOR_TAU_COOL_SEC] = dt_engine_sec / max(v_float64[IDX_EQ_COMPRESSOR_TAU_COOL_SEC], SAFE_FLOOR)

    # --- Evaporator (equipment-only) ---
    v_float64[IDX_EQ_EVAP_MASS_METAL_KG] = evap["metal_mass_kg"]
    v_float64[IDX_EQ_CP_HX_METAL_J_KG_K] = cp_metal
    v_float64[IDX_EQ_EVAP_THERMAL_MASS_METAL] = evap["metal_mass_kg"] * cp_metal
    v_float64[IDX_EQ_DT_OVER_EVAP_THERMAL_MASS_METAL] = dt_engine_sec / max(v_float64[IDX_EQ_EVAP_THERMAL_MASS_METAL], 1e-4)

    v_float64[IDX_EQ_EVAP_NOMINAL_AIRFLOW_M3_S] = evap["rated_airflow_m3_s"]
    v_float64[IDX_EQ_INV_EVAP_NOMINAL_AIRFLOW_M3_S] = 1.0 / max(v_float64[IDX_EQ_EVAP_NOMINAL_AIRFLOW_M3_S], SAFE_FLOOR)
    v_float64[IDX_EQ_EVAP_FLOOR_AIRFLOW_M3_S] = v_float64[IDX_EQ_EVAP_NOMINAL_AIRFLOW_M3_S] * 0.005
    v_float64[IDX_EQ_EVAP_PRESSURE_DROP_PA] = evap["coil_nominal_pressure_drop_pa"]   # kept for frost blockage (Phase 3)
    v_float64[IDX_EQ_EVAP_FAN_POWER_RATED_W] = evap["number_of_fans"] * evap["fan_power_each_w"]
    v_float64[IDX_EQ_EVAP_FAN_TAU_LAG_SEC] = evap["fan_inertia_time_constant_sec"]
    v_float64[IDX_EQ_DT_OVER_EVAP_FAN_TAU_LAG_SEC] = dt_engine_sec / max(v_float64[IDX_EQ_EVAP_FAN_TAU_LAG_SEC], SAFE_FLOOR)
    v_float64[IDX_EQ_VELOCITY_EXPONENT_UA_COIL_EVAP_AIR_SIDE] = evap["air_side_ua_velocity_exponent"]

    v_float64[IDX_EQ_EVAP_SURFACE_AREA_M2] = evap["external_surface_area_m2"]
    v_float64[IDX_EQ_EVAP_INV_SURFACE_AREA_M2] = 1.0 / v_float64[IDX_EQ_EVAP_SURFACE_AREA_M2]
    v_float64[IDX_EQ_EVAP_FRONTAL_AREA_M2] = evap["rated_airflow_m3_s"] / evap["rated_face_velocity_m_s"]
    v_float64[IDX_EQ_EVAP_INV_FRONTAL_AREA_M2] = 1.0 / v_float64[IDX_EQ_EVAP_FRONTAL_AREA_M2]
    v_float64[IDX_EQ_EVAP_FIN_PITCH_M] = evap["fin_spacing_mm"] * 1e-3
    v_float64[IDX_EQ_EVAP_FIN_THICKNESS_M] = evap["fin_thickness_mm"] * 1e-3
    v_float64[IDX_EQ_EVAP_FIN_EDGE_CONTACT_FACTOR] = v_float64[IDX_EQ_EVAP_FIN_THICKNESS_M] / v_float64[IDX_EQ_EVAP_FIN_PITCH_M]
    v_float64[IDX_EQ_EVAP_FIN_CLEAR_GAP_M] = v_float64[IDX_EQ_EVAP_FIN_PITCH_M] - v_float64[IDX_EQ_EVAP_FIN_THICKNESS_M]
    v_float64[IDX_EQ_EVAP_TUBE_CLEAR_GAP_M] = evap["tube_clear_gap_mm"] * 1e-3
    v_float64[IDX_EQ_EVAP_INV_FIN_CLEAR_GAP_M] = 1.0 / v_float64[IDX_EQ_EVAP_FIN_CLEAR_GAP_M]
    v_float64[IDX_EQ_EVAP_INV_TUBE_CLEAR_GAP_M] = 1.0 / v_float64[IDX_EQ_EVAP_TUBE_CLEAR_GAP_M]
    face_bridge_length_m = evap["face_bridge_length_mm"] * 1e-3
    v_float64[IDX_EQ_EVAP_INV_FACE_BRIDGE_LENGTH_M] = 1.0 / face_bridge_length_m
    v_float64[IDX_EQ_EVAP_FACE_CONTACT_FRACTION] = evap["face_contact_fraction"]
    v_float64[IDX_EQ_EVAP_FACE_H_W_M2_K] = evap["exposed_face_h_w_m2_k"]
    v_float64[IDX_EQ_EVAP_INV_CP_AIR_LEWIS_2_3] = 1.0 / (
        CP_AIR * evap["air_water_vapor_lewis_number"] ** (2.0 / 3.0)
    )

    airflow_proof = hw["evaporator_airflow_proof"]
    v_float64[IDX_EQ_EVAP_AIRFLOW_PROOF_TRIP_M_S] = airflow_proof["trip_velocity_m_s"]
    v_float64[IDX_EQ_EVAP_AIRFLOW_PROOF_RECOVERY_M_S] = (
        airflow_proof["trip_velocity_m_s"] * airflow_proof["recovery_multiplier"]
    )
    v_int64[IDX_EQ_EVAP_AIRFLOW_PROOF_DELAY_MS] = int(
        airflow_proof["condition_delay_sec"] * 1000.0
    )

    v_float64[IDX_EQ_DEFROST_HEATER_W] = evap["defrost_heater_w"]
    pan_heater_w = evap["defrost_drain_pan_heater_w"]
    if not 0.0 <= pan_heater_w <= evap["defrost_heater_w"]:
        raise ValueError("defrost_drain_pan_heater_w must be within total defrost heater rating")
    v_float64[IDX_EQ_DEFROST_PAN_HEATER_W] = pan_heater_w
    pan_air_conductance_w_k = evap["drain_pan_air_conductance_w_k"]
    if pan_air_conductance_w_k < 0.0:
        raise ValueError("drain_pan_air_conductance_w_k must be nonnegative")
    v_float64[IDX_EQ_DEFROST_PAN_AIR_CONDUCTANCE_W_K] = pan_air_conductance_w_k
    release_film_m = evap["face_release_melt_film_mm"] * 1e-3
    if release_film_m <= 0.0:
        raise ValueError("face_release_melt_film_mm must be positive")
    v_float64[IDX_EQ_FACE_RELEASE_BOND_MASS_PER_M2] = (
        RHO_ICE_KG_M3
        * v_float64[IDX_EQ_EVAP_FIN_EDGE_CONTACT_FACTOR]
        * release_film_m
    )

    v_float64[IDX_EQ_EVAP_FAN_CLEAN_PRESSURE_PA] = calibrate_fan_clean_pressure(
        fan_curve_matrix[COEFFS, :], fan_curve_matrix[BOUNDS, :], evap["rated_airflow_m3_s"])
    v_float64[IDX_EQ_EVAP_INV_V_DOT_CLEAN] = 1.0 / v_float64[IDX_EQ_EVAP_NOMINAL_AIRFLOW_M3_S]
    v_float64[IDX_EQ_EVAP_PRESSURE_OVER_SQ_V_DOT_CLEAN] = (
        v_float64[IDX_EQ_EVAP_FAN_CLEAN_PRESSURE_PA]
        * v_float64[IDX_EQ_EVAP_INV_V_DOT_CLEAN] ** 2
    )
    print(f"[fan] clean operating pressure {v_float64[IDX_EQ_EVAP_FAN_CLEAN_PRESSURE_PA]:.1f} Pa")

    v_float64[IDX_EQ_EVAP_TEMP_RELAX_ALPHA] = 1 - math.exp(-dt_engine_sec / evap["tau_off_sec"])
    v_float64[IDX_EQ_EVAP_COIL_G_FACE_W_K] = evap["effective_residual_conductance_w_k"]

    # --- Condenser (equipment-only) ---
    v_float64[IDX_EQ_COND_MASS_METAL_KG] = cond["metal_mass_kg"]
    v_float64[IDX_EQ_COND_THERMAL_MASS_METAL] = cond["metal_mass_kg"] * cp_metal
    v_float64[IDX_EQ_DT_OVER_COND_THERMAL_MASS_METAL] = dt_engine_sec / max(v_float64[IDX_EQ_COND_THERMAL_MASS_METAL], 1e-4)

    v_float64[IDX_EQ_COND_NOMINAL_AIRFLOW_M3_S] = cond["rated_airflow_m3_h"] * INV_3600
    v_float64[IDX_EQ_COND_FLOOR_AIRFLOW_M3_S] = v_float64[IDX_EQ_COND_NOMINAL_AIRFLOW_M3_S] * 0.015
    v_float64[IDX_EQ_COND_FAN_POWER_RATED_W] = cond["number_of_fans"] * cond["fan_power_each_w"]
    v_float64[IDX_EQ_COND_FAN_TAU_LAG_SEC] = cond["fan_inertia_time_constant_sec"]
    v_float64[IDX_EQ_DT_OVER_COND_FAN_TAU_LAG_SEC] = dt_engine_sec / max(v_float64[IDX_EQ_COND_FAN_TAU_LAG_SEC], SAFE_FLOOR)
    v_float64[IDX_EQ_VELOCITY_EXPONENT_UA_COIL_COND_AIR_SIDE] = cond["air_side_ua_velocity_exponent"]
    v_float64[IDX_EQ_COND_TEMP_RELAX_ALPHA] = 1 - math.exp(-dt_engine_sec / cond["tau_off_sec"])

    # --- Design references (facility) ---
    v_float64[IDX_EQ_DESIGN_EVAP_TD_K] = hw["evaporator_design_temperature_difference_k"]
    v_float64[IDX_EQ_DESIGN_COND_TD_K] = hw["condenser_design_temperature_difference_k"]
    v_float64[IDX_EQ_DESIGN_AMBIENT_TEMP_C] = loc["ambient_design_temperature_c"]
    v_float64[IDX_EQ_DESIGN_RHO_AIR_COND] = P_ATM_STANDARD / (R_DRY_AIR * (loc["ambient_design_temperature_c"] + 273.15))

    # --- Controls (facility) ---
    v_float64[IDX_EQ_REFRIG_CONTROL_DEADBAND_C] = hw["compressor_deadband_differential_c"]
    v_int64[IDX_EQ_REFRIG_CONTROL_EVALUATION_INTERVAL_MS] = int(hw["compressor_evaluation_interval_sec"] * 1e3)
    v_int64[IDX_EQ_REFRIG_CONTROL_MIN_IDLE_MS] = int(hw["compressor_minimum_idle_time_sec"] * 1e3)
    v_int64[IDX_EQ_REFRIG_CONTROL_MIN_RUN_MS] = int(hw["compressor_minimum_active_time_sec"] * 1e3)

    # --- Electric defrost controls (facility settings; heater rating is catalog equipment) ---
    defrost = hw["electric_defrost_control"]
    if defrost["interval_sec"] <= 0.0:
        raise ValueError("electric_defrost_control.interval_sec must be positive")
    if defrost["minimum_heat_sec"] < 0.0 or defrost["maximum_heat_sec"] <= defrost["minimum_heat_sec"]:
        raise ValueError("electric_defrost_control heat durations must satisfy 0 <= minimum < maximum")
    for key in ("settle_sec", "drip_sec", "fan_delay_max_sec", "airflow_proof_grace_sec", "termination_probe_hold_sec"):
        if defrost[key] < 0.0:
            raise ValueError(f"electric_defrost_control.{key} must be nonnegative")

    v_int64[IDX_EQ_DEFROST_ENABLED] = int(defrost["enabled"])
    v_int64[IDX_EQ_DEFROST_INTERVAL_MS] = int(defrost["interval_sec"] * 1e3)
    v_int64[IDX_EQ_DEFROST_SETTLE_MS] = int(defrost["settle_sec"] * 1e3)
    v_int64[IDX_EQ_DEFROST_MIN_HEAT_MS] = int(defrost["minimum_heat_sec"] * 1e3)
    v_int64[IDX_EQ_DEFROST_MAX_HEAT_MS] = int(defrost["maximum_heat_sec"] * 1e3)
    v_int64[IDX_EQ_DEFROST_DRIP_MS] = int(defrost["drip_sec"] * 1e3)
    v_int64[IDX_EQ_DEFROST_FAN_DELAY_MAX_MS] = int(defrost["fan_delay_max_sec"] * 1e3)
    v_int64[IDX_EQ_DEFROST_AIRFLOW_PROOF_GRACE_MS] = int(defrost["airflow_proof_grace_sec"] * 1e3)
    v_int64[IDX_EQ_DEFROST_TERMINATION_HOLD_MS] = int(defrost["termination_probe_hold_sec"] * 1e3)
    v_float64[IDX_EQ_DEFROST_TERMINATION_COIL_T_C] = defrost["termination_coil_temp_c"]
    v_float64[IDX_EQ_DEFROST_FAN_RESTART_COIL_T_C] = defrost["fan_restart_coil_temp_c"]

    # --- Humidifier (facility) ---
    v_float64[IDX_EQ_HUMIDIFIER_M_DOT_MAX_KG_S] = hw["humidifier_max_mass_flow_per_disc_L_hr"] * INV_3600
    v_float64[IDX_EQ_HUMIDIFIER_DISCS_PER_CLUSTER] = hw["humidifier_discs_per_cluster_count"]
    v_float64[IDX_EQ_HUMIDIFIER_INV_DISCS_PER_CLUSTER] = 1.0 / max(v_float64[IDX_EQ_HUMIDIFIER_DISCS_PER_CLUSTER], SAFE_FLOOR)
    v_float64[IDX_EQ_HUMIDIFIER_MAX_TOTAL_DISCS] = int(hw["humidifier_max_total_discs_ceiling"])
    v_float64[IDX_EQ_HUMIDIFIER_FAN_TAU_LAG_SEC] = hw["humidifier_fan_inertia_time_constant_sec"]
    v_float64[IDX_EQ_DT_OVER_HUMIDIFIER_FAN_TAU_LAG_SEC] = dt_engine_sec / max(v_float64[IDX_EQ_HUMIDIFIER_FAN_TAU_LAG_SEC], SAFE_FLOOR)

    v_float64[IDX_EQ_HUMIDIFIER_PID_INTERVAL_SEC] = aux["relative_humidity_pid_evaluation_interval_sec"]
    v_int64[IDX_EQ_HUMIDIFIER_PID_INTERVAL_MS] = int(v_float64[IDX_EQ_HUMIDIFIER_PID_INTERVAL_SEC] * 1e3)
    v_float64[IDX_EQ_HUMIDIFIER_PID_K_P] = aux["relative_humidity_pid_K_p"]
    v_float64[IDX_EQ_HUMIDIFIER_PID_K_I] = aux["relative_humidity_pid_K_i"]
    v_float64[IDX_EQ_HUMIDIFIER_PID_K_D] = aux["relative_humidity_pid_K_d"]
    v_float64[IDX_EQ_HUMIDIFIER_PID_MIN_OUTPUT] = aux["relative_humidity_pid_min_output"]
    v_float64[IDX_EQ_HUMIDIFIER_PID_MAX_OUTPUT] = aux["relative_humidity_pid_max_output"]

    # --- EEV / main valve (facility) ---
    v_int64[IDX_EQ_EEV_PID_INTERVAL_MS] = int(aux["superheat_pid_evaluation_interval_sec"] * 1e3)
    v_float64[IDX_EQ_EEV_PID_INTERVAL_SEC] = aux["superheat_pid_evaluation_interval_sec"]
    v_float64[IDX_EQ_EEV_PID_K_P] = aux["superheat_pid_K_p"]
    v_float64[IDX_EQ_EEV_PID_K_I] = aux["superheat_pid_K_i"]
    v_float64[IDX_EQ_EEV_PID_K_D] = aux["superheat_pid_K_d"]
    v_float64[IDX_EQ_EEV_PID_MAX_OUTPUT] = aux["superheat_pid_max_output"]
    v_float64[IDX_EQ_EEV_INV_PID_MAX_OUTPUT] = 1.0 / max(aux["superheat_pid_max_output"], SAFE_FLOOR)
    v_float64[IDX_EQ_EEV_PID_MIN_OUTPUT] = aux["superheat_pid_min_output"]
    v_float64[IDX_EQ_EEV_CLOSED_VALVE_SH_K] = aux["superheat_closed_valve_k"]
    v_float64[IDX_EQ_EEV_SUPERHEAT_K] = aux["superheat_target_setpoint_k"]

    v_float64[IDX_EQ_MAIN_VALVE_PROP_BAND_K] = aux["expansion_valve_proportional_band_k"]
    v_float64[IDX_EQ_INV_MAIN_VALVE_PROP_BAND_K] = 1.0 / max(v_float64[IDX_EQ_MAIN_VALVE_PROP_BAND_K], SAFE_FLOOR)
    v_float64[IDX_EQ_MAIN_VALVE_SETPOINT_C] = aux["expansion_valve_temperature_setpoint_c"]
    v_float64[IDX_EQ_MAIN_VALVE_SUBCOOL_APPROACH_K] = aux["expansion_valve_subcooling_approach_k"]

    _fill_sensor_channel(v_float64, v_int64, sens, "SENSOR_T",
        "temperature_sensor_sampling_interval_sec", "temperature_sensor_filter_window_sec",
        "temperature_sensor_noise_scale_c", "temperature_sensor_lower_limit_c", "temperature_sensor_upper_limit_c")

    _fill_sensor_channel(v_float64, v_int64, sens, "SENSOR_T_COIL_EVAP",
        "coil_temp_sensor_sampling_interval_sec", "coil_temp_sensor_filter_window_sec",
        "coil_temp_sensor_noise_scale_c", "coil_temp_sensor_lower_limit_c", "coil_temp_sensor_upper_limit_c")

    _fill_sensor_channel(v_float64, v_int64, sens, "SENSOR_EVAP_AIR_VELOCITY",
        "evap_air_velocity_sensor_sampling_interval_sec", "evap_air_velocity_sensor_filter_window_sec",
        "evap_air_velocity_sensor_noise_scale_m_s", "evap_air_velocity_sensor_lower_limit_m_s",
        "evap_air_velocity_sensor_upper_limit_m_s")

    _fill_sensor_channel(v_float64, v_int64, sens, "SENSOR_RH",
        "relative_humidity_sensor_sampling_interval_sec", "relative_humidity_sensor_filter_window_sec",
        "relative_humidity_sensor_noise_scale_frac", "relative_humidity_sensor_lower_limit_frac", "relative_humidity_sensor_upper_limit_frac")

    _fill_sensor_channel(v_float64, v_int64, sens, "SENSOR_T_SUCTION_LINE",
        "suction_line_temp_sensor_sampling_interval_sec", "suction_line_temp_sensor_filter_window_sec",
        "suction_line_temp_sensor_noise_scale_c", "suction_line_temp_sensor_lower_limit", "suction_line_temp_sensor_upper_limit")
    _fill_sensor_channel(v_float64, v_int64, sens, "SENSOR_P_SUCTION_LINE",
        "suction_line_pressure_sensor_sampling_interval_sec", "suction_line_pressure_sensor_filter_window_sec",
        "suction_line_pressure_sensor_noise_scale_bar", "suction_line_pressure_sensor_lower_limit", "suction_line_pressure_sensor_upper_limit")
    _fill_sensor_channel(v_float64, v_int64, sens, "SENSOR_T_LIQUID_LINE",
        "liquid_line_temp_sensor_sampling_interval_sec", "liquid_line_temp_sensor_filter_window_sec",
        "liquid_line_temp_sensor_noise_scale_c", "liquid_line_temp_sensor_lower_limit", "liquid_line_temp_sensor_upper_limit")
    _fill_sensor_channel(v_float64, v_int64, sens, "SENSOR_P_LIQUID_LINE",
        "liquid_line_pressure_sensor_sampling_interval_sec", "liquid_line_pressure_sensor_filter_window_sec",
        "liquid_line_pressure_sensor_noise_scale_bar", "liquid_line_pressure_sensor_lower_limit", "liquid_line_pressure_sensor_upper_limit")

    _fill_sensor_channel(v_float64, v_int64, sens, "SENSOR_COMPRESSOR_POWER",
        "compressor_power_sensor_sampling_interval_sec", "compressor_power_sensor_filter_window_sec",
        "compressor_power_sensor_noise_scale_w", "compressor_power_sensor_upper_limit_w")

    _fill_sensor_channel(v_float64, v_int64, sens, "SENSOR_T_ROOM_AIR_RETURN",
        "room_air_return_temp_sensor_sampling_interval_sec", "room_air_return_temp_sensor_filter_window_sec",
        "room_air_return_temp_sensor_noise_scale_c", "room_air_return_temp_sensor_lower_limit", "room_air_return_temp_sensor_upper_limit")
    _fill_sensor_channel(v_float64, v_int64, sens, "SENSOR_T_ROOM_AIR_SUPPLY",
        "room_air_supply_temp_sensor_sampling_interval_sec", "room_air_supply_temp_sensor_filter_window_sec",
        "room_air_supply_temp_sensor_noise_scale_c", "room_air_supply_temp_sensor_lower_limit", "room_air_supply_temp_sensor_upper_limit")
    _fill_sensor_channel(v_float64, v_int64, sens, "SENSOR_RH_ROOM_AIR_SUPPLY",
        "room_air_supply_rh_sensor_sampling_interval_sec", "room_air_supply_rh_sensor_filter_window_sec",
        "room_air_supply_rh_sensor_noise_scale_frac", "room_air_supply_rh_sensor_lower_limit_frac", "room_air_supply_rh_sensor_upper_limit_frac")
    _fill_sensor_channel(v_float64, v_int64, sens, "SENSOR_CO2_PPM",
        "co2_sensor_sampling_interval_sec", "co2_sensor_filter_window_sec",
        "co2_sensor_noise_scale_ppm", "co2_sensor_lower_limit", "co2_sensor_upper_limit")
    _fill_sensor_channel(v_float64, v_int64, sens, "SENSOR_O2_PCT",
        "o2_sensor_sampling_interval_sec", "o2_sensor_filter_window_sec",
        "o2_sensor_noise_scale_pct", "o2_sensor_lower_limit", "o2_sensor_upper_limit")

    _fill_sensor_channel(v_float64, v_int64, sens, "SENSOR_T_PULP",
        "pulp_temp_sensor_sampling_interval_sec", "pulp_temp_sensor_filter_window_sec",
        "pulp_temp_sensor_noise_scale_c", "pulp_temp_sensor_lower_limit", "pulp_temp_sensor_upper_limit")
    
    _fill_sensor_channel(v_float64, v_int64, sens, "SENSOR_LOAD_CELL",
        "load_cell_sensor_sampling_interval_sec", "load_cell_sensor_filter_window_sec",
        "load_cell_sensor_noise_scale_kg", "load_cell_sensor_lower_limit", "load_cell_sensor_upper_limit")

    v_int64[IDX_EQ_SENSOR_POLL_GCD_MS] = compute_sensor_poll_gcd_ms(sens, int(dt_engine_sec * 1e3))

    return v_float64, v_int64

# Sampling-interval keys of every channel written to room_telemetry_matrix. Pulp temperature and
# load cell are excluded on purpose: they go through the batch sensor ledger, not the telemetry matrix.
TELEMETRY_SENSOR_INTERVAL_KEYS = (
    "temperature_sensor_sampling_interval_sec",
    "coil_temp_sensor_sampling_interval_sec",
    "evap_air_velocity_sensor_sampling_interval_sec",
    "relative_humidity_sensor_sampling_interval_sec",
    "suction_line_temp_sensor_sampling_interval_sec",
    "suction_line_pressure_sensor_sampling_interval_sec",
    "liquid_line_temp_sensor_sampling_interval_sec",
    "liquid_line_pressure_sensor_sampling_interval_sec",
    "compressor_power_sensor_sampling_interval_sec",
    "room_air_return_temp_sensor_sampling_interval_sec",
    "room_air_supply_temp_sensor_sampling_interval_sec",
    "room_air_supply_rh_sensor_sampling_interval_sec",
    "co2_sensor_sampling_interval_sec",
    "o2_sensor_sampling_interval_sec",
)

def compute_sensor_poll_gcd_ms(sens: Dict[str, Any], dt_engine_ms: int) -> int:
    """GCD (ms) of all telemetry sampling intervals. Must be a positive multiple of the engine step."""
    gcd_ms = 0
    for key in TELEMETRY_SENSOR_INTERVAL_KEYS:
        gcd_ms = math.gcd(gcd_ms, int(sens[key] * 1e3))  # same ms conversion as _fill_sensor_channel
    if gcd_ms <= 0 or gcd_ms % dt_engine_ms != 0:
        raise ValueError(
            f"Telemetry sensor sampling intervals have GCD {gcd_ms} ms, which is not a positive "
            f"multiple of dt_engine ({dt_engine_ms} ms). Every sampling interval must be a multiple of the engine step."
        )
    return gcd_ms

def _alpha_from_window(interval_sec: float, window_sec: float) -> float:
    """EWMA smoothing coefficient from a sampling interval and an equivalent filter window."""
    if window_sec <= 0:
        return 1.0
    return 1.0 - np.exp(-interval_sec / window_sec)

def _fill_sensor_channel(v_float64, v_int64, sens, prefix, interval_key, window_key, noise_key, lower_key, upper_key=None):
    """Direct-key sensor channel fill. Every key must exist in `sens` -- KeyError if not."""
    interval = sens[interval_key]
    window = sens[window_key]
    base = f"IDX_EQ_{prefix}"
    v_int64[globals()[f"{base}_INTERVAL_MS"]] = int(interval * 1e3)
    v_float64[globals()[f"{base}_ALPHA"]] = _alpha_from_window(interval, window)
    v_float64[globals()[f"{base}_NOISE_SCALE"]] = sens[noise_key]

    if upper_key is None:
        # single-limit channel
        v_float64[globals()[f"{base}_LIMIT"]] = sens[lower_key]
    else:
        v_float64[globals()[f"{base}_LOWER_LIMIT"]] = sens[lower_key]
        v_float64[globals()[f"{base}_UPPER_LIMIT"]] = sens[upper_key]

# ==============================================================================
# EMPLOYEE PARAMETERS
# ==============================================================================
def build_employee_parameters_matrix(facility_cfg: Union[Dict[str, Any], str, Path] = "configs/facility_runtime_config.json") -> np.ndarray:
    facility_cfg = _load_json_source(facility_cfg, "Runtime configuration file")

    roster = facility_cfg["employee_workforce_roster"]
    m = np.zeros((len(roster), N_EMP_FIELDS), dtype=np.float64)
    for row, op in enumerate(roster):
        m[row, IDX_EMP_PARAM_CARELESSNESS] = op["base_carelessness_factor"]
        m[row, IDX_EMP_PARAM_CONVECTIVE_Q_MEAN] = op["convective_heat_load_mean_w"]
        m[row, IDX_EMP_PARAM_CONVECTIVE_Q_SCALE] = op["convective_heat_load_scale_w"]
        m[row, IDX_EMP_PARAM_CONVECTIVE_Q_LOW] = op["convective_heat_load_low_w"]
        m[row, IDX_EMP_PARAM_CONVECTIVE_Q_HIGH] = op["convective_heat_load_high_w"]
        m[row, IDX_EMP_PARAM_FATIGUE_SENSITIVITY] = op["fatigue_sensitivity_scale"]
        m[row, IDX_EMP_PARAM_ERROR_BASELINE_P] = op["error_baseline_probability"]
        m[row, IDX_EMP_PARAM_ERROR_SENSITIVITY] = op["error_sensitivity_slope"]
        m[row, IDX_EMP_PARAM_SLACK_LOW_BOUND_MS] = op["forklift_slack_low_bound_sec"] * 1e3
        m[row, IDX_EMP_PARAM_SLACK_HIGH_BOUND_MS] = op["forklift_slack_high_bound_sec"] * 1e3
    return m

def build_supervisor_parameters_matrix(facility_cfg: Union[Dict[str, Any], str, Path] = "configs/facility_runtime_config.json") -> np.ndarray:
    facility_cfg = _load_json_source(facility_cfg, "Runtime configuration file")

    roster = facility_cfg["supervisor_workforce_roster"]
    m = np.zeros((7, NUM_DOORS, N_EMP_FIELDS), dtype=np.float64)
    for sup in roster:
        day = sup["day_of_week"]
        door = sup["door_code"]
        m[day, door, IDX_EMP_PARAM_CARELESSNESS] = sup["base_carelessness_factor"]
        m[day, door, IDX_EMP_PARAM_FATIGUE_SENSITIVITY] = sup["fatigue_sensitivity_scale"]
        m[day, door, IDX_EMP_PARAM_ERROR_SENSITIVITY] = sup["error_sensitivity_slope"]
        m[day, door, IDX_EMP_PARAM_SLACK_LOW_BOUND_MS] = sup["forklift_slack_low_bound_sec"] * 1e3
        m[day, door, IDX_EMP_PARAM_SLACK_HIGH_BOUND_MS] = sup["forklift_slack_high_bound_sec"] * 1e3
        # IDX_EMP_PARAM_CONVECTIVE_Q_* and IDX_EMP_PARAM_ERROR_BASELINE_P intentionally
        # left at 0.0 -- not used for supervisors (no room heat load, baseline error
        # rate comes from IDX_ROOM_LOGISTICS_P_BASE_SUPERVISOR_FORGET_DOOR instead).
    return m

# ==============================================================================
# DOOR SPECS
# ==============================================================================
def build_door_specs_matrices(facility_cfg: Union[Dict[str, Any], str, Path] = "configs/facility_runtime_config.json"):
    facility_cfg = _load_json_source(facility_cfg, "Runtime configuration file")

    doors = facility_cfg["doorway_channels_mechanical_specs"]
    runtime = facility_cfg["simulation_runtime_settings"]
    m_float64 = np.zeros((len(doors), N_DOOR_FLOAT64_FIELDS), dtype=np.float64)
    m_int64 = np.zeros((len(doors), N_DOOR_INT64_FIELDS), dtype=np.int64)
    for door in doors:
        row = door["door_code"]
        m_float64[row, IDX_FACILITY_TAU_DOOR_SEC] = door["door_tau_sec"]
        m_float64[row, IDX_FACILITY_DT_OVER_TAU_DOOR_SEC] = runtime["dt_engine_seconds"] / max(door["door_tau_sec"], SAFE_FLOOR)
        m_int64[row, IDX_FACILITY_TAU_DOOR_MS] = int(door["door_tau_sec"] * 1e3)
        m_float64[row, IDX_FACILITY_DOOR_HEIGHT] = door["door_height_m"]
        m_float64[row, IDX_FACILITY_DOOR_WIDTH] = door["door_width_m"]
        m_float64[row, IDX_FACILITY_C_WIND_CAPTURE] = door["door_c_wind_capture"]
        m_float64[row, IDX_FACILITY_K_STRIP_BUOYANCY] = door["door_k_strip_buoyancy"]
        m_float64[row, IDX_FACILITY_C_RECT_DISCHARGE] = door["door_c_rectangular_discharge"]
        m_int64[row, IDX_FACILITY_DOOR_TRANSIT_BASELINE_MS] = int(door["door_transit_baseline_sec"] * 1e3)
        m_int64[row, IDX_FACILITY_DOOR_ALARM_TRIGGER_MS] = int(door["door_alarm_trigger_sec"] * 1e3)
    return m_float64, m_int64

# ==============================================================================
# FRUIT PARAMETERS: Respiration curve params
# ============================================================================== 
def build_fruit_parameter_arrays(facility_cfg: Union[Dict[str, Any], str, Path] = "configs/facility_runtime_config.json", respiration_db: Union[Dict[str, Any], str, Path] = "configs/fruit_respiration_curves.json"):
    facility_cfg = _load_json_source(facility_cfg, "Runtime configuration file")
    respiration_db = _load_json_source(respiration_db, "Fruit respiration database")

    crop_list = facility_cfg["crop_variety_parameters"]
    sched = facility_cfg["macro_scheduling_environment"]
    num_crops = len(crop_list)

    fruit_params_float64_matrix = np.zeros((num_crops, N_FRUIT_PARAM_FLOAT64_FIELDS), dtype=np.float64)
    fruit_params_int64_matrix = np.zeros((num_crops, N_FRUIT_PARAM_INT64_FIELDS), dtype=np.int64)
     
    num_crops = len(crop_list)
 
    fruit_curves_T = {}
    fruit_curves_R = {}
    max_curve_len = 0
    max_yearly_map_len = 0

    grading_scale_lengths_vector = np.zeros(num_crops, dtype=np.int64)
 
    for crop in crop_list:
        code = crop["fruit_type_code"]
        fruit_name_key = crop["fruit_name"].lower()

        yearly_weights = crop["historical_yearly_trend_weights"]
        baseline_year = int(min(yearly_weights, key=int))
        num_years = 1
        next_year = baseline_year
        for year_str in sorted(yearly_weights, key=int):
            year = int(year_str)
            if year == baseline_year:
                continue
            elif year - next_year == 1:
                next_year = year
                num_years += 1
            else:
                break

        if num_years > max_yearly_map_len:
            max_yearly_map_len = num_years

        fruit_params_int64_matrix[code, IDX_FRUIT_PARAM_YEARLY_MAP_LEN] = num_years
        fruit_params_int64_matrix[code, IDX_FRUIT_PARAM_YEARLY_MAP_BASELINE] = baseline_year
 
        T_lookup, R_lookup = get_respiration_curve(respiration_db, fruit_name_key)
        T_vec, R_vec, T_min, T_max, T_lo, T_hi, T_ref, Q_10, R_ref = get_fruit_respiration_parameters(T_lookup, R_lookup)
 
        fruit_curves_T[code] = np.asarray(T_vec, dtype=np.float64)
        fruit_curves_R[code] = np.asarray(R_vec, dtype=np.float64)
        curve_len = len(T_vec)
        if curve_len > max_curve_len:
            max_curve_len = curve_len
 
        fruit_params_int64_matrix[code, IDX_FRUIT_PARAM_FRUIT_TYPE_CODE] = int(crop["fruit_type_code"])
 
        fruit_params_int64_matrix[code, IDX_FRUIT_PARAM_TARGET_RETENTION_MS] = int(crop["target_retention_limit_days"] * MS_PER_DAY)
        fruit_params_int64_matrix[code, IDX_FRUIT_PARAM_BIOLOGICAL_SHELF_MS] = int(crop["biological_shelf_life_days"] * MS_PER_DAY)
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_CP_FRUIT] = crop["specific_heat_capacity_j_kg_k"]
 
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_TUNNEL_EXIT_TEMP_REF] = crop["tunnel_exit_temperature_reference_c"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_TUNNEL_EXIT_TEMP_DRIFT] = crop["tunnel_exit_temperature_drift_delta_k"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_TUNNEL_EXIT_TEMP_SCALE] = crop["tunnel_exit_temperature_scale_delta_k"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_TUNNEL_MAX_DECREASE] = crop["tunnel_exit_temperature_max_decrease_delta_k"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_TUNNEL_MAX_INCREASE] = crop["tunnel_exit_temperature_max_increase_delta_k"]
 
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_K_ZONE_LOW] = crop["thermal_conductance_k_zone_low"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_K_ZONE_HIGH] = crop["thermal_conductance_k_zone_high"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_K_ZONE_MOV_MULTIPLIER] = crop["thermal_conductance_movement_multiplier"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_K_BULK_BATCH_COUPLING] = crop["bulk_batch_fruit_coupling_k_constant"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_TRANSP_K_P] = crop["transpiration_coefficient_k_p"]
 
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_T_SETPOINT] = crop["refrigeration_target_t_setpoint_c"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_RH_SETPOINT] = crop["refrigeration_target_rh_setpoint_fraction"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_RH_DEADBAND_REFERENCE] = crop["rh_deadband_reference_fraction"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_RHO_LOAD_BULK] = crop["gross_rho_load_bulk_kg_m3"]
 
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_KM_O2] = crop["oxygen_affinity_fraction_km"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_KI_CO2] = crop["carbon_dioxide_inhibition_ppm"]
 
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_RESP_T_MIN] = T_min
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_RESP_T_MAX] = T_max
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_RESP_T_LO] = T_lo
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_RESP_T_HI] = T_hi
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_RESP_T_REF] = T_ref
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_RESP_R_REF] = R_ref
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_RESP_Q10] = Q_10
 
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_RESP_HEAT_J_KG_CO2] = crop["respiration_heat_j_mg_co2"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_INV_RQ_O2_CONS_CO2_PROD] = 1.0 / crop["respiratory_quotient_co2_prod_to_o2_cons"]
 
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_INTERNAL_ARRIVAL_BASE_P] = crop["internal_arrival_baseline_probability"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_BOX_HANDLING_MEAN_MS] = crop["case_picking_handling_mean_ms"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_BOX_HANDLING_SHAPE] = crop["case_picking_handling_shape"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_QUALITY_SENSITIVITY] = crop["quality_distribution_sensitivity_multiplier"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_CUSTOMER_QUALITY_SENSITIVITY] = crop["customer_quality_sensitivity_multiplier"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_PALLET_RETRIEVAL_DELAY_MEAN_MS] = crop["pallet_retrieval_delay_mean_sec"] * 1e3
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_PALLET_RETRIEVAL_DELAY_SHAPE] = crop["pallet_retrieval_delay_shape"]
 
        fruit_params_int64_matrix[code, IDX_FRUIT_PARAM_PALLET_UNIT_BOXES] = int(crop["pallet_max_unit_boxes_capacity"])
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_BOX_PACKAGING_MASS] = crop["box_packaging_tare_mass_kg"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_CP_PACKAGING] = crop["box_packaging_specific_heat_j_kg_k"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_PALLET_BASE_MASS] = crop["pallet_structural_base_tare_mass_kg"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_CP_PALLET] = crop["pallet_base_specific_heat_j_kg_k"]
 
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_BOX_NET_MASS_MEAN] = crop["box_net_fruit_mass_mean_kg"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_BOX_NET_MASS_SCALE] = crop["box_net_fruit_mass_scale_kg"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_BOX_LEGAL_MIN_MASS] = crop["box_net_fruit_mass_legal_minimum_kg"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_PALLET_NET_MASS_MEAN] = crop["box_net_fruit_mass_mean_kg"] * crop["pallet_max_unit_boxes_capacity"]
 
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_SCHED_BASE_P_URGENT_SHIP] = crop["scheduler_base_probability_urgent_ship"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_SCHED_BASE_P_PARTIAL_BATCH] = crop["scheduler_base_probability_partial_batch"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_SCHED_MIN_SHIP_MASS_KG] = crop["scheduler_min_shipment_mass_kg"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_SCHED_INV_MIN_SHIP_MASS_KG] = 1.0 / max(crop["scheduler_min_shipment_mass_kg"], 1.0)
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_SCHED_MIN_ARRIVED_MASS_KG] = crop["scheduler_min_arrived_mass_kg"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_SCHED_MAX_INVENTORY_KG] = crop["scheduler_max_inventory_capacity_kg"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_SCHED_ARRIVAL_SCALE] = crop["scheduler_arrival_volume_scale"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_SCHED_SHIPMENT_SCALE] = crop["scheduler_shipment_volume_scale"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_SCHED_ARRIVAL_SHAPE] = crop["scheduler_arrival_volume_shape"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_SCHED_SHIPMENT_SHAPE] = crop["scheduler_shipment_volume_shape"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_DISPATCH_MASS_MULT] = crop["scheduler_dispatch_mass_multiple_kg"]

        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_MARKET_CORRELATION] = crop["local_market_correlation"]
        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_MARKET_VOLATILITY_SCALE] = crop["market_volatility_scale"]
 
        fruit_params_int64_matrix[code, IDX_FRUIT_PARAM_RESP_CURVE_TRUE_LEN] = len(T_vec)

        fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_DOOR_LOCK_THRESHOLD_TEMP_C] = crop["door_lock_threshold_temp_c"]

        grading_scale_lengths_vector[code] = len(crop["base_quality_grade_distribution_probs"])
        fruit_params_int64_matrix[code, IDX_FRUIT_PARAM_GRADING_SCALE_LEN] = int(grading_scale_lengths_vector[code])

        fruit_params_int64_matrix[code, IDX_FRUIT_PARAM_RAMP_UP_DURATION_MS] = int(crop["standard_ramp_up_duration_hrs"] * MS_PER_HOUR)

        fruit_params_int64_matrix[code, IDX_FRUIT_PARAM_SCHED_INVENTORY_PALLET_CAPACITY] = int(crop["scheduler_inventory_pallet_capacity"])

        # fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_SCHED_ORDERS_PER_DAY_MULTIPLIER] = crop["scheduler_orders_per_day_multiplier"]
        # fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_SCHED_MAX_MOVES_PER_PALLET_DAY]  = crop["scheduler_max_moves_per_pallet_day"]
        # fruit_params_float64_matrix[code, IDX_FRUIT_PARAM_SCHED_SAFETY_BUFFER]             = crop["scheduler_safety_buffer"]
 
    max_grading_scale_len = np.max(grading_scale_lengths_vector)
    fruit_respiration_curves_matrix = np.zeros((num_crops, max_curve_len, 2), dtype=np.float64)
    base_quality_grade_probs_matrix = np.zeros((num_crops, max_grading_scale_len, 2), dtype=np.float64)
    # inventory_mass_per_grade_vector = np.zeros(max_grading_scale_len, dtype=np.float64)
    historical_yield_matrix = np.zeros((num_crops, max_yearly_map_len, 12), dtype=np.float64)

    for crop in crop_list:
        code = crop["fruit_type_code"]
        active_len = len(fruit_curves_T[code])
        fruit_respiration_curves_matrix[code, :active_len, IDX_FRUIT_RESP_CURVE_T] = fruit_curves_T[code]
        fruit_respiration_curves_matrix[code, :active_len, IDX_FRUIT_RESP_CURVE_R] = fruit_curves_R[code]

        yearly_weights_dict = crop["historical_yearly_trend_weights"]
        yearly_weights = [yearly_weights_dict[k] for k in sorted(yearly_weights_dict, key=int)]
        yearly_map_len = fruit_params_int64_matrix[code, IDX_FRUIT_PARAM_YEARLY_MAP_LEN]
        grading_scale_len = fruit_params_int64_matrix[code, IDX_FRUIT_PARAM_RESP_CURVE_TRUE_LEN]

        base_quality_grade_probs_matrix[code, :grading_scale_len, IDX_FRUIT_BASE_QUALITY_DISTRIBUTION] = np.asarray(
            crop["base_quality_grade_distribution_probs"], dtype=np.float64
        )[:grading_scale_len]
        base_quality_grade_probs_matrix[code, :grading_scale_len, IDX_FRUIT_BASE_QUALITY_DEMANDED] = np.asarray(
            crop["base_customer_required_grade_probs"], dtype=np.float64
        )[:grading_scale_len]
 
        monthly_vec = np.asarray(crop["historical_monthly_weights"], dtype=np.float64)
        yearly_vec = np.asarray(yearly_weights, dtype=np.float64)[:, np.newaxis]
        historical_yield_matrix[code, :yearly_map_len, :] = monthly_vec * yearly_vec
 
    return fruit_params_float64_matrix, fruit_params_int64_matrix, fruit_respiration_curves_matrix, historical_yield_matrix, base_quality_grade_probs_matrix # ,  inventory_mass_per_grade_vector
 
def build_work_schedule_arrays(facility_cfg: Union[Dict[str, Any], str, Path] = "configs/facility_runtime_config.json"):
    facility_cfg = _load_json_source(facility_cfg, "Runtime configuration file")

    sched = facility_cfg["macro_scheduling_environment"]
    day_of_week_weights = np.asarray(sched["day_of_week_labor_capacity_weights"], dtype=np.float64) 
    lambda_event_per_hour_matrix = np.zeros((2, 24), dtype=np.float64)
    lambda_event_per_hour_matrix[IDX_LAMBDA_DIRECTION_INBOUND, :] = np.asarray(sched["lambda_arrival_per_hour_list"], dtype=np.float64)
    lambda_event_per_hour_matrix[IDX_LAMBDA_DIRECTION_OUTBOUND, :] = np.asarray(sched["lambda_shipment_per_hour_list"], dtype=np.float64)

    return day_of_week_weights, lambda_event_per_hour_matrix

# ==============================================================================
# REFRIGERANT PROPERTIES + SURFACES
# ==============================================================================  
def load_fits_json_dictionary_map(
    target_refrigerants: Collection[str] = ("R404A", "R134a"),
    output_dir: Union[str, Path] = "polynomial_surfaces",
):
    fits_json_dictionary_map = {}
    base_path = Path(output_dir)

    for fluid in target_refrigerants:
        file_path = base_path / f"surface_fit_{fluid.lower()}.json"
        if not file_path.exists():
            # if "fit_refrigerant_surfaces" in globals() or fit_refrigerant_surfaces is None:
            if fit_refrigerant_surfaces is None:
                raise RuntimeError(
                    f"{file_path} doesn't exist and fit_refrigerant_surfaces.py isn't "
                    f"importable -- put it on PYTHONPATH so missing refrigerant fits can "
                    f"be auto-generated the way you built it to work."
                )

            sys.argv = [
                "fit_refrigerant_surfaces.py", 
                "--refrigerant", fluid, 
                "--output-folder", str(base_path)
            ]
            fit_refrigerant_surfaces.main()
        
            if not file_path.exists():
                raise FileNotFoundError(f"Generator run failed to produce expected file: {file_path}")

        with file_path.open("r") as f:
            fits_json_dictionary_map[fluid] = json.load(f)
            
    return fits_json_dictionary_map

def build_complete_thermodynamic_refrigerant_pool(
    facility_cfg: Union[Dict[str, Any], str, Path] = "configs/facility_runtime_config.json",
    equipment_catalog: Union[Dict[str, Any], str, Path] = "configs/catalog_equipment_config.json",
    save_fluid_map: bool = True,
    fits_output_dir: Union[str, Path] = "polynomial_surfaces",
):
    facility_cfg = _load_json_source(facility_cfg, "Runtime configuration file")
    equipment_catalog = _load_json_source(equipment_catalog, "Equipment catalog")
    comp, evap, cond = resolve_active_equipment(facility_cfg, equipment_catalog)

    compressor_maps = comp["refrigerants"]
    target_refrigerants = tuple(compressor_maps.keys())
    fluid_key_resolver = {name: i for i, name in enumerate(target_refrigerants)}
    num_fluids = len(target_refrigerants)

    fits_json_dictionary_map = load_fits_json_dictionary_map(target_refrigerants, fits_output_dir)

    V_disp = comp["displacement_cm3_rev"] * 1e-6
    n_hz = comp["nominal_speed_rpm"] / 60.0
    evap_airflow_m3_s = evap["rated_airflow_m3_s"]
    cond_airflow_m3_s = cond["rated_airflow_m3_h"] * INV_3600

    scalar_parameters_matrix = np.zeros((num_fluids, N_REFRIG_SCALAR_FIELDS), dtype=np.float64)
    surfaces_matrix = np.zeros((num_fluids, N_FITTED_SURFACES, 2, MAX_N_COEFFS), dtype=np.float64)
 
    for fluid_name, row_idx in fluid_key_resolver.items():
        refrig_parameter_map = compressor_maps[fluid_name]
        ratings = refrig_parameter_map["compressor_mechanical_ratings"]
        lp_trip_settings = refrig_parameter_map["low_pressure_trip_settings"]

        # --- Fluid scalars ---
        scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_GAMMA] = refrig_parameter_map["universal_fluid_constants"]["gamma_refrigerant"]
        scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_INV_GAMMA] = 1.0 / max(scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_GAMMA], SAFE_FLOOR)
        scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_RATED_SUPERHEAT_K] = ratings["superheat_k"]
        scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_RATED_SUBCOOLING] = ratings["subcooling_k"]
        scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_LP_TRIP_PRESSURE_CUT_OUT_BAR] = lp_trip_settings["cut_out_bar"]
        scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_LP_TRIP_PRESSURE_CUT_IN_BAR] = lp_trip_settings["cut_in_bar"]
 
        # --- Surfaces ---
        fit_node = fits_json_dictionary_map[fluid_name]["fits"]
        performance_fit_node = refrig_parameter_map["fitted_compressor_polynomials"]
        _write_surface(surfaces_matrix, row_idx, SURFACE_ANTOINE_EVAP, fit_node["antoine_evap"])
        _write_surface(surfaces_matrix, row_idx, SURFACE_ANTOINE_COND, fit_node["antoine_cond"])
        _write_surface(surfaces_matrix, row_idx, SURFACE_POLY1D_CUBIC_H4, fit_node["h4_liquid_enthalpy"])
        _write_surface(surfaces_matrix, row_idx, SURFACE_POLY2D_CUBIC_RHO, fit_node["rho_suction_density"])
        _write_surface(surfaces_matrix, row_idx, SURFACE_POLY2D_QUAD_H1, fit_node["h1_suction_enthalpy"])
        _write_surface(surfaces_matrix, row_idx, SURFACE_POLY3D_QUAD_H2S, fit_node["h2s_isentropic_enthalpy"]) 
        _write_surface(surfaces_matrix, row_idx, SURFACE_POLY1D_QUAD_ETA_ISEN, refrig_parameter_map["literature_isentropic_efficiency_fit"])
        _write_surface(surfaces_matrix, row_idx, SURFACE_POLY2D_CUBIC_RATED_COOLING_W, performance_fit_node["cooling_capacity_w"])
        _write_surface(surfaces_matrix, row_idx, SURFACE_POLY2D_CUBIC_RATED_POWER_INPUT_W, performance_fit_node["power_input_w"])
        _write_surface(surfaces_matrix, row_idx, SURFACE_POLY2D_CUBIC_RATED_MASS_FLOW_KG_S, performance_fit_node["mass_flow_kg_s"])

        # --- C_clearance ---
        t_evap_vec = np.array(ratings["evaporating_temperatures_c"])
        t_cond_vec = np.array(ratings["condensing_temperatures_c"])
        m_dot_matrix = np.array(ratings["mass_flow_kg_h"]) * INV_3600   # rows = T_cond, cols = T_evap
        inv_gamma = scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_INV_GAMMA]
        sh = scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_RATED_SUPERHEAT_K]
        rho_coeffs = surfaces_matrix[row_idx, SURFACE_POLY2D_CUBIC_RHO, COEFFS, :]
        rho_bounds = surfaces_matrix[row_idx, SURFACE_POLY2D_CUBIC_RHO, BOUNDS, :]
        antoine_evap_coeffs = surfaces_matrix[row_idx, SURFACE_ANTOINE_EVAP, COEFFS, :]
        antoine_cond_coeffs = surfaces_matrix[row_idx, SURFACE_ANTOINE_COND, COEFFS, :]

        x_vec, y_vec = [], []
        for i, tc in enumerate(t_cond_vec):
            for j, te in enumerate(t_evap_vec):
                if m_dot_matrix[i, j] <= 0.0:
                    continue
                if not (rho_bounds[0] <= te <= rho_bounds[1]):
                    continue
                rho = evaluate_poly2d_cubic_fit(te, sh, rho_coeffs, rho_bounds)
                log_pe = evaluate_antoine_fit(te, antoine_evap_coeffs)
                log_pc = evaluate_antoine_fit(tc, antoine_cond_coeffs)
                pr = math.exp(log_pc - log_pe)
                eta = m_dot_matrix[i, j] / (V_disp * n_hz * rho)
                x_vec.append(pr ** inv_gamma - 1.0)
                y_vec.append(1.0 - eta)
        x_vec, y_vec = np.array(x_vec), np.array(y_vec)
        m, b = least_squares_regression(x_vec, y_vec)
        scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_C_CLEARANCE] = m
        scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_ETA_VOL_ZERO] = 1.0 - b

        # --- Evaporator rated UA + refrigerant-side inverse ---
        r_e = _rating_for_fluid(evap, fluid_name, "Evaporator")
        ua_evap = calculate_ua_coil_rated(r_e["capacity_w"], evap_airflow_m3_s, r_e["air_inlet_temp_c"], r_e["delta_t1_k"], r_e["wet_factor"])
        scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_UA_COIL_EVAP_RATED] = ua_evap
        scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_INV_UA_COIL_EVAP_REFRIG_SIDE] = 1.0 / max(ua_evap * evap["ua_internal_scale"], SAFE_FLOOR)

        # --- Condenser rated UA + refrigerant-side inverse ---
        r_c = _rating_for_fluid(cond, fluid_name, "Condenser")
        ua_cond = calculate_ua_coil_rated(r_c["capacity_w"], cond_airflow_m3_s, r_c["air_inlet_temp_c"], r_c["delta_t_k"], r_c["wet_factor"])
        inv_ua_cond_refrig = 1.0 / max(ua_cond * cond["ua_internal_scale"], SAFE_FLOOR)
        scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_UA_COIL_COND_RATED] = ua_cond
        scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_INV_UA_COIL_COND_REFRIG_SIDE] = inv_ua_cond_refrig
        scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_TWO_OVER_UA_COIL_COND_REFRIG_SIDE] = 2.0 * inv_ua_cond_refrig

        # --- Build-time diagnostics (validation item 2) ---
        etas = 1.0 - y_vec
        # _, b_free = least_squares_regression(x_vec, y_vec)
        print(f"[{fluid_name}] C_clearance={scalar_parameters_matrix[row_idx, IDX_REFRIG_SCALAR_C_CLEARANCE]:.5f} "
              f"| eta_vol range=[{etas.min():.3f}, {etas.max():.3f}] | eta0={1.0 - b:.4f} "
              f"| UA_evap={ua_evap:.0f} W/K | UA_cond={ua_cond:.0f} W/K")

    # Save the fluid_key_resolver
    if save_fluid_map:
        with open("refrigerants_key_resolver.json", "w") as file:
            json.dump(fluid_key_resolver, file, indent=4)
            print("Saved refrigerants key resolver to 'refrigentants_key_resolver.json'")
 
    return scalar_parameters_matrix, surfaces_matrix

def _write_surface(surfaces_matrix, row_idx, surface_row, node):
    coeffs = np.array(node["coefficients"], dtype=np.float64)
    bounds = np.array(node.get("bounds", []), dtype=np.float64)
    surfaces_matrix[row_idx, surface_row, COEFFS, :len(coeffs)] = coeffs
    surfaces_matrix[row_idx, surface_row, BOUNDS, :len(bounds)] = bounds

def size_ledgers(sim_start_ms, sim_end_ms, safety_factor=1.75, facility_cfg: Union[Dict[str, Any], str, Path] = "configs/facility_runtime_config.json"):
    facility_cfg = _load_json_source(facility_cfg, "Runtime configuration file")

    simulated_days = (sim_end_ms - sim_start_ms) * 8.64e-7
    runtime = facility_cfg["simulation_runtime_settings"]
    hw = facility_cfg["refrigeration_and_humidifier_hardware"]
    aux = facility_cfg["auxiliary_environmental_hardware"]
    macro = facility_cfg["macro_scheduling_environment"]
    sens = facility_cfg["digital_sensor_monitoring_profiles"]

    # Compressor events: bounded by minimum idle+run cycle time
    min_cycle_sec = hw["compressor_minimum_idle_time_sec"] + hw["compressor_minimum_active_time_sec"]
    max_cycles_per_day = 86400.0 / min_cycle_sec
    compressor_event_rows = ceil_int(simulated_days * max_cycles_per_day * 2 * safety_factor)  # x2: on + off events per cycle
    defrost_interval_ms = int(hw["electric_defrost_control"]["interval_sec"] * 1e3)
    if defrost_interval_ms <= 0:
        raise ValueError("electric_defrost_control.interval_sec must be positive")
    # One row per completed cycle, plus room for a cycle resumed from an earlier chunk.
    defrost_event_rows = ceil_int((sim_end_ms - sim_start_ms) / defrost_interval_ms) + 2

    # Humidifier events: bounded by PID evaluation interval (can't toggle faster than it evaluates)
    humidifier_pid_interval_sec = aux["relative_humidity_pid_evaluation_interval_sec"]
    max_humidifier_events_per_day = 86400.0 / humidifier_pid_interval_sec
    humidifier_event_rows = ceil_int(simulated_days * max_humidifier_events_per_day * safety_factor)

    daily_arrival_rate = sum(macro["lambda_arrival_per_hour_list"])
    daily_shipment_rate = sum(macro["lambda_shipment_per_hour_list"])

    # Door events: bounded by real door-cycling physics -- each pallet crossing = one open + one close per door use
    # Reuse your existing arrival/dispatch event-rate estimates (already computed above in this function)
    max_door_events_per_day = (daily_arrival_rate + daily_shipment_rate) * 2 * safety_factor  # x2: open + close
    door_event_rows = ceil_int(simulated_days * max_door_events_per_day)
 
    results = {}
    for crop in facility_cfg["crop_variety_parameters"]:
        name = crop["fruit_name"]
 
        pallet_net_mass_mean_kg = crop["box_net_fruit_mass_mean_kg"] * crop["pallet_max_unit_boxes_capacity"]
 
        mean_pallets_per_arrival = (
            crop["scheduler_arrival_volume_shape"] * crop["scheduler_arrival_volume_scale"]
        ) / pallet_net_mass_mean_kg
        mean_pallets_per_shipment = (
            crop["scheduler_shipment_volume_shape"] * crop["scheduler_shipment_volume_scale"]
        ) / pallet_net_mass_mean_kg
 
        monthly = crop["historical_monthly_weights"]
        yearly = crop["historical_yearly_trend_weights"]
        peak_climate_yield_weight = max(m * y for m in monthly for y in yearly.values())
 
        inbound_order_rows = ceil_int(
            daily_arrival_rate * simulated_days * peak_climate_yield_weight * safety_factor
        )
        outbound_order_rows = ceil_int(
            daily_shipment_rate * simulated_days * peak_climate_yield_weight * safety_factor
        )
 
        inbound_pallet_history_rows = ceil_int(inbound_order_rows * mean_pallets_per_arrival)
        outbound_pallet_history_rows = ceil_int(outbound_order_rows * mean_pallets_per_shipment)

        readings_per_pallet = ceil_int(simulated_days * MS_PER_DAY / sens["load_cell_sensor_sampling_interval_sec"] / 1000.0)
        batch_fruit_sensor_rows = ceil_int(
            crop["scheduler_inventory_pallet_capacity"] * readings_per_pallet * safety_factor
        )
 
        results[name] = dict(
            inbound_order_rows=inbound_order_rows,
            outbound_order_rows=outbound_order_rows,
            inbound_pallet_history_rows=inbound_pallet_history_rows,
            outbound_pallet_history_rows=outbound_pallet_history_rows,
            batch_fruit_sensor_rows=batch_fruit_sensor_rows,
        )
 
    # Ledgers persist across crop rotations (room_ephemeral row-index counters are never
    # reset on rotation) -- size the shared arrays to cover every crop that will run in
    # the simulated window, not just one.
    combined = dict(
        N_INBOUND_ORDER_LEDGER_ROWS    = max(r["inbound_order_rows"] for r in results.values()),
        N_OUTBOUND_ORDER_LEDGER_ROWS   = max(r["outbound_order_rows"] for r in results.values()),
        N_INBOUND_PALLET_HISTORY_ROWS  = max(r["inbound_pallet_history_rows"] for r in results.values()),
        N_OUTBOUND_PALLET_HISTORY_ROWS = max(r["outbound_pallet_history_rows"] for r in results.values()),
        N_BATCH_SENSOR_ROWS            = max(r["batch_fruit_sensor_rows"] for r in results.values()),
        N_COMPRESSOR_EVENT_ROWS        = compressor_event_rows,
        N_DEFROST_EVENT_ROWS           = defrost_event_rows,
        N_HUMIDIFIER_EVENT_ROWS        = humidifier_event_rows,
        N_DOOR_EVENT_ROWS              = door_event_rows,
    )
 
    return results, combined

def ceil_int(x: float) -> int:
    return int(x) + (1 if x != int(x) else 0)

def build_ephemeral_arrays(
    sim_start_ms,
    sim_end_ms,
    base_quality_grade_probs_matrix,
    facility_cfg: Union[Dict[str, Any], str, Path] = "configs/facility_runtime_config.json",
    sentinel=SENTINEL_FLOAT64,
    epoch_ms_sentinel=MAX_EPOCH_TIMESTAMP_MS,
):
    facility_cfg = _load_json_source(facility_cfg, "Runtime configuration file")

    room_ephemeral_int64_vector     = np.zeros(N_ROOM_EPHEMERAL_INT64_FIELDS, dtype=np.int64)
    inventory_mass_per_grade_vector = np.zeros_like(base_quality_grade_probs_matrix[0, :, 0])
    available_pallet_indices        = np.empty(MAX_SCANNED_PALLETS, dtype=np.int64)
    box_removal_timestamps          = np.full(MAX_NUM_BOXES, SENTINEL_INT64, dtype=np.int64)
    forcing_cache_float64_vector    = np.empty(N_CACHE_FLOAT64_FIELDS, dtype=np.float64)
    forcing_cache_int64_vector      = np.empty(N_CACHE_INT64_FIELDS, dtype=np.int64)
    door_outcome_flags_matrix       = np.empty((MAX_ACTIVE_PALLETS, N_DOOR_OUTCOME_FIELDS), dtype=np.int8)

    _, ephemeral_ledgers_rows_sizes = size_ledgers(sim_start_ms=sim_start_ms, sim_end_ms=sim_end_ms, facility_cfg=facility_cfg)

    runtime = facility_cfg["simulation_runtime_settings"]
    bucket_ms = int(runtime["forcing_arrays_resample_interval_sec"] * 1e3)
    total_duration_ms = sim_end_ms - sim_start_ms
    total_steps = total_duration_ms // int(runtime["dt_engine_seconds"] * 1e3)

    room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_FORCING_BUCKET_MS] = bucket_ms
    room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_TOTAL_STEPS] = total_steps

    output_int64_matrix   = np.empty((total_steps, N_OUTPUT_INT64_FIELDS), dtype=np.int64)
    output_float64_matrix = np.empty((total_steps, N_OUTPUT_FLOAT64_FIELDS), dtype=np.float64)
    # One row per sensor-polling tick (every poll_gcd_ms), not per engine step. A tick every
    # `stride` steps fits in at most total_steps // stride + 1 rows whatever the starting phase.
    dt_engine_ms = int(runtime["dt_engine_seconds"] * 1e3)
    poll_gcd_ms = compute_sensor_poll_gcd_ms(facility_cfg["digital_sensor_monitoring_profiles"], dt_engine_ms)
    n_telemetry_rows = total_steps // (poll_gcd_ms // dt_engine_ms) + 1
    room_telemetry_matrix = np.full((n_telemetry_rows, N_DBT_EXPORT_FIELDS), SENTINEL_FLOAT64, dtype=np.float64)

    inbound_pallet_history_ledger  = np.full((ephemeral_ledgers_rows_sizes["N_INBOUND_PALLET_HISTORY_ROWS"], N_PALLET_HISTORY_FIELDS), sentinel, dtype=np.float64) 
    inbound_orders_ledger          = np.full((ephemeral_ledgers_rows_sizes["N_INBOUND_ORDER_LEDGER_ROWS"], N_ORDER_LEDGER_FIELDS), sentinel, dtype=np.float64) 
    outbound_pallet_history_ledger = np.full((ephemeral_ledgers_rows_sizes["N_OUTBOUND_PALLET_HISTORY_ROWS"], N_PALLET_HISTORY_FIELDS), sentinel, dtype=np.float64) 
    outbound_orders_ledger         = np.full((ephemeral_ledgers_rows_sizes["N_OUTBOUND_ORDER_LEDGER_ROWS"], N_ORDER_LEDGER_FIELDS), sentinel, dtype=np.float64)
    compressor_event_ledger        = np.zeros((ephemeral_ledgers_rows_sizes["N_COMPRESSOR_EVENT_ROWS"], N_COMPRESSOR_EVENT_FIELDS), dtype=np.int64)
    defrost_event_ledger          = np.zeros((ephemeral_ledgers_rows_sizes["N_DEFROST_EVENT_ROWS"], N_DEFROST_EVENT_FIELDS), dtype=np.int64)
    lp_trip_event_ledger           = np.zeros((ephemeral_ledgers_rows_sizes["N_COMPRESSOR_EVENT_ROWS"], N_LP_TRIP_EVENT_FIELDS), dtype=np.int64)
    humidifier_event_ledger        = np.zeros((ephemeral_ledgers_rows_sizes["N_HUMIDIFIER_EVENT_ROWS"], N_HUMIDIFIER_EVENT_FIELDS), dtype=np.int64)
    door_event_ledger              = np.zeros((ephemeral_ledgers_rows_sizes["N_DOOR_EVENT_ROWS"], N_DOOR_EVENT_FIELDS), dtype=np.int64)
    crop_rotation_event_ledger        = np.zeros((N_CROP_ROTATION_EVENT_ROWS, N_CROP_ROTATION_EVENT_FIELDS), dtype=np.int64)
    refrigerant_rotation_event_ledger = np.zeros((N_REFRIGERANT_ROTATION_EVENT_ROWS, N_REFRIGERANT_ROTATION_EVENT_FIELDS), dtype=np.int64)
    downtime_sequence_event_ledger    = np.full((N_DOWNTIME_SEQUENCE_EVENT_ROWS, N_DOWNTIME_SEQUENCE_EVENT_FIELDS), epoch_ms_sentinel, dtype=np.int64)
    batch_fruit_sensor_ledger      = np.zeros((ephemeral_ledgers_rows_sizes["N_BATCH_SENSOR_ROWS"], N_BATCH_SENSOR_FIELDS), dtype=np.float64)

    return dict(
        room_ephemeral_int64_vector       = room_ephemeral_int64_vector,
        inventory_mass_per_grade_vector   = inventory_mass_per_grade_vector,
        available_pallet_indices          = available_pallet_indices,
        box_removal_timestamps            = box_removal_timestamps,
        inbound_pallet_history_ledger     = inbound_pallet_history_ledger,
        inbound_orders_ledger             = inbound_orders_ledger,
        outbound_pallet_history_ledger    = outbound_pallet_history_ledger,
        outbound_orders_ledger            = outbound_orders_ledger,
        output_int64_matrix               = output_int64_matrix,
        output_float64_matrix             = output_float64_matrix,
        room_telemetry_matrix             = room_telemetry_matrix,
        batch_fruit_sensor_ledger         = batch_fruit_sensor_ledger,
        compressor_event_ledger           = compressor_event_ledger,
        defrost_event_ledger              = defrost_event_ledger,
        humidifier_event_ledger           = humidifier_event_ledger,
        door_event_ledger                 = door_event_ledger,
        crop_rotation_event_ledger        = crop_rotation_event_ledger,
        refrigerant_rotation_event_ledger = refrigerant_rotation_event_ledger,
        downtime_sequence_event_ledger    = downtime_sequence_event_ledger,
        lp_trip_event_ledger              = lp_trip_event_ledger,
        forcing_cache_float64_vector      = forcing_cache_float64_vector,
        forcing_cache_int64_vector        = forcing_cache_int64_vector,
        door_outcome_flags_matrix         = door_outcome_flags_matrix,
    )

def load_json_files(directory_path: Union[str, Path]) -> Dict[str, Any]:
    folder = Path(directory_path)
    json_data_map: Dict[str, Any] = {}
    
    for file in folder.glob("*.json"):
        try:
            json_data_map[file.stem] = json.loads(file.read_text(encoding='utf-8'))
        except json.JSONDecodeError as e:
            # Reraise a clean error highlighting the culprit file name
            raise RuntimeError(
                f"Invalid JSON format in file '{file.name}' at line {e.lineno}, column {e.colno}: {e.msg}"
            ) from e
            
    return json_data_map

    # with open(facility_cfg_path) as f:
    #     facility_cfg = json.load(f) 
    # with open(danfoss_db_path) as f:
    #     danfoss_db = json.load(f)
    # with open(respiration_db_path) as f:
    #     respiration_db = json.load(f)

    # return facility_cfg, danfoss_db, respiration_db

# ==============================================================================
# TOP-LEVEL ORCHESTRATOR
# ==============================================================================
def build_all_parameter_arrays(
    facility_cfg: Union[Dict[str, Any], str, Path] = "configs/facility_runtime_config.json",
    equipment_catalog: Union[Dict[str, Any], str, Path] = "configs/catalog_equipment_config.json",
    respiration_db: Union[Dict[str, Any], str, Path] = "configs/fruit_respiration_curves.json",
    location_id: Union[str, int] = None,
):
    if location_id is None:
        raise ValueError("build_all_parameter_arrays: location_id is required")
    facility_cfg = _load_json_source(facility_cfg, "Runtime configuration file")
    equipment_catalog = _load_json_source(equipment_catalog, "Equipment catalog")
    respiration_db = _load_json_source(respiration_db, "Respiration database")

    # Build arrays
    room_configurations_float64_vector, room_configurations_int64_vector = build_room_configurations_vectors(facility_cfg)
    employee_parameters_matrix = build_employee_parameters_matrix(facility_cfg)
    supervisor_parameters_matrix = build_supervisor_parameters_matrix(facility_cfg)
    door_specs_float64_matrix, door_specs_int64_matrix = build_door_specs_matrices(facility_cfg)
    day_of_week_weights, lambda_event_per_hour_matrix = build_work_schedule_arrays(facility_cfg)
    (
        fruit_parameters_float64_matrix,
        fruit_parameters_int64_matrix,
        fruit_respiration_curves_matrix,
        historical_yield_matrix,
        base_quality_grade_probs_matrix
    ) = build_fruit_parameter_arrays(facility_cfg, respiration_db)
    refrigerant_properties_matrix, refrigerant_surfaces_matrix = build_complete_thermodynamic_refrigerant_pool(facility_cfg, equipment_catalog)
    evaporator_fan_curve_matrix = build_evaporator_fan_curve_matrix(facility_cfg, equipment_catalog_path="configs/catalog_equipment_config.json", fits_output_dir="./polynomial_surfaces")
    equipment_configurations_float64_vector, equipment_configurations_int64_vector = build_equipment_configurations_vectors(evaporator_fan_curve_matrix, facility_cfg, equipment_catalog, location_id)

    return dict(
        room_configurations_float64_vector      = room_configurations_float64_vector,
        room_configurations_int64_vector        = room_configurations_int64_vector,
        equipment_configurations_float64_vector = equipment_configurations_float64_vector,
        equipment_configurations_int64_vector   = equipment_configurations_int64_vector,
        fruit_parameters_float64_matrix         = fruit_parameters_float64_matrix,
        fruit_parameters_int64_matrix           = fruit_parameters_int64_matrix,
        fruit_respiration_curves_matrix         = fruit_respiration_curves_matrix,
        historical_yield_matrix                 = historical_yield_matrix,
        day_of_week_weights                     = day_of_week_weights,
        base_quality_grade_probs_matrix         = base_quality_grade_probs_matrix,
        lambda_event_per_hour_matrix            = lambda_event_per_hour_matrix,
        employee_parameters_matrix              = employee_parameters_matrix,
        supervisor_parameters_matrix            = supervisor_parameters_matrix,
        door_specs_float64_matrix               = door_specs_float64_matrix,
        door_specs_int64_matrix                 = door_specs_int64_matrix,
        refrigerant_properties_matrix           = refrigerant_properties_matrix,
        refrigerant_surfaces_matrix             = refrigerant_surfaces_matrix,
        evaporator_fan_curve_matrix             = evaporator_fan_curve_matrix,
    )
 
 
if __name__ == "__main__":
    data = load_json_files("./configs")
    facility_cfg   = data["facility_runtime_config"]
    equipment_catalog = data["catalog_equipment_config"]
    respiration_db = data["fruit_respiration_curves"]
    parameter_arrays = build_all_parameter_arrays(facility_cfg, equipment_catalog, respiration_db)

    for k, v in parameter_arrays.items():
        print(f"{k}: shape={v.shape}" if isinstance(v, np.ndarray) else f"{k}: {type(v)}")
