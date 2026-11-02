"""
generate_dbt_seeds.py

Regenerates dbt seed CSVs from the same static config files the simulation
itself reads -- run this whenever facility_runtime_config.json,
catalog_equipment_config.json, or the polynomial_surfaces/*.json files
change, right before `dbt seed`. Keeps the seeds as a derived artifact of one
source of truth instead of a second copy to maintain by hand.

Deliberately NOT included (see conversation -- these are simulation-internal
quantities, or manufacturer curves that don't hold at your actual operating
conditions):
  - Danfoss cooling_capacity_w / power_input_w / mass_flow_kg_s curves --
    published at FIXED reference superheat/subcooling (10K/0K per EN 12900);
    this simulation has dynamic superheat/subcooling, so these curves give
    systematically wrong results at any operating point that isn't the rated
    condition. Superseded by the swept-volume method below.
  - h2s_isentropic_enthalpy, literature_isentropic_efficiency_fit -- only
    needed for a power-inversion mass-flow estimate that turned out to be
    unusable (compressor_power_w is clamped by a friction floor,
    max(W_indicated_w, W_motor_friction_overhead), which can't be inverted
    from the output alone).
  - crop_variety_parameters' stochastic/scheduling tuning knobs (arrival
    volume shape/scale, market_volatility_scale, quality sensitivity
    multipliers, etc.) -- simulation mechanics, not descriptive BI attributes

Included instead: the swept-volume method for mass flow, m_dot = V_disp *
speed * rho_suction * eta_vol -- forward physics, not an inversion of a
clamped quantity, and every input is either a real sensor-derived actual
state or a fixed hardware/refrigerant constant. Needs: antoine_evap/
antoine_cond (saturation temp from measured pressure), rho_suction and h1
(evaluated at ACTUAL measured superheat, not the rated 10K), h4 (from
measured liquid line temp directly), and the compressor/refrigerant physical
constants (displacement, speed, clearance ratio, gamma) needed for
eta_vol -- none of which depend on compressor_power_w or the Danfoss curves.

C_clearance itself is NOT a runtime-only value -- it's fully derivable from
fixed design/rated inputs (crop setpoint + equipment design deltas +
refrigerant rated-point curves), computed ONCE per (crop, refrigerant) pair,
matching derive_fixed_hardware_parameters() exactly. This DOES need the
Danfoss mass_flow_kg_s curve -- but only evaluated at the single fixed rated
design point (t_evap_rated, t_cond_rated), which is exactly the curve's own
calibration condition, not an arbitrary real-time operating point. That's
why it's fine to use here despite being excluded from the operational COP
path above -- different use, not a contradiction: the mismatch problem only
applies when a fixed-condition curve gets evaluated at conditions that don't
match its calibration, and here they're identical by construction.

Also NOT included: downtime_schedule.json / crop_rotation_dates /
refrigerant_rotation_dates. Those describe INTENT, not what actually
happened (rotation firing depends on real-time conditions like
no_fruit_activity) -- seed them separately, later, only if/when you build a
"planned vs. actual" schedule-adherence mart against the (not-yet-built)
crop_rotation_events / refrigerant_rotation_events ledgers.

Usage:
    python generate_dbt_seeds.py \
        --facility-config configs/facility_runtime_config.json \
        --catalog-config configs/catalog_equipment_config.json \
        --surfaces-dir polynomial_surfaces \
        --out-dir seeds
"""

import argparse
import csv
import json
from pathlib import Path
from time_calendar import get_iana_timezone
from utils import get_country_details

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")

# Maps each sensor-profile config key to the telemetry/batch_sensors tag_name
# it actually corresponds to -- this is the one piece of tribal knowledge
# that can't be derived automatically from the JSON key names alone, since
# the config's naming convention and the scadafier's tag names diverged
# slightly (e.g. "temperature_sensor" -> "t_room_c", not a 1:1 string match).
SENSOR_PROFILE_TO_TAG = {
    "temperature_sensor_sampling_interval_sec": "t_room_c",
    "relative_humidity_sensor_sampling_interval_sec": "rh_room_frac",
    "suction_line_temp_sensor_sampling_interval_sec": "t_suction_line_c",
    "suction_line_pressure_sensor_sampling_interval_sec": "p_suction_line_bar",
    "liquid_line_temp_sensor_sampling_interval_sec": "t_liquid_line_c",
    "liquid_line_pressure_sensor_sampling_interval_sec": "p_liquid_line_bar",
    "compressor_power_sensor_sampling_interval_sec": "compressor_power_w",
    "room_air_return_temp_sensor_sampling_interval_sec": "t_room_air_return_c",
    "room_air_supply_temp_sensor_sampling_interval_sec": "t_room_air_supply_c",
    "room_air_supply_rh_sensor_sampling_interval_sec": "rh_room_air_supply_frac",
    "co2_sensor_sampling_interval_sec": "co2_ppm",
    "o2_sensor_sampling_interval_sec": "o2_pct",
    "coil_temp_sensor_sampling_interval_sec": "t_coil_evap_c",
    "evap_air_velocity_sensor_sampling_interval_sec": "evap_air_velocity_m_s",
    "pulp_temp_sensor_sampling_interval_sec": "t_pulp_c",
    "load_cell_sensor_sampling_interval_sec": "mass_kg",
}

def generate_sensor_polling_seed(facility_cfg: dict, out_dir: Path):
    profiles = facility_cfg["digital_sensor_monitoring_profiles"]
    rows = []
    missing = []
    for config_key, tag_name in SENSOR_PROFILE_TO_TAG.items():
        if config_key not in profiles:
            missing.append(config_key)
            continue
        rows.append({
            "tag_name": tag_name,
            "expected_polling_interval_sec": profiles[config_key],
        })
    if missing:
        print(f"WARNING: sensor_polling_config -- config keys not found, skipped: {missing}")
    write_csv(out_dir / "seed_sensor_polling_config.csv", ["tag_name", "expected_polling_interval_sec"], rows)


def generate_compressor_curve_seed(compressor_cfg: dict, out_dir: Path, ref_map: dict):
    """
    Three seeds, three different grains -- checked against the actual JSON
    structure, not assumed:
      - compressor physical constants (displacement, speed): compressor-only.
        general_characteristics sits outside the refrigerants dict entirely.
      - refrigerant physical constants (gamma): refrigerant-only. Lives under
        "universal_fluid_constants" in the source JSON -- a pure substance
        property, doesn't vary by which compressor it's used in.
      - compressor rating conditions (rated superheat_k, rated subcooling_k):
        genuinely BOTH compressor- and refrigerant-dependent. Lives under
        "compressor_mechanical_ratings" -- this is the manufacturer's own
        published rating-test condition for that specific compressor run
        with that specific refrigerant, not a fluid property.
    Plus the mass_flow_kg_s curve, evaluated ONLY at the rated design point
    (see module docstring for why that's fine despite excluding the other
    Danfoss curves).
    """
    compressor_model_code = compressor_cfg["metadata"]["model"]
    displacement_m3_rev = compressor_cfg["displacement_cm3_rev"] / 1e6
    speed_hz = compressor_cfg["nominal_speed_rpm"] / 60.0

    write_csv(
        out_dir / "seed_compressor_physical_constants.csv",
        ["compressor_model_code", "compressor_displacement_m3_rev", "compressor_speed_hz"],
        [{
            "compressor_model_code": compressor_model_code,
            "compressor_displacement_m3_rev": displacement_m3_rev,
            "compressor_speed_hz": speed_hz,
        }],
    )

    refrigerant_const_rows = []
    rating_condition_rows = []
    mass_flow_rows = []
    for refrigerant_name, refrigerant_data in compressor_cfg["refrigerants"].items():
        # Resolve the string to its explicit integer code mapping
        ref_id = int(ref_map[refrigerant_name])

        refrigerant_const_rows.append({
            "refrigerant_code": ref_id,
            "refrigerant_name": refrigerant_name,
            "gamma_refrigerant": refrigerant_data["universal_fluid_constants"]["gamma_refrigerant"],
        })
        ratings = refrigerant_data["compressor_mechanical_ratings"]
        rating_condition_rows.append({
            "compressor_model_code": compressor_model_code,
            "refrigerant_code": ref_id,
            "refrigerant_name": refrigerant_name,
            "rated_superheat_k": ratings["superheat_k"],
            "rated_subcooling_k": ratings["subcooling_k"],
        })
        fit = refrigerant_data["fitted_compressor_polynomials"]["mass_flow_kg_s"]
        mass_flow_rows.append({
            "compressor_model_code": compressor_model_code,
            "refrigerant_code": ref_id,
            "refrigerant_name": refrigerant_name,
            "curve_name": "mass_flow_kg_s",
            "fit_type": fit["type"],
            "coefficients": json.dumps(fit["coefficients"]),
            "bounds": json.dumps(fit["bounds"]),
        })

    write_csv(
        out_dir / "seed_refrigerant_physical_constants.csv",
        ["refrigerant_code", "refrigerant_name", "gamma_refrigerant"],
        refrigerant_const_rows,
    )
    write_csv(
        out_dir / "seed_compressor_rating_conditions.csv",
        ["compressor_model_code", "refrigerant_code", "refrigerant_name", "rated_superheat_k", "rated_subcooling_k"],
        rating_condition_rows,
    )
    write_csv(
        out_dir / "seed_rated_mass_flow_curve.csv",
        ["compressor_model_code", "refrigerant_code", "refrigerant_name", "curve_name", "fit_type", "coefficients", "bounds"],
        mass_flow_rows,
    )
    print(
        "NOTE: C_clearance itself is NOT precomputed here -- it also depends "
        "on the crop's T_setpoint (via t_evap_rated), so it's a function of "
        "(crop, compressor, refrigerant), not refrigerant alone. Compute it "
        "in a dbt model joining seed_compressor_physical_constants + "
        "seed_refrigerant_physical_constants + seed_compressor_rating_conditions "
        "+ seed_rated_mass_flow_curve + seed_refrigerant_saturation_curves + "
        "seed_equipment_design_constants + seed_crop_variety_reference, "
        "cross-joined over crops x refrigerants -- see conversation for the "
        "exact formula."
    )


def generate_location_parameters_seeds(facility_cfg: dict, out_dir: Path):
    """
    Crop-independent, refrigerant-independent design/rated constants -- the
    other half of what's needed for t_evap_rated/t_cond_rated in the
    C_clearance derivation. Single row: this facility has one equipment
    config, not one per location/crop/refrigerant.
    """
    hw = facility_cfg["refrigeration_and_humidifier_hardware"]
    equipment_design_rows = []
    location_details_rows = []
    for location_data in facility_cfg["facility_location"]["locations"].values():
        location_id = int(location_data["id_code"])
        equipment_design_rows.append({
            "location_id": location_id,
            "evaporator_design_temperature_difference_k": hw["evaporator_design_temperature_difference_k"],
            "condenser_design_temperature_difference_k": hw["condenser_design_temperature_difference_k"],
            "ambient_design_temperature_c": location_data["ambient_design_temperature_c"],
            "compressor_deadband_differential_c": hw["compressor_deadband_differential_c"],
            "temperature_hysteresis_enter_margin_c": hw["temperature_hysteresis_enter_margin_c"],
            "temperature_hysteresis_exit_margin_c": hw["temperature_hysteresis_exit_margin_c"],
            "humidifier_max_total_discs_ceiling": hw["humidifier_max_total_discs_ceiling"],
            "humidifier_discs_per_cluster_count": hw["humidifier_discs_per_cluster_count"],
        })

        latitude, longitude = location_data["latitude"], location_data["longitude"]
        iana_timezone = get_iana_timezone(latitude, longitude)
        country_details = get_country_details(latitude, longitude)
        location_details_rows.append({
            "location_id": location_id,
            "region": location_data["name"],
            "country_name": country_details["name"],
            "country_code": country_details["code"],
            "latitude": latitude,
            "longitude": longitude,
            "iana_timezone": iana_timezone,
        })

    write_csv(
        out_dir / "seed_equipment_design_constants.csv",
        [
            "location_id",
            "evaporator_design_temperature_difference_k",
            "condenser_design_temperature_difference_k",
            "ambient_design_temperature_c",
            "compressor_deadband_differential_c",
            "temperature_hysteresis_enter_margin_c",
            "temperature_hysteresis_exit_margin_c",
            "humidifier_max_total_discs_ceiling",
            "humidifier_discs_per_cluster_count",
        ],
        equipment_design_rows,
    )

    write_csv(
        out_dir / "seed_location_details.csv",
        ["location_id", "region", "country_name", "country_code", "latitude", "longitude", "iana_timezone"],
        location_details_rows,
    )

def generate_refrigerant_saturation_seed(surfaces_dir: Path, out_dir: Path, ref_map: dict):
    """
    antoine_evap/antoine_cond (saturation temp from measured pressure),
    rho_suction and h1 (evaluated at ACTUAL measured superheat downstream in
    dbt, not the Danfoss-rated 10K), and h4 (from measured liquid line temp).
    h2s / literature_isentropic_efficiency_fit deliberately excluded -- only
    needed for the power-inversion mass-flow path that's not being used
    (see module docstring).
    """
    needed_curves = ["antoine_evap", "antoine_cond", "rho_suction_density", "h1_suction_enthalpy", "h4_liquid_enthalpy"]
    rows = []
    for surface_file in sorted(surfaces_dir.glob("surface_fit_*.json")):
        data = load_json(surface_file)
        refrigerant_name = data["metadata"]["fluid"]
        
        # Resolve the string to its explicit integer code mapping
        ref_id = int(ref_map[refrigerant_name])

        for curve_name in needed_curves:
            fit = data["fits"][curve_name]
            rows.append({
                "refrigerant_code": ref_id,
                "refrigerant_name": refrigerant_name,
                "curve_name": curve_name,
                "fit_type": fit["type"],
                "coefficients": json.dumps(fit["coefficients"]),
                "bounds": json.dumps(fit.get("bounds", [])),
            })
    write_csv(
        out_dir / "seed_refrigerant_saturation_curves.csv",
        ["refrigerant_code", "refrigerant_name", "curve_name", "fit_type", "coefficients", "bounds"],
        rows,
    )

def generate_crop_variety_seed(facility_cfg: dict, out_dir: Path):
    fieldnames = [
        "fruit_type_code", "fruit_name", "biological_shelf_life_days",
        "target_retention_limit_days", "door_lock_threshold_temp_c",
        "refrigeration_target_t_setpoint_c", "refrigeration_target_rh_setpoint_fraction",
        "chilling_injury_threshold_c", "freezing_injury_threshold_c", "heat_risk_threshold_c",
        "box_packaging_tare_mass_kg", "pallet_structural_base_tare_mass_kg",
    ]
    rows = []
    for crop in facility_cfg["crop_variety_parameters"]:
        rows.append({k: crop[k] for k in fieldnames})
    write_csv(out_dir / "seed_crop_variety_reference.csv", fieldnames, rows)

def generate_driver_seed(facility_cfg: dict, out_dir: Path):
    rows = [
        {"driver_code": w["operator_code"], "driver_name": w["operator_name"]}
        for w in facility_cfg["employee_workforce_roster"]
    ]
    write_csv(out_dir / "seed_driver_reference.csv", ["driver_code", "driver_name"], rows)


def generate_door_seed(facility_cfg: dict, out_dir: Path):
    rows = [
        {"door_code": d["door_code"], "door_name": d["door_name"]}
        for d in facility_cfg["doorway_channels_mechanical_specs"]
    ]
    write_csv(out_dir / "seed_door_reference.csv", ["door_code", "door_name"], rows)

def generate_dispatch_type_seed(facility_cfg: dict, out_dir: Path):
    rows = [
        {"dispatch_type_code": bt["code"], "dispatch_type_name": bt["name"]}
        for bt in facility_cfg["dispatch_types"]
    ]
    write_csv(out_dir / "seed_dispatch_type_reference.csv", ["dispatch_type_code", "dispatch_type_name"], rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facility-config", type=Path, default=Path("configs/facility_runtime_config.json"))
    parser.add_argument("--catalog-config", type=Path, default=Path("configs/catalog_equipment_config.json"))
    parser.add_argument("--surfaces-dir", type=Path, default=Path("polynomial_surfaces"))
    parser.add_argument("--out-dir", type=Path, default=Path("../dbt_project/seeds"))
    parser.add_argument("--resolver-file", type=Path, default=Path("refrigerants_key_resolver.json"))
    args = parser.parse_args()

    facility_cfg = load_json(args.facility_config)
    catalog = load_json(args.catalog_config)
    # The compressor the simulation actually runs is the one selected in the facility config.
    compressor_cfg = catalog["compressors"][str(facility_cfg["active_equipment_id_selection"]["compressor"])]
    ref_map = load_json(args.resolver_file)

    generate_sensor_polling_seed(facility_cfg, args.out_dir)
    generate_compressor_curve_seed(compressor_cfg, args.out_dir, ref_map)
    generate_location_parameters_seeds(facility_cfg, args.out_dir)
    generate_refrigerant_saturation_seed(args.surfaces_dir, args.out_dir, ref_map)
    generate_crop_variety_seed(facility_cfg, args.out_dir)
    generate_driver_seed(facility_cfg, args.out_dir)
    generate_door_seed(facility_cfg, args.out_dir)
    generate_dispatch_type_seed(facility_cfg, args.out_dir)

if __name__ == "__main__":
    main()
