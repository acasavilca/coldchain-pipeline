"""
fit_fan_curves.py

Fits the fan curve of the fan installed in a catalog evaporator: unit-level airflow [m3/s] as a function of the
static pressure increase [Pa], by least squares (quadratic). Datasheet points are read from the catalog ("fans"
section, per fan); the evaporator entry gives the fan id, the wiring key (e.g. "delta") and the number of fans.

Usage: python fit_fan_curves.py [--catalog-config-path ...] [--evaporator-id 1] [--output-folder fan_fits]
"""
import numpy as np
import json
import argparse
import sys
from pathlib import Path
from fit_refrigerant_surfaces import process_arrays_1d_fit, poly1d_quadratic_fit, to_lst

def main():
    parser = argparse.ArgumentParser(
        description="Fit fan curves and save to JSON."
    )
    parser.add_argument("--catalog-config-path", type=str, default="./configs/catalog_equipment_config.json", help="Equipment catalog json with datapoints (default: catalog_equipment_config.json)")
    parser.add_argument("--evaporator-id", type=int, default=1, help="Evaporator identifier (default: 1, model HMR-632-N-4Rx4F-E_delta)")
    parser.add_argument("--output-folder", type=str, default="./polynomial_surfaces", help="Folder for the fit json")
    args = parser.parse_args()

    # Load data
    with open(args.catalog_config_path, "r") as f:
        catalog_cfg = json.load(f)

    try:
        evaporator_cfg = catalog_cfg["evaporators"][str(args.evaporator_id)]
    except KeyError:
        sys.exit(f"Error: Evaporator ID '{args.evaporator_id}' not found in catalog.")

    fan_id = evaporator_cfg["metadata"]["fan_id"]
    try:
        fan_cfg = catalog_cfg["fans"][str(fan_id)]
    except KeyError:
        sys.exit(f"Error: Fan curves for Fan ID '{fan_id}' not found in catalog.")

    n_fans = evaporator_cfg["number_of_fans"]
    fan_wiring = evaporator_cfg["metadata"]["fan_configuration_key"]
    try:
        fan_curves = fan_cfg[fan_wiring]
    except KeyError:
        sys.exit(f"Error: wiring '{fan_wiring}' not found for Fan ID '{fan_id}'.")

    # x = static pressure increase [Pa]; y = unit airflow [m3/s] = per-fan airflow * number of fans (fans in parallel)
    x_pressure_pa = np.array(fan_curves["pressure_increase_pa"], dtype=np.float64)
    y_airflow_m3_s = np.array(fan_curves["air_flow_m3_h"], dtype=np.float64) * n_fans / 3600.0

    # Data consistency check
    if len(x_pressure_pa) != len(y_airflow_m3_s) or len(x_pressure_pa) < 4:
        sys.exit("Error: need at least 4 (pressure, airflow) points of equal length for a quadratic with a residual.")
    order = np.argsort(x_pressure_pa)
    x_pressure_pa, y_airflow_m3_s = x_pressure_pa[order], y_airflow_m3_s[order]
    if not np.all(np.diff(y_airflow_m3_s) < 0.0):
        sys.exit("Error: airflow must decrease monotonically with pressure (check the datasheet points).")

    xs, ys, bounds = process_arrays_1d_fit(x_pressure_pa, y_airflow_m3_s)
    results = poly1d_quadratic_fit(xs, ys, bounds)


    fan_fit_data = {
        "metadata": {
            "evaporator_id": args.evaporator_id,
            "evaporator_model": evaporator_cfg["metadata"].get("model"),
            "fan_id": fan_id,
            "fan_model": fan_cfg["metadata"].get("model"),
            "fan_wiring": fan_wiring,
            "number_of_fans": n_fans,
            "datasheet_air_density_kg_m3": fan_cfg["metadata"].get("air_density_kg_m3"),
            "x": "static pressure increase [Pa]",
            "y": "unit airflow [m3/s] (per-fan airflow x number of fans)",
            "normalization": "x_norm = 2*(x - bounds[0])/(bounds[1] - bounds[0]) - 1",
        },
        "data_points": {"pressure_increase_pa": to_lst(x_pressure_pa), "unit_airflow_m3_s": to_lst(y_airflow_m3_s)},
        "fit": {"coefficients": to_lst(results["c"]), "bounds": to_lst(results["bounds"]), "type": "poly1d_quadratic"},
        "diagnostics": {
            "rmse": float(results["rmse"]), "r_squared": float(results["r_squared"]),
            "max_abs_error": float(results["max_abs_error"]), "max_rel_pct_error": float(results["max_rel_pct_error"]),
            "matrix_condition_number": float(results["kappa_condition_number"]),
        },
    }

    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"fan_fit_evaporator_{args.evaporator_id}.json"
    with open(file_path, "w") as f:
        json.dump(fan_fit_data, f, indent=4)
    print(f"Saved fan fit to {file_path} | max abs error {results['max_abs_error']:.4f} m3/s "
          f"({results['max_rel_pct_error']:.2f}%)")

if __name__ == "__main__":
    main()
