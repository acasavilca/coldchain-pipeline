import numpy as np
import json
import argparse
from fit_refrigerant_surfaces import to_lst, poly2d_cubic_fit

def process_danfoss_arrays(
    x1_evap_axis,
    x2_cond_axis,
    raw_metric_grid_2d,
):
    # 1. Step through the 2D grid to filter out holes and pair up valid points
    valid_te = []
    valid_tc = []
    valid_y  = []

    for r_idx in range(len(x2_cond_axis)):
        for c_idx in range(len(x1_evap_axis)):
            val = raw_metric_grid_2d[r_idx, c_idx]
            
            # --- THE MISSING HOLE FILTER ---
            # Skip the empty cells completely so they don't warp our regression
            if val == 0.0:
                continue
                
            valid_te.append(x1_evap_axis[c_idx])
            valid_tc.append(x2_cond_axis[r_idx])
            valid_y.append(val)
            
    te_arr = np.array(valid_te, dtype=np.float64)
    tc_arr = np.array(valid_tc, dtype=np.float64)
    y_arr  = np.array(valid_y, dtype=np.float64)
    
    # 2. Extract bounds for your downstream evaluation functions
    te_min, te_max = np.min(te_arr), np.max(te_arr)
    tc_min, tc_max = np.min(tc_arr), np.max(tc_arr)
    
    # 3. Execute your normalization steps exactly like your other functions
    te_norm = 2 * (te_arr - te_min) / (te_max - te_min) - 1
    tc_norm = 2 * (tc_arr - tc_min) / (tc_max - tc_min) - 1

    bounds = np.array([te_min, te_max, tc_min, tc_max], dtype=np.float64)

    return te_norm, tc_norm, y_arr, bounds

def main():
    # 1. Initialize the parser
    parser = argparse.ArgumentParser(
        description="Fit compressor performance arrays for a given refrigerant and save to JSON."
    )

    # 2. Add arguments with your original values as defaults
    parser.add_argument(
        "--danfoss-config-path", type=str, default="refrigerant_properties_pool.json",
        help="Danfoss json with datapoints (default: refrigerant_properties_pool.json)"
    )

    parser.add_argument(
        "--refrigerant", type=str, default="R404A",
        help="Refrigerant name identifier (default: R404A)"
    )

    args = parser.parse_args()

    # Load data
    with open(args.danfoss_config_path, "r") as f:
        danfoss_db = json.load(f)
    refrigerant_db = danfoss_db["refrigerants"][args.refrigerant]["compressor_mechanical_ratings"]

    t_evap_vector = np.array(refrigerant_db["evaporating_temperatures_c"], dtype=np.float64)
    t_cond_vector = np.array(refrigerant_db["condensing_temperatures_c"], dtype=np.float64)

    q_cooling_matrix    = np.array(refrigerant_db["cooling_capacity_w"], dtype=np.float64)
    w_compressor_matrix = np.array(refrigerant_db["power_input_w"], dtype=np.float64)
    m_dot_matrix        = np.array(refrigerant_db["mass_flow_kg_h"], dtype=np.float64)

    # Fit cooling capacity curve
    X1, X2, y, bounds = process_danfoss_arrays(
        t_evap_vector,
        t_cond_vector,
        q_cooling_matrix,
    )
    q_cooling_fit_results = poly2d_cubic_fit(X1, X2, y, bounds)

    # Fit compressor power curve
    X1, X2, y, bounds = process_danfoss_arrays(
        t_evap_vector,
        t_cond_vector,
        w_compressor_matrix,
    )
    w_compressor_fit_results = poly2d_cubic_fit(X1, X2, y, bounds)

    # Fit mass flow curve
    X1, X2, y, bounds = process_danfoss_arrays(
        t_evap_vector,
        t_cond_vector,
        m_dot_matrix/3600.0, # convert from kg/h to kg/s
    )
    mass_flow_fit_results = poly2d_cubic_fit(X1, X2, y, bounds)

    refrigerant_data = {
        "metadata": {
            "fluid": args.refrigerant,
            "compressor_model": danfoss_db["metadata"]["compressor_siglepack_code_number"],
            "rating_standard": danfoss_db["metadata"]["rating_standard"],
            "rating_compressor_frequency": danfoss_db["metadata"]["rating_compressor_frequency"],
        },
        "fits": {
            "q_cooling_rated_w": {
                "coefficients": to_lst(q_cooling_fit_results["c"]),
                "bounds": to_lst(q_cooling_fit_results["bounds"]),
                "type": "poly2d_cubic"
            },
            "w_compressor_rated_w": {
                "coefficients": to_lst(w_compressor_fit_results["c"]),
                "bounds": to_lst(w_compressor_fit_results["bounds"]),
                "type": "poly2d_cubic"
            },
            "mass_flow_rated_kg_s": {
                "coefficients": to_lst(mass_flow_fit_results["c"]),
                "bounds": to_lst(mass_flow_fit_results["bounds"]),
                "type": "poly2d_cubic"
            }
        },
        "diagnostics": {
            "q_cooling_rated_w_fit": {
                "rmse": float(q_cooling_fit_results["rmse"]),
                "r_squared": float(q_cooling_fit_results["r_squared"]),
                "max_abs_error": float(q_cooling_fit_results["max_abs_error"]),
                "max_rel_pct_error": float(q_cooling_fit_results["max_rel_pct_error"]),
                "matrix_condition_number": float(q_cooling_fit_results["kappa_condition_number"]),
            },
            "w_compressor_rated_w_fit": {
                "rmse": float(w_compressor_fit_results["rmse"]),
                "r_squared": float(w_compressor_fit_results["r_squared"]),
                "max_abs_error": float(w_compressor_fit_results["max_abs_error"]),
                "max_rel_pct_error": float(w_compressor_fit_results["max_rel_pct_error"]),
                "matrix_condition_number": float(w_compressor_fit_results["kappa_condition_number"]),
            },
            "mass_flow_rated_kg_s_fit": {
                "rmse": float(mass_flow_fit_results["rmse"]),
                "r_squared": float(mass_flow_fit_results["r_squared"]),
                "max_abs_error": float(mass_flow_fit_results["max_abs_error"]),
                "max_rel_pct_error": float(mass_flow_fit_results["max_rel_pct_error"]),
                "matrix_condition_number": float(mass_flow_fit_results["kappa_condition_number"]),
            },
        },
    }

    file_name = f"surface_fit_{args.refrigerant.lower()}_MTZ160-4_performance.json"
    with open(file_name, "w") as f:
        json.dump(refrigerant_data, f, indent=4)

if __name__ == "__main__":
    main()
