import numpy as np
from scipy.optimize import curve_fit
import CoolProp.CoolProp as CP
import json
import argparse
from pathlib import Path

def process_arrays_1d_fit(
    X_range_1d,
    y_1d,
):
    x_min, x_max = np.min(X_range_1d), np.max(X_range_1d)
    X = 2 * (X_range_1d - x_min) / (x_max - x_min) - 1
    bounds = np.array([x_min, x_max], dtype=np.float64)
    return X, y_1d, bounds

def poly1d_quadratic_fit(
    X,
    y,
    bounds,
):
    V_matrix = np.column_stack((
        np.ones_like(X),
        X,
        X**2,
    ))

    c, residuals, rank, s = np.linalg.lstsq(V_matrix, y, rcond=None)
    res_val = residuals[0] if residuals.size > 0 else 0.0

    rmse = (res_val/len(X))**0.5
    target_std = np.std(y)
    relative_rmse = rmse/target_std if target_std != 0 else 0.0

    y_pred = V_matrix @ c
    absolute_errors = np.abs(y - y_pred)
    max_absolute_error = np.max(absolute_errors)

    relative_errors = np.where(y != 0, absolute_errors / np.abs(y), 0.0)
    max_relative_percentage_error = np.max(relative_errors) * 100

    results = {
        "c": c,
        "residuals": residuals,
        "rank": rank,
        "singular_values": s,
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "r_squared": 1 - (relative_rmse ** 2),
        "max_abs_error": max_absolute_error,
        "max_rel_pct_error": max_relative_percentage_error,
        "kappa_condition_number": s[0] / s[-1] if s[-1] != 0 else np.nan,
        "bounds": bounds,
    }
    return results

def poly1d_cubic_fit(
    X,
    y,
    bounds,
):
    V_matrix = np.column_stack((
        np.ones_like(X),
        X,
        X**2,
        X**3,
    ))

    c, residuals, rank, s = np.linalg.lstsq(V_matrix, y, rcond=None)
    res_val = residuals[0] if residuals.size > 0 else 0.0

    rmse = (res_val/len(X))**0.5
    target_std = np.std(y)
    relative_rmse = rmse/target_std if target_std != 0 else 0.0

    y_pred = V_matrix @ c
    absolute_errors = np.abs(y - y_pred)
    max_absolute_error = np.max(absolute_errors)

    relative_errors = np.where(y != 0, absolute_errors / np.abs(y), 0.0)
    max_relative_percentage_error = np.max(relative_errors) * 100

    results = {
        "c": c,
        "residuals": residuals,
        "rank": rank,
        "singular_values": s,
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "r_squared": 1 - (relative_rmse ** 2),
        "max_abs_error": max_absolute_error,
        "max_rel_pct_error": max_relative_percentage_error,
        "kappa_condition_number": s[0] / s[-1] if s[-1] != 0 else np.nan,
        "bounds": bounds,
    }
    return results

def process_arrays_2d_fit(
    X1_range_1d,
    X2_range_1d,
    y_2d,
):
    # 1. Store min/max bounds for your downstream evaluation functions
    x1_min, x1_max = np.min(X1_range_1d), np.max(X1_range_1d)
    x2_min, x2_max = np.min(X2_range_1d), np.max(X2_range_1d)

    # 2. Normalize inputs to [-1, 1] range to eliminate matrix collinearity
    X1_norm_1d = 2 * (X1_range_1d - x1_min) / (x1_max - x1_min) - 1
    X2_norm_1d = 2 * (X2_range_1d - x2_min) / (x2_max - x2_min) - 1

    X1_grid, X2_grid = np.meshgrid(X1_norm_1d, X2_norm_1d, indexing='ij')
    X1 = X1_grid.flatten()
    X2 = X2_grid.flatten()

    y = y_2d.flatten()

    bounds = np.array([x1_min, x1_max, x2_min, x2_max], dtype=np.float64)

    return X1, X2, y, bounds

def poly2d_quadratic_fit(
    X1,
    X2,
    y,
    bounds,
):
    V_matrix = np.column_stack((
        np.ones_like(X1),   # Degree 0: Intercept
        X1, X2,             # Degree 1: Linear terms
        X1**2, X2**2,       # Degree 2: Pure quadratic terms
        X1*X2,              # Degree 2: Cross-product interaction terms
    ))

    # y = y_2d.flatten()

    c, residuals, rank, s = np.linalg.lstsq(V_matrix, y, rcond=None)
    res_val = residuals[0] if residuals.size > 0 else 0.0

    rmse = (res_val/len(X1))**0.5
    target_std = np.std(y)
    relative_rmse = rmse/target_std if target_std != 0 else 0.0

    y_pred = V_matrix @ c
    absolute_errors = np.abs(y - y_pred)
    max_absolute_error = np.max(absolute_errors)

    relative_errors = np.where(y != 0, absolute_errors / np.abs(y), 0.0)
    max_relative_percentage_error = np.max(relative_errors) * 100

    results = {
        "c": c,
        "residuals": residuals,
        "rank": rank,
        "singular_values": s,
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "r_squared": 1 - (relative_rmse ** 2),
        "max_abs_error": max_absolute_error,
        "max_rel_pct_error": max_relative_percentage_error,
        "kappa_condition_number": s[0] / s[-1] if s[-1] != 0 else np.nan,
        "bounds": bounds,
    }
    return results

def poly2d_cubic_fit(
    X1,
    X2,
    y,
    bounds,
):
    V_matrix = np.column_stack((
        np.ones_like(X1),   # Degree 0: Intercept
        X1, X2,             # Degree 1: Linear terms
        X1**2, X2**2, X1*X2,      # Degree 2: Quadratic terms
        X1**3, X2**3, (X1**2)*X2, X1*(X2**2)              # Degree 3: Cubic terms
    ))

    # y = y_2d.flatten()

    c, residuals, rank, s = np.linalg.lstsq(V_matrix, y, rcond=None)
    res_val = residuals[0] if residuals.size > 0 else 0.0

    rmse = (res_val/len(X1))**0.5
    target_std = np.std(y)
    relative_rmse = rmse/target_std if target_std != 0 else 0.0

    y_pred = V_matrix @ c
    absolute_errors = np.abs(y - y_pred)
    max_absolute_error = np.max(absolute_errors)

    relative_errors = np.where(y != 0, absolute_errors / np.abs(y), 0.0)
    max_relative_percentage_error = np.max(relative_errors) * 100

    results = {
        "c": c,
        "residuals": residuals,
        "rank": rank,
        "singular_values": s,
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "r_squared": 1 - (relative_rmse ** 2),
        "max_abs_error": max_absolute_error,
        "max_rel_pct_error": max_relative_percentage_error,
        "kappa_condition_number": s[0] / s[-1] if s[-1] != 0 else np.nan,
        "bounds": bounds,
    }
    return results

def process_arrays_3d_fit(
    X1_range_1d,
    X2_range_1d,
    X3_range_1d,
    y_3d,
):
    # 1. Capture tracking boundaries for all three dimensions
    x1_min, x1_max = np.min(X1_range_1d), np.max(X1_range_1d)
    x2_min, x2_max = np.min(X2_range_1d), np.max(X2_range_1d)
    x3_min, x3_max = np.min(X3_range_1d), np.max(X3_range_1d)

    # 2. Standardize target axes to the numerical range system
    X1_norm_1d = 2 * (X1_range_1d - x1_min) / (x1_max - x1_min) - 1
    X2_norm_1d = 2 * (X2_range_1d - x2_min) / (x2_max - x2_min) - 1
    X3_norm_1d = 2 * (X3_range_1d - x3_min) / (x3_max - x3_min) - 1

    X1_grid, X2_grid, X3_grid = np.meshgrid(X1_norm_1d, X2_norm_1d, X3_norm_1d, indexing='ij')
    X1 = X1_grid.flatten()
    X2 = X2_grid.flatten()
    X3 = X3_grid.flatten()

    y = y_3d.flatten()

    bounds = np.array([x1_min, x1_max, x2_min, x2_max, x3_min, x3_max], dtype=np.float64)

    return X1, X2, X3, y, bounds


def poly3d_quadratic_fit(
    X1,
    X2,
    X3,
    y,
    bounds,
):
    V_matrix = np.column_stack((
        np.ones_like(X1),
        X1, X2, X3,          # 3: Linear terms
        X1**2, X2**2, X3**2, # 3: Pure quadratic terms
        X1 * X2,             # 3: Cross-product interaction terms
        X1 * X3,
        X2 * X3
    ))

    # y = y_3d.flatten()

    c, residuals, rank, s = np.linalg.lstsq(V_matrix, y, rcond=None)
    res_val = residuals[0] if residuals.size > 0 else 0.0

    rmse = (res_val/len(X1))**0.5
    target_std = np.std(y)
    relative_rmse = rmse/target_std if target_std != 0 else 0.0

    y_pred = V_matrix @ c
    absolute_errors = np.abs(y - y_pred)
    max_absolute_error = np.max(absolute_errors)

    relative_errors = np.where(y != 0, absolute_errors / np.abs(y), 0.0)
    max_relative_percentage_error = np.max(relative_errors) * 100

    results = {
        "c": c,
        "residuals": residuals,
        "rank": rank,
        "singular_values": s,
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "r_squared": 1 - (relative_rmse ** 2),
        "max_abs_error": max_absolute_error,
        "max_rel_pct_error": max_relative_percentage_error,
        "kappa_condition_number": s[0] / s[-1] if s[-1] != 0 else np.nan,
        "bounds": bounds,
    }
    return results

def antoine_function_logP(T, A1, A2, A3):
    return A1 + A2/(T + A3)

def antoine_fit(
    antoine_function,
    t_celsius_arr,
    p_bar_arr,
    initial_guess=[10.0, -2000.0, 245.0],
):
    X = t_celsius_arr
    y = np.log(p_bar_arr)

    try:
        popt, pcov = curve_fit(antoine_function, X, y, p0=initial_guess)
        std_errors = np.sqrt(np.diag(pcov))
        relative_errors = np.where(
            np.abs(popt) > 1e-10,
            std_errors / np.abs(popt),
            np.nan  # Returns NaN for near-zero parameters instead of crashing
        )

        y_pred = antoine_function(X, *popt)
        
        # 1. Standard Residuals and R-squared
        absolute_errors = np.abs(y - y_pred)
        ss_res = np.sum(absolute_errors ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # 2. Physical Engineering Errors (Crucial for Pressure logs)
        max_absolute_error = np.max(absolute_errors)
        rmse = (ss_res / len(y)) ** 0.5
        
        # Calculate localized percentage errors
        # Safe division in case pressure array contains 0 (unlikely for bar)
        pct_errors = np.where(y != 0, (absolute_errors / np.abs(y)) * 100, 0.0)
        max_relative_percentage_error = np.max(pct_errors)
        mean_relative_percentage_error = np.mean(pct_errors)
        
        results = {
            "A_antoine": popt,
            "cov": pcov,
            "standard_errors": std_errors,
            "relative_errors": relative_errors,
            "rmse": rmse,
            "r_squared": r_squared,
            "max_abs_error": max_absolute_error,
            "max_rel_pct_error": max_relative_percentage_error,
            "mean_rel_pct_error": mean_relative_percentage_error,
        }
        return results

    except Exception as e:
        print(f"Fitting failed: {e}")
        raise

def generate_input_arrays(
    fluid="R404A",
    num_points=50,
    t_evap_c_low=-20.0,
    t_evap_c_high=15.0,
    t_cond_c_low=25.0,
    t_cond_c_high=55.0,
    delta_t_sh_low=0.0,
    delta_t_sh_high=35.0,
    delta_t_sh_ref=11.1,
):
    if fluid == "R448A":
        fluid_string = "R32&R125&R1234yf&R134a&R1234ze(E)"
        AS = CP.AbstractState("HEOS", fluid_string)

        mass_fractions = [0.26, 0.26, 0.20, 0.21, 0.07]
        AS.set_mass_fractions(mass_fractions)

    elif fluid == "R449A":
        fluid_string = "R32&R125&R1234yf&R134a"
        AS = CP.AbstractState("HEOS", fluid_string)

        mass_fractions = [0.243, 0.247, 0.253, 0.257]
        AS.set_mass_fractions(mass_fractions)

    elif fluid == "R407F":
        fluid_string = "R32&R125&R134a"
        AS = CP.AbstractState("HEOS", fluid_string)

        mass_fractions = [0.30, 0.30, 0.40]
        AS.set_mass_fractions(mass_fractions)

    elif fluid == "R452A":
        fluid_string = "R32&R125&R1234yf"
        AS = CP.AbstractState("HEOS", fluid_string)

        mass_fractions = [0.11, 0.59, 0.30]
        AS.set_mass_fractions(mass_fractions)

    elif fluid == "R407A":
        fluid_string = "R32&R125&R134a"
        AS = CP.AbstractState("HEOS", fluid_string)

        mass_fractions = [0.20, 0.20, 0.60]
        AS.set_mass_fractions(mass_fractions)

    else:
        fluid_string = fluid
        AS = CP.AbstractState("HEOS", fluid_string)
    
    # Create a 2D grid of boiling and condensing temperatures
    t_evap_c_range = np.linspace(t_evap_c_low, t_evap_c_high, num_points)  # Handles both fruit room targets
    t_cond_c_range = np.linspace(t_cond_c_low, t_cond_c_high, num_points)   # Handles cold days to hot days
    delta_t_sh_range = np.linspace(delta_t_sh_low, delta_t_sh_high, num_points)
    
    t_evap_k_range = t_evap_c_range + 273.15
    t_cond_k_range = t_cond_c_range + 273.15
    
    y_p_evap_bar_1d = np.zeros_like(t_evap_c_range)
    y_p_cond_bar_1d = np.zeros_like(t_cond_c_range)
    y_h4_j_kg_1d = np.zeros_like(t_cond_c_range)
    
    y_rho_suction_kg_m3_2d = np.zeros((num_points, num_points), dtype=float)
    y_h1_j_kg_2d = np.zeros_like(y_rho_suction_kg_m3_2d)
    y_h2s_j_kg_2d = np.zeros_like(y_rho_suction_kg_m3_2d)

    y_h2s_j_kg_3d = np.zeros((num_points, num_points, num_points), dtype=float)
    
    for i, t_cond_k in enumerate(t_cond_k_range):
        AS.update(CP.QT_INPUTS, 0.0, t_cond_k)
        y_p_cond_bar_1d[i] = AS.p() / 1e5
        y_h4_j_kg_1d[i] = AS.hmass()
    p_cond_pa_arr = y_p_cond_bar_1d*1e5
    
    for i, t_evap_k in enumerate(t_evap_k_range):
        AS.update(CP.QT_INPUTS, 1.0, t_evap_k)
        p_evap_pa = AS.p()
        y_p_evap_bar_1d[i] = p_evap_pa / 1e5

        # t_dew_k = AS.T()
    
        for j, delta_t_sh in enumerate(delta_t_sh_range):
            t_inlet_k = t_evap_k + delta_t_sh
            # p_cond_pa = y_p_cond_bar_1d[j] * 1e5
    
            # Physics Boundary Check: Prevent the script from evaluating a temperature 
            # that is colder than the boiling point (which would break physics)
            # if superheat is exactly 0, force pure saturation vapor
            if t_inlet_k <= t_evap_k:
                # Fallback to the saturation line property if the grid coordinate is unphysical
                AS.update(CP.QT_INPUTS, 1.0, t_evap_k)
            else:
                # Update CoolProp using true decoupled Pressure and Temperature inputs
                AS.update(CP.PT_INPUTS, p_evap_pa, t_inlet_k + 0.001)
    
            rho_suction_kg_m3 = AS.rhomass()
            h1_j_kg = AS.hmass()
            s1_j_kg_k = AS.smass()
            y_rho_suction_kg_m3_2d[i, j] = rho_suction_kg_m3
            y_h1_j_kg_2d[i, j] = h1_j_kg


            for k, p_cond_pa in enumerate(p_cond_pa_arr):
                AS.update(CP.PSmass_INPUTS, p_cond_pa, s1_j_kg_k)
                y_h2s_j_kg_3d[i, j, k] = AS.hmass()
    
    p_evap_pa_arr = y_p_evap_bar_1d*1e5

    for i, (t_evap_k, p_evap_pa) in enumerate(zip(t_evap_k_range, p_evap_pa_arr)):
        t_inlet_k = t_evap_k + delta_t_sh_ref

        if t_inlet_k <= t_evap_k:
            AS.update(CP.QT_INPUTS, 1.0, t_evap_k)
        else:
            AS.update(CP.PT_INPUTS, p_evap_pa, t_inlet_k + 0.001)

        s1_j_kg_k = AS.smass()

        for j, p_cond_pa in enumerate(p_cond_pa_arr):
            AS.update(CP.PSmass_INPUTS, p_cond_pa, s1_j_kg_k)
            y_h2s_j_kg_2d[i, j] = AS.hmass()

    return (
        t_evap_c_range,
        t_cond_c_range,
        delta_t_sh_range,
        y_p_evap_bar_1d,
        y_p_cond_bar_1d,
        y_h4_j_kg_1d,
        y_rho_suction_kg_m3_2d,
        y_h1_j_kg_2d,
        y_h2s_j_kg_3d,
    )

def to_lst(val):
    return val.tolist() if hasattr(val, "tolist") else list(val)

def main():
    # 1. Initialize the parser
    parser = argparse.ArgumentParser(
        description="Fit thermodynamic property arrays for a given refrigerant and save to JSON."
    )

    # 2. Add arguments with your original values as defaults
    parser.add_argument(
        "--refrigerant", type=str, default="R404A",
        help="Refrigerant name identifier (default: R404A)"
    )

    parser.add_argument(
        "--num-points", type=int, default=50,
        help="Number of points in calculation grids (default: 50)"
    )

    parser.add_argument(
        "--delta-t-sh-ref", type=float, default=10.0,
        help="Reference superheat temperature delta (default: 10.0)"
    )
    
    # Evaporator temperature ranges
    parser.add_argument(
        "--t-evap-low", type=float, default=-30.0,
        help="Lower bound for evaporating temperature in C (default: -20.0)"
    )
    parser.add_argument(
        "--t-evap-high", type=float, default=15.0,
        help="Upper bound for evaporating temperature in C (default: 15.0)"
    )

    # Condenser temperature ranges
    parser.add_argument(
        "--t-cond-low", type=float, default=5.0,
        help="Lower bound for condensing temperature in C (default: 5.0)"
    )
    parser.add_argument(
        "--t-cond-high", type=float, default=55.0,
        help="Upper bound for condensing temperature in C (default: 55.0)"
    )

    # Superheat temperature ranges
    parser.add_argument(
        "--delta-t-sh-low", type=float, default=0.0,
        help="Lower bound for superheat temperature delta (default: 0.0)"
    )
    parser.add_argument(
        "--delta-t-sh-high", type=float, default=35.0,
        help="Upper bound for superheat temperature delta (default: 35.0)"
    )

    # Initial Guess List (parsed as a list of floats)
    parser.add_argument(
        "--initial-guess", type=float, nargs=3, default=[10.0, -2000.0, 245.0],
        help="Three space-separated float values for Antoine fit initial guess (default: 10.0 -2000.0 245.0)"
    )

    parser.add_argument(
        "--output-folder", type=str, default="polynomial_surfaces",
        help="Write output file in specified folder"
    )

    # 3. Parse arguments
    args = parser.parse_args()

    (
    t_evap_c_range,
    t_cond_c_range,
    delta_t_sh_range,
    y_p_evap_bar_1d,
    y_p_cond_bar_1d,
    y_h4_j_kg_1d,
    y_rho_suction_kg_m3_2d,
    y_h1_j_kg_2d,
    y_h2s_j_kg_3d,
    ) = generate_input_arrays(
        fluid           = args.refrigerant.upper(), 
        num_points      = args.num_points,
        t_evap_c_low    = args.t_evap_low,
        t_evap_c_high   = args.t_evap_high,
        t_cond_c_low    = args.t_cond_low,
        t_cond_c_high   = args.t_cond_high,
        delta_t_sh_low  = args.delta_t_sh_low,
        delta_t_sh_high = args.delta_t_sh_high,
        delta_t_sh_ref  = args.delta_t_sh_ref,
    ) 

    antoine_evap_results = antoine_fit(
        antoine_function=antoine_function_logP,
        t_celsius_arr=t_evap_c_range,
        p_bar_arr=y_p_evap_bar_1d,
        initial_guess=args.initial_guess,
    )
    A_evap = antoine_evap_results["A_antoine"]

    antoine_cond_results = antoine_fit(
        antoine_function=antoine_function_logP,
        t_celsius_arr=t_cond_c_range,
        p_bar_arr=y_p_cond_bar_1d,
        initial_guess=args.initial_guess,
    )
    A_cond = antoine_cond_results["A_antoine"]

    # h4 fitting
    X, y, bounds = process_arrays_1d_fit(t_cond_c_range, y_h4_j_kg_1d)
    h4_fit_results = poly1d_cubic_fit(
        X,
        y,
        bounds,
    )
    c_h4 = h4_fit_results["c"]

    # rho suction fitting
    X1, X2, y, bounds = process_arrays_2d_fit(t_evap_c_range, delta_t_sh_range, y_rho_suction_kg_m3_2d)
    rho_fit_results = poly2d_cubic_fit(
        X1,
        X2,
        y,
        bounds,
    )
    c_rho = rho_fit_results["c"]

    # h1 fitting
    X1, X2, y, bounds = process_arrays_2d_fit(t_evap_c_range, delta_t_sh_range, y_h1_j_kg_2d)
    h1_fit_results = poly2d_quadratic_fit(
        X1,
        X2,
        y,
        bounds,
    )
    c_h1 = h1_fit_results["c"]

    X1, X2, X3, y, bounds = process_arrays_3d_fit(t_evap_c_range, delta_t_sh_range, t_cond_c_range, y_h2s_j_kg_3d)
    h2s_fit_results = poly3d_quadratic_fit(
        X1,
        X2,
        X3,
        y,
        bounds,
    )
    c_h2s = h2s_fit_results["c"]

    refrigerant_data = {
        "metadata": {
            "fluid": args.refrigerant,
            "num_points": args.num_points,
        },
        "fits": {
            "antoine_evap": {
                "coefficients": to_lst(antoine_evap_results["A_antoine"]),
                "type": "nonlinear_antoine"
            },
            "antoine_cond": {
                "coefficients": to_lst(antoine_cond_results["A_antoine"]),
                "type": "nonlinear_antoine"
            },
            "h4_liquid_enthalpy": {
                "coefficients": to_lst(h4_fit_results["c"]),
                "bounds": to_lst(h4_fit_results["bounds"]),
                "type": "poly1d_cubic"
            },
            "rho_suction_density": {
                "coefficients": to_lst(rho_fit_results["c"]),
                "bounds": to_lst(rho_fit_results["bounds"]),
                "type": "poly2d_cubic"
            },
            "h1_suction_enthalpy": {
                "coefficients": to_lst(h1_fit_results["c"]),
                "bounds": to_lst(h1_fit_results["bounds"]),
                "type": "poly2d_quadratic"
            },
            "h2s_isentropic_enthalpy": {
                "coefficients": to_lst(h2s_fit_results["c"]),
                "bounds": to_lst(h2s_fit_results["bounds"]),
                "type": "poly3d_quadratic"
            }
        },
        "diagnostics": {
            "antoine_evap_fit": {
                "rmse": float(antoine_evap_results["rmse"]),
                "r_squared": float(antoine_evap_results["r_squared"]),
                "max_abs_error": float(antoine_evap_results["max_abs_error"]),
                "max_rel_pct_error": float(antoine_evap_results["max_rel_pct_error"]),
                "mean_rel_pct_error": float(antoine_evap_results["mean_rel_pct_error"]),
                "standard_errors": to_lst(antoine_evap_results["standard_errors"]),
            },
            "antoine_cond_fit": {
                "rmse": float(antoine_cond_results["rmse"]),
                "r_squared": float(antoine_cond_results["r_squared"]),
                "max_abs_error": float(antoine_cond_results["max_abs_error"]),
                "max_rel_pct_error": float(antoine_cond_results["max_rel_pct_error"]),
                "mean_rel_pct_error": float(antoine_cond_results["mean_rel_pct_error"]),
                "standard_errors": to_lst(antoine_cond_results["standard_errors"]),
            },
            "h4_liquid_enthalpy_fit": {
                "rmse": float(h4_fit_results["rmse"]),
                "r_squared": float(h4_fit_results["r_squared"]),
            },
            "rho_suction_density_fit": {
                "rmse": float(rho_fit_results["rmse"]),
                "r_squared": float(rho_fit_results["r_squared"]),
                "max_abs_error": float(rho_fit_results["max_abs_error"]),
                "max_rel_pct_error": float(rho_fit_results["max_rel_pct_error"]),
                "matrix_condition_number": float(rho_fit_results["kappa_condition_number"]),
            },
            "h1_suction_enthalpy_fit": {
                "rmse": float(h1_fit_results["rmse"]),
                "r_squared": float(h1_fit_results["r_squared"]),
                "max_abs_error": float(h1_fit_results["max_abs_error"]),
                "max_rel_pct_error": float(h1_fit_results["max_rel_pct_error"]),
                "matrix_condition_number": float(h1_fit_results["kappa_condition_number"]),
            },
            "h2s_isentropic_enthalpy_fit": {
                "rmse": float(h2s_fit_results["rmse"]),
                "r_squared": float(h2s_fit_results["r_squared"]),
                "max_abs_error": float(h2s_fit_results["max_abs_error"]),
                "max_rel_pct_error": float(h2s_fit_results["max_rel_pct_error"]),
                "matrix_condition_number": float(h2s_fit_results["kappa_condition_number"]),
            },
        },
    }

    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"surface_fit_{args.refrigerant.lower()}.json"
    with open(file_path, "w") as f:
        json.dump(refrigerant_data, f, indent=4)

if __name__ == "__main__":
    main()
