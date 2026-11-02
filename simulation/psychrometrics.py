"""
psychrometrics.py

Pure physics/psychrometrics formulas: saturation pressure, dew point, humidity ratio conversions, door infiltration, compressor volumetric efficiency, coil UA.
"""

import math
import numpy as np
from jit import njit

from constants import *
from matrix_index_maps import *
from utils import evaluate_poly1d_fit, evaluate_antoine_fit, numba_clip

@njit
def latent_heat_vaporization_j_kg(T_c):
    return (HARRISON_VAP_ENTHALPY_0C - HARRISON_VAP_TEMP_COEFF * T_c) * 1000.0

@njit
def latent_heat_sublimation_j_kg(T_c):
    # Murphy & Koop 2005 (Q. J. R. Meteorol. Soc. 131, 1539-1565), Eq. 5; T in K, result converted J/mol -> J/kg
    T = T_c + CELSIUS_TO_KELVIN
    L_j_mol = (
        MK_SUB_ENTHALPY_INTERCEPT
        + MK_SUB_ENTHALPY_LINEAR_COEFF * T
        - MK_SUB_ENTHALPY_QUAD_COEFF * (T * T)
        + MK_SUB_LOW_T_SCALE * math.exp(-(T * MK_SUB_INV_LOW_T_REF) * (T * MK_SUB_INV_LOW_T_REF))
    )
    return L_j_mol / M_H2O

@njit
def coil_surface_state(M_frost, T_coil_evap):
    # Ice surface if any frost remains or the coil is below 0 C.
    # While frost remains the surface temperature is capped at 0 C (melting plateau).
    is_ice = (M_frost > MASS_KG_FLOOR) or (T_coil_evap < 0.0)
    T_surface = min(T_coil_evap, 0.0) if is_ice else T_coil_evap
    return is_ice, T_surface

@njit
def p_sat_magnus(T_c, magnus_const_set):
    a, b, c = magnus_const_set
    P_sat = a * math.exp((b * T_c) / (T_c + c))
    return P_sat

@njit
def t_dew_point_magnus(T_c, W, P, magnus_const_set):
    P_w = partial_pressure_from_w(W, P)
    P_sat = p_sat_magnus(T_c, magnus_const_set)
    P_w = min(P_w, P_sat)
    if P_w <= 0.0:
        return -99.0
    a, b, c = magnus_const_set
    gamma = math.log(P_w / a)
    T_dew = (c * gamma) / (b - gamma)
    return T_dew


@njit
def partial_pressure_from_w(W, P):
    P_w = (W * P) / (MOLAR_RATIO_H2O_AIR + W)
    return P_w


@njit
def w_from_partial_pressure(P_w, P):
    W = MOLAR_RATIO_H2O_AIR * (P_w / (P - P_w))
    return W


@njit
def door_infiltration_orifice_slot(
    H_open,
    W_open,
    rho_air_source,
    rho_air_receiving,
    wind_speed,
    C_wind_capture,
    k_strip_buoyancy,
    C_rectangular_discharge,
):
    # Buoyancy: Calculate pressure at the midpoint of the actual physical opening
    g = 9.81
    A_open = H_open * W_open
    delta_P_buoyancy = max((rho_air_receiving - rho_air_source) * g * (H_open * 0.5), 0.0)
    delta_P_buoyancy *= k_strip_buoyancy

    # Wind: Calculate the wind pressure force (Velocity Pressure)
    v_wind_eff = C_wind_capture * wind_speed
    delta_P_wind = 0.5 * rho_air_source * (v_wind_eff * v_wind_eff)

    # Total Pressure Difference (Combined orthogonal forces)
    delta_P_total = math.sqrt(delta_P_buoyancy * delta_P_buoyancy + delta_P_wind * delta_P_wind)

    # Calculate actual volumetric flow rate via the Orifice Equation
    rho_air_avg = 0.5 * (rho_air_receiving + rho_air_source)
    Q_total = C_rectangular_discharge * A_open * math.sqrt((2.0 * delta_P_total) / rho_air_avg)
    m_dot_door = Q_total * rho_air_source

    return m_dot_door


@njit
def compressor_volumetric_efficiency(
    T_evap_c,
    T_cond_c,
    c_antoine_evap,
    c_antoine_cond,
    inv_gamma_refrigerant,
    C_clearance,
    eta_vol_zero
):
    # Calculate saturation pressures via customized Antoine fits
    log_P_evap_bar = evaluate_antoine_fit(T_evap_c, c_antoine_evap)
    log_P_cond_bar = evaluate_antoine_fit(T_cond_c, c_antoine_cond)
    pressure_ratio = max(math.exp(log_P_cond_bar - log_P_evap_bar), 1.0)

    # Calculate actual volumetric pumping efficiency
    eta_vol = eta_vol_zero - C_clearance * ((pressure_ratio ** inv_gamma_refrigerant) - 1.0)

    return max(eta_vol, 0.1), log_P_evap_bar, log_P_cond_bar, pressure_ratio


# @njit
# def calculate_ua_coil(
#     v_dot_air,
#     rho_air_source,
#     Q_rated,
#     td_design,
# ):
#     C = v_dot_air * rho_air_source * CP_AIR
#     epsilon = numba_clip(Q_rated / (C * td_design), 1e-4, 0.999)
#     UA_coil = -np.log(1.0 - epsilon) * C
#     return UA_coil

@njit
def calculate_ua_coil_rated(
    q_rated,
    v_dot_rated,
    t_air_inlet_c,
    delta_t_rated,
    wet_factor,
):
    rho_air = P_ATM_STANDARD / (R_DRY_AIR * (t_air_inlet_c + CELSIUS_TO_KELVIN))
    C_air = v_dot_rated * rho_air * CP_AIR
    Q_dry = q_rated / wet_factor
    epsilon = numba_clip(Q_dry / (C_air * delta_t_rated), 1e-4, 0.999)
    return -C_air * np.log(1 - epsilon)

@njit
def calculate_ua_coil_air(
    ua_coil_rated,
    v_dot_rated,
    v_dot_actual,
    n
):
    return ua_coil_rated * (v_dot_actual / v_dot_rated) ** n

# --- Frost functions
@njit
def frost_deposition_density_kg_m3(T_frost_surface_c, T_air_dew_point_c):
    return FROST_DENSITY_PREFACTOR_KG_M3 * math.exp(
        FROST_DENSITY_T_FS_COEFF_PER_C * T_frost_surface_c + FROST_DENSITY_T_DEW_COEFF_PER_C * T_air_dew_point_c)

@njit
def frost_conductivity_w_m_k(frost_density_kg_m3):
    rho = min(max(frost_density_kg_m3, FROST_K_DENSITY_MIN_KG_M3), FROST_K_DENSITY_MAX_KG_M3)
    return FROST_K_INTERCEPT_W_M_K + FROST_K_LINEAR_COEFF * rho + FROST_K_QUADRATIC_COEFF * (rho * rho)

@njit
def frost_layer_thickness_m(M_frost_kg, frost_density_kg_m3, coil_external_area_m2):
    return M_frost_kg / (frost_density_kg_m3 * coil_external_area_m2)     # uniform layer over the external area

@njit
def frost_thermal_resistance_k_w(thickness_m, frost_density_kg_m3, coil_external_area_m2):
    return thickness_m / (frost_conductivity_w_m_k(frost_density_kg_m3) * coil_external_area_m2)

@njit
def face_frost_covered_area(
    V_face_m3,
    evap_frontal_area_m2,
    inv_face_bridge_length_m,
):
    return min(evap_frontal_area_m2, V_face_m3 * inv_face_bridge_length_m) if V_face_m3 > 0.0 else 0.0

@njit
def frost_geometry(
    V_passage_m3,
    V_face_m3,
    inv_evap_surface_area_m2,
    evap_frontal_area_m2,
    inv_evap_frontal_area_m2,
    inv_face_bridge_length_m
):
    passage_thickness = V_passage_m3 * inv_evap_surface_area_m2
    if V_face_m3 > 0.0:
        face_covered_area = min(evap_frontal_area_m2, V_face_m3 * inv_face_bridge_length_m)
        face_thickness = V_face_m3 / face_covered_area
        face_covered_factor = face_covered_area * inv_evap_frontal_area_m2
    else:
        face_covered_area = 0.0
        face_thickness = 0.0
        face_covered_factor = 0.0
    return passage_thickness, face_thickness, face_covered_area, face_covered_factor

@njit
def passage_frost_free_flow_fraction(
    thickness_m,
    inv_fin_clear_gap_m,
    inv_tube_clear_gap_m,
):
    if thickness_m <= 0.0:
        return 1.0
    else:
        # Frost grows by 2*thickness across the fin gap and across the gap between tubes (free-flow area ratio, clean = 1)
        fin_factor = max(1.0 - 2.0 * thickness_m * inv_fin_clear_gap_m, 0.0)
        tube_factor = max(1.0 - 2.0 * thickness_m * inv_tube_clear_gap_m, 0.0)
        return fin_factor * tube_factor

@njit
def combined_frost_free_flow_fraction(passage_frost_free_flow_fraction, face_covered_factor):
    return min(passage_frost_free_flow_fraction, 1 - face_covered_factor)

@njit
def frost_airflow_ratio(free_flow_fraction, fan_curve_matrix, inv_v_dot_clean, p_over_sq_v_dot_clean):
    # Operating point: the fan delivers Q = evaluate_poly1d_fit(p, ...) at pressure p; the coil needs p = K*Q^2, K = K_clean * f^-n.
    # K*Q(p)^2 - p decreases with p, so there is one root; found by halving the pressure interval.
    if free_flow_fraction == 0.0:
        return 0.0
    else:
        c = fan_curve_matrix[COEFFS, :]
        bounds = fan_curve_matrix[BOUNDS, :]
        K = p_over_sq_v_dot_clean * free_flow_fraction ** (-FROST_BLOCKAGE_PRESSURE_EXPONENT)
        p_lo = bounds[0]
        p_hi = bounds[1]
        v_dot_at_p_hi = evaluate_poly1d_fit(p_hi, c, bounds)
        if K * v_dot_at_p_hi * v_dot_at_p_hi >= p_hi:
            # Past the last measured fan point (stall): ASSUMPTION, fan pressure held at its last measured value
            return math.sqrt(p_hi / K) * inv_v_dot_clean
        for _ in range(40):
            p_mid = 0.5 * (p_lo + p_hi)
            v_dot_mid = evaluate_poly1d_fit(p_mid, c, bounds)
            if K * v_dot_mid * v_dot_mid > p_mid:
                p_lo = p_mid
            else:
                p_hi = p_mid
        return evaluate_poly1d_fit(0.5 * (p_lo + p_hi), c, bounds) * inv_v_dot_clean

@njit
def frost_conductance_full_contact_layers(
    area_m2,
    thickness_m,
    conductivity_w_m_k,
):
    return 2.0 * conductivity_w_m_k * area_m2 / thickness_m


@njit
def frost_conductance_passage_to_inner_face(
    passage_conductivity_w_m_k,
    face_covered_area_m2,
    face_contact_fraction,
    face_bridge_length_m,
    face_thickness_m,
    face_conductivity_w_m_k,
):
    return 1.0 / (
        face_bridge_length_m / (face_contact_fraction * passage_conductivity_w_m_k * face_covered_area_m2)
        + 0.25 * face_thickness_m / (face_conductivity_w_m_k * face_covered_area_m2)
    )

@njit
def frost_conductance_metal_to_inner_face(
    face_conductivity_w_m_k,
    face_covered_area_m2,
    fin_edge_contact_factor,
    face_thickness_m,
):
    # Only the fin-edge fraction contacts metal; that contact is assumed ideal.
    # The inner face-cell center is one quarter-layer from the contact plane.
    return 4.0 * face_conductivity_w_m_k * (face_covered_area_m2 * fin_edge_contact_factor) / face_thickness_m

@njit
def passage_frost_temperatures_and_capacities(
    M_passage_kg,
    E_passage_j,
):
    C_passage = M_passage_kg * CP_ICE
    inv_C_passage = 1.0 / C_passage
    T_pass_c = E_passage_j * inv_C_passage
    return T_pass_c, C_passage, inv_C_passage

@njit
def face_frost_temperatures_and_capacities(
    M_face_kg,
    E_inner_face_j,
    E_outer_face_j,
):
    C_face_half = M_face_kg * CP_ICE * 0.5
    inv_C_face_half = 1.0 / C_face_half
    T_inner_face_c = inv_C_face_half * E_inner_face_j
    T_outer_face_c = inv_C_face_half * E_outer_face_j
    return T_inner_face_c, T_outer_face_c, C_face_half, inv_C_face_half

@njit
def calculate_energy_state_delta(
    temp_a,
    temp_b,
    C_a,
    inv_C_a,
    C_b,
    inv_C_b,
    conductance,
    dt_sec,
):
    return (
        (C_a * C_b) / (C_a + C_b) * (temp_a - temp_b)
        * (1 - math.exp(-conductance * dt_sec * (inv_C_a + inv_C_b)))
    )

@njit
def frost_densities(
    M_passage_kg,
    M_face_kg,
    V_passage_m3,
    V_face_m3
):
    rho_passage = M_passage_kg / V_passage_m3
    rho_face = M_face_kg / V_face_m3
    return rho_passage, rho_face

@njit
def frost_deposition_density(
    T_dew,
    T_surface,
    T_fs_floor,
):
    T_fs = numba_clip(T_surface, low=T_fs_floor, high=0.0) # frost surface ~ coil temperature, held inside the tested range
    return frost_deposition_density_kg_m3(T_fs, T_dew)
