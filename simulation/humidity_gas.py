"""
humidity_gas.py

Humidifier control, evaporator moisture exchange, CO2/O2 gas balance, moisture/fruit-mass updates, derived psychrometric state.
"""

import math
import numpy as np
from jit import njit

from constants import *
from matrix_index_maps import *
from psychrometrics import latent_heat_vaporization_j_kg, latent_heat_sublimation_j_kg, p_sat_magnus, partial_pressure_from_w, t_dew_point_magnus, w_from_partial_pressure, coil_surface_state, passage_frost_temperatures_and_capacities, face_frost_temperatures_and_capacities, face_frost_covered_area, frost_deposition_density
from sensors_control import pid_evaluation
from utils import numba_clip

@njit
def calculate_humidifier_variables(
    room_state_int64_vector,
    room_state_float64_vector,
    room_ephemeral_int64_vector,
    humidifier_event_ledger,
    equipment_configurations_float64_vector,
    equipment_configurations_int64_vector,
    active_crop_params_float64_vector,
    dt_engine_sec,
):
    if room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS] % equipment_configurations_int64_vector[IDX_EQ_HUMIDIFIER_PID_INTERVAL_MS] == 0:
        current_rh_error = active_crop_params_float64_vector[IDX_FRUIT_PARAM_RH_SETPOINT] - room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM_FILTERED_VAL]
        min_rh_pid_output = equipment_configurations_float64_vector[IDX_EQ_HUMIDIFIER_PID_MIN_OUTPUT]
        max_rh_pid_output = equipment_configurations_float64_vector[IDX_EQ_HUMIDIFIER_PID_MAX_OUTPUT]
        if not room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_SAFE_FOR_SHUTDOWN]:
            (
                rh_error,
                rh_p_term,
                rh_i_term,
                rh_d_term,
            ) = pid_evaluation(
                    current_error        = current_rh_error,
                    dt_sec               = equipment_configurations_float64_vector[IDX_EQ_HUMIDIFIER_PID_INTERVAL_SEC],
                    previous_error       = room_state_float64_vector[IDX_ROOM_STATE_HUMIDIFIER_ERROR],
                    integral_accumulator = room_state_float64_vector[IDX_ROOM_STATE_HUMIDIFIER_INTEGRAL_TERM],
                    K_p                  = equipment_configurations_float64_vector[IDX_EQ_HUMIDIFIER_PID_K_P],
                    K_i                  = equipment_configurations_float64_vector[IDX_EQ_HUMIDIFIER_PID_K_I],
                    K_d                  = equipment_configurations_float64_vector[IDX_EQ_HUMIDIFIER_PID_K_D],
                    min_pid_output       = min_rh_pid_output,
                    max_pid_output       = max_rh_pid_output,
                )
            num_discs_per_cluster = equipment_configurations_float64_vector[IDX_EQ_HUMIDIFIER_DISCS_PER_CLUSTER]
            inv_num_discs_per_cluster = equipment_configurations_float64_vector[IDX_EQ_HUMIDIFIER_INV_DISCS_PER_CLUSTER]
            max_piezo_discs = equipment_configurations_float64_vector[IDX_EQ_HUMIDIFIER_MAX_TOTAL_DISCS]
            rh_pid_steps = rh_p_term + rh_i_term + rh_d_term
            num_discs_theoretical = numba_clip(rh_pid_steps, min_rh_pid_output, max_rh_pid_output) * max_piezo_discs
            num_discs_on = int(min(round(num_discs_theoretical * inv_num_discs_per_cluster) * num_discs_per_cluster, max_piezo_discs))
            room_state_float64_vector[IDX_ROOM_STATE_HUMIDIFIER_ERROR] = rh_error
            room_state_float64_vector[IDX_ROOM_STATE_HUMIDIFIER_INTEGRAL_TERM] = rh_i_term
        else:
            num_discs_theoretical = 0
            num_discs_on = 0

        # Write humidifier_event_ledger
        if num_discs_on != room_state_int64_vector[IDX_ROOM_STATE_HUMIDIFIER_NUM_PIEZO_DISCS_ON]:
            row = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_HUMIDIFIER_EVENT_WRITE_IDX]
            humidifier_event_ledger[row, IDX_HUMIDIFIER_EVENT_TIMESTAMP_MS] = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]
            humidifier_event_ledger[row, IDX_HUMIDIFIER_EVENT_NUM_DISCS_ON] = num_discs_on
            room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_HUMIDIFIER_EVENT_WRITE_IDX] = row + 1

        room_state_int64_vector[IDX_ROOM_STATE_HUMIDIFIER_NUM_PIEZO_DISCS_ON] = num_discs_on

    m_dot_humidifier = room_state_float64_vector[IDX_ROOM_STATE_M_DOT_HUMIDIFIER_KG_S]
    f_evap_humid = (room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM] < 1.0) # tiny droplets completely evaporate
    m_dot_discs_target = equipment_configurations_float64_vector[IDX_EQ_HUMIDIFIER_M_DOT_MAX_KG_S] * f_evap_humid * room_state_int64_vector[IDX_ROOM_STATE_HUMIDIFIER_NUM_PIEZO_DISCS_ON]
    m_dot_humidifier += (m_dot_discs_target - m_dot_humidifier) * equipment_configurations_float64_vector[IDX_EQ_DT_OVER_HUMIDIFIER_FAN_TAU_LAG_SEC]
    m_dot_humidifier *= (m_dot_humidifier > SAFE_FLOOR)
    h_fg_room = latent_heat_vaporization_j_kg(room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C]) 
    Q_humidifier_cooling = m_dot_humidifier * h_fg_room
    room_state_float64_vector[IDX_ROOM_STATE_M_DOT_HUMIDIFIER_KG_S] = m_dot_humidifier
    return m_dot_humidifier, Q_humidifier_cooling

@njit
def calculate_evaporator_outlet_state(
    T_room,
    W_room,
    P_ambient,
    Q_sensible_air_to_passage_surface,
    inv_C_evap,
    inv_m_dot_air_evap,
    m_dot_water_removed,
):
    # -- Current Step ---
    T_evap_air_out = T_room - Q_sensible_air_to_passage_surface * inv_C_evap
    W_evap_out = W_room - m_dot_water_removed * inv_m_dot_air_evap
    P_w_evap_out = partial_pressure_from_w(W_evap_out, P_ambient)
    P_sat_evap_out = p_sat_magnus(T_evap_air_out, ALDUCHOV_LIQ_CONST_SET)
    RH_evap_air_out = numba_clip(P_w_evap_out / P_sat_evap_out, 0.0, 1.0)
    
    return T_evap_air_out, RH_evap_air_out

@njit
def calculate_evaporator_heat_and_moisture_exchange(
    room_state_float64_vector,
    forcing_cache_float64_vector,
    equipment_configurations_float64_vector,
    epsilon_evap,
    m_dot_air_evap,
    inv_m_dot_air_evap,
    C_evap,
    inv_C_evap,
    dt_engine_sec,
):
    # Room states
    T_room = room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C]
    W_room = room_state_float64_vector[IDX_ROOM_STATE_AIR_HUMIDITY_RATIO_W]
    P_ambient = forcing_cache_float64_vector[IDX_CACHE_P_AMBIENT]
    T_dew_room = t_dew_point_magnus(T_room, W_room, P_ambient, ALDUCHOV_LIQ_CONST_SET)

    # Coil states (includes frost)
    M_passage = room_state_float64_vector[IDX_ROOM_STATE_M_PASSAGE_ICE_KG]
    M_face = room_state_float64_vector[IDX_ROOM_STATE_M_FACE_ICE_KG]
    V_face = room_state_float64_vector[IDX_ROOM_STATE_V_FACE_ICE_M3]
    T_metal = room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAPORATOR]

    # Throughflow and front area without face ice see passage ice if present; otherwise metal.
    if M_passage > 0.0:
        T_substrate, _, _, = passage_frost_temperatures_and_capacities(
            M_passage,
            room_state_float64_vector[IDX_ROOM_STATE_E_PASSAGE_ICE_J]
        )
    else:
        T_substrate = T_metal

    substrate_is_ice = M_passage > 0.0 or T_substrate < 0.0
    substrate_sat_constants = (
        ALDUCHOV_ICE_CONST_SET if substrate_is_ice else ALDUCHOV_LIQ_CONST_SET
    )

    W_sat_substrate = w_from_partial_pressure(
        p_sat_magnus(T_substrate, substrate_sat_constants),
        P_ambient,
    )

    A_frontal = equipment_configurations_float64_vector[IDX_EQ_EVAP_FRONTAL_AREA_M2]
    A_face_ice_covered = face_frost_covered_area(
        V_face,
        A_frontal,
        equipment_configurations_float64_vector[IDX_EQ_EVAP_INV_FACE_BRIDGE_LENGTH_M],
    )
    A_face_uncovered = A_frontal - A_face_ice_covered

    h_face_w_m2_k = equipment_configurations_float64_vector[IDX_EQ_EVAP_FACE_H_W_M2_K]
    G_face_uncovered = h_face_w_m2_k * A_face_uncovered
    G_face_ice_covered = h_face_w_m2_k * A_face_ice_covered

    m_dot_passage_ice_deposited_throughflow = 0.0
    m_dot_face_ice_deposited_on_uncovered_front = 0.0
    m_dot_face_ice_deposited_on_covered_front = 0.0

    v_dot_passage_ice_deposited_throughflow = 0.0
    v_dot_face_ice_deposited_on_uncovered_front = 0.0
    v_dot_face_ice_deposited_on_covered_front = 0.0
    
    m_dot_passage_ice_sublimation_throughflow_requested = 0.0
    m_dot_passage_ice_sublimation_uncovered_front_requested = 0.0
    m_dot_face_ice_sublimation_covered_front = 0.0
    
    m_dot_condensed_throughflow = 0.0
    m_dot_condensed_uncovered_front = 0.0

    # 1. Air passing through the fin passages
    if m_dot_air_evap > 0.0:
        raw_m_dot_throughflow = m_dot_air_evap * epsilon_evap * (W_room - W_sat_substrate)
        if substrate_is_ice:
            if raw_m_dot_throughflow > 0.0: # Deposition
                m_dot_passage_ice_deposited_throughflow = raw_m_dot_throughflow
                v_dot_passage_ice_deposited_throughflow = (
                    m_dot_passage_ice_deposited_throughflow
                    / frost_deposition_density(T_dew_room, T_substrate, FROST_DENSITY_T_FS_MIN_C)
                )
            elif M_passage > 0.0: # Sublimation
                m_dot_passage_ice_sublimation_throughflow_requested = -raw_m_dot_throughflow
        elif raw_m_dot_throughflow > 0.0:
            m_dot_condensed_throughflow = raw_m_dot_throughflow

    inv_cp_air_lewis_2_3 = equipment_configurations_float64_vector[IDX_EQ_EVAP_INV_CP_AIR_LEWIS_2_3]
    # 2. Front without face ice: new frost joins face ice; exposed passage ice can sublimate.
    if A_face_uncovered > 0.0:
        raw_m_dot_uncovered_front = G_face_uncovered * inv_cp_air_lewis_2_3 * (W_room - W_sat_substrate)
        if substrate_is_ice:
            if raw_m_dot_uncovered_front > 0.0: # Deposition
                m_dot_face_ice_deposited_on_uncovered_front = raw_m_dot_uncovered_front
                v_dot_face_ice_deposited_on_uncovered_front = (
                    m_dot_face_ice_deposited_on_uncovered_front
                    / frost_deposition_density(T_dew_room, T_substrate, FROST_DENSITY_T_FS_MIN_C)
                )
            elif M_passage > 0.0: # Sublimation
                m_dot_passage_ice_sublimation_uncovered_front_requested = -raw_m_dot_uncovered_front
        elif raw_m_dot_uncovered_front > 0.0:
            m_dot_condensed_uncovered_front = raw_m_dot_uncovered_front

    # Both air paths can remove passage ice; scale their requests to its available mass.
    m_dot_passage_ice_sublimation_throughflow = 0.0
    m_dot_passage_ice_sublimation_uncovered_front = 0.0
    m_dot_passage_ice_sublimation_requested_total = m_dot_passage_ice_sublimation_throughflow_requested + m_dot_passage_ice_sublimation_uncovered_front_requested

    if m_dot_passage_ice_sublimation_requested_total > 0.0:
        fraction_available = min(M_passage / (dt_engine_sec * m_dot_passage_ice_sublimation_requested_total), 1.0)
        m_dot_passage_ice_sublimation_throughflow = m_dot_passage_ice_sublimation_throughflow_requested * fraction_available
        m_dot_passage_ice_sublimation_uncovered_front = m_dot_passage_ice_sublimation_uncovered_front_requested * fraction_available

    # 3. Exposed outer surface of existing face ice
    T_face_outer = 0.0  # No face-ice temperature is used if this block is skipped.
    outer_face_frost = M_face > 0.0 and A_face_ice_covered > 0.0
    if outer_face_frost:
        _, T_face_outer, _, _ = face_frost_temperatures_and_capacities(
            M_face,
            room_state_float64_vector[IDX_ROOM_STATE_E_FACE_INNER_ICE_J],
            room_state_float64_vector[IDX_ROOM_STATE_E_FACE_OUTER_ICE_J],
        )
        W_sat_face_ice = w_from_partial_pressure(
            p_sat_magnus(T_face_outer, ALDUCHOV_ICE_CONST_SET),
            P_ambient,
        )
        raw_m_dot_covered_front = G_face_ice_covered * inv_cp_air_lewis_2_3 * (W_room - W_sat_face_ice)
        if raw_m_dot_covered_front > 0.0: # Deposition
            m_dot_face_ice_deposited_on_covered_front = raw_m_dot_covered_front
            v_dot_face_ice_deposited_on_covered_front = (
                m_dot_face_ice_deposited_on_covered_front
                / frost_deposition_density(T_dew_room, T_face_outer, FROST_DENSITY_T_FS_MIN_C)
            )
        else: # Sublimation
            m_dot_face_ice_sublimation_covered_front = min(-raw_m_dot_covered_front, M_face / dt_engine_sec)

    m_dot_water_removed_throughflow = (
        m_dot_passage_ice_deposited_throughflow
        - m_dot_passage_ice_sublimation_throughflow
        + m_dot_condensed_throughflow
    )
    m_dot_water_removed_uncovered_front = (
        m_dot_face_ice_deposited_on_uncovered_front
        - m_dot_passage_ice_sublimation_uncovered_front
        + m_dot_condensed_uncovered_front
    )
    m_dot_water_removed_covered_front = (
        m_dot_face_ice_deposited_on_covered_front
        - m_dot_face_ice_sublimation_covered_front
    )
    m_dot_water_removed_total = (
        m_dot_water_removed_throughflow
        + m_dot_water_removed_uncovered_front
        + m_dot_water_removed_covered_front
    )

    # --- Sensible heat terms ---
    Q_sensible_air_to_passage_surface = 0.0
    if m_dot_air_evap > 0.0:
        Q_sensible_air_to_passage_surface = C_evap * epsilon_evap * (T_room - T_substrate)

    Q_sensible_air_to_uncovered_front = G_face_uncovered * (T_room - T_substrate)

    Q_sensible_air_to_outer_face_ice = 0.0
    if M_face > 0.0 and A_face_ice_covered > 0.0:
        Q_sensible_air_to_outer_face_ice = G_face_ice_covered * (T_room - T_face_outer)


    # --- Latent heat terms ---
    m_dot_ice_net_throughflow = (
        m_dot_passage_ice_deposited_throughflow - m_dot_passage_ice_sublimation_throughflow
    )
    m_dot_ice_net_uncovered_front = (
        m_dot_face_ice_deposited_on_uncovered_front - m_dot_passage_ice_sublimation_uncovered_front
    )
    Q_latent_throughflow = 0.0
    Q_latent_uncovered_front = 0.0
    Q_latent_covered_front = 0.0

    if m_dot_ice_net_throughflow != 0.0 or m_dot_ice_net_uncovered_front != 0.0:
        h_sub = latent_heat_sublimation_j_kg(T_substrate)
        Q_latent_throughflow += m_dot_ice_net_throughflow * h_sub
        Q_latent_uncovered_front += m_dot_ice_net_uncovered_front * h_sub

    if m_dot_condensed_throughflow > 0.0 or m_dot_condensed_uncovered_front > 0.0:
        h_vap = latent_heat_vaporization_j_kg(T_substrate)
        Q_latent_throughflow += m_dot_condensed_throughflow * h_vap
        Q_latent_uncovered_front += m_dot_condensed_uncovered_front * h_vap

    m_dot_ice_net_covered_front = (
        m_dot_face_ice_deposited_on_covered_front - m_dot_face_ice_sublimation_covered_front
    )

    if m_dot_ice_net_covered_front != 0.0:
        Q_latent_covered_front = m_dot_ice_net_covered_front * latent_heat_sublimation_j_kg(T_face_outer)

    dE_air_to_passage_surface_j = (Q_sensible_air_to_passage_surface + Q_latent_throughflow) * dt_engine_sec
    dE_air_to_uncovered_front_j = (Q_sensible_air_to_uncovered_front + Q_latent_uncovered_front) * dt_engine_sec
    Q_air_to_outer_face_ice = Q_sensible_air_to_outer_face_ice + Q_latent_covered_front

    (
        room_state_float64_vector[IDX_ROOM_STATE_T_EVAP_AIR],
        room_state_float64_vector[IDX_ROOM_STATE_RH_EVAP_AIR],
    ) = calculate_evaporator_outlet_state(
        T_room,
        W_room,
        P_ambient,
        Q_sensible_air_to_passage_surface,
        inv_C_evap,
        inv_m_dot_air_evap,
        m_dot_water_removed_throughflow,  # Through-coil moisture only
    ) 

    Q_sensible_air_to_face_total = Q_sensible_air_to_uncovered_front + Q_sensible_air_to_outer_face_ice
    Q_latent_total = Q_latent_throughflow + Q_latent_uncovered_front + Q_latent_covered_front
    m_dot_passage_ice_sublimation_total = (
        m_dot_passage_ice_sublimation_throughflow
        + m_dot_passage_ice_sublimation_uncovered_front
    )
    m_dot_condensed_total = m_dot_condensed_throughflow + m_dot_condensed_uncovered_front
    v_dot_face_ice_deposited_total = (
        v_dot_face_ice_deposited_on_uncovered_front
        + v_dot_face_ice_deposited_on_covered_front
    )

    return (
        Q_sensible_air_to_passage_surface,
        Q_sensible_air_to_face_total,
        Q_latent_total,
        dE_air_to_passage_surface_j,
        dE_air_to_uncovered_front_j,
        Q_air_to_outer_face_ice,
        m_dot_passage_ice_deposited_throughflow,
        m_dot_passage_ice_sublimation_total,
        m_dot_face_ice_deposited_on_uncovered_front,
        m_dot_face_ice_deposited_on_covered_front,
        m_dot_face_ice_sublimation_covered_front,
        m_dot_condensed_total,
        m_dot_water_removed_total,
        T_dew_room,
        v_dot_passage_ice_deposited_throughflow,
        v_dot_face_ice_deposited_total,
        T_substrate,
        T_face_outer, 
    ) 

@njit
def update_gas_balance(room_state_float64_vector, room_configurations_float64_vector, active_crop_params_float64_vector, CO2_prod_ppm_sec_bulk, f_exchange_sec, dt_engine_sec):
    CO2_ppm = room_state_float64_vector[IDX_ROOM_STATE_CO2_PPM]
    O2_pct = room_state_float64_vector[IDX_ROOM_STATE_O2_PCT]

    dCO2_ppm = (
        CO2_prod_ppm_sec_bulk
        + f_exchange_sec * (room_configurations_float64_vector[IDX_ROOM_OUTDOORS_CO2_PPM] - CO2_ppm)
    ) * dt_engine_sec
    room_state_float64_vector[IDX_ROOM_STATE_CO2_PPM] = max(CO2_ppm + dCO2_ppm, 0.0)

    O2_consumption_pct_sec_bulk = (CO2_prod_ppm_sec_bulk * 1e-4) * active_crop_params_float64_vector[IDX_FRUIT_PARAM_INV_RQ_O2_CONS_CO2_PROD]
    dO2_pct = (
        - O2_consumption_pct_sec_bulk 
        + f_exchange_sec * (room_configurations_float64_vector[IDX_ROOM_OUTDOORS_O2_PCT] - O2_pct)
    ) * dt_engine_sec
    room_state_float64_vector[IDX_ROOM_STATE_O2_PCT] = max(O2_pct + dO2_pct, 0.0)


@njit
def update_moisture_and_fruit_mass(room_state_float64_vector, door_states_float64_matrix, forcing_cache_float64_vector, m_dot_leak, m_dot_transpiration_bulk, m_dot_humidifier, m_dot_water_removed, inv_m_air_room, dt_engine_sec):
    W_room = room_state_float64_vector[IDX_ROOM_STATE_AIR_HUMIDITY_RATIO_W]
    W_plant = room_state_float64_vector[IDX_ROOM_STATE_ADJACENT_PLANT_AIR_HUMIDITY_RATIO_W]
    W_ambient = forcing_cache_float64_vector[IDX_CACHE_W_AMBIENT]
    fruit_mass_bulk_kg = room_state_float64_vector[IDX_ROOM_STATE_BULK_PULP_MASS]
    m_dot_door_int = door_states_float64_matrix[DOOR_CHANNEL_INTERNAL, IDX_DOOR_STATE_M_DOT_INFLOW]
    m_dot_door_ext = door_states_float64_matrix[DOOR_CHANNEL_EXTERNAL, IDX_DOOR_STATE_M_DOT_INFLOW]
    dW_room = (
        (m_dot_leak + m_dot_door_ext) * (W_ambient - W_room)
        + m_dot_door_int * (W_plant - W_room)
        + m_dot_transpiration_bulk
        + m_dot_humidifier
        - m_dot_water_removed
    ) * inv_m_air_room * dt_engine_sec
    room_state_float64_vector[IDX_ROOM_STATE_AIR_HUMIDITY_RATIO_W] = max(W_room + dW_room, 0.0)
    room_state_float64_vector[IDX_ROOM_STATE_BULK_PULP_MASS] = max(fruit_mass_bulk_kg - m_dot_transpiration_bulk * dt_engine_sec, 0.0)

@njit
def update_room_and_coil_psychrometrics(
    room_state_float64_vector,
    forcing_cache_float64_vector,
):
    P_ambient = forcing_cache_float64_vector[IDX_CACHE_P_AMBIENT]

    # --- Next Step ---
    T_room_updated = room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C]
    W_room_updated = room_state_float64_vector[IDX_ROOM_STATE_AIR_HUMIDITY_RATIO_W]
    T_coil_evap_updated = room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAPORATOR]
    M_passage_updated = room_state_float64_vector[IDX_ROOM_STATE_M_PASSAGE_ICE_KG]

    P_w_room = partial_pressure_from_w(W_room_updated, P_ambient)
    P_sat_room = p_sat_magnus(T_room_updated, ALDUCHOV_LIQ_CONST_SET)
    RH_room = numba_clip(P_w_room / P_sat_room, 0.0, 1.0)

    if M_passage_updated > 0.0:
        T_surface, _, _ = passage_frost_temperatures_and_capacities(
            M_passage_updated,
            room_state_float64_vector[IDX_ROOM_STATE_E_PASSAGE_ICE_J],
        )
        is_ice = True
    else:
        T_surface = T_coil_evap_updated
        is_ice = T_surface < 0.0
    P_sat_coil = p_sat_magnus(T_surface, ALDUCHOV_ICE_CONST_SET if is_ice else ALDUCHOV_LIQ_CONST_SET)
    W_coil_sat = w_from_partial_pressure(P_sat_coil, P_ambient)

    room_state_float64_vector[IDX_ROOM_STATE_P_W_ROOM] = P_w_room
    room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM] = RH_room
    room_state_float64_vector[IDX_ROOM_STATE_W_COIL_EVAP_SAT] = W_coil_sat
