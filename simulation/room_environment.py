"""
room_environment.py

Ambient forcing cache refresh plus room thermal balance: envelope/floor/infiltration heat loads, room/floor/fruit-bulk/adjacent-plant temperature updates.
"""

import math
import numpy as np
from jit import njit

from constants import *
from matrix_index_maps import *
from fruit_biology import calc_respiration_rate_mg_co2_per_kg_fruit_hr
from psychrometrics import door_infiltration_orifice_slot, latent_heat_vaporization_j_kg, p_sat_magnus, w_from_partial_pressure
from sensors_control import get_sensor_filtered_value
from utils import numba_clip, bounded_normal_draw


@njit
def select_magnus_constants(): # PLACEHOLDER
    # NOTE: For now, always use magnus constants for liquid water
    # TODO: Add the option to use ice constants after implementing defrost system
    magnus_const_set = ALDUCHOV_LIQ_CONST_SET
    return magnus_const_set


@njit
def calculate_forcing_data(
    # step, # NOTE: DEBUG
    forcing_matrix,
    local_utc_offset_ms_vector,
    forcing_cache_float64_vector,
    forcing_cache_int64_vector,
    room_state_int64_vector,
    room_state_float64_vector,
    forcing_start_time_ms,
    bucket_interval_ms,
    last_forcing_idx,
    room_configurations_float64_vector,
    equipment_configurations_float64_vector,
):
    # Calculate current forcing row index
    forcing_idx = (room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS] - forcing_start_time_ms) // bucket_interval_ms

    # Only execute when the 1-minute bucket actually rolls over
    if forcing_idx != last_forcing_idx:
        # Extract raw weather fields directly into the cache vector
        forcing_cache_float64_vector[IDX_CACHE_T_AMBIENT]     = forcing_matrix[forcing_idx, IDX_FORCING_AMBIENT_AIR_TEMP_C]
        forcing_cache_float64_vector[IDX_CACHE_P_AMBIENT]     = forcing_matrix[forcing_idx, IDX_FORCING_AMBIENT_SURF_PRESSURE_PA]
        forcing_cache_float64_vector[IDX_CACHE_INV_P_AMBIENT] = forcing_matrix[forcing_idx, IDX_FORCING_AMBIENT_INV_SURF_PRESSURE_PA]
        forcing_cache_float64_vector[IDX_CACHE_RH_AMBIENT]    = forcing_matrix[forcing_idx, IDX_FORCING_AMBIENT_RH_FRAC]
        forcing_cache_float64_vector[IDX_CACHE_GHI_W_M2]      = forcing_matrix[forcing_idx, IDX_FORCING_AMBIENT_IRRADIANCE_W_M2]
        forcing_cache_float64_vector[IDX_CACHE_WIND_SPEED]    = forcing_matrix[forcing_idx, IDX_FORCING_AMBIENT_WIND_SPEED_M_S]
        forcing_cache_float64_vector[IDX_CACHE_T_GROUND]      = forcing_matrix[forcing_idx, IDX_FORCING_AMBIENT_SOIL_TEMP_C]

        forcing_cache_int64_vector[IDX_CACHE_LOCAL_UTC_MS]    = local_utc_offset_ms_vector[forcing_idx]

        # Use clean local variables for intermediate physics math
        T_ambient  = forcing_cache_float64_vector[IDX_CACHE_T_AMBIENT]
        P_ambient  = forcing_cache_float64_vector[IDX_CACHE_P_AMBIENT]
        RH_ambient = forcing_cache_float64_vector[IDX_CACHE_RH_AMBIENT]
        ghi_w_m2   = forcing_cache_float64_vector[IDX_CACHE_GHI_W_M2]
        wind_speed = forcing_cache_float64_vector[IDX_CACHE_WIND_SPEED]

        # Ambient psychrometrics
        P_sat_ambient = p_sat_magnus(T_ambient, ALDUCHOV_LIQ_CONST_SET)
        P_w_ambient = RH_ambient * P_sat_ambient
        
        W_ambient = w_from_partial_pressure(P_w_ambient, P_ambient)
        forcing_cache_float64_vector[IDX_CACHE_W_AMBIENT] = W_ambient

        # print(
        #     "W_ambient from cache vector:", forcing_cache_float64_vector[IDX_CACHE_W_AMBIENT], "\n",
        #     "P_sat_ambient:", P_sat_ambient, "\n",
        #     "P_w_ambient:", P_w_ambient, "\n",
        #     "W_ambient from in-line caculation", w_from_partial_pressure(P_w_ambient, P_ambient), "\n",
        # )

        # Moist-air gas constant & density
        R_moist_ambient = R_DRY_AIR * (1.0 + MOISTURE_BUOYANCY_FACTOR * W_ambient)
        rho_air_ambient = P_ambient / (R_moist_ambient * (T_ambient + 273.15))

        # Ambient-dependent heat envelopes
        inv_h_o = 1.0 / (5.7 + 3.8 * wind_speed)
        forcing_cache_float64_vector[IDX_CACHE_T_SOL_AIR] = T_ambient + (room_configurations_float64_vector[IDX_ROOM_THERMAL_ALPHA_ROOF] * ghi_w_m2) * inv_h_o

        U_wall_ext_eff = 1.0 / (inv_h_o + room_configurations_float64_vector[IDX_ROOM_THERMAL_EXT_WALL_ENV_CONSTANT])
        U_roof_eff = 1.0 / (inv_h_o + room_configurations_float64_vector[IDX_ROOM_THERMAL_ROOF_ENV_CONSTANT])

        forcing_cache_float64_vector[IDX_CACHE_UA_WALLS_EXT] = U_wall_ext_eff * room_configurations_float64_vector[IDX_ROOM_EXTERNAL_WALL_AREA_M2]
        forcing_cache_float64_vector[IDX_CACHE_UA_ROOF]      = U_roof_eff * room_configurations_float64_vector[IDX_ROOM_ROOF_AREA_M2]

        forcing_cache_float64_vector[IDX_CACHE_F_WIND_PENALTY] = max(1.0 - 0.015 * wind_speed, 0.85)

        # --- Condenser hardware constants ---
        # Q_cond_rated = room_state_float64_vector[IDX_ROOM_STATE_RATED_Q_COOLING] + room_state_float64_vector[IDX_ROOM_STATE_RATED_W_COMPRESSOR]
        V_dot_air_cond_fully_on = equipment_configurations_float64_vector[IDX_EQ_COND_NOMINAL_AIRFLOW_M3_S] * forcing_cache_float64_vector[IDX_CACHE_F_WIND_PENALTY]
        td_design_cond = equipment_configurations_float64_vector[IDX_EQ_DESIGN_COND_TD_K]
        # #### DEBUG ####
        # print(
        #     "Function:", "calculate_forcing_data",
        #     "V_dot_evap:", room_state_float64_vector[IDX_ROOM_STATE_V_DOT_AIR_COND_ACTUAL], "\n",
        #     "rho_air_ambient:", rho_air_ambient, "\n",
        #     "Q_cond_rated:", Q_cond_rated, "\n",
        #     "td_design_cond:", td_design_cond, "\n",
        #     "V_dot_air_cond_fully_on:", V_dot_air_cond_fully_on, "\n",
        #     "Q_cooling_rated:", room_state_float64_vector[IDX_ROOM_STATE_RATED_Q_COOLING], "\n",
        #     "W_compressor_rated:", room_state_float64_vector[IDX_ROOM_STATE_RATED_W_COMPRESSOR], "\n",
        #     "V_dot_cond_nominal:", equipment_configurations_float64_vector[IDX_EQ_COND_NOMINAL_AIRFLOW_M3_S], "\n",
        #     "f_wind_penalty:", forcing_cache_float64_vector[IDX_CACHE_F_WIND_PENALTY], "\n",
        #     "f_wind_penalty in-line calc:", max(1.0 - 0.015 * forcing_cache_float64_vector[IDX_CACHE_WIND_SPEED], 0.85), "\n", 
        #     "wind_speed:", wind_speed, "\n",
        #     "wind_speed from cache:", forcing_cache_float64_vector[IDX_CACHE_WIND_SPEED], "\n",
        #     "last_forcing_idx:", last_forcing_idx, "\n",
        #     "forcing_idx:", forcing_idx, "\n",
        #     "epoch_timestamp_ms:", room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS], "\n",
        #     "forcing_start_time_ms:", forcing_start_time_ms, "\n",
        #     "bucket_interval_ms:", bucket_interval_ms, "\n",
        #     "T_ambient:" ,forcing_cache_float64_vector[IDX_CACHE_T_AMBIENT], "\n",
        #     "this step:", step, 2*"\n",
        # )
        # #### DEBUG ####

        # UA_coil_cond = calculate_ua_coil(                                                          
        #     room_state_float64_vector[IDX_ROOM_STATE_V_DOT_AIR_COND_ACTUAL], 
        #     rho_air_ambient,
        #     Q_cond_rated,
        #     td_design_cond,                                                                                                                                                                     
        # )

        # UA_coil_cond_floor = 0.005 * calculate_ua_coil(  
        #     V_dot_air_cond_fully_on,  
        #     rho_air_ambient,  
        #     Q_cond_rated,  
        #     td_design_cond,  
        # )

        # UA_coil_cond = max(UA_coil_cond, UA_coil_cond_floor)
        # t_ambient_design = equipment_configurations_float64_vector[IDX_EQ_DESIGN_AMBIENT_TEMP_C]
        # rho_air_cond_design = P_ATM_STANDARD / (R_DRY_AIR * (t_ambient_design + 273.15))
        # UA_coil_cond = calculate_ua_coil(                                                          
        #     V_dot_air_cond_fully_on, 
        #     equipment_configurations_float64_vector[IDX_EQ_DESIGN_RHO_AIR_COND],
        #     Q_cond_rated,
        #     td_design_cond,
        # )
        # UA_internal_cond = UA_coil_cond * equipment_configurations_float64_vector[IDX_EQ_COND_UA_INTERNAL_SCALE]

        forcing_cache_float64_vector[IDX_CACHE_V_DOT_COND_FULLY_ON] = V_dot_air_cond_fully_on
        # forcing_cache_float64_vector[IDX_CACHE_UA_COIL_COND] = UA_coil_cond
        # forcing_cache_float64_vector[IDX_CACHE_UA_INTERNAL_COND] = UA_internal_cond

        forcing_cache_float64_vector[IDX_CACHE_RHO_AIR_AMBIENT] = rho_air_ambient

        last_forcing_idx = forcing_idx

    # Return ONLY the tracker index. Cache vector modifications persist automatically.
    return last_forcing_idx


@njit
def calculate_air_properties(
    room_state_float64_vector,
    forcing_cache_float64_vector,
    # equipment_configurations_float64_vector,
    # magnus_const_set,
):
    # moist-air gas constants
    R_moist_room = R_DRY_AIR * (1 + MOISTURE_BUOYANCY_FACTOR * room_state_float64_vector[IDX_ROOM_STATE_AIR_HUMIDITY_RATIO_W])

    # moist-air gas densities
    rho_air_room = forcing_cache_float64_vector[IDX_CACHE_P_AMBIENT] / (R_moist_room * (room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C] + 273.15))

    # Q_evap_rated = room_state_float64_vector[IDX_ROOM_STATE_RATED_Q_COOLING]
    # td_design_evap = equipment_configurations_float64_vector[IDX_EQ_DESIGN_EVAP_TD_K]
    # V_dot_air_evap_fully_on = room_state_float64_vector[IDX_ROOM_STATE_V_DOT_EVAP_FULLY_ON]

    # ##### DEBUG ######
    # print(
    #     "function:", "calculate_air_properties",
    #     "V_dot_evap:", room_state_float64_vector[IDX_ROOM_STATE_V_DOT_AIR_EVAP_ACTUAL],
    #     "rho_air_room:", rho_air_room,
    #     "Q_evap_rated:", room_state_float64_vector[IDX_ROOM_STATE_RATED_Q_COOLING],
    #     "td_design_evap:", td_design_evap,
    #     "V_dot_air_evap_fully_on", V_dot_air_evap_fully_on
    # )
    # ##### DEBUG ######

    # UA_coil_evap = calculate_ua_coil(                                                     
    #     room_state_float64_vector[IDX_ROOM_STATE_V_DOT_AIR_EVAP_ACTUAL],                                                             
    #     rho_air_room,
    #     Q_evap_rated,                                                                      
    #     td_design_evap,                                                                    
    # )

    # UA_coil_evap_floor = 0.005 * calculate_ua_coil(
    #     V_dot_air_evap_fully_on,
    #     rho_air_room,
    #     Q_evap_rated,
    #     td_design_evap,
    # )

    # UA_coil_evap = max(UA_coil_evap, UA_coil_evap_floor)
    # UA_internal_evap = UA_coil_evap * equipment_configurations_float64_vector[IDX_EQ_EVAP_UA_INTERNAL_SCALE]

    return rho_air_room #  UA_coil_evap, UA_internal_evap


@njit
def calculate_envelope_heat_loads(
    room_state_float64_vector,
    forcing_cache_float64_vector,
    room_configurations_float64_vector,
):
    T_room = room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C]
    T_plant = room_state_float64_vector[IDX_ROOM_STATE_ADJACENT_PLANT_TEMP_C]

    # alpha_roof = room_configurations_float64_vector[IDX_ROOM_THERMAL_ALPHA_ROOF]
    # h_i_walls = room_configurations_float64_vector[IDX_ROOM_THERMAL_H_I_WALLS]
    # h_i_roof = room_configurations_float64_vector[IDX_ROOM_THERMAL_H_I_ROOF]
    # U_wall = room_configurations_float64_vector[IDX_ROOM_THERMAL_U_WALL]

    # A_wall_internal = room_configurations_float64_vector[IDX_ROOM_INTERNAL_WALL_AREA_M2]
    # A_walls_external = room_configurations_float64_vector[IDX_ROOM_EXTERNAL_WALL_AREA_M2]
    # A_roof = room_configurations_float64_vector[IDX_ROOM_ROOF_AREA_M2]

    # h_o = 5.7 + 3.8 * wind_speed
    # T_sol_air = T_ambient + (alpha_roof * ghi_w_m2) / h_o

    # U_wall_ext_eff = 1.0 / (inv_h_o + room_configurations_float64_vector[IDX_ROOM_THERMAL_EXT_WALL_ENV_CONSTANT])
    # U_roof_eff = 1.0 / (inv_h_o + room_configurations_float64_vector[IDX_ROOM_THERMAL_ROOF_ENV_CONSTANT])

    Q_walls_ext = forcing_cache_float64_vector[IDX_CACHE_UA_WALLS_EXT] * (forcing_cache_float64_vector[IDX_CACHE_T_AMBIENT] - T_room)
    Q_roof = forcing_cache_float64_vector[IDX_CACHE_UA_ROOF] * (forcing_cache_float64_vector[IDX_CACHE_T_SOL_AIR] - T_room)

    # U_wall_int_eff = 1.0 / (1.0/h_i_walls + 1.0/U_wall + 1.0/h_i_walls)
    Q_walls_int = room_configurations_float64_vector[IDX_ROOM_THERMAL_UA_WALLS_INT] * (T_plant - T_room)

    return Q_walls_ext, Q_walls_int, Q_roof


@njit
def calculate_floor_heat_terms(
    room_state_float64_vector,
    forcing_cache_float64_vector,
    room_configurations_float64_vector,
):
    T_room = room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C]
    T_floor = room_state_float64_vector[IDX_ROOM_STATE_FLOOR_SLAB_TEMP_C]

    # A_floor = room_configurations_float64_vector[IDX_ROOM_FLOOR_AREA_M2]
    # h_i_floor = room_configurations_float64_vector[IDX_ROOM_THERMAL_H_I_FLOOR]
    # U_ground = room_configurations_float64_vector[IDX_ROOM_THERMAL_U_FLOOR]
    
    Q_room_to_floor = room_configurations_float64_vector[IDX_ROOM_THERMAL_UA_ROOM_TO_FLOOR] * (T_room - T_floor)
    Q_floor_to_ground = room_configurations_float64_vector[IDX_ROOM_THERMAL_UA_FLOOR_TO_GROUND] * (T_floor - forcing_cache_float64_vector[IDX_CACHE_T_GROUND])

    return Q_room_to_floor, Q_floor_to_ground

@njit
def get_door_state(
    epoch_timestamp,
    open_milestone_timestamp,
    close_milestone_timestamp,
):
    if open_milestone_timestamp <= epoch_timestamp < close_milestone_timestamp:
        is_door_open = 1
    else:
        is_door_open = 0
    return is_door_open

@njit
def calculate_infiltration_terms(
    rng,
    room_state_int64_vector,
    room_state_float64_vector,
    door_states_float64_matrix,
    door_states_int64_matrix,
    room_ephemeral_int64_vector,
    forcing_cache_float64_vector,
    door_event_ledger,
    room_configurations_float64_vector,
    room_configurations_int64_vector,
    door_specs_float64_matrix,
    rho_air_room,
    m_air_room,
    inv_m_air_room,
    n_ach_actual_per_sec,
    dt_engine_sec,
):
    epoch_timestamp_ms = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]
    rho_air_plant = room_state_float64_vector[IDX_ROOM_STATE_ADJACENT_PLANT_RHO_AIR]
    T_ambient = forcing_cache_float64_vector[IDX_CACHE_T_AMBIENT]
    rho_air_ambient = forcing_cache_float64_vector[IDX_CACHE_RHO_AIR_AMBIENT]
    for door_code in range(door_states_int64_matrix.shape[0]):
        wind_speed = forcing_cache_float64_vector[IDX_CACHE_WIND_SPEED]
        is_door_open = get_door_state(
            epoch_timestamp_ms,
            door_states_int64_matrix[door_code, IDX_DOOR_STATE_OPEN_MS],
            door_states_int64_matrix[door_code, IDX_DOOR_STATE_CLOSE_MS],
        )

        # Write door_event_ledger
        if is_door_open != door_states_int64_matrix[door_code, IDX_DOOR_STATE_IS_OPEN]:
            row = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOOR_EVENT_WRITE_IDX]
            door_event_ledger[row, IDX_DOOR_EVENT_TIMESTAMP_MS] = epoch_timestamp_ms
            door_event_ledger[row, IDX_DOOR_EVENT_DOOR_CODE] = door_code
            door_event_ledger[row, IDX_DOOR_EVENT_IS_OPEN] = is_door_open
            room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOOR_EVENT_WRITE_IDX] = row + 1

        current_door_open_fraction = door_states_float64_matrix[door_code, IDX_DOOR_STATE_OPEN_FRACTION]

        target_door_open_fraction = np.float64(is_door_open)
        updated_door_open_fraction = (
            current_door_open_fraction + 
            (target_door_open_fraction - current_door_open_fraction) * door_specs_float64_matrix[door_code, IDX_FACILITY_DT_OVER_TAU_DOOR_SEC]
        )
        updated_door_open_fraction = numba_clip(updated_door_open_fraction, SAFE_FLOOR, 1.0)

        if updated_door_open_fraction > 1e-5:
            H_open_fraction = door_specs_float64_matrix[door_code, IDX_FACILITY_DOOR_HEIGHT] * updated_door_open_fraction
            is_external_door = (door_code == DOOR_CHANNEL_EXTERNAL)
            rho_air_source = (is_external_door * rho_air_ambient) + ((1.0 - is_external_door) * rho_air_plant)
            wind_speed *= is_external_door # zero for internal door
            door_states_float64_matrix[door_code, IDX_DOOR_STATE_M_DOT_INFLOW] = door_infiltration_orifice_slot(
                    H_open_fraction,
                    door_specs_float64_matrix[door_code, IDX_FACILITY_DOOR_WIDTH],
                    rho_air_source,
                    rho_air_room,
                    wind_speed,
                    door_specs_float64_matrix[door_code, IDX_FACILITY_C_WIND_CAPTURE],
                    door_specs_float64_matrix[door_code, IDX_FACILITY_K_STRIP_BUOYANCY],
                    door_specs_float64_matrix[door_code, IDX_FACILITY_C_RECT_DISCHARGE],
            )
        else:
            door_states_float64_matrix[door_code, IDX_DOOR_STATE_M_DOT_INFLOW] = 0.0

        door_states_float64_matrix[door_code, IDX_DOOR_STATE_OPEN_FRACTION] = updated_door_open_fraction
        door_states_int64_matrix[door_code, IDX_DOOR_STATE_IS_OPEN] = is_door_open

    T_room = room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C]
    T_plant = room_state_float64_vector[IDX_ROOM_STATE_ADJACENT_PLANT_TEMP_C]
    door_int_open_fraction = door_states_float64_matrix[DOOR_CHANNEL_INTERNAL, IDX_DOOR_STATE_OPEN_FRACTION]
    door_ext_open_fraction = door_states_float64_matrix[DOOR_CHANNEL_EXTERNAL, IDX_DOOR_STATE_OPEN_FRACTION]
    m_dot_door_int = door_states_float64_matrix[DOOR_CHANNEL_INTERNAL, IDX_DOOR_STATE_M_DOT_INFLOW]
    m_dot_door_ext = door_states_float64_matrix[DOOR_CHANNEL_EXTERNAL, IDX_DOOR_STATE_M_DOT_INFLOW]

    m_dot_leak = n_ach_actual_per_sec * m_air_room

    if door_int_open_fraction > 1e-5 and door_ext_open_fraction > 1e-5:
        penalty_scale = max(m_dot_door_int, m_dot_door_ext)
        cross_flow_factor = 1.0 + (room_configurations_float64_vector[IDX_ROOM_LOGISTICS_TD_COEFF] * penalty_scale)
        m_dot_total = m_dot_leak + (m_dot_door_int + m_dot_door_ext) * cross_flow_factor
    elif door_int_open_fraction <= 1e-5 and door_ext_open_fraction <= 1e-5:
        if epoch_timestamp_ms % room_configurations_int64_vector[IDX_ROOM_CFG_LEAK_NOISE_FACTOR_RESAMPLE_INTERVAL_MS] == 0:
            room_state_float64_vector[IDX_ROOM_STATE_LEAK_NOISE_FACTOR] = bounded_normal_draw(
                rng   = rng,
                mean  = 1.0,
                scale = 0.1,
                low   = 0.85,
                high  = 1.15,
            )
        m_dot_total = m_dot_leak * room_state_float64_vector[IDX_ROOM_STATE_LEAK_NOISE_FACTOR] + m_dot_door_ext + m_dot_door_int
    else:
        m_dot_total = m_dot_leak + m_dot_door_ext + m_dot_door_int

    f_exchange_sec = m_dot_total * inv_m_air_room

    Q_inf_sens_ext = (m_dot_leak + m_dot_door_ext) * CP_AIR * (T_ambient - T_room)
    Q_inf_sens_int = m_dot_door_int * CP_AIR * (T_plant - T_room)

    return Q_inf_sens_int, Q_inf_sens_ext, f_exchange_sec, m_dot_leak


@njit
def update_floor_temperature(room_state_float64_vector, room_configurations_float64_vector, Q_room_to_floor, Q_floor_to_ground, dt_engine_sec):
    dT_floor = (
        Q_room_to_floor 
        - Q_floor_to_ground
    ) * room_configurations_float64_vector[IDX_ROOM_DT_OVER_FLOOR_THERMAL_MASS]
    room_state_float64_vector[IDX_ROOM_STATE_FLOOR_SLAB_TEMP_C] += dT_floor


@njit
def update_room_temperature(room_state_float64_vector, Q_walls_ext, Q_walls_int, Q_roof, Q_room_to_floor, Q_fans, Q_room_exchange_bulk, Q_inf_sens_ext, Q_inf_sens_int, Q_sensible_forced, Q_sensible_face, Q_humidifier_cooling, Q_pan_to_room_w, C_room, dt_engine_sec):
    dT_room = (
        Q_walls_ext + Q_walls_int + Q_roof
        - Q_room_to_floor
        + Q_fans
        + Q_pan_to_room_w
        + Q_room_exchange_bulk
        + Q_inf_sens_ext + Q_inf_sens_int
        + room_state_float64_vector[IDX_ROOM_STATE_BULK_ACTIVE_DRIVERS_CONVECTIVE_Q]
        - Q_sensible_forced
        - Q_sensible_face
        - Q_humidifier_cooling
    ) / max(C_room, 1e-4) * dt_engine_sec
    room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C] += dT_room
    return dT_room

@njit
def update_fruit_bulk_temperature(room_state_float64_vector, C_composite_fruit_bulk, Q_batch_to_bulk_exchange, dt_engine_sec):
    if room_state_float64_vector[IDX_ROOM_STATE_BULK_PULP_MASS] <= MASS_KG_FLOOR:
        room_state_float64_vector[IDX_ROOM_STATE_BULK_PULP_TEMP_C] = room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C]
    else:
        dT_pulp_bulk = Q_batch_to_bulk_exchange / max(C_composite_fruit_bulk, 1e-4) * dt_engine_sec
        room_state_float64_vector[IDX_ROOM_STATE_BULK_PULP_TEMP_C] += dT_pulp_bulk


@njit
def update_adjacent_plant_states(
    room_state_int64_vector,
    room_state_float64_vector,
    forcing_cache_float64_vector,
    room_configurations_float64_vector,
    dt_plant_ms,
    dt_plant_sec,
    # magnus_const_set,
):
    if room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS] % dt_plant_ms == 0:
        # ==============================================================================
        # TEMPERATURE PROGRESSION PATH
        # ==============================================================================
        T_plant   = room_state_float64_vector[IDX_ROOM_STATE_ADJACENT_PLANT_TEMP_C]
        T_plant_a = room_configurations_float64_vector[IDX_ROOM_PLANT_TEMP_COEFF_A]
        T_plant_b = room_configurations_float64_vector[IDX_ROOM_PLANT_TEMP_COEFF_B]
        T_ambient = forcing_cache_float64_vector[IDX_CACHE_T_AMBIENT]

        T_plant_target = T_plant_a + T_plant_b * T_ambient
        T_plant += room_configurations_float64_vector[IDX_ROOM_PLANT_DT_OVER_TAU_THERMAL] * (T_plant_target - T_plant)

        # ==============================================================================
        # HUMIDITY PROGRESSION PATH
        # ==============================================================================
        RH_plant   = room_state_float64_vector[IDX_ROOM_STATE_ADJACENT_PLANT_RH]
        RH_plant_a = room_configurations_float64_vector[IDX_ROOM_PLANT_RH_COEFF_A]
        RH_plant_b = room_configurations_float64_vector[IDX_ROOM_PLANT_RH_COEFF_B]
        RH_ambient = forcing_cache_float64_vector[IDX_CACHE_RH_AMBIENT]
        P_ambient  = forcing_cache_float64_vector[IDX_CACHE_P_AMBIENT]

        RH_plant_target = RH_plant_a + RH_plant_b * RH_ambient
        next_rh_plant   = RH_plant + room_configurations_float64_vector[IDX_ROOM_PLANT_DT_OVER_TAU_VAPOR] * (RH_plant_target - RH_plant)
        RH_plant = numba_clip(next_rh_plant, 0.0, 1.0)

        # Plant psychrometrics 
        P_sat_plant = p_sat_magnus(T_plant, ALDUCHOV_LIQ_CONST_SET)
        P_w_plant = RH_plant * P_sat_plant
        W_plant = w_from_partial_pressure(P_w_plant, P_ambient)
        R_moist_plant = R_DRY_AIR * (1 + MOISTURE_BUOYANCY_FACTOR * W_plant)
        rho_air_plant = P_ambient / (R_moist_plant * (T_plant + 273.15))

        # Update state vector
        room_state_float64_vector[IDX_ROOM_STATE_ADJACENT_PLANT_TEMP_C] = T_plant
        room_state_float64_vector[IDX_ROOM_STATE_ADJACENT_PLANT_RH] = RH_plant
        room_state_float64_vector[IDX_ROOM_STATE_ADJACENT_PLANT_AIR_HUMIDITY_RATIO_W] = W_plant
        room_state_float64_vector[IDX_ROOM_STATE_ADJACENT_PLANT_RHO_AIR] = rho_air_plant


@njit
def calculate_room_capacity_terms(
    room_state_float64_vector,
    room_configurations_float64_vector,
    active_crop_params_float64_vector,
    rho_air_room,
):
    V_room_m3 = room_configurations_float64_vector[IDX_ROOM_GEOM_VOLUME]
    V_load_eff_m3 = (
        room_state_float64_vector[IDX_ROOM_STATE_BULK_BOX_PACKAGING_MASS]
        + room_state_float64_vector[IDX_ROOM_STATE_BULK_PULP_MASS]
    ) / active_crop_params_float64_vector[IDX_FRUIT_PARAM_RHO_LOAD_BULK]
    f_free = numba_clip(1.0 - V_load_eff_m3 / V_room_m3, 0.05, 1.0)
    V_free_m3 = f_free * V_room_m3
            
     # effective room air capacity
    C_air = V_free_m3 * rho_air_room * CP_AIR
    C_room = C_air + room_configurations_float64_vector[IDX_ROOM_PANELS_THERMAL_MASS]
    m_air_room = max(V_free_m3 * rho_air_room, 0.05)
    inv_m_air_room = 1 / m_air_room
    # inv_V_free_m3 = V_free_m3
    
    return V_free_m3, f_free, C_room, m_air_room, inv_m_air_room


@njit
def calculate_fruit_heat_terms(
    rng,
    room_state_float64_vector,
    room_state_int64_vector,
    forcing_cache_float64_vector,
    pallet_physics_float64_matrix,
    pallet_metadata_int64_matrix,
    outbound_wave_queue_matrix,
    box_flags_ledger,
    box_masses_ledger,
    partial_dispatch_box_temperatures_ledger,
    airflow_pallets_compact_vector,
    room_ephemeral_int64_vector,
    batch_fruit_sensor_ledger,

    room_configurations_float64_vector,
    equipment_configurations_float64_vector,
    equipment_configurations_int64_vector,
    active_crop_params_float64_vector,
    active_crop_params_int64_vector,
    active_crop_respiration_curves_matrix,

    V_free_m3,

    dt_engine_sec,
    dt_engine_ms,
):
    Q_room_exchange_bulk = 0.0
    Q_batch_to_bulk_exchange = 0.0
    Q_respiration_bulk = 0.0
    Q_evap_fruit_bulk = 0.0
    C_composite_fruit_bulk = 0.0
    fruit_mass_bulk_kg = 0.0
    water_loss_bulk_kg = 0.0
    CO2_prod_ppm_sec_bulk = 0.0
    m_dot_transpiration_bulk = 0.0
    if room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_AIRFLOW] > 0:
        # Unpack fruit-dependent parameters
        h_respiration_j_kg    = active_crop_params_float64_vector[IDX_FRUIT_PARAM_RESP_HEAT_J_KG_CO2] # NOTE: should this be fruit dependent? Or is it more of a universal constant units are energy(J)/mass(kg)
        K_m_O2                = active_crop_params_float64_vector[IDX_FRUIT_PARAM_KM_O2]
        K_i_CO2               = active_crop_params_float64_vector[IDX_FRUIT_PARAM_KI_CO2]
        movement_multiplier   = active_crop_params_float64_vector[IDX_FRUIT_PARAM_K_ZONE_MOV_MULTIPLIER] # NOTE: k_zone movement multiplier, ask Claude for reasonable values
        k_p                   = active_crop_params_float64_vector[IDX_FRUIT_PARAM_TRANSP_K_P]
        k_bulk_batch_coupling = active_crop_params_float64_vector[IDX_FRUIT_PARAM_K_BULK_BATCH_COUPLING] # NOTE: ask Claude for value
        Cp_fruit              = active_crop_params_float64_vector[IDX_FRUIT_PARAM_CP_FRUIT]
        Cp_packaging          = active_crop_params_float64_vector[IDX_FRUIT_PARAM_CP_PACKAGING]
        Cp_pallet             = active_crop_params_float64_vector[IDX_FRUIT_PARAM_CP_PALLET]

        n_resp = active_crop_params_int64_vector[IDX_FRUIT_PARAM_RESP_CURVE_TRUE_LEN]
        T_vector = active_crop_respiration_curves_matrix[:n_resp, IDX_FRUIT_RESP_CURVE_T]
        R_vector = active_crop_respiration_curves_matrix[:n_resp, IDX_FRUIT_RESP_CURVE_R]
        T_min    = active_crop_params_float64_vector[IDX_FRUIT_PARAM_RESP_T_MIN]
        T_max    = active_crop_params_float64_vector[IDX_FRUIT_PARAM_RESP_T_MAX]
        T_lo     = active_crop_params_float64_vector[IDX_FRUIT_PARAM_RESP_T_LO]
        T_hi     = active_crop_params_float64_vector[IDX_FRUIT_PARAM_RESP_T_HI]
        T_ref    = active_crop_params_float64_vector[IDX_FRUIT_PARAM_RESP_T_REF]
        R_ref    = active_crop_params_float64_vector[IDX_FRUIT_PARAM_RESP_R_REF]
        Q_10     = active_crop_params_float64_vector[IDX_FRUIT_PARAM_RESP_Q10]

        # Unpatch bulk state variables
        T_room = room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C]
        P_w_room = room_state_float64_vector[IDX_ROOM_STATE_P_W_ROOM]
        O2_pct = room_state_float64_vector[IDX_ROOM_STATE_O2_PCT]
        CO2_ppm = room_state_float64_vector[IDX_ROOM_STATE_CO2_PPM]
        T_pulp_bulk = room_state_float64_vector[IDX_ROOM_STATE_BULK_PULP_TEMP_C]

        inv_P_ambient = forcing_cache_float64_vector[IDX_CACHE_INV_P_AMBIENT]

        epoch_timestamp_ms = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]
        sample_t_pulp_sensor = (epoch_timestamp_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_T_PULP_INTERVAL_MS] == 0)
        t_pulp_sensor_alpha       = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_PULP_ALPHA]
        t_pulp_sensor_noise_scale = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_PULP_NOISE_SCALE]
        t_pulp_sensor_lower_limit = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_PULP_LOWER_LIMIT]
        t_pulp_sensor_upper_limit = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_PULP_UPPER_LIMIT]

        load_cell_sensor_alpha       = equipment_configurations_float64_vector[IDX_EQ_SENSOR_LOAD_CELL_ALPHA]
        load_cell_sensor_noise_scale = equipment_configurations_float64_vector[IDX_EQ_SENSOR_LOAD_CELL_NOISE_SCALE]
        load_cell_sensor_lower_limit = equipment_configurations_float64_vector[IDX_EQ_SENSOR_LOAD_CELL_LOWER_LIMIT]
        load_cell_sensor_upper_limit = equipment_configurations_float64_vector[IDX_EQ_SENSOR_LOAD_CELL_UPPER_LIMIT]

        # Michaelis-Menten non-competitive suppression equation
        f_gas_multiplier = (O2_pct / (K_m_O2 + O2_pct)) * (1 / (1 + CO2_ppm / K_i_CO2))

        inv_V_free_m3 = 1 / V_free_m3
        # inv_P_ambient = 1 / P_ambient

        fruit_type_code = room_state_int64_vector[IDX_ROOM_STATE_ACTIVE_FRUIT_TYPE_CODE]

        for i in range(room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_AIRFLOW]):
            pallet_slot = airflow_pallets_compact_vector[i]
            # if pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IS_IN_AIRFLOW]:
            # is_partial_dispatch = False
            if pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_DISPATCH_RUN]:
                q_read_slot = pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_QUEUE_READ_SLOT]
                if not outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IN_FULL_DISPATCH]:
                    is_partial_dispatch = True
                else:
                    is_partial_dispatch = False
            else:
                is_partial_dispatch = False
            # is_partial_dispatch = (
            #     pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_DISPATCH_RUN]
            #     and not outbound_wave_queue_matrix[dispatch_queue_slot, IDX_WAVE_IN_FULL_DISPATCH]
            #     # and not pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH]
            # )
            total_box_slots_to_iterate = pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_ACTUAL_BOX_COUNT] if not is_partial_dispatch else pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_BOX_SLOTS_COUNT]
            # Fruit-related updates
            # if pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IS_IN_AIRFLOW]:
            # Calculate mass-dependent terms
            T_pulp_batch = pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PULP_TEMPERATURE]
            h_fg_fruit_batch = latent_heat_vaporization_j_kg(T_pulp_batch)
            R_batch_mg_co2_per_kg_fruit_hr = calc_respiration_rate_mg_co2_per_kg_fruit_hr(
                T_pulp_batch,
                T_vector, R_vector,
                T_min, T_max,
                T_lo, T_hi, T_ref, R_ref, Q_10,
            )

            m_dot_batch_kg_co2_per_kg_fruit_sec = (R_batch_mg_co2_per_kg_fruit_hr * (1e-6 * INV_3600)) * f_gas_multiplier
            k_respiration_batch_j_per_kg_fruit_sec = m_dot_batch_kg_co2_per_kg_fruit_sec * h_respiration_j_kg
            V_dot_batch_ml_co2_per_kg_fruit_sec = (m_dot_batch_kg_co2_per_kg_fruit_sec * INV_M_CO2) * (R_IDEAL_GAS * (T_pulp_batch + 273.15) * inv_P_ambient) * 1e6
            P_sat_pulp_batch = p_sat_magnus(T_pulp_batch, ALDUCHOV_LIQ_CONST_SET)
            k_p_pressure_batch = k_p*(P_sat_pulp_batch - P_w_room)

            # C_fruit_batch = 0.0
            # C_packaging_batch = 0.0
            # C_pallet_batch = 0.0
            # fruit_mass_batch_kg = 0.0
            Q_respiration_batch = 0.0
            Q_evap_fruit_batch = 0.0
            m_dot_transpiration_batch = 0.0
            water_loss_per_batch_kg = 0.0
            CO2_prod_per_batch_ppm_sec = 0.0
            C_composite_batch = 0.0
            k_zone_sampled = pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_K_ZONE]
            k_zone_eff_batch = k_zone_sampled * movement_multiplier if box_flags_ledger[0, pallet_slot, IDX_BOX_IS_IN_TRANSIT] else k_zone_sampled

            # for box_idx in range(max_num_boxes):
            for box_idx in range(total_box_slots_to_iterate):
                fruit_mass_in_box_kg = box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_ACTUAL]
                # if fruit_mass_in_box_kg == SENTINEL_FLOAT64:
                #     continue
                # else:
                # k_zone_effective =  k_zone_sampled * movement_multiplier if box_flags_ledger[box_idx, pallet_slot, IDX_BOX_IS_IN_TRANSIT] else k_zone_sampled

                if fruit_mass_in_box_kg > 0.0:
                    # Box slots that have fruit in them
                    if box_flags_ledger[box_idx, pallet_slot, IDX_BOX_IS_IN_PALLET]:
                        # All boxes in same pallet share the same temperature
                        m_dot_transpiration_box = max(k_p_pressure_batch * fruit_mass_in_box_kg, 0.0)
                        water_loss_per_box_kg = m_dot_transpiration_box * dt_engine_sec

                        CO2_prod_per_box_ppm_sec = V_dot_batch_ml_co2_per_kg_fruit_sec * fruit_mass_in_box_kg * inv_V_free_m3
  
                        Q_respiration_box = k_respiration_batch_j_per_kg_fruit_sec * fruit_mass_in_box_kg
                        Q_evap_fruit_box = m_dot_transpiration_box * h_fg_fruit_batch
                        # For non-partial-dispatch events, all boxes in pallet have same k_zone_effective
                        # k_zone_batch_eff = k_zone_effective

                        # FOR NEXT STEP
                        box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_ACTUAL] = max(fruit_mass_in_box_kg - water_loss_per_box_kg, 0.0)

                        # C_packaging_batch += C_packaging_box
                        # C_fruit_batch += C_fruit_box
                        # fruit_mass_batch_kg += fruit_mass_in_box_kg
                        m_dot_transpiration_batch += m_dot_transpiration_box
                        water_loss_per_batch_kg += water_loss_per_box_kg
                        CO2_prod_per_batch_ppm_sec += CO2_prod_per_box_ppm_sec
                        Q_respiration_batch += Q_respiration_box
                        Q_evap_fruit_batch += Q_evap_fruit_box

                    elif (
                        is_partial_dispatch
                        and box_flags_ledger[box_idx, pallet_slot, IDX_BOX_IS_IN_TRANSIT]
                        # Only boxes in transit during partial dispatches are 'detached'
                        # If box is not longer in transit during a partial dispatch it means it's gone
                    ):
                        # Dispatched boxes become their own system
                        T_pulp_box = partial_dispatch_box_temperatures_ledger[box_idx, pallet_slot]
                        h_fg_fruit_box = latent_heat_vaporization_j_kg(T_pulp_box)
                        R_box_mg_co2_per_kg_fruit_hr = calc_respiration_rate_mg_co2_per_kg_fruit_hr(
                            T_pulp_box,
                            T_vector, R_vector,
                            T_min, T_max,
                            T_lo, T_hi, T_ref, R_ref, Q_10,
                        )

                        # Michaelis-Menten non-competitive suppression equation
                        m_dot_box_kg_co2_per_kg_fruit_sec = (R_box_mg_co2_per_kg_fruit_hr * (1e-6 * INV_3600)) * f_gas_multiplier
                        k_respiration_box_j_per_kg_fruit_sec = m_dot_box_kg_co2_per_kg_fruit_sec * h_respiration_j_kg
                        V_dot_box_ml_co2_per_kg_fruit_sec = (m_dot_box_kg_co2_per_kg_fruit_sec * INV_M_CO2) * (R_IDEAL_GAS * (T_pulp_batch + 273.15) * inv_P_ambient) * 1e6

                        P_sat_pulp_box = p_sat_magnus(T_pulp_box, ALDUCHOV_LIQ_CONST_SET)
                        m_dot_transpiration_box = max(k_p*(P_sat_pulp_box - P_w_room) * fruit_mass_in_box_kg, 0.0)

                        # Q_room_exchange_box
                        box_packaging_mass = box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_BOX_PACKAGE_MASS, IDX_BOX_LEDGER_ACTUAL]
                        new_pallet_fraction_mass = box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NEW_PALLET_MASS, IDX_BOX_LEDGER_ACTUAL]
                        C_fruit_box = fruit_mass_in_box_kg * Cp_fruit
                        C_new_pallet_box = new_pallet_fraction_mass * Cp_pallet
                        C_packaging_box = box_packaging_mass * Cp_packaging
                        C_composite_box = C_fruit_box + C_packaging_box + C_new_pallet_box

                        CO2_prod_per_box_ppm_sec = V_dot_box_ml_co2_per_kg_fruit_sec * fruit_mass_in_box_kg * inv_V_free_m3
                        water_loss_per_box_kg = m_dot_transpiration_box * dt_engine_sec
                        Q_evap_fruit_box = m_dot_transpiration_box * h_fg_fruit_box
                        Q_respiration_box = k_respiration_box_j_per_kg_fruit_sec * fruit_mass_in_box_kg
                        k_zone_eff = k_zone_sampled * movement_multiplier # Detached boxes are in movement
                        Q_room_exchange_box = C_composite_box * k_zone_eff * (T_pulp_box - T_room)

                        # All updates go directly to bulk
                        # Boxes only talk with the room and are no longer part of the 'blob'
                        # Therefore, only Q_room_exchange_box is added to the bulk heat term counterpart
                        # fruit_mass_bulk_kg += fruit_mass_in_box_kg
                        m_dot_transpiration_bulk += m_dot_transpiration_box
                        water_loss_bulk_kg += water_loss_per_box_kg
                        CO2_prod_ppm_sec_bulk += CO2_prod_per_box_ppm_sec
                        Q_room_exchange_bulk += Q_room_exchange_box

                        # Update mass and temperature for next step
                        # Boxes become their own system and only talk with the room
                        dT_pulp_box = (
                            - Q_room_exchange_box
                            + Q_respiration_box
                            - Q_evap_fruit_box
                        ) / max(C_composite_box, 1e-4) * dt_engine_sec
                        partial_dispatch_box_temperatures_ledger[box_idx, pallet_slot] += dT_pulp_box
                        # FOR NEXT STEP
                        box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_ACTUAL] = max(fruit_mass_in_box_kg - water_loss_per_box_kg, 0.0)

                # pallet_fraction_mass = box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_PALLET_MASS, IDX_BOX_LEDGER_ACTUAL]
                # C_pallet_batch += pallet_fraction_mass * Cp_pallet
            # if 150318 <= step <= 150320 and pallet_slot == 0:
            #     print(pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PULP_MASS], pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_BOX_PACKAGING_MASS], pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PALLET_MASS])
            
            pulp_mass_batch = pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PULP_MASS]
            C_fruit_batch = pulp_mass_batch * Cp_fruit
            C_packaging_batch = pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_BOX_PACKAGING_MASS] * Cp_packaging
            C_pallet_batch = pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PALLET_MASS] * Cp_pallet

            C_composite_batch = C_fruit_batch + C_packaging_batch + C_pallet_batch

            Q_room_exchange_batch = C_composite_batch * k_zone_eff_batch * (T_pulp_batch - T_room)
            Q_bulk_to_batch_exchange = C_composite_batch * k_bulk_batch_coupling * (T_pulp_batch - T_pulp_bulk) # NOTE: Added C_composite_batch

            dT_pulp_batch = (
                - Q_bulk_to_batch_exchange
                - Q_room_exchange_batch
                + Q_respiration_batch
                - Q_evap_fruit_batch
            ) / max(C_composite_batch, 1e-4) * dt_engine_sec

            pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PULP_TEMPERATURE] += dT_pulp_batch

            if sample_t_pulp_sensor and pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IS_PALLET_IN_PLACE]:
                # Pulp temperature sensor
                t_pulp_filtered_val = get_sensor_filtered_value(
                    rng                     = rng,
                    true_value              = pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PULP_TEMPERATURE],
                    previous_filtered_value = pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PULP_TEMP_FILTERED_VAL],
                    alpha                   = t_pulp_sensor_alpha,
                    noise_scale             = t_pulp_sensor_noise_scale,
                    lower_limit             = t_pulp_sensor_lower_limit,
                    upper_limit             = t_pulp_sensor_upper_limit,
                )
                pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PULP_TEMP_FILTERED_VAL] = t_pulp_filtered_val

                # Load cell
                true_mass = pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PULP_MASS] + pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_BOX_PACKAGING_MASS]
                mass_filtered_val = get_sensor_filtered_value(
                    rng                     = rng,
                    true_value              = true_mass,
                    previous_filtered_value = pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_MASS_FILTERED_VAL],  # new field, add to pallet_physics matrix
                    alpha                   = load_cell_sensor_alpha,
                    noise_scale             = load_cell_sensor_noise_scale,
                    lower_limit             = load_cell_sensor_lower_limit,
                    upper_limit             = load_cell_sensor_upper_limit,
                )

                pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_MASS_FILTERED_VAL] = mass_filtered_val
                
                # Write into dbt export array
                batch_sensor_write_idx = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_BATCH_SENSOR_WRITE_IDX]
                batch_fruit_sensor_ledger[batch_sensor_write_idx, IDX_BATCH_SENSOR_TIMESTAMP_MS] = epoch_timestamp_ms 
                batch_fruit_sensor_ledger[batch_sensor_write_idx, IDX_BATCH_SENSOR_BATCH_ID] = pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_BATCH_ID]
                # batch_fruit_sensor_ledger[batch_sensor_write_idx, IDX_BATCH_SENSOR_FRUIT_TYPE_CODE] = fruit_type_code 
                batch_fruit_sensor_ledger[batch_sensor_write_idx, IDX_BATCH_SENSOR_PALLET_SLOT] = pallet_slot 
                batch_fruit_sensor_ledger[batch_sensor_write_idx, IDX_BATCH_SENSOR_T_PULP] = t_pulp_filtered_val
                batch_fruit_sensor_ledger[batch_sensor_write_idx, IDX_BATCH_SENSOR_MASS] = mass_filtered_val
                room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_BATCH_SENSOR_WRITE_IDX] = batch_sensor_write_idx + 1

            pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AGE_MS] += dt_engine_ms

            # FOR NEXT STEP
            pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PULP_MASS] = max(pulp_mass_batch - water_loss_per_batch_kg, 0.0)

            C_composite_fruit_bulk += C_composite_batch
            m_dot_transpiration_bulk += m_dot_transpiration_batch
            water_loss_bulk_kg += water_loss_per_batch_kg
            CO2_prod_ppm_sec_bulk += CO2_prod_per_batch_ppm_sec
            Q_respiration_bulk += Q_respiration_batch
            Q_room_exchange_bulk += Q_room_exchange_batch
            Q_evap_fruit_bulk += Q_evap_fruit_batch
            Q_batch_to_bulk_exchange += Q_bulk_to_batch_exchange

        # if 150319 <= step <= 150321:
        #     print(step, "pallet_slot:", pallet_slot, "T_pulp_batch:", T_pulp_batch, "T_room:", T_room, "k_zone_eff_batch:", k_zone_eff_batch, "C_composite_batch:", C_composite_batch, 
        #           "Q_room_exchange_batch:", Q_room_exchange_batch, "Q_bulk_to_batch_exchange:", Q_bulk_to_batch_exchange)
    # if 150319 <= step <= 150321:
    #     print(step, "T_room:", room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C]) # , "k_zone_eff_batch:", k_zone_eff_batch, "C_composite_batch:", C_composite_batch, 
    #           # "Q_room_exchange_batch:", Q_room_exchange_batch, "Q_bulk_to_batch_exchange:", Q_bulk_to_batch_exchange)


    return (
        Q_respiration_bulk,
        Q_room_exchange_bulk,
        Q_evap_fruit_bulk,
        Q_batch_to_bulk_exchange,
        CO2_prod_ppm_sec_bulk,
        C_composite_fruit_bulk,
        m_dot_transpiration_bulk,
        water_loss_bulk_kg,
    )
