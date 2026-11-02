"""
refrigeration_cycle.py

The refrigeration cycle itself: rated hardware constants, compressor performance curves, the cycle solver loop, coil temperature updates.
"""

import math
from numba import njit

from constants import *
from matrix_index_maps import *
from psychrometrics import (
    calculate_ua_coil_air,
    compressor_volumetric_efficiency,
    frost_thermal_resistance_k_w,
    frost_geometry,
    passage_frost_free_flow_fraction,
    combined_frost_free_flow_fraction,
    frost_airflow_ratio,
    frost_conductivity_w_m_k,
    frost_conductance_full_contact_layers,
    frost_conductance_passage_to_inner_face,
    frost_conductance_metal_to_inner_face,
    passage_frost_temperatures_and_capacities,
    face_frost_temperatures_and_capacities,
    calculate_energy_state_delta,
)
from sensors_control import pid_evaluation, defrost_time_controller, update_evaporator_air_velocity_sensor
from humidity_gas import calculate_evaporator_heat_and_moisture_exchange
from fruit_biology import compute_market_scarcity_matrix
from utils import evaluate_antoine_fit, evaluate_poly1d_fit, evaluate_poly2d_cubic_fit, evaluate_poly2d_quadratic_fit, evaluate_poly3d_quadratic_fit, invert_antoine_fit, numba_clip

@njit
def update_downtime_and_ramp_up_schedule(
    rng,
    room_state_int64_vector,
    room_state_float64_vector,
    room_ephemeral_int64_vector,
    equipment_configurations_float64_vector,
    downtime_timestamps_ms_vector,
    valid_crop_rotation_timestamps_ms_vector,
    valid_refrigerant_rotation_timestamps_ms_vector,
    fruit_parameters_float64_matrix,
    fruit_parameters_int64_matrix,
    refrigerant_properties_matrix,
    refrigerant_surfaces_matrix,
    historical_yield_matrix,
    base_quality_grade_probs_matrix,
    fruit_respiration_curves_matrix,

    active_crop_params_float64_vector,
    active_crop_params_int64_vector,
    active_crop_historical_yield_matrix,
    active_crop_market_scarcity_matrix,
    active_crop_base_grades_probs_matrix,
    active_crop_respiration_curves_matrix,
    active_refrigerant_properties_vector,
    active_refrigerant_surfaces_matrix,

    crop_rotation_event_ledger,
    refrigerant_rotation_event_ledger,
    downtime_sequence_event_ledger,

    dt_engine_ms,
    sentinel=MAX_EPOCH_TIMESTAMP_MS,
):
    epoch_timestamp_ms = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]
    next_time_step_ms = epoch_timestamp_ms + dt_engine_ms

    is_fruit_switch_in_process       = room_state_int64_vector[IDX_ROOM_STATE_IS_FRUIT_SWITCH_IN_PROCESS]
    is_refrigerant_switch_in_process = room_state_int64_vector[IDX_ROOM_STATE_IS_REFRIGERANT_SWITCH_IN_PROCESS]
    standard_ramp_up_scheduled       = room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_SCHEDULED]

    crop_rotation_scan_idx             = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_CROP_ROTATION_SCAN_IDX]
    refrigerant_rotation_scan_idx      = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_REFRIGERANT_ROTATION_SCAN_IDX]
    downtime_idx                       = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_IDX]
    is_downtime_in_simulated_timeframe = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_IS_DOWNTIME_IN_TIMEFRAME]

    no_fruit_activity = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_NO_FRUIT_ACTIVITY] 

    # Crop rotation: counter persists forever, drives active_crop_row
    if is_fruit_switch_in_process and no_fruit_activity:
        print("CROP SWITCH FIRING at:", epoch_timestamp_ms, "| no_fruit_activity:", no_fruit_activity) ### DEBUG ###
        is_fruit_switch_in_process = False
        room_state_int64_vector[IDX_ROOM_STATE_CROP_ROTATION_COUNTER] += 1
        # Slice fruit parameter arrays
        active_crop_row = room_state_int64_vector[IDX_ROOM_STATE_CROP_ROTATION_COUNTER] % fruit_parameters_float64_matrix.shape[0]

        # 1. Update 1D vectors first
        active_crop_params_float64_vector[:]           = fruit_parameters_float64_matrix[active_crop_row, ...]
        active_crop_params_int64_vector[:]             = fruit_parameters_int64_matrix[active_crop_row, ...]
        room_state_int64_vector[IDX_ROOM_STATE_ACTIVE_FRUIT_TYPE_CODE] = active_crop_params_int64_vector[IDX_FRUIT_PARAM_FRUIT_TYPE_CODE]
       
        # 2. Extract specific active slice lengths for the new crop
        n_grades = active_crop_params_int64_vector[IDX_FRUIT_PARAM_GRADING_SCALE_LEN]
        n_map    = active_crop_params_int64_vector[IDX_FRUIT_PARAM_YEARLY_MAP_LEN]
        n_resp   = active_crop_params_int64_vector[IDX_FRUIT_PARAM_RESP_CURVE_TRUE_LEN]

        print(
            "active_crop_respiration_curves_matrix[:n_resp, ...]", active_crop_respiration_curves_matrix[:n_resp, ...].shape, "\n",
            "active_crop_base_grades_probs_matrix[:n_grades, ...]", base_quality_grade_probs_matrix[active_crop_row, :n_grades, ...].shape, "\n",
            "fruit_respiration_curves_matrix[active_crop_row, :n_resp, ...]", fruit_respiration_curves_matrix[active_crop_row, :n_resp, ...].shape, "\n",
            "base_quality_grade_probs_matrix[active_crop_row, :n_grades, ...]", base_quality_grade_probs_matrix[active_crop_row, :n_grades, ...].shape, "\n",
        )

        # 3. Slice destination buffers by the exact matching lengths on BOTH sides
        # active_crop_inventory_mass_per_grade_vector[:n_grades] = inventory_mass_per_grade_vector[:n_grades]
        active_crop_historical_yield_matrix[:n_map, ...]       = historical_yield_matrix[active_crop_row, :n_map, ...]
        active_crop_market_scarcity_matrix[:n_map, ...]        = compute_market_scarcity_matrix(
            rng,
            active_crop_historical_yield_matrix[:n_map, ...],
            active_crop_params_float64_vector[IDX_FRUIT_PARAM_MARKET_CORRELATION],
            active_crop_params_float64_vector[IDX_FRUIT_PARAM_MARKET_VOLATILITY_SCALE],
        )
        active_crop_base_grades_probs_matrix[:n_grades, ...]   = base_quality_grade_probs_matrix[active_crop_row, :n_grades, ...]
        active_crop_respiration_curves_matrix[:n_resp, ...]    = fruit_respiration_curves_matrix[active_crop_row, :n_resp, ...]

        # crop rotation block
        crop_rotation_row = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_CROP_ROTATION_EVENT_WRITE_IDX]
        crop_rotation_event_ledger[crop_rotation_row, IDX_CROP_ROTATION_EVENT_TIMESTAMP_MS] = epoch_timestamp_ms
        crop_rotation_event_ledger[crop_rotation_row, IDX_CROP_ROTATION_EVENT_FRUIT_TYPE_CODE] = active_crop_params_int64_vector[IDX_FRUIT_PARAM_FRUIT_TYPE_CODE]
        room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_CROP_ROTATION_EVENT_WRITE_IDX] = crop_rotation_row + 1

        # Recalculate ALL fixed hardware parameters
        (
            room_state_float64_vector[IDX_ROOM_STATE_RATED_COMPRESSOR_Q_COOLING],
            room_state_float64_vector[IDX_ROOM_STATE_RATED_COMPRESSOR_W_ELECT],
            room_state_float64_vector[IDX_ROOM_STATE_RATED_COMPRESSOR_REFRIG_M_DOT],
            room_state_float64_vector[IDX_ROOM_STATE_V_DOT_EVAP_FULLY_ON],
            # room_state_float64_vector[IDX_ROOM_STATE_UA_COIL_EVAP],
            # room_state_float64_vector[IDX_ROOM_STATE_UA_INTERNAL_EVAP],
            # room_state_float64_vector[IDX_ROOM_STATE_C_CLEARANCE], 
        ) = derive_fixed_hardware_parameters(
            room_state_float64_vector,
            equipment_configurations_float64_vector,
            active_crop_params_float64_vector,
            # active_refrigerant_properties_vector,
            active_refrigerant_surfaces_matrix,
        )

    # Refrigerant rotation: same pattern, separate counter
    if is_refrigerant_switch_in_process and no_fruit_activity:
        is_refrigerant_switch_in_process = False
        room_state_int64_vector[IDX_ROOM_STATE_REFRIGERANT_ROTATION_COUNTER] += 1
        # Slice refrigerant parameter arrays
        active_refrigerant_row = room_state_int64_vector[IDX_ROOM_STATE_REFRIGERANT_ROTATION_COUNTER] % refrigerant_properties_matrix.shape[0]  
        room_state_int64_vector[IDX_ROOM_STATE_ACTIVE_REFRIGERANT_CODE] = active_refrigerant_row
        active_refrigerant_properties_vector[:] = refrigerant_properties_matrix[active_refrigerant_row, ...]
        active_refrigerant_surfaces_matrix[...] = refrigerant_surfaces_matrix[active_refrigerant_row, ...]
        # Recalculate ALL fixed hardware parameters
        (
            room_state_float64_vector[IDX_ROOM_STATE_RATED_COMPRESSOR_Q_COOLING],
            room_state_float64_vector[IDX_ROOM_STATE_RATED_COMPRESSOR_W_ELECT],
            room_state_float64_vector[IDX_ROOM_STATE_RATED_COMPRESSOR_REFRIG_M_DOT],
            room_state_float64_vector[IDX_ROOM_STATE_V_DOT_EVAP_FULLY_ON],
            # room_state_float64_vector[IDX_ROOM_STATE_UA_COIL_EVAP],
            # room_state_float64_vector[IDX_ROOM_STATE_UA_INTERNAL_EVAP],
            # room_state_float64_vector[IDX_ROOM_STATE_C_CLEARANCE], 
        ) = derive_fixed_hardware_parameters(
            room_state_float64_vector,
            equipment_configurations_float64_vector,
            active_crop_params_float64_vector,
            # active_refrigerant_properties_vector,
            active_refrigerant_surfaces_matrix,
        )

        # refrigerant rotation block
        refrigerant_rotation_row = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_REFRIGERANT_ROTATION_EVENT_WRITE_IDX]
        refrigerant_rotation_event_ledger[refrigerant_rotation_row, IDX_REFRIGERANT_ROTATION_EVENT_TIMESTAMP_MS] = epoch_timestamp_ms
        refrigerant_rotation_event_ledger[refrigerant_rotation_row, IDX_REFRIGERANT_ROTATION_EVENT_REFRIGERANT_CODE] = active_refrigerant_row
        room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_REFRIGERANT_ROTATION_EVENT_WRITE_IDX] = refrigerant_rotation_row + 1

    if is_downtime_in_simulated_timeframe:
        if crop_rotation_scan_idx < len(valid_crop_rotation_timestamps_ms_vector):
            if epoch_timestamp_ms == valid_crop_rotation_timestamps_ms_vector[crop_rotation_scan_idx]:
                print("CROP SWITCH TRIGGERED (waiting for no_fruit_activity) at:", epoch_timestamp_ms) ### DEBUG ###
                is_fruit_switch_in_process = True
                crop_rotation_scan_idx += 1

        if refrigerant_rotation_scan_idx < len(valid_refrigerant_rotation_timestamps_ms_vector):
            if epoch_timestamp_ms == valid_refrigerant_rotation_timestamps_ms_vector[refrigerant_rotation_scan_idx]:
                is_refrigerant_switch_in_process = True
                refrigerant_rotation_scan_idx += 1

        if epoch_timestamp_ms == downtime_timestamps_ms_vector[downtime_idx]:
            downtime_row = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_SEQUENCE_EVENT_WRITE_IDX]
            downtime_sequence_event_ledger[downtime_row, IDX_DOWNTIME_SEQUENCE_EVENT_SCHEDULED_START_TIMESTAMP_MS] = epoch_timestamp_ms
            if downtime_sequence_event_ledger[downtime_row, IDX_DOWNTIME_SEQUENCE_EVENT_TIMESTAMP_MS] == sentinel:
                downtime_sequence_event_ledger[downtime_row, IDX_DOWNTIME_SEQUENCE_EVENT_TIMESTAMP_MS] = epoch_timestamp_ms

        if next_time_step_ms == downtime_timestamps_ms_vector[downtime_idx + 1]:
            if not no_fruit_activity: # or not room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_REFRIGERATION_SYSTEM_FULLY_OFF]:
                # Delay the schedule milestone boundary just before the expiration boundary is reached
                downtime_timestamps_ms_vector[downtime_idx + 1] += MS_PER_5_MINUTES
                print("DOWNTIME BOUNDARY PUSHED to:", downtime_timestamps_ms_vector[downtime_idx + 1], "| no_fruit_activity:", no_fruit_activity, "| fully_off:", room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_REFRIGERATION_SYSTEM_FULLY_OFF])

        downtime_scheduled = not standard_ramp_up_scheduled and (
            downtime_timestamps_ms_vector[downtime_idx] <= epoch_timestamp_ms < downtime_timestamps_ms_vector[downtime_idx + 1]
        )
        downtime_safe_for_shutdown = downtime_scheduled and no_fruit_activity

        if downtime_safe_for_shutdown:
            downtime_row = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_SEQUENCE_EVENT_WRITE_IDX]
            if downtime_sequence_event_ledger[downtime_row, IDX_DOWNTIME_SEQUENCE_EVENT_SHUTDOWN_TIMESTAMP_MS] == sentinel:
                downtime_sequence_event_ledger[downtime_row, IDX_DOWNTIME_SEQUENCE_EVENT_SHUTDOWN_TIMESTAMP_MS] = epoch_timestamp_ms
                if downtime_sequence_event_ledger[downtime_row, IDX_DOWNTIME_SEQUENCE_EVENT_TIMESTAMP_MS] == sentinel:
                    downtime_sequence_event_ledger[downtime_row, IDX_DOWNTIME_SEQUENCE_EVENT_TIMESTAMP_MS] = epoch_timestamp_ms

        # ### DEBUG ###
        # if downtime_scheduled:
        #     print("DOWNTIME ACTIVE at:", epoch_timestamp_ms, "| safe_for_shutdown:", downtime_safe_for_shutdown)
        # ### DEBUG ###

        if epoch_timestamp_ms == downtime_timestamps_ms_vector[downtime_idx + 1]:
            room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_START_MS] = downtime_timestamps_ms_vector[downtime_idx + 1]
            room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_END_MS] = downtime_timestamps_ms_vector[downtime_idx + 1] + active_crop_params_int64_vector[IDX_FRUIT_PARAM_RAMP_UP_DURATION_MS]
            print("SETTING RAMP-UP START at boundary:", epoch_timestamp_ms, "downtime_idx:", downtime_idx) ### DEBUG ###

            if downtime_idx < len(downtime_timestamps_ms_vector) - 2:
                downtime_idx += 2
            else:
                is_downtime_in_simulated_timeframe = False
                downtime_scheduled = False
                downtime_safe_for_shutdown = False
    else:
        downtime_scheduled = False
        downtime_safe_for_shutdown = False

    if standard_ramp_up_scheduled and next_time_step_ms == room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_END_MS]:
        t_setpoint = active_crop_params_float64_vector[IDX_FRUIT_PARAM_T_SETPOINT]
        t_deadband = equipment_configurations_float64_vector[IDX_EQ_REFRIG_CONTROL_DEADBAND_C]
        rh_setpoint = active_crop_params_float64_vector[IDX_FRUIT_PARAM_RH_SETPOINT]
        rh_deadband_reference = active_crop_params_float64_vector[IDX_FRUIT_PARAM_RH_DEADBAND_REFERENCE]
        t_room_filtered_val = room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_FILTERED_VAL]
        rh_room_filtered_val = room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM_FILTERED_VAL]
        # if t_room_filtered_val > (t_setpoint + 1.5*t_deadband) or rh_room_filtered_val < (rh_setpoint - rh_deadband_reference):
        right_conditions = (
            (t_setpoint - t_deadband) < t_room_filtered_val < (t_setpoint + 1.5*t_deadband)
            and (rh_setpoint - 1.5*rh_deadband_reference) < rh_room_filtered_val < 1.0
        )
        # if t_room_filtered_val > (t_setpoint + 1.5*t_deadband) or rh_room_filtered_val < (rh_setpoint - 1.5*rh_deadband_reference):
        if not right_conditions:
            print("RAMP-UP EXTENDED at:", epoch_timestamp_ms, "| T_room:", t_room_filtered_val, "RH:", rh_room_filtered_val) ### DEBUG ###
            print(
                "TOTAL_BATCHES_IN_ROOM", room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_ROOM], "\n",
                "TOTAL_ARRIVALS_PLANNED", room_state_int64_vector[IDX_ROOM_STATE_TOTAL_ARRIVALS_PLANNED], "\n",
                "TOTAL_DISPATCHES_PLANNED", room_state_int64_vector[IDX_ROOM_STATE_TOTAL_DISPATCHES_PLANNED], "\n",
            )
            room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_END_MS] += MS_PER_5_MINUTES

    if epoch_timestamp_ms == room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_START_MS]:
        sampled_coil_health = rng.beta(a=9.0, b=0.5)
        room_state_float64_vector[IDX_ROOM_STATE_ACTIVE_COIL_EFFICIENCY] = max(0.70, sampled_coil_health)
        print("RAMP-UP START at:", epoch_timestamp_ms, "| coil efficiency sampled:", room_state_float64_vector[IDX_ROOM_STATE_ACTIVE_COIL_EFFICIENCY]) ### DEBUG ###
        print(
            "TOTAL_BATCHES_IN_ROOM", room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_ROOM], "\n",
            "TOTAL_ARRIVALS_PLANNED", room_state_int64_vector[IDX_ROOM_STATE_TOTAL_ARRIVALS_PLANNED], "\n",
            "TOTAL_DISPATCHES_PLANNED", room_state_int64_vector[IDX_ROOM_STATE_TOTAL_DISPATCHES_PLANNED], "\n",
        )
        downtime_row = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_SEQUENCE_EVENT_WRITE_IDX]
        downtime_sequence_event_ledger[downtime_row, IDX_DOWNTIME_SEQUENCE_EVENT_RAMP_UP_START_TIMESTAMP_MS] = epoch_timestamp_ms
        if downtime_sequence_event_ledger[downtime_row, IDX_DOWNTIME_SEQUENCE_EVENT_TIMESTAMP_MS] == sentinel:
            downtime_sequence_event_ledger[downtime_row, IDX_DOWNTIME_SEQUENCE_EVENT_TIMESTAMP_MS] = epoch_timestamp_ms

        # Recalculate ALL fixed hardware parameters
        (
            room_state_float64_vector[IDX_ROOM_STATE_RATED_COMPRESSOR_Q_COOLING],
            room_state_float64_vector[IDX_ROOM_STATE_RATED_COMPRESSOR_W_ELECT],
            room_state_float64_vector[IDX_ROOM_STATE_RATED_COMPRESSOR_REFRIG_M_DOT],
            room_state_float64_vector[IDX_ROOM_STATE_V_DOT_EVAP_FULLY_ON],
            # room_state_float64_vector[IDX_ROOM_STATE_UA_COIL_EVAP],
            # room_state_float64_vector[IDX_ROOM_STATE_UA_INTERNAL_EVAP],
            # room_state_float64_vector[IDX_ROOM_STATE_C_CLEARANCE], 
        ) = derive_fixed_hardware_parameters(
            room_state_float64_vector,
            equipment_configurations_float64_vector,
            active_crop_params_float64_vector,
            # active_refrigerant_properties_vector,
            active_refrigerant_surfaces_matrix,
        )

    standard_ramp_up_scheduled = room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_START_MS] <= epoch_timestamp_ms < room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_END_MS]
    if epoch_timestamp_ms == room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_END_MS]:
        print("RAMP-UP END at:", epoch_timestamp_ms)
        print(
            "TOTAL_BATCHES_IN_ROOM", room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_ROOM], "\n",
            "TOTAL_ARRIVALS_PLANNED", room_state_int64_vector[IDX_ROOM_STATE_TOTAL_ARRIVALS_PLANNED], "\n",
            "TOTAL_DISPATCHES_PLANNED", room_state_int64_vector[IDX_ROOM_STATE_TOTAL_DISPATCHES_PLANNED], "\n",
        )
        downtime_row = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_SEQUENCE_EVENT_WRITE_IDX]
        downtime_sequence_event_ledger[downtime_row, IDX_DOWNTIME_SEQUENCE_EVENT_RAMP_UP_END_TIMESTAMP_MS] = epoch_timestamp_ms
        if downtime_sequence_event_ledger[downtime_row, IDX_DOWNTIME_SEQUENCE_EVENT_TIMESTAMP_MS] == sentinel:
            downtime_sequence_event_ledger[downtime_row, IDX_DOWNTIME_SEQUENCE_EVENT_TIMESTAMP_MS] = epoch_timestamp_ms
        room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_SEQUENCE_EVENT_WRITE_IDX] = downtime_row + 1

    room_state_int64_vector[IDX_ROOM_STATE_IS_FRUIT_SWITCH_IN_PROCESS]       = is_fruit_switch_in_process
    room_state_int64_vector[IDX_ROOM_STATE_IS_REFRIGERANT_SWITCH_IN_PROCESS] = is_refrigerant_switch_in_process
    room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_SCHEDULED]                = standard_ramp_up_scheduled

    room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_CROP_ROTATION_SCAN_IDX]        = crop_rotation_scan_idx
    room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_REFRIGERANT_ROTATION_SCAN_IDX] = refrigerant_rotation_scan_idx
    room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_IDX]                  = downtime_idx
    room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_IS_DOWNTIME_IN_TIMEFRAME]      = is_downtime_in_simulated_timeframe
    room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_SCHEDULED]            = downtime_scheduled
    room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_SAFE_FOR_SHUTDOWN]    = downtime_safe_for_shutdown


@njit
def derive_fixed_hardware_parameters(
    room_state_float64_vector,
    equipment_configurations_float64_vector,
    active_crop_params_float64_vector,
    active_refrigerant_surfaces_matrix,
):
    td_design_evap = equipment_configurations_float64_vector[IDX_EQ_DESIGN_EVAP_TD_K]
    td_design_cond = equipment_configurations_float64_vector[IDX_EQ_DESIGN_COND_TD_K]
    t_setpoint = active_crop_params_float64_vector[IDX_FRUIT_PARAM_T_SETPOINT]
    t_evap_rated = t_setpoint - td_design_evap
    t_cond_rated = equipment_configurations_float64_vector[IDX_EQ_DESIGN_AMBIENT_TEMP_C] + td_design_cond
    
    Q_cooling_w_rated = evaluate_poly2d_cubic_fit(
        t_evap_rated,
        t_cond_rated,
        active_refrigerant_surfaces_matrix[SURFACE_POLY2D_CUBIC_RATED_COOLING_W, COEFFS, :],
        active_refrigerant_surfaces_matrix[SURFACE_POLY2D_CUBIC_RATED_COOLING_W, BOUNDS, :],
    )
    W_compressor_w_rated = evaluate_poly2d_cubic_fit(
        t_evap_rated,
        t_cond_rated,
        active_refrigerant_surfaces_matrix[SURFACE_POLY2D_CUBIC_RATED_POWER_INPUT_W, COEFFS, :],
        active_refrigerant_surfaces_matrix[SURFACE_POLY2D_CUBIC_RATED_POWER_INPUT_W, BOUNDS, :],
    )
    m_dot_refrig_kg_s_rated = evaluate_poly2d_cubic_fit(
        t_evap_rated,
        t_cond_rated,
        active_refrigerant_surfaces_matrix[SURFACE_POLY2D_CUBIC_RATED_MASS_FLOW_KG_S, COEFFS, :],
        active_refrigerant_surfaces_matrix[SURFACE_POLY2D_CUBIC_RATED_MASS_FLOW_KG_S, BOUNDS, :],
    )
    # --- Evaporator hardware constants ---
    # Q_evap_rated = Q_cooling_w_rated

    V_dot_air_evap_fully_on = equipment_configurations_float64_vector[IDX_EQ_EVAP_NOMINAL_AIRFLOW_M3_S] * room_state_float64_vector[IDX_ROOM_STATE_ACTIVE_COIL_EFFICIENCY]
    # UA_coil_evap_air = equipment_configurations_float64_vector[IDX_EQ_UA_COIL_EVAP_RATED] * () ** 0.5 # NOTE: Could be made into parameter

    # rho_air_evap_design = P_ATM_STANDARD / (R_DRY_AIR * (t_setpoint + 273.15))
    # UA_coil_evap = calculate_ua_coil(
    #     V_dot_air_evap_fully_on,
    #     rho_air_evap_design,
    #     Q_evap_rated,
    #     td_design_evap,
    # )

    # UA_internal_evap = UA_coil_evap * equipment_configurations_float64_vector[IDX_EQ_EVAP_UA_INTERNAL_SCALE]
    # 
    # log_P_suction_rated = evaluate_antoine_fit(t_evap_rated, active_refrigerant_surfaces_matrix[SURFACE_ANTOINE_EVAP, COEFFS, :])
    # log_P_discharge_rated = evaluate_antoine_fit(t_cond_rated, active_refrigerant_surfaces_matrix[SURFACE_ANTOINE_COND, COEFFS, :])
    # pressure_ratio_rated = max(math.exp(log_P_discharge_rated - log_P_suction_rated), 1.0)
    # 
    # rho_suction_rated = evaluate_poly2d_cubic_fit(
    #     t_evap_rated,
    #     active_refrigerant_properties_vector[IDX_REFRIG_SCALAR_RATED_SUPERHEAT_K],
    #     active_refrigerant_surfaces_matrix[SURFACE_POLY2D_CUBIC_RHO, COEFFS, :],
    #     active_refrigerant_surfaces_matrix[SURFACE_POLY2D_CUBIC_RHO, BOUNDS, :],
    # )
    # 
    # m_dot_refrig_ideal = (
    #     equipment_configurations_float64_vector[IDX_EQ_COMPRESSOR_V_DISPLACEMENT_M3] * 
    #     equipment_configurations_float64_vector[IDX_EQ_COMPRESSOR_SPEED_HZ] *
    #     rho_suction_rated
    # )
    # 
    # eta_vol_rated = m_dot_refrig_kg_s_rated / m_dot_refrig_ideal

    # C_clearance = (1.0 - eta_vol_rated) / ((pressure_ratio_rated ** (active_refrigerant_properties_vector[IDX_REFRIG_INV_SCALAR_GAMMA])) - 1.0)

    return(
        Q_cooling_w_rated,
        W_compressor_w_rated,
        m_dot_refrig_kg_s_rated,
        V_dot_air_evap_fully_on,
        # UA_coil_evap,
        # UA_internal_evap,
        # C_clearance,
    )

@njit
def update_fan_states(
    rng,
    room_state_float64_vector,
    room_state_int64_vector,
    forcing_cache_float64_vector,
    room_ephemeral_int64_vector,
    defrost_event_ledger,
    equipment_configurations_float64_vector,
    equipment_configurations_int64_vector,
    evaporator_fan_curve_matrix,
    dt_engine_sec,
):
    compressor_on = room_state_int64_vector[IDX_ROOM_STATE_REFRIGERATION_IS_ACTIVE]

    # --- Evaporator block ---
    passage_thickness, face_thickness, face_covered_area, face_covered_factor = frost_geometry(
        room_state_float64_vector[IDX_ROOM_STATE_V_PASSAGE_ICE_M3],
        room_state_float64_vector[IDX_ROOM_STATE_V_FACE_ICE_M3],
        equipment_configurations_float64_vector[IDX_EQ_EVAP_INV_SURFACE_AREA_M2],
        equipment_configurations_float64_vector[IDX_EQ_EVAP_FRONTAL_AREA_M2],
        equipment_configurations_float64_vector[IDX_EQ_EVAP_INV_FRONTAL_AREA_M2],
        equipment_configurations_float64_vector[IDX_EQ_EVAP_INV_FACE_BRIDGE_LENGTH_M],
    )

    passage_frost_free_flow_frac = passage_frost_free_flow_fraction(
        passage_thickness,
        equipment_configurations_float64_vector[IDX_EQ_EVAP_INV_FIN_CLEAR_GAP_M],
        equipment_configurations_float64_vector[IDX_EQ_EVAP_INV_TUBE_CLEAR_GAP_M],
    )
    combined_frost_free_flow_frac = combined_frost_free_flow_fraction(passage_frost_free_flow_frac, face_covered_factor)

    if combined_frost_free_flow_frac == 0.0 or combined_frost_free_flow_frac == 1.0:
        frost_flow_ratio = combined_frost_free_flow_frac
    else:
        frost_flow_ratio = frost_airflow_ratio(
            combined_frost_free_flow_frac,
            evaporator_fan_curve_matrix,
            equipment_configurations_float64_vector[IDX_EQ_EVAP_INV_V_DOT_CLEAN],
            equipment_configurations_float64_vector[IDX_EQ_EVAP_PRESSURE_OVER_SQ_V_DOT_CLEAN],
        )

    evap_fan_modulation_frac = room_state_float64_vector[IDX_ROOM_STATE_EVAP_FAN_MODULATION_FRAC]
    cond_fan_modulation_frac = room_state_float64_vector[IDX_ROOM_STATE_COND_FAN_MODULATION_FRAC]

    defrost_time_controller(
        room_state_float64_vector,
        room_state_int64_vector,
        room_ephemeral_int64_vector,
        defrost_event_ledger,
        equipment_configurations_float64_vector,
        equipment_configurations_int64_vector,
    )

    # fans_off = (
    #     room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_SAFE_FOR_SHUTDOWN]
    #     and not compressor_on
    #     and not room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_SCHEDULED]
    #     and defrost_phase in (DF_SETTLE, DF_HEAT, DF_DRIP)
    # )

    full_downtime = (
        room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_SAFE_FOR_SHUTDOWN]
        and not room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_SCHEDULED]
        and not compressor_on
    )

    defrost_phase = room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE]
    evap_off = full_downtime or defrost_phase in (DF_SETTLE, DF_HEAT, DF_DRIP, DF_FAN_DELAY)
    cond_off = full_downtime or defrost_phase in (DF_SETTLE, DF_HEAT, DF_DRIP)

    target_evap_fan = 0.0 if evap_off else 1.0
    evap_fan_modulation_frac += (target_evap_fan - evap_fan_modulation_frac) * equipment_configurations_float64_vector[IDX_EQ_DT_OVER_EVAP_FAN_TAU_LAG_SEC]
    evap_fan_modulation_frac = numba_clip(evap_fan_modulation_frac, SAFE_FLOOR, 1.0)

    target_cond_fan = 0.0 if cond_off else 1.0
    cond_fan_modulation_frac += (target_cond_fan - cond_fan_modulation_frac) * equipment_configurations_float64_vector[IDX_EQ_DT_OVER_COND_FAN_TAU_LAG_SEC]
    cond_fan_modulation_frac = numba_clip(cond_fan_modulation_frac, SAFE_FLOOR, 1.0)

    room_state_float64_vector[IDX_ROOM_STATE_V_DOT_AIR_EVAP_ACTUAL] = room_state_float64_vector[IDX_ROOM_STATE_V_DOT_EVAP_FULLY_ON] * evap_fan_modulation_frac * frost_flow_ratio if evap_fan_modulation_frac > SAFE_FLOOR else 0.0
    room_state_float64_vector[IDX_ROOM_STATE_V_DOT_AIR_COND_ACTUAL] = forcing_cache_float64_vector[IDX_CACHE_V_DOT_COND_FULLY_ON] * cond_fan_modulation_frac if cond_fan_modulation_frac > SAFE_FLOOR else 0.0
    room_state_float64_vector[IDX_ROOM_STATE_EVAP_FAN_MODULATION_FRAC] = evap_fan_modulation_frac
    room_state_float64_vector[IDX_ROOM_STATE_COND_FAN_MODULATION_FRAC] = cond_fan_modulation_frac

    evaporator_fan_motor_ready = evap_fan_modulation_frac >= 0.90
    condenser_fan_motor_ready = cond_fan_modulation_frac >= 0.90

    update_evaporator_air_velocity_sensor(
        rng,
        room_state_float64_vector,
        room_state_int64_vector,
        equipment_configurations_float64_vector,
        equipment_configurations_int64_vector,
    )

    evap_airflow_ok = update_evaporator_airflow_proof(
        room_state_float64_vector,
        room_state_int64_vector,
        equipment_configurations_float64_vector,
        equipment_configurations_int64_vector,
    )

    return passage_thickness, passage_frost_free_flow_frac, combined_frost_free_flow_frac, face_thickness, face_covered_area, face_covered_factor, frost_flow_ratio, evaporator_fan_motor_ready, condenser_fan_motor_ready, evap_airflow_ok


@njit
def update_evaporator_airflow_proof(
    room_state_float64_vector,
    room_state_int64_vector,
    equipment_configurations_float64_vector,
    equipment_configurations_int64_vector,
):
    """Update the flow-switch contact from its sensed velocity and return a permit.

    The contact reports the delayed sensor decision. The permit bypasses it while
    the evaporator fan is intentionally off in FAN_DELAY and briefly after
    fan release, while the fan ramps and the contact completes its delay.
    """
    velocity_m_s = room_state_float64_vector[IDX_ROOM_STATE_EVAP_AIR_VELOCITY_SENSED_M_S]
    proved = room_state_int64_vector[IDX_ROOM_STATE_EVAP_AIRFLOW_PROVED]
    condition_since_ms = room_state_int64_vector[IDX_ROOM_STATE_EVAP_AIRFLOW_PROOF_CONDITION_SINCE_MS]
    epoch_ms = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]
    if proved:
        condition_met = velocity_m_s < equipment_configurations_float64_vector[
            IDX_EQ_EVAP_AIRFLOW_PROOF_TRIP_M_S
        ]
    else:
        condition_met = velocity_m_s >= equipment_configurations_float64_vector[
            IDX_EQ_EVAP_AIRFLOW_PROOF_RECOVERY_M_S
        ]

    if condition_met:
        if condition_since_ms == 0:
            condition_since_ms = epoch_ms
        elif epoch_ms - condition_since_ms >= equipment_configurations_int64_vector[
            IDX_EQ_EVAP_AIRFLOW_PROOF_DELAY_MS
        ]:
            proved = 0 if proved else 1
            condition_since_ms = 0
    else:
        condition_since_ms = 0

    room_state_int64_vector[IDX_ROOM_STATE_EVAP_AIRFLOW_PROVED] = proved
    room_state_int64_vector[IDX_ROOM_STATE_EVAP_AIRFLOW_PROOF_CONDITION_SINCE_MS] = condition_since_ms

    defrost_phase = room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE]
    airflow_interlock_permit = (
        proved == 1
        or defrost_phase == DF_FAN_DELAY
        or (
            defrost_phase == DF_NORMAL
            and epoch_ms < room_state_int64_vector[IDX_ROOM_STATE_EVAP_AIRFLOW_PROOF_GRACE_END_MS]
        )
    )
    return airflow_interlock_permit


@njit
def calculate_airflow_variables(
    room_state_float64_vector,
    forcing_cache_float64_vector,
    room_configurations_float64_vector,
    equipment_configurations_float64_vector,
    active_refrigerant_properties_vector,
    rho_air_room,
    frost_thickness_m,
    frost_rho,
):
    V_dot_air_evap_actual = room_state_float64_vector[IDX_ROOM_STATE_V_DOT_AIR_EVAP_ACTUAL]
    V_dot_air_cond_actual = room_state_float64_vector[IDX_ROOM_STATE_V_DOT_AIR_COND_ACTUAL]

    UA_coil_evap_air = calculate_ua_coil_air(
        active_refrigerant_properties_vector[IDX_REFRIG_SCALAR_UA_COIL_EVAP_RATED],
        equipment_configurations_float64_vector[IDX_EQ_EVAP_NOMINAL_AIRFLOW_M3_S],
        V_dot_air_evap_actual,
        equipment_configurations_float64_vector[IDX_EQ_VELOCITY_EXPONENT_UA_COIL_EVAP_AIR_SIDE],
    )
    if frost_thickness_m > 0.0:
        # The passage ice node is at the layer center. Air crosses half the
        # layer here; update_evaporator_energies() supplies the other half
        # between that node and the metal.
        R_air_to_passage = 0.5 * frost_thermal_resistance_k_w(
            frost_thickness_m,
            frost_rho,
            equipment_configurations_float64_vector[IDX_EQ_EVAP_SURFACE_AREA_M2],
        )
        UA_coil_evap_air = UA_coil_evap_air / (1.0 + UA_coil_evap_air * R_air_to_passage)

    UA_coil_cond_air = calculate_ua_coil_air(
        active_refrigerant_properties_vector[IDX_REFRIG_SCALAR_UA_COIL_COND_RATED],
        equipment_configurations_float64_vector[IDX_EQ_COND_NOMINAL_AIRFLOW_M3_S],
        V_dot_air_cond_actual,
        equipment_configurations_float64_vector[IDX_EQ_VELOCITY_EXPONENT_UA_COIL_COND_AIR_SIDE],
    )
    # UA_coil_evap = room_state_float64_vector[IDX_ROOM_STATE_UA_COIL_EVAP]
    # UA_coil_cond = forcing_cache_float64_vector[IDX_CACHE_UA_COIL_COND]

    m_dot_air_evap = V_dot_air_evap_actual * rho_air_room
    inv_m_dot_air_evap = 1.0 / max(m_dot_air_evap, 1e-4)
    C_evap = m_dot_air_evap * CP_AIR
    inv_C_evap = inv_m_dot_air_evap * INV_CP_AIR
    epsilon_evap = 1.0 - math.exp(-UA_coil_evap_air * inv_C_evap)

    m_dot_air_cond = V_dot_air_cond_actual * forcing_cache_float64_vector[IDX_CACHE_RHO_AIR_AMBIENT]
    C_cond = m_dot_air_cond * CP_AIR
    epsilon_cond = 1.0 - math.exp(-UA_coil_cond_air / max(C_cond, 1e-4))

    evap_fan_nominal_airflow = equipment_configurations_float64_vector[IDX_EQ_INV_EVAP_NOMINAL_AIRFLOW_M3_S]
    evap_fan_airflow_ratio = V_dot_air_evap_actual * evap_fan_nominal_airflow

    # Fan pressure and heat scale via fluid mechanics power laws
    delta_P_coils_pa_actual = equipment_configurations_float64_vector[IDX_EQ_EVAP_PRESSURE_DROP_PA] * (evap_fan_airflow_ratio ** 2)
    # Q_fans = (V_dot_air_evap_actual * delta_P_coils_pa_actual) * equipment_configurations_float64_vector[IDX_EQ_INV_FAN_LUMPED_EFFICIENCY]
    evap_fan_modulation_frac = room_state_float64_vector[IDX_ROOM_STATE_EVAP_FAN_MODULATION_FRAC]
    Q_fans = equipment_configurations_float64_vector[IDX_EQ_EVAP_FAN_POWER_RATED_W] * (evap_fan_modulation_frac * evap_fan_modulation_frac * evap_fan_modulation_frac) if evap_fan_modulation_frac > SAFE_FLOOR else 0.0 # TODO: Fit fan curve

    # Gas infiltration
    n_ach_passive_floor = room_configurations_float64_vector[IDX_ROOM_NOMINAL_ACH_PER_SEC] * 0.20
    n_ach_forced_nominal = room_configurations_float64_vector[IDX_ROOM_NOMINAL_ACH_PER_SEC] * 0.80
    # gasket leaks scale non-linearly via the 1.3 ASHRAE Power Law
    forced_leakage_decay = evap_fan_airflow_ratio ** 1.3

    n_ach_actual_per_sec = n_ach_passive_floor + (n_ach_forced_nominal * forced_leakage_decay)

    return Q_fans, C_evap, inv_C_evap, C_cond, epsilon_evap, epsilon_cond, m_dot_air_evap, inv_m_dot_air_evap, n_ach_actual_per_sec, UA_coil_evap_air, UA_coil_cond_air

@njit
def low_suction_pressure_trip(
    room_state_int64_vector,
    p_suction_line_filtered,
    p_cut_in,
    p_cut_out
):
    if p_suction_line_filtered <= p_cut_out:
        room_state_int64_vector[IDX_ROOM_STATE_LP_TRIPPED] = 1
    elif p_suction_line_filtered >= p_cut_in:
        room_state_int64_vector[IDX_ROOM_STATE_LP_TRIPPED] = 0

@njit
def update_refrigeration_state(
    room_state_float64_vector,
    room_state_int64_vector,
    room_ephemeral_int64_vector,
    compressor_event_ledger,
    lp_trip_event_ledger,
    equipment_configurations_float64_vector,
    equipment_configurations_int64_vector,
    active_crop_params_float64_vector,
    active_refrigerant_properties_vector,
    evaporator_fan_motor_ready,
    condenser_fan_motor_ready,
    evap_airflow_ok,
    dt_engine_sec,
):
    epoch_timestamp_ms = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]
    compressor_on = room_state_int64_vector[IDX_ROOM_STATE_REFRIGERATION_IS_ACTIVE]
    cooling_frac = room_state_float64_vector[IDX_ROOM_STATE_COOLING_FRACTION]
    downtime_safe_for_shutdown = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_SAFE_FOR_SHUTDOWN]
    lp_tripped_prev = room_state_int64_vector[IDX_ROOM_STATE_LP_TRIPPED]
    switch_reason = 0

    # Apply the flow-switch interlock every step, including between thermostat
    # polls; the permit includes the intentional fan-delay and release grace.
    if compressor_on and not evap_airflow_ok:
        compressor_on = 0
        room_state_int64_vector[IDX_ROOM_STATE_LAST_COMPRESSOR_SWITCH_OFF_MS] = epoch_timestamp_ms
        switch_reason = -2

    if epoch_timestamp_ms % equipment_configurations_int64_vector[IDX_EQ_REFRIG_CONTROL_EVALUATION_INTERVAL_MS] == 0:
        # State variables
        # V_dot_air_evap_actual = room_state_float64_vector[IDX_ROOM_STATE_V_DOT_AIR_EVAP_ACTUAL]
        t_room_filtered_val = room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_FILTERED_VAL]
        defrost_phase = room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE]
        
        # Parameters
        t_setpoint = active_crop_params_float64_vector[IDX_FRUIT_PARAM_T_SETPOINT]
        t_deadband = equipment_configurations_float64_vector[IDX_EQ_REFRIG_CONTROL_DEADBAND_C]
        compressor_minimum_idle_time_ms = equipment_configurations_int64_vector[IDX_EQ_REFRIG_CONTROL_MIN_IDLE_MS]
        compressor_minimum_active_time_ms = equipment_configurations_int64_vector[IDX_EQ_REFRIG_CONTROL_MIN_RUN_MS]

        if t_room_filtered_val > t_setpoint + t_deadband:
            room_state_int64_vector[IDX_ROOM_STATE_THERMOSTAT_CALL] = 1
        elif t_room_filtered_val <= t_setpoint:
            room_state_int64_vector[IDX_ROOM_STATE_THERMOSTAT_CALL] = 0

        low_suction_pressure_trip(
            room_state_int64_vector,
            room_state_float64_vector[IDX_ROOM_STATE_P_SUCTION_LINE_FILTERED_VAL],
            active_refrigerant_properties_vector[IDX_REFRIG_SCALAR_LP_TRIP_PRESSURE_CUT_IN_BAR],
            active_refrigerant_properties_vector[IDX_REFRIG_SCALAR_LP_TRIP_PRESSURE_CUT_OUT_BAR]
        )

        is_evap_fan_delayed = defrost_phase == DF_FAN_DELAY

        motors_ok = (
            condenser_fan_motor_ready
            and (evaporator_fan_motor_ready or is_evap_fan_delayed)
        )

        if (
            not compressor_on 
            and not downtime_safe_for_shutdown
            and defrost_phase in (DF_NORMAL, DF_FAN_DELAY)
            and motors_ok and evap_airflow_ok
            and not room_state_int64_vector[IDX_ROOM_STATE_LP_TRIPPED]
        ):
            if room_state_int64_vector[IDX_ROOM_STATE_THERMOSTAT_CALL]:
                if epoch_timestamp_ms - room_state_int64_vector[IDX_ROOM_STATE_LAST_COMPRESSOR_SWITCH_OFF_MS] >= compressor_minimum_idle_time_ms:
                    compressor_on = 1
                    room_state_int64_vector[IDX_ROOM_STATE_LAST_COMPRESSOR_SWITCH_ON_MS] = epoch_timestamp_ms

        elif compressor_on:
            if room_state_int64_vector[IDX_ROOM_STATE_LP_TRIPPED]:
                compressor_on = 0
                room_state_int64_vector[IDX_ROOM_STATE_LAST_COMPRESSOR_SWITCH_OFF_MS] = epoch_timestamp_ms
                switch_reason = -1
            elif (
                not room_state_int64_vector[IDX_ROOM_STATE_THERMOSTAT_CALL]
                or defrost_phase in (DF_WAIT_FOR_STOP, DF_SETTLE)
                or downtime_safe_for_shutdown
            ):
                if epoch_timestamp_ms - room_state_int64_vector[IDX_ROOM_STATE_LAST_COMPRESSOR_SWITCH_ON_MS] >= compressor_minimum_active_time_ms:
                    compressor_on = 0
                    room_state_int64_vector[IDX_ROOM_STATE_LAST_COMPRESSOR_SWITCH_OFF_MS] = epoch_timestamp_ms

        if lp_tripped_prev != room_state_int64_vector[IDX_ROOM_STATE_LP_TRIPPED]:
            row = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_LP_TRIP_EVENT_WRITE_IDX]
            lp_trip_event_ledger[row, IDX_LP_TRIP_EVENT_TIMESTAMP_MS] = epoch_timestamp_ms
            lp_trip_event_ledger[row, IDX_LP_TRIP_EVENT_TRIPPED] = room_state_int64_vector[IDX_ROOM_STATE_LP_TRIPPED]
            room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_LP_TRIP_EVENT_WRITE_IDX] = row + 1

    # Compressor transitions also occur between thermostat polls when the
    # airflow switch opens, so the ledger belongs outside the modulo block.
    if compressor_on != room_state_int64_vector[IDX_ROOM_STATE_REFRIGERATION_IS_ACTIVE]:
        row = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_COMPRESSOR_EVENT_WRITE_IDX]
        compressor_event_ledger[row, IDX_COMPRESSOR_EVENT_TIMESTAMP_MS] = epoch_timestamp_ms
        compressor_event_ledger[row, IDX_COMPRESSOR_EVENT_IS_ACTIVE] = compressor_on
        compressor_event_ledger[row, IDX_COMPRESSOR_EVENT_SWITCH_REASON] = switch_reason
        room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_COMPRESSOR_EVENT_WRITE_IDX] = row + 1
    room_state_int64_vector[IDX_ROOM_STATE_REFRIGERATION_IS_ACTIVE] = compressor_on

    target_cooling = 1.0 if compressor_on else 0.0
    cooling_frac += (target_cooling - cooling_frac) * equipment_configurations_float64_vector[IDX_EQ_DT_OVER_COMPRESSOR_TAU_COOL_SEC]
    room_state_float64_vector[IDX_ROOM_STATE_COOLING_FRACTION] = numba_clip(cooling_frac, SAFE_FLOOR, 1.0)

@njit
def compressor_metrics(
    T_evap_c,
    T_cond_c,
    dt_sh_actual_k,
    T_liquid_outlet_c,
    cooling_frac,
    Q_refrigerant_evap_w,
    W_compressor_rated_w,
    V_disp_m3_rev,
    compressor_speed_hz,
    compressor_idling_friction_factor,
    inv_gamma_refrig,
    C_clearance,
    eta_vol_zero,
    active_refrigerant_surfaces_matrix,
):
    # -----------------------------------------------------------------
    # STEP 1: Evaluate 2D Surface Polynomials (Suction Side)
    # -----------------------------------------------------------------
    # Density (X1 = T_evap, X2 = dt_sh_actual)
    rho_suction = evaluate_poly2d_cubic_fit(
        x1     = T_evap_c,
        x2     = dt_sh_actual_k,
        c      = active_refrigerant_surfaces_matrix[SURFACE_POLY2D_CUBIC_RHO, COEFFS, :],
        bounds = active_refrigerant_surfaces_matrix[SURFACE_POLY2D_CUBIC_RHO, BOUNDS, :],
    )

    # Inlet Enthalpy h1 (X1 = T_evap, X2 = dt_sh_actual)
    h1 = evaluate_poly2d_quadratic_fit(
        x1     = T_evap_c,
        x2     = dt_sh_actual_k,
        c      = active_refrigerant_surfaces_matrix[SURFACE_POLY2D_QUAD_H1, COEFFS, :],
        bounds = active_refrigerant_surfaces_matrix[SURFACE_POLY2D_QUAD_H1, BOUNDS, :],
    )

    # -----------------------------------------------------------------
    # STEP 2: Evaluate 2D Surface Polynomial (Discharge Side)
    # -----------------------------------------------------------------
    # Ideal Isentropic Enthalpy h2s (X1 = T_evap, X2 = T_cond)
    h2s = evaluate_poly3d_quadratic_fit(
        x1     = T_evap_c,
        x2     = dt_sh_actual_k,
        x3     = T_cond_c,
        c      = active_refrigerant_surfaces_matrix[SURFACE_POLY3D_QUAD_H2S, COEFFS, :],
        bounds = active_refrigerant_surfaces_matrix[SURFACE_POLY3D_QUAD_H2S, BOUNDS, :],
    )

    # -----------------------------------------------------------------
    # STEP 3: Evaluate 1D Polynomial (Liquid Line)
    # -----------------------------------------------------------------
    # Liquid Enthalpy h4 entering Evaporator (Cubic fit of T_cond)
    h4 = evaluate_poly1d_fit(
        x      = T_liquid_outlet_c,
        c      = active_refrigerant_surfaces_matrix[SURFACE_POLY1D_CUBIC_H4, COEFFS, :],
        bounds = active_refrigerant_surfaces_matrix[SURFACE_POLY1D_CUBIC_H4, BOUNDS, :],
    )

    # -----------------------------------------------------------------
    # STEP 4: Convert Thermal Capacity to Refrigerant Mass Flow
    # -----------------------------------------------------------------
    # Enthalpy delta across evaporator (J/kg). Max safeguard keeps it positive.
    dh_evap_j_kg = max(h1 - h4, 5.0)
    m_dot_refrig_ref = Q_refrigerant_evap_w / dh_evap_j_kg

    # -----------------------------------------------------------------
    # STEP 5: Calculate Dynamic Efficiencies & Power Boundaries
    # -----------------------------------------------------------------
    eta_vol, log_P_evap_bar, log_P_cond_bar, pressure_ratio = compressor_volumetric_efficiency(
        T_evap_c,
        T_cond_c,
        active_refrigerant_surfaces_matrix[SURFACE_ANTOINE_EVAP, COEFFS, :],
        active_refrigerant_surfaces_matrix[SURFACE_ANTOINE_COND, COEFFS, :],
        inv_gamma_refrig,
        C_clearance,
        eta_vol_zero,
    )

    # Calculate maximum hardware mass flow capacity at this operating point
    m_dot_refrig_max = V_disp_m3_rev * compressor_speed_hz * rho_suction * eta_vol # * cooling_frac

    dh_isen_j_kg = max(h2s - h1, 0.1)

    eta_isen = evaluate_poly1d_fit(
        pressure_ratio,
        active_refrigerant_surfaces_matrix[SURFACE_POLY1D_QUAD_ETA_ISEN, COEFFS, :],
        active_refrigerant_surfaces_matrix[SURFACE_POLY1D_QUAD_ETA_ISEN, BOUNDS, :],
    )
    eta_isen = numba_clip(eta_isen, 0.35, 0.70)

    # Actual Electrical Power Consumption drawn from the wall (Watts)
    m_dot_refrig_actual = min(m_dot_refrig_ref, m_dot_refrig_max)
    W_fluid_isentropic_w = m_dot_refrig_actual * dh_isen_j_kg
    W_indicated_w = W_fluid_isentropic_w / eta_isen
    W_motor_friction_overhead = W_compressor_rated_w * compressor_idling_friction_factor

    W_compressor_w = max(W_indicated_w, W_motor_friction_overhead)

    # Total heat rejected into the condenser loop (Watts)
    # Follows the first law of thermodynamics for hermetic systems
    Q_cooling_delivered_w = m_dot_refrig_actual * dh_evap_j_kg
    Q_refrigerant_cond_w = Q_cooling_delivered_w + W_compressor_w

    # Calculate real-world cop
    cop = Q_cooling_delivered_w / max(W_compressor_w, 1.0)

    return (
        cop,
        W_compressor_w,
        Q_refrigerant_cond_w,
        rho_suction,
        h2s,
        h1,
        h4,
        m_dot_refrig_actual,
    )


@njit
def execute_solver_loop(
    room_state_float64_vector,
    room_configurations_float64_vector,
    room_configurations_int64_vector,
    equipment_configurations_float64_vector,
    active_refrigerant_properties_vector,
    active_refrigerant_surfaces_matrix,
    W_compressor_rated_w,
    Q_refrigerant_evap_w,
    dt_sh_actual_k,
    sentinel=SENTINEL_FLOAT64,
):
    cooling_frac          = room_state_float64_vector[IDX_ROOM_STATE_COOLING_FRACTION]
    inv_valve_prop_band_k = equipment_configurations_float64_vector[IDX_EQ_INV_MAIN_VALVE_PROP_BAND_K]
    T_coil_evap_c         = room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAPORATOR]
    T_coil_cond_c         = room_state_float64_vector[IDX_ROOM_STATE_T_COIL_CONDENSER]
    T_cond_c              = room_state_float64_vector[IDX_ROOM_STATE_T_COND]
    T_liquid_outlet_c     = room_state_float64_vector[IDX_ROOM_STATE_T_LIQUID_OUTLET]
    T_valve_setpoint_c    = equipment_configurations_float64_vector[IDX_EQ_MAIN_VALVE_SETPOINT_C]
    subcooling_approach_k = equipment_configurations_float64_vector[IDX_EQ_MAIN_VALVE_SUBCOOL_APPROACH_K]    
    V_disp_m3_rev         = equipment_configurations_float64_vector[IDX_EQ_COMPRESSOR_V_DISPLACEMENT_M3]
    compressor_speed_hz   = equipment_configurations_float64_vector[IDX_EQ_COMPRESSOR_SPEED_HZ]
    compressor_idling_friction_factor = equipment_configurations_float64_vector[IDX_EQ_COMPRESSOR_IDLING_FRICTION]
    inv_UA_internal_evap_w_k      = active_refrigerant_properties_vector[IDX_REFRIG_SCALAR_INV_UA_COIL_EVAP_REFRIG_SIDE]
    inv_UA_internal_cond_w_k      = active_refrigerant_properties_vector[IDX_REFRIG_SCALAR_INV_UA_COIL_COND_REFRIG_SIDE]
    two_over_UA_internal_cond_w_k = active_refrigerant_properties_vector[IDX_REFRIG_SCALAR_TWO_OVER_UA_COIL_COND_REFRIG_SIDE]
    C_clearance = active_refrigerant_properties_vector[IDX_REFRIG_SCALAR_C_CLEARANCE]
    eta_vol_zero = active_refrigerant_properties_vector[IDX_REFRIG_SCALAR_ETA_VOL_ZERO]
    
    # Refrigerant constants
    inv_gamma_refrig   = active_refrigerant_properties_vector[IDX_REFRIG_SCALAR_INV_GAMMA]
    c_antoine_evap     = active_refrigerant_surfaces_matrix[SURFACE_ANTOINE_EVAP, COEFFS, :]
    c_antoine_cond     = active_refrigerant_surfaces_matrix[SURFACE_ANTOINE_COND, COEFFS, :]
    c_h1               = active_refrigerant_surfaces_matrix[SURFACE_POLY2D_QUAD_H1, COEFFS, :]
    bounds_h1          = active_refrigerant_surfaces_matrix[SURFACE_POLY2D_QUAD_H1, BOUNDS, :]
    c_h4               = active_refrigerant_surfaces_matrix[SURFACE_POLY1D_CUBIC_H4, COEFFS, :]
    bounds_h4          = active_refrigerant_surfaces_matrix[SURFACE_POLY1D_CUBIC_H4, BOUNDS, :]

    # Avoid unbounded-type errors:
    Q_refrigerant_evap_w_old = sentinel
    Q_refrigerant_cond_w     = sentinel
    T_evap_c                 = sentinel
    log_P_evap_bar           = sentinel
    log_P_cond_bar           = sentinel
    cop                      = sentinel
    W_compressor_w           = sentinel
    h2s_metrics              = sentinel
    h1_metrics               = sentinel
    h1                       = sentinel
    m_dot_refrig_max         = sentinel

    solver_tolerance = room_configurations_float64_vector[IDX_ROOM_CFG_NUMERICAL_SOLVER_TOLERANCE]
    max_iterations = max(room_configurations_int64_vector[IDX_ROOM_CFG_MAX_NUMERICAL_SOLVER_ITERS], 2)
    n_iters = 0
    for _ in range(max_iterations):
        Q_refrigerant_evap_w_old = Q_refrigerant_evap_w

        # A. Evaporator supply side
        T_evap_c = T_coil_evap_c - Q_refrigerant_evap_w * inv_UA_internal_evap_w_k

        # B. Compressor Metrics & Electrical Power
        (
            cop,
            W_compressor_w,
            Q_refrigerant_cond_w,
            rho_suction,
            h2s_metrics,
            h1_metrics,
            h4_metrics,
            m_dot_refrig_actual,
        ) = compressor_metrics(
            T_evap_c,
            T_cond_c,
            dt_sh_actual_k,
            T_liquid_outlet_c,
            cooling_frac,
            Q_refrigerant_evap_w,
            W_compressor_rated_w,
            V_disp_m3_rev,
            compressor_speed_hz,
            compressor_idling_friction_factor,
            inv_gamma_refrig,
            C_clearance,
            eta_vol_zero,
            active_refrigerant_surfaces_matrix,
        )

        # C. Condenser Side Temperature Update
        T_cond_natural_c = T_coil_cond_c + Q_refrigerant_cond_w * inv_UA_internal_cond_w_k
        
        if T_cond_natural_c < T_valve_setpoint_c:
            valve_flood_frac = (T_valve_setpoint_c - T_cond_natural_c) * inv_valve_prop_band_k
            valve_flood_frac = numba_clip(valve_flood_frac, SAFE_FLOOR, 1.0)
        else:
            valve_flood_frac = 0.0

        T_cond_c = T_coil_cond_c + (Q_refrigerant_cond_w * two_over_UA_internal_cond_w_k) / (2.0 - valve_flood_frac)
        # / (UA_internal_cond_w_k * (1.0 - 0.5 * valve_flood_frac))
        T_liquid_outlet_c = T_cond_c - valve_flood_frac * (T_cond_c - (T_coil_cond_c + subcooling_approach_k))
        T_liquid_outlet_c = numba_clip(T_liquid_outlet_c, low=T_coil_cond_c, high=T_cond_c)

        # D. Re-calculate actual compressor pumping limit
        eta_vol_actual, log_P_evap_bar, log_P_cond_bar, _ = compressor_volumetric_efficiency(
            T_evap_c,
            T_cond_c,
            c_antoine_evap,
            c_antoine_cond,
            inv_gamma_refrig,
            C_clearance,
            eta_vol_zero,
        )

        # E. Calculate Maximum Refrigerant Mass Flow Capability (kg/s)
        m_dot_refrig_max = V_disp_m3_rev * compressor_speed_hz * rho_suction * eta_vol_actual # * cooling_frac

        # F. Set Absolute Cooling Capacity Limit
        h1 = evaluate_poly2d_quadratic_fit(
            T_evap_c,
            dt_sh_actual_k,
            c_h1,
            bounds_h1,
        )

        h4 = evaluate_poly1d_fit(
            T_liquid_outlet_c,
            c_h4,
            bounds_h4,
        )

        Q_refrigerant_evap_w = m_dot_refrig_max * (h1 - h4) * cooling_frac

        n_iters += 1
        if n_iters > 1 and abs(Q_refrigerant_evap_w - Q_refrigerant_evap_w_old) < solver_tolerance:
            break
    absolute_error = abs(Q_refrigerant_evap_w - Q_refrigerant_evap_w_old)
    relative_error = abs(Q_refrigerant_evap_w - Q_refrigerant_evap_w_old) / Q_refrigerant_evap_w

    return (
        Q_refrigerant_evap_w,
        Q_refrigerant_cond_w,
        T_evap_c,
        T_cond_c,
        T_liquid_outlet_c,
        log_P_evap_bar,
        log_P_cond_bar,
        cop,
        W_compressor_w,
        valve_flood_frac,
        h2s_metrics,
        h1_metrics,
        h4_metrics,
        # m_dot_refrig_actual,
        h1,
        h4,
        m_dot_refrig_max,
        absolute_error,
        relative_error,
        n_iters,
    )


@njit
def execute_refrigeration_cycle(
    room_state_float64_vector,
    room_state_int64_vector,
    forcing_cache_float64_vector,
    room_configurations_float64_vector,
    room_configurations_int64_vector,
    equipment_configurations_float64_vector,
    equipment_configurations_int64_vector,
    active_refrigerant_properties_vector,
    active_refrigerant_surfaces_matrix,
    sentinel=SENTINEL_FLOAT64,
):
    if not room_state_int64_vector[IDX_ROOM_STATE_REFRIGERATION_IS_ACTIVE]:
        # -----------------------------------------------------
        # BRANCH A: COMPRESSOR IS IDLE (Pure State-Variable Mass)
        # -----------------------------------------------------
        # Compressor forces are dead
        Q_refrigerant_evap = 0.0
        Q_refrigerant_cond = 0.0
        W_compressor = 0.0
        valve_flood_frac = 0.0
        # dt_valve_penalty = 0.0
        cop = sentinel
        T_evap = room_state_float64_vector[IDX_ROOM_STATE_T_EVAP]
        T_cond = room_state_float64_vector[IDX_ROOM_STATE_T_COND]
        T_evap += equipment_configurations_float64_vector[IDX_EQ_EVAP_TEMP_RELAX_ALPHA] * (room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAPORATOR] - T_evap) # coil is warmer, so t_evap drifts upward
        T_cond += equipment_configurations_float64_vector[IDX_EQ_COND_TEMP_RELAX_ALPHA] * (room_state_float64_vector[IDX_ROOM_STATE_T_COIL_CONDENSER] - T_cond) # coil is colder, so t_cond drift downward
        T_liquid_outlet = T_cond
        log_P_evap_bar = evaluate_antoine_fit(T_evap, active_refrigerant_surfaces_matrix[SURFACE_ANTOINE_EVAP, COEFFS, :])
        log_P_cond_bar = evaluate_antoine_fit(T_cond, active_refrigerant_surfaces_matrix[SURFACE_ANTOINE_COND, COEFFS, :])
        sh_pid_steps = 0.0      # Physically drive the EEV closed
        sh_error = 0.0        # Clear history
        sh_i_term = 0.0           # RESET ACCUMULATOR: Prevents windup while off
        dt_sh_actual = equipment_configurations_float64_vector[IDX_EQ_EEV_CLOSED_VALVE_SH_K]     # Set a safe, closed-valve target state
        m_dot_refrig_max = 0.0
        n_iters = 0
        absolute_error = sentinel
    else:
        # -----------------------------------------------------
        # BRANCH B: COMPRESSOR IS ACTIVE (Execute Solver Loop)
        # -----------------------------------------------------
        if room_state_float64_vector[IDX_ROOM_STATE_Q_REFRIG_EVAP] == 0.0:
            # Case 1: Compressor just woke up.
            # Seed with reasonable values
            Q_refrigerant_evap = (
                room_state_float64_vector[IDX_ROOM_STATE_RATED_COMPRESSOR_Q_COOLING] 
                * (1.0 - 0.007 * (forcing_cache_float64_vector[IDX_CACHE_T_AMBIENT] - 35.0))
                * max(room_state_float64_vector[IDX_ROOM_STATE_COOLING_FRACTION], 0.05)
            )
        else:
            # Case 2: Compressor was already running normally.
            # Pull forward true historical timeline data for instant 1-step convergence.
            Q_refrigerant_evap = room_state_float64_vector[IDX_ROOM_STATE_Q_REFRIG_EVAP]

        # T_cond = room_state_float64_vector[IDX_ROOM_STATE_T_COND]
        # T_liquid_outlet = room_state_float64_vector[IDX_ROOM_STATE_T_LIQUID_LINE]

        if room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS] % equipment_configurations_int64_vector[IDX_EQ_EEV_PID_INTERVAL_MS] == 0:
            t_evap_sensor_calculated = invert_antoine_fit(
                math.log(room_state_float64_vector[IDX_ROOM_STATE_P_SUCTION_LINE_FILTERED_VAL]),
                active_refrigerant_surfaces_matrix[SURFACE_ANTOINE_EVAP, COEFFS, :],
            )
            dt_sh_sensor_calculated = room_state_float64_vector[IDX_ROOM_STATE_T_SUCTION_LINE_FILTERED_VAL] - t_evap_sensor_calculated
            current_sh_error = dt_sh_sensor_calculated - equipment_configurations_float64_vector[IDX_EQ_EEV_SUPERHEAT_K]

            (
                sh_error,
                sh_p_term,
                sh_i_term,
                sh_d_term,
            ) = pid_evaluation(
                current_error        = current_sh_error,
                dt_sec               = equipment_configurations_float64_vector[IDX_EQ_EEV_PID_INTERVAL_SEC],
                previous_error       = room_state_float64_vector[IDX_ROOM_STATE_EEV_SH_ERROR],
                integral_accumulator = room_state_float64_vector[IDX_ROOM_STATE_EEV_INTEGRAL_TERM], 
                K_p                  = equipment_configurations_float64_vector[IDX_EQ_EEV_PID_K_P],
                K_i                  = equipment_configurations_float64_vector[IDX_EQ_EEV_PID_K_I],
                K_d                  = equipment_configurations_float64_vector[IDX_EQ_EEV_PID_K_D],
                min_pid_output       = equipment_configurations_float64_vector[IDX_EQ_EEV_PID_MIN_OUTPUT],
                max_pid_output       = equipment_configurations_float64_vector[IDX_EQ_EEV_PID_MAX_OUTPUT],
            )
            room_state_float64_vector[IDX_ROOM_STATE_EEV_SH_ERROR] = sh_error
            room_state_float64_vector[IDX_ROOM_STATE_EEV_PROPORTIONAL_TERM] = sh_p_term
            room_state_float64_vector[IDX_ROOM_STATE_EEV_INTEGRAL_TERM] = sh_i_term
            room_state_float64_vector[IDX_ROOM_STATE_EEV_DERIVATIVE_TERM] = sh_d_term

        sh_pid_steps = (
            room_state_float64_vector[IDX_ROOM_STATE_EEV_PROPORTIONAL_TERM] +
            room_state_float64_vector[IDX_ROOM_STATE_EEV_INTEGRAL_TERM] +
            room_state_float64_vector[IDX_ROOM_STATE_EEV_DERIVATIVE_TERM]
        )
        sh_pid_steps = numba_clip(
            sh_pid_steps,
            equipment_configurations_float64_vector[IDX_EQ_EEV_PID_MIN_OUTPUT],
            equipment_configurations_float64_vector[IDX_EQ_EEV_PID_MAX_OUTPUT],
        )
        valve_opening_frac = sh_pid_steps * equipment_configurations_float64_vector[IDX_EQ_EEV_INV_PID_MAX_OUTPUT]
        dt_sh_actual = 2.0 + 18.0 * ((1.0 - valve_opening_frac) ** 2.0)

        # UA_internal_cond = forcing_cache_float64_vector[IDX_CACHE_UA_INTERNAL_COND]
        W_compressor_rated = room_state_float64_vector[IDX_ROOM_STATE_RATED_COMPRESSOR_W_ELECT]
        # Execute solver
        (
            Q_refrigerant_evap,
            Q_refrigerant_cond,
            T_evap,
            T_cond,
            T_liquid_outlet,
            log_P_evap_bar,
            log_P_cond_bar,
            cop,
            W_compressor,
            valve_flood_frac,
            h2s_metrics,
            h1_metrics,
            h4_metrics,
            h1,
            h4,
            m_dot_refrig_max,
            absolute_error,
            relative_error,
            n_iters,
        ) = execute_solver_loop(
            room_state_float64_vector,
            room_configurations_float64_vector,
            room_configurations_int64_vector,
            equipment_configurations_float64_vector,
            active_refrigerant_properties_vector,
            active_refrigerant_surfaces_matrix,
            W_compressor_rated,
            Q_refrigerant_evap,
            dt_sh_actual,
        )

    P_evap_bar = math.exp(log_P_evap_bar)
    P_cond_bar = math.exp(log_P_cond_bar)
    T_evap_outlet = T_evap + dt_sh_actual

    # Update state vector
    room_state_float64_vector[IDX_ROOM_STATE_Q_REFRIG_EVAP] = Q_refrigerant_evap
    room_state_float64_vector[IDX_ROOM_STATE_Q_REFRIG_COND] = Q_refrigerant_cond
    room_state_float64_vector[IDX_ROOM_STATE_T_EVAP] = T_evap
    room_state_float64_vector[IDX_ROOM_STATE_T_COND] = T_cond
    room_state_float64_vector[IDX_ROOM_STATE_T_LIQUID_OUTLET] = T_liquid_outlet

    return cop, P_evap_bar, P_cond_bar, T_evap_outlet, W_compressor, m_dot_refrig_max, valve_flood_frac, n_iters, absolute_error

@njit
def thermal_node_temperature_from_energy(E_j, C_ice_or_metal_j_k, phase_mass_kg):
    """Temperature for a fixed-mass metal or ice/water cell during one heat step.

    Ice energy is measured from ice at 0 C. Positive energy first melts ice
    at 0 C; only energy above the full fusion requirement warms liquid water.
    Meltwater is removed from the ice inventory after the heat step.
    """
    if phase_mass_kg == 0.0:  # metal
        return E_j / C_ice_or_metal_j_k
    if E_j <= 0.0:
        return E_j / C_ice_or_metal_j_k
    E_full_melt_j = phase_mass_kg * H_FUSION
    if E_j <= E_full_melt_j:
        return 0.0
    return (E_j - E_full_melt_j) / (phase_mass_kg * CP_LIQ_WATER)


@njit
def phase_aware_pair_energy_delta(
    E_a_j,
    E_b_j,
    C_a_j_k,
    C_b_j_k,
    phase_mass_a_kg,
    phase_mass_b_kg,
    conductance_w_k,
    dt_sec,
):
    """Energy from a to b; conserve pair energy across the 0 C melt plateau.

    The existing exact two-capacity solution is retained when both cells stay
    in their sensible ranges. At a phase crossing, solve the monotone backward
    Euler pair equation with temperatures evaluated at the new energies.
    """
    if conductance_w_k <= 0.0 or dt_sec <= 0.0:
        return 0.0

    T_a = thermal_node_temperature_from_energy(E_a_j, C_a_j_k, phase_mass_a_kg)
    T_b = thermal_node_temperature_from_energy(E_b_j, C_b_j_k, phase_mass_b_kg)
    inv_C_a = 1.0 / C_a_j_k
    inv_C_b = 1.0 / C_b_j_k
    dE_sensible_j = calculate_energy_state_delta(
        T_a, T_b, C_a_j_k, inv_C_a, C_b_j_k, inv_C_b,
        conductance_w_k, dt_sec,
    )
    a_stays_sensible = phase_mass_a_kg == 0.0 or (
        E_a_j <= 0.0 and E_a_j - dE_sensible_j <= 0.0
    )
    b_stays_sensible = phase_mass_b_kg == 0.0 or (
        E_b_j <= 0.0 and E_b_j + dE_sensible_j <= 0.0
    )
    if a_stays_sensible and b_stays_sensible:
        return dE_sensible_j

    G_dt = conductance_w_k * dt_sec
    dE_explicit_j = G_dt * (T_a - T_b)
    if dE_explicit_j >= 0.0:
        lo = 0.0
        hi = dE_explicit_j
    else:
        lo = dE_explicit_j
        hi = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        T_a_new = thermal_node_temperature_from_energy(
            E_a_j - mid, C_a_j_k, phase_mass_a_kg,
        )
        T_b_new = thermal_node_temperature_from_energy(
            E_b_j + mid, C_b_j_k, phase_mass_b_kg,
        )
        residual = mid - G_dt * (T_a_new - T_b_new)
        if residual > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@njit
def update_evaporator_energies(
    room_state_float64_vector,
    equipment_configurations_float64_vector,
    passage_thickness_m,
    face_thickness_m,
    face_covered_area_m2,
    face_contact_fraction,
    face_bridge_length_m,
    Q_coil_defrost_heater_w,
    dE_air_forced_j,
    dE_face_bare_j,
    Q_surface_to_outer_w,
    dE_ice_mass_p_j,
    dE_ice_mass_i_j,
    dE_ice_mass_o_j,
    dt_engine_sec,
):
    # Inputs ending in _j are step energies. Q_surface_to_outer_w is a rate,
    # converted to joules below. The air and mass exchange code supplies them;
    # this function routes energy through the active metal/ice thermal nodes.
    M_passage = room_state_float64_vector[IDX_ROOM_STATE_M_PASSAGE_ICE_KG]
    V_passage = room_state_float64_vector[IDX_ROOM_STATE_V_PASSAGE_ICE_M3]
    M_face = room_state_float64_vector[IDX_ROOM_STATE_M_FACE_ICE_KG]
    V_face = room_state_float64_vector[IDX_ROOM_STATE_V_FACE_ICE_M3]
    has_passage = M_passage > 0.0
    has_face = M_face > 0.0

    # An absent node has no capacity or conductance. Catch inconsistent inputs
    # instead of losing their energy by silently routing them to nowhere.
    if has_passage:
        if V_passage <= 0.0 or passage_thickness_m <= 0.0:
            raise ValueError("passage ice needs positive volume and thickness")
    elif dE_ice_mass_p_j != 0.0:
        raise ValueError("passage mass energy requires passage ice")
    if has_face:
        if V_face <= 0.0 or face_thickness_m <= 0.0 or face_covered_area_m2 <= 0.0:
            raise ValueError("face ice needs positive volume, thickness, and covered area")
    elif dE_ice_mass_i_j != 0.0 or dE_ice_mass_o_j != 0.0 or Q_surface_to_outer_w != 0.0:
        raise ValueError("face energy input requires face ice")

    C_metal = equipment_configurations_float64_vector[IDX_EQ_EVAP_THERMAL_MASS_METAL]
    if C_metal <= 0.0:
        raise ValueError("evaporator metal capacity must be positive")
    inv_C_metal = 1.0 / C_metal
    E_metal = C_metal * room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAPORATOR]
    E_metal += (
        Q_coil_defrost_heater_w
        - room_state_float64_vector[IDX_ROOM_STATE_Q_REFRIG_EVAP]
    ) * dt_engine_sec

    E_passage = 0.0
    C_passage = 0.0
    inv_C_passage = 0.0
    G_mp = 0.0
    k_passage = 0.0
    if has_passage:
        E_passage = room_state_float64_vector[IDX_ROOM_STATE_E_PASSAGE_ICE_J]
        _, C_passage, inv_C_passage = passage_frost_temperatures_and_capacities(M_passage, E_passage)
        k_passage = frost_conductivity_w_m_k(M_passage / V_passage)
        G_mp = frost_conductance_full_contact_layers(
            equipment_configurations_float64_vector[IDX_EQ_EVAP_SURFACE_AREA_M2],
            passage_thickness_m,
            k_passage,
        )
        E_passage += dE_air_forced_j + dE_face_bare_j + dE_ice_mass_p_j
    else:
        E_metal += dE_air_forced_j + dE_face_bare_j

    E_inner = 0.0
    E_outer = 0.0
    C_face_half = 0.0
    inv_C_face_half = 0.0
    G_pi = 0.0
    G_mi = 0.0
    G_io = 0.0
    if has_face:
        E_inner = room_state_float64_vector[IDX_ROOM_STATE_E_FACE_INNER_ICE_J] + dE_ice_mass_i_j
        E_outer = (
            room_state_float64_vector[IDX_ROOM_STATE_E_FACE_OUTER_ICE_J]
            + Q_surface_to_outer_w * dt_engine_sec
            + dE_ice_mass_o_j
        )
        _, _, C_face_half, inv_C_face_half = face_frost_temperatures_and_capacities(
            M_face, E_inner, E_outer,
        )
        k_face = frost_conductivity_w_m_k(M_face / V_face)
        G_io = frost_conductance_full_contact_layers(face_covered_area_m2, face_thickness_m, k_face)
        if has_passage:
            G_pi = frost_conductance_passage_to_inner_face(
                k_passage,
                face_covered_area_m2,
                face_contact_fraction,
                face_bridge_length_m,
                face_thickness_m,
                k_face,
            )
        else:
            # With no passage ice, contact is limited to the exposed fin edges.
            G_mi = frost_conductance_metal_to_inner_face(
                k_face,
                face_covered_area_m2,
                equipment_configurations_float64_vector[IDX_EQ_EVAP_FIN_EDGE_CONTACT_FACTOR],
                face_thickness_m,
            )

    # Two half sweeps reduce link-order bias. Pair transfers conserve energy;
    # the phase-aware helper holds ice/water at 0 C during partial melting.
    half_dt = 0.5 * dt_engine_sec
    M_face_half = 0.5 * M_face
    for sweep in range(2):
        if sweep == 0:
            if has_passage:
                dE_m_to_p = phase_aware_pair_energy_delta(
                    E_metal, E_passage,
                    C_metal, C_passage, 0.0, M_passage, G_mp, half_dt,
                )
                E_metal -= dE_m_to_p
                E_passage += dE_m_to_p
            if has_face:
                if has_passage:
                    dE_p_to_i = phase_aware_pair_energy_delta(
                        E_passage, E_inner,
                        C_passage, C_face_half, M_passage, M_face_half, G_pi, half_dt,
                    )
                    E_passage -= dE_p_to_i
                    E_inner += dE_p_to_i
                else:
                    dE_m_to_i = phase_aware_pair_energy_delta(
                        E_metal, E_inner,
                        C_metal, C_face_half, 0.0, M_face_half, G_mi, half_dt,
                    )
                    E_metal -= dE_m_to_i
                    E_inner += dE_m_to_i
                dE_i_to_o = phase_aware_pair_energy_delta(
                    E_inner, E_outer,
                    C_face_half, C_face_half, M_face_half, M_face_half, G_io, half_dt,
                )
                E_inner -= dE_i_to_o
                E_outer += dE_i_to_o
        else:
            if has_face:
                dE_i_to_o = phase_aware_pair_energy_delta(
                    E_inner, E_outer,
                    C_face_half, C_face_half, M_face_half, M_face_half, G_io, half_dt,
                )
                E_inner -= dE_i_to_o
                E_outer += dE_i_to_o
                if has_passage:
                    dE_p_to_i = phase_aware_pair_energy_delta(
                        E_passage, E_inner,
                        C_passage, C_face_half, M_passage, M_face_half, G_pi, half_dt,
                    )
                    E_passage -= dE_p_to_i
                    E_inner += dE_p_to_i
                else:
                    dE_m_to_i = phase_aware_pair_energy_delta(
                        E_metal, E_inner,
                        C_metal, C_face_half, 0.0, M_face_half, G_mi, half_dt,
                    )
                    E_metal -= dE_m_to_i
                    E_inner += dE_m_to_i
            if has_passage:
                dE_m_to_p = phase_aware_pair_energy_delta(
                    E_metal, E_passage,
                    C_metal, C_passage, 0.0, M_passage, G_mp, half_dt,
                )
                E_metal -= dE_m_to_p
                E_passage += dE_m_to_p

    room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAPORATOR] = E_metal * inv_C_metal
    room_state_float64_vector[IDX_ROOM_STATE_E_PASSAGE_ICE_J] = E_passage
    room_state_float64_vector[IDX_ROOM_STATE_E_FACE_INNER_ICE_J] = E_inner
    room_state_float64_vector[IDX_ROOM_STATE_E_FACE_OUTER_ICE_J] = E_outer

@njit
def update_condenser_temperature(
    room_state_float64_vector,
    forcing_cache_float64_vector,
    equipment_configurations_float64_vector,
    C_cond,
    epsilon_cond,
):
    T_coil_cond = room_state_float64_vector[IDX_ROOM_STATE_T_COIL_CONDENSER]
    Q_out_to_ambient = C_cond * epsilon_cond * (
        T_coil_cond - forcing_cache_float64_vector[IDX_CACHE_T_AMBIENT]
    )
    dT_coil_cond = (
        room_state_float64_vector[IDX_ROOM_STATE_Q_REFRIG_COND]
        - Q_out_to_ambient
    ) * equipment_configurations_float64_vector[IDX_EQ_DT_OVER_COND_THERMAL_MASS_METAL]
    room_state_float64_vector[IDX_ROOM_STATE_T_COIL_CONDENSER] += dT_coil_cond

@njit
def clear_passage_ice_that_melts_within_step(
    room_state_float64_vector,
    equipment_configurations_float64_vector,
    dt_engine_sec,
):
    """Resolve a passage-ice disappearance before selecting the air boundary.

    A vanishingly thin 0 C layer can melt through metal contact in less than
    one engine step. If it stays in the air boundary for the whole step, the
    forced-air load is evaluated at 0 C even after the metal is exposed.
    """
    M_passage = room_state_float64_vector[IDX_ROOM_STATE_M_PASSAGE_ICE_KG]
    if M_passage <= 0.0:
        return 0.0

    T_metal = room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAPORATOR]
    if T_metal <= 0.0:
        return 0.0

    V_passage = room_state_float64_vector[IDX_ROOM_STATE_V_PASSAGE_ICE_M3]
    if V_passage <= 0.0:
        raise ValueError("passage ice needs positive volume")
    E_passage = room_state_float64_vector[IDX_ROOM_STATE_E_PASSAGE_ICE_J]
    T_passage, _, _ = passage_frost_temperatures_and_capacities(M_passage, E_passage)
    if T_metal <= T_passage:
        return 0.0

    C_metal = equipment_configurations_float64_vector[IDX_EQ_EVAP_THERMAL_MASS_METAL]
    energy_to_melt_j = -E_passage + M_passage * H_FUSION
    if energy_to_melt_j <= 0.0 or C_metal * T_metal < energy_to_melt_j:
        return 0.0

    A_passage = equipment_configurations_float64_vector[IDX_EQ_EVAP_SURFACE_AREA_M2]
    thickness_m = V_passage / A_passage
    k_passage = frost_conductivity_w_m_k(M_passage / V_passage)
    G_metal_to_passage = frost_conductance_full_contact_layers(
        A_passage, thickness_m, k_passage,
    )
    if G_metal_to_passage * (T_metal - T_passage) * dt_engine_sec < energy_to_melt_j:
        return 0.0

    room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAPORATOR] = (
        C_metal * T_metal - energy_to_melt_j
    ) / C_metal
    room_state_float64_vector[IDX_ROOM_STATE_M_PASSAGE_ICE_KG] = 0.0
    room_state_float64_vector[IDX_ROOM_STATE_V_PASSAGE_ICE_M3] = 0.0
    room_state_float64_vector[IDX_ROOM_STATE_E_PASSAGE_ICE_J] = 0.0
    return M_passage


@njit
def update_evaporator_state(
    room_state_float64_vector,
    forcing_cache_float64_vector,
    equipment_configurations_float64_vector,
    epsilon_evap,
    m_dot_air_evap,
    inv_m_dot_air_evap,
    C_evap,
    inv_C_evap,
    C_room,
    dt_engine_sec,
):
    # A sub-microgram numerical deposit cannot be represented as a thermal
    # layer: V/A then approaches zero and its conductance diverges. Remove
    # these unresolved ice inventories before any geometry or heat transfer.
    if 0.0 < room_state_float64_vector[IDX_ROOM_STATE_M_PASSAGE_ICE_KG] <= MASS_KG_FLOOR:
        room_state_float64_vector[IDX_ROOM_STATE_M_PASSAGE_ICE_KG] = 0.0
        room_state_float64_vector[IDX_ROOM_STATE_V_PASSAGE_ICE_M3] = 0.0
        room_state_float64_vector[IDX_ROOM_STATE_E_PASSAGE_ICE_J] = 0.0
    if 0.0 < room_state_float64_vector[IDX_ROOM_STATE_M_FACE_ICE_KG] <= MASS_KG_FLOOR:
        room_state_float64_vector[IDX_ROOM_STATE_M_FACE_ICE_KG] = 0.0
        room_state_float64_vector[IDX_ROOM_STATE_V_FACE_ICE_M3] = 0.0
        room_state_float64_vector[IDX_ROOM_STATE_E_FACE_INNER_ICE_J] = 0.0
        room_state_float64_vector[IDX_ROOM_STATE_E_FACE_OUTER_ICE_J] = 0.0
        room_state_float64_vector[IDX_ROOM_STATE_FACE_BOND_MELTED_KG] = 0.0

    dm_passage_melted_before_exchange = clear_passage_ice_that_melts_within_step(
        room_state_float64_vector,
        equipment_configurations_float64_vector,
        dt_engine_sec,
    )

    # Calculate exchange terms using the surface that remains exposed this step.
    (
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
    ) = calculate_evaporator_heat_and_moisture_exchange(
        room_state_float64_vector,
        forcing_cache_float64_vector,
        equipment_configurations_float64_vector,
        epsilon_evap,
        m_dot_air_evap,
        inv_m_dot_air_evap,
        C_evap,
        inv_C_evap,
        dt_engine_sec,
    )

    M_passage_prev = room_state_float64_vector[IDX_ROOM_STATE_M_PASSAGE_ICE_KG]
    V_passage_prev = room_state_float64_vector[IDX_ROOM_STATE_V_PASSAGE_ICE_M3]
    M_face_prev = room_state_float64_vector[IDX_ROOM_STATE_M_FACE_ICE_KG]
    V_face_prev = room_state_float64_vector[IDX_ROOM_STATE_V_FACE_ICE_M3]

    passage_thickness, face_thickness, face_covered_area, _ = frost_geometry(
        V_passage_prev,
        V_face_prev,
        equipment_configurations_float64_vector[IDX_EQ_EVAP_INV_SURFACE_AREA_M2],
        equipment_configurations_float64_vector[IDX_EQ_EVAP_FRONTAL_AREA_M2],
        equipment_configurations_float64_vector[IDX_EQ_EVAP_INV_FRONTAL_AREA_M2],
        equipment_configurations_float64_vector[IDX_EQ_EVAP_INV_FACE_BRIDGE_LENGTH_M],
    )

    Q_defrost_total_w = room_state_float64_vector[IDX_ROOM_STATE_Q_DEFROST_HEATER_W]
    Q_defrost_pan_w = min(
        Q_defrost_total_w,
        equipment_configurations_float64_vector[IDX_EQ_DEFROST_PAN_HEATER_W],
    )
    Q_defrost_coil_w = Q_defrost_total_w - Q_defrost_pan_w

    # The three mass-energy inputs are zero here because deposited and
    # sublimated ice energy is applied after the inventory update below.
    update_evaporator_energies(
        room_state_float64_vector,
        equipment_configurations_float64_vector,
        passage_thickness,
        face_thickness,
        face_covered_area,
        equipment_configurations_float64_vector[IDX_EQ_EVAP_FACE_CONTACT_FRACTION],
        1.0 / equipment_configurations_float64_vector[IDX_EQ_EVAP_INV_FACE_BRIDGE_LENGTH_M],
        Q_defrost_coil_w,
        dE_air_to_passage_surface_j,
        dE_air_to_uncovered_front_j,
        Q_air_to_outer_face_ice,
        0.0,
        0.0,
        0.0,
        dt_engine_sec,
    )

    # Passage ice
    dm_passage_added = m_dot_passage_ice_deposited_throughflow * dt_engine_sec
    dm_passage_sublimated = m_dot_passage_ice_sublimation_total * dt_engine_sec
    dv_passage_added = v_dot_passage_ice_deposited_throughflow * dt_engine_sec
    
    M_passage = max(M_passage_prev + dm_passage_added - dm_passage_sublimated, 0.0)
    V_passage = (
        V_passage_prev * (1.0 - dm_passage_sublimated / M_passage_prev) if M_passage_prev > 0.0 else 0.0 # Sublimation
    ) + dv_passage_added # deposition
    if M_passage <= MASS_KG_FLOOR:
        M_passage = 0.0
        V_passage = 0.0
    
    # Face ice
    dm_face_added_uncovered = m_dot_face_ice_deposited_on_uncovered_front * dt_engine_sec
    dm_face_added_covered = m_dot_face_ice_deposited_on_covered_front * dt_engine_sec
    dm_face_sublimated = m_dot_face_ice_sublimation_covered_front * dt_engine_sec
    dv_face_added = v_dot_face_ice_deposited_total * dt_engine_sec
    
    M_face = max(M_face_prev + (dm_face_added_uncovered + dm_face_added_covered) - dm_face_sublimated, 0.0)
    V_face = (
        V_face_prev * (1.0 - dm_face_sublimated / M_face_prev) if M_face_prev > 0.0 else 0.0 # Sublimation
    ) + dv_face_added # Deposition
    if M_face <= MASS_KG_FLOOR:
        M_face = 0.0
        V_face = 0.0

    room_state_float64_vector[IDX_ROOM_STATE_M_PASSAGE_ICE_KG] = M_passage
    room_state_float64_vector[IDX_ROOM_STATE_V_PASSAGE_ICE_M3] = V_passage
    room_state_float64_vector[IDX_ROOM_STATE_M_FACE_ICE_KG] = M_face
    room_state_float64_vector[IDX_ROOM_STATE_V_FACE_ICE_M3] = V_face

    # Add the sensible energy carried by deposited or sublimated ice.
    if M_passage == 0.0:
        room_state_float64_vector[IDX_ROOM_STATE_E_PASSAGE_ICE_J] = 0.0
    else:
        E_passage = room_state_float64_vector[IDX_ROOM_STATE_E_PASSAGE_ICE_J]
        dE_ice_mass_p_j = CP_ICE * dm_passage_added * T_substrate
        if M_passage_prev > 0.0:
            dE_ice_mass_p_j -= (dm_passage_sublimated / M_passage_prev) * E_passage
        room_state_float64_vector[IDX_ROOM_STATE_E_PASSAGE_ICE_J] += dE_ice_mass_p_j

    if M_face == 0.0:
        room_state_float64_vector[IDX_ROOM_STATE_E_FACE_INNER_ICE_J] = 0.0
        room_state_float64_vector[IDX_ROOM_STATE_E_FACE_OUTER_ICE_J] = 0.0
    else:
        E_inner = room_state_float64_vector[IDX_ROOM_STATE_E_FACE_INNER_ICE_J]
        E_outer = room_state_float64_vector[IDX_ROOM_STATE_E_FACE_OUTER_ICE_J]
        E_face_added_j = CP_ICE * (
            dm_face_added_uncovered * T_substrate
            + dm_face_added_covered * T_face_outer
        )
        face_sublimated_fraction = dm_face_sublimated / M_face_prev if M_face_prev > 0.0 else 0.0
        dE_ice_mass_i_j = 0.5 * E_face_added_j - face_sublimated_fraction * E_inner
        dE_ice_mass_o_j = 0.5 * E_face_added_j - face_sublimated_fraction * E_outer
        room_state_float64_vector[IDX_ROOM_STATE_E_FACE_INNER_ICE_J] += dE_ice_mass_i_j
        room_state_float64_vector[IDX_ROOM_STATE_E_FACE_OUTER_ICE_J] += dE_ice_mass_o_j

    M_passage, V_passage, E_passage, dm_passage_melted, E_passage_liquid_excess_j = melt_passage_ice(
        M_passage,
        V_passage,
        room_state_float64_vector[IDX_ROOM_STATE_E_PASSAGE_ICE_J],
    )
    M_face, V_face, E_inner, E_outer, dm_face_melted, dm_face_inner_melted, E_face_liquid_excess_j = melt_face_ice(
        M_face,
        V_face,
        room_state_float64_vector[IDX_ROOM_STATE_E_FACE_INNER_ICE_J],
        room_state_float64_vector[IDX_ROOM_STATE_E_FACE_OUTER_ICE_J],
    )

    # A thin attachment film at fin edges can melt before the entire face slab.
    # Its latent heat has already been charged by melt_face_ice; the remaining
    # slab falls into the pan once that measured inner-cell melt is sufficient.
    dm_face_shed = 0.0
    E_face_shed_j = 0.0
    bond_melted_kg = room_state_float64_vector[IDX_ROOM_STATE_FACE_BOND_MELTED_KG]
    if M_face > 0.0:
        # A clearly subfreezing inner cell represents a refrozen attachment.
        # Do not reset for tiny negative cell energy during active melting:
        # equal-mass cell redistribution can leave it just below zero.
        T_face_inner_c = 2.0 * E_inner / (M_face * CP_ICE)
        if T_face_inner_c <= FACE_BOND_REFREEZE_TEMP_C and dm_face_inner_melted == 0.0:
            bond_melted_kg = 0.0
        bond_melted_kg += dm_face_inner_melted
        face_area_after_melt_m2 = min(
            equipment_configurations_float64_vector[IDX_EQ_EVAP_FRONTAL_AREA_M2],
            V_face * equipment_configurations_float64_vector[IDX_EQ_EVAP_INV_FACE_BRIDGE_LENGTH_M],
        )
        bond_release_mass_kg = (
            face_area_after_melt_m2
            * equipment_configurations_float64_vector[IDX_EQ_FACE_RELEASE_BOND_MASS_PER_M2]
        )
        if bond_melted_kg >= bond_release_mass_kg:
            dm_face_shed = M_face
            E_face_shed_j = E_inner + E_outer
            M_face = 0.0
            V_face = 0.0
            E_inner = 0.0
            E_outer = 0.0
            bond_melted_kg = 0.0
    else:
        bond_melted_kg = 0.0

    room_state_float64_vector[IDX_ROOM_STATE_FACE_BOND_MELTED_KG] = bond_melted_kg
    dm_pan_ice_melted, Q_pan_to_room_w = update_drain_pan_ice(
        room_state_float64_vector,
        dm_face_shed,
        E_face_shed_j,
        Q_defrost_pan_w,
        C_room,
        equipment_configurations_float64_vector[IDX_EQ_DEFROST_PAN_AIR_CONDUCTANCE_W_K],
        dt_engine_sec,
    )
    room_state_float64_vector[IDX_ROOM_STATE_M_PASSAGE_ICE_KG] = M_passage
    room_state_float64_vector[IDX_ROOM_STATE_V_PASSAGE_ICE_M3] = V_passage
    room_state_float64_vector[IDX_ROOM_STATE_E_PASSAGE_ICE_J] = E_passage
    room_state_float64_vector[IDX_ROOM_STATE_M_FACE_ICE_KG] = M_face
    room_state_float64_vector[IDX_ROOM_STATE_V_FACE_ICE_M3] = V_face
    room_state_float64_vector[IDX_ROOM_STATE_E_FACE_INNER_ICE_J] = E_inner
    room_state_float64_vector[IDX_ROOM_STATE_E_FACE_OUTER_ICE_J] = E_outer

    dm_ice_melted = (
        dm_passage_melted_before_exchange + dm_passage_melted + dm_face_melted
    )
    m_dot_melt = dm_ice_melted / dt_engine_sec
    Q_melt = m_dot_melt * H_FUSION
    E_liquid_excess_j = E_passage_liquid_excess_j + E_face_liquid_excess_j
    m_dot_ice_deposited = (
        m_dot_passage_ice_deposited_throughflow
        + m_dot_face_ice_deposited_on_uncovered_front
        + m_dot_face_ice_deposited_on_covered_front
    )
    m_dot_ice_sublimated = (
        m_dot_passage_ice_sublimation_total
        + m_dot_face_ice_sublimation_covered_front
    )
    return (
        Q_sensible_air_to_passage_surface,
        Q_sensible_air_to_face_total,
        Q_latent_total,
        m_dot_ice_deposited,
        m_dot_ice_sublimated,
        m_dot_condensed_total,
        m_dot_water_removed_total,
        T_dew_room,
        m_dot_melt,
        Q_melt,
        E_liquid_excess_j,
        dm_face_shed / dt_engine_sec,
        dm_pan_ice_melted / dt_engine_sec,
        Q_defrost_coil_w,
        Q_defrost_pan_w,
        Q_pan_to_room_w,
    )

@njit
def melt_ice_cell(mass_kg, energy_j):
    """Project one ice cell onto the 0 C melt plateau after heat exchange."""
    if mass_kg == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    if energy_j <= 0.0:
        return mass_kg, energy_j, 0.0, 0.0
    dm_melted = min(energy_j * INV_H_FUSION, mass_kg)
    mass_remaining = mass_kg - dm_melted
    if mass_remaining > 0.0:
        return mass_remaining, 0.0, dm_melted, 0.0
    liquid_excess_j = energy_j - dm_melted * H_FUSION
    return 0.0, 0.0, dm_melted, liquid_excess_j


@njit
def melt_passage_ice(M, V, E):
    """Melt the passage cell; positive excess energy leaves with meltwater."""
    if M == 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    M_new, E_new, dm_melted, liquid_excess_j = melt_ice_cell(M, E)
    V_new = V * (M_new / M) if M_new > 0.0 else 0.0
    return M_new, V_new, E_new, dm_melted, liquid_excess_j


@njit
def melt_face_ice(M, V, E_inner, E_outer):
    """Melt face cells independently, then conservatively restore equal cell masses."""
    if M == 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    half_mass = 0.5 * M
    M_inner, E_inner, dm_inner, excess_inner_j = melt_ice_cell(half_mass, E_inner)
    M_outer, E_outer, dm_outer, excess_outer_j = melt_ice_cell(half_mass, E_outer)

    M_new = M_inner + M_outer
    dm_melted = dm_inner + dm_outer
    liquid_excess_j = excess_inner_j + excess_outer_j
    if M_new == 0.0:
        return 0.0, 0.0, 0.0, 0.0, dm_melted, dm_inner, liquid_excess_j

    V_new = V * (M_new / M)
    new_half_mass = 0.5 * M_new
    if M_inner >= new_half_mass:
        # Move the surplus inner ice into the new outer cell.
        E_inner_new = E_inner * (new_half_mass / M_inner)
        E_outer_new = E_outer + (E_inner - E_inner_new)
    else:
        # Move the required outer ice into the new inner cell.
        E_outer_moved = E_outer * ((new_half_mass - M_inner) / M_outer)
        E_inner_new = E_inner + E_outer_moved
        E_outer_new = E_outer - E_outer_moved
    return M_new, V_new, E_inner_new, E_outer_new, dm_melted, dm_inner, liquid_excess_j


@njit
def update_drain_pan_ice(
    room_state_float64_vector,
    dm_ice_shed_kg,
    E_ice_shed_j,
    Q_pan_heater_w,
    C_room,
    G_room_to_pan_w_k,
    dt_engine_sec,
):
    """Receive shed ice; exchange heat with room; melt and drain liquid."""
    M_pan = room_state_float64_vector[IDX_ROOM_STATE_M_PAN_ICE_KG] + dm_ice_shed_kg
    E_pan = room_state_float64_vector[IDX_ROOM_STATE_E_PAN_ICE_J] + E_ice_shed_j
    if M_pan > 0.0:
        E_pan += Q_pan_heater_w * dt_engine_sec
        E_room = C_room * room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C]
        dE_room_to_pan_j = phase_aware_pair_energy_delta(
            E_room,
            E_pan,
            C_room,
            M_pan * CP_ICE,
            0.0,
            M_pan,
            G_room_to_pan_w_k,
            dt_engine_sec,
        )
        M_pan, E_pan, dm_pan_melted, E_unused_j = melt_ice_cell(
            M_pan, E_pan + dE_room_to_pan_j,
        )
        Q_pan_to_room_w = (E_unused_j - dE_room_to_pan_j) / dt_engine_sec
    else:
        M_pan = 0.0
        E_pan = 0.0
        dm_pan_melted = 0.0
        Q_pan_to_room_w = Q_pan_heater_w
    room_state_float64_vector[IDX_ROOM_STATE_M_PAN_ICE_KG] = M_pan
    room_state_float64_vector[IDX_ROOM_STATE_E_PAN_ICE_J] = E_pan
    return dm_pan_melted, Q_pan_to_room_w
