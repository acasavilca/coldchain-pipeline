"""
Output logging for run_simulator(). Call once per step, right before the
`epoch_timestamp_ms += dt_engine_ms` line, replacing your `# TODO: Write output
arrays` comment.

Deliberately takes local loop variables (cop, P_evap_bar, m_dot_refrig_max,
valve_flood_frac, n_iters, absolute_error, downtime_scheduled,
standard_ramp_up_scheduled, T_ambient, RH_ambient, V_free_m3-derived fan targets,
Q_cooling_w_rated) as explicit arguments instead of re-deriving them from
room_state_float64_vector -- several of these (cop, m_dot_refrig_max,
valve_flood_frac, n_iters, absolute_error, Q_cooling_w_rated,
W_compressor_w_rated) are NOT stored in the state vector at all, they only exist
as return values inside the step. Pulling them from state would either be
impossible or would silently read a stale value from a prior step.
"""

from matrix_index_maps import *  # noqa: F401,F403
from constants import SENTINEL_FLOAT64, CP_ICE
from numba import njit
import math

@njit
def write_output_arrays(
    step,
    output_float64_matrix,
    output_int64_matrix,
    room_state_float64_vector,
    room_state_int64_vector,
    door_states_float64_matrix,
    door_states_int64_matrix,
    forcing_cache_float64_vector,
    room_ephemeral_int64_vector,
    active_crop_params_int64_vector,
    cop,
    P_evap_bar,
    P_cond_bar,
    W_compressor,
    m_dot_refrig_max,
    valve_flood_frac,
    n_iters,
    absolute_error,
    m_dot_humidifier,
    Q_sensible_forced,
    Q_sensible_face,
    Q_latent,
    Q_fans,
    T_dew_room,
    Q_inf_sens_int,
    Q_inf_sens_ext,
    f_exchange_sec,
    m_dot_leak,
    m_dot_water_removed,
    m_dot_frost,
    m_dot_sublimated,
    m_dot_condensed,
    Q_melt,
    m_dot_melt,
    E_liquid_excess_j,
    m_dot_face_ice_shed,
    m_dot_pan_ice_melted,
    Q_coil_defrost_heater_w,
    Q_pan_defrost_heater_w,
    Q_pan_to_room_w,
    UA_coil_evap_air,
    T_evap_outlet,
):
    # --- output_int64_matrix ---
    output_int64_matrix[step, IDX_OUTPUT_EPOCH_TIMESTAMP_MS] = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]
    output_int64_matrix[step, IDX_OUTPUT_REFRIGERATION_IS_ACTIVE] = room_state_int64_vector[IDX_ROOM_STATE_REFRIGERATION_IS_ACTIVE]
    output_int64_matrix[step, IDX_OUTPUT_HUMIDIFIER_NUM_DISCS_ON] = room_state_int64_vector[IDX_ROOM_STATE_HUMIDIFIER_NUM_PIEZO_DISCS_ON]
    output_int64_matrix[step, IDX_OUTPUT_TOTAL_BATCHES_IN_ROOM] = room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_ROOM]
    output_int64_matrix[step, IDX_OUTPUT_TOTAL_ARRIVALS_PLANNED] = room_state_int64_vector[IDX_ROOM_STATE_TOTAL_ARRIVALS_PLANNED]
    output_int64_matrix[step, IDX_OUTPUT_TOTAL_DISPATCHES_PLANNED] = room_state_int64_vector[IDX_ROOM_STATE_TOTAL_DISPATCHES_PLANNED]
    output_int64_matrix[step, IDX_OUTPUT_DOWNTIME_SCHEDULED] = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_SCHEDULED]
    output_int64_matrix[step, IDX_OUTPUT_DOWNTIME_SAFE_SHUTDOWN] = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_SAFE_FOR_SHUTDOWN]
    output_int64_matrix[step, IDX_OUTPUT_STANDARD_RAMP_UP_SCHEDULED] = room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_SCHEDULED]
    output_int64_matrix[step, IDX_OUTPUT_ACTIVE_FRUIT_TYPE_CODE] = active_crop_params_int64_vector[IDX_FRUIT_PARAM_FRUIT_TYPE_CODE]
    output_int64_matrix[step, IDX_OUTPUT_ACTIVE_REFRIGERANT_CODE] = room_state_int64_vector[IDX_ROOM_STATE_ACTIVE_REFRIGERANT_CODE]
    output_int64_matrix[step, IDX_OUTPUT_SOLVER_N_ITERS] = n_iters

    output_int64_matrix[step, IDX_OUTPUT_INT_DOOR_IS_OPEN] = door_states_int64_matrix[DOOR_CHANNEL_INTERNAL, IDX_DOOR_STATE_IS_OPEN]
    output_int64_matrix[step, IDX_OUTPUT_EXT_DOOR_IS_OPEN] = door_states_int64_matrix[DOOR_CHANNEL_EXTERNAL, IDX_DOOR_STATE_IS_OPEN]

    output_int64_matrix[step, IDX_OUTPUT_LP_TRIPPED] = room_state_int64_vector[IDX_ROOM_STATE_LP_TRIPPED]
    output_int64_matrix[step, IDX_OUTPUT_THERMOSTAT_CALL] = room_state_int64_vector[IDX_ROOM_STATE_THERMOSTAT_CALL]
    output_int64_matrix[step, IDX_OUTPUT_EVAP_AIRFLOW_PROVED] = room_state_int64_vector[IDX_ROOM_STATE_EVAP_AIRFLOW_PROVED]
    output_int64_matrix[step, IDX_OUTPUT_DEFROST_PHASE] = room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE]

    # --- output_float64_matrix ---
    output_float64_matrix[step, IDX_OUTPUT_INT_DOOR_OPEN_FRACTION] = door_states_float64_matrix[DOOR_CHANNEL_INTERNAL, IDX_DOOR_STATE_OPEN_FRACTION]
    output_float64_matrix[step, IDX_OUTPUT_EXT_DOOR_OPEN_FRACTION] = door_states_float64_matrix[DOOR_CHANNEL_EXTERNAL, IDX_DOOR_STATE_OPEN_FRACTION]
    output_float64_matrix[step, IDX_OUPUT_INT_M_DOT_AIR_INFLOW] = door_states_float64_matrix[DOOR_CHANNEL_INTERNAL, IDX_DOOR_STATE_M_DOT_INFLOW]
    output_float64_matrix[step, IDX_OUPUT_EXT_M_DOT_AIR_INFLOW] = door_states_float64_matrix[DOOR_CHANNEL_EXTERNAL, IDX_DOOR_STATE_M_DOT_INFLOW]

    output_float64_matrix[step, IDX_OUTPUT_T_ROOM_C] = room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C]
    output_float64_matrix[step, IDX_OUTPUT_T_ROOM_FILTERED_C] = room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_FILTERED_VAL]
    output_float64_matrix[step, IDX_OUTPUT_RH_ROOM] = room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM]
    output_float64_matrix[step, IDX_OUTPUT_RH_ROOM_FILTERED] = room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM_FILTERED_VAL]
    output_float64_matrix[step, IDX_OUTPUT_W_ROOM_HUMIDITY_RATIO] = room_state_float64_vector[IDX_ROOM_STATE_AIR_HUMIDITY_RATIO_W]
    output_float64_matrix[step, IDX_OUTPUT_CO2_PPM] = room_state_float64_vector[IDX_ROOM_STATE_CO2_PPM]
    output_float64_matrix[step, IDX_OUTPUT_O2_PCT] = room_state_float64_vector[IDX_ROOM_STATE_O2_PCT]

    output_float64_matrix[step, IDX_OUTPUT_W_AMBIENT] = forcing_cache_float64_vector[IDX_CACHE_W_AMBIENT]
    output_float64_matrix[step, IDX_OUTPUT_P_AMBIENT] = forcing_cache_float64_vector[IDX_CACHE_P_AMBIENT]

    output_float64_matrix[step, IDX_OUTPUT_T_FLOOR_C] = room_state_float64_vector[IDX_ROOM_STATE_FLOOR_SLAB_TEMP_C]
    output_float64_matrix[step, IDX_OUTPUT_T_ADJACENT_PLANT_C] = room_state_float64_vector[IDX_ROOM_STATE_ADJACENT_PLANT_TEMP_C]

    output_float64_matrix[step, IDX_OUTPUT_T_PULP_BULK_C] = room_state_float64_vector[IDX_ROOM_STATE_BULK_PULP_TEMP_C]
    output_float64_matrix[step, IDX_OUTPUT_BULK_FRUIT_MASS_KG] = room_state_float64_vector[IDX_ROOM_STATE_BULK_PULP_MASS]

    output_float64_matrix[step, IDX_OUTPUT_T_COIL_EVAP_C] = room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAPORATOR]
    output_float64_matrix[step, IDX_OUTPUT_T_COIL_EVAP_FILTERED_C] = room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAP_FILTERED_VAL]
    output_float64_matrix[step, IDX_OUTPUT_Q_DEFROST_HEATER_W] = room_state_float64_vector[IDX_ROOM_STATE_Q_DEFROST_HEATER_W]
    output_float64_matrix[step, IDX_OUTPUT_Q_COIL_DEFROST_HEATER_W] = Q_coil_defrost_heater_w
    output_float64_matrix[step, IDX_OUTPUT_Q_PAN_DEFROST_HEATER_W] = Q_pan_defrost_heater_w
    output_float64_matrix[step, IDX_OUTPUT_Q_PAN_TO_ROOM_W] = Q_pan_to_room_w
    output_float64_matrix[step, IDX_OUTPUT_T_COIL_COND_C] = room_state_float64_vector[IDX_ROOM_STATE_T_COIL_CONDENSER]
    output_float64_matrix[step, IDX_OUTPUT_T_EVAP_C] = room_state_float64_vector[IDX_ROOM_STATE_T_EVAP]
    output_float64_matrix[step, IDX_OUTPUT_T_COND_C] = room_state_float64_vector[IDX_ROOM_STATE_T_COND]
    output_float64_matrix[step, IDX_OUTPUT_T_LIQUID_LINE_C] = room_state_float64_vector[IDX_ROOM_STATE_T_LIQUID_OUTLET]
    output_float64_matrix[step, IDX_OUTPUT_T_SUCTION_LINE_C] = T_evap_outlet

    output_float64_matrix[step, IDX_OUTPUT_P_EVAP_BAR] = P_evap_bar
    output_float64_matrix[step, IDX_OUTPUT_P_COND_BAR] = P_cond_bar
    output_float64_matrix[step, IDX_OUTPUT_COP] = cop
    output_float64_matrix[step, IDX_OUTPUT_W_COMPRESSOR_W] = W_compressor
    output_float64_matrix[step, IDX_OUTPUT_M_DOT_REFRIG_MAX] = m_dot_refrig_max
    output_float64_matrix[step, IDX_OUTPUT_VALVE_FLOOD_FRAC] = valve_flood_frac
    output_float64_matrix[step, IDX_OUTPUT_SOLVER_ABS_ERROR] = absolute_error
    output_float64_matrix[step, IDX_OUTPUT_Q_REFRIG_EVAP_W] = room_state_float64_vector[IDX_ROOM_STATE_Q_REFRIG_EVAP]
    output_float64_matrix[step, IDX_OUTPUT_Q_REFRIG_COND_W] = room_state_float64_vector[IDX_ROOM_STATE_Q_REFRIG_COND]

    output_float64_matrix[step, IDX_OUTPUT_Q_COOLING_W_RATED] = room_state_float64_vector[IDX_ROOM_STATE_RATED_COMPRESSOR_Q_COOLING]
    output_float64_matrix[step, IDX_OUTPUT_W_COMPRESSOR_W_RATED] = room_state_float64_vector[IDX_ROOM_STATE_RATED_COMPRESSOR_W_ELECT]

    output_float64_matrix[step, IDX_OUTPUT_V_DOT_AIR_EVAP_ACTUAL] = room_state_float64_vector[IDX_ROOM_STATE_V_DOT_AIR_EVAP_ACTUAL]
    output_float64_matrix[step, IDX_OUTPUT_EVAP_AIR_VELOCITY_SENSED_M_S] = room_state_float64_vector[IDX_ROOM_STATE_EVAP_AIR_VELOCITY_SENSED_M_S]
    output_float64_matrix[step, IDX_OUTPUT_V_DOT_AIR_COND_ACTUAL] = room_state_float64_vector[IDX_ROOM_STATE_V_DOT_AIR_COND_ACTUAL]
    output_float64_matrix[step, IDX_OUTPUT_COOLING_FRACTION] = room_state_float64_vector[IDX_ROOM_STATE_COOLING_FRACTION]

    output_float64_matrix[step, IDX_OUTPUT_M_DOT_HUMIDIFIER_KG_S] = m_dot_humidifier

    output_float64_matrix[step, IDX_OUTPUT_T_AMBIENT_C] = forcing_cache_float64_vector[IDX_CACHE_T_AMBIENT]
    output_float64_matrix[step, IDX_OUTPUT_RH_AMBIENT]  = forcing_cache_float64_vector[IDX_CACHE_RH_AMBIENT]

    output_float64_matrix[step, IDX_OUTPUT_T_SUCTION_LINE_FILTERED] = room_state_float64_vector[IDX_ROOM_STATE_T_SUCTION_LINE_FILTERED_VAL]
    output_float64_matrix[step, IDX_OUTPUT_P_SUCTION_LINE_FILTERED] = room_state_float64_vector[IDX_ROOM_STATE_P_SUCTION_LINE_FILTERED_VAL]
    output_float64_matrix[step, IDX_OUTPUT_T_LIQUID_LINE_FILTERED] = room_state_float64_vector[IDX_ROOM_STATE_T_LIQUID_LINE_FILTERED_VAL]
    output_float64_matrix[step, IDX_OUTPUT_P_LIQUID_LINE_FILTERED] = room_state_float64_vector[IDX_ROOM_STATE_P_LIQUID_LINE_FILTERED_VAL]
    output_float64_matrix[step, IDX_OUTPUT_COMPRESSOR_POWER_FILTERED] = room_state_float64_vector[IDX_ROOM_STATE_COMPRESSOR_POWER_FILTERED_VAL]

    output_float64_matrix[step, IDX_OUTPUT_Q_SENSIBLE_FORCED]      = Q_sensible_forced 
    output_float64_matrix[step, IDX_OUTPUT_Q_SENSIBLE_FACE]      = Q_sensible_face 
    output_float64_matrix[step, IDX_OUTPUT_Q_LATENT]        = Q_latent
    output_float64_matrix[step, IDX_OUTPUT_Q_FANS]          = Q_fans
    output_float64_matrix[step, IDX_OUTPUT_T_DEW_ROOM]      = T_dew_room
    output_float64_matrix[step, IDX_OUTPUT_Q_INF_SENS_INT]  = Q_inf_sens_int
    output_float64_matrix[step, IDX_OUTPUT_Q_INF_SENS_EXT]  = Q_inf_sens_ext
    output_float64_matrix[step, IDX_OUTPUT_F_EXCHANGE_SEC]  = f_exchange_sec
    output_float64_matrix[step, IDX_OUTPUT_M_DOT_LEAK]      = m_dot_leak

    output_float64_matrix[step, IDX_OUTPUT_M_DOT_WATER_REMOVED] = m_dot_water_removed
    output_float64_matrix[step, IDX_OUTPUT_M_DOT_FROST]         = m_dot_frost
    output_float64_matrix[step, IDX_OUTPUT_M_DOT_SUBLIMATED]    = m_dot_sublimated
    output_float64_matrix[step, IDX_OUTPUT_M_DOT_CONDENSED]     = m_dot_condensed
    output_float64_matrix[step, IDX_OUTPUT_Q_MELT]              = Q_melt
    output_float64_matrix[step, IDX_OUTPUT_M_DOT_MELT]          = m_dot_melt
    output_float64_matrix[step, IDX_OUTPUT_M_DOT_FACE_ICE_SHED] = m_dot_face_ice_shed
    output_float64_matrix[step, IDX_OUTPUT_M_DOT_PAN_ICE_MELTED] = m_dot_pan_ice_melted
    output_float64_matrix[step, IDX_OUTPUT_M_PAN_ICE_KG] = room_state_float64_vector[IDX_ROOM_STATE_M_PAN_ICE_KG]
    output_float64_matrix[step, IDX_OUTPUT_FACE_BOND_MELTED_KG] = room_state_float64_vector[IDX_ROOM_STATE_FACE_BOND_MELTED_KG]
    output_float64_matrix[step, IDX_OUTPUT_E_LIQUID_EXCESS_J]  = E_liquid_excess_j

    M_passage = room_state_float64_vector[IDX_ROOM_STATE_M_PASSAGE_ICE_KG]
    M_face = room_state_float64_vector[IDX_ROOM_STATE_M_FACE_ICE_KG]
    V_passage = room_state_float64_vector[IDX_ROOM_STATE_V_PASSAGE_ICE_M3]
    V_face = room_state_float64_vector[IDX_ROOM_STATE_V_FACE_ICE_M3]
    output_float64_matrix[step, IDX_OUTPUT_M_FROST_KG] = M_passage + M_face
    output_float64_matrix[step, IDX_OUTPUT_W_COIL_EVAP_SAT]     = room_state_float64_vector[IDX_ROOM_STATE_W_COIL_EVAP_SAT]

    output_float64_matrix[step, IDX_OUTPUT_V_FROST_M3] = V_passage + V_face
    output_float64_matrix[step, IDX_OUTPUT_M_PASSAGE_ICE_KG] = M_passage
    output_float64_matrix[step, IDX_OUTPUT_M_FACE_ICE_KG] = M_face
    output_float64_matrix[step, IDX_OUTPUT_V_PASSAGE_ICE_M3] = V_passage
    output_float64_matrix[step, IDX_OUTPUT_V_FACE_ICE_M3] = V_face
    output_float64_matrix[step, IDX_OUTPUT_T_PASSAGE_ICE_C] = (
        room_state_float64_vector[IDX_ROOM_STATE_E_PASSAGE_ICE_J] / (M_passage * CP_ICE)
        if M_passage > 0.0 else SENTINEL_FLOAT64
    )
    if M_face > 0.0:
        inv_C_face_half = 2.0 / (M_face * CP_ICE)
        output_float64_matrix[step, IDX_OUTPUT_T_FACE_INNER_ICE_C] = (
            room_state_float64_vector[IDX_ROOM_STATE_E_FACE_INNER_ICE_J] * inv_C_face_half
        )
        output_float64_matrix[step, IDX_OUTPUT_T_FACE_OUTER_ICE_C] = (
            room_state_float64_vector[IDX_ROOM_STATE_E_FACE_OUTER_ICE_J] * inv_C_face_half
        )
    else:
        output_float64_matrix[step, IDX_OUTPUT_T_FACE_INNER_ICE_C] = SENTINEL_FLOAT64
        output_float64_matrix[step, IDX_OUTPUT_T_FACE_OUTER_ICE_C] = SENTINEL_FLOAT64
    output_float64_matrix[step, IDX_OUTPUT_EVAP_FAN_MODULATION_FRAC] = room_state_float64_vector[IDX_ROOM_STATE_EVAP_FAN_MODULATION_FRAC]
    output_float64_matrix[step, IDX_OUTPUT_COND_FAN_MODULATION_FRAC] = room_state_float64_vector[IDX_ROOM_STATE_COND_FAN_MODULATION_FRAC]
    output_float64_matrix[step, IDX_OUTPUT_UA_COIL_EVAP_AIR] = UA_coil_evap_air

# --- write_room_telemetry_export_row ---
# Called every step immediately after write_output_arrays (see engine.py) --
# same "per-step output emission" role, just targeting the sensor-realistic
# telemetry matrix (the one that becomes the parquet export) instead of the
# raw physics-debug matrix above. Kept in the same file on purpose.

@njit
def write_room_telemetry_export_row(
    room_telemetry_matrix,
    room_state_int64_vector,
    room_state_float64_vector,
    # forcing_cache_float64_vector,
    equipment_configurations_int64_vector,
    room_ephemeral_int64_vector,
):
    # One row per polling tick: steps that cannot be a tick for any channel write nothing.
    # The row counter lives in the ephemeral vector (trimmed on load, see run_outputs_io).
    epoch_ms = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]
    if epoch_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_POLL_GCD_MS] != 0:
        return
    export_row = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_TELEMETRY_EXPORT_WRITE_IDX]
    room_telemetry_matrix[export_row, IDX_EXPORT_TIMESTAMP_MS] = epoch_ms

    if epoch_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_T_INTERVAL_MS] == 0:
        room_telemetry_matrix[export_row, IDX_EXPORT_T_ROOM] = room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_FILTERED_VAL]
    if epoch_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_RH_INTERVAL_MS] == 0:
        room_telemetry_matrix[export_row, IDX_EXPORT_RH_ROOM] = room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM_FILTERED_VAL]
    if epoch_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_T_SUCTION_LINE_INTERVAL_MS] == 0:
        room_telemetry_matrix[export_row, IDX_EXPORT_T_SUCTION_LINE] = room_state_float64_vector[IDX_ROOM_STATE_T_SUCTION_LINE_FILTERED_VAL]
    if epoch_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_P_SUCTION_LINE_INTERVAL_MS] == 0:
        room_telemetry_matrix[export_row, IDX_EXPORT_P_SUCTION_LINE] = room_state_float64_vector[IDX_ROOM_STATE_P_SUCTION_LINE_FILTERED_VAL]
    if epoch_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_T_LIQUID_LINE_INTERVAL_MS] == 0:
        room_telemetry_matrix[export_row, IDX_EXPORT_T_LIQUID_LINE] = room_state_float64_vector[IDX_ROOM_STATE_T_LIQUID_LINE_FILTERED_VAL]
    if epoch_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_P_LIQUID_LINE_INTERVAL_MS] == 0:
        room_telemetry_matrix[export_row, IDX_EXPORT_P_LIQUID_LINE] = room_state_float64_vector[IDX_ROOM_STATE_P_LIQUID_LINE_FILTERED_VAL]
    if epoch_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_COMPRESSOR_POWER_INTERVAL_MS] == 0:
        room_telemetry_matrix[export_row, IDX_EXPORT_COMPRESSOR_POWER] = room_state_float64_vector[IDX_ROOM_STATE_COMPRESSOR_POWER_FILTERED_VAL]
    if epoch_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_T_ROOM_AIR_RETURN_INTERVAL_MS] == 0:
        room_telemetry_matrix[export_row, IDX_EXPORT_T_ROOM_AIR_RETURN] = room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_AIR_RETURN_FILTERED_VAL]
    if epoch_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_T_ROOM_AIR_SUPPLY_INTERVAL_MS] == 0:
        room_telemetry_matrix[export_row, IDX_EXPORT_T_ROOM_AIR_SUPPLY] = room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_AIR_SUPPLY_FILTERED_VAL]
    if epoch_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_RH_ROOM_AIR_SUPPLY_INTERVAL_MS] == 0:
        room_telemetry_matrix[export_row, IDX_EXPORT_RH_ROOM_AIR_SUPPLY] = room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM_AIR_SUPPLY_FILTERED_VAL]
    if epoch_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_CO2_PPM_INTERVAL_MS] == 0:
        room_telemetry_matrix[export_row, IDX_EXPORT_CO2_PPM] = room_state_float64_vector[IDX_ROOM_STATE_CO2_PPM_FILTERED_VAL]
    if epoch_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_O2_PCT_INTERVAL_MS] == 0:
        room_telemetry_matrix[export_row, IDX_EXPORT_O2_PCT] = room_state_float64_vector[IDX_ROOM_STATE_O2_PCT_FILTERED_VAL]

    if epoch_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_T_COIL_EVAP_INTERVAL_MS] == 0:
        room_telemetry_matrix[export_row, IDX_EXPORT_T_COIL_EVAP] = room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAP_FILTERED_VAL]
    if epoch_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_EVAP_AIR_VELOCITY_INTERVAL_MS] == 0:
        room_telemetry_matrix[export_row, IDX_EXPORT_EVAP_AIR_VELOCITY] = room_state_float64_vector[IDX_ROOM_STATE_EVAP_AIR_VELOCITY_SENSED_M_S]

    room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_TELEMETRY_EXPORT_WRITE_IDX] = export_row + 1
    # room_telemetry_matrix[export_row, IDX_EXPORT_T_AMBIENT] = forcing_cache_float64_vector[IDX_CACHE_T_AMBIENT]
    # room_telemetry_matrix[export_row, IDX_EXPORT_RH_AMBIENT] = forcing_cache_float64_vector[IDX_CACHE_RH_AMBIENT]
