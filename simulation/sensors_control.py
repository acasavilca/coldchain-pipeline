"""
sensors_control.py

Sensor filtering and control-loop primitives: exponential sensor filters, PID evaluation, per-step sensor-reading updates, door-state lookup.
"""

import math
import numpy as np
from jit import njit

from constants import *
from matrix_index_maps import *
from utils import numba_clip


@njit
def get_sensor_filtered_value(
    rng,
    true_value,
    previous_filtered_value,
    alpha,
    noise_scale,
    lower_limit,
    upper_limit,
):
    noise = rng.normal(loc=0, scale=noise_scale)
    raw_input = true_value + noise
    filtered_value = (alpha * raw_input) + ((1 - alpha) * previous_filtered_value)
    return numba_clip(filtered_value, lower_limit, upper_limit)


@njit
def update_evaporator_air_velocity_sensor(
    rng,
    room_state_float64_vector,
    room_state_int64_vector,
    equipment_configurations_float64_vector,
    equipment_configurations_int64_vector,
):
    """Sample the flow switch's internal velocity estimate; hold it between samples.

    The controller receives the switch contact, not this internal measurement.
    """
    epoch_ms = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]
    sample_interval_ms = equipment_configurations_int64_vector[
        IDX_EQ_SENSOR_EVAP_AIR_VELOCITY_INTERVAL_MS
    ]
    if epoch_ms % sample_interval_ms == 0:
        representative_velocity_m_s = (
            room_state_float64_vector[IDX_ROOM_STATE_V_DOT_AIR_EVAP_ACTUAL]
            * equipment_configurations_float64_vector[IDX_EQ_EVAP_INV_FRONTAL_AREA_M2]
        )
        room_state_float64_vector[IDX_ROOM_STATE_EVAP_AIR_VELOCITY_SENSED_M_S] = (
            get_sensor_filtered_value(
                rng,
                representative_velocity_m_s,
                room_state_float64_vector[IDX_ROOM_STATE_EVAP_AIR_VELOCITY_SENSED_M_S],
                equipment_configurations_float64_vector[IDX_EQ_SENSOR_EVAP_AIR_VELOCITY_ALPHA],
                equipment_configurations_float64_vector[IDX_EQ_SENSOR_EVAP_AIR_VELOCITY_NOISE_SCALE],
                equipment_configurations_float64_vector[IDX_EQ_SENSOR_EVAP_AIR_VELOCITY_LOWER_LIMIT],
                equipment_configurations_float64_vector[IDX_EQ_SENSOR_EVAP_AIR_VELOCITY_UPPER_LIMIT],
            )
        )

@njit
def pid_evaluation(
    current_error,
    dt_sec,
    previous_error,
    integral_accumulator,
    K_p,
    K_i,
    K_d,
    min_pid_output,
    max_pid_output,
):
    p_term = K_p * current_error
    temp_i_term = integral_accumulator + (K_i * current_error * dt_sec)
    d_term = K_d * (current_error - previous_error)/dt_sec
    temp_pid_output = p_term + temp_i_term + d_term
    if (
        # Saturated high and still demanding moisture
        (temp_pid_output > max_pid_output and current_error > 0)
        # Saturated low and room is over-saturated
        or (temp_pid_output < min_pid_output and current_error < 0)
    ):
        # Freeze the accumulator
        i_term = integral_accumulator
    else:
        # Business as usual: accept the newly calculated integration step
        i_term = temp_i_term
    return (
        current_error,
        p_term,
        i_term,
        d_term,
    )

@njit
def update_sensor_readings(
    rng,
    room_state_int64_vector,
    room_state_float64_vector,
    equipment_configurations_float64_vector,
    equipment_configurations_int64_vector,
    T_evap_outlet,
    P_evap_bar,
    P_cond_bar,
    W_compressor_w,
):
    epoch_timestamp_ms = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]

    # Every channel below samples on a multiple of the GCD of all intervals, so no channel can
    # fire on any other step. RNG draws happen only inside the per-channel branches, so skipping
    # here leaves the random stream unchanged.
    if epoch_timestamp_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_POLL_GCD_MS] != 0:
        return

    # 1. Room temperature sensor
    if epoch_timestamp_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_T_INTERVAL_MS] == 0:
        t_room_filtered_val = get_sensor_filtered_value(
            rng                     = rng,
            true_value              = room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C],
            previous_filtered_value = room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_FILTERED_VAL],
            alpha                   = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_ALPHA],
            noise_scale             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_NOISE_SCALE],
            lower_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_LOWER_LIMIT],
            upper_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_UPPER_LIMIT],
        )
        room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_FILTERED_VAL] = t_room_filtered_val

    # 2. Room RH sensor
    if epoch_timestamp_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_RH_INTERVAL_MS] == 0:
        rh_room_filtered_val = get_sensor_filtered_value(
            rng                     = rng,
            true_value              = room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM],
            previous_filtered_value = room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM_FILTERED_VAL],
            alpha                   = equipment_configurations_float64_vector[IDX_EQ_SENSOR_RH_ALPHA],
            noise_scale             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_RH_NOISE_SCALE],
            lower_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_RH_LOWER_LIMIT],
            upper_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_RH_UPPER_LIMIT],
        )
        room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM_FILTERED_VAL] = rh_room_filtered_val

    # 3. Suction line temperature sensor
    # Senses refrigerant temperature at the suction line, after it's picked up superheat
    if epoch_timestamp_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_T_SUCTION_LINE_INTERVAL_MS] == 0:
        t_suction_line_filtered_val = get_sensor_filtered_value(
            rng                     = rng,
            true_value              = T_evap_outlet,
            previous_filtered_value = room_state_float64_vector[IDX_ROOM_STATE_T_SUCTION_LINE_FILTERED_VAL],
            alpha                   = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_SUCTION_LINE_ALPHA],
            noise_scale             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_SUCTION_LINE_NOISE_SCALE],
            lower_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_SUCTION_LINE_LOWER_LIMIT],
            upper_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_SUCTION_LINE_UPPER_LIMIT],
        )
        room_state_float64_vector[IDX_ROOM_STATE_T_SUCTION_LINE_FILTERED_VAL] = t_suction_line_filtered_val

    # 4. Suction line pressure sensor
    # Pressure gauge read saturation pressure before the point of full vaporization 
    if epoch_timestamp_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_P_SUCTION_LINE_INTERVAL_MS] == 0:
        p_suction_line_filtered_val = get_sensor_filtered_value(
            rng                     = rng,
            true_value              = P_evap_bar,
            previous_filtered_value = room_state_float64_vector[IDX_ROOM_STATE_P_SUCTION_LINE_FILTERED_VAL],
            alpha                   = equipment_configurations_float64_vector[IDX_EQ_SENSOR_P_SUCTION_LINE_ALPHA],
            noise_scale             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_P_SUCTION_LINE_NOISE_SCALE],
            lower_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_P_SUCTION_LINE_LOWER_LIMIT],
            upper_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_P_SUCTION_LINE_UPPER_LIMIT],
        )
        room_state_float64_vector[IDX_ROOM_STATE_P_SUCTION_LINE_FILTERED_VAL] = p_suction_line_filtered_val

    # 5. Liquid line temperature sensor
    # Senses refrigerant temperature post-subcooling
    if epoch_timestamp_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_T_LIQUID_LINE_INTERVAL_MS] == 0:
        t_liquid_line_filtered_val = get_sensor_filtered_value(
            rng                     = rng,
            true_value              = room_state_float64_vector[IDX_ROOM_STATE_T_LIQUID_OUTLET],
            previous_filtered_value = room_state_float64_vector[IDX_ROOM_STATE_T_LIQUID_LINE_FILTERED_VAL],
            alpha                   = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_LIQUID_LINE_ALPHA],
            noise_scale             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_LIQUID_LINE_NOISE_SCALE],
            lower_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_LIQUID_LINE_LOWER_LIMIT],
            upper_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_LIQUID_LINE_UPPER_LIMIT],
        )
        room_state_float64_vector[IDX_ROOM_STATE_T_LIQUID_LINE_FILTERED_VAL] = t_liquid_line_filtered_val

    # 6. Liquid line pressure sensor
    # Senses refrigerant temperature before subcooling
    if epoch_timestamp_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_P_LIQUID_LINE_INTERVAL_MS] == 0:
        p_liquid_line_filtered_val = get_sensor_filtered_value(
            rng                     = rng,
            true_value              = P_cond_bar,
            previous_filtered_value = room_state_float64_vector[IDX_ROOM_STATE_P_LIQUID_LINE_FILTERED_VAL],
            alpha                   = equipment_configurations_float64_vector[IDX_EQ_SENSOR_P_LIQUID_LINE_ALPHA],
            noise_scale             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_P_LIQUID_LINE_NOISE_SCALE],
            lower_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_P_LIQUID_LINE_LOWER_LIMIT],
            upper_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_P_LIQUID_LINE_UPPER_LIMIT],
        )
        room_state_float64_vector[IDX_ROOM_STATE_P_LIQUID_LINE_FILTERED_VAL] = p_liquid_line_filtered_val

    # 7. Power input sensor
    if epoch_timestamp_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_COMPRESSOR_POWER_INTERVAL_MS] == 0:
        w_compressor_filtered_val = get_sensor_filtered_value(
            rng                     = rng,
            true_value              = W_compressor_w,
            previous_filtered_value = room_state_float64_vector[IDX_ROOM_STATE_COMPRESSOR_POWER_FILTERED_VAL],
            alpha                   = equipment_configurations_float64_vector[IDX_EQ_SENSOR_COMPRESSOR_POWER_ALPHA],
            noise_scale             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_COMPRESSOR_POWER_NOISE_SCALE],
            lower_limit             = 0.0,
            upper_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_COMPRESSOR_POWER_LIMIT],
        )
        room_state_float64_vector[IDX_ROOM_STATE_COMPRESSOR_POWER_FILTERED_VAL] = w_compressor_filtered_val

    # 8. Return air temperature sensor
    if epoch_timestamp_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_T_ROOM_AIR_RETURN_INTERVAL_MS] == 0:
        t_room_air_return_filtered_val = get_sensor_filtered_value(
            rng                     = rng,
            true_value              = room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C],
            previous_filtered_value = room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_AIR_RETURN_FILTERED_VAL],
            alpha                   = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_ROOM_AIR_RETURN_ALPHA],
            noise_scale             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_ROOM_AIR_RETURN_NOISE_SCALE],
            lower_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_ROOM_AIR_RETURN_LOWER_LIMIT],
            upper_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_ROOM_AIR_RETURN_UPPER_LIMIT],
        )
        room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_AIR_RETURN_FILTERED_VAL] = t_room_air_return_filtered_val

    # 9. Supply air temperature sensor
    if epoch_timestamp_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_T_ROOM_AIR_SUPPLY_INTERVAL_MS] == 0:
        t_room_air_supply_filtered_val = get_sensor_filtered_value(
            rng                     = rng,
            true_value              = room_state_float64_vector[IDX_ROOM_STATE_T_EVAP_AIR],
            previous_filtered_value = room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_AIR_SUPPLY_FILTERED_VAL],
            alpha                   = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_ROOM_AIR_SUPPLY_ALPHA],
            noise_scale             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_ROOM_AIR_SUPPLY_NOISE_SCALE],
            lower_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_ROOM_AIR_SUPPLY_LOWER_LIMIT],
            upper_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_ROOM_AIR_SUPPLY_UPPER_LIMIT],
        )
        room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_AIR_SUPPLY_FILTERED_VAL] = t_room_air_supply_filtered_val

    # Evaporator defrost termination probe: metal temperature, sampled after the thermal update.
    if epoch_timestamp_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_T_COIL_EVAP_INTERVAL_MS] == 0:
        t_coil_evap_filtered_val = get_sensor_filtered_value(
            rng                     = rng,
            true_value              = room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAPORATOR],
            previous_filtered_value = room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAP_FILTERED_VAL],
            alpha                   = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_COIL_EVAP_ALPHA],
            noise_scale             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_COIL_EVAP_NOISE_SCALE],
            lower_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_COIL_EVAP_LOWER_LIMIT],
            upper_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_T_COIL_EVAP_UPPER_LIMIT],
        )
        room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAP_FILTERED_VAL] = t_coil_evap_filtered_val

    # 10. Supply air relative humidity sensor
    if epoch_timestamp_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_RH_ROOM_AIR_SUPPLY_INTERVAL_MS] == 0:
        rh_room_air_supply_filtered_val = get_sensor_filtered_value(
            rng                     = rng,
            true_value              = room_state_float64_vector[IDX_ROOM_STATE_RH_EVAP_AIR],
            previous_filtered_value = room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM_AIR_SUPPLY_FILTERED_VAL],
            alpha                   = equipment_configurations_float64_vector[IDX_EQ_SENSOR_RH_ROOM_AIR_SUPPLY_ALPHA],
            noise_scale             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_RH_ROOM_AIR_SUPPLY_NOISE_SCALE],
            lower_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_RH_ROOM_AIR_SUPPLY_LOWER_LIMIT],
            upper_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_RH_ROOM_AIR_SUPPLY_UPPER_LIMIT],
        )
        room_state_float64_vector[IDX_ROOM_STATE_RH_ROOM_AIR_SUPPLY_FILTERED_VAL] = rh_room_air_supply_filtered_val

    # 11. CO2 sensor
    if epoch_timestamp_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_CO2_PPM_INTERVAL_MS] == 0:
        co2_ppm_supply_filtered_val = get_sensor_filtered_value(
            rng                     = rng,
            true_value              = room_state_float64_vector[IDX_ROOM_STATE_CO2_PPM],
            previous_filtered_value = room_state_float64_vector[IDX_ROOM_STATE_CO2_PPM_FILTERED_VAL],
            alpha                   = equipment_configurations_float64_vector[IDX_EQ_SENSOR_CO2_PPM_ALPHA],
            noise_scale             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_CO2_PPM_NOISE_SCALE],
            lower_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_CO2_PPM_LOWER_LIMIT],
            upper_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_CO2_PPM_UPPER_LIMIT],
        )
        room_state_float64_vector[IDX_ROOM_STATE_CO2_PPM_FILTERED_VAL] = co2_ppm_supply_filtered_val

    # 12. O2 sensor
    if epoch_timestamp_ms % equipment_configurations_int64_vector[IDX_EQ_SENSOR_O2_PCT_INTERVAL_MS] == 0:
        o2_pct_supply_filtered_val = get_sensor_filtered_value(
            rng                     = rng,
            true_value              = room_state_float64_vector[IDX_ROOM_STATE_O2_PCT],
            previous_filtered_value = room_state_float64_vector[IDX_ROOM_STATE_O2_PCT_FILTERED_VAL],
            alpha                   = equipment_configurations_float64_vector[IDX_EQ_SENSOR_O2_PCT_ALPHA],
            noise_scale             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_O2_PCT_NOISE_SCALE],
            lower_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_O2_PCT_LOWER_LIMIT],
            upper_limit             = equipment_configurations_float64_vector[IDX_EQ_SENSOR_O2_PCT_UPPER_LIMIT],
        )
        room_state_float64_vector[IDX_ROOM_STATE_O2_PCT_FILTERED_VAL] = o2_pct_supply_filtered_val

@njit
def defrost_time_controller(
    room_state_float64_vector,
    room_state_int64_vector,
    room_ephemeral_int64_vector,
    defrost_event_ledger,
    equipment_configurations_float64_vector,
    equipment_configurations_int64_vector,
):
    if equipment_configurations_int64_vector[IDX_EQ_DEFROST_ENABLED]:
        phase = room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE]
        now = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]

        if phase == DF_NORMAL and now >= room_state_int64_vector[IDX_ROOM_STATE_NEXT_DEFROST_MS]:
            room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE] = DF_WAIT_FOR_STOP
            room_state_int64_vector[IDX_ROOM_STATE_NEXT_DEFROST_MS] = now + equipment_configurations_int64_vector[IDX_EQ_DEFROST_INTERVAL_MS]
            room_state_int64_vector[IDX_ROOM_STATE_DEFROST_CYCLE_START_MS] = now
            room_state_int64_vector[IDX_ROOM_STATE_DEFROST_TERMINATION_REASON] = 0
            room_state_float64_vector[IDX_ROOM_STATE_DEFROST_MELTED_MASS_ACCUM_KG] = 0.0
            room_state_float64_vector[IDX_ROOM_STATE_DEFROST_COIL_MELTED_MASS_ACCUM_KG] = 0.0
            room_state_float64_vector[IDX_ROOM_STATE_DEFROST_PAN_MELTED_MASS_ACCUM_KG] = 0.0
            room_state_float64_vector[IDX_ROOM_STATE_DEFROST_FACE_SHED_MASS_ACCUM_KG] = 0.0

        elif phase == DF_WAIT_FOR_STOP and not room_state_int64_vector[IDX_ROOM_STATE_REFRIGERATION_IS_ACTIVE]:
            room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE] = DF_SETTLE
            room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS] = now

        elif (
            phase == DF_SETTLE
            and (now - room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS]) >= equipment_configurations_int64_vector[IDX_EQ_DEFROST_SETTLE_MS]
        ):
            room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE] = DF_HEAT
            room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS] = now
            room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PROBE_ABOVE_SINCE_MS] = -1

        elif phase == DF_HEAT:
            probe_above_since_ms = room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PROBE_ABOVE_SINCE_MS]
            if room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAP_FILTERED_VAL] >= equipment_configurations_float64_vector[IDX_EQ_DEFROST_TERMINATION_COIL_T_C]:
                if probe_above_since_ms < 0:
                    probe_above_since_ms = now
            else:
                probe_above_since_ms = -1
            room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PROBE_ABOVE_SINCE_MS] = probe_above_since_ms
            if (
                probe_above_since_ms >= 0
                and (now - probe_above_since_ms) >= equipment_configurations_int64_vector[IDX_EQ_DEFROST_TERMINATION_HOLD_MS]
                and (now - room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS]) >= equipment_configurations_int64_vector[IDX_EQ_DEFROST_MIN_HEAT_MS] 
            ):
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE] = DF_DRIP
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS] = now
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PROBE_ABOVE_SINCE_MS] = -1
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_TERMINATION_REASON] = DF_TERMINATION_PROBE
            elif (now - room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS]) >= equipment_configurations_int64_vector[IDX_EQ_DEFROST_MAX_HEAT_MS]:
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE] = DF_DRIP
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS] = now
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PROBE_ABOVE_SINCE_MS] = -1
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_TERMINATION_REASON] = DF_TERMINATION_TIMEOUT

        elif (
            phase == DF_DRIP
            and (now - room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS]) >= equipment_configurations_int64_vector[IDX_EQ_DEFROST_DRIP_MS]
        ):
            room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE] = DF_FAN_DELAY
            room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS] = now

        elif phase == DF_FAN_DELAY:
            phase_start_ms = room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS]
            compressor_restarted = (
                room_state_int64_vector[IDX_ROOM_STATE_REFRIGERATION_IS_ACTIVE]
                and room_state_int64_vector[IDX_ROOM_STATE_LAST_COMPRESSOR_SWITCH_ON_MS] >= phase_start_ms
            )
            coil_cold_enough = (
                compressor_restarted
                and room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAP_FILTERED_VAL] <= equipment_configurations_float64_vector[IDX_EQ_DEFROST_FAN_RESTART_COIL_T_C]
            )
            fan_delay_timed_out = (now - phase_start_ms) >= equipment_configurations_int64_vector[IDX_EQ_DEFROST_FAN_DELAY_MAX_MS]

            if coil_cold_enough or fan_delay_timed_out:
                row = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DEFROST_EVENT_WRITE_IDX]
                if row >= defrost_event_ledger.shape[0]:
                    raise ValueError("defrost_event_ledger capacity exceeded")
                defrost_event_ledger[row, IDX_DEFROST_EVENT_START_MS] = room_state_int64_vector[IDX_ROOM_STATE_DEFROST_CYCLE_START_MS]
                defrost_event_ledger[row, IDX_DEFROST_EVENT_END_MS] = now
                defrost_event_ledger[row, IDX_DEFROST_EVENT_TERMINATION_REASON] = room_state_int64_vector[IDX_ROOM_STATE_DEFROST_TERMINATION_REASON]
                defrost_event_ledger[row, IDX_DEFROST_EVENT_FROST_MELTED_G] = int(round(1000.0 * room_state_float64_vector[IDX_ROOM_STATE_DEFROST_MELTED_MASS_ACCUM_KG]))
                defrost_event_ledger[row, IDX_DEFROST_EVENT_COIL_ICE_MELTED_G] = int(round(1000.0 * room_state_float64_vector[IDX_ROOM_STATE_DEFROST_COIL_MELTED_MASS_ACCUM_KG]))
                defrost_event_ledger[row, IDX_DEFROST_EVENT_PAN_ICE_MELTED_G] = int(round(1000.0 * room_state_float64_vector[IDX_ROOM_STATE_DEFROST_PAN_MELTED_MASS_ACCUM_KG]))
                defrost_event_ledger[row, IDX_DEFROST_EVENT_FACE_ICE_SHED_G] = int(round(1000.0 * room_state_float64_vector[IDX_ROOM_STATE_DEFROST_FACE_SHED_MASS_ACCUM_KG]))
                room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DEFROST_EVENT_WRITE_IDX] = row + 1
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE] = DF_NORMAL
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS] = now
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_CYCLE_START_MS] = -1
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_TERMINATION_REASON] = 0
                room_state_int64_vector[IDX_ROOM_STATE_EVAP_AIRFLOW_PROOF_GRACE_END_MS] = (
                    now + equipment_configurations_int64_vector[IDX_EQ_DEFROST_AIRFLOW_PROOF_GRACE_MS]
                )

    heater_power_w = 0.0
    if (equipment_configurations_int64_vector[IDX_EQ_DEFROST_ENABLED]
            and room_state_int64_vector[IDX_ROOM_STATE_DEFROST_HEATER_HEALTHY]):
        active_phase = room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE]
        if active_phase == DF_HEAT:
            heater_power_w = equipment_configurations_float64_vector[IDX_EQ_DEFROST_HEATER_W]
        elif active_phase == DF_DRIP:
            heater_power_w = equipment_configurations_float64_vector[IDX_EQ_DEFROST_PAN_HEATER_W]
    room_state_float64_vector[IDX_ROOM_STATE_Q_DEFROST_HEATER_W] = heater_power_w
