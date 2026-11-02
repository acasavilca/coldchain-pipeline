import math
from jit import njit

from matrix_index_maps import *
from constants import *

@njit
def defrost_controller(
    room_state_float64_vector,
    room_state_int64_vector,
    equipment_configurations_float64_vector,
    equipment_configurations_int64_vector,
):
    if equipment_configurations_int64_vector[IDX_EQ_DEFROST_ENABLED]:
        phase = room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE]
        now = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]

        if phase == DF_NORMAL and now >= room_state_int64_vector[IDX_ROOM_STATE_NEXT_DEFROST_MS]:
            room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE] = DF_WAIT_FOR_STOP
            room_state_int64_vector[IDX_ROOM_STATE_NEXT_DEFROST_MS] = now + equipment_configurations_int64_vector[IDX_EQ_DEFROST_INTERVAL_MS]

        elif phase == DF_WAIT_FOR_STOP and not room_state_int64_vector[IDX_ROOM_STATE_REFRIGERATION_IS_ACTIVE]:
            room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE] = DF_SETTLE
            room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS] = now

        elif (
            phase == DF_SETTLE
            and (now - room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS]) >= equipment_configurations_int64_vector[IDX_EQ_DEFROST_SETTLE_MS]
        ):
            room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE] = DF_HEAT
            room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS] = now

        elif phase == DF_HEAT:
            if (
                room_state_float64_vector[IDX_ROOM_STATE_T_COIL_EVAP_FILTERED_VAL] >= equipment_configurations_float64_vector[IDX_EQ_DEFROST_TERMINATION_COIL_T_C]
                and (now - room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS]) >= equipment_configurations_int64_vector[IDX_EQ_DEFROST_MIN_HEAT_MS] 
            ):
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE] = DF_DRIP
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS] = now
            elif (now - room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS]) >= equipment_configurations_int64_vector[IDX_EQ_DEFROST_MAX_HEAT_MS]:
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE] = DF_DRIP
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS] = now

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
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE] = DF_NORMAL
                room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE_START_MS] = now
                room_state_int64_vector[IDX_ROOM_STATE_EVAP_AIRFLOW_PROOF_GRACE_END_MS] = (
                    now + equipment_configurations_int64_vector[IDX_EQ_DEFROST_AIRFLOW_PROOF_GRACE_MS]
                )

    room_state_float64_vector[IDX_ROOM_STATE_Q_DEFROST_HEATER_W] = (
        equipment_configurations_float64_vector[IDX_EQ_DEFROST_HEATER_W]
        if (
            equipment_configurations_int64_vector[IDX_EQ_DEFROST_ENABLED]
            and room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE] == DF_HEAT
            and room_state_int64_vector[IDX_ROOM_STATE_DEFROST_HEATER_HEALTHY]
        )
        else 0.0
    )
