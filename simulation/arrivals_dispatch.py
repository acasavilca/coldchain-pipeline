"""
arrivals_dispatch.py

The stochastic logistics engine: inventory parameters, event planning, arrival/dispatch timing decisions, compact-vector id bookkeeping.
"""

import math
import numpy as np
from numba import njit

from constants import *
from matrix_index_maps import *
from fruit_biology import draw_fruit_quality_grade
from time_calendar import calculate_time_dependent_weights, compute_local_calendar_components
from utils import bounded_normal_draw, draw_single_box_net_weight, draw_tunnel_exit_fruit_temp, gamma_draw, round_up_to_multiple
from workforce_behavior import calculate_actual_carelessness_factor, calculate_fatigue_factor, compute_probability_of_error


@njit
def add_id_element(compact_vector, inverse_lookup, counter, new_slot_id):
    # Guardrail against duplicates or full capacity
    if inverse_lookup[new_slot_id] != -1:
        return counter # Already tracked, do nothing
    if counter >= len(compact_vector):
        raise ValueError("Compact vector overflow.")
    
    # Place value at the end of the active slice
    compact_vector[counter] = new_slot_id
    
    # Store its position in the inverse tracking map
    inverse_lookup[new_slot_id] = counter
    
    return counter + 1


@njit
def remove_element_by_id(compact_vector, inverse_lookup, counter, slot_id_to_remove):
    # 1. Get the current index of the ID we want to delete
    idx_to_remove = inverse_lookup[slot_id_to_remove]
    
    # If the ID isn't in our active vector, exit early
    if idx_to_remove == -1:
        return counter
        
    last_valid_idx = counter - 1
    last_slot_id = compact_vector[last_valid_idx]
    
    # 2. If it's NOT the last element, swap the last element into the dead slot
    if idx_to_remove != last_valid_idx:
        # Move the data in the main tracking array
        compact_vector[idx_to_remove] = last_slot_id
        # Update the inverse lookup pointer for that last element
        inverse_lookup[last_slot_id] = idx_to_remove
        
    # 3. Clean up the trailing edge states
    compact_vector[last_valid_idx] = 0
    inverse_lookup[slot_id_to_remove] = -1 # De-register the removed ID
    
    return counter - 1


@njit
def update_main_logistics_and_operation_flags(
    room_state_int64_vector,
    room_state_float64_vector,
    room_ephemeral_int64_vector,
    equipment_configurations_float64_vector,
):
    room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_NO_FRUIT_ACTIVITY] = (
        room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_ROOM] == 0
        and room_state_int64_vector[IDX_ROOM_STATE_TOTAL_ARRIVALS_PLANNED] == 0
        and room_state_int64_vector[IDX_ROOM_STATE_TOTAL_DISPATCHES_PLANNED] == 0
    )
    room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_REFRIGERATION_SYSTEM_FULLY_OFF] = (
        not room_state_int64_vector[IDX_ROOM_STATE_REFRIGERATION_IS_ACTIVE] 
        and room_state_float64_vector[IDX_ROOM_STATE_EVAP_FAN_MODULATION_FRAC] <= 0.01
        and room_state_float64_vector[IDX_ROOM_STATE_COND_FAN_MODULATION_FRAC] <= 0.01
    )


@njit
def calculate_inventory_parameters(
    rng,
    room_state_int64_vector,
    room_configurations_float64_vector,
    room_configurations_int64_vector,
    employee_parameters_matrix,
    pallet_metadata_int64_matrix,
    outbound_wave_queue_matrix,
    outbound_orders_ledger,
    room_ephemeral_int64_vector,
    active_crop_params_float64_vector,
    active_crop_params_int64_vector,
    true_len_inventory_mass_per_grade_vector,
    box_masses_ledger,
    airflow_pallets_compact_vector,
    f_fatigue,
    dt_engine_ms,
):
    n_grades = active_crop_params_int64_vector[IDX_FRUIT_PARAM_GRADING_SCALE_LEN]
    true_len_inventory_mass_per_grade_vector[...] = 0.0
    accumulated_weighted_mass = 0.0
    schedule_spoilage_dump = False
    total_boxes_withdrawn = 0
    dispatch_type = 0
    assigned_door_code = DOOR_CHANNEL_INTERNAL
    is_first_pallet = True
    # max_queue_size = outbound_wave_queue_matrix.shape[0]
    absolute_spoilage_max_ms = active_crop_params_int64_vector[IDX_FRUIT_PARAM_BIOLOGICAL_SHELF_MS]
    target_retention_ms = active_crop_params_int64_vector[IDX_FRUIT_PARAM_TARGET_RETENTION_MS]
    # max_num_pallets = pallet_metadata_int64_matrix.shape[0]
    # max_num_boxes = box_masses_ledger.shape[0]
    active_driver_code = rng.integers(0, employee_parameters_matrix.shape[0])
    epoch_timestamp_ms = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]
    is_time_for_spoilage_check = epoch_timestamp_ms % room_configurations_int64_vector[IDX_ROOM_LOGISTICS_SPOIL_CHECK_INTERVAL_MS] == 0

    # scan_idx = room_state_int64_vector[IDX_ROOM_STATE_FIFO_SCAN_IDX]
    n_active = room_state_int64_vector[IDX_ROOM_STATE_NUM_ACTIVE_BATCHES]
    next_order_id = room_state_int64_vector[IDX_ROOM_STATE_NEXT_ORDER_ID]
    total_dispatches_planned = room_state_int64_vector[IDX_ROOM_STATE_TOTAL_DISPATCHES_PLANNED]
    dispatch_wave_write_ptr = room_state_int64_vector[IDX_ROOM_STATE_DISPATCH_WAVE_WRITE_PTR]
    
    if is_time_for_spoilage_check:
        p_spoilage_override = compute_probability_of_error(
            base_carelessness   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CARELESSNESS],
            baseline_p_error    = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_P_BASE_SPOILAGE_OVERRIDE],   # Shared Global Baseline!
            fatigue_factor      = f_fatigue,
            fatigue_sensitivity = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_FATIGUE_SENSITIVITY],
            error_sensitivity   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_ERROR_SENSITIVITY],
        )
        macro_order_delay_ms = gamma_draw(
            rng   = rng,
            mean  = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_MACRO_DUMP_DELAY_MS],
            shape = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_MACRO_DELAY_SHAPE],
        )
    else:
        p_spoilage_override = 1.0

    for i in range(room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_AIRFLOW]):
        pallet_slot = airflow_pallets_compact_vector[i]
        marked_for_disposal = False
        employee_notices = rng.random()
        # Only evaluate slots in inventory
        if pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IS_IN_INVENTORY]:
            grade_idx = pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_QUALITY_GRADE]

            # Check for spoiled pallets
            if pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AGE_MS] > absolute_spoilage_max_ms:
                if is_time_for_spoilage_check:
                    is_available_for_removal = (
                        not pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_ARRIVAL_RUN]
                        and pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AVAILABLE_BOX_COUNT] > 0
                        # and not pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH]
                        # or not pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_DISPATCH_RUN]
                    )
                    if is_available_for_removal:
                        age_days = pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AGE_MS] / MS_PER_DAY
                        max_days = absolute_spoilage_max_ms / MS_PER_DAY
                        # print(
                        #     "DEBUG: Pallet slot", pallet_slot,
                        #     "is age", age_days,
                        #     "days (Max:", max_days,
                        #     "). In inventory:", pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IS_IN_INVENTORY],
                        #     ", Boxes:", pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AVAILABLE_BOX_COUNT]
                        # )
                        # print(f"DEBUG: Pallet slot {pallet_slot} is age {age_days:.2f} days (Max: {max_days}). In inventory: {pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IS_IN_INVENTORY]}, Boxes: {pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AVAILABLE_BOX_COUNT]}")
                        # carelessness_factor
                        if employee_notices >= p_spoilage_override:
                            if total_dispatches_planned < MAX_QUEUE_SIZE:
                                marked_for_disposal = True
                                if not schedule_spoilage_dump:
                                    schedule_spoilage_dump = True
                                    dispatch_type = -1

            pallet_net_fruit_mass = 0.0

            for box_idx in range(pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AVAILABLE_BOX_COUNT]):
                actual_mass = box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_ACTUAL]
                # if actual_mass == SENTINEL_FLOAT64:
                #     continue
                target_mass = box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TARGET]
                # start_mass  = box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_START]

                if not marked_for_disposal:
                    # Condition A: Pallet is currently arriving (Mass is stepping up)
                    if pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_ARRIVAL_RUN]:
                        pallet_net_fruit_mass += target_mass
                    # Condition B: Pallet is undergoing dispatch (Extract only the remaining base layer)
                    elif pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_DISPATCH_RUN]:
                        if target_mass > 0.0:
                            # If the box target is greater than 0, it stays on the rack
                            pallet_net_fruit_mass += actual_mass
                    # Condition C: Static inventory resting on the racks normally
                    elif pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AVAILABLE_BOX_COUNT] > 0:
                    # elif not pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH]:
                        pallet_net_fruit_mass += actual_mass

                elif target_mass > 0.0:
                    box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TARGET] = 0.0
                    box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_BOX_PACKAGE_MASS, IDX_BOX_LEDGER_TARGET] = 0.0
                    # spoiled_mass_to_remove += actual_mass

            # Schedule dispatch
            if marked_for_disposal:
                q_write_slot = dispatch_wave_write_ptr % MAX_QUEUE_SIZE
                num_boxes_available = pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AVAILABLE_BOX_COUNT]
                num_boxes_taken_from_pallet = pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AVAILABLE_BOX_COUNT]
                # pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH] = 1

                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_TARGET_ROOM_SLOT] = pallet_slot
                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_BOXES_TO_EXTRACT] = num_boxes_taken_from_pallet
                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_PARENT_ORDER_ID]  = next_order_id
                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IS_EVENT_SCHEDULED] = 1
                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_ACTIVE_DRIVER_CODE] = active_driver_code
                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_ASSIGNED_DOOR_CODE] = assigned_door_code
                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_ERROR_THRESHOLD_SCALED] = int(rng.random() * 1_000_000)
                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IN_FULL_DISPATCH] = 1

                box_start_idx = num_boxes_available - 1
                box_end_idx = num_boxes_available - num_boxes_taken_from_pallet
                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_BOX_START_IDX] = box_start_idx
                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_BOX_END_IDX] = box_end_idx


                if is_first_pallet:
                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IS_FIRST_IN_MACRO_ORDER] = 1
                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_MACRO_ORDER_START_MS] = epoch_timestamp_ms + round_up_to_multiple(macro_order_delay_ms, dt_engine_ms)
                    is_first_pallet = False
                else:
                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IS_FIRST_IN_MACRO_ORDER] = 0

                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IS_LAST_IN_MACRO_ORDER] = 0

                pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AVAILABLE_BOX_COUNT] -= num_boxes_taken_from_pallet
                total_boxes_withdrawn += num_boxes_taken_from_pallet

                total_dispatches_planned += 1
                dispatch_wave_write_ptr += 1
                n_active -= 1

            true_len_inventory_mass_per_grade_vector[grade_idx] += pallet_net_fruit_mass

            # Calculate chronological urgency weight
            if pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AGE_MS] > target_retention_ms:
                excess_time_ms = pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AGE_MS] - target_retention_ms

                time_weight = 1 + (excess_time_ms / target_retention_ms)
                time_weight = min(time_weight, 5.0)

                accumulated_weighted_mass += pallet_net_fruit_mass * time_weight
            else:
                accumulated_weighted_mass += pallet_net_fruit_mass

    if schedule_spoilage_dump:
        outbound_wave_queue_matrix[(dispatch_wave_write_ptr - 1) % MAX_QUEUE_SIZE, IDX_WAVE_IS_LAST_IN_MACRO_ORDER] = 1

        outbound_order_row_idx = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_OUTBOUND_ORDER_ROW_IDX]
        outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_ORDER_ID] = next_order_id # id
        outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_TIMESTAMP_MS] = epoch_timestamp_ms # timestamp
        outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_TARGET_FRUIT_MASS] = SENTINEL_FLOAT64 # requested mass out
        outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_TOTAL_BOX_COUNT] = total_boxes_withdrawn
        outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_MINIMUM_GRADE] = SENTINEL_FLOAT64
        outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_CROP_TYPE] = active_crop_params_int64_vector[IDX_FRUIT_PARAM_FRUIT_TYPE_CODE]
        outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_ASSIGNED_DOOR_CODE] = assigned_door_code
        outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_ASSIGNED_DRIVER] = active_driver_code
        outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_DISPATCH_TYPE] = dispatch_type

        next_order_id += 1
        outbound_order_row_idx += 1
        
        room_state_int64_vector[IDX_ROOM_STATE_NUM_ACTIVE_BATCHES] = n_active
        room_state_int64_vector[IDX_ROOM_STATE_NEXT_ORDER_ID] = next_order_id
        room_state_int64_vector[IDX_ROOM_STATE_TOTAL_DISPATCHES_PLANNED] = total_dispatches_planned
        room_state_int64_vector[IDX_ROOM_STATE_DISPATCH_WAVE_WRITE_PTR] = dispatch_wave_write_ptr
        room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_OUTBOUND_ORDER_ROW_IDX] = outbound_order_row_idx

    urgency_scalar = accumulated_weighted_mass * active_crop_params_float64_vector[IDX_FRUIT_PARAM_SCHED_INV_MIN_SHIP_MASS_KG]
    # urgency_scalar = accumulated_weighted_mass * active_crop_params_float64_vector[IDX_FRUIT_PARAM_SCHED_INV_MIN_SHIP_MASS_KG] # 1 / max(1.0, active_crop_params_float64_vector[IDX_FRUIT_PARAM_SCHED_MIN_SHIP_MASS_KG])
    return urgency_scalar


@njit
def calculate_ramp_up_arrival_scaler(
    epoch_timestamp_ms,
    downtime_scheduled,
    ramp_up_scheduled,
    ramp_up_start_ms,
    nominal_ramp_up_duration_ms,
    settling_window_ms,
    base_scaler=0.7,
):
    # 1. Total shutdown during scheduled downtime
    if downtime_scheduled:
        return 0.0

    # 2. Steady-state commercial operation
    if not ramp_up_scheduled:
        return 1.0

    nominal_ramp_end_ms = ramp_up_start_ms + nominal_ramp_up_duration_ms

    # 3. Strict lockout during the primary pulldown phase
    if epoch_timestamp_ms < nominal_ramp_end_ms:
        return 0.0

    # 4. Smooth ramp from base_scaler to 1.0 during the extended settling tail
    elapsed_settling_ms = epoch_timestamp_ms - nominal_ramp_end_ms
    progress = min(elapsed_settling_ms / max(settling_window_ms, 1), 1.0)

    # Smoothstep interpolation (S-curve) to prevent abrupt boundary steps
    smooth_progress = progress * progress * (3.0 - 2.0 * progress)

    return base_scaler + (1.0 - base_scaler) * smooth_progress


@njit
def compute_dynamic_settling_window(
    t_room_filtered,
    t_setpoint,
    base_window_ms=MS_PER_HOUR * 2,
    max_window_ms=MS_PER_HOUR * 12,
):
    # Temperature error magnitude (absolute difference from setpoint)
    temp_error = abs(t_room_filtered - t_setpoint)
    
    # Scale settling window dynamically based on how far the room is from setpoint
    # e.g., 1 hour per degree of error above a 1.0°C deadband tolerance
    excess_error = max(0.0, temp_error - 1.0)
    dynamic_window_ms = base_window_ms + int(excess_error * MS_PER_HOUR)
    
    return min(dynamic_window_ms, max_window_ms)


@njit
def event_planner(
    rng,
    room_state_float64_vector,
    room_state_int64_vector,
    forcing_cache_int64_vector,
    room_configurations_float64_vector,
    room_configurations_int64_vector,
    active_crop_params_float64_vector,
    active_crop_params_int64_vector,
    active_crop_historical_yield_matrix,
    active_crop_market_scarcity_matrix,
    active_crop_base_grades_probs_matrix,
    inventory_mass_per_grade_vector,
    employee_parameters_matrix,
    room_ephemeral_int64_vector,
    downtime_timestamps_ms_vector,
    lambda_event_per_hour_matrix,
    day_of_week_weights,

    pallet_metadata_int64_matrix,
    pallet_timeline_int64_matrix,
    outbound_wave_queue_matrix,
    outbound_orders_ledger,
    inbound_wave_queue_matrix,
    inbound_orders_ledger,
    box_masses_ledger,

    available_pallet_indices,
    airflow_pallets_compact_vector,

    dt_engine_ms,
    dt_planner_ms,
    dt_planner_hr,
):
    event_code = 0
    epoch_timestamp_ms = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]
    if epoch_timestamp_ms % dt_planner_ms == 0:
        downtime_scheduled = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_SCHEDULED]
        baseline_year = active_crop_params_int64_vector[IDX_FRUIT_PARAM_YEARLY_MAP_BASELINE]
        year, month, _, hour, day_of_week, _, is_last_weekday = compute_local_calendar_components(epoch_timestamp_ms, forcing_cache_int64_vector[IDX_CACHE_LOCAL_UTC_MS])
        # print("epoch_timestamp_ms:", epoch_timestamp_ms, "year:", year, "month:", month, "day_of_week:", day_of_week)
    
        f_fatigue = calculate_fatigue_factor(
            hour,
            day_of_week,
            is_last_weekday,
        )

        # Calculate weight due to historical yield and day of week
        n_map = active_crop_params_int64_vector[IDX_FRUIT_PARAM_YEARLY_MAP_LEN]
        climate_yield_weight, weekly_labor_weight, throughput_weight = calculate_time_dependent_weights( 
            year,
            month,
            day_of_week,
            active_crop_historical_yield_matrix[:n_map, ...],
            day_of_week_weights,
            baseline_year,
        )

        n_grades = active_crop_params_int64_vector[IDX_FRUIT_PARAM_GRADING_SCALE_LEN]
        true_len_inventory_mass_per_grade_vector = inventory_mass_per_grade_vector[:n_grades]
        # Calculate inventory parameters
        urgency_scalar = calculate_inventory_parameters(
            rng,
            room_state_int64_vector,
            room_configurations_float64_vector,
            room_configurations_int64_vector,
            employee_parameters_matrix,
            pallet_metadata_int64_matrix,
            outbound_wave_queue_matrix,
            outbound_orders_ledger,
            room_ephemeral_int64_vector,
            active_crop_params_float64_vector,
            active_crop_params_int64_vector,
            true_len_inventory_mass_per_grade_vector,
            box_masses_ledger,
            airflow_pallets_compact_vector,
            f_fatigue,
            dt_engine_ms,
        )

        inventory_mass_kg = np.sum(true_len_inventory_mass_per_grade_vector)

        # bypass_temperature_checks = False
        # force_clear_room = False
        # forced_urgency_modifier = 0.0
        # forced_dispatch_urgency_factor = 0.0
        current_base_min_ship_mass = active_crop_params_float64_vector[IDX_FRUIT_PARAM_SCHED_MIN_SHIP_MASS_KG]

        if (
            downtime_scheduled
            and not room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_NO_FRUIT_ACTIVITY]
            and inventory_mass_kg > MASS_KG_FLOOR
        ): # Downtime currently happening but there's still fruit inside
            downtime_idx = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_IDX]
            time_until_deadline_ms = downtime_timestamps_ms_vector[downtime_idx + 1] - epoch_timestamp_ms

            # Threshold 1: The 12-Hour Critical Gate (Total Emergency Evacuation)
            if time_until_deadline_ms < MS_PER_12_HOURS:
                bypass_temperature_checks = True
                force_clear_room = True
                forced_urgency_modifier = 1.0
                forced_dispatch_urgency_factor = 1.0
                current_base_min_ship_mass /= 500.0

            # Threshold 2: The 24-Hour Urgent Gate (Force Room Clear Mode)
            elif time_until_deadline_ms < MS_PER_24_HOURS:
                bypass_temperature_checks = False
                force_clear_room = True
                forced_urgency_modifier = 1.0
                forced_dispatch_urgency_factor = 1.0
                current_base_min_ship_mass /= 50.0

            # Threshold 3: The 48-Hour Warning Gate (Dampen Economic Shipment Thresholds)
            elif time_until_deadline_ms < MS_PER_48_HOURS:
                bypass_temperature_checks = False
                force_clear_room = False
                forced_urgency_modifier = 1.0
                forced_dispatch_urgency_factor = 1.0
                current_base_min_ship_mass /= 5.0

            # Threshold 4: The 72-Hour Operational Buffer Gate (Raise Shipment Urgency Only)
            elif time_until_deadline_ms < MS_PER_72_HOURS:
                bypass_temperature_checks = False
                force_clear_room = False
                forced_urgency_modifier = 0.0
                forced_dispatch_urgency_factor = 1.0
            else:
                bypass_temperature_checks = False
                force_clear_room = False
                forced_urgency_modifier = 0.0
                forced_dispatch_urgency_factor = 0.0

        else:
            # Normal commercial operations rule the time step
            bypass_temperature_checks = False
            force_clear_room = False
            forced_urgency_modifier = 0.0
            forced_dispatch_urgency_factor = 0.0


        lambda_arrival_per_hour = lambda_event_per_hour_matrix[IDX_LAMBDA_DIRECTION_INBOUND, hour]
        lambda_shipment_per_hour = lambda_event_per_hour_matrix[IDX_LAMBDA_DIRECTION_OUTBOUND, hour]
        
        #### DEBUG ####
        # p_arrival = 0.0
        # p_arrival = min(1.0, lambda_arrival_per_hour * dt_planner_hr * throughput_weight * weekly_labor_weight) if not downtime_scheduled else 0.0
        # base_p_shipment = min(1.0, lambda_shipment_per_hour * dt_planner_hr * throughput_weight * weekly_labor_weight)

        # arrival_scaler = 0.5 if room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_SCHEDULED] else 1.0 # lower p_arrival during ramp-up
        # if not downtime_scheduled:
        #     if room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_SCHEDULED]:
        #         if epoch_timestamp_ms >= room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_START_MS] + active_crop_params_int64_vector[IDX_FRUIT_PARAM_RAMP_UP_DURATION_MS]:
        #             arrival_scaler = 0.7
        #         else:
        #             arrival_scaler = 0.0
        #     else:
        #         arrival_scaler = 1.0
        # else:
        #     arrival_scaler = 0.0
        settling_window_ms = compute_dynamic_settling_window(
            t_room_filtered = room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_FILTERED_VAL],
            t_setpoint      = active_crop_params_float64_vector[IDX_FRUIT_PARAM_T_SETPOINT],
            base_window_ms  = MS_PER_HOUR // 12,         # Minimum 1-hour buffer
            max_window_ms   = MS_PER_HOUR,     # Cap at 8 hours max
        )

        arrival_scaler = calculate_ramp_up_arrival_scaler(
            epoch_timestamp_ms          = epoch_timestamp_ms,
            downtime_scheduled          = downtime_scheduled,
            ramp_up_scheduled           = room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_SCHEDULED],
            ramp_up_start_ms            = room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_START_MS],
            nominal_ramp_up_duration_ms = active_crop_params_int64_vector[IDX_FRUIT_PARAM_RAMP_UP_DURATION_MS],
            settling_window_ms          = settling_window_ms,  # Window to ramp from 0.7 -> 1.0
            base_scaler                 = 0.95,
        )
        p_arrival = min(1.0, lambda_arrival_per_hour * dt_planner_hr * throughput_weight * arrival_scaler) # if not downtime_scheduled else 0.0

        base_p_shipment = min(1.0, lambda_shipment_per_hour * dt_planner_hr * throughput_weight)

        # Scale economic triggers
        economic_min_ship_mass = current_base_min_ship_mass * climate_yield_weight # throughput_weight

        # Run continuous inventory mapping and register spoilage dump waves

        # # Calculate inventory parameters
        # urgency_scalar, active_crop_inventory_mass_per_grade_vector = calculate_inventory_parameters(
        #     rng,
        #     room_state_int64_vector,
        #     room_configurations_float64_vector,
        #     room_configurations_int64_vector,
        #     employee_parameters_matrix,
        #     pallet_metadata_int64_matrix,
        #     outbound_wave_queue_matrix,
        #     outbound_orders_ledger,
        #     room_ephemeral_int64_vector,
        #     active_crop_params_float64_vector,
        #     active_crop_params_int64_vector,
        #     raw_active_crop_inventory_mass_per_grade_vector,
        #     box_masses_ledger,
        #     airflow_pallets_compact_vector,
        #     f_fatigue,
        #     dt_engine_ms,
        # )


        # max_inventory_mass_kg = active_crop_params_vector[IDX_FRUIT_PARAM_SCHED_MAX_INVENTORY_KG]
        # if inventory_mass_kg >= max_inventory_mass_kg:
        #     p_arrival = 0.0
        inventory_pallet_capacity = active_crop_params_int64_vector[IDX_FRUIT_PARAM_SCHED_INVENTORY_PALLET_CAPACITY]
        num_pallets_in_inventory = np.sum(pallet_metadata_int64_matrix[:, IDX_PALLET_METADATA_IS_IN_INVENTORY])
        num_pallets_pending_arrival = room_state_int64_vector[IDX_ROOM_STATE_TOTAL_ARRIVALS_PLANNED]
        if (num_pallets_in_inventory + num_pallets_pending_arrival) >= inventory_pallet_capacity:
            p_arrival = 0.0
        if inventory_mass_kg <= MASS_KG_FLOOR:
            base_p_shipment = 0.0

        urgency_modifier = max(urgency_scalar, forced_dispatch_urgency_factor) * active_crop_params_float64_vector[IDX_FRUIT_PARAM_SCHED_BASE_P_URGENT_SHIP] * dt_planner_hr

        # Still restrict shipments to working hours
        p_shipment = min(1.0, base_p_shipment + max(urgency_modifier, forced_urgency_modifier)) * (lambda_shipment_per_hour > 0.0)

        # DEBUG#
        # print("PRINT CHECK IN planner function")
        # Arrival into cold room
        if rng.random() < p_arrival:
            # Once the die has been cast, if an arrival is the outcome, as reasonable mass must arrive
            raw_sampled_mass = rng.gamma(
                shape = active_crop_params_float64_vector[IDX_FRUIT_PARAM_SCHED_ARRIVAL_SHAPE], 
                scale = active_crop_params_float64_vector[IDX_FRUIT_PARAM_SCHED_ARRIVAL_SCALE] * climate_yield_weight
            )
            
            box_net_fruit_mass_mean_kg     = active_crop_params_float64_vector[IDX_FRUIT_PARAM_BOX_NET_MASS_MEAN]
            pallet_max_unit_boxes_capacity = active_crop_params_int64_vector[IDX_FRUIT_PARAM_PALLET_UNIT_BOXES]
            mean_total_net_weight          = active_crop_params_float64_vector[IDX_FRUIT_PARAM_PALLET_NET_MASS_MEAN]

            total_target_mass_delta = math.ceil(raw_sampled_mass / mean_total_net_weight) * mean_total_net_weight
            # available_rack_space_kg = max(0.0, max_inventory_mass_kg - inventory_mass_kg)

            # if total_target_mass_delta > available_rack_space_kg:
            #     total_target_mass_delta = math.ceil(available_rack_space_kg / mean_total_net_weight) * mean_total_net_weight

            # if total_target_mass_delta >= mean_total_net_weight:
            #     event_code = 1
            # else:
            #     event_code = 0

            p_partial_batch = max(active_crop_params_float64_vector[IDX_FRUIT_PARAM_SCHED_BASE_P_PARTIAL_BATCH] * (2.0 - climate_yield_weight), 0.0)
            p_partial_batch = min(p_partial_batch, 1.0)

            if rng.random() < p_partial_batch:
                mass_reduction_fraction = rng.uniform(0.3, 0.7)
                # total_target_mass_delta -= mass_reduction_fraction * typical_net_weight_per_pallet
                total_target_mass_delta -= mass_reduction_fraction * mean_total_net_weight
                total_target_mass_delta = math.ceil(total_target_mass_delta / mean_total_net_weight) * mean_total_net_weight

            total_boxes_arrival_target = int(total_target_mass_delta / box_net_fruit_mass_mean_kg)
            total_pallets_arrival_target = int(math.ceil(total_boxes_arrival_target / pallet_max_unit_boxes_capacity))
            available_pallet_rack_spaces = inventory_pallet_capacity - num_pallets_in_inventory

            # ### DEBUG ####
            # print("scale used:", active_crop_params_vector[IDX_FRUIT_PARAM_SCHED_ARRIVAL_SCALE] * climate_yield_weight)
            # print("raw_sampled_mass:", raw_sampled_mass)
            # print("event_code:", event_code)
            # ### DEBUG ####

            if total_pallets_arrival_target > available_pallet_rack_spaces:
                event_code = 0
            else:
                event_code = 1

            if event_code == 1:
                # Roll a random integer draw to select an employee on duty
                active_driver_code = rng.integers(0, employee_parameters_matrix.shape[0])
                
                pallet_count = 0
                total_box_count = 0
                total_target_net_weight_all_pallets = 0.0
                is_first_pallet = True
                macro_order_delay_ms = gamma_draw(
                    rng, 
                    mean=room_configurations_float64_vector[IDX_ROOM_LOGISTICS_MACRO_ARR_DELAY_MS], 
                    shape=room_configurations_float64_vector[IDX_ROOM_LOGISTICS_MACRO_DELAY_SHAPE]
                )
                macro_order_delay_ms = round_up_to_multiple(macro_order_delay_ms, dt_engine_ms)

                quality_grade = draw_fruit_quality_grade(
                    climate_yield_weight,
                    active_crop_base_grades_probs_matrix[:n_grades, IDX_FRUIT_BASE_QUALITY_DISTRIBUTION],
                    active_crop_params_float64_vector[IDX_FRUIT_PARAM_QUALITY_SENSITIVITY],
                    room_state_int64_vector[IDX_ROOM_STATE_ACTIVE_FRUIT_TYPE_CODE],
                    rng,
                )

                p_internal_base = active_crop_params_float64_vector[IDX_FRUIT_PARAM_INTERNAL_ARRIVAL_BASE_P]
                p_external_base = 1.0 - p_internal_base
                p_external_scaled = min(0.40, p_external_base * climate_yield_weight)
                total_route_weight = p_internal_base + p_external_scaled
                p_internal_normalized = p_internal_base / total_route_weight

                assigned_door_code = 1 - (rng.random() < p_internal_normalized) # internal door is 0

                box_net_mass_mean    = active_crop_params_float64_vector[IDX_FRUIT_PARAM_BOX_NET_MASS_MEAN]
                box_net_mass_scale   = active_crop_params_float64_vector[IDX_FRUIT_PARAM_BOX_NET_MASS_SCALE]
                legal_minimum_weight = active_crop_params_float64_vector[IDX_FRUIT_PARAM_BOX_LEGAL_MIN_MASS]

                n_active = room_state_int64_vector[IDX_ROOM_STATE_NUM_ACTIVE_BATCHES]
                next_batch_id = room_state_int64_vector[IDX_ROOM_STATE_NEXT_BATCH_ID]
                next_inbound_delivery_id = room_state_int64_vector[IDX_ROOM_STATE_NEXT_INBOUND_DELIVERY_ID]
                total_arrivals_planned = room_state_int64_vector[IDX_ROOM_STATE_TOTAL_ARRIVALS_PLANNED]
                arrival_wave_write_ptr = room_state_int64_vector[IDX_ROOM_STATE_ARRIVAL_WAVE_WRITE_PTR]
                inbound_order_row_idx = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_INBOUND_ORDER_ROW_IDX]

                while (
                    n_active < MAX_ACTIVE_PALLETS
                    and total_box_count < total_boxes_arrival_target
                ):
                    # Define pallet_net_weight
                    # target_pallet_net_weight = typical_net_weight_per_pallet if total_target_mass_delta >= typical_net_weight_per_pallet else total_target_mass_delta
                    # total_target_net_weight_all_pallets += target_pallet_net_weight # Update

                    # Define slot for batch/pallet
                    q_write_slot = arrival_wave_write_ptr % MAX_ACTIVE_PALLETS

                    pallet_count += 1

                    # Determine actual mass_delta
                    mass_delta = 0.0
                    box_count = 0
                    # while mass_delta <= target_pallet_net_weight:
                    while box_count < pallet_max_unit_boxes_capacity:
                        if total_box_count >= total_boxes_arrival_target:
                            break
                        box_net_mass = draw_single_box_net_weight(
                            rng            = rng,
                            mean_pack      = box_net_mass_mean,
                            std_pack       = box_net_mass_scale,
                            legal_minimum  = legal_minimum_weight,
                        )
                        box_masses_ledger[box_count, q_write_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TARGET] = box_net_mass
                        box_masses_ledger[box_count, q_write_slot, IDX_BOX_LEDGER_BOX_PACKAGE_MASS, IDX_BOX_LEDGER_TARGET] = active_crop_params_float64_vector[IDX_FRUIT_PARAM_BOX_PACKAGING_MASS]
                        box_masses_ledger[box_count, q_write_slot, ..., IDX_BOX_LEDGER_START] = 0.0
                        box_masses_ledger[box_count, q_write_slot, ..., IDX_BOX_LEDGER_TRANSIT] = 0.0
                        box_masses_ledger[box_count, q_write_slot, ..., IDX_BOX_LEDGER_ACTUAL] = 0.0

                        mass_delta += box_net_mass
                        box_count += 1
                        total_box_count += 1

                    # Add pallet weight to box ledger
                    pallet_weight_fraction_per_box = active_crop_params_float64_vector[IDX_FRUIT_PARAM_PALLET_BASE_MASS] / box_count
                    for box_idx in range(box_count):
                        box_masses_ledger[box_idx, q_write_slot, IDX_BOX_LEDGER_PALLET_MASS, IDX_BOX_LEDGER_TARGET] = pallet_weight_fraction_per_box

                    # Fill tracking arrays
                    pallet_metadata_int64_matrix[q_write_slot, IDX_PALLET_METADATA_BATCH_ID] = next_batch_id
                    pallet_metadata_int64_matrix[q_write_slot, IDX_PALLET_METADATA_QUALITY_GRADE] = quality_grade
                    pallet_metadata_int64_matrix[q_write_slot, IDX_PALLET_METADATA_CROP_TYPE] = active_crop_params_int64_vector[IDX_FRUIT_PARAM_FRUIT_TYPE_CODE]
                    # pallet_metadata_int64_matrix[q_write_slot, IDX_PALLET_METADATA_AVAILABLE_BOX_COUNT] = box_count

                    # Commit their combined forklift convective load directly to the slot physics column
                    # pallet_physics_float64_matrix[q_write_slot, IDX_PALLET_PHYSICS_ACTIVE_DRIVER_CONVECTIVE_Q] = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q]

                    pallet_timeline_int64_matrix[q_write_slot, IDX_TIMELINE_PALLET_REGISTRATION_MS] = epoch_timestamp_ms

                    inbound_wave_queue_matrix[q_write_slot, IDX_WAVE_TARGET_ROOM_SLOT]   = q_write_slot
                    inbound_wave_queue_matrix[q_write_slot, IDX_WAVE_BOXES_TO_ARRIVE]    = box_count
                    inbound_wave_queue_matrix[q_write_slot, IDX_WAVE_PARENT_ORDER_ID]    = next_inbound_delivery_id
                    inbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IS_EVENT_SCHEDULED] = 1
                    inbound_wave_queue_matrix[q_write_slot, IDX_WAVE_ACTIVE_DRIVER_CODE] = active_driver_code
                    inbound_wave_queue_matrix[q_write_slot, IDX_WAVE_ASSIGNED_DOOR_CODE] = assigned_door_code
                    inbound_wave_queue_matrix[q_write_slot, IDX_WAVE_BYPASS_TEMPERATURE_CHECK] = bypass_temperature_checks
                    inbound_wave_queue_matrix[q_write_slot, IDX_WAVE_ERROR_THRESHOLD_SCALED] = int(rng.random() * 1_000_000)

                    box_start_idx = box_count - 1
                    box_end_idx = 0
                    inbound_wave_queue_matrix[q_write_slot, IDX_WAVE_BOX_START_IDX] = box_start_idx
                    inbound_wave_queue_matrix[q_write_slot, IDX_WAVE_BOX_END_IDX] = box_end_idx

                    if is_first_pallet:
                        inbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IS_FIRST_IN_MACRO_ORDER] = 1
                        inbound_wave_queue_matrix[q_write_slot, IDX_WAVE_MACRO_ORDER_START_MS] = epoch_timestamp_ms + macro_order_delay_ms
                        is_first_pallet = False
                    else:
                        inbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IS_FIRST_IN_MACRO_ORDER] = 0

                    inbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IS_LAST_IN_MACRO_ORDER] = 0

                    # Update batch tracking parameters
                    total_target_net_weight_all_pallets += mass_delta
                    total_arrivals_planned += 1
                    arrival_wave_write_ptr += 1
                    n_active += 1
                    next_batch_id += 1
                    
                if event_code == 1:
                    # is_regular = not assigned_door_code
                    inbound_wave_queue_matrix[(arrival_wave_write_ptr - 1) % MAX_ACTIVE_PALLETS, IDX_WAVE_IS_LAST_IN_MACRO_ORDER] = 1

                    inbound_order_row_idx = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_INBOUND_ORDER_ROW_IDX]
                    inbound_orders_ledger[inbound_order_row_idx, IDX_ORDER_LEDGER_ORDER_ID] = next_inbound_delivery_id # id
                    inbound_orders_ledger[inbound_order_row_idx, IDX_ORDER_LEDGER_TIMESTAMP_MS] = epoch_timestamp_ms # timestamp
                    inbound_orders_ledger[inbound_order_row_idx, IDX_ORDER_LEDGER_TARGET_FRUIT_MASS] = total_target_net_weight_all_pallets
                    inbound_orders_ledger[inbound_order_row_idx, IDX_ORDER_LEDGER_TOTAL_BOX_COUNT] = total_box_count
                    inbound_orders_ledger[inbound_order_row_idx, IDX_ORDER_LEDGER_TOTAL_PALLET_COUNT] = pallet_count
                    inbound_orders_ledger[inbound_order_row_idx, IDX_ORDER_LEDGER_CROP_TYPE] = active_crop_params_int64_vector[IDX_FRUIT_PARAM_FRUIT_TYPE_CODE]
                    inbound_orders_ledger[inbound_order_row_idx, IDX_ORDER_LEDGER_ASSIGNED_DOOR_CODE] = assigned_door_code
                    inbound_orders_ledger[inbound_order_row_idx, IDX_ORDER_LEDGER_ASSIGNED_DRIVER] = active_driver_code
                    # inbound_orders_ledger[inbound_order_row_idx, IDX_ORDER_LEDGER_DISPATCH_TYPE] = sentinel # is_regular * 1.0 # Regular arrivals happen by internal door

                    next_inbound_delivery_id += 1
                    inbound_order_row_idx += 1

                    room_state_int64_vector[IDX_ROOM_STATE_NUM_ACTIVE_BATCHES]         = n_active
                    room_state_int64_vector[IDX_ROOM_STATE_NEXT_BATCH_ID]              = next_batch_id
                    room_state_int64_vector[IDX_ROOM_STATE_TOTAL_ARRIVALS_PLANNED]     = total_arrivals_planned
                    room_state_int64_vector[IDX_ROOM_STATE_ARRIVAL_WAVE_WRITE_PTR]     = arrival_wave_write_ptr
                    room_state_int64_vector[IDX_ROOM_STATE_NEXT_INBOUND_DELIVERY_ID]   = next_inbound_delivery_id
                    room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_INBOUND_ORDER_ROW_IDX] = inbound_order_row_idx

        # Shipment out of cold room through external door
        if rng.random() < p_shipment:
            # Roll for the size of the resquested fruit mass
            raw_mass = rng.gamma(
                shape = active_crop_params_float64_vector[IDX_FRUIT_PARAM_SCHED_SHIPMENT_SHAPE], 
                scale = active_crop_params_float64_vector[IDX_FRUIT_PARAM_SCHED_SHIPMENT_SCALE] * climate_yield_weight,
            ) # MODIFIED
            requested_mass_out = round_up_to_multiple(raw_mass, active_crop_params_float64_vector[IDX_FRUIT_PARAM_DISPATCH_MASS_MULT])

            if force_clear_room:
                minimum_grade_requested = active_crop_params_int64_vector[IDX_FRUIT_PARAM_GRADING_SCALE_LEN] - 1 # worst grade: 0 (best) to n (worst)
            else:
                # active_fruit_type_code = room_state_int64_vector[IDX_ROOM_STATE_ACTIVE_FRUIT_TYPE_CODE]
                market_scarcity_weight, _, _ = calculate_time_dependent_weights( 
                    year,
                    month,
                    day_of_week,
                    active_crop_market_scarcity_matrix[:n_map, ...],
                    day_of_week_weights,
                    baseline_year,
                )

                if room_state_int64_vector[IDX_ROOM_STATE_ACTIVE_FRUIT_TYPE_CODE] == FRUIT_TYPE_AVOCADO:
                    # Avocados handle shortages via wider quality acceptance thresholds
                    demand_weight_factor = market_scarcity_weight # 2.0 - market_scarcity_weight 
                else:
                    # Berries experience extreme cosmetic demands when prices spike
                    demand_weight_factor = 1.0 / max(0.1, market_scarcity_weight)

                minimum_grade_requested = draw_fruit_quality_grade(
                    demand_weight_factor,
                    active_crop_base_grades_probs_matrix[:n_grades, IDX_FRUIT_BASE_QUALITY_DEMANDED],
                    active_crop_params_float64_vector[IDX_FRUIT_PARAM_CUSTOMER_QUALITY_SENSITIVITY],
                    room_state_int64_vector[IDX_ROOM_STATE_ACTIVE_FRUIT_TYPE_CODE],
                    rng,
                )

            valid_inventory_mass_kg = 0.0
            grade_idx = 0
            while grade_idx <= minimum_grade_requested: # A: 0, B: 1, C:2
                valid_inventory_mass_kg += true_len_inventory_mass_per_grade_vector[grade_idx]
                grade_idx += 1

            if not force_clear_room:
                if requested_mass_out <= valid_inventory_mass_kg:
                    # If the request is worth the cost of dispatching for the season, execute
                    if requested_mass_out > economic_min_ship_mass:
                        event_code = 2
                    # If not, do we break protocol due to avoid letting batches go to waste?
                    else:
                        p_bypass_protocol = min(1.0, urgency_scalar) # CHECK IF IT'S THIS OR urgency_modifier
                        # Roll a second time to symbolize decision-making in emergency situations
                        if rng.random() < p_bypass_protocol and requested_mass_out >= (economic_min_ship_mass*0.25):
                            event_code = 2
                    dispatch_type = 1
                else:
                    event_code = 0
            else:
                requested_mass_out = valid_inventory_mass_kg
                event_code = 2
                dispatch_type = 0

            if event_code == 2:
                # max_queue_size = outbound_wave_queue_matrix.shape[0]
                assigned_door_code = DOOR_CHANNEL_EXTERNAL
                pulp_mass_total = 0.0
                total_boxes_withdrawn = 0
                unemptied_pallet_in_sequence = False
                active_driver_code = rng.integers(0, employee_parameters_matrix.shape[0])
                is_first_pallet = True
                macro_order_delay_ms = gamma_draw(
                    rng, 
                    mean=room_configurations_float64_vector[IDX_ROOM_LOGISTICS_MACRO_DISP_DELAY_MS], 
                    shape=room_configurations_float64_vector[IDX_ROOM_LOGISTICS_MACRO_DELAY_SHAPE]
                )
                macro_order_delay_ms = round_up_to_multiple(macro_order_delay_ms, dt_engine_ms)
                available_pallet_indices[...] = SENTINEL_INT64

                scan_idx = room_state_int64_vector[IDX_ROOM_STATE_FIFO_SCAN_IDX]
                n_active = room_state_int64_vector[IDX_ROOM_STATE_NUM_ACTIVE_BATCHES]
                next_order_id = room_state_int64_vector[IDX_ROOM_STATE_NEXT_ORDER_ID]
                total_dispatches_planned = room_state_int64_vector[IDX_ROOM_STATE_TOTAL_DISPATCHES_PLANNED]
                dispatch_wave_write_ptr = room_state_int64_vector[IDX_ROOM_STATE_DISPATCH_WAVE_WRITE_PTR]
                outbound_order_row_idx = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_OUTBOUND_ORDER_ROW_IDX]

                pallet_count = 0 # NOTE: FIXED, ADDED THIS LINE
                target_pallet_count = 0
                for _ in range(MAX_ACTIVE_PALLETS):
                    if total_dispatches_planned < MAX_QUEUE_SIZE:
                        if pulp_mass_total < requested_mass_out:
                            pallet_slot = scan_idx % MAX_ACTIVE_PALLETS
                            q_write_slot = dispatch_wave_write_ptr % MAX_QUEUE_SIZE
                            is_pallet_available_for_dispatch = (
                                pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IS_IN_INVENTORY]
                                and pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AVAILABLE_BOX_COUNT] > 0
                                # and not pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH] # DEBUG
                            ) # unavailable slots are those marked for full dispatch or those that are no longer in inventory

                            # the scan will pass by the unavailable slots but stop at the first available slot
                            if is_pallet_available_for_dispatch:
                                available_pallet_indices[pallet_count] = scan_idx
                                pallet_grade = pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_QUALITY_GRADE]
                                if pallet_grade > minimum_grade_requested:
                                    if not unemptied_pallet_in_sequence:
                                        unemptied_pallet_in_sequence = True
                                        room_state_int64_vector[IDX_ROOM_STATE_FIFO_SCAN_IDX] = scan_idx

                                # PASS 1: exact matches
                                elif pallet_grade == minimum_grade_requested:
                                    num_boxes_taken_from_pallet = 0
                                    num_boxes_available = pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AVAILABLE_BOX_COUNT]
                                    for box_idx in range(num_boxes_available - 1, -1, -1):
                                        # Check if box is gone or marked for dispatch
                                        if box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TARGET] == 0.0:
                                            continue
                                        net_weight_in_box = box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_ACTUAL]
                                        # Mark box spaces for dispatch by zeroing target values
                                        box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TARGET] = 0.0
                                        box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_BOX_PACKAGE_MASS, IDX_BOX_LEDGER_TARGET] = 0.0
                                        pulp_mass_total += net_weight_in_box
                                        num_boxes_taken_from_pallet += 1
                                        total_boxes_withdrawn += 1
                                        if pulp_mass_total >= requested_mass_out:
                                            break

                                    if num_boxes_taken_from_pallet >= num_boxes_available:
                                        # pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH] = 1
                                        outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IN_FULL_DISPATCH] = 1
                                        n_active -= 1
                                    else:
                                        outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IN_FULL_DISPATCH] = 0
                                        if not unemptied_pallet_in_sequence:
                                            unemptied_pallet_in_sequence = True
                                            room_state_int64_vector[IDX_ROOM_STATE_FIFO_SCAN_IDX] = scan_idx

                                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_TARGET_ROOM_SLOT] = pallet_slot
                                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_BOXES_TO_EXTRACT] = num_boxes_taken_from_pallet
                                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_PARENT_ORDER_ID] = next_order_id
                                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IS_EVENT_SCHEDULED] = 1
                                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_ACTIVE_DRIVER_CODE] = active_driver_code
                                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_ASSIGNED_DOOR_CODE] = assigned_door_code
                                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_BYPASS_TEMPERATURE_CHECK] = bypass_temperature_checks
                                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_ERROR_THRESHOLD_SCALED] = int(rng.random() * 1_000_000)

                                    box_start_idx = num_boxes_available - 1
                                    box_end_idx = num_boxes_available - num_boxes_taken_from_pallet
                                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_BOX_START_IDX] = box_start_idx
                                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_BOX_END_IDX] = box_end_idx

                                    if is_first_pallet:
                                        outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IS_FIRST_IN_MACRO_ORDER] = 1
                                        outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_MACRO_ORDER_START_MS] = epoch_timestamp_ms +  macro_order_delay_ms
                                        is_first_pallet = False
                                    else:
                                        outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IS_FIRST_IN_MACRO_ORDER] = 0

                                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IS_LAST_IN_MACRO_ORDER] = 0

                                    pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AVAILABLE_BOX_COUNT] -= num_boxes_taken_from_pallet
                                   
                                    target_pallet_count += 1
                                    total_dispatches_planned += 1
                                    dispatch_wave_write_ptr += 1
                                pallet_count += 1
                            scan_idx += 1

                # PASS 2: Better-grade fallback sweep
                for i in range(pallet_count):
                    if total_dispatches_planned < MAX_QUEUE_SIZE:
                        if pulp_mass_total < requested_mass_out:
                            scan_idx = available_pallet_indices[i]
                            pallet_slot = scan_idx % MAX_ACTIVE_PALLETS
                            pallet_grade = pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_QUALITY_GRADE]
                            if pallet_grade < minimum_grade_requested:
                                q_write_slot = dispatch_wave_write_ptr % MAX_QUEUE_SIZE
                                num_boxes_taken_from_pallet = 0
                                num_boxes_available = pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AVAILABLE_BOX_COUNT]
                                for box_idx in range(num_boxes_available - 1, -1, -1):
                                    # Check if box is gone or marked for dispatch
                                    if box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TARGET] == 0.0:
                                        continue
                                    net_weight_in_box = box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_ACTUAL]
                                    # Mark box spaces for dispatch by zeroing target values
                                    box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TARGET] = 0.0
                                    box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_BOX_PACKAGE_MASS, IDX_BOX_LEDGER_TARGET] = 0.0
                                    pulp_mass_total += net_weight_in_box
                                    num_boxes_taken_from_pallet += 1
                                    total_boxes_withdrawn += 1
                                    if pulp_mass_total >= requested_mass_out:
                                        break

                                if num_boxes_taken_from_pallet >= num_boxes_available:
                                    # pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH] = 1
                                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IN_FULL_DISPATCH] = 1
                                    n_active -= 1
                                else:
                                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IN_FULL_DISPATCH] = 0
                                    if not unemptied_pallet_in_sequence:
                                        unemptied_pallet_in_sequence = True
                                        room_state_int64_vector[IDX_ROOM_STATE_FIFO_SCAN_IDX] = scan_idx

                                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_TARGET_ROOM_SLOT] = pallet_slot
                                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_BOXES_TO_EXTRACT] = num_boxes_taken_from_pallet
                                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_PARENT_ORDER_ID] = next_order_id
                                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IS_EVENT_SCHEDULED] = 1
                                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_ACTIVE_DRIVER_CODE] = active_driver_code
                                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_ASSIGNED_DOOR_CODE] = assigned_door_code
                                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_BYPASS_TEMPERATURE_CHECK] = bypass_temperature_checks
                                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_ERROR_THRESHOLD_SCALED] = int(rng.random() * 1_000_000)

                                box_start_idx = num_boxes_available - 1
                                box_end_idx = num_boxes_available - num_boxes_taken_from_pallet
                                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_BOX_START_IDX] = box_start_idx
                                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_BOX_END_IDX] = box_end_idx

                                if is_first_pallet:
                                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IS_FIRST_IN_MACRO_ORDER] = 1
                                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_MACRO_ORDER_START_MS] = epoch_timestamp_ms + macro_order_delay_ms
                                    is_first_pallet = False
                                else:
                                    outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IS_FIRST_IN_MACRO_ORDER] = 0

                                outbound_wave_queue_matrix[q_write_slot, IDX_WAVE_IS_LAST_IN_MACRO_ORDER] = 0

                                pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AVAILABLE_BOX_COUNT] -= num_boxes_taken_from_pallet
                               
                                target_pallet_count += 1
                                total_dispatches_planned += 1
                                dispatch_wave_write_ptr += 1

                if not unemptied_pallet_in_sequence:
                    room_state_int64_vector[IDX_ROOM_STATE_FIFO_SCAN_IDX] = scan_idx

                if total_boxes_withdrawn > 0:
                    outbound_wave_queue_matrix[(dispatch_wave_write_ptr - 1) % MAX_QUEUE_SIZE, IDX_WAVE_IS_LAST_IN_MACRO_ORDER] = 1

                    outbound_order_row_idx = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_OUTBOUND_ORDER_ROW_IDX]
                    outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_ORDER_ID] = next_order_id # id
                    outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_TIMESTAMP_MS] = epoch_timestamp_ms # timestamp
                    outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_TARGET_FRUIT_MASS] = requested_mass_out # requested mass out
                    outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_TOTAL_BOX_COUNT] = total_boxes_withdrawn
                    outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_TOTAL_PALLET_COUNT] = target_pallet_count
                    outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_MINIMUM_GRADE] = minimum_grade_requested
                    outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_CROP_TYPE] = active_crop_params_int64_vector[IDX_FRUIT_PARAM_FRUIT_TYPE_CODE]
                    outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_ASSIGNED_DOOR_CODE] = assigned_door_code
                    outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_ASSIGNED_DRIVER] = active_driver_code
                    outbound_orders_ledger[outbound_order_row_idx, IDX_ORDER_LEDGER_DISPATCH_TYPE] = dispatch_type

                    next_order_id += 1
                    outbound_order_row_idx += 1

                    # Update state arrays
                    room_state_int64_vector[IDX_ROOM_STATE_NUM_ACTIVE_BATCHES] = n_active
                    room_state_int64_vector[IDX_ROOM_STATE_NEXT_ORDER_ID] = next_order_id
                    room_state_int64_vector[IDX_ROOM_STATE_TOTAL_DISPATCHES_PLANNED] = total_dispatches_planned
                    room_state_int64_vector[IDX_ROOM_STATE_DISPATCH_WAVE_WRITE_PTR] = dispatch_wave_write_ptr
                    room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_OUTBOUND_ORDER_ROW_IDX] = outbound_order_row_idx


@njit
def arrival_timing_decision(
    rng,
    room_state_int64_vector,
    room_state_float64_vector,
    door_states_int64_matrix,
    forcing_cache_int64_vector,
    room_configurations_float64_vector,
    room_configurations_int64_vector,
    active_crop_params_float64_vector,
    door_specs_int64_matrix,
    employee_parameters_matrix,
    supervisor_parameters_matrix, # TODO: ADD to initialization script
    room_ephemeral_int64_vector,
    door_outcome_flags_matrix,
    inbound_pallet_history_ledger,

    pallet_metadata_int64_matrix,
    pallet_physics_float64_matrix,
    pallet_timeline_int64_matrix,
    inbound_wave_queue_matrix,
    box_masses_ledger,
    box_flags_ledger,

    airflow_pallets_compact_vector,
    inverse_airflow_pallets_lookup_vector,
    crossing_blobs_compact_vector,
    inverse_crossing_blobs_lookup_vector,

    dt_engine_ms,
):
    if not room_state_int64_vector[IDX_ROOM_STATE_RAMP_UP_SCHEDULED]: # arrival_wave_write_ptr
        epoch_timestamp_ms = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]
        q_read_slot = room_state_int64_vector[IDX_ROOM_STATE_ARRIVAL_WAVE_READ_PTR] % MAX_ACTIVE_PALLETS
        if inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IS_EVENT_SCHEDULED]:
            active_room_slot = inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_TARGET_ROOM_SLOT]
            assigned_door_code = inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_ASSIGNED_DOOR_CODE]
            active_driver_code = inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_ACTIVE_DRIVER_CODE]
            tau_door_ms = door_specs_int64_matrix[assigned_door_code, IDX_FACILITY_TAU_DOOR_MS]

            if not pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IN_ARRIVAL_RUN]:
                local_utc_offset_ms = forcing_cache_int64_vector[IDX_CACHE_LOCAL_UTC_MS]
                if inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IS_FIRST_IN_MACRO_ORDER]:
                    if door_states_int64_matrix[assigned_door_code, IDX_DOOR_STATE_IS_BUSY]:
                        is_door_available = False
                    else:
                        is_door_available = True
                    is_operator_ready = epoch_timestamp_ms >= inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_MACRO_ORDER_START_MS]
                else:
                    is_door_available = True
                    is_operator_ready = True

                if is_door_available and is_operator_ready:
                    # Calculate probabilty of error
                    _, _, _, hour, day_of_week, _, is_last_weekday = compute_local_calendar_components(epoch_timestamp_ms, local_utc_offset_ms)
                    f_fatigue = calculate_fatigue_factor(
                        hour,
                        day_of_week,
                        is_last_weekday,
                    )

                    p_gate_override = compute_probability_of_error(
                        base_carelessness   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CARELESSNESS],
                        baseline_p_error    = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_P_BASE_GATE_OVERRIDE],
                        fatigue_factor      = f_fatigue,
                        fatigue_sensitivity = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_FATIGUE_SENSITIVITY],
                        error_sensitivity   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_ERROR_SENSITIVITY],
                    ) # * 0.0 #### DEBUG ####
                    wave_error_threshold = inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_ERROR_THRESHOLD_SCALED] * 1e-6

                    # Make decision
                    if (
                        room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_FILTERED_VAL] < active_crop_params_float64_vector[IDX_FRUIT_PARAM_DOOR_LOCK_THRESHOLD_TEMP_C]
                        or wave_error_threshold < p_gate_override
                    ):
                        t_anchor_ms = epoch_timestamp_ms # anchor timestamp
                        # enter_sequence_door_was_shut = False

                        if inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IS_FIRST_IN_MACRO_ORDER]:
                            door_states_int64_matrix[assigned_door_code, IDX_DOOR_STATE_IS_BUSY] = 1
                            # delay_mean = door_specs_matrix[assigned_door_code, IDX_DOOR_PARAM_STAGING_MEAN_MS]
                            # delay_shape = door_specs_matrix[assigned_door_code, IDX_DOOR_PARAM_STAGING_SHAPE]
                            # first_pallet_delay = gamma_draw(rng, mean=delay_mean, shape=delay_shape)
                            # t_anchor_ms += round_up_to_multiple(first_pallet_delay, dt_engine_ms)
                        
                        pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IN_ARRIVAL_RUN] = 1
                        pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_QUEUE_READ_SLOT] = q_read_slot

                        pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_ACTIVE_DRIVER_CONVECTIVE_Q] = bounded_normal_draw(
                            rng   = rng,
                            mean  = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q_MEAN],
                            scale = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q_SCALE],
                            low   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q_LOW],
                            high  = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q_HIGH],
                        ) 

                        slack_low_bound_ms       = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_SLACK_LOW_BOUND_MS]
                        slack_high_bound_ms      = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_SLACK_HIGH_BOUND_MS]
                        door_transit_baseline_ms = door_specs_int64_matrix[assigned_door_code, IDX_FACILITY_DOOR_TRANSIT_BASELINE_MS]
                        staging_delay_mean_ms    = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_STAGING_DELAY_MEAN_MS]
                        staging_delay_shape      = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_STAGING_DELAY_SHAPE]  
                        outbound_leg_mean_ms     = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_OUTBOUND_LEG_MEAN_MS] 
                        outbound_leg_shape       = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_OUTBOUND_LEG_SHAPE]   
                        return_leg_mean_ms       = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_RETURN_LEG_MEAN_MS]   
                        return_leg_shape         = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_RETURN_LEG_SHAPE]

                        superv_slack_low_bound_ms  = supervisor_parameters_matrix[day_of_week, assigned_door_code, IDX_EMP_PARAM_SLACK_LOW_BOUND_MS]
                        superv_slack_high_bound_ms = supervisor_parameters_matrix[day_of_week, assigned_door_code, IDX_EMP_PARAM_SLACK_HIGH_BOUND_MS]

                        # Log arrival
                        pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ARRIVAL_LOGGING_MS] = t_anchor_ms

                        # 1. Staging delay
                        staging_delay_ms = gamma_draw(rng, staging_delay_mean_ms, staging_delay_shape)
                        t_anchor_ms += round_up_to_multiple(staging_delay_ms, dt_engine_ms)

                        # Pulse 1: Entrance
                        pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_OPEN_MS] = t_anchor_ms # door starts opening

                        operator_random_term = rng.uniform(slack_low_bound_ms, slack_high_bound_ms)
                        t_anchor_ms += tau_door_ms + round_up_to_multiple(operator_random_term, dt_engine_ms)

                        # Pallet enters room
                        pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_BOUNDARY_CROSS_START_MS] = t_anchor_ms
                        operator_random_term = rng.uniform(slack_low_bound_ms, slack_high_bound_ms)
                        t_anchor_ms += round_up_to_multiple(door_transit_baseline_ms + operator_random_term, dt_engine_ms)
                        pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_BOUNDARY_CROSS_END_MS] = t_anchor_ms

                        operator_random_term = rng.uniform(slack_low_bound_ms, slack_high_bound_ms)
                        standard_close_time_ms = t_anchor_ms + round_up_to_multiple(operator_random_term, dt_engine_ms)

                        decision_t_anchor_ms = t_anchor_ms
                        door_outcome_flags_matrix[active_room_slot, :] = 0

                        # PENDING
                        _, _, _, hour, day_of_week, _, is_last_weekday = compute_local_calendar_components(standard_close_time_ms, local_utc_offset_ms)
                        f_fatigue = calculate_fatigue_factor(
                            hour,
                            day_of_week,
                            is_last_weekday,
                        )

                        p_forget_door = compute_probability_of_error(
                            base_carelessness   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CARELESSNESS],
                            baseline_p_error    = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_P_BASE_FORGET_DOOR],
                            fatigue_factor      = f_fatigue,
                            fatigue_sensitivity = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_FATIGUE_SENSITIVITY],
                            error_sensitivity   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_ERROR_SENSITIVITY],
                        ) # * 0.0 ### DEBUG ###

                        p_employee_self_catch_miss = compute_probability_of_error(
                            base_carelessness   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CARELESSNESS],
                            baseline_p_error    = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_P_BASE_FORGET_DOOR_REMEDY],  # new, independent constant
                            fatigue_factor      = f_fatigue,
                            fatigue_sensitivity = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_FATIGUE_SENSITIVITY],
                            error_sensitivity   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_ERROR_SENSITIVITY],
                        )
                        
                        draw_forget = rng.random()
                        draw_remembers = rng.random()
                        draw_supervisor_notices = rng.random()

                        # Cast the die for employee forgetting to shut the door
                        if draw_forget < p_forget_door:
                            p_supervisor_miss_intervention = compute_probability_of_error(
                                base_carelessness   = supervisor_parameters_matrix[day_of_week, assigned_door_code, IDX_EMP_PARAM_CARELESSNESS], # Supervisors should more attentive than subordinates
                                baseline_p_error    = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_P_BASE_SUPERVISOR_FORGET_DOOR],
                                fatigue_factor      = f_fatigue, # Re-use fatigue factor since it's at the hour grain
                                fatigue_sensitivity = supervisor_parameters_matrix[day_of_week, assigned_door_code, IDX_EMP_PARAM_FATIGUE_SENSITIVITY],
                                error_sensitivity   = supervisor_parameters_matrix[day_of_week, assigned_door_code, IDX_EMP_PARAM_ERROR_SENSITIVITY],
                            )
                            # Give an opportunity to employees: cast die again to see if they will remember to close the door before they reach the pallet space
                            employee_remembers = draw_remembers >= p_employee_self_catch_miss
                            supervisor_notices = draw_supervisor_notices >= p_supervisor_miss_intervention
                            if employee_remembers:
                                # Label
                                door_outcome_flags_matrix[active_room_slot, IDX_DOOR_OUTCOME_DRIVER_NOTICED] = 1
                            if supervisor_notices:
                                door_outcome_flags_matrix[active_room_slot, IDX_DOOR_OUTCOME_SUPERVISOR_NOTICED] = 1
                            if not employee_remembers and not supervisor_notices:
                                door_outcome_flags_matrix[active_room_slot, IDX_DOOR_OUTCOME_UNNOTICED] = 1
                        else:
                            # Business as usual
                            pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = standard_close_time_ms
                            t_anchor_ms = standard_close_time_ms
                            # enter_sequence_door_was_shut = True
                            
                        # if rng.random() < p_forget_door:
                        #     pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = SENTINEL_INT64
                        # else:
                        #     pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = standard_close_time_ms
                        #     t_anchor_ms = standard_close_time_ms

                        # Outbound leg
                        carelessness_factor = calculate_actual_carelessness_factor(
                            base_carelessness   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CARELESSNESS],
                            fatigue_factor      = f_fatigue,
                            fatigue_sensitivity = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_FATIGUE_SENSITIVITY],
                        )
                        # outbund_leg_ms = gamma_draw(rng, outbound_leg_mean_ms, outbound_leg_shape) * carelessness_factor
                        # Outbound leg
                        outbund_leg_ms = gamma_draw(rng, outbound_leg_mean_ms, outbound_leg_shape) * carelessness_factor
                        outbund_leg_ms = round_up_to_multiple(outbund_leg_ms, dt_engine_ms)

                        fast_catch_resolved = False
                        if door_outcome_flags_matrix[active_room_slot, IDX_DOOR_OUTCOME_DRIVER_NOTICED]:
                            # Add a fraction of outbund_leg_ms
                            fraction = rng.uniform(0.3, 0.6) # NOTE: Draw a reasonable fraction
                            driver_notice_time_ms = round_up_to_multiple(fraction * outbund_leg_ms, dt_engine_ms)
                            if driver_notice_time_ms < door_specs_int64_matrix[assigned_door_code, IDX_FACILITY_DOOR_ALARM_TRIGGER_MS]:
                                back_and_forth_trip = round_up_to_multiple(2 * fraction * outbund_leg_ms, dt_engine_ms)
                                pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = t_anchor_ms + back_and_forth_trip
                                outbund_leg_ms += round_up_to_multiple(back_and_forth_trip, dt_engine_ms)
                                fast_catch_resolved = True

                        t_anchor_ms += outbund_leg_ms

                        # Place pallet with boxes
                        placement_delay_ms = gamma_draw(
                            rng,
                            active_crop_params_float64_vector[IDX_FRUIT_PARAM_PALLET_RETRIEVAL_DELAY_MEAN_MS],
                            active_crop_params_float64_vector[IDX_FRUIT_PARAM_PALLET_RETRIEVAL_DELAY_SHAPE],
                        )
                        t_anchor_ms += round_up_to_multiple(placement_delay_ms, dt_engine_ms)

                        pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ARRIVED_IN_PLACE_MS] = t_anchor_ms

                        _, _, _, hour, day_of_week, _, is_last_weekday = compute_local_calendar_components(t_anchor_ms, local_utc_offset_ms)
                    
                        f_fatigue = calculate_fatigue_factor(
                                hour,
                                day_of_week,
                                is_last_weekday,
                        )
                        carelessness_factor = calculate_actual_carelessness_factor(
                            base_carelessness   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CARELESSNESS],
                            fatigue_factor      = f_fatigue,
                            fatigue_sensitivity = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_FATIGUE_SENSITIVITY],
                        )
                    
                        # Return leg
                        return_leg_ms = gamma_draw(rng, return_leg_mean_ms, return_leg_shape)
                        t_anchor_ms += round_up_to_multiple(return_leg_ms*carelessness_factor, dt_engine_ms)

                        driver_opens_door_for_exit_ts_ms = t_anchor_ms # + round_up_to_multiple(rng.uniform(slack_low_bound_ms, slack_high_bound_ms), dt_engine_ms)

                        if door_outcome_flags_matrix[active_room_slot, IDX_DOOR_OUTCOME_UNNOTICED]:
                            pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = t_anchor_ms
                        elif fast_catch_resolved:
                            pass
                        elif door_outcome_flags_matrix[active_room_slot, IDX_DOOR_OUTCOME_SUPERVISOR_NOTICED]:
                            # Calculate at which timestamp would supervisor shut the door
                            superv_slack_ms = rng.uniform(superv_slack_low_bound_ms, superv_slack_high_bound_ms)
                            supervisor_shuts_door_ts_ms = decision_t_anchor_ms + round_up_to_multiple(door_specs_int64_matrix[assigned_door_code, IDX_FACILITY_DOOR_ALARM_TRIGGER_MS] + superv_slack_ms, dt_engine_ms)
                            if (driver_opens_door_for_exit_ts_ms - supervisor_shuts_door_ts_ms) > 20_000:
                                # Supervisor shuts door
                                pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = supervisor_shuts_door_ts_ms
                                pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_SUPERVISOR_INTERVENED] = 1
                            else:
                                # Supervisor lets driver keep the door open
                                pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = t_anchor_ms
                        elif door_outcome_flags_matrix[active_room_slot, IDX_DOOR_OUTCOME_DRIVER_NOTICED]:
                            pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = t_anchor_ms


                        # `if not enter_sequence_door_was_shut:
                        # `    if door_outcome_flags_matrix[active_room_slot, IDX_DOOR_OUTCOME_SUPERVISOR_NOTICED]:
                        # `        # Calculate at which timestamp would supervisor shut the door
                        # `        superv_slack_ms = rng.uniform(superv_slack_low_bound_ms, superv_slack_high_bound_ms)
                        # `        supervisor_shuts_door_ts_ms = decision_t_anchor_ms + round_up_to_multiple(door_specs_matrix[assigned_door_code, IDX_FACILITY_DOOR_ALARM_TRIGGER_MS] + superv_slack_ms, dt_engine_ms)
                        # `        # supervisor_shuts_door_ts_ms = decision_t_anchor_ms + room_configurations_float64_vector[IDX_FACILITY_DOOR_ALARM_TRIGGER_MS] + rng.uniform(superv_slack_low_bound_ms, superv_slack_high_bound_ms)
                        # `        if (driver_opens_door_for_exit_ts_ms - supervisor_shuts_door_ts_ms) > 30_000: # 30 sec buffer so they won't overlap. tau_door_ms is less than 5 sec
                        # `            # Supervisor shuts door
                        # `            pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = supervisor_shuts_door_ts_ms
                        # `        else:
                        # `            # Supervisor lets driver keep the door open
                        # `            pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = t_anchor_ms
                        # `    else:
                        # `        # Door goes unnoticed so it's kept open
                        # `        pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = t_anchor_ms

                        # Door kept open means IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS = IDX_TIMELINE_PALLET_EXIT_DOOR_OPEN_MS
                        pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_DOOR_OPEN_MS] = t_anchor_ms

                        # # Exit sequence
                        # pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_DOOR_OPEN_MS] = t_anchor_ms
                        # if pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] == SENTINEL_INT64:
                        #     pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = t_anchor_ms

                        operator_random_term = rng.uniform(slack_low_bound_ms, slack_high_bound_ms)
                        t_anchor_ms += tau_door_ms + round_up_to_multiple(operator_random_term, dt_engine_ms)

                        pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_BOUNDARY_CROSS_START_MS] = t_anchor_ms
                        operator_random_term = rng.uniform(slack_low_bound_ms, slack_high_bound_ms)
                        t_anchor_ms += round_up_to_multiple(door_transit_baseline_ms + operator_random_term, dt_engine_ms)
                        pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_BOUNDARY_CROSS_END_MS] = t_anchor_ms

                        operator_random_term = rng.uniform(slack_low_bound_ms, slack_high_bound_ms)
                        t_anchor_ms += round_up_to_multiple(operator_random_term, dt_engine_ms)
                        pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_DOOR_CLOSE_MS] = t_anchor_ms

                        buffer_random_term = rng.uniform(slack_low_bound_ms, slack_high_bound_ms)
                        t_anchor_ms += tau_door_ms + round_up_to_multiple(buffer_random_term, dt_engine_ms)

                        pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_SEQUENCE_COMPLETE_MS] = t_anchor_ms

            if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ARRIVAL_LOGGING_MS]:
                num_boxes_total = inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_BOXES_TO_ARRIVE]
                net_mass_total = 0.0

                for box_idx in range(num_boxes_total):
                    net_mass_per_box = box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TARGET]
                    if net_mass_per_box == SENTINEL_FLOAT64:
                        break
                    net_mass_total += net_mass_per_box

                gross_mass_total = net_mass_total + num_boxes_total * active_crop_params_float64_vector[IDX_FRUIT_PARAM_BOX_PACKAGING_MASS]

                incoming_fruit_temperature = draw_tunnel_exit_fruit_temp(
                    rng          = rng,
                    temp_ref     = active_crop_params_float64_vector[IDX_FRUIT_PARAM_TUNNEL_EXIT_TEMP_REF],
                    temp_drift   = active_crop_params_float64_vector[IDX_FRUIT_PARAM_TUNNEL_EXIT_TEMP_DRIFT],
                    temp_scale   = active_crop_params_float64_vector[IDX_FRUIT_PARAM_TUNNEL_EXIT_TEMP_SCALE],
                    max_decrease = active_crop_params_float64_vector[IDX_FRUIT_PARAM_TUNNEL_MAX_DECREASE],
                    max_increase = active_crop_params_float64_vector[IDX_FRUIT_PARAM_TUNNEL_MAX_INCREASE]
                )

                pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_PULP_TEMPERATURE] = incoming_fruit_temperature
                 #  = incoming_fruit_temperature
                pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_AGE_MS] = 0

                # micro pallet history ledgers
                inbound_history_row_idx = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_INBOUND_HISTORY_ROW_IDX]
                inbound_pallet_history_ledger[inbound_history_row_idx, IDX_HISTORICAL_PALLET_BATCH_ID] = pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_BATCH_ID]
                inbound_pallet_history_ledger[inbound_history_row_idx, IDX_HISTORICAL_PALLET_PARENT_ORDER_ID] = inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_PARENT_ORDER_ID]
                inbound_pallet_history_ledger[inbound_history_row_idx, IDX_HISTORICAL_PALLET_TIMESTAMP_MS] = epoch_timestamp_ms
                inbound_pallet_history_ledger[inbound_history_row_idx, IDX_HISTORICAL_PALLET_QUALITY_GRADE] = pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_QUALITY_GRADE]
                inbound_pallet_history_ledger[inbound_history_row_idx, IDX_HISTORICAL_PALLET_PULP_TEMP] = incoming_fruit_temperature
                # inbound_pallet_history_ledger[inbound_history_row_idx, IDX_HISTORICAL_PALLET_PULP_TEMP_FILTERED_VAL] = incoming_fruit_temperature
                inbound_pallet_history_ledger[inbound_history_row_idx, IDX_HISTORICAL_PALLET_GROSS_START] = gross_mass_total
                inbound_pallet_history_ledger[inbound_history_row_idx, IDX_HISTORICAL_PALLET_BOX_START] = num_boxes_total
                # inbound_pallet_history_ledger[inbound_history_row_idx, IDX_HISTORICAL_PALLET_GROSS_MOVED] = 0.0
                # inbound_pallet_history_ledger[inbound_history_row_idx, IDX_HISTORICAL_PALLET_BOX_MOVED] = 0
                # inbound_pallet_history_ledger[inbound_history_row_idx, IDX_HISTORICAL_PALLET_GROSS_START_FILTERED_VAL] = gross_mass_total

                if pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_SUPERVISOR_INTERVENED]:
                    inbound_pallet_history_ledger[inbound_history_row_idx, IDX_HISTORICAL_PALLET_SUPERVISOR_INTERVENTION_MS] = pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS]
                    pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_SUPERVISOR_INTERVENED] = 0
                else:
                    inbound_pallet_history_ledger[inbound_history_row_idx, IDX_HISTORICAL_PALLET_SUPERVISOR_INTERVENTION_MS] = MAX_EPOCH_TIMESTAMP_MS


                room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_INBOUND_HISTORY_ROW_IDX] += 1

    # pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_ACTIVE_DRIVER_CONVECTIVE_Q] = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q]
            if pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_BOUNDARY_CROSS_END_MS] <= epoch_timestamp_ms < pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_BOUNDARY_CROSS_START_MS]:
                if epoch_timestamp_ms % room_configurations_int64_vector[IDX_ROOM_LOGISTICS_DRIVER_Q_SAMPLING_INTERVAL_MS] == 0:
                    current_active_driver_convective_q = pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_ACTIVE_DRIVER_CONVECTIVE_Q]
                    updated_active_driver_convective_q = bounded_normal_draw(
                        rng = rng,
                        mean  = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q_MEAN],
                        scale = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q_SCALE],
                        low   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q_LOW],
                        high  = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q_HIGH],
                    )
                    pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_ACTIVE_DRIVER_CONVECTIVE_Q] = updated_active_driver_convective_q
                    room_state_float64_vector[IDX_ROOM_STATE_BULK_ACTIVE_DRIVERS_CONVECTIVE_Q] += (updated_active_driver_convective_q - current_active_driver_convective_q) 

            if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_OPEN_MS]:
                door_states_int64_matrix[assigned_door_code, IDX_DOOR_STATE_OPEN_MS] = pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_OPEN_MS]
                door_states_int64_matrix[assigned_door_code, IDX_DOOR_STATE_CLOSE_MS] = pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS]

            if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_DOOR_OPEN_MS]:
                door_states_int64_matrix[assigned_door_code, IDX_DOOR_STATE_OPEN_MS] = pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_DOOR_OPEN_MS]
                door_states_int64_matrix[assigned_door_code, IDX_DOOR_STATE_CLOSE_MS] = pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_DOOR_CLOSE_MS]

            # if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS]:
            #     supervisor_intervened = pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_SUPERVISOR_INTERVENED]
            #     inbound_history_row_idx = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_INBOUND_HISTORY_ROW_IDX]
            #     if supervisor_intervened:
            #         inbound_pallet_history_ledger[inbound_history_row_idx, IDX_HISTORICAL_PALLET_SUPERVISOR_SHUT_DOOR] = supervisor_intervened
            #     pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_SUPERVISOR_INTERVENED] = 0

            if (
                epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_BOUNDARY_CROSS_START_MS]
                or epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_BOUNDARY_CROSS_START_MS]
            ):
                updated_blobs_counter = add_id_element(
                    crossing_blobs_compact_vector,
                    inverse_crossing_blobs_lookup_vector,
                    room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BLOBS_CROSSING],
                    active_room_slot
                )
                room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BLOBS_CROSSING] = updated_blobs_counter

            if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_BOUNDARY_CROSS_START_MS]:
                pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IS_IN_AIRFLOW] = 1
                pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IS_CROSSING_BOUNDARY] = 1
                num_boxes_to_arrive = inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_BOXES_TO_ARRIVE]
                pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_PULP_MASS] = 0.0
                pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_BOX_PACKAGING_MASS] = 0.0
                pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_PALLET_MASS] = 0.0

                # Write in the actual number of boxes per pallet in the room
                pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_ACTUAL_BOX_COUNT] = num_boxes_to_arrive

                pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_BOX_SLOTS_COUNT] = num_boxes_to_arrive

                # Sample k_zone to randomly assign space in the room (with a certain degree of exposure to the air)
                pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_K_ZONE] = rng.uniform(
                    active_crop_params_float64_vector[IDX_FRUIT_PARAM_K_ZONE_LOW], 
                    active_crop_params_float64_vector[IDX_FRUIT_PARAM_K_ZONE_HIGH],
                )

                for box_idx in range(num_boxes_to_arrive):
                    box_flags_ledger[box_idx, active_room_slot, IDX_BOX_IS_IN_PALLET] = 1
                    box_flags_ledger[box_idx, active_room_slot, IDX_BOX_IS_IN_TRANSIT] = 1

                updated_airflow_pallets_counter = add_id_element(
                    airflow_pallets_compact_vector,
                    inverse_airflow_pallets_lookup_vector,
                    room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_AIRFLOW],
                    active_room_slot,
                )
                room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_AIRFLOW] = updated_airflow_pallets_counter

            # if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_BOUNDARY_CROSS_START_MS]:

            if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ARRIVED_IN_PLACE_MS]:
                pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IS_PALLET_IN_PLACE] = 1
                pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_PULP_TEMP_FILTERED_VAL] = pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_PULP_TEMPERATURE]
                pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_MASS_FILTERED_VAL] = pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_PULP_MASS] + pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_BOX_PACKAGING_MASS]
                for box_idx in range(inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_BOXES_TO_ARRIVE]): # NOTE: FIXED, SWAPPED FROM num_boxes_to_arrive
                    box_flags_ledger[box_idx, active_room_slot, IDX_BOX_IS_IN_TRANSIT] = 0

            if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_SEQUENCE_COMPLETE_MS]:
                inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IS_EVENT_SCHEDULED] = 0
                pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IN_ARRIVAL_RUN] = 0
                pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IS_IN_INVENTORY] = 1

                if inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IS_LAST_IN_MACRO_ORDER]:
                    door_states_int64_matrix[assigned_door_code, IDX_DOOR_STATE_IS_BUSY] = 0

                pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_AVAILABLE_BOX_COUNT] = inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_BOXES_TO_ARRIVE]

                room_state_int64_vector[IDX_ROOM_STATE_ARRIVAL_WAVE_READ_PTR] += 1
                room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_ROOM] += 1
                room_state_int64_vector[IDX_ROOM_STATE_TOTAL_ARRIVALS_PLANNED] -= 1


@njit
def dispatch_timing_decision(
    step, # DEBUG
    rng,
    room_state_int64_vector,
    room_state_float64_vector,
    door_states_int64_matrix,
    forcing_cache_int64_vector,
    room_configurations_float64_vector,
    room_configurations_int64_vector,
    active_crop_params_float64_vector,
    door_specs_int64_matrix,
    employee_parameters_matrix,
    supervisor_parameters_matrix, # TODO: ADD to initialization script
    room_ephemeral_int64_vector,
    door_outcome_flags_matrix,
    outbound_pallet_history_ledger,
    partial_dispatch_box_temperatures_ledger,
    box_removal_timestamps,

    pallet_metadata_int64_matrix,
    pallet_physics_float64_matrix,
    pallet_timeline_int64_matrix,
    outbound_wave_queue_matrix,
    box_masses_ledger,
    box_flags_ledger,

    crossing_blobs_compact_vector,
    inverse_crossing_blobs_lookup_vector,

    dt_engine_ms,
):
    q_read_slot = room_state_int64_vector[IDX_ROOM_STATE_DISPATCH_WAVE_READ_PTR] % MAX_QUEUE_SIZE
    if outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IS_EVENT_SCHEDULED]:
        epoch_timestamp_ms = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]
        active_room_slot = outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_TARGET_ROOM_SLOT]
        # outbound_history_row_idx = room_state_int64_vector[IDX_ROOM_STATE_OUTBOUND_HISTORY_ROW_IDX]

        assigned_door_code = outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_ASSIGNED_DOOR_CODE]
        active_driver_code = outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_ACTIVE_DRIVER_CODE]
        num_boxes_requested = outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_BOXES_TO_EXTRACT]
        tau_door_ms = door_specs_int64_matrix[assigned_door_code, IDX_FACILITY_TAU_DOOR_MS]

        if not pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IN_DISPATCH_RUN]:
            # carelessness_factor = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CARELESSNESS]
            # p_gate_override = compute_probability_of_error(carelessness_factor, baseline_p=0.05, sensitivity=1.2) if not bypass_temperature_checks else 1.0

            local_utc_offset_ms = forcing_cache_int64_vector[IDX_CACHE_LOCAL_UTC_MS] 

            if outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IS_FIRST_IN_MACRO_ORDER]:
                if door_states_int64_matrix[assigned_door_code, IDX_DOOR_STATE_IS_BUSY]:
                    is_door_available = False
                else:
                    is_door_available = True
                is_operator_ready = epoch_timestamp_ms >= outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_MACRO_ORDER_START_MS]
            else:
                is_door_available = True
                is_operator_ready = True

            if is_door_available and is_operator_ready:
                # Calculate probabilty of error
                _, _, _, hour, day_of_week, _, is_last_weekday = compute_local_calendar_components(epoch_timestamp_ms, local_utc_offset_ms)
                f_fatigue = calculate_fatigue_factor(
                    hour,
                    day_of_week,
                    is_last_weekday,
                )
                p_gate_override = compute_probability_of_error(
                    base_carelessness   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CARELESSNESS],
                    baseline_p_error    = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_P_BASE_GATE_OVERRIDE], # Shared Global Baseline!
                    fatigue_factor      = f_fatigue,
                    fatigue_sensitivity = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_FATIGUE_SENSITIVITY],
                    error_sensitivity   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_ERROR_SENSITIVITY],
                ) # * 0.0 ### DEBUG ###
                wave_error_threshold = outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_ERROR_THRESHOLD_SCALED] * 1e-6

                # Make decision
                if (
                    room_state_float64_vector[IDX_ROOM_STATE_T_ROOM_FILTERED_VAL] < active_crop_params_float64_vector[IDX_FRUIT_PARAM_DOOR_LOCK_THRESHOLD_TEMP_C]
                    or wave_error_threshold < p_gate_override
                ):
                    t_anchor_ms = epoch_timestamp_ms # anchor timestamp
                    # enter_sequence_door_was_shut = False
                    
                    if outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IS_FIRST_IN_MACRO_ORDER]:
                        door_states_int64_matrix[assigned_door_code, IDX_DOOR_STATE_IS_BUSY] = 1
                        # delay_mean = facility_constants_matrix[assigned_door_code, IDX_FACILITY_STAGING_MEAN_MS]
                        # delay_shape = door_specs_matrix[assigned_door_code, IDX_DOOR_PARAM_STAGING_SHAPE]
                        # first_pallet_delay = gamma_draw(rng, mean=delay_mean, shape=delay_shape)
                        # t_anchor_ms += round_up_to_multiple(first_pallet_delay, dt_engine_ms)

                    # box_removal_timestamps[...] = SENTINEL_FLOAT64

                    pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IN_DISPATCH_RUN] = 1
                    pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_QUEUE_READ_SLOT] = q_read_slot

                    # Draw employee terms
                    pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_ACTIVE_DRIVER_CONVECTIVE_Q] = bounded_normal_draw(
                        rng = rng,
                        mean  = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q_MEAN],
                        scale = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q_SCALE],
                        low   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q_LOW],
                        high  = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q_HIGH],
                    )

                    slack_low_bound_ms       = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_SLACK_LOW_BOUND_MS]
                    slack_high_bound_ms      = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_SLACK_HIGH_BOUND_MS]
                    door_transit_baseline_ms = door_specs_int64_matrix[assigned_door_code, IDX_FACILITY_DOOR_TRANSIT_BASELINE_MS]
                    staging_delay_mean_ms    = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_STAGING_DELAY_MEAN_MS]
                    staging_delay_shape      = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_STAGING_DELAY_SHAPE]  
                    outbound_leg_mean_ms     = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_OUTBOUND_LEG_MEAN_MS] 
                    outbound_leg_shape       = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_OUTBOUND_LEG_SHAPE]   
                    return_leg_mean_ms       = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_RETURN_LEG_MEAN_MS]   
                    return_leg_shape         = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_RETURN_LEG_SHAPE]


                    superv_slack_low_bound_ms  = supervisor_parameters_matrix[day_of_week, assigned_door_code, IDX_EMP_PARAM_SLACK_LOW_BOUND_MS]
                    superv_slack_high_bound_ms = supervisor_parameters_matrix[day_of_week, assigned_door_code, IDX_EMP_PARAM_SLACK_HIGH_BOUND_MS]

                    # 1. Staging delay
                    staging_delay_ms = gamma_draw(rng, staging_delay_mean_ms, staging_delay_shape)
                    t_anchor_ms += round_up_to_multiple(staging_delay_ms, dt_engine_ms)

                    # Pulse 1: Entrance
                    pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_OPEN_MS] = t_anchor_ms # door starts opening

                    # Forklift crosses boundary
                    operator_random_term = rng.uniform(slack_low_bound_ms, slack_high_bound_ms)
                    t_anchor_ms += tau_door_ms + round_up_to_multiple(operator_random_term, dt_engine_ms)

                    # Operator starts entering room
                    pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_BOUNDARY_CROSS_START_MS] = t_anchor_ms

                    operator_random_term = rng.uniform(slack_low_bound_ms, slack_high_bound_ms)
                    t_anchor_ms += round_up_to_multiple(door_transit_baseline_ms + operator_random_term, dt_engine_ms)

                    # Operator finishes crossing boundary
                    pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_BOUNDARY_CROSS_END_MS] = t_anchor_ms

                    operator_random_term = rng.uniform(slack_low_bound_ms, slack_high_bound_ms)
                    # t_anchor_ms += round_up_to_multiple(operator_random_term, dt_engine_ms)
                    standard_close_time_ms = t_anchor_ms + round_up_to_multiple(operator_random_term, dt_engine_ms)

                    decision_t_anchor_ms = t_anchor_ms
                    door_outcome_flags_matrix[active_room_slot, :] = 0

                    _, _, _, hour, day_of_week, _, is_last_weekday = compute_local_calendar_components(standard_close_time_ms, local_utc_offset_ms)
                    f_fatigue = calculate_fatigue_factor(
                        hour,
                        day_of_week,
                        is_last_weekday,
                    )
                    p_forget_door = compute_probability_of_error(
                        base_carelessness   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CARELESSNESS],
                        baseline_p_error    = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_P_BASE_FORGET_DOOR],   # Shared Global Baseline!
                        fatigue_factor      = f_fatigue,
                        fatigue_sensitivity = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_FATIGUE_SENSITIVITY],
                        error_sensitivity   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_ERROR_SENSITIVITY],
                    ) # * 0.0 ### DEBUG ###

                    p_employee_self_catch_miss = compute_probability_of_error(
                        base_carelessness   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CARELESSNESS],
                        baseline_p_error    = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_P_BASE_FORGET_DOOR_REMEDY],  # new, independent constant
                        fatigue_factor      = f_fatigue,
                        fatigue_sensitivity = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_FATIGUE_SENSITIVITY],
                        error_sensitivity   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_ERROR_SENSITIVITY],
                    )

                    draw_forget = rng.random()
                    draw_remembers = rng.random()
                    draw_supervisor_notices = rng.random()

                    # Cast the die for employee forgetting to shut the door
                    if draw_forget < p_forget_door:
                        p_supervisor_miss_intervention = compute_probability_of_error(
                            base_carelessness   = supervisor_parameters_matrix[day_of_week, assigned_door_code, IDX_EMP_PARAM_CARELESSNESS], # Supervisors should more attentive than subordinates
                            baseline_p_error    = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_P_BASE_SUPERVISOR_FORGET_DOOR],
                            fatigue_factor      = f_fatigue, # Re-use fatigue factor since it's at the hour grain
                            fatigue_sensitivity = supervisor_parameters_matrix[day_of_week, assigned_door_code, IDX_EMP_PARAM_FATIGUE_SENSITIVITY],
                            error_sensitivity   = supervisor_parameters_matrix[day_of_week, assigned_door_code, IDX_EMP_PARAM_ERROR_SENSITIVITY],
                        )
                        # Give an opportunity to employees: cast die again to see if they will remember to close the door before they reach the pallet space
                        employee_remembers = draw_remembers >= p_employee_self_catch_miss
                        supervisor_notices = draw_supervisor_notices >= p_supervisor_miss_intervention
                        if employee_remembers:
                            # Label
                            door_outcome_flags_matrix[active_room_slot, IDX_DOOR_OUTCOME_DRIVER_NOTICED] = 1 
                        if supervisor_notices:
                            door_outcome_flags_matrix[active_room_slot, IDX_DOOR_OUTCOME_SUPERVISOR_NOTICED] = 1
                        if not employee_remembers and not supervisor_notices:
                            door_outcome_flags_matrix[active_room_slot, IDX_DOOR_OUTCOME_UNNOTICED] = 1

                    else:
                        # Business as usual
                        pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = standard_close_time_ms
                        t_anchor_ms = standard_close_time_ms
                        # enter_sequence_door_was_shut = True

                    # if rng.random() < p_forget_door:
                    #     pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = SENTINEL_INT64
                    # else:
                    #     pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = standard_close_time_ms
                    #     t_anchor_ms = standard_close_time_ms

                    carelessness_factor = calculate_actual_carelessness_factor(
                        base_carelessness   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CARELESSNESS],
                        fatigue_factor      = f_fatigue,
                        fatigue_sensitivity = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_FATIGUE_SENSITIVITY],
                    )
                    # Outbound leg
                    outbund_leg_ms = gamma_draw(rng, outbound_leg_mean_ms, outbound_leg_shape) * carelessness_factor
                    outbund_leg_ms = round_up_to_multiple(outbund_leg_ms, dt_engine_ms)

                    fast_catch_resolved = False
                    if door_outcome_flags_matrix[active_room_slot, IDX_DOOR_OUTCOME_DRIVER_NOTICED]: 
                        # Add a fraction of outbund_leg_ms
                        fraction = rng.uniform(0.3, 0.6) # NOTE: Draw a reasonable fraction
                        driver_notice_time_ms = round_up_to_multiple(fraction * outbund_leg_ms, dt_engine_ms)
                        if driver_notice_time_ms < door_specs_int64_matrix[assigned_door_code, IDX_FACILITY_DOOR_ALARM_TRIGGER_MS]:
                            back_and_forth_trip = round_up_to_multiple(2 * fraction * outbund_leg_ms, dt_engine_ms)
                            pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = t_anchor_ms + back_and_forth_trip
                            outbund_leg_ms += back_and_forth_trip
                            fast_catch_resolved = True

                    t_anchor_ms += outbund_leg_ms

                    # Retrieve boxes
                    pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_DISPATCH_LOGGING_MS] = t_anchor_ms
                    retrieval_delay_ms = round_up_to_multiple(
                        gamma_draw(
                            rng,
                            active_crop_params_float64_vector[IDX_FRUIT_PARAM_PALLET_RETRIEVAL_DELAY_MEAN_MS],
                            active_crop_params_float64_vector[IDX_FRUIT_PARAM_PALLET_RETRIEVAL_DELAY_SHAPE],
                        ), 
                        dt_engine_ms
                    )

                    penalty_mean  = active_crop_params_float64_vector[IDX_FRUIT_PARAM_BOX_HANDLING_MEAN_MS]
                    penalty_shape = active_crop_params_float64_vector[IDX_FRUIT_PARAM_BOX_HANDLING_SHAPE]

                    pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_REMOVAL_START_MS] = t_anchor_ms + retrieval_delay_ms

                    # if not pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH]:
                    if not outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IN_FULL_DISPATCH]:
                        # partial dispatch
                        box_removal_timestamps[...] = SENTINEL_FLOAT64
                        total_penalty_term_ms = 0
                        local_t_anchor = t_anchor_ms + retrieval_delay_ms
                        for i in range(num_boxes_requested):
                            raw_per_box_penalty_ms = gamma_draw(rng, penalty_mean, penalty_shape)
                            per_box_penalty_ms = round_up_to_multiple(raw_per_box_penalty_ms, dt_engine_ms)
                            local_t_anchor += per_box_penalty_ms
                            total_penalty_term_ms += per_box_penalty_ms
                            box_removal_timestamps[i] = local_t_anchor

                        # box_handling_penalty_ms = round_up_to_multiple(per_box_penalty_raw_ms * num_boxes_requested, dt_engine_ms)
                        retrieval_delay_ms += total_penalty_term_ms

                    t_anchor_ms += retrieval_delay_ms

                    pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_REMOVAL_END_MS] = t_anchor_ms

                    # pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET__MS] = t_anchor_ms

                    _, _, _, hour, day_of_week, _, is_last_weekday = compute_local_calendar_components(standard_close_time_ms, local_utc_offset_ms)
                    f_fatigue = calculate_fatigue_factor(
                        hour,
                        day_of_week,
                        is_last_weekday,
                    )
                    carelessness_factor = calculate_actual_carelessness_factor(
                        base_carelessness   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CARELESSNESS],
                        fatigue_factor      = f_fatigue,
                        fatigue_sensitivity = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_FATIGUE_SENSITIVITY],
                    )

                    # Return leg
                    return_leg_ms = gamma_draw(rng, return_leg_mean_ms, return_leg_shape)
                    t_anchor_ms += round_up_to_multiple(return_leg_ms*carelessness_factor, dt_engine_ms)

                    # Exit sequence
                    driver_opens_door_for_exit_ts_ms = t_anchor_ms # + round_up_to_multiple(rng.uniform(slack_low_bound_ms, slack_high_bound_ms), dt_engine_ms)

                    # if not enter_sequence_door_was_shut:
                    if door_outcome_flags_matrix[active_room_slot, IDX_DOOR_OUTCOME_UNNOTICED]:
                        pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = t_anchor_ms
                    elif fast_catch_resolved:
                        pass # already correctly set by the driver's fast catch
                    elif door_outcome_flags_matrix[active_room_slot, IDX_DOOR_OUTCOME_SUPERVISOR_NOTICED]:
                        # Calculate at which timestamp would supervisor shut the door
                        superv_slack_ms = rng.uniform(superv_slack_low_bound_ms, superv_slack_high_bound_ms)
                        supervisor_shuts_door_ts_ms = decision_t_anchor_ms + round_up_to_multiple(door_specs_int64_matrix[assigned_door_code, IDX_FACILITY_DOOR_ALARM_TRIGGER_MS] + superv_slack_ms, dt_engine_ms)
                        if (driver_opens_door_for_exit_ts_ms - supervisor_shuts_door_ts_ms) > 20_000:
                            # Supervisor shuts door
                            pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = supervisor_shuts_door_ts_ms
                            pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_SUPERVISOR_INTERVENED] = 1
                        else:
                            # Supervisor lets driver keep the door open
                            pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = t_anchor_ms
                    elif door_outcome_flags_matrix[active_room_slot, IDX_DOOR_OUTCOME_DRIVER_NOTICED]:
                        pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = t_anchor_ms

                        # if pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] == TOKEN_DOOR_NOTICED_BY_SUPERVISOR:
                        #     # Calculate at which timestamp would supervisor shut the door
                        #     superv_slack_ms = rng.uniform(superv_slack_low_bound_ms, superv_slack_high_bound_ms)
                        #     supervisor_shuts_door_ts_ms = decision_t_anchor_ms + round_up_to_multiple(door_specs_matrix[assigned_door_code, IDX_FACILITY_DOOR_ALARM_TRIGGER_MS] + superv_slack_ms, dt_engine_ms)
                        #     # supervisor_shuts_door_ts_ms = decision_t_anchor_ms + room_configurations_float64_vector[IDX_FACILITY_DOOR_ALARM_TRIGGER_MS] + rng.uniform(superv_slack_low_bound_ms, superv_slack_high_bound_ms)
                        #     if (driver_opens_door_for_exit_ts_ms - supervisor_shuts_door_ts_ms) > 30_000: # 30 sec buffer so they won't overlap. tau_door_ms is less than 5 sec
                        #         # Supervisor shuts door
                        #         pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = supervisor_shuts_door_ts_ms
                        #     else:
                        #         # Supervisor lets driver keep the door open
                        #         pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = t_anchor_ms
                        # else:
                        #     # Door goes unnoticed so it's kept open
                        #     pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = t_anchor_ms

                    # Door kept open means IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS = IDX_TIMELINE_PALLET_EXIT_DOOR_OPEN_MS
                    pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_DOOR_OPEN_MS] = t_anchor_ms

                    # pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_DOOR_OPEN_MS] = t_anchor_ms
                    # if pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] == SENTINEL_INT64:
                    #     pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS] = t_anchor_ms

                    operator_random_term = rng.uniform(slack_low_bound_ms, slack_high_bound_ms)
                    t_anchor_ms += tau_door_ms + round_up_to_multiple(operator_random_term, dt_engine_ms)

                    pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_BOUNDARY_CROSS_START_MS] = t_anchor_ms

                    operator_random_term = rng.uniform(slack_low_bound_ms, slack_high_bound_ms)
                    t_anchor_ms += round_up_to_multiple(door_transit_baseline_ms + operator_random_term, dt_engine_ms)
                    pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_BOUNDARY_CROSS_END_MS] = t_anchor_ms

                    operator_random_term = rng.uniform(slack_low_bound_ms, slack_high_bound_ms)
                    t_anchor_ms += round_up_to_multiple(operator_random_term, dt_engine_ms)
                    pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_DOOR_CLOSE_MS] = t_anchor_ms

                    buffer_random_term = rng.uniform(slack_low_bound_ms, slack_high_bound_ms)
                    t_anchor_ms += tau_door_ms + round_up_to_multiple(buffer_random_term, dt_engine_ms)

                    pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_SEQUENCE_COMPLETE_MS] = t_anchor_ms

        # Time-dependent actions
        # Pre-retrieval update
        if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_DISPATCH_LOGGING_MS]:
            # num_boxes_withdrawn = 0
            # net_mass_withdrawn = 0.0
            box_count = 0
            total_net_mass = 0.0
            # max_num_boxes = box_masses_ledger.shape[0]
            # num_boxes_in_pallet = pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_AVAILABLE_BOX_COUNT]
            num_boxes_withdrawn = 0
            net_mass_withdrawn = 0.0
            # is_full_dispatch = pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH]
            is_full_dispatch =  outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IN_FULL_DISPATCH]
            # must_register_first_box_for_removal = not is_full_dispatch
            first_box_registered = False

            w_start = outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_BOX_START_IDX]
            w_end = outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_BOX_END_IDX]
            # for box_idx in range(pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_BOX_SLOTS_COUNT] - 1, -1, -1):
            for box_idx in range(w_start, w_end - 1, -1):
                net_mass_per_box = box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_ACTUAL]
                if net_mass_per_box == SENTINEL_FLOAT64:
                    continue
                else:
                    package_mass_per_box_slot = box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_BOX_PACKAGE_MASS, IDX_BOX_LEDGER_ACTUAL]
                    pallet_mass_per_box_slot = box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_PALLET_MASS, IDX_BOX_LEDGER_ACTUAL]
                    if net_mass_per_box > 0.0: # Actual box movement
                        box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_START] = net_mass_per_box
                        box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TRANSIT] = net_mass_per_box
                        # Target should be 0.0 for boxes marked for dispatch and net_mass_per_box elsewhere, if functions were correctly written
                        # box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TARGET] = 0.0
                        # net_mass_per_pallet += net_mass_per_box
                        if not first_box_registered and not is_full_dispatch:
                                # Initialize box removal counter
                                pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_TARGET_BOX_IDX] = box_idx
                                pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_TARGET_BOX_COUNT] = 0
                                first_box_registered = True

                        total_net_mass += net_mass_per_box
                        box_count += 1
                    else: # Boxes have already been moved
                        box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_START] = 0.0
                        box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TRANSIT] = 0.0
                        # box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TARGET] = 0.0

                    box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_BOX_PACKAGE_MASS, IDX_BOX_LEDGER_START] = package_mass_per_box_slot
                    box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_BOX_PACKAGE_MASS, IDX_BOX_LEDGER_TRANSIT] = package_mass_per_box_slot
                    # Same target logic should apply here
                    # box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_BOX_PACKAGE_MASS, IDX_BOX_LEDGER_TARGET] = 0.0

                    box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_PALLET_MASS, IDX_BOX_LEDGER_START] = pallet_mass_per_box_slot
                    box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_PALLET_MASS, IDX_BOX_LEDGER_TRANSIT] = pallet_mass_per_box_slot
                    
                    if is_full_dispatch:
                        box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_PALLET_MASS, IDX_BOX_LEDGER_TARGET] = 0.0
                    else:
                        box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_PALLET_MASS, IDX_BOX_LEDGER_TARGET] = pallet_mass_per_box_slot

                    if box_count == num_boxes_requested:
                        num_boxes_withdrawn = box_count
                        net_mass_withdrawn = total_net_mass

            if net_mass_withdrawn == 0.0 or num_boxes_withdrawn == 0:
                print(
                    "box_count", box_count, "\n",
                    "total_net_mass", total_net_mass, "\n",
                    "num_boxes_requested", num_boxes_requested, "\n",
                    "active_room_slot", active_room_slot, "\n",
                    "is_full_dispatch", is_full_dispatch, "\n",
                    "box_slots_count", pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_BOX_SLOTS_COUNT], "\n",
                    "fruit_type", pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_CROP_TYPE], "\n",
                    "timestamp", room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS], "\n",
                    "step", step, "\n",
                )

            net_mass_total = pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_PULP_MASS]
            gross_mass_total = net_mass_total + box_count * active_crop_params_float64_vector[IDX_FRUIT_PARAM_BOX_PACKAGING_MASS]
            gross_mass_withdrawn = net_mass_withdrawn + num_boxes_withdrawn * active_crop_params_float64_vector[IDX_FRUIT_PARAM_BOX_PACKAGING_MASS]

            outbound_history_row_idx = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_OUTBOUND_HISTORY_ROW_IDX]
            outbound_pallet_history_ledger[outbound_history_row_idx, IDX_HISTORICAL_PALLET_BATCH_ID] = pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_BATCH_ID] # batch_id
            outbound_pallet_history_ledger[outbound_history_row_idx, IDX_HISTORICAL_PALLET_PARENT_ORDER_ID] = outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_PARENT_ORDER_ID]
            outbound_pallet_history_ledger[outbound_history_row_idx, IDX_HISTORICAL_PALLET_TIMESTAMP_MS] = epoch_timestamp_ms # timestamp
            outbound_pallet_history_ledger[outbound_history_row_idx, IDX_HISTORICAL_PALLET_QUALITY_GRADE] = pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_QUALITY_GRADE] # quality_grade
            outbound_pallet_history_ledger[outbound_history_row_idx, IDX_HISTORICAL_PALLET_PULP_TEMP] = pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_PULP_TEMPERATURE] # batch fruit pulp temperature at dispatch
            # outbound_pallet_history_ledger[outbound_history_row_idx, IDX_HISTORICAL_PALLET_PULP_TEMP_FILTERED_VAL] = pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_PULP_TEMP_FILTERED_VAL]
            outbound_pallet_history_ledger[outbound_history_row_idx, IDX_HISTORICAL_PALLET_GROSS_START] = gross_mass_total # mass_at_dispatch
            outbound_pallet_history_ledger[outbound_history_row_idx, IDX_HISTORICAL_PALLET_BOX_START] = box_count # number of boxes at dispatch
            outbound_pallet_history_ledger[outbound_history_row_idx, IDX_HISTORICAL_PALLET_GROSS_MOVED] = gross_mass_withdrawn # mass_removed
            outbound_pallet_history_ledger[outbound_history_row_idx, IDX_HISTORICAL_PALLET_BOX_MOVED] = num_boxes_withdrawn
            # outbound_pallet_history_ledger[outbound_history_row_idx, IDX_HISTORICAL_PALLET_GROSS_START_FILTERED_VAL] = pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_MASS_FILTERED_VAL]

            if pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_SUPERVISOR_INTERVENED]:
                outbound_pallet_history_ledger[outbound_history_row_idx, IDX_HISTORICAL_PALLET_SUPERVISOR_INTERVENTION_MS] = pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS]
                pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_SUPERVISOR_INTERVENED] = 0
            else:
                outbound_pallet_history_ledger[outbound_history_row_idx, IDX_HISTORICAL_PALLET_SUPERVISOR_INTERVENTION_MS] = MAX_EPOCH_TIMESTAMP_MS

            room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_OUTBOUND_HISTORY_ROW_IDX] += 1

        if pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_BOUNDARY_CROSS_END_MS] <= epoch_timestamp_ms < pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_BOUNDARY_CROSS_START_MS]:
            if epoch_timestamp_ms % room_configurations_int64_vector[IDX_ROOM_LOGISTICS_DRIVER_Q_SAMPLING_INTERVAL_MS] == 0:
                current_active_driver_convective_q = pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_ACTIVE_DRIVER_CONVECTIVE_Q]
                updated_active_driver_convective_q = bounded_normal_draw(
                    rng = rng,
                    mean  = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q_MEAN],
                    scale = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q_SCALE],
                    low   = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q_LOW],
                    high  = employee_parameters_matrix[active_driver_code, IDX_EMP_PARAM_CONVECTIVE_Q_HIGH],
                )
                pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_ACTIVE_DRIVER_CONVECTIVE_Q] = updated_active_driver_convective_q
                room_state_float64_vector[IDX_ROOM_STATE_BULK_ACTIVE_DRIVERS_CONVECTIVE_Q] += (updated_active_driver_convective_q - current_active_driver_convective_q) 

        # if (
        #     epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_REMOVAL_START_MS]
        #     and not pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH]
        # ):
        #     # New pallet in airflow that will be counted
        #     room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_AIRFLOW] += 1

        if pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_REMOVAL_START_MS] <= epoch_timestamp_ms <= pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_REMOVAL_END_MS]:
                total_target_boxes = num_boxes_requested
                new_pallet_mass_fraction = active_crop_params_float64_vector[IDX_FRUIT_PARAM_PALLET_BASE_MASS] / total_target_boxes
                target_box_idx = pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_TARGET_BOX_IDX]
                # Partial dispatch, boxes are removed sequencially
                if (
                    # not pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH]
                    not outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IN_FULL_DISPATCH]
                    and pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_TARGET_BOX_COUNT] < total_target_boxes
                    and epoch_timestamp_ms == box_removal_timestamps[target_box_idx]
                ):
                    box_flags_ledger[target_box_idx, active_room_slot, IDX_BOX_IS_IN_PALLET] = 0
                    box_flags_ledger[target_box_idx, active_room_slot, IDX_BOX_IS_IN_TRANSIT] = 1

                    # Add mass of pallet that was brought for the partial dispatch
                    box_masses_ledger[target_box_idx, active_room_slot, IDX_BOX_LEDGER_NEW_PALLET_MASS, IDX_BOX_LEDGER_ACTUAL] = new_pallet_mass_fraction
                    partial_dispatch_box_temperatures_ledger[target_box_idx, active_room_slot] = pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_PULP_TEMPERATURE]

                    pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_TARGET_BOX_IDX] -= 1
                    pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_TARGET_BOX_COUNT] += 1

                    # Remove the gross mass of the target box from pallet trackers
                    current_pallet_pulp_mass = pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_PULP_MASS]
                    current_pallet_box_packaging_mass = pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_BOX_PACKAGING_MASS]
                    target_box_pulp_mass = box_masses_ledger[target_box_idx, active_room_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_ACTUAL]
                    target_box_packaging_mass = box_masses_ledger[target_box_idx, active_room_slot, IDX_BOX_LEDGER_BOX_PACKAGE_MASS, IDX_BOX_LEDGER_ACTUAL]
                    pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_PULP_MASS] = max(current_pallet_pulp_mass - target_box_pulp_mass, 0.0)
                    pallet_physics_float64_matrix[active_room_slot, IDX_PALLET_PHYSICS_BOX_PACKAGING_MASS] = max(current_pallet_box_packaging_mass - target_box_packaging_mass, 0.0)

                # Full dispatch, whole pallet is removed
                elif (
                    # pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH]
                    outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IN_FULL_DISPATCH]
                    and epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_REMOVAL_END_MS]
                ):
                    pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IS_PALLET_IN_PLACE] = 0
                    # Change all box slots too true, so empty box slots also receive the scaled k_zone
                    box_flags_ledger[..., active_room_slot, IDX_BOX_IS_IN_TRANSIT] = 1

        # if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_PICKED_UP_MS]:
        #     num_boxes_withdrawn = outbound_pallets_ledger[outbound_history_row_idx, IDX_HISTORICAL_PALLET_BOX_MOVED]
        #     # Activate transit flag
        #     for box_idx in range(box_flags_ledger.shape[0]):
        #         if box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_ACTUAL] == SENTINEL_FLOAT64:
        #             continue
        #         # Check which boxes are actually being removed
        #         # Boxes with target = 0.0 have been marked for removal
        #         if box_masses_ledger[box_idx, active_room_slot, IDX_BOX_LEDGER_PALLET_MASS, IDX_BOX_LEDGER_TARGET] == 0.0:
        #             box_flags_ledger[box_idx, active_room_slot, IDX_BOX_IS_IN_TRANSIT] = 1
        if (
            epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_BOUNDARY_CROSS_START_MS]
            or epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_BOUNDARY_CROSS_START_MS]
        ):
            updated_blobs_counter = add_id_element(
                crossing_blobs_compact_vector,
                inverse_crossing_blobs_lookup_vector,
                room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BLOBS_CROSSING],
                active_room_slot,
            )
            room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BLOBS_CROSSING] = updated_blobs_counter

        # if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_BOUNDARY_CROSS_START_MS]:
        #     if pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH]:
        #         room_state_float64_vector[IDX_ROOM_STATE_TOTAL_OBJECTS_CROSSING] += 1 # Operator
        #     else:
        #         room_state_float64_vector[IDX_ROOM_STATE_TOTAL_OBJECTS_CROSSING] += 2 # Operator + new pallet

        if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_BOUNDARY_CROSS_START_MS]:
            pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IS_CROSSING_BOUNDARY] = 1

        if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_OPEN_MS]:
            door_states_int64_matrix[assigned_door_code, IDX_DOOR_STATE_OPEN_MS] = pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_OPEN_MS]
            door_states_int64_matrix[assigned_door_code, IDX_DOOR_STATE_CLOSE_MS] = pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS]

        if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_DOOR_OPEN_MS]:
            door_states_int64_matrix[assigned_door_code, IDX_DOOR_STATE_OPEN_MS] = pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_DOOR_OPEN_MS]
            door_states_int64_matrix[assigned_door_code, IDX_DOOR_STATE_CLOSE_MS] = pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_DOOR_CLOSE_MS]

        # if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_ENTER_DOOR_CLOSE_MS]:
        #     supervisor_intervened = pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_SUPERVISOR_INTERVENED]
        #     outbound_history_row_idx = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_OUTBOUND_HISTORY_ROW_IDX]
        #     if supervisor_intervened:
        #         outbound_pallet_history_ledger[outbound_history_row_idx, IDX_HISTORICAL_PALLET_SUPERVISOR_SHUT_DOOR] = supervisor_intervened
        #     pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_SUPERVISOR_INTERVENED] = 0


        if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_EXIT_BOUNDARY_CROSS_END_MS]:
            pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_ACTUAL_BOX_COUNT] -= num_boxes_requested
            
        if epoch_timestamp_ms == pallet_timeline_int64_matrix[active_room_slot, IDX_TIMELINE_PALLET_SEQUENCE_COMPLETE_MS]:
            pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IN_DISPATCH_RUN] = 0
            outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IS_EVENT_SCHEDULED] = 0

            if outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IS_LAST_IN_MACRO_ORDER]:
                door_states_int64_matrix[assigned_door_code, IDX_DOOR_STATE_IS_BUSY] = 0

            # if pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH]:
            #     pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH] = 0
            if outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IN_FULL_DISPATCH]:
                outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IN_FULL_DISPATCH] = 0
                room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_ROOM] -= 1
            # else:
            #     partial_dispatch_box_temperatures_ledger[...] = SENTINEL_FLOAT64

            room_state_int64_vector[IDX_ROOM_STATE_TOTAL_DISPATCHES_PLANNED] -= 1
            room_state_int64_vector[IDX_ROOM_STATE_DISPATCH_WAVE_READ_PTR] += 1

            # Reset helper arrays
            partial_dispatch_box_temperatures_ledger[..., active_room_slot] = SENTINEL_FLOAT64
            pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_TARGET_BOX_IDX] = SENTINEL_INT64
            pallet_metadata_int64_matrix[active_room_slot, IDX_PALLET_METADATA_TARGET_BOX_COUNT] = SENTINEL_INT64


@njit
def update_mass_states(
    # step, ### DEBUG ###
    room_state_float64_vector,
    room_state_int64_vector,
    pallet_physics_float64_matrix,
    pallet_metadata_int64_matrix,
    pallet_timeline_int64_matrix,
    inbound_wave_queue_matrix,
    outbound_wave_queue_matrix,
    box_flags_ledger,
    box_masses_ledger,

    crossing_blobs_compact_vector,
    inverse_crossing_blobs_lookup_vector,

    airflow_pallets_compact_vector,
    inverse_airflow_pallets_lookup_vector,

    dt_engine_ms,
):
    # max_num_pallets = pallet_metadata_int64_matrix.shape[0]
    # max_num_boxes = box_masses_ledger.shape[0]

    # Q_forklift_operator_sensible_bulk = 0.0
    # gross_mass_bulk_kg = 0.0

    # room_state_float64_vector
    # IDX_STATE_PHYSICS_FORKLIFT_DRIVERS_Q_CONVECTIVE
    # IDX_STATE_PHYSICS_FRUIT_GROSS_MASS_BULK
    room_net_step_pulp_dm_accumulation = 0.0
    room_net_step_box_packaging_dm_accumulation = 0.0
    room_net_step_pallet_dm_accumulation = 0.0
    bulk_operator_delta_Q = 0.0
    total_blobs_crossing = room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BLOBS_CROSSING]
    epoch_timestamp_ms = room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS]
    next_epoch_timestamp_ms = epoch_timestamp_ms + dt_engine_ms

    # if not SKIP_PALLET_LOOPS:
    for i in range(total_blobs_crossing - 1, -1, -1):
        pallet_slot = crossing_blobs_compact_vector[i]
        pallet_net_step_pulp_dm_accumulation = 0.0
        pallet_net_step_box_packaging_dm_accumulation = 0.0
        pallet_net_step_pallet_dm_accumulation = 0.0
        # Operator movement logic
        # if pallet_timeline_int64_matrix[pallet_slot, IDX_TIMELINE_PALLET_ENTER_BOUNDARY_CROSS_START_MS] <= epoch_timestamp_ms < pallet_timeline_int64_matrix[pallet_slot, IDX_TIMELINE_PALLET_EXIT_BOUNDARY_CROSS_END_MS]:
        is_entering = epoch_timestamp_ms < pallet_timeline_int64_matrix[pallet_slot, IDX_TIMELINE_PALLET_ENTER_BOUNDARY_CROSS_END_MS]
        # is_exiting = epoch_timestamp_ms >= pallet_timeline_int64_matrix[pallet_slot, IDX_TIMELINE_PALLET_EXIT_BOUNDARY_CROSS_START_MS]
        Q_forklift_operator_sensible_draw = pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_ACTIVE_DRIVER_CONVECTIVE_Q]
        if is_entering:
            timeline_start_idx = IDX_TIMELINE_PALLET_ENTER_BOUNDARY_CROSS_START_MS
            timeline_end_idx = IDX_TIMELINE_PALLET_ENTER_BOUNDARY_CROSS_END_MS
            operator_starting_point = 0.0
            operator_end_point = Q_forklift_operator_sensible_draw
        else:
            timeline_start_idx = IDX_TIMELINE_PALLET_EXIT_BOUNDARY_CROSS_START_MS
            timeline_end_idx = IDX_TIMELINE_PALLET_EXIT_BOUNDARY_CROSS_END_MS
            operator_starting_point = Q_forklift_operator_sensible_draw
            operator_end_point = 0.0

        total_window_ms = pallet_timeline_int64_matrix[pallet_slot, timeline_end_idx] - pallet_timeline_int64_matrix[pallet_slot, timeline_start_idx]
        # ### DEBUG ####
        # if total_window_ms <= 0:
        #     print("ZERO WINDOW at step", step, "pallet_slot", pallet_slot, "start_idx", timeline_start_idx,
        #           pallet_timeline_int64_matrix[pallet_slot, timeline_start_idx],
        #           pallet_timeline_int64_matrix[pallet_slot, timeline_end_idx])
        # #### DEBUG ####

        elapsed_transit_ms = next_epoch_timestamp_ms - pallet_timeline_int64_matrix[pallet_slot, timeline_start_idx]
        completion_fraction = min(elapsed_transit_ms/total_window_ms, 1)

        # Compute the constant rate of change per millisecond (the Euler derivative)
        dQ_per_ms = (operator_end_point - operator_starting_point) / total_window_ms

        # Floating-Point Guardrail: Identify the exact steps to prevent drift
        is_first_step = (epoch_timestamp_ms == pallet_timeline_int64_matrix[pallet_slot, timeline_start_idx])
        is_final_step = (next_epoch_timestamp_ms >= pallet_timeline_int64_matrix[pallet_slot, timeline_end_idx])

        if is_first_step:
            # First step delta: integrated forward from the exact start point
            current_step_value_at_t1 = operator_starting_point + (dQ_per_ms * dt_engine_ms)
            operator_delta_Q = current_step_value_at_t1 - operator_starting_point
        
        elif is_final_step:
            # CRITICAL GUARDRAIL: Force the final position to be exactly operator_end_point.
            # This completely eliminates floating-point drift accumulation at the window boundary.
            previous_step_theoretical_t0 = operator_end_point - (dQ_per_ms * dt_engine_ms)
            operator_delta_Q = operator_end_point - previous_step_theoretical_t0
            updated_blobs_counter = remove_element_by_id(
                crossing_blobs_compact_vector,
                inverse_crossing_blobs_lookup_vector,
                room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BLOBS_CROSSING],
                pallet_slot
            )
            room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BLOBS_CROSSING] = updated_blobs_counter

        else:
            # Standard Euler Step Delta for intermediate intervals
            operator_delta_Q = dQ_per_ms * dt_engine_ms

        bulk_operator_delta_Q += operator_delta_Q # * 0.2 ### NOTE: DEBUG ###
        # Q_forklift_operator_sensible_actual = operator_starting_point * (1 - completion_fraction) + operator_end_point * completion_fraction

        if pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IS_CROSSING_BOUNDARY]:
            # is_partial_dispatch_in_progress = (
            #     pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_DISPATCH_RUN]
            #     and not outbound_wave_queue_matrix[dispatch_q_read_slot, IDX_WAVE_IN_FULL_DISPATCH]
            #     # and not pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH]
            # )
            is_partial_dispatch_in_progress = False
            is_full_dispatch_in_progress = False
            q_read_slot = pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_QUEUE_READ_SLOT]
            if pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_DISPATCH_RUN]:
                w_start = outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_BOX_START_IDX]
                w_end   = outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_BOX_END_IDX]
                if not outbound_wave_queue_matrix[q_read_slot, IDX_WAVE_IN_FULL_DISPATCH]:
                    is_partial_dispatch_in_progress = True
                else:
                    is_full_dispatch_in_progress = True
            else:
                w_start = inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_BOX_START_IDX]
                w_end   = inbound_wave_queue_matrix[q_read_slot, IDX_WAVE_BOX_END_IDX]

            # for box_idx in range(pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_BOX_SLOTS_COUNT]):
            for box_idx in range(w_start, w_end - 1, -1):
                starting_mass = box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_START]
                transit_mass = box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TRANSIT]
                target_mass = box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TARGET]
                new_transit_mass = starting_mass + completion_fraction * (target_mass - starting_mass)
                dm_step = new_transit_mass - transit_mass
                # if step == 150320 and pallet_slot == 0:
                # # if 150319 <= step <= 150321 and pallet_slot == 0:
                #     print("OPQ CHECK", step, dQ_per_ms, operator_delta_Q, is_first_step, is_final_step, total_window_ms)
                #     print("SUM CHECK", pallet_net_step_pulp_dm_accumulation, pallet_net_step_box_packaging_dm_accumulation, pallet_net_step_pallet_dm_accumulation)
                    # print(box_idx, pallet_slot, starting_mass, target_mass, completion_fraction, total_window_ms, dm_step,
                    #     "BOX_SLOTS_COUNT:", pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_BOX_SLOTS_COUNT])
                #     print(box_idx, pallet_slot, starting_mass, target_mass, completion_fraction, total_window_ms, dm_step)
                  # box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TRANSIT] = new_transit_mass

                box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TRANSIT] = new_transit_mass
                current_box_net_mass = box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_ACTUAL]
                updated_box_net_mass = max(current_box_net_mass + dm_step, 0.0)
                # Track the real, bounded mass delta for this box
                net_fruit_dm = updated_box_net_mass - current_box_net_mass
                box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_ACTUAL] = updated_box_net_mass
                # Update total pulp mass in room
                room_net_step_pulp_dm_accumulation += net_fruit_dm
                # Only update batch mass if box is in pallet
                if box_flags_ledger[box_idx, pallet_slot, IDX_BOX_IS_IN_PALLET]:
                    pallet_net_step_pulp_dm_accumulation += net_fruit_dm
                    # pallet_net_step_packaging_dm_accumulation += net_fruit_dm
                    # pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PULP_MASS] += (updated_box_net_mass - current_box_net_mass)
                    
                elif is_partial_dispatch_in_progress:
                    # New pallet brought for partial dispatches
                    # New pallet's only purpose is to contribute to outgoing boxes heat capacity
                    current_pallet_fraction_mass = box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NEW_PALLET_MASS, IDX_BOX_LEDGER_ACTUAL]
                    updated_pallet_fraction_mass = max(current_pallet_fraction_mass * (1 - completion_fraction), 0.0)
                    pallet_fraction_dm = updated_pallet_fraction_mass - current_pallet_fraction_mass
                    # Only update the accumulator for total packaging mass in room
                    # room_net_step_packaging_dm_accumulation += pallet_fraction_dm
                    # pallet_net_step_dm_accumulation += (updated_pallet_fraction_mass - current_pallet_fraction_mass)
                    box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NEW_PALLET_MASS, IDX_BOX_LEDGER_ACTUAL] = updated_pallet_fraction_mass
                    
                for partition_idx in range(1, 3):
                    material_start_mass = box_masses_ledger[box_idx, pallet_slot, partition_idx, IDX_BOX_LEDGER_START]
                    material_target_mass = box_masses_ledger[box_idx, pallet_slot, partition_idx, IDX_BOX_LEDGER_TARGET]
                    material_actual_mass = material_start_mass * (1 - completion_fraction) + material_target_mass * completion_fraction
                    updated_material_mass = max(material_actual_mass, 0.0)
                    material_dm = updated_material_mass - box_masses_ledger[box_idx, pallet_slot, partition_idx, IDX_BOX_LEDGER_ACTUAL]
                    # Accumulate partition mass shifts (package/wood/cardboard) into the step delta
                    # if box_flags_ledger[box_idx, pallet_slot, IDX_BOX_IS_IN_PALLET]:
                    if partition_idx == IDX_BOX_LEDGER_BOX_PACKAGE_MASS:
                        room_net_step_box_packaging_dm_accumulation += material_dm
                        if box_flags_ledger[box_idx, pallet_slot, IDX_BOX_IS_IN_PALLET]:
                            pallet_net_step_box_packaging_dm_accumulation += material_dm

                    elif partition_idx == IDX_BOX_LEDGER_PALLET_MASS:
                        pallet_net_step_pallet_dm_accumulation += material_dm
                        room_net_step_pallet_dm_accumulation += material_dm
                    
                    box_masses_ledger[box_idx, pallet_slot, partition_idx, IDX_BOX_LEDGER_ACTUAL] = updated_material_mass

                    #     and partition_idx == IDX_BOX_LEDGER_BOX_PACKAGE_MASS
                    # ):
                    #     # Only update here for box packaging masses (from boxes currently sitting on pallet)
                    #     pallet_net_step_box_packaging_dm_accumulation += material_dm
                    # elif partition_idx == IDX_BOX_LEDGER_PALLET_MASS:
                    #     # Always update for pallet fraction mass (which stays until full dispatches)
                    #     pallet_net_step_pallet_dm_accumulation += material_dm
                    # # Account for everything, including 'detached' boxes from partial dispatches
                    # room_net_step_packaging_dm_accumulation += material_dm # Should detached the new pallet be counter for V_free calculation?

                if (
                    is_final_step
                    and box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_TARGET] == 0.0
                ):
                    # Transit flags in dispatch can be deactivated when box is gone
                    # They are only relevant for net fruit mass and temperature updates
                    box_flags_ledger[box_idx, pallet_slot, IDX_BOX_IS_IN_TRANSIT] = 0

                    if  (  
                        box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_BOX_PACKAGE_MASS, IDX_BOX_LEDGER_TARGET] == 0.0
                        and box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_PALLET_MASS, IDX_BOX_LEDGER_TARGET] == 0.0
                    ):
                        # Only deactivate slots when the whole pallet is finally gone
                        box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_ACTUAL] = SENTINEL_FLOAT64
                        box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_BOX_PACKAGE_MASS, IDX_BOX_LEDGER_ACTUAL] = SENTINEL_FLOAT64
                        box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_PALLET_MASS, IDX_BOX_LEDGER_ACTUAL] = SENTINEL_FLOAT64
                    
            # Accumulate the total calculated delta step into your new state variable
            pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PULP_MASS] = max(pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PULP_MASS] + pallet_net_step_pulp_dm_accumulation, 0.0)
            pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_BOX_PACKAGING_MASS] = max(pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_BOX_PACKAGING_MASS] + pallet_net_step_box_packaging_dm_accumulation, 0.0)
            pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PALLET_MASS] = max(pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PALLET_MASS] + pallet_net_step_pallet_dm_accumulation, 0.0)

            if is_final_step:
                pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IS_CROSSING_BOUNDARY] = 0
                # if pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH]:
                if is_full_dispatch_in_progress:
                    # Remove pallet from airflow
                    updated_airflow_pallets_counter = remove_element_by_id(
                        airflow_pallets_compact_vector,
                        inverse_airflow_pallets_lookup_vector,
                        room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_AIRFLOW],
                        pallet_slot
                    )
                    room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_AIRFLOW] = updated_airflow_pallets_counter

                    pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IS_IN_AIRFLOW] = 0
                    pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IS_IN_INVENTORY] = 0
                    pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_AGE_MS] = SENTINEL_INT64
                    pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_BOX_SLOTS_COUNT] = 0
                    pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PULP_MASS] = 0.0
                    pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_BOX_PACKAGING_MASS] = 0.0
                    pallet_physics_float64_matrix[pallet_slot, IDX_PALLET_PHYSICS_PALLET_MASS] = 0.0

    if total_blobs_crossing > 0:
        room_state_float64_vector[IDX_ROOM_STATE_BULK_PULP_MASS] = max(room_state_float64_vector[IDX_ROOM_STATE_BULK_PULP_MASS] + room_net_step_pulp_dm_accumulation, 0.0)
        room_state_float64_vector[IDX_ROOM_STATE_BULK_BOX_PACKAGING_MASS] = max(room_state_float64_vector[IDX_ROOM_STATE_BULK_BOX_PACKAGING_MASS] + room_net_step_box_packaging_dm_accumulation, 0.0)
        room_state_float64_vector[IDX_ROOM_STATE_BULK_PALLET_MASS] = max(room_state_float64_vector[IDX_ROOM_STATE_BULK_PALLET_MASS] + room_net_step_pallet_dm_accumulation, 0.0)
        
        room_state_float64_vector[IDX_ROOM_STATE_BULK_ACTIVE_DRIVERS_CONVECTIVE_Q] = max(room_state_float64_vector[IDX_ROOM_STATE_BULK_ACTIVE_DRIVERS_CONVECTIVE_Q] + bulk_operator_delta_Q, 0.0)
                # # Calculate gross mass inside room
                # if pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IS_IN_AIRFLOW]:
                #     gross_mass_bulk_kg += box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_ACTUAL]
                #     gross_mass_bulk_kg += box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_BOX_PACKAGE_MASS, IDX_BOX_LEDGER_ACTUAL]
                #     gross_mass_bulk_kg += box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_PALLET_MASS, IDX_BOX_LEDGER_ACTUAL]
                #     if (
                #         not box_flags_ledger[box_idx, pallet_slot, IDX_BOX_IS_IN_PALLET]
                #         and box_flags_ledger[box_idx, pallet_slot, IDX_BOX_IS_IN_TRANSIT]
                #         and is_partial_dispatch_in_progress
                #     ):
                #         # Include extra mass of pallet brought for partial dispatches
                #         gross_mass_bulk_kg += box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NEW_PALLET_MASS, IDX_BOX_LEDGER_ACTUAL]
                #     # is_partial_dispatch_in_progress = (
                #     #     pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_DISPATCH_RUN]
                #     #     and not pallet_metadata_int64_matrix[pallet_slot, IDX_PALLET_METADATA_IN_FULL_DISPATCH]
                #     # )
                #     # for box_idx in range(box_masses_ledger.shape[0]):
                #     #     fruit_mass_in_box_kg = box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NET_MASS, IDX_BOX_LEDGER_ACTUAL]
                #     #     if fruit_mass_in_box_kg == SENTINEL_FLOAT64:
                #     #         continue
                #     #     else:
                #     #         gross_mass_bulk_kg += box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_BOX_PACKAGE_MASS, IDX_BOX_LEDGER_ACTUAL]
                #     #         gross_mass_bulk_kg += box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_PALLET_MASS, IDX_BOX_LEDGER_ACTUAL]
                #     #         if fruit_mass_in_box_kg > 0.0:
                #     #             gross_mass_bulk_kg += fruit_mass_in_box_kg
                #     #             if (
                #     #                 not box_flags_ledger[box_idx, pallet_slot, IDX_BOX_IS_IN_PALLET]
                #     #                 and box_flags_ledger[box_idx, pallet_slot, IDX_BOX_IS_IN_TRANSIT]
                #     #                 and is_partial_dispatch_in_progress
                #     #             ):
                #     #                 gross_mass_bulk_kg += box_masses_ledger[box_idx, pallet_slot, IDX_BOX_LEDGER_NEW_PALLET_MASS, IDX_BOX_LEDGER_ACTUAL]
