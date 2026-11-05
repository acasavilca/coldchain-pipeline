"""
engine.py

The step-loop entry point, run_simulator(). Everything it calls lives in
wrappers.py (per-step physics/state updates), simulator_engine.py (schedule
and downtime bookkeeping), and output_writer.py (per-step array writes).

Formerly big_loop_corrections.py. That file's original docstring documented
three bugs (calculate_forcing_data's missing alpha_roof lookup,
derive_fixed_hardware_parameters' missing rho_air_ambient/forcing_cache_vector
wiring, and calculate_fruit_heat_terms' lowercase INV_3600/INV_M_CO2 typos) --
all three are already fixed in wrappers.py in this version, so the corrections
have been folded in and this file now just holds the loop itself.
"""

from constants import *
from matrix_index_maps import *
from room_environment import *
from refrigeration_cycle import *
from humidity_gas import *
from sensors_control import *
from workforce_behavior import *
from arrivals_dispatch import *
from output_writer import *

# Recompile this orchestration function each process: Numba's disk cache does
# not reliably invalidate it when an inlined physics helper changes.
@njit
def run_simulator(
    # RNG object
    rng,
    # State arrays
    room_state_float64_vector,
    room_state_int64_vector, # contains stating epoch
    pallet_metadata_int64_matrix,
    pallet_timeline_int64_matrix,
    pallet_physics_float64_matrix,
    outbound_wave_queue_matrix,
    inbound_wave_queue_matrix,
    box_masses_ledger,
    box_flags_ledger,
    partial_dispatch_box_temperatures_ledger,
    door_states_int64_matrix,
    door_states_float64_matrix,
    # Sliced arrays (states too)
    active_crop_params_float64_vector,
    active_crop_params_int64_vector,
    active_crop_historical_yield_matrix,
    active_crop_market_scarcity_matrix,
    active_crop_base_grades_probs_matrix,
    # active_crop_inventory_mass_per_grade_vector,
    active_crop_respiration_curves_matrix,
    active_refrigerant_properties_vector,
    active_refrigerant_surfaces_matrix,
    # Processed downtime/rotation arrays
    downtime_timestamps_ms_vector,
    valid_crop_rotation_timestamps_ms_vector,
    valid_refrigerant_rotation_timestamps_ms_vector,

   # Simulation exclusive ending epoch
    sim_end_epoch_ms,

    # Forcing arrays/scalar
    forcing_matrix,
    local_utc_offset_ms_vector,
    forcing_start_time_ms, # scalar
    
    # Parameter arrays
    room_configurations_float64_vector,
    room_configurations_int64_vector,
    equipment_configurations_float64_vector,
    equipment_configurations_int64_vector,
    fruit_parameters_float64_matrix,
    fruit_parameters_int64_matrix,
    fruit_respiration_curves_matrix,
    historical_yield_matrix,
    day_of_week_weights,
    base_quality_grade_probs_matrix,
    lambda_event_per_hour_matrix,
    employee_parameters_matrix,
    supervisor_parameters_matrix,
    door_specs_float64_matrix,
    door_specs_int64_matrix,
    refrigerant_properties_matrix,
    refrigerant_surfaces_matrix,
    evaporator_fan_curve_matrix,

    # Ephemeral arrays
    room_ephemeral_int64_vector,
    inventory_mass_per_grade_vector,
    available_pallet_indices,
    box_removal_timestamps,
    crossing_blobs_compact_vector,
    inverse_crossing_blobs_lookup_vector,
    airflow_pallets_compact_vector,
    inverse_airflow_pallets_lookup_vector,
    forcing_cache_float64_vector,
    forcing_cache_int64_vector,
    door_outcome_flags_matrix,
    # Output arrays (ephemeral too)
    inbound_pallet_history_ledger,
    inbound_orders_ledger,
    outbound_pallet_history_ledger,
    outbound_orders_ledger,
    room_telemetry_matrix,
    batch_fruit_sensor_ledger,
    compressor_event_ledger,
    humidifier_event_ledger,
    door_event_ledger,
    crop_rotation_event_ledger,
    refrigerant_rotation_event_ledger,
    downtime_sequence_event_ledger,
    lp_trip_event_ledger,
    defrost_event_ledger,

    # Output arrays for testing
    output_int64_matrix,
    output_float64_matrix,
):
    # Unpack frequently used variables
    dt_engine_sec = room_configurations_float64_vector[IDX_ROOM_CFG_DT_ENGINE_SEC]
    dt_engine_ms  = room_configurations_int64_vector[IDX_ROOM_CFG_DT_ENGINE_MS]

    dt_plant_sec  = room_configurations_float64_vector[IDX_ROOM_PLANT_DT_STEP_SEC]
    dt_plant_ms   = room_configurations_int64_vector[IDX_ROOM_PLANT_DT_STEP_MS]

    dt_planner_hr = room_configurations_float64_vector[IDX_ROOM_LOGISTICS_PLANNER_INTERVAL_HR]
    dt_planner_ms = room_configurations_int64_vector[IDX_ROOM_LOGISTICS_PLANNER_INTERVAL_MS]

    # Initialize local CPU register scalars for forcing data
    last_forcing_idx = -1

    # Simulation loop
    # When units are not specified, assume SI units
    total_steps = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_TOTAL_STEPS]
    forcing_bucket_ms = room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_FORCING_BUCKET_MS]

    # NOTE: For now, always use magnus constants for liquid water
    # TODO: Add the option to use ice constants after implementing defrost system
    # magnus_const_set = select_magnus_constants()

    for k in range(total_steps):
        # Was: unpack_forcing_data(...) -- that function doesn't exist, and
        # its expected 8-value return doesn't match what calculate_forcing_data
        # actually does (it mutates forcing_cache_vector and returns only the
        # updated bucket-tracker index).
        last_forcing_idx = calculate_forcing_data(
            forcing_matrix,
            local_utc_offset_ms_vector,
            forcing_cache_float64_vector,
            forcing_cache_int64_vector,
            room_state_int64_vector,
            room_state_float64_vector,
            forcing_start_time_ms,
            forcing_bucket_ms,
            last_forcing_idx,
            room_configurations_float64_vector,
            equipment_configurations_float64_vector,
        )

        update_main_logistics_and_operation_flags(
            room_state_int64_vector,
            room_state_float64_vector,
            room_ephemeral_int64_vector,
            equipment_configurations_float64_vector,
        )

        update_downtime_and_ramp_up_schedule(
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
            # inventory_mass_per_grade_vector,
            fruit_respiration_curves_matrix,

            active_crop_params_float64_vector,
            active_crop_params_int64_vector,
            active_crop_historical_yield_matrix,
            active_crop_market_scarcity_matrix,
            active_crop_base_grades_probs_matrix,
            # active_crop_inventory_mass_per_grade_vector,
            active_crop_respiration_curves_matrix,
            active_refrigerant_properties_vector,
            active_refrigerant_surfaces_matrix,

            crop_rotation_event_ledger,
            refrigerant_rotation_event_ledger,
            downtime_sequence_event_ledger,

            dt_engine_ms,
        )

        event_planner(
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
        )

        arrival_timing_decision(
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
        )

        dispatch_timing_decision(
            k,
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
        )

        # Was: calculate_air_properties(room_state_float64_vector, T_ambient, RH_ambient, P_ambient)
        # unpacked as a 6-tuple -- def only takes 3 args and returns 1 value.
        rho_air_room = calculate_air_properties(
            room_state_float64_vector,
            forcing_cache_float64_vector,
        )

        update_mass_states( # TODO: REMOVE DEBUG argument
            # k,
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
        )

        # Was missing the 5th return value (inv_m_air_room) entirely.
        V_free_m3, f_free, C_room, m_air_room, inv_m_air_room = calculate_room_capacity_terms(
            room_state_float64_vector,
            room_configurations_float64_vector,
            active_crop_params_float64_vector,
            rho_air_room,
        )
        # NOTE: calculate_room_capacity_terms is missing its @njit decorator in
        # wrappers.py -- add it, or every njit caller upstream of it will fail
        # to compile.

        (
            Q_respiration_bulk, Q_room_exchange_bulk, Q_evap_fruit_bulk,
            Q_batch_to_bulk_exchange, CO2_prod_ppm_sec_bulk,
            C_composite_fruit_bulk, m_dot_transpiration_bulk, water_loss_bulk_kg,
        ) = calculate_fruit_heat_terms(
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
        )

        # # Needs forcing_cache_vector added as a parameter (see bug #2 above).
        # (
        #     Q_cooling_w_rated, W_compressor_w_rated, m_dot_refrig_kg_s_rated,
        #     V_dot_air_evap_fully_on, V_dot_air_cond_fully_on,
        #     UA_coil_evap, UA_internal_evap, UA_coil_cond, UA_internal_cond,
        #     C_clearance,
        # ) = derive_rated_hardware_constants(
        #     room_state_float64_vector, forcing_cache_vector,
        #     equipment_configurations_vector,
        #     active_crop_params_vector, active_refrigerant_properties_vector,
        #     active_refrigerant_surfaces_matrix,
        #     rho_air_room,
        # )

        (
            passage_thickness_m,
            passage_open_fraction,
            combined_open_fraction,
            face_thickness_m,
            face_covered_area_m2,
            face_covered_fraction,
            frost_flow_ratio,
            evaporator_fan_motor_ready,
            condenser_fan_motor_ready,
            evap_airflow_ok,
        ) = update_fan_states(
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
        )

        # Was called with the wrong args entirely and only 6 of the 9 return
        # values unpacked.
        (
            Q_fans, C_evap, inv_C_evap, C_cond, epsilon_evap, epsilon_cond,
            m_dot_air_evap, inv_m_dot_air_evap, n_ach_actual_per_sec, UA_coil_evap_air, UA_coil_cond_air,
        ) = calculate_airflow_variables(
            room_state_float64_vector,
            forcing_cache_float64_vector,
            room_configurations_float64_vector,
            equipment_configurations_float64_vector,
            active_refrigerant_properties_vector,
            rho_air_room,
            passage_thickness_m,
        )

        update_refrigeration_state(
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
        )

        m_dot_humidifier, Q_humidifier_cooling = calculate_humidifier_variables(
            room_state_int64_vector,
            room_state_float64_vector,
            room_ephemeral_int64_vector,
            humidifier_event_ledger,
            equipment_configurations_float64_vector,
            equipment_configurations_int64_vector,
            active_crop_params_float64_vector,
            dt_engine_sec,
        )

        Q_walls_ext, Q_walls_int, Q_roof = calculate_envelope_heat_loads(
            room_state_float64_vector, forcing_cache_float64_vector, room_configurations_float64_vector,
        )

        Q_room_to_floor, Q_floor_to_ground = calculate_floor_heat_terms(
            room_state_float64_vector, forcing_cache_float64_vector, room_configurations_float64_vector,
        )

        (
            cop, P_evap_bar, P_cond_bar, T_evap_outlet, W_compressor,
            m_dot_refrig_max, valve_flood_frac, n_iters, absolute_error,
        ) = execute_refrigeration_cycle(
            room_state_float64_vector,
            room_state_int64_vector,
            forcing_cache_float64_vector,
            room_configurations_float64_vector,
            room_configurations_int64_vector,
            equipment_configurations_float64_vector,
            equipment_configurations_int64_vector,
            active_refrigerant_properties_vector,
            active_refrigerant_surfaces_matrix,
        )

        (
            Q_sensible_forced, Q_sensible_face, Q_latent,
            m_dot_frost, m_dot_sublimated, m_dot_condensed, m_dot_water_removed,
            T_dew_room, m_dot_melt, Q_melt, E_liquid_excess_j,
            m_dot_face_ice_shed, m_dot_pan_ice_melted,
            Q_coil_defrost_heater_w, Q_pan_defrost_heater_w, Q_pan_to_room_w,
        ) = update_evaporator_state(
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
        )
        if room_state_int64_vector[IDX_ROOM_STATE_DEFROST_PHASE] != DF_NORMAL:
            coil_melted_kg = m_dot_melt * dt_engine_sec
            pan_melted_kg = m_dot_pan_ice_melted * dt_engine_sec
            face_shed_kg = m_dot_face_ice_shed * dt_engine_sec
            room_state_float64_vector[IDX_ROOM_STATE_DEFROST_COIL_MELTED_MASS_ACCUM_KG] += coil_melted_kg
            room_state_float64_vector[IDX_ROOM_STATE_DEFROST_PAN_MELTED_MASS_ACCUM_KG] += pan_melted_kg
            room_state_float64_vector[IDX_ROOM_STATE_DEFROST_MELTED_MASS_ACCUM_KG] += coil_melted_kg + pan_melted_kg
            room_state_float64_vector[IDX_ROOM_STATE_DEFROST_FACE_SHED_MASS_ACCUM_KG] += face_shed_kg

        Q_inf_sens_int, Q_inf_sens_ext, f_exchange_sec, m_dot_leak = calculate_infiltration_terms(
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
        )

        update_sensor_readings(
            rng,
            room_state_int64_vector,
            room_state_float64_vector,
            equipment_configurations_float64_vector,
            equipment_configurations_int64_vector,
            T_evap_outlet,
            P_evap_bar,
            P_cond_bar,
            W_compressor,
        )

        update_floor_temperature(room_state_float64_vector, room_configurations_float64_vector, Q_room_to_floor, Q_floor_to_ground, dt_engine_sec)

        dT_room = update_room_temperature(
            room_state_float64_vector,
            Q_walls_ext, Q_walls_int, Q_roof, Q_room_to_floor, Q_fans,
            Q_room_exchange_bulk, Q_inf_sens_ext, Q_inf_sens_int,
            Q_sensible_forced, Q_sensible_face, Q_humidifier_cooling,
            Q_pan_to_room_w,
            C_room, dt_engine_sec,
        )

        if 1495808 <= k <= 1495815:
            print(
                    "STEP", k,
                    "T_room:", room_state_float64_vector[IDX_ROOM_STATE_AIR_TEMP_C],
                    "dT_room:", dT_room,
                    "Q_bulk_ex:", Q_room_exchange_bulk,
                    "Q_sens:", Q_sensible_forced,
                    "Q_fans:", Q_fans,
                    "Q_env:", Q_walls_ext + Q_walls_int + Q_roof,
                    "C_room:", C_room
            )

        update_condenser_temperature(
            room_state_float64_vector,
            forcing_cache_float64_vector,
            equipment_configurations_float64_vector,
            C_cond,
            epsilon_cond,
        )

        update_fruit_bulk_temperature(
            room_state_float64_vector, C_composite_fruit_bulk,
            Q_batch_to_bulk_exchange, dt_engine_sec,
        )

        update_gas_balance(
            room_state_float64_vector, room_configurations_float64_vector,
            active_crop_params_float64_vector, CO2_prod_ppm_sec_bulk, f_exchange_sec, dt_engine_sec,
        )

        update_moisture_and_fruit_mass(
            room_state_float64_vector, door_states_float64_matrix, forcing_cache_float64_vector,
            m_dot_leak, m_dot_transpiration_bulk, m_dot_humidifier, m_dot_water_removed, inv_m_air_room, dt_engine_sec)

        update_room_and_coil_psychrometrics(
            room_state_float64_vector,
            forcing_cache_float64_vector,
        )

        update_adjacent_plant_states(
            room_state_int64_vector,
            room_state_float64_vector,
            forcing_cache_float64_vector,
            room_configurations_float64_vector,
            dt_plant_ms,
            dt_plant_sec,
        )

        write_output_arrays(
            k,
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
        )
        write_room_telemetry_export_row(
            room_telemetry_matrix,
            room_state_int64_vector,
            room_state_float64_vector,
            # forcing_cache_float64_vector,
            equipment_configurations_int64_vector,
            room_ephemeral_int64_vector,
        )

        if room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_ROOM] == 0 and k % 36000 == 0:
            print(k, room_state_float64_vector[IDX_ROOM_STATE_BULK_PULP_MASS])
        # if room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS] % 60_000 == 0:
        #     print(
        #         "W_ambient from cache vector:", forcing_cache_vector[IDX_CACHE_W_AMBIENT], "\n",
        #         # "P_sat_ambient:", P_sat_ambient, "\n",
        #         # "P_w_ambient:", P_w_ambient, "\n",
        #         "P_ambient from cache vector:", forcing_cache_vector[IDX_CACHE_P_AMBIENT], "\n",
        #     )

        # if room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS] % 60_000 == 0 and k >= 144_000:  # Log once per simulated minute to avoid flooding
        #     print("Time MS:", room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS])
        #     print("Queue Read:", room_state_int64_vector[IDX_ROOM_STATE_ARRIVAL_WAVE_READ_PTR])
        #     print("Queue Write:", room_state_int64_vector[IDX_ROOM_STATE_ARRIVAL_WAVE_WRITE_PTR])

        ### DEBUG ###
        # if room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS] % 1000 == 0:
        #     if 144000 <= k <= 180000:
        #         print(
        #             k,
        #             room_state_int64_vector[IDX_ROOM_STATE_TOTAL_BATCHES_IN_ROOM],
        #             room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_NO_FRUIT_ACTIVITY],
        #             room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_SCHEDULED],
        #             room_ephemeral_int64_vector[IDX_ROOM_EPHEMERAL_DOWNTIME_SAFE_FOR_SHUTDOWN],
        #             room_state_int64_vector[IDX_ROOM_STATE_REFRIGERATION_IS_ACTIVE],
        #             room_state_float64_vector[IDX_ROOM_PHYSICS_V_DOT_AIR_EVAP_ACTUAL],
        #             room_state_float64_vector[IDX_ROOM_PHYSICS_T_ROOM_FILTERED_VAL],
        #             room_state_float64_vector[IDX_ROOM_PHYSICS_RH_ROOM],
        #             room_state_float64_vector[IDX_ROOM_PHYSICS_RH_ROOM_FILTERED_VAL],
        #             room_state_float64_vector[IDX_ROOM_PHYSICS_AIR_HUMIDITY_RATIO_W],
        #             room_state_float64_vector[IDX_ROOM_PHYSICS_AIR_TEMP_C],
        #             room_state_float64_vector[IDX_ROOM_PHYSICS_P_W_ROOM],
        #         )

        ### DEBUG ###
        # ### DEBUG ###
        # if 150320 <= k <= 150331:
        #     print(k, Q_walls_ext, Q_walls_int, Q_roof, Q_room_to_floor, Q_fans,
        #           Q_room_exchange_bulk, Q_inf_sens_ext, Q_inf_sens_int,
        #           room_state_float64_vector[IDX_ROOM_PHYSICS_BULK_ACTIVE_DRIVERS_CONVECTIVE_Q],
        #           Q_sensible, Q_humidifier_cooling, C_room)
        # ### DEBUG ###

        room_state_int64_vector[IDX_ROOM_STATE_EPOCH_TIMESTAMP_MS] += dt_engine_ms

    if room_state_int64_vector[IDX_ROOM_STATE_IS_FIRST_RUN_FLAG]:
        room_state_int64_vector[IDX_ROOM_STATE_IS_FIRST_RUN_FLAG] = 0
