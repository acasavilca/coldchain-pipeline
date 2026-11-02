"""
workforce_behavior.py

Human-behavior model for workers/drivers (fatigue, carelessness, error probability) plus the sampling primitives (gamma/normal draws, erfinv) that only exist to serve it.
"""

import math
import numpy as np
from numba import njit

from constants import *
from matrix_index_maps import *
from utils import numba_clip

@njit
def calculate_actual_carelessness_factor(base_carelessness, fatigue_factor, fatigue_sensitivity):
    return base_carelessness * (1.0 + (fatigue_factor - 1.0) * fatigue_sensitivity)

@njit
def compute_probability_of_error(base_carelessness, baseline_p_error, fatigue_factor, fatigue_sensitivity, error_sensitivity): # compute_probability_of_error(base_carelessnes, baseline_p=0.05, error_sensitivity=1.5):
    """
    Maps a physical slacking factor (1.0 to 2.0+) safely into a bounded probability (0.0 to 1.0).
    - baseline_p: The error rate of a flawless, perfect driver (factor = 1.0).
    - sensitivity: How aggressively the error rate scales as slacking increases.
    """
    # Scale carelessness by time-dependent fatigue
    carelessness_factor = calculate_actual_carelessness_factor(base_carelessness, fatigue_factor, fatigue_sensitivity)

    # Shift factor so 1.0 baseline evaluates starting from 0.0
    delta_slacking = max(0.0, carelessness_factor - 1.0)
    
    # Logistic sigmoid mapping function
    p_error = baseline_p_error + (1.0 - baseline_p_error) * (1.0 - math.exp(-error_sensitivity * delta_slacking))
    return numba_clip(p_error, 0.0, 0.99)

@njit
def calculate_fatigue_factor(
    hour,
    day_of_week,
    is_last_weekday,
):
    # 1. Start with a baseline fatigue factor of 1.0 (Standard performance)
    f_fatigue = 1.0

    # 2. Add an Hour-of-Day Shift Penalty (The Circadian Dip)
    if hour >= 22 or hour <= 4:
        f_fatigue += 0.15  # Sluggishness increases by 15% at night
    elif hour >= 13 and hour <= 15:
        f_fatigue += 0.05 # Standard post-lunch food coma lull

    # 3. Add a Day-of-the-Week Fatigue Penalty
    # (0 = Sunday, 1 = Monday, 2 = Tuesday, 3 = Wednesday, 4 = Thursday, 5 = Friday, 6 = Saturday)
    # Fridays and weekends see an operational surge in employee distractions
    if day_of_week == 5:   # Friday
        f_fatigue += 0.10  # Distractions climb as the weekend approaches
    elif day_of_week == 6 or day_of_week == 0:
        f_fatigue += 0.18  # Skeleton crews experience severe fatigue spikes

    # 4. Add an End-of-Month Rush Modifier
    # During the last week of the fiscal month, drivers rush to hit inventory targets, dropping safety standards
    if is_last_weekday:
        f_fatigue += 0.12
    return f_fatigue

# NOTE: For future implementation
@njit
def select_driver(
    rng,
    driver_busy_until_matrix_ms,
    free_drivers_pool,
    target_start_ms,
    num_drivers,
    sentinel=SENTINEL_INT64,
):
    available_count = 0
    must_wait_for_driver = 0
    free_drivers_pool[...] = SENTINEL_INT64
    for d_idx in range(num_drivers):
        if driver_busy_until_matrix_ms[d_idx] <= target_start_ms:
            free_drivers_pool[available_count] = d_idx
            available_count += 1
    if available_count > 0:
        # CASE A: Drivers are available -> Randomly sample among the free ones
        random_pool_idx = rng.integers(0, available_count)
        active_driver_code = free_drivers_pool[random_pool_idx]
    else:
        # CASE B: No drivers are free -> The batch MUST move anyway.
        # Find the driver who will become free earliest to minimize the overlap violation.
        must_wait_for_driver = 1
        earliest_free_time = MAX_EPOCH_TIMESTAMP_MS
        fallback_driver = 0
        
        for d_idx in range(num_drivers):
            if driver_busy_until_matrix_ms[d_idx] < earliest_free_time:
                earliest_free_time = driver_busy_until_matrix_ms[d_idx]
                fallback_driver = d_idx
                
        active_driver_code = fallback_driver
    return active_driver_code, must_wait_for_driver 
