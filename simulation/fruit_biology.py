"""
fruit_biology.py

Fruit respiration curves and quality-grade sampling.
"""

import math
import numpy as np
import pandas as pd
from jit import njit

from constants import *
from matrix_index_maps import *
from utils import least_squares_regression, numba_clip


def get_respiration_curve(respiration_database, fruit_type):
    df = (
        pd.DataFrame(respiration_database["data"])
        .T
        .rename_axis("fruit")
        .reset_index()
    )

    df.columns = ["fruit"] + [int(c) for c in df.columns[1:]]

    df_long = df.melt(
        id_vars="fruit",
        var_name="temperature_C",
        value_name="respiration"
    )

    sub = df_long[df_long["fruit"] == fruit_type]

    if sub.empty:
        raise ValueError(f"Fruit {fruit_type} not found in respiration DB")

    else:
        T_lookup = pd.to_numeric(sub["temperature_C"].values, errors="coerce")
        R_lookup = pd.to_numeric(sub["respiration"].values, errors="coerce")

    mask = ~np.isnan(R_lookup)

    return np.asarray(T_lookup[mask], dtype=float), np.asarray(R_lookup[mask], dtype=float)


@njit(fastmath=False)
def get_t_ref(T, T_min, T_max):
    mid = (T_min + T_max) / 2

    # Initialize with the first element
    best_t = T[0]
    best_diff = abs(best_t - mid)

    for i in range(1, len(T)):
        t = T[i]
        diff = abs(t - mid)

        # This mimics the (diff, t) tuple comparison logic
        if diff < best_diff:
            best_diff = diff
            best_t = t
        elif diff == best_diff:
            if t < best_t:
                best_t = t

    return best_t


@njit
def get_fruit_respiration_parameters(T_lookup, R_lookup):
    order = np.argsort(T_lookup)
    T = T_lookup[order]
    R = R_lookup[order]
    T_max = np.max(T)
    T_min = np.min(T)
    buffer = 0.5*np.median(np.diff(T)) if len(T) > 1 else 0.0
    T_lo = T_min - buffer
    T_hi = T_max + buffer

    T_ref = get_t_ref(T, T_min, T_max)
    x = T - T_ref
    y = np.log(R)

    m, b = least_squares_regression(x, y)

    Q_10 = math.exp(10*m)
    R_ref = math.exp(b)

    return T, R, T_min, T_max, T_lo, T_hi, T_ref, Q_10, R_ref


@njit
def calc_respiration_rate_mg_co2_per_kg_fruit_hr(T_pulp, T, R, T_min, T_max, T_lo, T_hi, T_ref, R_ref, Q_10):
    idx = np.searchsorted(T, T_pulp)

    if idx < len(T) and T[idx] == T_pulp:
        R_fruit = R[idx]
    elif idx > 0 and T[idx - 1] == T_pulp:
        R_fruit = R[idx - 1]
    elif T_pulp < T_min or T_pulp > T_max:
        T_pulp_eff = numba_clip(T_pulp, T_lo, T_hi)
        R_fruit = R_ref * Q_10**((T_pulp_eff - T_ref) / 10.0)
    else:
        T_1, T_2 = T[idx - 1], T[idx]
        R_1, R_2 = R[idx - 1], R[idx]
        R_fruit = R_1 * (R_2 / R_1)**((T_pulp - T_1) / (T_2 - T_1))

    return R_fruit


@njit
def draw_fruit_quality_grade(
    seasonal_weight_factor,
    base_quality_probabilities,
    quality_sensitivity_multiplier,
    active_fruit_type_code,
    rng,
):
    num_grades = base_quality_probabilities.shape[0]
    scaled_probabilities = np.empty(num_grades, dtype=np.float64)
    is_avocado = (active_fruit_type_code == FRUIT_TYPE_AVOCADO)

    # AVOCADO LOGIC: Invert the factor relative to the baseline of 1.0
    # low-yield year means the tree carries fewer fruits, so it pumps all its nutrients into them, making them larger and higher quality (Premium)
    # If factor is 0.60 (low yield), it becomes 1.40 (high quality)
    # If factor is 1.40 (high yield), it becomes 0.60 (lower quality)
    effective_factor = max(2.0 - seasonal_weight_factor, 0.1) if is_avocado else seasonal_weight_factor
    
    # If seasonal_weight_factor < 1 (Bad season), the exponent shifts mass to higher indices (lower quality)
    # If seasonal_weight_factor > 1 (Good season), the exponent shifts mass to lower indices (higher quality)
    # exponent = 1.0 / max(0.1, seasonal_weight_factor * quality_sensitivity_multiplier)
    # Good season (>1) creates a negative delta, Bad season (<1) creates a positive delta
    delta = (1.0 - effective_factor) * quality_sensitivity_multiplier

    total_sum = 0.0
    for idx in range(num_grades):
        # The exponent scales linearly by index:
        # If Good season (delta is negative): high indices get a LARGER exponent, suppressing poor quality.
        # If Bad season (delta is positive): high indices get a SMALLER exponent, boosting poor quality.
        if is_avocado:
            exponent = max(0.1, 1.0 - idx * delta)
        else:
            exponent = max(0.01, (1.0 - delta) ** idx)
        scaled_probabilities[idx] = base_quality_probabilities[idx] ** exponent
        total_sum += scaled_probabilities[idx]

    # Normalize
    for idx in range(num_grades):
        scaled_probabilities[idx] /= total_sum

    # Clamp
    # no grade can end up more likely than the grade above it
    for idx in range(1, num_grades):
        if scaled_probabilities[idx] > scaled_probabilities[idx-1]:
            scaled_probabilities[idx] = scaled_probabilities[idx-1]

    # Categorical Inverse Transform Sampling Draw
    u = rng.random()
    cumulative_sum = 0.0
    sampled_grade = num_grades - 1

    for idx in range(num_grades):
        cumulative_sum += scaled_probabilities[idx]
        if u < cumulative_sum:
            sampled_grade = idx
            break

    return sampled_grade

@njit
def compute_market_scarcity_matrix(
    rng,
    historical_yield_matrix,
    local_market_correlation,
    market_volatility_scale,
):
    independent_market_noise = rng.lognormal(mean=0.0, sigma=market_volatility_scale, size=historical_yield_matrix.shape)
    return (
        local_market_correlation * historical_yield_matrix
        + (1.0 - local_market_correlation) * independent_market_noise
    )
