import json
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from numba import njit, types, typeof
from numba.typed import Dict
import math
from constants import SAFE_FLOOR
import pycountry
import reverse_geocoder as rg

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # Converts full numpy arrays to standard nested Python lists
        if isinstance(obj, np.ndarray):
            return obj.tolist() 
        # Converts numpy data types/scalars (like np.int64) to native Python types
        if isinstance(obj, (np.void, np.number)):
            return obj.item()
        return super().default(obj)

def get_country_details(lat, lng):
    result = rg.search([(lat, lng)])[0]
    cc = result['cc']  # e.g., 'BR'
    
    # Look up full name safely
    country = pycountry.countries.get(alpha_2=cc)
    full_name = country.name if country else "Unknown Country"
    
    return {"code": cc, "name": full_name}

def get_persistent_session():
    session = requests.Session()
    # Retry on 504, 502, 503, 500
    retry_strategy = Retry(
        total=5, 
        backoff_factor=2,
        status_forcelist=[500, 502, 503],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    return session

def create_typed_dict(
    key_type=types.unicode_type,
    value_type=types.float64,
    value_structure='scalar',
):
    if value_structure == 'scalar':
        return Dict.empty(
            key_type=key_type,
            value_type=value_type,
        )
    elif value_structure == 'array':
        return Dict.empty(
            key_type=key_type,
            value_type=value_type[:],
        )
    else:
        raise ValueError("Enter a valid data structure: 'scalar' or 'array'")

def plain_dict_to_typed_dict(plain_dict, empty_typed_dict):
    dict_type = typeof(empty_typed_dict)
    cast_key = dict_type.key_type # .get_python_type()
    cast_value = dict_type.value_type # .get_python_type()
    for key, value in plain_dict.items():
        empty_typed_dict[cast_key(key)] = cast_value(value)
    return empty_typed_dict

def typed_dict_to_plain_dict(typed_dict):
    return {k: typed_dict[k] for k in typed_dict}

def clean_rng_state(obj):
    """Recursively converts NumPy ints/arrays into JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: clean_rng_state(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Converts uint32 arrays into [1, 2, 3] lists
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()    # Converts numpy scalars into native int/float
    return obj

def get_list_element_type(lst):
    """Recursively finds the type of the underlying data inside a nested list."""
    if isinstance(lst, list) and len(lst) > 0:
        return get_list_element_type(lst[0])
    return type(lst)

def process_kestra_payload(kestra_payload_dict):
    rng = np.random.default_rng()
    rng.bit_generator.state = kestra_payload_dict["rng_state"]

    data_states = kestra_payload_dict["data_states"]
    data_types = kestra_payload_dict["_types"]

    simulation_state = {}

    for key, dtype_str in data_types.items():
        np_scalar_type = getattr(np, dtype_str)

        raw_value = data_states[key]

        if isinstance(raw_value, list):
            simulation_state[key] = np.array(raw_value, dtype=np_scalar_type)
        else: 
            simulation_state[key] = np_scalar_type(raw_value)

    return simulation_state, rng

@njit
def wrapped_index(index, size):
    # double division to account for negative indices
    return ((index % size) + size) % size

@njit
def concat_datetime_to_int(year, month, day, hour, minute, second):
    # Combines components into a single YYYYMMDDHHMMSS fixed-length integer
    fixed_len_int = (
        (year * 10_000_000_000)
        + (month * 100_000_000)
        + (day * 1_000_000)
        + (hour * 10_000)
        + (minute * 100)
        + second
    )
    return fixed_len_int

@njit
def sort_both_axes(arr):
    out = arr.copy()
    rows, cols = out.shape
    for i in range(rows):
        out[i, :] = np.sort(out[i, :])
    for j in range(cols):
        out[:, j] = np.sort(out[:, j])
    return out

@njit
def fast_vstack_numba(matrix_tuple):
    total_rows = 0
    for m in matrix_tuple:
        total_rows += m.shape[0]

    cols = matrix_tuple[0].shape[1]
    out = np.empty((total_rows, cols), dtype=matrix_tuple[0].dtype)

    current_row = 0
    for m in matrix_tuple:
        h = m.shape[0]
        out[current_row : current_row + h, :] = m
        current_row += h

    return out

class OldNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# --- Absorbed from the old simulator.py / physics_helpers.py split ---
# These are generic curve-fit/clip/regression math with no domain knowledge
# (no room state, no fruit, no refrigerant) -- same tier as sort_both_axes /
# fast_vstack_numba above, so they belong here rather than in a one-off file.

@njit(cache=True)
def numba_clip(value, low, high):
    if value < low:
        return low
    elif value > high:
        return high
    else:
        return value


@njit(cache=True)
def least_squares_regression(x, y):
    # --- Manual Replacement for np.polyfit(x, y, 1) ---
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x*x)
    sum_xy = np.sum(x*y)

    denominator = (n * sum_xx - sum_x**2)

    if denominator == 0:
        m, b = 0.0, 0.0  # Avoid division by zero for single points or identical X values
    else:
        m = (n * sum_xy - sum_x * sum_y) / denominator
        b = (sum_y - m * sum_x) / n
    return m, b

@njit(cache=True)
def least_squares_regression_through_origin(x, y):
    return np.sum(x * y) / max(np.sum(x * x), SAFE_FLOOR)

@njit(cache=True)
def evaluate_poly1d_fit(
    x,
    c,
    bounds,
    clip_values=True,
    # sentinel=SENTINEL_FLOAT64,
):
    x_min, x_max = bounds[:2]

    if x_min == x_max: # Placeholder for no bounds
        x_norm = x
    else:
        if clip_values:
            x = numba_clip(x, x_min, x_max)
        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1

    y_pred = 0
    for d, c_term in enumerate(c):
        # if c_term == sentinel:
        #     break
        y_pred += c_term * (x_norm ** d)
    return y_pred


@njit(cache=True)
def evaluate_poly2d_quadratic_fit(
    x1,
    x2,
    c,
    bounds,
    clip_values=True,
):
    x1_min, x1_max, x2_min, x2_max = bounds[:4]

    if clip_values:
        x1 = numba_clip(x1, x1_min, x1_max)
        x2 = numba_clip(x2, x2_min, x2_max)

    x1_norm = 2 * (x1 - x1_min) / (x1_max - x1_min) - 1
    x2_norm = 2 * (x2 - x2_min) / (x2_max - x2_min) - 1

    y_pred = (
        c[0] +
        c[1] * x1_norm + c[2] * x2_norm +
        c[3] * x1_norm**2 + c[4] * x2_norm**2 + c[5] * x1_norm * x2_norm
    )
    return y_pred


@njit(cache=True)
def evaluate_poly2d_cubic_fit(
    x1,
    x2,
    c,
    bounds,
    clip_values=True,
):
    x1_min, x1_max, x2_min, x2_max = bounds[:4]
   
    if clip_values:
        x1 = numba_clip(x1, x1_min, x1_max)
        x2 = numba_clip(x2, x2_min, x2_max)

    x1_norm = 2 * (x1 - x1_min) / (x1_max - x1_min) - 1
    x2_norm = 2 * (x2 - x2_min) / (x2_max - x2_min) - 1

    y_pred = (
        c[0] +
        c[1] * x1_norm + c[2] * x2_norm +
        c[3] * x1_norm**2 + c[4] * x2_norm**2 + c[5] * x1_norm * x2_norm +
        c[6] * x1_norm**3 + c[7] * x2_norm**3 + c[8] * (x1_norm**2) * x2_norm + c[9] * x1_norm * (x2_norm**2)
    )
    return y_pred


@njit(cache=True)
def evaluate_poly3d_quadratic_fit(
    x1,
    x2,
    x3,
    c,
    bounds,
    clip_values=True,
):
    x1_min, x1_max, x2_min, x2_max, x3_min, x3_max = bounds[:6]
    
    if clip_values:
        x1 = numba_clip(x1, x1_min, x1_max)
        x2 = numba_clip(x2, x2_min, x2_max)
        x3 = numba_clip(x3, x3_min, x3_max)

    x1_norm = 2 * (x1 - x1_min) / (x1_max - x1_min) - 1
    x2_norm = 2 * (x2 - x2_min) / (x2_max - x2_min) - 1
    x3_norm = 2 * (x3 - x3_min) / (x3_max - x3_min) - 1
    
    # Match the order: [1, x1, x2, x3, x1^2, x2^2, x3^2, x1*x2, x1*x3, x2*x3]
    y_pred = (
        c[0] +
        c[1] * x1_norm + c[2] * x2_norm + c[3] * x3_norm +
        c[4] * (x1_norm**2) + c[5] * (x2_norm**2) + c[6] * (x3_norm**2) +
        c[7] * (x1_norm * x2_norm) + c[8] * (x1_norm * x3_norm) + c[9] * (x2_norm * x3_norm)
    )
    return y_pred


@njit(cache=True)
def evaluate_antoine_fit(
    T_c,
    A_antoine,
):
    log_P_bar = A_antoine[0] + A_antoine[1]/(T_c + A_antoine[2])
    return log_P_bar


@njit(cache=True)
def invert_antoine_fit(
    log_P_bar,
    A_antoine,
):
    T_c = A_antoine[1]/(log_P_bar - A_antoine[0]) - A_antoine[2]
    return T_c


@njit(cache=True)
def gamma_draw(
    rng,
    mean,
    shape,
):
    scale = mean / shape
    return float(rng.gamma(shape, scale))

@njit
def round_up_to_multiple(
    number,
    multiple,
):
    return int(math.ceil(number / multiple) * multiple)

@njit(cache=True)
def erfinv(x):
    if x >= 1.0:
        return 5.0  # Safe high Z-score ceiling truncation
    if x <= -1.0:
        return -5.0 # Safe low Z-score floor truncation
    
    # Winitzki approximation
    a = 0.14705
    log_term = math.log(1.0 - x**2)
    term1 = 2.0 / (math.pi * a) + log_term / 2.0
    inner_sqrt = term1**2 - log_term / a

    radicand = math.sqrt(inner_sqrt) - term1
    if radicand < 0.0:
        radicand = 0.0

    val = math.sqrt(radicand)
    
    if x >= 0.0:
        return val
    else:
        return -val

@njit
def draw_single_box_net_weight(
    rng,
    mean_pack,
    std_pack,
    legal_minimum,
):
    limit_z = (legal_minimum - mean_pack) / (std_pack * math.sqrt(2.0))

    cdf_floor = 0.5 * (1.0 + math.erf(limit_z))

    u = rng.random()
    sampled_cdf = cdf_floor + u * (1.0 - cdf_floor)

    z_sampled = math.sqrt(2.0) * erfinv(2.0 * sampled_cdf - 1.0)

    box_net_weight = legal_minimum + std_pack * z_sampled

    return box_net_weight

@njit
def bounded_normal_draw(
    rng,
    mean,
    scale,
    low,
    high,
):
    # 1. Convert physical bounds into standard normal Z-scores
    low_z = (low - mean) / (scale * math.sqrt(2.0))
    high_z =(high - mean) / (scale * math.sqrt(2.0))

    # 2. Calculate the exact CDF probability window boundaries using the error function
    cdf_low = 0.5 * (1.0 + math.erf(low_z))
    cdf_high = 0.5 * (1.0 + math.erf(high_z))

    # 3. Sample a flat uniform line strictly within this legal probability window
    u = rng.random()
    sampled_cdf = cdf_low + u * (cdf_high - cdf_low)

    # 4. Map the uniform sample back to the normal distribution scale via the Inverse CDF
    z_sampled = math.sqrt(2.0) * erfinv(2.0 * sampled_cdf - 1.0)

    return mean + z_sampled * scale

@njit
def draw_tunnel_exit_fruit_temp(
    rng,
    temp_ref,
    temp_drift,
    temp_scale,
    max_decrease,
    max_increase,
):
    mean_val = temp_ref + temp_drift

    # 1. Convert physical bounds into standard normal Z-scores
    low_z = (-max_decrease - temp_drift) / (temp_scale * math.sqrt(2.0))
    high_z = (max_increase - temp_drift) / (temp_scale * math.sqrt(2.0))

    # 2. Calculate the exact CDF probability window boundaries using the error function
    cdf_low = 0.5 * (1.0 + math.erf(low_z))
    cdf_high = 0.5 * (1.0 + math.erf(high_z))

    # 3. Sample a flat uniform line strictly within this legal probability window
    u = rng.random()
    sampled_cdf = cdf_low + u * (cdf_high - cdf_low)

    # 4. Map the uniform sample back to the normal distribution scale via the Inverse CDF
    z_sampled = math.sqrt(2.0) * erfinv(2.0 * sampled_cdf - 1.0)

    # 5. Scale back to true temperature degrees celsius
    incoming_temperature = mean_val + z_sampled * temp_scale
    
    return incoming_temperature

