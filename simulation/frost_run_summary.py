import numpy as np
from constants import *
from matrix_index_maps import *
from run_outputs_io import load_run_outputs

out = load_run_outputs("run_outputs")
f, i = out["output_float64_matrix"], out["output_int64_matrix"]
t0 = i[0, IDX_OUTPUT_EPOCH_TIMESTAMP_MS]
hrs = (i[:, IDX_OUTPUT_EPOCH_TIMESTAMP_MS] - t0) / MS_PER_HOUR
step = 72000  # 2 simulated hours at 0.1 s
print(f"{'h':>5} {'duty':>5} {'V_evap':>7} {'M_frost':>8} {'T_coil':>7} {'T_evap':>7} {'T_room':>7}")
for k in range(0, len(f), step):
    s = slice(k, k + step)
    print(f"{hrs[k]:5.1f} {i[s, IDX_OUTPUT_REFRIGERATION_IS_ACTIVE].mean():5.2f} {f[s, IDX_OUTPUT_V_DOT_AIR_EVAP_ACTUAL].mean():7.2f} "
          f"{f[s, IDX_OUTPUT_M_FROST_KG].max():8.2f} {f[s, IDX_OUTPUT_T_COIL_EVAP_C].mean():7.1f} "
          f"{f[s, IDX_OUTPUT_T_EVAP_C].mean():7.1f} {f[s, IDX_OUTPUT_T_ROOM_C].mean():7.1f}")
run = hrs > 2.0
print("NaN anywhere:", np.isnan(f).any(), "| min evaporator airflow after 2 h:", f[run, IDX_OUTPUT_V_DOT_AIR_EVAP_ACTUAL].min().round(3),
      "| peak M_frost:", np.nanmax(f[:, IDX_OUTPUT_M_FROST_KG]).round(2))
