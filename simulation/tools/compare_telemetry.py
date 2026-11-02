"""
Compare two telemetry parquet files (before / after a code change) row for row.

    python tools/compare_telemetry.py baseline_telemetry.parquet new_telemetry.parquet

Both runs must start from the same state.npz / rng_state.json and the same config, otherwise
the noise draws differ and the values legitimately differ. Exit code 0 = identical.
"""
import sys
import pandas as pd

a = pd.read_parquet(sys.argv[1])
b = pd.read_parquet(sys.argv[2])
print(f"rows: {len(a)} vs {len(b)}")
print("columns equal:", list(a.columns) == list(b.columns))
key = [c for c in ("timestamp_ms", "event_timestamp", "location_id", "tag_name") if c in a.columns and c in b.columns]
a = a.sort_values(key).reset_index(drop=True)
b = b.sort_values(key).reset_index(drop=True)
same = a.shape == b.shape and a.equals(b)
print("identical:", same)
if not same and a.shape == b.shape:
    diff = (a != b) & ~(a.isna() & b.isna())
    print("differing cells per column:\n", diff.sum()[diff.sum() > 0])
sys.exit(0 if same else 1)
