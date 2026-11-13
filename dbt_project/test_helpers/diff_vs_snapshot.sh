#!/usr/bin/env bash
# Compare <model> with snap_<model>: row diffs both ways (EXCEPT DISTINCT) and per-day count/hash mismatches.
# Every number printed must be 0.
# Usage: ./diff_vs_snapshot.sh <model> [partition_col] [--except "colA,colB"] [--round "colC,colD"] [--digits 6]
#   partition_col defaults to the table's partitioning column
#   --except  columns left out of the comparison (e.g. dbt_updated_at)
#   --round   float columns rounded to --digits (default 6) before comparing
source "$(dirname "$0")/_common.sh"
[ $# -ge 1 ] || { sed -n 2,9p "$0"; exit 1; }
M="$1"; shift
PCOL=""; EXCEPT=""; ROUND=""; DIGITS=6
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then PCOL="$1"; shift; fi
while [ $# -gt 0 ]; do
  case "$1" in
    --except) EXCEPT="$2"; shift 2;;
    --round)  ROUND="$2";  shift 2;;
    --digits) DIGITS="$2"; shift 2;;
    *) echo "unknown option $1"; exit 1;;
  esac
done
[ -n "$PCOL" ] || PCOL="$(partition_col_of "$M")"
[ -n "$PCOL" ] || { echo "could not detect partition column for $M; pass it explicitly"; exit 1; }

EX_ALL="$EXCEPT"
RD_EXPR=""
if [ -n "$ROUND" ]; then
  EX_ALL="${EX_ALL:+$EX_ALL,}$ROUND"
  for c in ${ROUND//,/ }; do RD_EXPR="${RD_EXPR}, round(${c}, ${DIGITS}) as ${c}"; done
fi
if [ -n "$EX_ALL" ]; then SEL="select * except (${EX_ALL})${RD_EXPR}"; else SEL="select *"; fi

for pair in "$M snap_$M" "snap_$M $M"; do
  set -- $pair
  echo -n "$1 minus $2 (rows): "
  bqq "select count(*) from (${SEL} from $(TBL $1) except distinct ${SEL} from $(TBL $2))"
done
echo -n "$M mismatched days (count/hash, by ${PCOL}): "
bqq "with a as (select cast(${PCOL} as date) d, count(*) n, bit_xor(farm_fingerprint(to_json_string(t))) h from (${SEL} from $(TBL ${M})) t group by 1),
          b as (select cast(${PCOL} as date) d, count(*) n, bit_xor(farm_fingerprint(to_json_string(t))) h from (${SEL} from $(TBL snap_${M})) t group by 1)
     select count(*) from a full outer join b on a.d = b.d
     where a.d is null or b.d is null or a.n != b.n or a.h != b.h"
