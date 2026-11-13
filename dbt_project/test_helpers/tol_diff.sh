#!/usr/bin/env bash
# Compare <model> with snap_<model>, tolerating float noise (BigQuery sums floats in a varying order,
# so avg()/sum() columns can differ in the last bits between runs of the same query).
#   - every NON-float column must match exactly (EXCEPT DISTINCT both ways)
#   - every FLOAT64 column must match within a tolerance: abs(a-b) <= tol * max(|a|, |b|, 1); null vs value counts as a mismatch
#   - rows present on one side only are counted
# Every number printed must be 0 (max_err must be <= tol).
# Usage: ./tol_diff.sh <model> <key columns, comma separated> [--except "colA,colB"] [--tol 1e-9]
#   keys must identify a row (compared as strings, null = null, so null keys are fine)
source "$(dirname "$0")/_common.sh"
[ $# -ge 2 ] || { sed -n 2,11p "$0"; exit 1; }
M="$1"; KEYS="$2"; shift 2
EXCEPT=""; TOL="1e-9"
while [ $# -gt 0 ]; do
  case "$1" in
    --except) EXCEPT="$2"; shift 2;;
    --tol)    TOL="$2";    shift 2;;
    *) echo "unknown option $1"; exit 1;;
  esac
done

# FLOAT64 columns of the model, minus the excepted ones
FLOATS=""
for c in $(bqq "select string_agg(column_name order by ordinal_position) from ${FQ}.INFORMATION_SCHEMA.COLUMNS
                where table_name = '${M}' and data_type = 'FLOAT64'" | tr -d '"' | tr ',' ' '); do
  case ",${EXCEPT}," in *",${c},"*) ;; *) FLOATS="${FLOATS:+$FLOATS,}$c";; esac
done
echo "float columns compared with tolerance ${TOL}: ${FLOATS:-none}"

EX_ALL="${EXCEPT}"
[ -n "$FLOATS" ] && EX_ALL="${EX_ALL:+$EX_ALL,}${FLOATS}"
if [ -n "$EX_ALL" ]; then SEL="select * except (${EX_ALL})"; else SEL="select *"; fi

for pair in "$M snap_$M" "snap_$M $M"; do
  set -- $pair
  echo -n "$1 minus $2 (non-float columns, rows): "
  bqq "select count(*) from (${SEL} from $(TBL $1) except distinct ${SEL} from $(TBL $2))"
done

if [ -n "$FLOATS" ]; then
  # keys compared as strings with a sentinel for null (null = null). No FULL OUTER JOIN: BigQuery only allows
  # plain field equalities there, so "only on one side" is counted with NOT EXISTS
  ON=""; for k in ${KEYS//,/ }; do ON="${ON:+$ON and }ifnull(cast(a.${k} as string), '<null>') = ifnull(cast(b.${k} as string), '<null>')"; done
  MISMATCH=""; ERR=""
  for f in ${FLOATS//,/ }; do
    MISMATCH="${MISMATCH:+$MISMATCH or }((a.${f} is null) != (b.${f} is null)) or abs(a.${f} - b.${f}) > ${TOL} * greatest(abs(a.${f}), abs(b.${f}), 1.0)"
    ERR="${ERR:+$ERR, }ifnull(abs(a.${f} - b.${f}) / greatest(abs(a.${f}), abs(b.${f}), 1.0), 0)"
  done
  A="$(TBL ${M})"; B="$(TBL snap_${M})"
  echo "rows only in snapshot / only in table / float mismatches / max_err:"
  bqq "select
         (select count(*) from ${B} as b where not exists (select 1 from ${A} as a where ${ON})),
         (select count(*) from ${A} as a where not exists (select 1 from ${B} as b where ${ON})),
         (select countif(${MISMATCH}) from ${A} as a join ${B} as b on ${ON}),
         (select max(greatest(${ERR})) from ${A} as a join ${B} as b on ${ON})"
fi
