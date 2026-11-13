#!/usr/bin/env bash
# Show how ingested_at is spread in a table (min, p20, p50, p80, max) to pick a cutoff for a slice test.
# Usage: ./ingest_spread.sh <table> [<table> ...]
source "$(dirname "$0")/_common.sh"
[ $# -ge 1 ] || { echo "usage: $0 <table> [<table> ...]"; exit 1; }
for t in "$@"; do
  echo "== $t"
  bq query --use_legacy_sql=false --format=pretty --quiet \
    "select min(ingested_at) mn,
            approx_quantiles(ingested_at, 10)[offset(2)] p20,
            approx_quantiles(ingested_at, 10)[offset(5)] p50,
            approx_quantiles(ingested_at, 10)[offset(8)] p80,
            max(ingested_at) mx, count(*) n
     from $(TBL ${t})" 2>/dev/null
done
