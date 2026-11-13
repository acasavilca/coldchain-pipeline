#!/usr/bin/env bash

# 1. Set default values
LOCATIONS="1 2 3 4 5 6 7 8 9 10"
MAX_PARALLEL=4
START_DATE="2025-04-30T19:00:00"
DAYS=1
ITERATIONS=30

# Helper function for --help
usage() {
    echo "Usage: $0 [options]"
    echo "  -l, --locations \"1 2 3\"   Space-separated location IDs (default: \"$LOCATIONS\")"
    echo "  -p, --parallel N          Max parallel processes (default: $MAX_PARALLEL)"
    echo "  -s, --start-date DATE     Start date ISO string (default: $START_DATE)"
    echo "  -d, --days N              Days per run (default: $DAYS)"
    echo "  -i, --iterations N        Number of sequential iterations (default: $ITERATIONS)"
    echo "  -h, --help                Show this help message"
    exit 0
}

# 2. Parse command-line flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--locations)
            LOCATIONS="$2"
            shift 2
            ;;
        -p|--parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        -s|--start-date)
            START_DATE="$2"
            shift 2
            ;;
        -d|--days)
            DAYS="$2"
            shift 2
            ;;
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

mkdir -p logs

# 3. Export variables so the inner bash -c subshells can read them
echo "Running locations [$LOCATIONS] with max $MAX_PARALLEL parallel jobs..."

echo "$LOCATIONS" | tr ' ' '\n' | xargs -P "$MAX_PARALLEL" -I {} bash -c '
    loc="$1"
    echo "Starting location $loc"
    for i in $(seq 1 "$ITERATIONS"); do
        python run.py --location "$loc" --start-date "$START_DATE" --days "$DAYS" --no-run-outputs || {
            echo "Location $loc failed on iteration $i"
            break
        }
    done
    echo "Finished location $loc"
' _ {} > logs/all_runs.log 2>&1

echo "Done! Logs saved to logs/all_runs.log"
