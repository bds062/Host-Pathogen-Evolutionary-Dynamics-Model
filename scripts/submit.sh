#!/bin/bash
# Convenience wrapper that applies the SB_* environment overrides as sbatch flags.
#   SB_ACCOUNT=myacct SB_PARTITION=general ./scripts/submit.sh 3_4
set -euo pipefail
ARGS=()
[ -n "${SB_ACCOUNT:-}" ]   && ARGS+=(--account="$SB_ACCOUNT")
[ -n "${SB_PARTITION:-}" ] && ARGS+=(--partition="$SB_PARTITION")
[ -n "${SB_QOS:-}" ]       && ARGS+=(--qos="$SB_QOS")
[ -n "${SB_TIME:-}" ]      && ARGS+=(--time="$SB_TIME")
[ -n "${SB_CPUS:-}" ]      && ARGS+=(--cpus-per-task="$SB_CPUS")
exec sbatch "${ARGS[@]}" "$(dirname "$0")/run_figure.sbatch" "$@"
