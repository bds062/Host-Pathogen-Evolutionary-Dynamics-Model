#!/bin/bash
# Run one figure from test/figs.py on a SLURM cluster.
#
#   sbatch scripts/run_figure.sbatch 3_4          # one figure
#   sbatch scripts/run_figure.sbatch 3_4 4_4      # several in one job
#
# Cluster-specific settings come from the environment, so nothing here is tied to a
# particular site. Override any of them at submit time, e.g.
#
#   SB_ACCOUNT=myacct SB_PARTITION=general SB_QOS=long \
#       sbatch scripts/run_figure.sbatch 3_4
#
# The Floquet sweeps (3_4, 4_4) take roughly an hour on 32 cores; everything else
# reuses the cached .npz embeddings and finishes in seconds.
#SBATCH --job-name=hpedm_fig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: sbatch scripts/run_figure.sbatch <figure-key> [figure-key ...]" >&2
    echo "run 'python test/figs.py --list' for the available keys" >&2
    exit 2
fi

# Repo root, resolved from this script rather than from any absolute path.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

mkdir -p "$REPO_DIR/slurm_logs"
cd "$REPO_DIR/test"
echo "Host: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK:-?}  Start: $(date)"
"$PYTHON" -u figs.py --fig "$@"
echo "End: $(date)"
