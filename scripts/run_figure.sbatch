#!/bin/bash
# Run one figure from test/figs.py on a SLURM cluster.
#
#   sbatch scripts/run_figure.sbatch sweeps          # every simulation (hours)
#   sbatch scripts/run_figure.sbatch species fig6    # one sweep, then its figure
#
# Cluster-specific settings come from the environment, so nothing here is tied to a
# particular site. Override any of them at submit time, e.g.
#
#   SB_ACCOUNT=myacct SB_PARTITION=general SB_QOS=long \
#       sbatch scripts/run_figure.sbatch sweeps
#
# Each invasion-fitness sweep (maturation, susceptibility, species) takes roughly an
# hour on 32 cores; the figures redraw from the cached .npz files in seconds.
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

# Repo root. SLURM runs a spooled copy of this script, so its own path is useless;
# use HPEDM_REPO (set by submit.sh), else the directory sbatch was called from
# (submit from the repo root), else this script's location when run with bash.
REPO_DIR="${HPEDM_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [ ! -f "$REPO_DIR/test/figs.py" ]; then
    echo "cannot find test/figs.py under $REPO_DIR; submit from the repo root" >&2
    exit 2
fi
PYTHON="${PYTHON:-python3}"

mkdir -p "$REPO_DIR/slurm_logs"
cd "$REPO_DIR/test"
echo "Host: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK:-?}  Start: $(date)"
"$PYTHON" -u figs.py --fig "$@"
echo "End: $(date)"
