#!/bin/bash
# Run ERASURE's published Adult benchmark config: 1-layer MLP, 5% forget set, seed 0.
# 38 unlearner configurations in one process; about 68 minutes on a DGX Spark.
#
# Note: the process segfaults at teardown (exit 139) *after* every evaluation has been
# written. Check the number of records in the results file, not the exit code.
set -u

ERASURE_DIR=${ERASURE_DIR:-$HOME/ERASURE}
VENV=${VENV:-$HOME/envs/erasure}

source "$VENV/bin/activate"
cd "$ERASURE_DIR"
mkdir -p dev/ICLR_benchmark/results/tabular   # where the config's SaveValues measure writes

CFG='configs/benchmark/tabular/adult/fs_05/adult_1mlp_05%_seed0.jsonc'
echo "=== start $(date -Is) ==="
time python main.py "$CFG"
echo "=== exit $? at $(date -Is) ==="
