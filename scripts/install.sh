#!/bin/bash
# Install ERASURE's dependencies without letting pip touch torch/torchvision.
#
# On aarch64 + Blackwell (DGX Spark) torch has to come from the cu130 wheel index. If pip
# resolves torch from ERASURE's requirements.txt it can quietly swap in a CPU-only wheel,
# and everything still runs, just ~100x slower. So torch is installed first from the right
# index and then filtered out of the requirements.
#
# On x86 + CUDA 12 set TORCH_INDEX=https://download.pytorch.org/whl/cu121 (or whatever
# matches your driver).
set -euo pipefail

ERASURE_DIR=${ERASURE_DIR:-$HOME/ERASURE}
VENV=${VENV:-$HOME/envs/erasure}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}

[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip

if ! python -c "import torch" 2>/dev/null; then
    pip install torch torchvision --index-url "$TORCH_INDEX"
fi

cd "$ERASURE_DIR"
grep -vE '^(torch|torchvision)$' requirements.txt > /tmp/erasure_req_no_torch.txt
echo "--- installing ---"
cat /tmp/erasure_req_no_torch.txt
pip install -r /tmp/erasure_req_no_torch.txt

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
