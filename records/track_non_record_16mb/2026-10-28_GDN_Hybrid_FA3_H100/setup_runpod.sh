#!/usr/bin/env bash
# One-time setup on a fresh RunPod (or any Hopper Linux box), from the repo root:
#   bash records/track_non_record_16mb/2026-10-28_GDN_Hybrid_FA3_H100/setup_runpod.sh
#
# Pins the exact environment these runs used: Python 3.12 (uv-managed), torch 2.9.1+cu128,
# fla 0.4.2, kernels 0.16.0 (FA3 from kernels-community, downloaded from the HF hub on first use).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv > /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Must be a uv-managed CPython: a venv on a distro python without dev headers makes Triton
# fail to build its launcher ("Python.h: No such file or directory").
if [[ ! -x .venv/bin/python ]]; then
  uv python install 3.12
  uv venv --managed-python --python 3.12 .venv
fi
VIRTUAL_ENV=.venv uv pip install torch==2.9.1 --torch-backend=cu128
VIRTUAL_ENV=.venv uv pip install -r "$HERE/requirements.txt"

# 8 shards is data-identical to 80 for these run lengths (a 32k run reads ~2.4 shards).
.venv/bin/python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 8

echo "=== environment check"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
.venv/bin/python - <<'EOF'
import torch, triton, fla
from fla.ops.kda import chunk_kda  # noqa: F401
from kernels import get_kernel
print(f"torch {torch.__version__} triton {triton.__version__} fla {fla.__version__} "
      f"sm{''.join(map(str, torch.cuda.get_device_capability()))} gpus={torch.cuda.device_count()}")
get_kernel("kernels-community/flash-attn3", version=1)
print("FA3 kernel: ok")
EOF
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench/check_kda.py "$HERE/train_gpt_gdn.py" 2>&1 | grep -E "^\[|PASS|FAIL"
echo "=== setup ok. Next:"
echo "  DRY_RUN=1 bash $HERE/run_core.sh   # inspect the job plan"
echo "  bash $HERE/run_core.sh"
