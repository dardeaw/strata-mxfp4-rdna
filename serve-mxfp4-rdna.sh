#!/bin/bash
# Default launcher for the MXFP4-for-RDNA grow configuration.
# Dense weights stay on the GPU, the VRAM goes to the expert cache (--kv-grow),
# and the expert products use the gfx12 FP8 W4A8 path.
set -e
cd "$(dirname "$0")"
export STRATA_MXFP4_FP8=1
export STRATA_MXFP4_FOLDED=1
exec .venv/bin/python -m serve.server --engine strata \
  --config configs/mxfp4-rdna.json --port "${PORT:-8081}" --host "${HOST:-0.0.0.0}"
