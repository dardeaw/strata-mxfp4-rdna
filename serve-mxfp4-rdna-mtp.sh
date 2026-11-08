#!/bin/bash
# Decode-oriented launcher: in-file MTP drafting (--spec 2 --spec-min-p 0.5),
# the learned expert profile, and --expert-profile-save so the adaptive tier keeps
# learning the real traffic (saved every 10 min and on QUIT).
set -e
cd "$(dirname "$0")"
# MXFP4 compute: FP8 E4M3 on RDNA4 (gfx12); the engine auto-detects the card
# and falls back to FP16 on RDNA3 / any GPU without FP8 WMMA.  FOLDED converts
# the nibbles to E4M3 in the GEMM, so there is no separate dequant pass.
export STRATA_MXFP4_FP8=1
export STRATA_MXFP4_FOLDED=1
exec .venv/bin/python -m serve.server --engine strata \
  --config configs/mxfp4-rdna-mtp.json --port "${PORT:-8081}" --host "${HOST:-0.0.0.0}"
