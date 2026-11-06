#!/bin/bash
# Decode-oriented launcher: in-file MTP drafting (--spec 2 --spec-min-p 0.5),
# the learned expert profile, and --expert-profile-save so the adaptive tier keeps
# learning the real traffic (saved every 10 min and on QUIT).
set -e
cd "$(dirname "$0")"
exec .venv/bin/python -m serve.server --engine strata \
  --config configs/mxfp4-rdna-mtp.json --port "${PORT:-8081}"
