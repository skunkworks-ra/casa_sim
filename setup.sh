#!/usr/bin/env bash
#
# setup.sh -- one-shot environment + data bring-up for casa_sim.
#
# Idempotent: safe to re-run. Each step skips work that is already done.
#
# Steps:
#   1. pixi install                -- create the conda/pixi environment
#   2. pixi run fetch-trecs        -- download T-RECS AGN + SFG catalogs
#   3. pixi run build-morphology   -- regenerate the morphology template library
#   4. pixi run test               -- sanity-check the install
#
# Data locations (no hardcoded user paths):
#   - T-RECS catalogs:  $CASA_SIM_TRECS_DIR  (default: ./data/trecs)
#   - Morphology lib:   ./data/morphology/lib  (rebuilt from ./data/morphology/tng_raw)
#   - VLA .cfg files:   resolved from the active CASA data install (casaconfig)
#
# Usage:
#   bash setup.sh                  # full bring-up
#   bash setup.sh --no-trecs       # skip the multi-GB catalog download
#   bash setup.sh --no-test        # skip the test sanity check

set -euo pipefail
cd "$(dirname "$0")"

FETCH_TRECS=1
RUN_TEST=1
for arg in "$@"; do
  case "$arg" in
    --no-trecs) FETCH_TRECS=0 ;;
    --no-test)  RUN_TEST=0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v pixi >/dev/null 2>&1; then
  echo "ERROR: pixi not found on PATH. Install pixi first: https://pixi.sh" >&2
  exit 1
fi

echo "==> [1/4] pixi install"
pixi install

if [[ "$FETCH_TRECS" == "1" ]]; then
  echo "==> [2/4] fetch T-RECS catalogs (AGN + SFG; multi-GB, may take a while)"
  pixi run fetch-trecs --pops agn sfg
else
  echo "==> [2/4] skipping T-RECS fetch (--no-trecs)"
fi

echo "==> [3/4] build morphology template library from data/morphology/tng_raw"
pixi run build-morphology

if [[ "$RUN_TEST" == "1" ]]; then
  echo "==> [4/4] sanity check: test suite"
  pixi run test
else
  echo "==> [4/4] skipping test sanity check (--no-test)"
fi

echo
echo "Setup complete. Generate a corpus field with:"
echo "    pixi run python tools/smoketest_corpus.py"
