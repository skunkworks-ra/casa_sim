# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A CASA-based radio astronomy simulator organized as a modular 8-stage pipeline. Config-driven (YAML), supports parameter sweeps via Cartesian product expansion, and outputs CASA Measurement Sets (.ms). Intended to be upstreamed as a simulation skill into wildcat and data-analyst repos.

## Python Environment

No separate install needed — reuse the ms-inspect pixi env which has casatools, casatasks, astropy:

```bash
/home/pjaganna/.claude/plugins/cache/ms-inspect/ms-inspect/0.1.0/.pixi/envs/default/bin/python
```

## Running the Simulator

```bash
PYTHON=/home/pjaganna/.claude/plugins/cache/ms-inspect/ms-inspect/0.1.0/.pixi/envs/default/bin/python

# Single config
$PYTHON simulate.py config.yaml

# Dry run (validate config only)
$PYTHON simulate.py config.yaml --dry-run

# Specific pipeline stages only
$PYTHON simulate.py config.yaml --stage 1,2,3
```

## Running Tests

```bash
# All integration tests
$PYTHON tests/run_all.py

# Generate reference outputs (first-time or after intentional changes)
$PYTHON tests/run_all.py --generate-reference

# Single test
$PYTHON tests/integration/test_3c286.py
$PYTHON tests/integration/test_3c286.py --generate-reference
$PYTHON tests/integration/test_vla_basic.py --config-only
```

## Repository Structure

```
casa_sim/           # installable package (source modules)
  __init__.py       # exposes load_config_with_sweep, run_single, run_sweep
  config.py         # YAML parsing, validation, sweep expansion (no CASA deps)
  simulate.py       # 8-stage pipeline orchestration + CLI main()
  skymodel.py       # sky model resolution (component list, image, Faraday)
  predict.py        # visibility prediction (ft_dft | sm_predict | tclean)
  corrupt.py        # noise, gains, Faraday corruption
  observation.py    # SPW, field, scan timing setup
  observatory.py    # telescope/antenna setup
  sanity.py         # sanity imaging + fidelity metrics
  sweep.py          # Cartesian product sweep expansion
simulate.py         # thin CLI shim → delegates to casa_sim.simulate.main()
tests/
  configs/          # YAML configs for integration tests
  data/             # generated test data (component lists etc.)
  integration/      # test scripts
  reference/        # reference JSON outputs (committed)
  run_all.py        # test runner
outputs/            # runtime outputs: MS, images, logs (gitignored)
pixi.toml           # dependency declaration
pyproject.toml      # package metadata
```

## Architecture

### 8-Stage Pipeline (`casa_sim/simulate.py`)

`run_single()` runs one `SimConfig` through all stages. `sweep.run_sweep()` iterates over sweep points.

Stages: observatory → observation → sky model → predict → corrupt → sanity image → write outputs.

### Predictor Routing (`casa_sim/predict.py`)

Three predictor strategies, both writing directly to the DATA column:
- `ft_dft`: `casatasks.ft(complist=...)` — Stokes I/V only (parallel hands)
- `sm_predict`: `sm.predict(complist=...)` — full Stokes IQUV via circular feed Jones matrices
- `tclean`: `tclean(savemodel=...)` — for mosaic/awproject gridders

**Critical**: both `ft_dft` and `sm_predict` write directly to DATA. Do NOT call `_copy_model_to_data()` after them.

### Config System (`casa_sim/config.py`)

- YAML → dataclass tree with validation
- `load_config_with_sweep()` is the primary entry point
- No CASA tool imports — uses `astropy.units` for unit arithmetic
- `apply_override()` supports dotpath config updates

### CASA Tool Lifecycle

Tools instantiated once per `run_single()` call, passed as dependencies, cleaned up in `finally`. Never cache across runs.
