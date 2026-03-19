# casa_sim

CASA-based radio interferometry simulation framework. Config-driven 8-stage pipeline that produces calibrated Measurement Sets from YAML specifications. Designed to be used as a simulation skill in wildcat and data-analyst workflows.

---

## Environment

Requires `casatools`, `casatasks`, `astropy`, `pyyaml`, `numpy`.

```bash
# Using pixi (recommended)
pixi install

# Or install into your own env
pip install casatools casatasks astropy pyyaml numpy
pip install -e .   # installs casa_sim package
```

---

## Quick Start

```bash
# Run a simulation
$PYTHON simulate.py tests/configs/vla_3c286.yaml

# Dry run — validate config only
$PYTHON simulate.py tests/configs/vla_3c286.yaml --dry-run

# Run specific pipeline stages
$PYTHON simulate.py tests/configs/vla_3c286.yaml --stage 1,2,3
```

---

## Pipeline Stages

| # | Stage | Module |
|---|-------|--------|
| 1 | parse & validate config | `config.py` |
| 2 | set up observatory (antennas, feeds) | `observatory.py` |
| 3 | configure observation (fields, SPWs, scans) | `observation.py` |
| 4 | resolve sky model (component list or image, Faraday) | `skymodel.py` |
| 5 | predict visibilities | `predict.py` |
| 6 | apply corruptions (noise, gains) | `corrupt.py` |
| 7 | sanity imaging + fidelity metrics | `sanity.py` |
| 8 | write output summary | `simulate.py` |

---

## Config Reference

All simulations are driven by a YAML file. Full example with all sections:

```yaml
name: sim_vla_3c286          # output MS will be {name}.ms

observatory:
  mode: canned               # canned | custom
  feeds: "perfect R L"       # perfect R L | perfect X Y
  canned:
    telescope: VLA
    cfg_file: null            # null = use CASA default for telescope
    antlist: [W01, W02, E01]  # null = use all antennas

observation:
  epoch: "UTC 2024/06/15/06:00:00"
  integration_time: "10s"
  use_hourangle: true
  fields:
    - name: 3C286
      direction: "J2000 13h31m08.2899s +30d30m32.959s"
  spws:
    - name: LBand
      freq: "1.4GHz"           # channel 0 centre frequency
      deltafreq: "0.2GHz"      # total bandwidth
      nchan: 5
      stokes: "RR RL LR LL"   # RR LL = Stokes I only; add RL LR for full-Stokes
  observe_calls:
    - field: 3C286
      spw: LBand
      start_time: "-2.0h"      # relative to epoch (h, m, s supported)
      stop_time: "+2.0h"

sky_model:
  stokes: IQUV                 # I | IQUV
  mode: component_list         # component_list | image_native | image_extrapolate
  cl_path: tests/data/3c286_lband.cl   # pre-built component list (OR use sources: below)
  faraday:                     # optional — omit to disable
    enabled: true
    rm_mode: global            # global = single RM for whole image
    rm_value: 50.0             # rad/m^2
    ref_freq: "1.25GHz"

prediction:
  gridder: standard            # standard | mosaic | awproject
  predictor: sm_predict        # auto | ft_dft | sm_predict | tclean
  cell: "9arcsec"              # null = auto-derived from array geometry
  imsize: 256                  # null = auto-derived
  normtype: flatsky

corruption:
  seed: 42
  noise:
    enabled: true
    mode: simplenoise          # simplenoise | per_baseline
    value: "0.05Jy"
  gains:
    enabled: false
    mode: fbm
    amplitude: 0.0

imaging:
  enabled: true
  deconvolver: hogbom          # hogbom | clark | mtmfs
  pbcor: false
  niter: 500
  nterms: 2                    # mtmfs only
```

### Inline Source Definitions

Instead of providing a pre-built `.cl` file, you can define sources directly in YAML. The pipeline auto-builds the component list and configures Faraday rotation from per-source RM values.

`sources` and `cl_path` are mutually exclusive — use one or the other.

```yaml
sky_model:
  stokes: IQUV
  mode: component_list
  sources:
    - name: 3C286
      direction: "J2000 13h31m08.29s +30d30m32.96s"
      flux: [14.9]                   # [I] or [I, Q, U, V]
      ref_freq: "1.4GHz"            # per-source reference frequency
      spectral_index: [-0.61]       # [alpha] or [alpha, beta] (curvature)
      shape: point                   # point | gaussian | disk
      rm: 50.0                       # rotation measure (rad/m^2), per-source
      frac_pol: 0.10                 # fractional linear polarization (derives Q, U from I)
      chi: 33.0                      # EVPA in degrees (required with frac_pol)

    - name: extended_src
      direction: "J2000 13h32m00s +30d35m00s"
      flux: [1.0, 0.2, 0.1, 0.0]   # explicit IQUV — cannot combine with frac_pol
      ref_freq: "1.4GHz"
      spectral_index: [-0.7, 0.1]   # alpha + curvature: S = S0 * (nu/nu0)^(a + b*ln(nu/nu0))
      shape: gaussian
      major: "10arcsec"
      minor: "5arcsec"
      pa: "45deg"
      rm: -30.0
```

**Source fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Source label (for logging) |
| `direction` | yes | J2000 sky position |
| `flux` | yes | `[I]` or `[I, Q, U, V]` in Jy |
| `ref_freq` | yes | Reference frequency for flux and spectral model |
| `spectral_index` | no | `[alpha]` or `[alpha, beta]`. Default: `[0.0]` |
| `shape` | no | `point` (default), `gaussian`, or `disk` |
| `major`, `minor`, `pa` | gaussian/disk | Angular size and position angle |
| `rm` | no | Rotation measure in rad/m^2. Default: `0.0` |
| `frac_pol` | no | Fractional linear polarization (0–1). Derives Q, U from Stokes I |
| `chi` | with frac_pol | EVPA in degrees. Required when `frac_pol` is set |

**Polarization derivation:** when `flux` is `[I]` and `frac_pol` + `chi` are given:
- `Q = I * frac_pol * cos(2 * chi)`
- `U = I * frac_pol * sin(2 * chi)`
- `V = 0`

**Per-source RM:** Faraday rotation is auto-configured from source RM values. If all sources share the same RM, a global scalar is used. If sources have different RMs, a 2D RM map is built from source positions.

### Predictor Selection

| Predictor | Use when | Stokes |
|-----------|----------|--------|
| `ft_dft` | component list + Stokes I only | I, V (parallel hands only) |
| `sm_predict` | component list + full Stokes IQUV | I, Q, U, V |
| `tclean` | image-based + mosaic/awproject gridder | I, Q, U, V |
| `auto` | selects ft_dft for component_list + standard, tclean otherwise | — |

**Rule:** for full-Stokes simulations always set `predictor: sm_predict` explicitly.

---

## Parameter Sweeps

Add a `sweep` block to run a Cartesian product over any config parameter:

```yaml
sweep:
  axes:
    - parameter: corruption.noise.value
      values: ["0.01Jy", "0.1Jy", "1.0Jy"]
    - parameter: observation.integration_time
      values: ["10s", "30s"]
```

Produces one MS per combination: `{name}_value_0.01Jy_integration_time_10s.ms` etc.

---

## Python API

```python
from casa_sim import load_config_with_sweep, run_single, run_sweep

# Single simulation
cfg, sweep_cfgs, raw = load_config_with_sweep("config.yaml")
metrics = run_single(cfg)

# Sweep
run_sweep(sweep_cfgs, raw)
```

---

## Tests

```bash
# Full suite (auto-generates missing references on first run)
$PYTHON tests/run_all.py

# Generate / regenerate all references
$PYTHON tests/run_all.py --generate-reference

# Individual tests
$PYTHON tests/integration/test_vla_basic.py        # Stokes I point source
$PYTHON tests/integration/test_3c286.py            # Full-Stokes 3C286 (Perley & Butler 2013)
$PYTHON tests/integration/test_faraday.py          # RM recovery
$PYTHON tests/integration/test_sweep.py            # Noise sweep
```

### Test Structure

Each test has three categories:
- **A — Structural**: MS metadata (fields, SPWs, antennas, correlations)
- **B — Predict-recover**: peak flux per Stokes within tolerance (I: 2%, Q/U: 5%)
- **C — Noise**: RMS within 5% of reference (seed=42 for reproducibility)
- **D — Faraday** (`test_faraday.py`): recovered RM within 10% of input
- **E — Sweep** (`test_sweep.py`): MS count, naming, RMS monotonicity

---

## Outputs

All runtime outputs land in the working directory:
- `{name}.ms` — simulated Measurement Set
- `{name}_sources.cl` — auto-built component list (when using inline `sources:`)
- `{name}_rm_map.im` — auto-built RM map (when sources have different per-source RMs)
- `{name}_skymodel_fromcl.im` — component list evaluated to image (Faraday/spectral line paths)
- `{name}_skymodel_faraday.im` — sky model after Faraday rotation
- `{name}_sanity.image` — tclean sanity image (IQUV cube if full-Stokes)
- `{name}_sanity.json` — fidelity metrics (peak intensity per Stokes, residual RMS, recovered RM)
- `{name}_sweep_index.json` — sweep parameter map (sweep runs only)

Add `outputs/` to your working directory to keep the repo root clean — it is gitignored.
