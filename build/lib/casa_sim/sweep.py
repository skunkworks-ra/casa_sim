"""
sweep.py — CASA Simulation Framework, Module 8

Responsibilities:
  - Expand sweep axes into Cartesian product of configs
  - Run pipeline per sweep point
  - Construct MS names per sweep point
  - Write {name}_sweep_index.json

Design notes:
  - Each sweep point gets its own output MS name via _make_ms_name()
  - sweep index JSON uses dataclasses.asdict() for config snapshot
  - pipeline_fn = simulate.run_single — sweep passes configs to it
  - MS names: {name}_{short_param}_{value}.ms (single axis)
              {name}_{p1}_{v1}_{p2}_{v2}.ms  (multi-axis)
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List

if TYPE_CHECKING:
    from .config import SimConfig, SweepConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_sweep(cfg: "SimConfig", raw: dict, pipeline_fn: Callable) -> List[dict]:
    """
    Expand sweep config, run pipeline per point, write sweep index.

    Args:
        cfg:         Base SimConfig (with sweep block)
        raw:         Original raw YAML dict (needed for apply_override round-trip)
        pipeline_fn: Callable(SimConfig) → dict of metrics.
                     Typically simulate.run_single.

    Returns:
        List of {ms_name, config_snapshot, metrics} dicts.
    """
    from .config import expand_sweep_from_raw, validate_config, derive_imaging_params

    if cfg.sweep is None or not cfg.sweep.axes:
        log.info("[sweep] No sweep block — running single pipeline")
        metrics = pipeline_fn(cfg)
        return [{'ms_name': cfg.name + '.ms',
                 'config_snapshot': _config_to_dict(cfg),
                 'metrics': metrics}]

    axes = cfg.sweep.axes
    value_lists = [axis.values for axis in axes]
    dotpaths = [axis.parameter for axis in axes]

    results = []

    for combo in itertools.product(*value_lists):
        overrides = dict(zip(dotpaths, combo))

        # Build MS name for this sweep point
        ms_name = _make_ms_name(cfg.name, overrides)

        # Apply overrides and build config
        from .config import apply_override, _parse_simconfig_from_raw
        import copy
        raw_copy = copy.deepcopy(raw)
        for dotpath, value in overrides.items():
            raw_copy = apply_override(raw_copy, dotpath, value)
        # Override the name to produce distinct MS names
        raw_copy['name'] = ms_name.replace('.ms', '')

        swept_cfg = _parse_simconfig_from_raw(raw_copy)
        validate_config(swept_cfg)
        swept_cfg = derive_imaging_params(swept_cfg)

        log.info("[sweep] Running point: %s", overrides)
        try:
            metrics = pipeline_fn(swept_cfg)
        except Exception as e:
            log.error("[sweep] Pipeline failed for %s: %s", ms_name, e)
            metrics = {'error': str(e)}

        results.append({
            'ms_name': ms_name,
            'config_snapshot': _config_to_dict(swept_cfg),
            'metrics': metrics
        })

    _write_sweep_index(results, cfg.name)
    log.info("[sweep] Sweep complete: %d point(s)", len(results))
    return results


# ---------------------------------------------------------------------------
# MS naming
# ---------------------------------------------------------------------------

def _make_ms_name(base_name: str, overrides: Dict[str, Any]) -> str:
    """
    Construct MS name from base name and sweep overrides.

    Single axis:  {name}_{short_param}_{value}.ms
    Multi-axis:   {name}_{p1}_{v1}_{p2}_{v2}.ms
    """
    parts = [base_name]
    for dotpath, value in overrides.items():
        short = dotpath.split('.')[-1]
        parts.append(short)
        parts.append(str(value))
    raw_name = '_'.join(parts) + '.ms'
    return _sanitize_name(raw_name)


def _sanitize_name(name: str) -> str:
    """
    Strip characters that are not alphanumeric, dot, or underscore.
    Preserves .ms suffix.
    """
    # Replace common problematic characters
    name = name.replace('/', '_').replace('\\', '_').replace(' ', '_')
    # Keep only alphanumeric, dot, underscore
    name = re.sub(r'[^\w.]', '_', name)
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)
    return name


# ---------------------------------------------------------------------------
# Sweep index output
# ---------------------------------------------------------------------------

def _write_sweep_index(results: List[dict], base_name: str) -> None:
    """
    Write {name}_sweep_index.json.
    Maps each MS name to its full resolved config snapshot.
    """
    index = {r['ms_name']: r['config_snapshot'] for r in results}
    path = f"{base_name}_sweep_index.json"
    with open(path, 'w') as fh:
        json.dump(index, fh, indent=2)
    log.info("[sweep] Sweep index written: %s", path)


# ---------------------------------------------------------------------------
# Config serialization
# ---------------------------------------------------------------------------

def _config_to_dict(cfg: "SimConfig") -> dict:
    """
    Serialize SimConfig to a plain dict for JSON output.
    Uses dataclasses.asdict() with None filtering for derived fields.
    """
    raw = dataclasses.asdict(cfg)
    # Remove internal derived fields that start with underscore
    # dataclasses.asdict includes them with mangled names
    keys_to_remove = [k for k in raw if k.startswith('_')]
    for k in keys_to_remove:
        del raw[k]
    return raw
