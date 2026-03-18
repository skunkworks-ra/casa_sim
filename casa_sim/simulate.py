"""
simulate.py — CASA Simulation Framework, Module 9 (Entry Point)

Responsibilities:
  - Top-level pipeline orchestration (all 8 stages)
  - CASA tool instantiation and teardown
  - CLI: config.yaml [--dry-run] [--stage N,M,...]
  - Sweep dispatch

Usage:
  python simulate.py config.yaml
  python simulate.py config.yaml --dry-run
  python simulate.py config.yaml --stage 1,2,3

Stage numbering:
  1  parse_and_validate
  2  resolve_observatory
  3  resolve_observation
  4  resolve_sky_model
  5  predict
  6  corrupt
  7  sanity_image
  8  write_outputs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional, Set

import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_single(cfg, stages: Optional[Set[int]] = None) -> dict:
    """
    Run all pipeline stages for a single SimConfig.

    Args:
        cfg:    SimConfig instance (validated and derived)
        stages: Set of stage numbers to run. None means all stages.

    Returns:
        Metrics dict from sanity imaging (empty dict if disabled or skipped).
    """
    def _should_run(stage_num: int) -> bool:
        return stages is None or stage_num in stages

    # ---- Instantiate CASA tools ------------------------------------------
    # Tools are instantiated once per run_single() call and cleaned up in
    # the finally block to avoid CASA table cache errors.
    from casatools import (simulator, image, table, coordsys, measures,
                           componentlist, quanta, ctsys, ms as mstool)
    from casatasks import flagdata, mstransform

    sm = simulator()
    ia = image()
    tb = table()
    me = measures()
    qa = quanta()
    cl = componentlist()
    myms = mstool()

    # simutil for canned observatory reading
    from casatasks.private import simutil as _simutil
    mysu = _simutil.simutil()

    msname = cfg.name + '.ms'

    try:
        # ---- Stage 1: parse_and_validate ---------------------------------
        # Already done by caller (load_config_with_sweep validates).
        # Log derived params here for user verification.
        if _should_run(1):
            log.info("[stage 1] Config: name=%s cell=%s imsize=%s",
                     cfg.name, cfg.effective_cell, cfg.effective_imsize)
            log.info("[stage 1] Predictor: %s gridder=%s",
                     cfg.prediction.predictor, cfg.prediction.gridder)

        # ---- Stage 2: resolve_observatory --------------------------------
        if _should_run(2):
            log.info("[stage 2] Resolving observatory...")
            os.system(f'rm -rf {msname}')
            sm.open(ms=msname)

            from .observatory import resolve_observatory
            resolve_observatory(cfg, sm, me, mysu, ctsys)

        # ---- Stage 3: resolve_observation --------------------------------
        if _should_run(3):
            log.info("[stage 3] Resolving observation...")
            from .observation import resolve_observation
            resolve_observation(cfg, msname, sm, me, qa, flagdata)

        # ---- Stage 4: resolve_sky_model ----------------------------------
        sky_model_path = None
        if _should_run(4):
            log.info("[stage 4] Resolving sky model...")
            from .skymodel import resolve_sky_model
            sky_model_path = resolve_sky_model(cfg, ia, cl, qa, me)

        # ---- Stage 5: predict --------------------------------------------
        if _should_run(5):
            if sky_model_path is None:
                raise RuntimeError(
                    "Stage 5 requires sky_model_path from Stage 4. "
                    "Run Stage 4 before Stage 5."
                )
            log.info("[stage 5] Predicting visibilities...")
            from .predict import predict
            predict(cfg, msname, sky_model_path, sm, tb, mstransform)

        # ---- Stage 6: corrupt --------------------------------------------
        if _should_run(6):
            log.info("[stage 6] Applying corruptions...")
            from .corrupt import corrupt
            corrupt(cfg, msname, sm, myms)

        # ---- Stage 7: sanity_image (optional) ----------------------------
        metrics = {}
        if _should_run(7) and cfg.imaging.enabled:
            log.info("[stage 7] Running sanity imaging...")
            from .sanity import sanity_image
            from casatasks import imstat
            metrics = sanity_image(cfg, msname, ia, imstat)

        # ---- Stage 8: write_outputs -------------------------------------
        if _should_run(8):
            _write_outputs(cfg, msname, metrics)

        return metrics

    finally:
        # Teardown: close all tools to avoid CASA table cache errors
        _close_tools(ia, cl, sm)


def _close_tools(ia, cl, sm) -> None:
    """Safe tool teardown."""
    for tool, name in [(ia, 'ia'), (cl, 'cl'), (sm, 'sm')]:
        try:
            tool.close()
        except Exception:
            pass
        try:
            tool.done()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Stage 8: output summary
# ---------------------------------------------------------------------------

def _write_outputs(cfg, msname: str, metrics: dict) -> None:
    """Print summary of all output paths to stdout."""
    print("\n" + "=" * 60)
    print(f"Simulation complete: {cfg.name}")
    print("=" * 60)
    print(f"  MS:          {msname}")

    # Intermediate sky model images
    for suffix in ['_skymodel_fromcl.im', '_skymodel_extrapolated.im',
                   '_skymodel_faraday.im', '_skymodel_withlines.im']:
        path = cfg.name + suffix
        if os.path.exists(path):
            print(f"  Sky model:   {path}")

    # Sanity imaging outputs
    for suffix in ['.image', '.residual', '.pb', '.psf']:
        path = cfg.name + '_sanity' + suffix
        if os.path.exists(path):
            print(f"  Sanity img:  {path}")

    sanity_json = cfg.name + '_sanity.json'
    if os.path.exists(sanity_json):
        print(f"  Metrics:     {sanity_json}")

    sweep_json = cfg.name + '_sweep_index.json'
    if os.path.exists(sweep_json):
        print(f"  Sweep index: {sweep_json}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='CASA Simulation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('config', help='Path to YAML config file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate config and print derived params, then exit')
    parser.add_argument('--stage', type=str, default=None,
                        help='Run only specified stages (comma-separated, e.g. 1,2,3)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging verbosity (default: INFO)')
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # ---- Load and validate -----------------------------------------------
    from .config import load_config_with_sweep
    try:
        base_cfg, sweep_configs, raw = load_config_with_sweep(args.config)
    except Exception as e:
        log.error("Config load failed: %s", e)
        sys.exit(1)

    # ---- Dry run ---------------------------------------------------------
    if args.dry_run:
        _dry_run(base_cfg)
        sys.exit(0)

    # ---- Parse stage filter ----------------------------------------------
    stages = None
    if args.stage:
        try:
            stages = {int(s.strip()) for s in args.stage.split(',')}
            log.info("Running stages: %s", sorted(stages))
            _warn_partial_stages(stages)
        except ValueError:
            log.error("Invalid --stage value: '%s'. Use comma-separated integers.", args.stage)
            sys.exit(1)

    # ---- Run -------------------------------------------------------------
    if base_cfg.sweep is None or not base_cfg.sweep.axes:
        # Single simulation
        run_single(base_cfg, stages=stages)
    else:
        # Sweep
        from .sweep import run_sweep
        run_sweep(base_cfg, raw, lambda cfg: run_single(cfg, stages=stages))


def _dry_run(cfg) -> None:
    """Print fully resolved config as YAML and exit."""
    import dataclasses

    print("\n--- Dry Run: Resolved Configuration ---\n")
    print(f"name:          {cfg.name}")
    print(f"cell:          {cfg.effective_cell}")
    print(f"imsize:        {cfg.effective_imsize}")
    print(f"predictor:     {cfg.prediction.predictor}")
    print(f"gridder:       {cfg.prediction.gridder}")
    print(f"sky_model:     {cfg.sky_model.mode} / stokes={cfg.sky_model.stokes}")
    print(f"noise:         enabled={cfg.corruption.noise.enabled} "
          f"mode={cfg.corruption.noise.mode} value={cfg.corruption.noise.value}")
    print(f"gains:         enabled={cfg.corruption.gains.enabled}")
    print(f"imaging:       enabled={cfg.imaging.enabled}")
    print(f"seed:          {cfg.corruption.seed}")

    if cfg.sweep:
        print(f"\nsweep axes:    {len(cfg.sweep.axes)}")
        for ax in cfg.sweep.axes:
            print(f"  {ax.parameter}: {ax.values}")

    print("\n--- Validation passed ---\n")


def _warn_partial_stages(stages: set) -> None:
    """Warn about stage subsets that are known to be invalid."""
    # Stage 5 requires Stage 4 output (sky_model_path)
    if 5 in stages and 4 not in stages:
        log.warning("--stage includes 5 but not 4: sky_model_path will be None. "
                    "Stage 5 will fail unless you are debugging and sky model exists on disk.")
    # Stage 3 requires Stage 2 to have created the MS
    if 3 in stages and 2 not in stages:
        log.warning("--stage includes 3 but not 2: MS may not exist yet.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    main()
