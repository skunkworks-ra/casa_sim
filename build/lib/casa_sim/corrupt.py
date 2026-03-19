"""
corrupt.py — CASA Simulation Framework, Module 6

Responsibilities:
  - Apply gain errors then noise (fixed physical order — not configurable)
  - Seed control for reproducibility
  - Three noise modes: simplenoise | tsys-atm | per_baseline

Design notes:
  - seed=None → do NOT call sm.setseed() at all. Never substitute 0 or -1.
  - /100.0 scaling from reference notebook addDishNoise() is simulation-specific
    and is NOT carried over here. noise_scale=1.0 by default.
  - sm.setseed() must be called before sm.setnoise() and sm.corrupt()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import SimConfig, CorruptionConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def corrupt(cfg: "SimConfig", msname: str, sm, myms) -> None:
    """
    Apply corruptions in physically correct order: gains first, noise second.
    Order is fixed and not configurable.

    Args:
        cfg:    SimConfig instance
        msname: Path to the MS (DATA column must be populated)
        sm:     CASA simulator tool
        myms:   CASA ms tool
    """
    corr = cfg.corruption

    if corr.gains.enabled:
        _apply_gains(corr, msname, sm)

    if corr.noise.enabled:
        mode = corr.noise.mode
        if mode == 'simplenoise':
            _apply_simplenoise(corr, msname, sm)
        elif mode == 'tsys-atm':
            _apply_tsys_atm(corr, msname, sm)
        elif mode == 'per_baseline':
            _apply_per_baseline_noise(corr, msname, myms)
        else:
            raise ValueError(f"Unknown noise mode '{mode}'")
    else:
        log.info("[corrupt] Noise disabled — skipping")


# ---------------------------------------------------------------------------
# Gain corruption
# ---------------------------------------------------------------------------

def _apply_gains(corr: "CorruptionConfig", msname: str, sm) -> None:
    """
    Apply antenna gain errors using sm.setgain(mode='fbm').
    Seed set before corruption if not None.
    """
    sm.openfromms(msname)

    if corr.seed is not None:
        sm.setseed(corr.seed)
        log.info("[corrupt] Gain seed set: %d", corr.seed)
    else:
        log.warning("[corrupt] Gain seed=None — gain results non-reproducible")

    sm.setgain(mode=corr.gains.mode, amplitude=corr.gains.amplitude)
    sm.corrupt()
    sm.close()
    log.info("[corrupt] Gain corruption applied: mode=%s amplitude=%.4f",
             corr.gains.mode, corr.gains.amplitude)


# ---------------------------------------------------------------------------
# Noise modes
# ---------------------------------------------------------------------------

def _apply_simplenoise(corr: "CorruptionConfig", msname: str, sm) -> None:
    """
    Apply simple Gaussian noise via sm.setnoise(mode='simplenoise').
    Seed must be set before setnoise() and corrupt().
    """
    sm.openfromms(msname)

    if corr.seed is not None:
        sm.setseed(corr.seed)
        log.info("[corrupt] Noise seed set: %d", corr.seed)
    else:
        log.warning("[corrupt] Noise seed=None — noise results non-reproducible")

    sm.setnoise(mode='simplenoise', simplenoise=corr.noise.value)
    sm.corrupt()
    sm.close()
    log.info("[corrupt] simplenoise applied: value=%s", corr.noise.value)


def _apply_tsys_atm(corr: "CorruptionConfig", msname: str, sm) -> None:
    """
    Apply atmospheric noise via sm.setnoise(mode='tsys-atm').
    Seed not applicable for tsys-atm mode (per design doc).
    """
    sm.openfromms(msname)
    sm.setnoise(mode='tsys-atm')
    sm.corrupt()
    sm.close()
    log.info("[corrupt] tsys-atm noise applied")


def _apply_per_baseline_noise(corr: "CorruptionConfig", msname: str, myms) -> None:
    """
    Add Gaussian noise to DATA column scaled by per-baseline SIGMA values.
    SIGMA is set by sm.observe() based on dish diameters.

    Seed controls np.random.seed() before the iteration loop.
    noise_scale=1.0 (design default — /100 from reference notebook omitted by design).
    """
    noise_scale = getattr(corr, 'noise_scale', 1.0)

    if corr.seed is not None:
        np.random.seed(corr.seed)
        log.info("[corrupt] per_baseline seed set: %d", corr.seed)
    else:
        log.warning("[corrupt] per_baseline seed=None — noise results non-reproducible")

    myms.open(msname, nomodify=False)
    myms.iterinit(interval=1000)
    myms.iterorigin()

    moretodo = True
    while moretodo:
        dat = myms.getdata(items=['SIGMA', 'DATA'])
        shp = dat['data'].shape

        # Broadcast sigma [npol, nrow] → [npol, nchan, nrow]
        sigma_broadcast = np.repeat(dat['sigma'][:, np.newaxis, :], shp[1], axis=1)

        noise_re = np.random.normal(loc=0.0, scale=sigma_broadcast) * noise_scale
        noise_im = np.random.normal(loc=0.0, scale=sigma_broadcast) * noise_scale

        dat['data'] = dat['data'] + noise_re + 1j * noise_im
        myms.putdata(dat)

        moretodo = myms.iternext()

    myms.close()
    log.info("[corrupt] per_baseline noise applied: noise_scale=%.4f", noise_scale)
