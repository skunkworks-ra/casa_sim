"""
sanity.py — CASA Simulation Framework, Module 7

Responsibilities:
  - Run tclean sanity imaging (optional, imaging.enabled)
  - Extract fidelity metrics per channel/Stokes
  - Estimate RM via lambda^2 fit if Faraday enabled
  - Write {name}_sanity.json

Design notes:
  - imstat() with axes=[0,1] gives per-channel statistics
  - Sanity tclean outputs prefixed {name}_sanity.* to avoid collision
  - _estimate_rm uses np.unwrap() before fitting; slope is RM in rad/m^2
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .config import SimConfig

log = logging.getLogger(__name__)

_C_LIGHT = 2.99792458e8   # m/s


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def sanity_image(cfg: "SimConfig", msname: str, ia, imstat_fn) -> dict:
    """
    Run optional sanity imaging and compute fidelity metrics.

    Args:
        cfg:        SimConfig instance
        msname:     Path to the MS
        ia:         CASA image tool
        imstat_fn:  casatasks.imstat function

    Returns:
        Metrics dict. Empty dict if imaging.enabled is False.
    """
    if not cfg.imaging.enabled:
        log.info("[sanity] imaging.enabled=False — skipping")
        return {}

    _run_tclean(cfg, msname)
    metrics = _extract_metrics(cfg, ia, imstat_fn)
    _write_sanity_log(metrics, f"{cfg.name}_sanity.json")

    log.info("[sanity] Metrics written to %s_sanity.json", cfg.name)
    return metrics


# ---------------------------------------------------------------------------
# tclean run
# ---------------------------------------------------------------------------

def _run_tclean(cfg: "SimConfig", msname: str) -> None:
    """
    Run tclean for sanity imaging.
    Outputs prefixed {name}_sanity.* to avoid collision with user imaging.
    """
    from casatasks import tclean

    img = cfg.imaging
    imagename = f"{cfg.name}_sanity"
    cell = cfg.effective_cell
    imsize = cfg.effective_imsize

    if cell is None or imsize is None:
        raise ValueError(
            "cell/imsize must be derived before sanity imaging. "
            "Call derive_imaging_params() first."
        )

    total_nchan = sum(s.nchan for s in cfg.observation.spws)
    specmode = 'mfs' if total_nchan == 1 else 'cube'

    # Determine gridder — must match prediction gridder
    gridder = cfg.prediction.gridder

    tclean_args = dict(
        vis=msname,
        imagename=imagename,
        imsize=imsize,
        cell=cell,
        gridder=gridder,
        specmode=specmode,
        deconvolver=img.deconvolver,
        niter=img.niter,
        pbcor=img.pbcor,
        stokes=cfg.sky_model.stokes,
        datacolumn='data',
        normtype=cfg.prediction.normtype,
        wbawp=True,
        pblimit=0.05,
        conjbeams=False,
    )

    if img.deconvolver == 'mtmfs' and img.nterms is not None:
        tclean_args['nterms'] = img.nterms

    os.system(f'rm -rf {imagename}.*')
    tclean(**tclean_args)
    log.info("[sanity] tclean complete: %s specmode=%s gridder=%s", imagename, specmode, gridder)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _extract_metrics(cfg: "SimConfig", ia, imstat_fn) -> dict:
    """
    Compute fidelity metrics from sanity imaging outputs.

    Metrics:
      - residual_rms: per channel (per Stokes if IQUV)
      - peak_intensity: per Stokes (channel 0)
      - pb_at_peak: PB value at peak pixel
      - rm_estimate: recovered RM (if faraday.enabled)
    """
    imagename = f"{cfg.name}_sanity"
    metrics = {}

    # ---- Residual RMS per channel ----------------------------------------
    residual_path = f"{imagename}.residual"
    if os.path.exists(residual_path):
        try:
            # axes=[0,1] gives per-channel statistics
            rstat = imstat_fn(residual_path, axes=[0, 1])
            metrics['residual_rms'] = list(rstat.get('rms', []))
            log.info("[sanity] Residual RMS (all channels): %s",
                     [f"{v:.4e}" for v in metrics['residual_rms']])
        except Exception as e:
            log.warning("[sanity] imstat on residual failed: %s", e)
            metrics['residual_rms'] = []
    else:
        log.warning("[sanity] Residual image not found: %s", residual_path)
        metrics['residual_rms'] = []

    # ---- Peak intensity per Stokes (channel 0) ---------------------------
    image_path = f"{imagename}.image"
    if os.path.exists(image_path):
        try:
            peaks = {}
            stokes_labels = _stokes_labels(cfg.sky_model.stokes)
            for s_idx, label in enumerate(stokes_labels):
                # imstat on stokes+channel slice — no full-cube getchunk needed
                stat = imstat_fn(image_path, axes=[0, 1],
                                 stokes=label, chans='0')
                if stat.get('max'):
                    peaks[label] = float(np.max(np.abs(stat['max'])))
            metrics['peak_intensity'] = peaks
            log.info("[sanity] Peak intensity chan0: %s", peaks)

            # PB value at peak Stokes I pixel — locate via imstat maxpos
            pb_path = f"{imagename}.pb"
            if os.path.exists(pb_path):
                stat_i = imstat_fn(image_path, stokes='I', chans='0')
                maxpos = stat_i.get('maxpos', [])
                if len(maxpos) >= 2:
                    px, py = int(maxpos[0]), int(maxpos[1])
                    ia.open(pb_path)
                    pb_plane = ia.getchunk(blc=[px, py, 0, 0], trc=[px, py, 0, 0])
                    ia.close()
                    metrics['pb_at_peak'] = float(pb_plane.flat[0])
                    log.info("[sanity] PB at peak: %.4f", metrics['pb_at_peak'])

        except Exception as e:
            log.warning("[sanity] Image metric extraction failed: %s", e)

    # ---- RM estimate (Faraday runs only) ---------------------------------
    if (cfg.sky_model.faraday and cfg.sky_model.faraday.enabled
            and os.path.exists(image_path)):
        try:
            rm_est = _estimate_rm(image_path, cfg, ia)
            metrics['rm_estimate_rad_per_m2'] = rm_est
            log.info("[sanity] Recovered RM estimate: %.2f rad/m^2", rm_est)
        except Exception as e:
            log.warning("[sanity] RM estimation failed: %s", e)

    return metrics


# ---------------------------------------------------------------------------
# RM estimation
# ---------------------------------------------------------------------------

def _estimate_rm(imagename: str, cfg: "SimConfig", ia) -> float:
    """
    Estimate RM from Q/U spectrum at peak Stokes I pixel.

    Method:
      1. Find peak pixel in Stokes I (channel 0)
      2. Extract Q, U spectra at that pixel
      3. PA = 0.5 * arctan2(U, Q) per channel
      4. Unwrap with np.unwrap() before fitting
      5. Fit PA vs lambda^2 with np.polyfit(lam2, PA, 1) — slope is RM

    Returns:
      RM in rad/m^2
    """
    from .skymodel import _get_chan_freqs_from_csys

    if cfg.sky_model.stokes != 'IQUV':
        raise ValueError(
            f"RM estimation requires stokes=IQUV. Got: {cfg.sky_model.stokes}"
        )

    stokes_idx = {'I': 0, 'Q': 1, 'U': 2, 'V': 3}
    i_idx, q_idx, u_idx = stokes_idx['I'], stokes_idx['Q'], stokes_idx['U']

    # Find peak Stokes I pixel via imstat — avoids loading the full cube
    from casatasks import imstat as _imstat
    stat_i = _imstat(imagename, stokes='I', chans='0')
    maxpos = stat_i.get('maxpos', [])
    if len(maxpos) < 2:
        raise ValueError("imstat did not return a valid maxpos for Stokes I")
    px, py = int(maxpos[0]), int(maxpos[1])

    # Read the coordinate system and shape for frequency axis
    ia.open(imagename)
    csys = ia.coordsys()
    shp = ia.shape()

    # Extract Q and U spectra at peak pixel only — one row of the cube
    nchan = shp[3]
    nstokes = shp[2]
    q_spec_col = ia.getchunk(blc=[px, py, q_idx, 0], trc=[px, py, q_idx, nchan - 1])
    u_spec_col = ia.getchunk(blc=[px, py, u_idx, 0], trc=[px, py, u_idx, nchan - 1])
    ia.close()

    Q_spec = q_spec_col.ravel()
    U_spec = u_spec_col.ravel()

    # PA per channel — unwrap in 2*PA domain (2π periodicity) to handle nπ ambiguity
    two_PA = np.arctan2(U_spec, Q_spec)       # 2χ, range [-π, π]
    two_PA_unwrapped = np.unwrap(two_PA)       # unwrap with default period=2π
    PA_unwrapped = two_PA_unwrapped / 2.0      # back to PA

    # lambda^2 per channel
    chan_freqs = _get_chan_freqs_from_csys(csys, shp[3])
    lam2 = (_C_LIGHT / chan_freqs) ** 2

    # Linear fit: PA = RM * lambda^2 + chi_0  → slope = RM
    coeffs = np.polyfit(lam2, PA_unwrapped, 1)
    rm_est = coeffs[0]

    return float(rm_est)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_sanity_log(metrics: dict, log_path: str) -> None:
    """Serialize metrics dict to JSON."""
    with open(log_path, 'w') as fh:
        json.dump(metrics, fh, indent=2)
    log.info("[sanity] Metrics written: %s", log_path)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _stokes_labels(stokes_str: str) -> list:
    """Return list of Stokes plane labels for a given sky_model.stokes."""
    if stokes_str == 'IQUV':
        return ['I', 'Q', 'U', 'V']
    return ['I']
