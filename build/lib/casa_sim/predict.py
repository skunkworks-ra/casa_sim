"""
predict.py — CASA Simulation Framework, Module 5

Responsibilities:
  - Route to correct predictor (ft_dft | sm_predict | tclean)
  - Execute prediction
  - Copy MODEL_DATA → DATA (except ft_dft which writes directly to DATA)
  - Initialize WEIGHT_SPECTRUM for per_baseline noise mode only

CRITICAL:
  ft_dft writes directly to DATA column.
  copyModelToData() MUST NOT be called for ft_dft.
  This is explicitly guarded below with a comment to prevent future regression.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import SimConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def predict(cfg: "SimConfig", msname: str, sky_model_path: str, sm, tb, mstransform_fn) -> None:
    """
    Route to the correct predictor and execute visibility prediction.

    Args:
        cfg:              SimConfig instance
        msname:           Path to the MS
        sky_model_path:   Path to sky model image or component list
        sm:               CASA simulator tool
        tb:               CASA table tool
        mstransform_fn:   casatasks.mstransform function
    """
    from .config import _resolve_predictor_auto

    effective_predictor = (_resolve_predictor_auto(cfg)
                           if cfg.prediction.predictor == 'auto'
                           else cfg.prediction.predictor)

    # Apply sm_predict → tclean demotion
    if (effective_predictor == 'sm_predict'
            and cfg.prediction.gridder in ('mosaic', 'awproject')):
        log.warning("[predict] sm_predict demoted to tclean for gridder=%s",
                    cfg.prediction.gridder)
        effective_predictor = 'tclean'

    log.info("[predict] Using predictor: %s", effective_predictor)

    if effective_predictor == 'ft_dft':
        # ft_dft writes directly to DATA — DO NOT call copyModelToData() after this
        _predict_ft_dft(msname, sky_model_path)
        # ^^^ DATA column is already populated. copyModelToData() is intentionally skipped.

    elif effective_predictor == 'sm_predict':
        # sm.predict() writes directly to DATA — DO NOT call copyModelToData() after this
        _predict_sm(msname, sky_model_path, sm)

    elif effective_predictor == 'tclean':
        _predict_tclean(msname, sky_model_path, cfg)
        _copy_model_to_data(msname, tb, mstransform_fn,
                            init_weight_spectrum=(cfg.corruption.noise.mode == 'per_baseline'))

    else:
        raise ValueError(f"Unknown predictor '{effective_predictor}'")


# ---------------------------------------------------------------------------
# Predictor implementations
# ---------------------------------------------------------------------------

def _predict_ft_dft(msname: str, cl_path: str) -> None:
    """
    Predict visibilities using the DFT Fourier transform task.
    Writes directly to DATA column — no MODEL_DATA involved.
    """
    from casatasks import ft
    ft(vis=msname, complist=cl_path, incremental=False, usescratch=True)
    log.info("[predict] ft_dft complete: cl=%s → %s", cl_path, msname)


def _predict_sm(msname: str, model_path: str, sm) -> None:
    """
    Predict visibilities using sm.predict() (standard gridder + static PB).
    Accepts either a component list (.cl) or an image path.
    Writes to MODEL_DATA column.
    """
    sm.openfromms(msname)
    if model_path.endswith('.cl'):
        sm.predict(complist=model_path, incremental=False)
    else:
        sm.predict(imagename=model_path, incremental=False)
    sm.close()
    log.info("[predict] sm_predict complete: model=%s → %s", model_path, msname)


def _predict_tclean(msname: str, image_path: str, cfg: "SimConfig") -> None:
    """
    Predict visibilities using tclean with savemodel='modelcolumn'.
    Writes to MODEL_DATA column.
    Supports all gridders.
    """
    from casatasks import tclean

    cell = cfg.effective_cell
    imsize = cfg.effective_imsize

    if cell is None or imsize is None:
        raise ValueError(
            "cell and imsize must be set before tclean prediction. "
            "Call derive_imaging_params() first."
        )

    # Remove stale predict outputs to avoid stale state
    os.system('rm -rf sim_predict.*')

    tclean(
        vis=msname,
        startmodel=image_path,
        imagename='sim_predict',
        savemodel='modelcolumn',
        imsize=imsize,
        cell=cell,
        specmode='cube',
        interpolation='nearest',
        nchan=-1,
        gridder=cfg.prediction.gridder,
        normtype=cfg.prediction.normtype,
        wbawp=True,
        pblimit=0.05,
        conjbeams=False,
        calcres=False,
        calcpsf=True,
        niter=0,
        wprojplanes=1
    )
    log.info("[predict] tclean prediction complete: image=%s gridder=%s → %s",
             image_path, cfg.prediction.gridder, msname)


# ---------------------------------------------------------------------------
# Post-prediction data column management
# ---------------------------------------------------------------------------

def _copy_model_to_data(msname: str, tb, mstransform_fn,
                         init_weight_spectrum: bool = False) -> None:
    """
    Copy MODEL_DATA → DATA, zero MODEL_DATA.
    Optionally initialize WEIGHT_SPECTRUM column for per_baseline noise mode.

    Matches the reference notebook copyModelToData() exactly.
    """
    tb.open(msname, nomodify=False)
    moddata = tb.getcol(columnname='MODEL_DATA')
    tb.putcol(columnname='DATA', value=moddata)
    moddata.fill(0.0)
    tb.putcol(columnname='MODEL_DATA', value=moddata)
    tb.close()
    log.info("[predict] MODEL_DATA → DATA copied, MODEL_DATA zeroed: %s", msname)

    if init_weight_spectrum:
        _init_weight_spectrum(msname, mstransform_fn)


def _init_weight_spectrum(msname: str, mstransform_fn) -> None:
    """
    Initialize WEIGHT_SPECTRUM column via mstransform.
    Called only when corruption.noise.mode == 'per_baseline'.
    """
    tmp = 'tmp_addedcol.ms'
    os.system(f'rm -rf {tmp}')
    mstransform_fn(vis=msname, outputvis=tmp, datacolumn='DATA', usewtspectrum=True)
    os.system(f'rm -rf {msname}')
    os.system(f'cp -r {tmp} {msname}')
    os.system(f'rm -rf {tmp}')
    log.info("[predict] WEIGHT_SPECTRUM initialized: %s", msname)


# ---------------------------------------------------------------------------
# Routing utility (exposed for external use)
# ---------------------------------------------------------------------------

def route_predictor(cfg: "SimConfig") -> str:
    """Return the resolved predictor string for this config."""
    from .config import _resolve_predictor_auto
    effective = (_resolve_predictor_auto(cfg)
                 if cfg.prediction.predictor == 'auto'
                 else cfg.prediction.predictor)
    if (effective == 'sm_predict'
            and cfg.prediction.gridder in ('mosaic', 'awproject')):
        return 'tclean'
    return effective
