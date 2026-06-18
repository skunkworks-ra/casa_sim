from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from .component_list import (
    _apply_cl_stokes_spectrum, _build_rm_map_from_sources, build_component_list,
)
from .faraday import _apply_faraday_rotation
from .spectral import _apply_spectral_extrapolation, _inject_spectral_lines
from .trecs import _build_trecs_sky_model
from .utils import _get_chan_freqs_from_csys, _get_stokes_indices_from_csys

if TYPE_CHECKING:
    from ..config import SimConfig

log = logging.getLogger(__name__)


def resolve_sky_model(cfg: "SimConfig", ia, cl, qa, me,
                      msname: str = None, tb=None) -> str:
    """
    Execute the full sky model resolution pipeline.

    Returns path to the final sky model image (or cl path for ft_dft).
    """
    from ..config import _resolve_predictor_auto, FaradayConfig

    sm_cfg = cfg.sky_model
    effective_predictor = (_resolve_predictor_auto(cfg)
                           if cfg.prediction.predictor == 'auto'
                           else cfg.prediction.predictor)

    # ---- Pre-step: build .cl from inline sources if provided -------------

    if sm_cfg.mode == 'component_list' and sm_cfg.sources and not sm_cfg.cl_path:
        auto_cl_path = f"{cfg.name}_sources.cl"
        build_component_list(sm_cfg.sources, auto_cl_path, cl)
        sm_cfg.cl_path = auto_cl_path

        any_rm = any(src.rm != 0.0 for src in sm_cfg.sources)
        if any_rm and (sm_cfg.faraday is None or not sm_cfg.faraday.enabled):
            all_same_rm = len(set(src.rm for src in sm_cfg.sources if src.rm != 0.0)) == 1
            ref_freq = sm_cfg.sources[0].ref_freq
            if all_same_rm:
                rm_val = next(src.rm for src in sm_cfg.sources if src.rm != 0.0)
                sm_cfg.faraday = FaradayConfig(
                    enabled=True, rm_mode='global',
                    rm_value=rm_val, ref_freq=ref_freq,
                )
                log.info("[skymodel] Auto-configured Faraday: global RM=%.2f ref_freq=%s",
                         rm_val, ref_freq)
            else:
                rm_map_path = _build_rm_map_from_sources(
                    sm_cfg.sources, cfg, ia, qa, me)
                sm_cfg.faraday = FaradayConfig(
                    enabled=True, rm_mode='map',
                    rm_value=rm_map_path, ref_freq=ref_freq,
                )
                log.info("[skymodel] Auto-configured Faraday: RM map=%s ref_freq=%s",
                         rm_map_path, ref_freq)

    # ---- Sub-stage 4a: Base model ----------------------------------------

    if sm_cfg.mode == 'component_list':
        needs_image = (sm_cfg.faraday and sm_cfg.faraday.enabled) or sm_cfg.spectral_lines
        if effective_predictor in ('ft_dft', 'sm_predict') and not needs_image:
            log.info("[skymodel] 4a: component_list + %s → returning cl path: %s",
                     effective_predictor, sm_cfg.cl_path)
            if sm_cfg.cl_stokes_spectrum:
                _apply_cl_stokes_spectrum(sm_cfg, cl)
            return sm_cfg.cl_path

        fromcl_path = f"{cfg.name}_skymodel_fromcl.im"
        _make_empty_image(cfg, ia, qa, fromcl_path, msname, tb)
        _eval_complist(sm_cfg.cl_path, fromcl_path, cl, ia)
        log.info("[skymodel] 4a: component_list evaluated to image: %s", fromcl_path)
        current_image = fromcl_path

    elif sm_cfg.mode == 'image_native':
        current_image = sm_cfg.image_path
        log.info("[skymodel] 4a: image_native → %s", current_image)

    elif sm_cfg.mode == 'image_extrapolate':
        extrap_path = f"{cfg.name}_skymodel_extrapolated.im"
        _apply_spectral_extrapolation(sm_cfg.image_path, sm_cfg, extrap_path, ia, qa)
        log.info("[skymodel] 4a: image_extrapolate → %s", extrap_path)
        current_image = extrap_path

    elif sm_cfg.mode == 't_recs':
        trecs_path = f"{cfg.name}_skymodel_trecs.im"
        _build_trecs_sky_model(cfg, sm_cfg.trecs, trecs_path, ia, qa, msname, tb)
        log.info("[skymodel] 4a: t_recs sky model → %s", trecs_path)
        current_image = trecs_path

    elif sm_cfg.mode == 'corpus_mix':
        corpus_path = f"{cfg.name}_skymodel_corpus.im"
        _build_corpus_mix_sky_model(cfg, sm_cfg, corpus_path, ia, qa, msname, tb)
        log.info("[skymodel] 4a: corpus_mix sky model → %s", corpus_path)
        current_image = corpus_path

    else:
        raise ValueError(f"Unknown sky_model.mode: '{sm_cfg.mode}'")

    # ---- Sub-stage 4b: Faraday rotation ----------------------------------

    if sm_cfg.faraday and sm_cfg.faraday.enabled:
        faraday_path = f"{cfg.name}_skymodel_faraday.im"
        _apply_faraday_rotation(current_image, sm_cfg, faraday_path, ia, qa)
        log.info("[skymodel] 4b: Faraday rotation applied → %s", faraday_path)
        current_image = faraday_path

    # ---- Sub-stage 4c: Spectral line injection ---------------------------

    if sm_cfg.spectral_lines:
        lines_path = f"{cfg.name}_skymodel_withlines.im"
        _inject_spectral_lines(current_image, sm_cfg, lines_path, ia, me, qa)
        log.info("[skymodel] 4c: Spectral lines injected → %s", lines_path)
        current_image = lines_path

    log.info("[skymodel] Final sky model image: %s", current_image)
    return current_image


def _make_empty_image(cfg: "SimConfig", ia, qa, out_path: str,
                      msname: str = None, tb=None) -> None:
    """
    Create an empty CASA image sized to match the simulation configuration.
    """
    cell_str = cfg.effective_cell
    imsize = cfg.effective_imsize

    if cell_str is None or imsize is None:
        raise ValueError(
            "cell and imsize must be derived before calling _make_empty_image(). "
            "Call derive_imaging_params() first."
        )

    first_spw = cfg.observation.spws[0]
    first_field = cfg.observation.fields[0]
    n_stokes = 4 if cfg.sky_model.stokes == 'IQUV' else 1
    nchan = first_spw.nchan

    parts = first_field.direction.strip().split()
    ra_str, dec_str = parts[1], parts[2]

    if msname is not None and tb is not None:
        chan_freqs = _read_ms_chan_freqs(msname, tb)
        ref_freq_hz = chan_freqs[0]
        if len(chan_freqs) > 1:
            chan_width_hz = chan_freqs[1] - chan_freqs[0]
        else:
            chan_width_hz = qa.convert(qa.quantity(first_spw.deltafreq), 'Hz')['value']
        nchan = len(chan_freqs)
        log.info("[skymodel] Image freq grid from MS: ref=%.6e Hz, width=%.6e Hz, nchan=%d",
                 ref_freq_hz, chan_width_hz, nchan)
    else:
        ref_freq_hz = qa.convert(qa.quantity(first_spw.freq), 'Hz')['value']
        chan_width_hz = qa.convert(qa.quantity(first_spw.deltafreq), 'Hz')['value']

    os.system(f'rm -rf {out_path}')
    ia.close()
    ia.fromshape(out_path, [imsize, imsize, n_stokes, nchan], overwrite=True)

    cs = ia.coordsys()
    cs.setunits(['rad', 'rad', '', 'Hz'])

    cell_rad = qa.convert(qa.quantity(cell_str), 'rad')['value']
    cs.setincrement([-cell_rad, cell_rad], 'direction')

    ra_rad = qa.convert(qa.quantity(ra_str), 'rad')['value']
    dec_rad = qa.convert(qa.quantity(dec_str), 'rad')['value']
    cs.setreferencevalue([ra_rad, dec_rad], type='direction')

    cs.setreferencevalue(f'{ref_freq_hz}Hz', 'spectral')
    cs.setreferencepixel([0], 'spectral')
    cs.setincrement(f'{chan_width_hz}Hz', 'spectral')

    ia.setcoordsys(cs.torecord())
    ia.setbrightnessunit('Jy/pixel')
    ia.set(0.0)
    ia.close()


def _read_ms_chan_freqs(msname: str, tb) -> "np.ndarray":
    """Read channel frequencies (Hz) from the MS SPECTRAL_WINDOW table."""
    tb.open(msname + '/SPECTRAL_WINDOW')
    chan_freqs = tb.getcol('CHAN_FREQ')  # shape: [nchan, nspw]
    tb.close()
    return chan_freqs[:, 0]


def _eval_complist(cl_path: str, im_path: str, cl, ia) -> None:
    """Evaluate a component list onto an existing empty image."""
    cl.open(cl_path)
    ia.open(im_path)
    ia.modify(cl.torecord(), subtract=False)
    ia.close()
    cl.done()


def _build_corpus_mix_sky_model(cfg, sm_cfg, out_path: str, ia, qa,
                                 msname: str = None, tb=None) -> None:
    """
    Build the corpus_mix sky model: T-RECS compact sources + extended morphology.

    Assembly steps:
      1. Paint T-RECS sources into a full-Stokes IQUV CASA image (identical to
         the t_recs branch).
      2. Build the extended morphology field via build_field() — a (imsize, imsize)
         float32 numpy array in Jy/pixel.  The cell_arcsec is read from the config
         so source angular sizes are config-appropriate.
      3. Add the morphology into the Stokes-I channel(s) of the CASA image.
         Non-negativity is preserved because both components are non-negative.

    The morphology field type is either fixed (corpus_morphology.field_type) or
    sampled per call from the balance weights (when field_type is null).

    Notes
    -----
    - The morphology is Stokes-I only (total intensity proxy).  Q/U/V from the
      morphology layer are left as zero; all polarisation comes from T-RECS.
    - POINT_ONLY produces a zero morphology array, so the result is identical to
      pure t_recs for that field type.
    - The T-RECS catalog loading call is the slow step (~seconds for AGN at
      flux_floor_jy > 1 mJy).  See trecs.py for tuning advice.
    """
    from .morphology_field import build_field, FieldType, sample_field_type
    from .utils import _get_stokes_indices_from_csys

    cm_cfg = sm_cfg.corpus_morphology

    # ---- Step 1: paint T-RECS sources into the CASA image -----------------

    _build_trecs_sky_model(cfg, sm_cfg.trecs, out_path, ia, qa, msname, tb)
    log.info("[skymodel/corpus_mix] T-RECS image built: %s", out_path)

    # ---- Step 2: build extended morphology field ---------------------------

    # Derive cell in arcsec from the config (same grid as the CASA image)
    cell_str = cfg.effective_cell
    cell_arcsec = float(qa.convert(qa.quantity(cell_str), 'arcsec')['value'])
    imsize = cfg.effective_imsize

    # Parse field centre from the first observation field
    first_field = cfg.observation.fields[0]
    parts = first_field.direction.strip().split()
    ra_str, dec_str = parts[1], parts[2]
    ra_deg = float(qa.convert(qa.quantity(ra_str), 'deg')['value'])
    dec_deg = float(qa.convert(qa.quantity(dec_str), 'deg')['value'])

    # Representative frequency for the morphology FITS header
    first_spw = cfg.observation.spws[0]
    freq_hz = float(qa.convert(qa.quantity(first_spw.freq), 'Hz')['value'])

    # Determine field type
    rng = np.random.default_rng(cm_cfg.seed)
    if cm_cfg.field_type is not None:
        field_type = FieldType(cm_cfg.field_type)
        log.info("[skymodel/corpus_mix] Using fixed field_type: %s", field_type.value)
    else:
        # Build balance dict mapping FieldType → weight
        if cm_cfg.balance is not None:
            balance = {FieldType(k): v for k, v in cm_cfg.balance.items()}
        else:
            balance = None   # use DEFAULT_FIELD_TYPE_BALANCE
        field_type = sample_field_type(rng, balance=balance)
        log.info("[skymodel/corpus_mix] Sampled field_type: %s", field_type.value)

    # repo_root: explicit override or auto-detect from module location
    repo_root = cm_cfg.repo_root  # may be None → auto-detected in build_field

    morph_image, morph_meta = build_field(
        rng,
        field_type,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        cell_arcsec=cell_arcsec,
        imsize=imsize,
        freq_hz=freq_hz,
        repo_root=repo_root,
    )
    log.info(
        "[skymodel/corpus_mix] Morphology built: type=%s total_flux=%.4g Jy "
        "cell=%.3f arcsec imsize=%d",
        field_type.value, float(morph_image.sum()), cell_arcsec, imsize,
    )

    # ---- Step 3: composite morphology into Stokes I of the CASA image -----

    ia.open(out_path)
    csys = ia.coordsys()
    shp = ia.shape()   # [nx, ny, nstokes, nchan]
    nx, ny, nstokes, nchan = shp

    stokes_idx = _get_stokes_indices_from_csys(csys)
    i_idx = stokes_idx.get('I', 0)

    img_data = ia.getchunk()   # shape [nx, ny, nstokes, nchan]

    # morph_image is (imsize, imsize) = (nx, ny) — same grid by construction.
    # Add into all channels of the I Stokes plane.
    for ch in range(nchan):
        img_data[:, :, i_idx, ch] = (
            img_data[:, :, i_idx, ch] + morph_image.astype(np.float64)
        )

    # Enforce non-negativity (guard: both inputs are non-negative, so this is
    # a no-op under normal operation, but guards against numerical noise)
    img_data[:, :, i_idx, :] = np.clip(img_data[:, :, i_idx, :], 0.0, None)

    ia.putchunk(img_data)
    ia.close()

    log.info("[skymodel/corpus_mix] Morphology composited into Stokes I of %s", out_path)
