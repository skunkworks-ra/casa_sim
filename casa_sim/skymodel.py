"""
skymodel.py — CASA Simulation Framework, Module 4

Responsibilities:
  - Sub-stage 4a: Base model resolution (component list | image_native | image_extrapolate)
  - Sub-stage 4b: Faraday rotation
  - Sub-stage 4c: Spectral line injection
  - Returns path to final sky model image (or cl path for ft_dft)

Design constraints:
  - Speed of light: 2.99792458e8 m/s (no astropy.constants)
  - _apply_faraday_rotation uses ia.getchunk()/ia.putchunk() NOT ia.modify()
  - _get_stokes_indices queries ia.coordsys() — never hardcoded
  - All intermediate images kept on disk and logged at INFO level
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .config import SimConfig, SkyModelConfig, SourceDef

log = logging.getLogger(__name__)

_C_LIGHT = 2.99792458e8   # m/s


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def resolve_sky_model(cfg: "SimConfig", ia, cl, qa, me,
                      msname: str = None, tb=None) -> str:
    """
    Execute the full sky model resolution pipeline.

    Returns:
        Path to the final sky model image handed to the predictor.
        For ft_dft path (component_list + standard gridder), returns the
        component list path directly.

    Side effects:
        Writes intermediate images to disk. Logs all paths at INFO level.
    """
    from .config import _resolve_predictor_auto, FaradayConfig

    sm_cfg = cfg.sky_model
    effective_predictor = (_resolve_predictor_auto(cfg)
                           if cfg.prediction.predictor == 'auto'
                           else cfg.prediction.predictor)

    # ---- Pre-step: build .cl from inline sources if provided -------------

    if sm_cfg.mode == 'component_list' and sm_cfg.sources and not sm_cfg.cl_path:
        # Filter sources to image FoV
        sm_cfg.sources = _filter_sources_by_fov(sm_cfg.sources, cfg, qa)
        if not sm_cfg.sources:
            raise RuntimeError(
                "No sources remain after FoV filter — all sources are outside "
                f"the image ({cfg.effective_imsize}px × {cfg.effective_cell}). "
                "Increase imsize or check source directions."
            )

        auto_cl_path = f"{cfg.name}_sources.cl"
        build_component_list(sm_cfg.sources, auto_cl_path, cl)
        sm_cfg.cl_path = auto_cl_path

        # Auto-configure Faraday from per-source RM values
        any_rm = any(src.rm != 0.0 for src in sm_cfg.sources)
        if any_rm and (sm_cfg.faraday is None or not sm_cfg.faraday.enabled):
            all_same_rm = len(set(src.rm for src in sm_cfg.sources if src.rm != 0.0)) == 1
            # Use first source ref_freq as Faraday ref_freq
            ref_freq = sm_cfg.sources[0].ref_freq
            if all_same_rm:
                # Single RM value → global mode
                rm_val = next(src.rm for src in sm_cfg.sources if src.rm != 0.0)
                sm_cfg.faraday = FaradayConfig(
                    enabled=True, rm_mode='global',
                    rm_value=rm_val, ref_freq=ref_freq,
                )
                log.info("[skymodel] Auto-configured Faraday: global RM=%.2f ref_freq=%s",
                         rm_val, ref_freq)
            else:
                # Multiple distinct RMs → build RM map
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
            # Use the .cl directly — no image needed
            if sm_cfg.cl_path is None:
                raise RuntimeError(
                    "component_list mode but cl_path is None — "
                    "config must provide cl_path, sources, or sources_file"
                )
            log.info("[skymodel] 4a: component_list + %s → returning cl path: %s",
                     effective_predictor, sm_cfg.cl_path)
            if sm_cfg.cl_stokes_spectrum:
                _apply_cl_stokes_spectrum(sm_cfg, cl)
            return sm_cfg.cl_path

        # Faraday/spectral-line path or mosaic/awproject: evaluate CL onto an image
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


# ---------------------------------------------------------------------------
# Sub-stage 4a helpers
# ---------------------------------------------------------------------------

def _make_empty_image(cfg: "SimConfig", ia, qa, out_path: str,
                      msname: str = None, tb=None) -> None:
    """
    Create an empty CASA image sized to match the simulation configuration.
    If msname and tb are provided, reads the actual channel frequencies from the
    MS SPECTRAL_WINDOW table to guarantee the image frequency grid matches the MS.
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

    # Parse direction
    parts = first_field.direction.strip().split()
    ra_str, dec_str = parts[1], parts[2]

    # Read actual channel frequencies from the MS if available
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

    # Spectral axis — use frequencies read from MS
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
    # First SPW, all channels
    return chan_freqs[:, 0]


def _eval_complist(cl_path: str, im_path: str, cl, ia) -> None:
    """Evaluate a component list onto an existing empty image."""
    cl.open(cl_path)
    ia.open(im_path)
    ia.modify(cl.torecord(), subtract=False)
    ia.close()
    cl.done()


def _apply_cl_stokes_spectrum(sm_cfg: "SkyModelConfig", cl) -> None:
    """
    Apply per-component Stokes spectral variation via cl.setstokesspectrum().
    Called only when cl_stokes_spectrum is non-empty.
    """
    cl.open(sm_cfg.cl_path)
    for entry in sm_cfg.cl_stokes_spectrum:
        idx = entry.component_index
        if entry.type == 'spectral_index' and entry.index is not None:
            cl.setstokesspectrum(
                which=idx,
                type='spectral index',
                index=entry.index,
                reffreq=entry.ref_freq
            )
            log.info("[skymodel] cl_stokes_spectrum: component %d spectral index applied", idx)
        elif entry.type == 'tabular':
            # tabular — set per-frequency Stokes values directly
            # CASA componentlist doesn't expose a direct "tabular Stokes" spectrum API;
            # this would require building a custom spectrum record.
            # Flag as non-standard and skip with a warning.
            log.warning("[skymodel] cl_stokes_spectrum tabular mode not supported "
                        "via CASA componentlist API for component %d — skipping", idx)
    cl.done()


# ---------------------------------------------------------------------------
# Sub-stage 4a: Build component list from inline sources
# ---------------------------------------------------------------------------

def _filter_sources_by_fov(sources: list, cfg: "SimConfig", qa) -> list:
    """
    Drop sources outside the image FoV defined by cell and imsize.
    """
    imsize = cfg.effective_imsize
    cell_str = cfg.effective_cell
    if imsize is None or cell_str is None:
        return sources

    cell_deg = qa.convert(qa.quantity(cell_str), 'deg')['value']
    hw_deg = (imsize / 2.0) * cell_deg  # half-width in degrees

    first_field = cfg.observation.fields[0]
    parts = first_field.direction.strip().split()
    pc_ra_deg = qa.convert(qa.quantity(parts[1]), 'deg')['value']
    pc_dec_deg = qa.convert(qa.quantity(parts[2]), 'deg')['value']
    cos_dec = np.cos(np.radians(pc_dec_deg))

    kept = []
    for src in sources:
        sparts = src.direction.strip().split()
        sra = qa.convert(qa.quantity(sparts[1]), 'deg')['value']
        sdec = qa.convert(qa.quantity(sparts[2]), 'deg')['value']
        if abs((sra - pc_ra_deg) * cos_dec) <= hw_deg and abs(sdec - pc_dec_deg) <= hw_deg:
            kept.append(src)

    log.info("[skymodel] FoV filter: kept %d / %d sources (imsize=%d cell=%s)",
             len(kept), len(sources), imsize, cell_str)
    return kept


def build_component_list(sources: list, cl_path: str, cl) -> None:
    """
    Build a CASA component list from inline SourceDef entries.

    For each source:
      - flux: [I] → Stokes I only; [I,Q,U,V] → full Stokes
      - If frac_pol + chi given (and flux is [I]): derive Q = I*p*cos(2χ), U = I*p*sin(2χ)
      - spectral_index: [alpha] or [alpha, beta] (curvature)
      - shape: point | gaussian | disk

    Args:
        sources: list of SourceDef dataclass instances
        cl_path: output path for the .cl table directory
        cl:      CASA componentlist tool instance
    """
    import math
    os.system(f'rm -rf {cl_path}')
    cl.done()

    for src in sources:
        # Resolve flux vector
        flux = list(src.flux)
        if len(flux) == 1 and src.frac_pol is not None:
            I = flux[0]
            p = src.frac_pol
            chi_rad = math.radians(src.chi)
            Q = I * p * math.cos(2.0 * chi_rad)
            U = I * p * math.sin(2.0 * chi_rad)
            V = 0.0
            flux = [I, Q, U, V]
            log.info("[skymodel] Source '%s': derived IQUV=[%.4f, %.4f, %.4f, %.4f] "
                     "from frac_pol=%.4f chi=%.1f°",
                     src.name, I, Q, U, V, p, src.chi)

        # Scalar flux for Stokes-I-only, list for full Stokes
        if len(flux) == 1:
            flux_arg = flux[0]
        else:
            flux_arg = flux

        # Spectral index: CASA takes scalar or list
        # [alpha] → scalar, [alpha, beta] → list (curvature term)
        spix = src.spectral_index
        index_arg = spix[0] if len(spix) == 1 else spix

        # Shape parameters
        shape_kwargs = {}
        if src.shape in ('gaussian', 'disk'):
            shape_kwargs['majoraxis'] = src.major
            shape_kwargs['minoraxis'] = src.minor
            shape_kwargs['positionangle'] = src.pa or '0deg'

        cl.addcomponent(
            dir=src.direction,
            flux=flux_arg,
            fluxunit='Jy',
            freq=src.ref_freq,
            shape=src.shape,
            spectrumtype='spectral index',
            index=index_arg,
            **shape_kwargs,
        )
        log.info("[skymodel] Source '%s': flux=%s spix=%s shape=%s @ %s",
                 src.name, flux, spix, src.shape, src.direction)

    cl.rename(filename=cl_path)
    cl.done()
    log.info("[skymodel] Component list built: %s (%d sources)", cl_path, len(sources))


def _build_rm_map_from_sources(sources: list, cfg: "SimConfig",
                                ia, qa, me) -> str:
    """
    Build a 2D RM map image from per-source RM values.

    Each source's RM is placed at its sky position in the image. The map
    is used by the existing _apply_faraday_rotation() with rm_mode='map'.

    Returns path to the RM map image.
    """
    rm_path = f"{cfg.name}_rm_map.im"
    cell_str = cfg.effective_cell
    imsize = cfg.effective_imsize

    first_spw = cfg.observation.spws[0]
    first_field = cfg.observation.fields[0]

    # Parse field center direction
    parts = first_field.direction.strip().split()
    ra_str, dec_str = parts[1], parts[2]

    os.system(f'rm -rf {rm_path}')
    ia.close()
    # RM map: [nx, ny, 1, 1] — single Stokes, single channel
    ia.fromshape(rm_path, [imsize, imsize, 1, 1], overwrite=True)

    cs = ia.coordsys()
    cs.setunits(['rad', 'rad', '', 'Hz'])

    cell_rad = qa.convert(qa.quantity(cell_str), 'rad')['value']
    cs.setincrement([-cell_rad, cell_rad], 'direction')

    ra_rad = qa.convert(qa.quantity(ra_str), 'rad')['value']
    dec_rad = qa.convert(qa.quantity(dec_str), 'rad')['value']
    cs.setreferencevalue([ra_rad, dec_rad], type='direction')

    freq_hz = qa.convert(qa.quantity(first_spw.freq), 'Hz')['value']
    cs.setreferencevalue(f'{freq_hz}Hz', 'spectral')

    ia.setcoordsys(cs.torecord())
    ia.set(0.0)   # initialize to zero RM everywhere

    pix = ia.getchunk()

    for src in sources:
        if src.rm == 0.0:
            continue
        px, py = _sky_to_pixel(src.direction, cs, qa, me)
        # Clamp to image bounds
        px = max(0, min(px, imsize - 1))
        py = max(0, min(py, imsize - 1))
        pix[px, py, 0, 0] = src.rm
        log.info("[skymodel] RM map: source '%s' RM=%.2f rad/m² at pixel (%d,%d)",
                 src.name, src.rm, px, py)

    ia.putchunk(pix)
    ia.close()
    log.info("[skymodel] RM map built: %s", rm_path)
    return rm_path


# ---------------------------------------------------------------------------
# Sub-stage 4a: Spectral extrapolation
# ---------------------------------------------------------------------------

def _apply_spectral_extrapolation(image_path: str, sm_cfg: "SkyModelConfig",
                                   out_path: str, ia, qa) -> None:
    """
    Scale each spatial plane per channel using power-law spectral index.
    Operates per Stokes plane independently.

    Physical caveat (per design doc): correct for optically thin emission only.
    """
    from .config import _parse_freq_to_hz

    ia.open(image_path)
    pix = ia.getchunk()          # [nx, ny, nstokes, nchan]
    csys = ia.coordsys()
    shp = ia.shape()
    ia.close()

    chan_freqs = _get_chan_freqs_from_csys(csys, shp[3])
    nu0 = _parse_freq_to_hz(sm_cfg.ref_freq)

    # Find reference channel (nearest to ref_freq)
    ref_chan = int(np.argmin(np.abs(chan_freqs - nu0)))

    out_pix = np.zeros_like(pix)

    if sm_cfg.alpha_mode == 'global':
        alpha = float(sm_cfg.alpha_value)
        for stokes_plane in range(shp[2]):
            for chan in range(shp[3]):
                scale = (chan_freqs[chan] / nu0) ** alpha
                out_pix[:, :, stokes_plane, chan] = (
                    pix[:, :, stokes_plane, ref_chan] * scale
                )
    else:  # map
        # alpha_value is a path to an alpha image [nx, ny, 1, 1] or [nx, ny]
        ia.open(sm_cfg.alpha_value)
        alpha_map = ia.getchunk()
        ia.close()
        # Squeeze to [nx, ny]
        alpha_2d = alpha_map[:, :, 0, 0] if alpha_map.ndim == 4 else alpha_map

        for stokes_plane in range(shp[2]):
            for chan in range(shp[3]):
                scale = (chan_freqs[chan] / nu0) ** alpha_2d
                out_pix[:, :, stokes_plane, chan] = (
                    pix[:, :, stokes_plane, ref_chan] * scale
                )

    # Write output
    os.system(f'rm -rf {out_path}')
    ia.fromarray(outfile=out_path, pixels=out_pix,
                 csys=csys.torecord(), overwrite=True)
    ia.close()
    log.info("[skymodel] Spectral extrapolation complete: ref_chan=%d ref_freq=%.4eHz",
             ref_chan, nu0)


# ---------------------------------------------------------------------------
# Sub-stage 4b: Faraday rotation
# ---------------------------------------------------------------------------

def _apply_faraday_rotation(image_path: str, sm_cfg: "SkyModelConfig",
                             out_path: str, ia, qa) -> None:
    """
    Apply Faraday rotation to Q and U planes per channel.

    Convention (IAU, Condon & Ransom):
      RM > 0 → PA rotates toward positive PA with increasing lambda^2
      delta_chi = RM * (lambda^2 - lambda_0^2)
      Zero net rotation at channel nearest ref_freq.

    Reads from image_path, writes rotated result to out_path.
    Does NOT modify image_path.
    """
    from .config import _parse_freq_to_hz

    ia.open(image_path)
    pix = ia.getchunk()          # [nx, ny, nstokes, nchan]
    csys = ia.coordsys()
    shp = ia.shape()
    ia.close()

    stokes_idx = _get_stokes_indices_from_csys(csys)
    if 'Q' not in stokes_idx or 'U' not in stokes_idx:
        raise ValueError(
            "Faraday rotation requires Q and U Stokes planes. "
            f"Found: {list(stokes_idx.keys())}"
        )

    chan_freqs = _get_chan_freqs_from_csys(csys, shp[3])

    # Reference wavelength^2
    nu0 = _parse_freq_to_hz(sm_cfg.faraday.ref_freq)
    lam0_sq = (_C_LIGHT / nu0) ** 2

    # RM: scalar or 2D map
    if sm_cfg.faraday.rm_mode == 'global':
        rm = float(sm_cfg.faraday.rm_value)
    else:
        ia_tmp = ia.__class__()   # new image tool instance for RM map
        ia_tmp.open(sm_cfg.faraday.rm_value)
        rm_arr = ia_tmp.getchunk()
        ia_tmp.close()
        rm = rm_arr[:, :, 0, 0] if rm_arr.ndim == 4 else rm_arr

    q_idx = stokes_idx['Q']
    u_idx = stokes_idx['U']

    # Apply rotation to a copy — never modify the input image
    out_pix = pix.copy()
    for chan in range(shp[3]):
        lam_sq = (_C_LIGHT / chan_freqs[chan]) ** 2
        angle = rm * (lam_sq - lam0_sq)   # delta_chi in radians

        Q0 = pix[:, :, q_idx, chan]
        U0 = pix[:, :, u_idx, chan]

        out_pix[:, :, q_idx, chan] = Q0 * np.cos(2.0 * angle) - U0 * np.sin(2.0 * angle)
        out_pix[:, :, u_idx, chan] = Q0 * np.sin(2.0 * angle) + U0 * np.cos(2.0 * angle)

    os.system(f'rm -rf {out_path}')
    ia.fromarray(outfile=out_path, pixels=out_pix,
                 csys=csys.torecord(), overwrite=True)
    ia.close()

    log.info("[skymodel] Faraday rotation applied: rm_mode=%s ref_freq=%s nchan=%d",
             sm_cfg.faraday.rm_mode, sm_cfg.faraday.ref_freq, shp[3])


# ---------------------------------------------------------------------------
# Sub-stage 4c: Spectral line injection
# ---------------------------------------------------------------------------

def _inject_spectral_lines(image_path: str, sm_cfg: "SkyModelConfig",
                            out_path: str, ia, me, qa) -> None:
    """
    Add spectral line flux to specific channel planes.
    Modes: point, gaussian, image.
    """
    os.system(f'rm -rf {out_path}')
    os.system(f'cp -r {image_path} {out_path}')

    ia.open(out_path)
    pix = ia.getchunk()
    csys = ia.coordsys()
    stokes_idx = _get_stokes_indices_from_csys(csys)

    for line in sm_cfg.spectral_lines:
        s_idx = stokes_idx.get(line.stokes)
        if s_idx is None:
            log.warning("[skymodel] Spectral line '%s': stokes '%s' not in image — skipping",
                        line.name, line.stokes)
            continue

        for i, chan in enumerate(line.channels):
            flux = line.flux_profile[i]

            if line.mode == 'point':
                px, py = _sky_to_pixel(line.direction, csys, qa, me)
                pix[px, py, s_idx, chan] += flux
                log.info("[skymodel] Line '%s' chan %d: point flux %.4f Jy at (%d,%d)",
                         line.name, chan, flux, px, py)

            elif line.mode == 'gaussian':
                gauss = _gaussian_plane(line, csys, ia, qa, i, pix.shape)
                pix[:, :, s_idx, chan] += gauss * flux
                log.info("[skymodel] Line '%s' chan %d: gaussian flux %.4f Jy",
                         line.name, chan, flux)

            elif line.mode == 'image':
                line_pix = _load_image_plane(line.image_path, chan, ia)
                pix[:, :, s_idx, chan] += line_pix
                log.info("[skymodel] Line '%s' chan %d: image plane added", line.name, chan)

    ia.putchunk(pix)
    ia.close()
    log.info("[skymodel] Spectral line injection complete: %d line(s)", len(sm_cfg.spectral_lines))


def _gaussian_plane(line, csys, ia, qa, flux_index: int, shape: tuple) -> np.ndarray:
    """
    Generate a 2D Gaussian brightness distribution normalized to unit peak.
    Caller multiplies by flux.
    """
    nx, ny = shape[0], shape[1]
    # Pixel scale from coordinate system
    inc = csys.increment()['numeric']   # radians per pixel [ra, dec, ...]
    cell_ra_rad = abs(inc[0])
    cell_dec_rad = abs(inc[1])

    major_rad = qa.convert(qa.quantity(line.major), 'rad')['value']
    minor_rad = qa.convert(qa.quantity(line.minor), 'rad')['value']
    pa_rad = qa.convert(qa.quantity(line.pa), 'rad')['value']

    sigma_major_pix = (major_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))) / cell_dec_rad
    sigma_minor_pix = (minor_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))) / cell_ra_rad

    # Reference pixel from center of image
    cx = nx // 2
    cy = ny // 2

    y_idx, x_idx = np.mgrid[0:nx, 0:ny]
    dx = x_idx - cx
    dy = y_idx - cy

    # Rotate by position angle
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    dx_rot = dx * cos_pa - dy * sin_pa
    dy_rot = dx * sin_pa + dy * cos_pa

    gauss = np.exp(-0.5 * ((dx_rot / sigma_minor_pix) ** 2 +
                            (dy_rot / sigma_major_pix) ** 2))
    return gauss


def _load_image_plane(image_path: str, chan: int, ia) -> np.ndarray:
    """Load a single channel plane from an image."""
    ia.open(image_path)
    shp = ia.shape()
    pix = ia.getchunk(blc=[0, 0, 0, chan], trc=[shp[0], shp[1], 0, chan])
    ia.close()
    return pix[:, :, 0, 0]


# ---------------------------------------------------------------------------
# Coordinate system utilities
# ---------------------------------------------------------------------------

def _get_chan_freqs(image_path: str, ia) -> np.ndarray:
    """Return array of channel centre frequencies in Hz."""
    ia.open(image_path)
    csys = ia.coordsys()
    shp = ia.shape()
    nchan = shp[3]
    ia.close()
    return _get_chan_freqs_from_csys(csys, nchan)


def _get_chan_freqs_from_csys(csys, nchan: int) -> np.ndarray:
    """Extract channel frequencies in Hz from an open coordinate system."""
    ref_val = csys.referencevalue()['numeric']
    inc = csys.increment()['numeric']
    ref_pix = csys.referencepixel()['numeric']

    # Axis ordering: [RA, DEC, Stokes, Freq] — frequency is axis index 3
    # Find spectral axis index from axis coordinate types
    axis_types = csys.axiscoordinatetypes()
    try:
        spec_axis = axis_types.index('Spectral')
    except ValueError:
        # Fallback: assume axis 3
        spec_axis = 3

    freq0 = ref_val[spec_axis]
    dfreq = inc[spec_axis]
    rpix = ref_pix[spec_axis]

    return np.array([freq0 + (i - rpix) * dfreq for i in range(nchan)])


def _get_stokes_indices(image_path: str, ia) -> dict:
    """Return dict mapping Stokes name to pixel index, e.g. {'I':0,'Q':1,'U':2,'V':3}."""
    ia.open(image_path)
    csys = ia.coordsys()
    shp = ia.shape()
    ia.close()
    return _get_stokes_indices_from_csys(csys, n_stokes=shp[2])


def _get_stokes_indices_from_csys(csys, n_stokes: int = None) -> dict:
    """
    Build Stokes index dict from csys.stokes() — returns e.g. ['I','Q','U','V'].
    """
    try:
        labels = csys.stokes()
        return {name: i for i, name in enumerate(labels)}
    except Exception:
        return {}


def _sky_to_pixel(direction_str: str, csys, qa, me) -> tuple:
    """
    Convert a J2000 direction string to (px, py) integer pixel coordinates.
    Uses csys.topixel().
    """
    parts = direction_str.strip().split()
    # Build a world coordinate vector [RA_rad, Dec_rad, Stokes_0, Freq_0]
    ra_rad = qa.convert(qa.quantity(parts[1]), 'rad')['value']
    dec_rad = qa.convert(qa.quantity(parts[2]), 'rad')['value']

    # csys.topixel() needs a complete world vector matching number of axes
    n_axes = len(csys.referencevalue()['numeric'])
    world = list(csys.referencevalue()['numeric'])
    world[0] = ra_rad
    world[1] = dec_rad

    pixel = csys.topixel(world)['numeric']
    px = int(round(pixel[0]))
    py = int(round(pixel[1]))
    return px, py
