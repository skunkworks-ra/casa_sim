from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from .utils import _get_chan_freqs_from_csys, _sky_to_pixel

if TYPE_CHECKING:
    from ..config import SkyModelConfig

log = logging.getLogger(__name__)


def _apply_spectral_extrapolation(image_path: str, sm_cfg: "SkyModelConfig",
                                   out_path: str, ia, qa) -> None:
    """
    Scale each spatial plane per channel using power-law spectral index.
    """
    from ..config import _parse_freq_to_hz

    ia.open(image_path)
    csys = ia.coordsys()
    shp = ia.shape()             # [nx, ny, nstokes, nchan]
    nx, ny, nstokes, nchan = shp

    chan_freqs = _get_chan_freqs_from_csys(csys, nchan)
    nu0 = _parse_freq_to_hz(sm_cfg.ref_freq)
    ref_chan = int(np.argmin(np.abs(chan_freqs - nu0)))

    blc_ref = [0, 0, 0, ref_chan]
    trc_ref = [nx - 1, ny - 1, nstokes - 1, ref_chan]
    ref_plane = ia.getchunk(blc=blc_ref, trc=trc_ref)  # [nx, ny, nstokes, 1]
    ia.close()

    if sm_cfg.alpha_mode != 'global':
        ia.open(sm_cfg.alpha_value)
        alpha_map = ia.getchunk()
        ia.close()
        alpha = alpha_map[:, :, 0, 0] if alpha_map.ndim == 4 else alpha_map
    else:
        alpha = float(sm_cfg.alpha_value)

    os.system(f'rm -rf {out_path}')
    ia.fromshape(out_path, list(shp), csys=csys.torecord(), overwrite=True)

    for chan in range(nchan):
        scale = (chan_freqs[chan] / nu0) ** alpha
        out_plane = ref_plane * scale[..., np.newaxis, np.newaxis] if np.ndim(scale) == 2 \
                    else ref_plane * scale
        ia.putchunk(out_plane, blc=[0, 0, 0, chan])

    ia.close()
    log.info("[skymodel] Spectral extrapolation complete: ref_chan=%d ref_freq=%.4eHz",
             ref_chan, nu0)


def _inject_spectral_lines(image_path: str, sm_cfg: "SkyModelConfig",
                            out_path: str, ia, me, qa) -> None:
    """Add spectral line flux to specific channel planes."""
    from .utils import _get_stokes_indices_from_csys

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
    """Generate a 2D Gaussian brightness distribution normalized to unit peak."""
    nx, ny = shape[0], shape[1]
    inc = csys.increment()['numeric']
    cell_ra_rad = abs(inc[0])
    cell_dec_rad = abs(inc[1])

    major_rad = qa.convert(qa.quantity(line.major), 'rad')['value']
    minor_rad = qa.convert(qa.quantity(line.minor), 'rad')['value']
    pa_rad = qa.convert(qa.quantity(line.pa), 'rad')['value']

    sigma_major_pix = (major_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))) / cell_dec_rad
    sigma_minor_pix = (minor_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))) / cell_ra_rad

    cx = nx // 2
    cy = ny // 2

    y_idx, x_idx = np.mgrid[0:nx, 0:ny]
    dx = x_idx - cx
    dy = y_idx - cy

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
