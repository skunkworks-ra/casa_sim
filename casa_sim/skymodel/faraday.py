from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from .utils import _C_LIGHT, _get_chan_freqs_from_csys, _get_stokes_indices_from_csys

if TYPE_CHECKING:
    from ..config import SkyModelConfig

log = logging.getLogger(__name__)


def _apply_faraday_rotation(image_path: str, sm_cfg: "SkyModelConfig",
                             out_path: str, ia, qa) -> None:
    """
    Apply Faraday rotation to Q and U planes per channel.

    Convention (IAU, Condon & Ransom):
      delta_chi = RM * (lambda^2 - lambda_0^2)
      Zero net rotation at channel nearest ref_freq.
    """
    from ..config import _parse_freq_to_hz

    ia.open(image_path)
    csys = ia.coordsys()
    shp = ia.shape()       # [nx, ny, nstokes, nchan]

    stokes_idx = _get_stokes_indices_from_csys(csys)
    if 'Q' not in stokes_idx or 'U' not in stokes_idx:
        ia.close()
        raise ValueError(
            "Faraday rotation requires Q and U Stokes planes. "
            f"Found: {list(stokes_idx.keys())}"
        )

    chan_freqs = _get_chan_freqs_from_csys(csys, shp[3])
    nu0 = _parse_freq_to_hz(sm_cfg.faraday.ref_freq)
    lam0_sq = (_C_LIGHT / nu0) ** 2

    if sm_cfg.faraday.rm_mode == 'global':
        rm = float(sm_cfg.faraday.rm_value)
    else:
        ia_tmp = ia.__class__()
        ia_tmp.open(sm_cfg.faraday.rm_value)
        rm_arr = ia_tmp.getchunk()
        ia_tmp.close()
        rm = rm_arr[:, :, 0, 0] if rm_arr.ndim == 4 else rm_arr

    q_idx = stokes_idx['Q']
    u_idx = stokes_idx['U']
    nx, ny, nstokes, nchan = shp

    ia.close()
    os.system(f'rm -rf {out_path}')
    ia.fromshape(out_path, list(shp), csys=csys.torecord(), overwrite=True)
    ia.close()

    ia_in = ia.__class__()
    ia_in.open(image_path)
    ia_out = ia.__class__()
    ia_out.open(out_path)

    for chan in range(nchan):
        blc = [0, 0, 0, chan]
        trc = [nx - 1, ny - 1, nstokes - 1, chan]
        plane = ia_in.getchunk(blc=blc, trc=trc)   # [nx, ny, nstokes, 1]

        lam_sq = (_C_LIGHT / chan_freqs[chan]) ** 2
        angle = rm * (lam_sq - lam0_sq)

        Q0 = plane[:, :, q_idx, 0].copy()
        U0 = plane[:, :, u_idx, 0].copy()
        plane[:, :, q_idx, 0] = Q0 * np.cos(2.0 * angle) - U0 * np.sin(2.0 * angle)
        plane[:, :, u_idx, 0] = Q0 * np.sin(2.0 * angle) + U0 * np.cos(2.0 * angle)
        ia_out.putchunk(plane, blc=blc)

    ia_in.close()
    ia_out.close()
    log.info("[skymodel] Faraday rotation applied: rm_mode=%s ref_freq=%s nchan=%d",
             sm_cfg.faraday.rm_mode, sm_cfg.faraday.ref_freq, nchan)
