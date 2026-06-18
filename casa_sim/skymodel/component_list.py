from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING

import numpy as np

from .utils import _sky_to_pixel

if TYPE_CHECKING:
    from ..config import SimConfig, SkyModelConfig, SourceDef

log = logging.getLogger(__name__)


def build_component_list(sources: list, cl_path: str, cl) -> None:
    """
    Build a CASA component list from inline SourceDef entries.
    """
    os.system(f'rm -rf {cl_path}')
    cl.done()

    for src in sources:
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

        if len(flux) == 1:
            flux_arg = flux[0]
        else:
            flux_arg = flux

        spix = src.spectral_index
        index_arg = spix[0] if len(spix) == 1 else spix

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


def _apply_cl_stokes_spectrum(sm_cfg: "SkyModelConfig", cl) -> None:
    """
    Apply per-component Stokes spectral variation via cl.setstokesspectrum().
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
            log.warning("[skymodel] cl_stokes_spectrum tabular mode not supported "
                        "via CASA componentlist API for component %d — skipping", idx)
    cl.done()


def _build_rm_map_from_sources(sources: list, cfg: "SimConfig",
                                ia, qa, me) -> str:
    """
    Build a 2D RM map image from per-source RM values.
    Returns path to the RM map image.
    """
    rm_path = f"{cfg.name}_rm_map.im"
    cell_str = cfg.effective_cell
    imsize = cfg.effective_imsize

    first_spw = cfg.observation.spws[0]
    first_field = cfg.observation.fields[0]

    parts = first_field.direction.strip().split()
    ra_str, dec_str = parts[1], parts[2]

    os.system(f'rm -rf {rm_path}')
    ia.close()
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
    ia.set(0.0)

    pix = ia.getchunk()

    for src in sources:
        if src.rm == 0.0:
            continue
        px, py = _sky_to_pixel(src.direction, cs, qa, me)
        px = max(0, min(px, imsize - 1))
        py = max(0, min(py, imsize - 1))
        pix[px, py, 0, 0] = src.rm
        log.info("[skymodel] RM map: source '%s' RM=%.2f rad/m² at pixel (%d,%d)",
                 src.name, src.rm, px, py)

    ia.putchunk(pix)
    ia.close()
    log.info("[skymodel] RM map built: %s", rm_path)
    return rm_path
