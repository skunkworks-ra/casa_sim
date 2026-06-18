from __future__ import annotations

import numpy as np

_C_LIGHT = 2.99792458e8   # m/s


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

    axis_types = csys.axiscoordinatetypes()
    try:
        spec_axis = axis_types.index('Spectral')
    except ValueError:
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
    ra_rad = qa.convert(qa.quantity(parts[1]), 'rad')['value']
    dec_rad = qa.convert(qa.quantity(parts[2]), 'rad')['value']

    n_axes = len(csys.referencevalue()['numeric'])
    world = list(csys.referencevalue()['numeric'])
    world[0] = ra_rad
    world[1] = dec_rad

    pixel = csys.topixel(world)['numeric']
    px = int(round(pixel[0]))
    py = int(round(pixel[1]))
    return px, py
