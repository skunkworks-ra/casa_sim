"""
observatory.py — CASA Simulation Framework, Module 2

Responsibilities:
  - Resolve canned or custom observatory definition to sm.setconfig() + sm.setfeed()
  - Antenna list masking
  - Observatory position conversion (known | geodetic | itrf)

Design constraints:
  - No file I/O beyond reading CASA cfg files via ctsys/mysu
  - coordsystem='global' for custom arrays, 'local' is not used (reference notebook
    shows 'global' for both ALMA and NGVLA; only 'local' is for true local frame arrays)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import SimConfig, ObservatoryConfig

log = logging.getLogger(__name__)


def resolve_observatory(cfg: "SimConfig", sm, me, mysu, ctsys) -> None:
    """
    Main entry point. Resolves observatory config to sm.setconfig() + sm.setfeed().

    Args:
        cfg:   SimConfig instance
        sm:    CASA simulator tool (already opened via sm.open())
        me:    CASA measures tool
        mysu:  casatasks.private.simutil.simutil instance
        ctsys: casatools.ctsys
    """
    obs = cfg.observatory

    if obs.mode == 'canned':
        _resolve_canned(obs, sm, me, mysu, ctsys)
    else:
        _resolve_custom(obs, sm, me)

    sm.setfeed(mode=obs.feeds, pol=[''])
    log.info("[observatory] Feed mode set: %s", obs.feeds)


# ---------------------------------------------------------------------------
# Canned path
# ---------------------------------------------------------------------------

def _resolve_canned(obs: "ObservatoryConfig", sm, me, mysu, ctsys) -> None:
    """
    Resolve a canned telescope config to sm.setconfig().

    Steps:
      1. Locate cfg file (explicit path or CASA default for telescope)
      2. Read antenna table via mysu.readantenna()
      3. Apply antlist mask if specified
      4. Call sm.setconfig()
    """
    canned = obs.canned
    tel = canned.telescope.upper()

    # --- Locate cfg file ---
    if canned.cfg_file is not None:
        antennalist = canned.cfg_file
        log.info("[observatory] Using explicit cfg file: %s", antennalist)
    else:
        antennalist = _default_cfg_file(tel, ctsys)
        log.info("[observatory] Using default cfg file for %s: %s", tel, antennalist)

    # --- Read antenna table ---
    result = mysu.readantenna(antennalist)
    # readantenna returns (x, y, z, d, an, an2, telname, obspos)
    x, y, z, d, an = result[0], result[1], result[2], result[3], result[4]
    telname = result[6]

    # --- Apply antlist mask ---
    if canned.antlist is not None:
        x, y, z, d, an = _apply_antlist_mask(x, y, z, d, an, canned.antlist)
        log.info("[observatory] Antenna mask applied: %d antenna(s) selected", len(an))

    # --- Observatory position ---
    # For canned telescopes use me.observatory() with the standard name
    obspos_name = _telescope_to_obsname(tel)
    referencelocation = me.observatory(obspos_name)

    # --- sm.setconfig ---
    sm.setconfig(
        telescopename=telname,
        x=list(x),
        y=list(y),
        z=list(z),
        dishdiameter=list(d),
        mount=['alt-az'],
        antname=list(an),
        coordsystem='global',
        referencelocation=referencelocation
    )
    log.info("[observatory] sm.setconfig() called: telescope=%s, n_ant=%d", telname, len(an))


def _default_cfg_file(tel: str, ctsys) -> str:
    """Return the CASA default cfg file path for a known telescope."""
    _cfg_map = {
        'VLA':   'alma/simmos/vla.d.cfg',
        'ALMA':  'alma/simmos/alma.all.cfg',
        'NGVLA': 'alma/simmos/ngvla-core-revC.cfg',
    }
    if tel not in _cfg_map:
        raise ValueError(
            f"Unknown canned telescope '{tel}'. Known: {list(_cfg_map.keys())}. "
            f"Provide cfg_file explicitly."
        )
    import os
    return os.path.join(ctsys.resolve("alma/simmos"), _cfg_map[tel].split('/')[-1])


def _telescope_to_obsname(tel: str) -> str:
    """Map telescope name to CASA observatory name for me.observatory()."""
    _obs_map = {
        'VLA':   'VLA',
        'ALMA':  'ALMA',
        'NGVLA': 'VLA',   # ngVLA uses VLA site in reference notebooks
    }
    return _obs_map.get(tel, tel)


# ---------------------------------------------------------------------------
# Custom path
# ---------------------------------------------------------------------------

def _resolve_custom(obs: "ObservatoryConfig", sm, me) -> None:
    """
    Resolve a custom observatory definition to sm.setconfig().

    Steps:
      1. Build x, y, z, d, an arrays from config
      2. Convert obspos to a measures position record
      3. Call sm.setconfig()
    """
    custom = obs.custom
    ants = custom.antennas

    x = np.array([a.x for a in ants])
    y = np.array([a.y for a in ants])
    z = np.array([a.z for a in ants])
    d = np.array([a.diameter for a in ants])
    an = [a.name for a in ants]

    referencelocation = _convert_obspos(custom.obspos, me)

    sm.setconfig(
        telescopename=custom.telname,
        x=list(x),
        y=list(y),
        z=list(z),
        dishdiameter=list(d),
        mount=[custom.mount],
        antname=an,
        coordsystem='global',
        referencelocation=referencelocation
    )
    log.info("[observatory] sm.setconfig() called (custom): telescope=%s, n_ant=%d",
             custom.telname, len(an))


def _convert_obspos(obspos_cfg, me) -> dict:
    """
    Convert obspos config to a CASA measures position record.

    Modes:
      known    — me.observatory(name)
      geodetic — me.position('WGS84', lon, lat, alt)
      itrf     — me.position('ITRF', x, y, z)
    """
    mode = obspos_cfg.mode
    val = obspos_cfg.value

    if mode == 'known':
        # val is a string telescope name
        if not me.isposition(me.observatory(val)):
            raise ValueError(
                f"Observatory name '{val}' not found in CASA measures table. "
                f"Check me.obslist() for valid names."
            )
        return me.observatory(val)

    if mode == 'geodetic':
        # val is a dict with lat, lon, alt
        lat = str(val['lat'])
        lon = str(val['lon'])
        alt = float(val['alt'])
        return me.position('WGS84', lon, lat, f'{alt}m')

    if mode == 'itrf':
        # val is a dict with x, y, z in meters
        return me.position('ITRF',
                           f"{val['x']}m",
                           f"{val['y']}m",
                           f"{val['z']}m")

    raise ValueError(f"Unknown obspos mode '{mode}'. Expected: known | geodetic | itrf")


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _apply_antlist_mask(x, y, z, d, an, antlist) -> tuple:
    """
    Filter antenna arrays to those whose names appear in antlist.
    Uses np.isin() matching the reference notebook exactly.

    Returns filtered (x, y, z, d, an) as numpy arrays.
    """
    an_arr = np.array(an)
    mask = np.isin(an_arr, antlist)

    if not np.any(mask):
        raise ValueError(
            f"antlist filter produced zero antennas. "
            f"Requested: {antlist}. Available: {list(an_arr)}"
        )

    return (
        np.array(x)[mask],
        np.array(y)[mask],
        np.array(z)[mask],
        np.array(d)[mask],
        list(an_arr[mask])
    )
