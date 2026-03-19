"""
observation.py — CASA Simulation Framework, Module 3

Responsibilities:
  - SPW setup via sm.setspwindow()
  - Field setup via sm.setfield()
  - Timing via sm.settimes()
  - Limits and autocorrelations
  - sm.observe() calls
  - flagdata unflag after sm.close()
  - freqresolution defaulting (qa available here)

Design notes:
  - freqresolution = deltafreq/nchan if not specified in config
    Done here because qa is available. config.py stores None for this field.
  - All calls in the required order per Section 3, Stage 3.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import SimConfig, ObservationConfig

log = logging.getLogger(__name__)


def resolve_observation(cfg: "SimConfig", msname: str, sm, me, qa, flagdata_fn) -> None:
    """
    Main entry point. Sets up the full observation structure.

    Args:
        cfg:         SimConfig instance
        msname:      Path to the MS being constructed
        sm:          CASA simulator tool (already open via sm.open(msname))
        me:          CASA measures tool
        qa:          CASA quanta tool
        flagdata_fn: casatasks.flagdata (passed to avoid circular import)
    """
    obs = cfg.observation

    _setup_spws(obs, sm, qa)
    _setup_fields(obs, sm, me)
    _setup_timing(obs, sm, me)

    sm.setlimits(shadowlimit=0.01, elevationlimit='1deg')
    sm.setauto(autocorrwt=0.0)

    _run_observe_calls(obs, sm)

    sm.close()
    log.info("[observation] sm.close() called")

    # Unflag everything — elevation/shadow flags not wanted for simulation
    flagdata_fn(vis=msname, mode='unflag')
    log.info("[observation] flagdata unflag applied to %s", msname)


# ---------------------------------------------------------------------------
# SPW setup
# ---------------------------------------------------------------------------

def _setup_spws(obs: "ObservationConfig", sm, qa) -> None:
    """
    Call sm.setspwindow() for each SPW in config.
    freqresolution defaults to deltafreq/nchan if not specified.
    """
    for spw in obs.spws:
        freqres = _resolve_freqresolution(spw, qa)
        sm.setspwindow(
            spwname=spw.name,
            freq=spw.freq,
            deltafreq=spw.deltafreq,
            freqresolution=freqres,
            nchannels=spw.nchan,
            stokes=spw.stokes
        )
        log.info("[observation] SPW '%s': freq=%s deltafreq=%s nchan=%d stokes=%s freqres=%s",
                 spw.name, spw.freq, spw.deltafreq, spw.nchan, spw.stokes, freqres)


def _resolve_freqresolution(spw, qa) -> str:
    """
    Return freqresolution string. If not specified in config, derive as deltafreq/nchan.
    Uses qa.convert() for unit-safe arithmetic.
    """
    if spw.freqresolution is not None:
        return spw.freqresolution

    # deltafreq / nchan — qa arithmetic
    delta_hz = qa.convert(qa.quantity(spw.deltafreq), 'Hz')['value']
    res_hz = delta_hz / spw.nchan
    # Format with reasonable precision
    if res_hz >= 1e6:
        return f"{res_hz / 1e6:.6g}MHz"
    elif res_hz >= 1e3:
        return f"{res_hz / 1e3:.6g}kHz"
    else:
        return f"{res_hz:.6g}Hz"


# ---------------------------------------------------------------------------
# Field setup
# ---------------------------------------------------------------------------

def _setup_fields(obs: "ObservationConfig", sm, me) -> None:
    """
    Call sm.setfield() for each field in config.
    Direction string is space-delimited: "J2000 RA Dec"
    """
    for fld in obs.fields:
        direction = _parse_direction(fld.direction, me)
        sm.setfield(sourcename=fld.name, sourcedirection=direction)
        log.info("[observation] Field '%s': direction=%s", fld.name, fld.direction)


def _parse_direction(direction_str: str, me) -> dict:
    """
    Parse a direction string like "J2000 19h59m28.5s +40d40m00.0s" into a
    CASA measures direction record.
    """
    parts = direction_str.strip().split()
    if len(parts) != 3:
        raise ValueError(
            f"direction string must have 3 space-separated components "
            f"(frame RA Dec), got: '{direction_str}'"
        )
    return me.direction(rf=parts[0], v0=parts[1], v1=parts[2])


# ---------------------------------------------------------------------------
# Timing setup
# ---------------------------------------------------------------------------

def _setup_timing(obs: "ObservationConfig", sm, me) -> None:
    """
    Call sm.settimes() with integration time, hourangle mode, and reference epoch.
    epoch string: "UTC 2020/10/4/00:00:00"
    """
    epoch = _parse_epoch(obs.epoch, me)
    sm.settimes(
        integrationtime=obs.integration_time,
        usehourangle=obs.use_hourangle,
        referencetime=epoch
    )
    log.info("[observation] Timing: integration=%s usehourangle=%s epoch=%s",
             obs.integration_time, obs.use_hourangle, obs.epoch)


def _parse_epoch(epoch_str: str, me) -> dict:
    """
    Parse epoch string "UTC 2020/10/4/00:00:00" into a CASA measures epoch record.
    """
    parts = epoch_str.strip().split(None, 1)
    if len(parts) != 2:
        raise ValueError(
            f"epoch string must be '<frame> <value>', got: '{epoch_str}'"
        )
    return me.epoch(parts[0], parts[1])


# ---------------------------------------------------------------------------
# Observe calls
# ---------------------------------------------------------------------------

def _run_observe_calls(obs: "ObservationConfig", sm) -> None:
    """
    Call sm.observe() for each (field, spw, start, stop) tuple.
    Explicit list — no implicit Cartesian product.
    """
    for call in obs.observe_calls:
        sm.observe(
            sourcename=call.field,
            spwname=call.spw,
            starttime=call.start_time,
            stoptime=call.stop_time
        )
        log.info("[observation] sm.observe: field=%s spw=%s start=%s stop=%s",
                 call.field, call.spw, call.start_time, call.stop_time)
