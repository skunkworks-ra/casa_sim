from __future__ import annotations

import copy
import itertools
import logging
from typing import Any, List

import astropy.units as u
import numpy as np
import yaml

from .dataclasses import SimConfig
from .parsing import _parse_simconfig_from_raw
from .validation import validate_config

log = logging.getLogger(__name__)

_C_LIGHT = 2.99792458e8  # m/s


def _parse_freq_to_hz(freq_str: str) -> float:
    """Parse a frequency string like '1.0GHz' to Hz."""
    return float(u.Quantity(freq_str).to(u.Hz).value)


def _parse_angle_to_rad(angle_str: str) -> float:
    """Parse an angle string like '2.5arcsec' to radians."""
    return float(u.Quantity(angle_str).to(u.rad).value)


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _resolve_predictor_auto(cfg: SimConfig) -> str:
    """
    Resolve 'auto' predictor to a concrete predictor string.
    Routing table from Section 6. Does NOT mutate cfg.

    Returns: 'ft_dft' | 'sm_predict' | 'tclean'
    """
    sm = cfg.sky_model
    pred = cfg.prediction

    if pred.predictor != 'auto':
        if (pred.predictor == 'sm_predict'
                and pred.gridder in ('mosaic', 'awproject')):
            log.warning("[router] sm_predict demoted to tclean for gridder=%s",
                        pred.gridder)
            return 'tclean'
        return pred.predictor

    if sm.mode in ('image_extrapolate', 't_recs', 'corpus_mix'):
        return 'tclean'

    if sm.mode == 'component_list':
        if pred.gridder == 'standard':
            return 'ft_dft'
        else:
            return 'tclean'

    if sm.mode == 'image_native':
        if pred.gridder == 'standard':
            return 'sm_predict'
        else:
            return 'tclean'

    return 'tclean'


def _canned_telescope_geometry(tel: str):
    """
    Return (D_max_m, D_min_m) conservative estimates for known telescopes.
    These are order-of-magnitude estimates for derivation only.
    """
    _known = {
        'VLA':   (1030.0,  25.0),
        'ALMA':  (160.0,    7.0),
        'NGVLA': (1000.0,   6.0),
    }
    if tel in _known:
        return _known[tel]
    log.warning("[derive_imaging_params] Unknown telescope '%s' — using fallback "
                "D_max=1000m D_min=12m. Override cell/imsize in config.", tel)
    return (1000.0, 12.0)


def derive_imaging_params(cfg: SimConfig) -> SimConfig:
    """
    Derive cell and imsize from observatory geometry and SPW definitions.
    Populates cfg._derived_cell and cfg._derived_imsize.

    Returns a new SimConfig (does not mutate in place).
    """
    cfg = copy.deepcopy(cfg)

    effective_predictor = _resolve_predictor_auto(cfg) if cfg.prediction.predictor == 'auto' \
        else cfg.prediction.predictor

    if cfg.prediction.cell is not None and cfg.prediction.imsize is not None:
        log.info("[derive_imaging_params] User specified cell=%s imsize=%d — skipping derivation",
                 cfg.prediction.cell, cfg.prediction.imsize)
        return cfg

    first_spw = cfg.observation.spws[0]
    freq_hz = _parse_freq_to_hz(first_spw.freq)
    delta_hz = _parse_freq_to_hz(first_spw.deltafreq)
    center_freq_hz = freq_hz + 0.5 * delta_hz * (first_spw.nchan - 1)
    lam_m = _C_LIGHT / center_freq_hz

    obs_cfg = cfg.observatory
    if obs_cfg.mode == 'canned':
        tel = obs_cfg.canned.telescope.upper()
        d_max_m, d_min_m = _canned_telescope_geometry(tel)
        log.info("[derive_imaging_params] Canned telescope %s: D_max=%.1fm D_min=%.1fm",
                 tel, d_max_m, d_min_m)
    else:
        ants = obs_cfg.custom.antennas
        diameters = [a.diameter for a in ants]
        d_min_m = min(diameters)
        positions = np.array([[a.x, a.y, a.z] for a in ants])
        if len(positions) > 1:
            diffs = positions[:, None, :] - positions[None, :, :]
            d_max_m = float(np.max(np.linalg.norm(diffs, axis=-1)))
        else:
            d_max_m = diameters[0]
        log.info("[derive_imaging_params] Custom array: D_max=%.1fm D_min=%.1fm",
                 d_max_m, d_min_m)

    if cfg.prediction.cell is None:
        cell_rad = lam_m / (5.0 * d_max_m)
        cell_arcsec = float((cell_rad * u.rad).to(u.arcsec).value)
        cell_str = f"{cell_arcsec:.4g}arcsec"
        cfg._derived_cell = cell_str
        log.info("[derive_imaging_params] Derived cell = %s (lambda=%.4fm, D_max=%.1fm)",
                 cell_str, lam_m, d_max_m)
    else:
        cell_rad = _parse_angle_to_rad(cfg.prediction.cell)

    if cfg.prediction.imsize is None:
        pb_fwhm_rad = 1.02 * lam_m / d_min_m
        n_pixels = pb_fwhm_rad / cell_rad
        imsize = _next_power_of_2(int(np.ceil(n_pixels)))
        cfg._derived_imsize = imsize
        log.info("[derive_imaging_params] Derived imsize = %d "
                 "(PB_FWHM=%.2f arcmin, cell=%.4f arcsec)",
                 imsize,
                 float((pb_fwhm_rad * u.rad).to(u.arcmin).value),
                 float((cell_rad * u.rad).to(u.arcsec).value))

    return cfg


def apply_override(config: dict, dotpath: str, value: Any) -> dict:
    """
    Apply a dot-path override to a raw config dict (pre-dataclass).
    Returns a deep copy with the override applied.
    """
    keys = dotpath.split('.')
    out = copy.deepcopy(config)
    node = out
    for k in keys[:-1]:
        if k not in node:
            raise KeyError(f"Sweep dotpath '{dotpath}' invalid: key '{k}' not found")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"Sweep dotpath '{dotpath}' invalid: leaf '{keys[-1]}' not found")
    node[keys[-1]] = value
    return out


def expand_sweep(cfg: SimConfig) -> List[SimConfig]:
    if cfg.sweep is None or not cfg.sweep.axes:
        return [cfg]
    raise NotImplementedError(
        "Call expand_sweep_from_raw(raw_dict, cfg) instead of expand_sweep(cfg). "
        "expand_sweep() cannot round-trip a SimConfig to raw dict without YAML."
    )


def expand_sweep_from_raw(raw: dict, base_cfg: SimConfig) -> List[SimConfig]:
    """
    Expand sweep axes from the raw YAML dict.
    Returns list of validated SimConfig instances, one per Cartesian product point.
    Returns [base_cfg] if no sweep block.
    """
    if base_cfg.sweep is None or not base_cfg.sweep.axes:
        return [base_cfg]

    axes = base_cfg.sweep.axes
    value_lists = [axis.values for axis in axes]
    dotpaths = [axis.parameter for axis in axes]

    configs = []
    for combo in itertools.product(*value_lists):
        raw_copy = copy.deepcopy(raw)
        for dotpath, value in zip(dotpaths, combo):
            raw_copy = apply_override(raw_copy, dotpath, value)
        swept_cfg = _parse_simconfig_from_raw(raw_copy)
        validate_config(swept_cfg)
        swept_cfg = derive_imaging_params(swept_cfg)
        configs.append(swept_cfg)

    log.info("[expand_sweep] Expanded %d sweep point(s) from %d axis/axes",
             len(configs), len(axes))
    return configs


def load_config_with_sweep(path: str) -> tuple:
    """
    Full entry point: load YAML, validate, derive, expand sweep.

    Returns:
        (base_cfg, sweep_configs, raw_dict)
        sweep_configs is [base_cfg] if no sweep block.
    """
    with open(path, 'r') as fh:
        raw = yaml.safe_load(fh)

    base_cfg = _parse_simconfig_from_raw(raw)
    validate_config(base_cfg)
    base_cfg = derive_imaging_params(base_cfg)
    sweep_configs = expand_sweep_from_raw(raw, base_cfg)

    return base_cfg, sweep_configs, raw
