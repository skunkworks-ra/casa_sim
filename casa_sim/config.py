"""
config.py — CASA Simulation Framework, Module 1

Responsibilities:
  - YAML parsing into a typed dataclass tree
  - All validation rules from Section 7
  - cell/imsize derivation (astropy.units, no CASA tools)
  - Sweep expansion (Cartesian product)

Design constraints:
  - Zero CASA tool dependencies (qa, ia, etc. are NOT imported here)
  - freqresolution stored as str | None; resolved in observation.py with qa
  - derive_imaging_params() uses astropy.units for unit arithmetic
"""

from __future__ import annotations

import copy
import itertools
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# astropy.units for unit-safe arithmetic in derive_imaging_params().
# Already a dependency in the reference notebooks.
import astropy.units as u
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Structured config validation error."""

    def __init__(self, rule: str, field_path: str, message: str):
        self.rule = rule
        self.field_path = field_path
        self.message = message
        super().__init__(f"[{rule}] {field_path}: {message}")


# ---------------------------------------------------------------------------
# Dataclasses — one per top-level config block
# ---------------------------------------------------------------------------

@dataclass
class ObsposConfig:
    mode: str                          # known | geodetic | itrf
    value: Any                         # str | dict


@dataclass
class AntennaEntry:
    name: str
    x: float
    y: float
    z: float
    diameter: float


@dataclass
class CannedObservatory:
    telescope: str
    cfg_file: Optional[str] = None
    antlist: Optional[List[str]] = None


@dataclass
class CustomObservatory:
    telname: str
    mount: str
    obspos: ObsposConfig
    antennas: List[AntennaEntry] = field(default_factory=list)


@dataclass
class ObservatoryConfig:
    mode: str                          # canned | custom
    feeds: str                         # e.g. "perfect R L"
    canned: Optional[CannedObservatory] = None
    custom: Optional[CustomObservatory] = None


@dataclass
class FieldConfig:
    name: str
    direction: str                     # e.g. "J2000 19h59m28.5s +40d40m00.0s"


@dataclass
class SpwConfig:
    name: str
    freq: str                          # e.g. "1.0GHz"
    deltafreq: str                     # e.g. "0.2GHz"
    nchan: int
    stokes: str                        # e.g. "RR LL"
    freqresolution: Optional[str] = None   # None → resolved in observation.py


@dataclass
class ObserveCall:
    field: str
    spw: str
    start_time: str
    stop_time: str


@dataclass
class ObservationConfig:
    epoch: str
    integration_time: str
    use_hourangle: bool
    fields: List[FieldConfig]
    spws: List[SpwConfig]
    observe_calls: List[ObserveCall]


@dataclass
class CLStokesSpectrum:
    component_index: int
    type: str                          # spectral_index | tabular
    ref_freq: Optional[str] = None
    index: Optional[List[float]] = None   # [alpha_I, alpha_fraclin, alpha_pa, alpha_fraccir]
    freqs: Optional[List[float]] = None
    I: Optional[List[float]] = None
    Q: Optional[List[float]] = None
    U: Optional[List[float]] = None
    V: Optional[List[float]] = None


@dataclass
class SourceDef:
    """Inline source definition for building a component list from YAML."""
    name: str
    direction: str                     # e.g. "J2000 13h31m08.29s +30d30m32.96s"
    flux: List[float]                  # [I] or [I,Q,U,V]
    ref_freq: str                      # e.g. "1.4GHz"
    spectral_index: List[float] = field(default_factory=lambda: [0.0])  # [alpha] or [alpha, beta]
    shape: str = 'point'               # point | gaussian | disk
    major: Optional[str] = None        # e.g. "10arcsec" (gaussian/disk)
    minor: Optional[str] = None
    pa: Optional[str] = None           # position angle
    rm: float = 0.0                    # rotation measure rad/m^2
    frac_pol: Optional[float] = None   # fractional linear polarization (derives Q,U from I)
    chi: Optional[float] = None        # EVPA in degrees (used with frac_pol)


@dataclass
class FaradayConfig:
    enabled: bool
    rm_mode: str                       # global | map
    rm_value: Union[float, str]        # scalar rad/m^2 or image path
    ref_freq: str


@dataclass
class SpectralLineConfig:
    name: str
    stokes: str
    channels: List[int]
    flux_profile: List[float]
    mode: str                          # point | gaussian | image
    direction: Optional[str] = None
    major: Optional[str] = None
    minor: Optional[str] = None
    pa: Optional[str] = None
    image_path: Optional[str] = None


@dataclass
class SkyModelConfig:
    stokes: str                        # I | IQUV
    mode: str                          # component_list | image_native | image_extrapolate
    cl_path: Optional[str] = None
    sources: Optional[List[SourceDef]] = None   # inline source definitions → auto-builds .cl
    sources_file: Optional[str] = None          # path to external YAML with sources list
    flux_cutoff: Optional[float] = None         # Jy; drop sources fainter than this
    cl_stokes_spectrum: Optional[List[CLStokesSpectrum]] = None
    image_path: Optional[str] = None
    ref_freq: Optional[str] = None     # image_extrapolate
    alpha_mode: Optional[str] = None   # global | map
    alpha_value: Optional[Union[float, str]] = None
    faraday: Optional[FaradayConfig] = None
    spectral_lines: List[SpectralLineConfig] = field(default_factory=list)


@dataclass
class PredictionConfig:
    gridder: str                       # standard | mosaic | awproject
    predictor: str                     # auto | ft_dft | sm_predict | tclean
    normtype: str                      # flatsky | flatnoise
    cell: Optional[str] = None        # None → derived
    imsize: Optional[int] = None      # None → derived


@dataclass
class NoiseConfig:
    enabled: bool
    mode: str                          # simplenoise | tsys-atm | per_baseline
    value: Optional[str] = None        # Jy string; simplenoise only


@dataclass
class GainsConfig:
    enabled: bool
    mode: str                          # fbm
    amplitude: float = 0.0


@dataclass
class CorruptionConfig:
    seed: Optional[int]                # None → non-reproducible
    noise: NoiseConfig = field(default_factory=lambda: NoiseConfig(enabled=False, mode='simplenoise'))
    gains: GainsConfig = field(default_factory=lambda: GainsConfig(enabled=False, mode='fbm'))


@dataclass
class ImagingConfig:
    enabled: bool
    deconvolver: str = 'hogbom'
    nterms: Optional[int] = None
    pbcor: bool = False
    niter: int = 500


@dataclass
class SweepAxis:
    parameter: str                     # dot-path e.g. "corruption.noise.value"
    values: List[Any]


@dataclass
class SweepConfig:
    axes: List[SweepAxis]


@dataclass
class SimConfig:
    """Top-level config. Carries derived imaging params after derive_imaging_params()."""
    name: str
    observatory: ObservatoryConfig
    observation: ObservationConfig
    sky_model: SkyModelConfig
    prediction: PredictionConfig
    corruption: CorruptionConfig
    imaging: ImagingConfig
    sweep: Optional[SweepConfig] = None

    # Derived in derive_imaging_params() — not in YAML
    _derived_cell: Optional[str] = field(default=None, repr=False)
    _derived_imsize: Optional[int] = field(default=None, repr=False)

    @property
    def effective_cell(self) -> Optional[str]:
        return self.prediction.cell or self._derived_cell

    @property
    def effective_imsize(self) -> Optional[int]:
        return self.prediction.imsize or self._derived_imsize


# ---------------------------------------------------------------------------
# YAML parsing helpers
# ---------------------------------------------------------------------------

def _require(d: dict, key: str, context: str) -> Any:
    if key not in d or d[key] is None:
        raise ConfigError("missing_required_field", f"{context}.{key}",
                          f"Required field '{key}' missing in {context}")
    return d[key]


def _parse_obspos(d: dict) -> ObsposConfig:
    mode = _require(d, 'mode', 'obspos')
    value = _require(d, 'value', 'obspos')
    return ObsposConfig(mode=mode, value=value)


def _parse_custom_observatory(d: dict) -> CustomObservatory:
    antennas = [
        AntennaEntry(
            name=a['name'],
            x=float(a['x']),
            y=float(a['y']),
            z=float(a['z']),
            diameter=float(a['diameter'])
        )
        for a in _require(d, 'antennas', 'custom')
    ]
    return CustomObservatory(
        telname=_require(d, 'telname', 'custom'),
        mount=_require(d, 'mount', 'custom'),
        obspos=_parse_obspos(_require(d, 'obspos', 'custom')),
        antennas=antennas
    )


def _parse_observatory(d: dict) -> ObservatoryConfig:
    mode = _require(d, 'mode', 'observatory')
    feeds = _require(d, 'feeds', 'observatory')
    canned = None
    custom = None
    if mode == 'canned':
        cd = _require(d, 'canned', 'observatory')
        canned = CannedObservatory(
            telescope=_require(cd, 'telescope', 'observatory.canned'),
            cfg_file=cd.get('cfg_file'),
            antlist=cd.get('antlist')
        )
    elif mode == 'custom':
        custom = _parse_custom_observatory(_require(d, 'custom', 'observatory'))
    else:
        raise ConfigError("invalid_value", "observatory.mode",
                          f"mode must be 'canned' or 'custom', got '{mode}'")
    return ObservatoryConfig(mode=mode, feeds=feeds, canned=canned, custom=custom)


def _parse_observation(d: dict) -> ObservationConfig:
    fields = [FieldConfig(name=f['name'], direction=f['direction'])
              for f in _require(d, 'fields', 'observation')]
    spws = [
        SpwConfig(
            name=s['name'],
            freq=s['freq'],
            deltafreq=s['deltafreq'],
            nchan=int(s['nchan']),
            stokes=s['stokes'],
            freqresolution=s.get('freqresolution')   # None if absent
        )
        for s in _require(d, 'spws', 'observation')
    ]
    calls = [
        ObserveCall(
            field=c['field'],
            spw=c['spw'],
            start_time=c['start_time'],
            stop_time=c['stop_time']
        )
        for c in _require(d, 'observe_calls', 'observation')
    ]
    return ObservationConfig(
        epoch=_require(d, 'epoch', 'observation'),
        integration_time=_require(d, 'integration_time', 'observation'),
        use_hourangle=bool(_require(d, 'use_hourangle', 'observation')),
        fields=fields,
        spws=spws,
        observe_calls=calls
    )


def _parse_faraday(d: dict) -> FaradayConfig:
    return FaradayConfig(
        enabled=bool(d.get('enabled', False)),
        rm_mode=d.get('rm_mode', 'global'),
        rm_value=d.get('rm_value', 0.0),
        ref_freq=d.get('ref_freq', '')
    )


def _parse_spectral_lines(lst: list) -> List[SpectralLineConfig]:
    result = []
    for s in lst:
        result.append(SpectralLineConfig(
            name=s['name'],
            stokes=s['stokes'],
            channels=list(s['channels']),
            flux_profile=list(s['flux_profile']),
            mode=s['mode'],
            direction=s.get('direction'),
            major=s.get('major'),
            minor=s.get('minor'),
            pa=s.get('pa'),
            image_path=s.get('image_path')
        ))
    return result


def _parse_cl_stokes_spectrum(lst: list) -> List[CLStokesSpectrum]:
    result = []
    for s in lst:
        result.append(CLStokesSpectrum(
            component_index=int(s['component_index']),
            type=s['type'],
            ref_freq=s.get('ref_freq'),
            index=s.get('index'),
            freqs=s.get('freqs'),
            I=s.get('I'),
            Q=s.get('Q'),
            U=s.get('U'),
            V=s.get('V')
        ))
    return result


def _parse_sources(lst: list) -> List[SourceDef]:
    result = []
    for s in lst:
        flux_raw = s.get('flux', [1.0])
        if isinstance(flux_raw, (int, float)):
            flux_raw = [float(flux_raw)]
        else:
            flux_raw = [float(f) for f in flux_raw]
        spix_raw = s.get('spectral_index', [0.0])
        if isinstance(spix_raw, (int, float)):
            spix_raw = [float(spix_raw)]
        else:
            spix_raw = [float(v) for v in spix_raw]
        result.append(SourceDef(
            name=_require(s, 'name', 'sources'),
            direction=_require(s, 'direction', 'sources'),
            flux=flux_raw,
            ref_freq=_require(s, 'ref_freq', 'sources'),
            spectral_index=spix_raw,
            shape=s.get('shape', 'point'),
            major=s.get('major'),
            minor=s.get('minor'),
            pa=s.get('pa'),
            rm=float(s.get('rm', 0.0)),
            frac_pol=float(s['frac_pol']) if s.get('frac_pol') is not None else None,
            chi=float(s['chi']) if s.get('chi') is not None else None,
        ))
    return result


def _load_sources_file(sources_file: str, config_dir: str) -> List[SourceDef]:
    """Load source definitions from an external YAML file."""
    path = Path(sources_file)
    if not path.is_absolute():
        path = Path(config_dir) / path
    if not path.exists():
        raise ConfigError(
            "sources_file_not_found",
            "sky_model.sources_file",
            f"Sources file not found: {path}"
        )
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "sources" not in raw:
        raise ConfigError(
            "invalid_sources_file",
            "sky_model.sources_file",
            f"Sources file must contain a top-level 'sources' list: {path}"
        )
    log.info("[config] Loaded %d sources from %s", len(raw["sources"]), path)
    return _parse_sources(raw["sources"])


def _parse_sky_model(d: dict, config_dir: str = ".") -> SkyModelConfig:
    faraday = None
    if 'faraday' in d and d['faraday']:
        faraday = _parse_faraday(d['faraday'])
    cl_ss = None
    if 'cl_stokes_spectrum' in d and d['cl_stokes_spectrum']:
        cl_ss = _parse_cl_stokes_spectrum(d['cl_stokes_spectrum'])
    lines = _parse_spectral_lines(d.get('spectral_lines') or [])

    sources = None
    sources_file = d.get('sources_file')
    flux_cutoff = d.get('flux_cutoff')
    if flux_cutoff is not None:
        flux_cutoff = float(flux_cutoff)
    if 'sources' in d and d['sources']:
        sources = _parse_sources(d['sources'])
    elif sources_file:
        sources = _load_sources_file(sources_file, config_dir)

    # Apply flux cutoff
    if sources and flux_cutoff is not None:
        n_before = len(sources)
        sources = [s for s in sources if s.flux[0] >= flux_cutoff]
        log.info("[config] flux_cutoff=%.2e Jy: kept %d / %d sources",
                 flux_cutoff, len(sources), n_before)
        if not sources:
            raise ConfigError(
                "flux_cutoff_removed_all_sources",
                "sky_model.flux_cutoff",
                f"flux_cutoff={flux_cutoff} removed all {n_before} sources — "
                "lower the cutoff or add brighter sources"
            )

    return SkyModelConfig(
        stokes=d.get('stokes', 'I'),
        mode=_require(d, 'mode', 'sky_model'),
        cl_path=d.get('cl_path'),
        sources=sources,
        sources_file=sources_file,
        flux_cutoff=flux_cutoff,
        cl_stokes_spectrum=cl_ss,
        image_path=d.get('image_path'),
        ref_freq=d.get('ref_freq'),
        alpha_mode=d.get('alpha_mode'),
        alpha_value=d.get('alpha_value'),
        faraday=faraday,
        spectral_lines=lines
    )


def _parse_prediction(d: dict) -> PredictionConfig:
    return PredictionConfig(
        gridder=_require(d, 'gridder', 'prediction'),
        predictor=d.get('predictor', 'auto'),
        normtype=d.get('normtype', 'flatsky'),
        cell=d.get('cell'),
        imsize=d.get('imsize')
    )


def _parse_corruption(d: dict) -> CorruptionConfig:
    noise_d = d.get('noise', {})
    gains_d = d.get('gains', {})
    noise = NoiseConfig(
        enabled=bool(noise_d.get('enabled', False)),
        mode=noise_d.get('mode', 'simplenoise'),
        value=noise_d.get('value')
    )
    gains = GainsConfig(
        enabled=bool(gains_d.get('enabled', False)),
        mode=gains_d.get('mode', 'fbm'),
        amplitude=float(gains_d.get('amplitude', 0.0))
    )
    return CorruptionConfig(
        seed=d.get('seed'),    # None is valid
        noise=noise,
        gains=gains
    )


def _parse_imaging(d: dict) -> ImagingConfig:
    return ImagingConfig(
        enabled=bool(d.get('enabled', False)),
        deconvolver=d.get('deconvolver', 'hogbom'),
        nterms=d.get('nterms'),
        pbcor=bool(d.get('pbcor', False)),
        niter=int(d.get('niter', 500))
    )


def _parse_sweep(d: dict) -> SweepConfig:
    axes = [SweepAxis(parameter=a['parameter'], values=list(a['values']))
            for a in d.get('axes', [])]
    return SweepConfig(axes=axes)


# ---------------------------------------------------------------------------
# Public: load_config
# ---------------------------------------------------------------------------

def load_config(path: str) -> SimConfig:
    """
    Parse a YAML config file into a SimConfig dataclass tree.
    Does NOT validate — call validate_config() separately.
    """
    config_dir = str(Path(path).resolve().parent)
    with open(path, 'r') as fh:
        raw = yaml.safe_load(fh)

    name = _require(raw, 'name', 'root')
    observatory = _parse_observatory(_require(raw, 'observatory', 'root'))
    observation = _parse_observation(_require(raw, 'observation', 'root'))
    sky_model = _parse_sky_model(_require(raw, 'sky_model', 'root'), config_dir)
    prediction = _parse_prediction(_require(raw, 'prediction', 'root'))
    corruption = _parse_corruption(raw.get('corruption', {}))
    imaging = _parse_imaging(raw.get('imaging', {}))
    sweep = _parse_sweep(raw['sweep']) if 'sweep' in raw else None

    return SimConfig(
        name=name,
        observatory=observatory,
        observation=observation,
        sky_model=sky_model,
        prediction=prediction,
        corruption=corruption,
        imaging=imaging,
        sweep=sweep
    )


# ---------------------------------------------------------------------------
# Public: validate_config — all Section 7 rules
# ---------------------------------------------------------------------------

def validate_config(cfg: SimConfig) -> None:
    """
    Enforce all validation rules from Section 7.
    Raises ConfigError on hard errors.
    Logs warnings via logging module.
    """
    sm = cfg.sky_model
    pred = cfg.prediction
    obs = cfg.observation
    corr = cfg.corruption
    img = cfg.imaging

    # ---- Predictor / gridder compatibility --------------------------------

    # Resolve effective predictor for validation purposes
    effective_predictor = _resolve_predictor_auto(cfg) if pred.predictor == 'auto' else pred.predictor

    if (effective_predictor == 'ft_dft'
            and pred.gridder in ('mosaic', 'awproject')):
        raise ConfigError(
            "incompatible_predictor_gridder",
            "prediction.predictor",
            "ft_dft predictor incompatible with mosaic/awproject gridder"
        )

    if (effective_predictor == 'sm_predict'
            and pred.gridder in ('mosaic', 'awproject')):
        log.warning("[validation] prediction.predictor: sm_predict demoted to tclean "
                    "for gridder=%s", pred.gridder)

    # ---- Sky model --------------------------------------------------------

    if sm.mode == 'image_extrapolate' and not sm.ref_freq:
        raise ConfigError(
            "missing_ref_freq",
            "sky_model.ref_freq",
            "ref_freq required for image_extrapolate mode"
        )

    if (sm.mode == 'image_extrapolate'
            and sm.alpha_mode == 'map'
            and not _is_valid_path(sm.alpha_value)):
        raise ConfigError(
            "invalid_alpha_map_path",
            "sky_model.alpha_value",
            "alpha_value must be a valid image path for alpha_mode=map"
        )

    # ---- Sources (inline component list) -----------------------------------

    if sm.sources and sm.cl_path:
        raise ConfigError(
            "sources_and_cl_path_exclusive",
            "sky_model",
            "sources and cl_path are mutually exclusive — use one or the other"
        )

    if sm.sources_file and sm.cl_path:
        raise ConfigError(
            "sources_file_and_cl_path_exclusive",
            "sky_model",
            "sources_file and cl_path are mutually exclusive"
        )

    if sm.mode == 'component_list' and not sm.cl_path and not sm.sources:
        raise ConfigError(
            "component_list_no_source",
            "sky_model",
            "mode=component_list requires either cl_path, sources, or sources_file"
        )

    if sm.sources:
        if sm.mode != 'component_list':
            raise ConfigError(
                "sources_require_component_list_mode",
                "sky_model.mode",
                "sources require mode: component_list"
            )
        for i, src in enumerate(sm.sources):
            if len(src.flux) not in (1, 4):
                raise ConfigError(
                    "invalid_flux_length",
                    f"sky_model.sources[{i}].flux",
                    f"flux must have 1 (I) or 4 (IQUV) elements, got {len(src.flux)}"
                )
            if len(src.spectral_index) not in (1, 2):
                raise ConfigError(
                    "invalid_spectral_index_length",
                    f"sky_model.sources[{i}].spectral_index",
                    f"spectral_index must have 1 (alpha) or 2 (alpha, beta) elements, "
                    f"got {len(src.spectral_index)}"
                )
            if src.frac_pol is not None and len(src.flux) == 4:
                raise ConfigError(
                    "frac_pol_with_iquv_flux",
                    f"sky_model.sources[{i}]",
                    "frac_pol cannot be used with explicit IQUV flux — use one or the other"
                )
            if src.frac_pol is not None and src.chi is None:
                raise ConfigError(
                    "frac_pol_requires_chi",
                    f"sky_model.sources[{i}]",
                    "chi (EVPA in degrees) is required when frac_pol is specified"
                )
            if src.shape in ('gaussian', 'disk'):
                if not src.major or not src.minor:
                    raise ConfigError(
                        "missing_shape_params",
                        f"sky_model.sources[{i}]",
                        f"major and minor required for shape={src.shape}"
                    )

    # ---- Faraday ----------------------------------------------------------

    if sm.faraday and sm.faraday.enabled:
        if sm.stokes != 'IQUV':
            raise ConfigError(
                "faraday_requires_iquv",
                "sky_model.stokes",
                "Faraday rotation requires stokes: IQUV"
            )
        if sm.faraday.rm_mode == 'map' and not _is_valid_path(sm.faraday.rm_value):
            raise ConfigError(
                "invalid_rm_map_path",
                "sky_model.faraday.rm_value",
                "rm_value must be a valid image path for rm_mode=map"
            )
        if not sm.faraday.ref_freq:
            raise ConfigError(
                "missing_faraday_ref_freq",
                "sky_model.faraday.ref_freq",
                "faraday.ref_freq required when faraday.enabled is true"
            )

    # ---- Spectral lines ---------------------------------------------------

    for i, line in enumerate(sm.spectral_lines):
        if line.mode == 'image' and not line.image_path:
            raise ConfigError(
                "missing_line_image_path",
                f"sky_model.spectral_lines[{i}].image_path",
                "image_path required for spectral line image mode"
            )
        if len(line.flux_profile) != len(line.channels):
            raise ConfigError(
                "flux_profile_length_mismatch",
                f"sky_model.spectral_lines[{i}].flux_profile",
                f"flux_profile length ({len(line.flux_profile)}) must match "
                f"channels length ({len(line.channels)})"
            )

    # ---- Observe calls reference valid field/spw names --------------------

    field_names = {f.name for f in obs.fields}
    spw_names = {s.name for s in obs.spws}

    for i, call in enumerate(obs.observe_calls):
        if call.field not in field_names:
            raise ConfigError(
                "undefined_field_reference",
                f"observation.observe_calls[{i}].field",
                f"observe_call references undefined field name '{call.field}'"
            )
        if call.spw not in spw_names:
            raise ConfigError(
                "undefined_spw_reference",
                f"observation.observe_calls[{i}].spw",
                f"observe_call references undefined spw name '{call.spw}'"
            )

    # ---- Imaging ----------------------------------------------------------

    if img.enabled and img.deconvolver == 'mtmfs' and img.nterms is None:
        raise ConfigError(
            "missing_nterms",
            "imaging.nterms",
            "nterms required for mtmfs deconvolver"
        )

    # ---- Corruption -------------------------------------------------------

    if corr.noise.enabled and corr.noise.mode == 'simplenoise' and not corr.noise.value:
        raise ConfigError(
            "missing_noise_value",
            "corruption.noise.value",
            "noise value required for simplenoise mode"
        )

    # ---- Warnings (log, do not raise) -------------------------------------

    total_nchan = sum(s.nchan for s in obs.spws)
    if sm.stokes == 'IQUV':
        for s in obs.spws:
            n_prods = len(s.stokes.strip().split())
            if n_prods < 4:
                log.warning("[validation] sky_model.stokes=IQUV but SPW '%s' has "
                            "only %d correlation product(s): %s",
                            s.name, n_prods, s.stokes)

    if (pred.cell is None or pred.imsize is None) and effective_predictor == 'ft_dft':
        log.warning("[validation] prediction.cell/imsize not needed for ft_dft, ignoring")

    if corr.seed is None:
        log.warning("[validation] corruption.seed=null — results will not be reproducible")


def _is_valid_path(value: Any) -> bool:
    """Return True if value is a non-empty string (path existence not checked here)."""
    return isinstance(value, str) and len(value.strip()) > 0


# ---------------------------------------------------------------------------
# Predictor routing (also used by predict.py — single source of truth)
# ---------------------------------------------------------------------------

def _resolve_predictor_auto(cfg: SimConfig) -> str:
    """
    Resolve 'auto' predictor to a concrete predictor string.
    Routing table from Section 6. Does NOT mutate cfg.

    Returns: 'ft_dft' | 'sm_predict' | 'tclean'
    """
    sm = cfg.sky_model
    pred = cfg.prediction

    if pred.predictor != 'auto':
        # User override — apply demotion rule then return
        if (pred.predictor == 'sm_predict'
                and pred.gridder in ('mosaic', 'awproject')):
            log.warning("[router] sm_predict demoted to tclean for gridder=%s",
                        pred.gridder)
            return 'tclean'
        return pred.predictor

    # image_extrapolate always uses tclean
    if sm.mode == 'image_extrapolate':
        return 'tclean'

    if sm.mode == 'component_list':
        if pred.gridder == 'standard':
            return 'ft_dft'
        else:  # mosaic or awproject
            return 'tclean'

    if sm.mode == 'image_native':
        if pred.gridder == 'standard':
            return 'sm_predict'
        else:
            return 'tclean'

    # Fallback — should not be reached after validation
    return 'tclean'


# ---------------------------------------------------------------------------
# Public: derive_imaging_params
# ---------------------------------------------------------------------------

# Speed of light — consistent with skymodel.py (no astropy.constants dependency)
_C_LIGHT = 2.99792458e8  # m/s


def _parse_freq_to_hz(freq_str: str) -> float:
    """
    Parse a frequency string like '1.0GHz', '90GHz', '1.25GHz' to Hz.
    Uses astropy.units for correctness.
    """
    # astropy can parse '1.0GHz' directly
    return u.Unit(freq_str).to(u.Hz) if False else float(u.Quantity(freq_str).to(u.Hz).value)


def _parse_angle_to_rad(angle_str: str) -> float:
    """Parse an angle string like '2.5arcsec', '1.0arcmin' to radians."""
    return float(u.Quantity(angle_str).to(u.rad).value)


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def derive_imaging_params(cfg: SimConfig) -> SimConfig:
    """
    Derive cell and imsize from observatory geometry and SPW definitions.
    Populates cfg._derived_cell and cfg._derived_imsize.

    Skipped if predictor (effective) is ft_dft.
    User-specified cell/imsize always take precedence.

    Derivation:
      center_freq = freq of first SPW (center channel)
      lambda_m = c / center_freq
      D_max = longest baseline proxy (max antenna separation or max dish diameter if single dish)
      cell ~ lambda / (5 * D_max)   [standard Nyquist-like sampling]
      D_min = smallest dish diameter
      PB_FWHM ~ 1.02 * lambda / D_min   [radians]
      imsize = PB_FWHM / cell, rounded up to next power of 2

    Returns a new SimConfig (does not mutate in place).
    """
    cfg = copy.deepcopy(cfg)

    effective_predictor = _resolve_predictor_auto(cfg) if cfg.prediction.predictor == 'auto' \
        else cfg.prediction.predictor

    # ft_dft: cell/imsize not used for prediction but may be needed for sanity imaging

    # User already specified both — nothing to derive
    if cfg.prediction.cell is not None and cfg.prediction.imsize is not None:
        log.info("[derive_imaging_params] User specified cell=%s imsize=%d — skipping derivation",
                 cfg.prediction.cell, cfg.prediction.imsize)
        return cfg

    # --- Frequency ---
    first_spw = cfg.observation.spws[0]
    freq_hz = _parse_freq_to_hz(first_spw.freq)
    delta_hz = _parse_freq_to_hz(first_spw.deltafreq)
    center_freq_hz = freq_hz + 0.5 * delta_hz * (first_spw.nchan - 1)
    lam_m = _C_LIGHT / center_freq_hz

    # --- Baseline / dish geometry ---
    obs_cfg = cfg.observatory
    if obs_cfg.mode == 'canned':
        # Cannot know actual baseline lengths without reading the cfg file (CASA tool needed).
        # Use a safe conservative estimate: D_max from known telescope defaults.
        # This is a best-effort derivation; user can override with explicit cell/imsize.
        tel = obs_cfg.canned.telescope.upper()
        d_max_m, d_min_m = _canned_telescope_geometry(tel)
        log.info("[derive_imaging_params] Canned telescope %s: D_max=%.1fm D_min=%.1fm",
                 tel, d_max_m, d_min_m)
    else:
        # Custom: compute from antenna list
        ants = obs_cfg.custom.antennas
        diameters = [a.diameter for a in ants]
        d_min_m = min(diameters)
        # Max baseline length from ITRF coordinates
        positions = np.array([[a.x, a.y, a.z] for a in ants])
        if len(positions) > 1:
            diffs = positions[:, None, :] - positions[None, :, :]
            d_max_m = float(np.max(np.linalg.norm(diffs, axis=-1)))
        else:
            d_max_m = diameters[0]
        log.info("[derive_imaging_params] Custom array: D_max=%.1fm D_min=%.1fm",
                 d_max_m, d_min_m)

    # --- Cell size ---
    if cfg.prediction.cell is None:
        cell_rad = lam_m / (5.0 * d_max_m)
        cell_arcsec = float((cell_rad * u.rad).to(u.arcsec).value)
        # Round to 3 significant figures for readability
        cell_str = f"{cell_arcsec:.4g}arcsec"
        cfg._derived_cell = cell_str
        log.info("[derive_imaging_params] Derived cell = %s (lambda=%.4fm, D_max=%.1fm)",
                 cell_str, lam_m, d_max_m)
    else:
        cell_rad = _parse_angle_to_rad(cfg.prediction.cell)

    # --- Image size ---
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


def _canned_telescope_geometry(tel: str):
    """
    Return (D_max_m, D_min_m) conservative estimates for known telescopes.
    These are order-of-magnitude estimates for derivation only.
    VLA D-config max baseline ~1.03 km, C-config ~3.4 km, B ~11 km, A ~36 km.
    Without reading the cfg file we default to a mid-range estimate.
    User should override cell/imsize explicitly for precise work.
    """
    # (D_max_baseline_m, D_min_dish_m)
    _known = {
        'VLA':   (1030.0,  25.0),    # D-config; conservative
        'ALMA':  (160.0,    7.0),    # compact configuration
        'NGVLA': (1000.0,   6.0),    # SBA dishes
    }
    if tel in _known:
        return _known[tel]
    # Unknown telescope: fall back to a 1 km / 12 m guess with a warning
    log.warning("[derive_imaging_params] Unknown telescope '%s' — using fallback "
                "D_max=1000m D_min=12m. Override cell/imsize in config.", tel)
    return (1000.0, 12.0)


# ---------------------------------------------------------------------------
# Public: apply_override
# ---------------------------------------------------------------------------

def apply_override(config: dict, dotpath: str, value: Any) -> dict:
    """
    Apply a dot-path override to a raw config dict (pre-dataclass).
    Returns a deep copy with the override applied.

    Raises KeyError for invalid paths.
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


# ---------------------------------------------------------------------------
# Public: expand_sweep
# ---------------------------------------------------------------------------

def expand_sweep(cfg: SimConfig) -> List[SimConfig]:
    """
    Expand the sweep block into a list of SimConfig instances.
    Returns [cfg] (length 1) if no sweep block — uniform downstream interface.

    Uses Cartesian product of all axes.
    """
    if cfg.sweep is None or not cfg.sweep.axes:
        return [cfg]

    # Re-serialize cfg to a raw dict for apply_override, then re-parse.
    # This avoids a brittle dataclass-to-dict path — we reload from the
    # original YAML via a round-trip through _simconfig_to_dict().
    # NOTE: caller must pass the original raw dict for reliable sweep.
    # expand_sweep_from_raw() is the primary entry point for sweep.
    raise NotImplementedError(
        "Call expand_sweep_from_raw(raw_dict, cfg) instead of expand_sweep(cfg). "
        "expand_sweep() cannot round-trip a SimConfig to raw dict without YAML."
    )


def expand_sweep_from_raw(raw: dict, base_cfg: SimConfig,
                          config_dir: str = ".") -> List[SimConfig]:
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
        # Re-parse the modified raw dict (no file I/O)
        swept_cfg = _parse_simconfig_from_raw(raw_copy, config_dir)
        validate_config(swept_cfg)
        swept_cfg = derive_imaging_params(swept_cfg)
        configs.append(swept_cfg)

    log.info("[expand_sweep] Expanded %d sweep point(s) from %d axis/axes",
             len(configs), len(axes))
    return configs


def _parse_simconfig_from_raw(raw: dict, config_dir: str = ".") -> SimConfig:
    """Parse a raw dict (as loaded from YAML) into a SimConfig."""
    name = _require(raw, 'name', 'root')
    observatory = _parse_observatory(_require(raw, 'observatory', 'root'))
    observation = _parse_observation(_require(raw, 'observation', 'root'))
    sky_model = _parse_sky_model(_require(raw, 'sky_model', 'root'), config_dir)
    prediction = _parse_prediction(_require(raw, 'prediction', 'root'))
    corruption = _parse_corruption(raw.get('corruption', {}))
    imaging = _parse_imaging(raw.get('imaging', {}))
    sweep = _parse_sweep(raw['sweep']) if 'sweep' in raw else None
    return SimConfig(
        name=name,
        observatory=observatory,
        observation=observation,
        sky_model=sky_model,
        prediction=prediction,
        corruption=corruption,
        imaging=imaging,
        sweep=sweep
    )


# ---------------------------------------------------------------------------
# Public: load_config_with_sweep
# ---------------------------------------------------------------------------

def load_config_with_sweep(path: str) -> tuple[SimConfig, List[SimConfig], dict]:
    """
    Full entry point: load YAML, validate, derive, expand sweep.

    Returns:
        (base_cfg, sweep_configs, raw_dict)
        sweep_configs is [base_cfg] if no sweep block.
    """
    config_dir = str(Path(path).resolve().parent)
    with open(path, 'r') as fh:
        raw = yaml.safe_load(fh)

    base_cfg = _parse_simconfig_from_raw(raw, config_dir)
    validate_config(base_cfg)
    base_cfg = derive_imaging_params(base_cfg)
    sweep_configs = expand_sweep_from_raw(raw, base_cfg, config_dir)

    return base_cfg, sweep_configs, raw
