from __future__ import annotations

import logging
from typing import Any, Dict, List

import yaml

from .dataclasses import (
    AntennaEntry, CannedObservatory, CLStokesSpectrum, ConfigError,
    CorpusMorphologyConfig, CorruptionConfig, CustomObservatory, FieldConfig,
    FaradayConfig, GainsConfig, ImagingConfig, NoiseConfig, ObservationConfig,
    ObservatoryConfig, ObserveCall, ObsposConfig, PredictionConfig,
    SimConfig, SkyModelConfig, SourceDef, SpectralLineConfig, SpwConfig,
    SweepAxis, SweepConfig, TRecsConfig, TRecsPolarizationConfig,
    TRecsSpectralConfig,
)

log = logging.getLogger(__name__)


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
            freqresolution=s.get('freqresolution')
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


def _parse_dist(d: dict) -> Dict[str, Any]:
    return dict(d)


def _parse_trecs(d: dict) -> TRecsConfig:
    cp_raw = _require(d, 'catalog_paths', 'sky_model.trecs')
    if not isinstance(cp_raw, dict) or not cp_raw:
        raise ConfigError("invalid_trecs_catalog_paths", "sky_model.trecs.catalog_paths",
                          "catalog_paths must be a dict with at least one of: agn, sfg")
    catalog_paths = {k: str(v) for k, v in cp_raw.items()}

    spec_d = _require(d, 'spectral', 'sky_model.trecs')
    spectral = TRecsSpectralConfig(
        mode=spec_d.get('mode', 'trecs_sed'),
        ref_freq=spec_d.get('ref_freq', '1.4GHz'),
        spidx_dist=_parse_dist(spec_d['spidx_dist']) if spec_d.get('spidx_dist') else None,
    )

    pol_d = _require(d, 'polarization', 'sky_model.trecs')
    polarization = TRecsPolarizationConfig(
        pol_fraction_source=pol_d.get('pol_fraction_source', 'trecs'),
        pol_fraction_dist=_parse_dist(pol_d['pol_fraction_dist']) if pol_d.get('pol_fraction_dist') else None,
        pol_spidx_dist=_parse_dist(pol_d['pol_spidx_dist']) if pol_d.get('pol_spidx_dist') else None,
        rm_dist=_parse_dist(pol_d['rm_dist']) if pol_d.get('rm_dist') else None,
        chi0_dist=_parse_dist(pol_d['chi0_dist']) if pol_d.get('chi0_dist') else None,
        n_rm_components=pol_d.get('n_rm_components', 1),
    )

    tile_center = list(d.get('tile_center_deg', [2.5, 2.5]))

    return TRecsConfig(
        catalog_paths=catalog_paths,
        flux_floor_jy=float(d.get('flux_floor_jy', 1e-4)),
        flux_floor_col=str(d.get('flux_floor_col', 'I1400')),
        field_size_arcsec=float(d.get('field_size_arcsec', 3600.0)),
        tile_center_deg=tile_center,
        seed=int(d.get('seed', 42)),
        spectral=spectral,
        polarization=polarization,
        readme_path=d.get('readme_path'),
    )


def _parse_corpus_morphology(d: dict) -> CorpusMorphologyConfig:
    balance = None
    if d.get('balance') is not None:
        balance = {str(k): float(v) for k, v in d['balance'].items()}
    return CorpusMorphologyConfig(
        field_type=d.get('field_type'),          # null OK → sampled per field
        balance=balance,
        seed=int(d.get('seed', 0)),
        repo_root=d.get('repo_root'),
    )


def _parse_sky_model(d: dict) -> SkyModelConfig:
    faraday = None
    if 'faraday' in d and d['faraday']:
        faraday = _parse_faraday(d['faraday'])
    cl_ss = None
    if 'cl_stokes_spectrum' in d and d['cl_stokes_spectrum']:
        cl_ss = _parse_cl_stokes_spectrum(d['cl_stokes_spectrum'])
    lines = _parse_spectral_lines(d.get('spectral_lines') or [])
    sources = None
    if 'sources' in d and d['sources']:
        sources = _parse_sources(d['sources'])
    trecs = None
    if 'trecs' in d and d['trecs']:
        trecs = _parse_trecs(d['trecs'])
    corpus_morphology = None
    if 'corpus_morphology' in d and d['corpus_morphology']:
        corpus_morphology = _parse_corpus_morphology(d['corpus_morphology'])
    return SkyModelConfig(
        stokes=d.get('stokes', 'I'),
        mode=_require(d, 'mode', 'sky_model'),
        cl_path=d.get('cl_path'),
        sources=sources,
        cl_stokes_spectrum=cl_ss,
        image_path=d.get('image_path'),
        ref_freq=d.get('ref_freq'),
        alpha_mode=d.get('alpha_mode'),
        alpha_value=d.get('alpha_value'),
        faraday=faraday,
        spectral_lines=lines,
        trecs=trecs,
        corpus_morphology=corpus_morphology,
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
        seed=d.get('seed'),
        noise=noise,
        gains=gains
    )


def _parse_imaging(d: dict) -> ImagingConfig:
    return ImagingConfig(
        enabled=bool(d.get('enabled', False)),
        deconvolver=d.get('deconvolver', 'hogbom'),
        nterms=d.get('nterms'),
        pbcor=bool(d.get('pbcor', False)),
        niter=int(d.get('niter', 500)),
        export_fits=bool(d.get('export_fits', False)),
        pblimit=float(d.get('pblimit', 0.05)),
        specmode=d.get('specmode'),
        stokes=d.get('stokes'),
        imsize=(int(d['imsize']) if d.get('imsize') is not None else None),
    )


def _parse_sweep(d: dict) -> SweepConfig:
    axes = [SweepAxis(parameter=a['parameter'], values=list(a['values']))
            for a in d.get('axes', [])]
    return SweepConfig(axes=axes)


def _parse_simconfig_from_raw(raw: dict) -> SimConfig:
    """Parse a raw dict (as loaded from YAML) into a SimConfig."""
    name = _require(raw, 'name', 'root')
    observatory = _parse_observatory(_require(raw, 'observatory', 'root'))
    observation = _parse_observation(_require(raw, 'observation', 'root'))
    sky_model = _parse_sky_model(_require(raw, 'sky_model', 'root'))
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


def load_config(path: str) -> SimConfig:
    """
    Parse a YAML config file into a SimConfig dataclass tree.
    Does NOT validate — call validate_config() separately.
    """
    with open(path, 'r') as fh:
        raw = yaml.safe_load(fh)
    return _parse_simconfig_from_raw(raw)
