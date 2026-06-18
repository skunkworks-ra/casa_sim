"""
test_config.py — Design-level tests for casa_sim/config/

Covers:
  - _resolve_predictor_auto: full routing table
  - validate_config: all documented hard-error conditions
  - derive_imaging_params: derivation logic and skip conditions
  - expand_sweep_from_raw: Cartesian product expansion
  - apply_override: valid and invalid dotpaths

No CASA dependency — all tests run under pytest-fast.
"""

from __future__ import annotations

import textwrap
import tempfile
import os

import pytest

from casa_sim.config import (
    ConfigError,
    FaradayConfig,
    ImagingConfig,
    SkyModelConfig,
    load_config,
    validate_config,
    _resolve_predictor_auto,
    derive_imaging_params,
    expand_sweep_from_raw,
    apply_override,
    _parse_simconfig_from_raw,
)


# ---------------------------------------------------------------------------
# Minimal YAML builder — produces a valid base config as a string
# ---------------------------------------------------------------------------

_BASE_YAML = textwrap.dedent("""
    name: test_cfg
    observatory:
      mode: canned
      feeds: "perfect R L"
      canned:
        telescope: VLA
    observation:
      epoch: "UTC 2020/10/4/00:00:00"
      integration_time: "300s"
      use_hourangle: true
      fields:
        - name: src1
          direction: "J2000 19h59m28.5s +40d40m00.0s"
      spws:
        - name: LBand
          freq: "1.0GHz"
          deltafreq: "0.05GHz"
          nchan: 4
          stokes: "RR LL RL LR"
      observe_calls:
        - field: src1
          spw: LBand
          start_time: "-0.5h"
          stop_time: "+0.5h"
    sky_model:
      stokes: I
      mode: component_list
      cl_path: dummy.cl
    prediction:
      gridder: standard
      predictor: auto
      cell: 2arcsec
      imsize: 64
      normtype: flatsky
    corruption:
      seed: 42
      noise:
        enabled: false
        mode: simplenoise
    imaging:
      enabled: false
""")


def _load_yaml_str(yaml_str: str):
    import yaml
    raw = yaml.safe_load(yaml_str)
    cfg = _parse_simconfig_from_raw(raw)
    return cfg, raw


def _load_modified(overrides: dict):
    """Load the base YAML then apply a dict of dotpath→value overrides."""
    import yaml
    raw = yaml.safe_load(_BASE_YAML)
    for dotpath, value in overrides.items():
        raw = apply_override(raw, dotpath, value)
    return _parse_simconfig_from_raw(raw), raw


# ---------------------------------------------------------------------------
# _resolve_predictor_auto routing table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,gridder,expected", [
    ('component_list',   'standard',  'ft_dft'),
    ('component_list',   'mosaic',    'tclean'),
    ('component_list',   'awproject', 'tclean'),
    ('image_native',     'standard',  'sm_predict'),
    ('image_native',     'mosaic',    'tclean'),
    ('image_native',     'awproject', 'tclean'),
    ('image_extrapolate','standard',  'tclean'),
    ('t_recs',           'standard',  'tclean'),
    ('t_recs',           'mosaic',    'tclean'),
])
def test_predictor_routing_auto(mode, gridder, expected):
    cfg, _ = _load_modified({
        'sky_model.mode': mode,
        'prediction.gridder': gridder,
        'prediction.predictor': 'auto',
    })
    # Patch fields required by specific modes to avoid downstream errors
    if mode == 'image_extrapolate':
        cfg.sky_model.ref_freq = '1.0GHz'
        cfg.sky_model.alpha_mode = 'global'
        cfg.sky_model.alpha_value = -0.7
    if mode == 't_recs':
        cfg.sky_model.trecs = object()  # routing only checks mode, not trecs content

    assert _resolve_predictor_auto(cfg) == expected


def test_predictor_explicit_passthrough():
    cfg, _ = _load_modified({'prediction.predictor': 'ft_dft'})
    assert _resolve_predictor_auto(cfg) == 'ft_dft'


def test_predictor_sm_predict_demoted_to_tclean_for_mosaic():
    cfg, _ = _load_modified({
        'prediction.predictor': 'sm_predict',
        'prediction.gridder': 'mosaic',
    })
    assert _resolve_predictor_auto(cfg) == 'tclean'


# ---------------------------------------------------------------------------
# validate_config — hard error conditions
# ---------------------------------------------------------------------------

def _validate(overrides: dict):
    cfg, _ = _load_modified(overrides)
    validate_config(cfg)


def test_validate_passes_for_valid_config():
    cfg, _ = _load_yaml_str(_BASE_YAML)
    validate_config(cfg)  # must not raise


def test_validate_faraday_requires_iquv():
    cfg, _ = _load_modified({'sky_model.stokes': 'I'})
    cfg.sky_model.faraday = FaradayConfig(
        enabled=True, rm_mode='global', rm_value=50.0, ref_freq='1.0GHz'
    )
    with pytest.raises(ConfigError, match='faraday_requires_iquv'):
        validate_config(cfg)


def test_validate_faraday_requires_ref_freq():
    cfg, _ = _load_modified({'sky_model.stokes': 'IQUV'})
    cfg.sky_model.faraday = FaradayConfig(
        enabled=True, rm_mode='global', rm_value=50.0, ref_freq=''
    )
    with pytest.raises(ConfigError, match='missing_faraday_ref_freq'):
        validate_config(cfg)


def test_validate_image_extrapolate_requires_ref_freq():
    cfg, _ = _load_modified({'sky_model.mode': 'image_extrapolate'})
    cfg.sky_model.ref_freq = None
    with pytest.raises(ConfigError, match='missing_ref_freq'):
        validate_config(cfg)


def test_validate_sources_and_cl_path_exclusive():
    import yaml
    from casa_sim.config import SourceDef
    cfg, _ = _load_yaml_str(_BASE_YAML)
    cfg.sky_model.sources = [
        SourceDef(name='s1', direction='J2000 0h0m0s +0d0m0s',
                  flux=[1.0], ref_freq='1GHz')
    ]
    cfg.sky_model.cl_path = 'existing.cl'
    with pytest.raises(ConfigError, match='sources_and_cl_path_exclusive'):
        validate_config(cfg)


def test_validate_sources_require_component_list_mode():
    from casa_sim.config import SourceDef
    cfg, _ = _load_modified({'sky_model.mode': 'image_native'})
    cfg.sky_model.sources = [
        SourceDef(name='s1', direction='J2000 0h0m0s +0d0m0s',
                  flux=[1.0], ref_freq='1GHz')
    ]
    cfg.sky_model.cl_path = None
    with pytest.raises(ConfigError, match='sources_require_component_list_mode'):
        validate_config(cfg)


def test_validate_frac_pol_requires_chi():
    from casa_sim.config import SourceDef
    cfg, _ = _load_yaml_str(_BASE_YAML)
    cfg.sky_model.cl_path = None
    cfg.sky_model.sources = [
        SourceDef(name='s1', direction='J2000 0h0m0s +0d0m0s',
                  flux=[1.0], ref_freq='1GHz', frac_pol=0.1, chi=None)
    ]
    with pytest.raises(ConfigError, match='frac_pol_requires_chi'):
        validate_config(cfg)


def test_validate_frac_pol_incompatible_with_iquv_flux():
    from casa_sim.config import SourceDef
    cfg, _ = _load_yaml_str(_BASE_YAML)
    cfg.sky_model.cl_path = None
    cfg.sky_model.sources = [
        SourceDef(name='s1', direction='J2000 0h0m0s +0d0m0s',
                  flux=[1.0, 0.0, 0.0, 0.0], ref_freq='1GHz',
                  frac_pol=0.1, chi=45.0)
    ]
    with pytest.raises(ConfigError, match='frac_pol_with_iquv_flux'):
        validate_config(cfg)


def test_validate_t_recs_requires_iquv():
    cfg, _ = _load_modified({
        'sky_model.mode': 't_recs',
        'sky_model.stokes': 'I',
    })
    # Attach a minimal trecs block so the "missing_trecs_block" check is bypassed
    from casa_sim.config import TRecsConfig, TRecsSpectralConfig, TRecsPolarizationConfig
    cfg.sky_model.trecs = TRecsConfig(
        catalog_paths={'agn': '/nonexistent.dat'},
        flux_floor_jy=1e-4,
        flux_floor_col='I1400',
        field_size_arcsec=3600.0,
        tile_center_deg=[2.5, 2.5],
        seed=42,
        spectral=TRecsSpectralConfig(mode='trecs_sed', ref_freq='1.4GHz'),
        polarization=TRecsPolarizationConfig(
            pol_fraction_source='trecs',
            rm_dist={'kind': 'uniform', 'low': -100.0, 'high': 100.0},
            chi0_dist={'kind': 'uniform', 'low': 0.0, 'high': 3.14159},
        ),
    )
    with pytest.raises(ConfigError, match='trecs_requires_iquv'):
        validate_config(cfg)


def _make_trecs_cfg(n_rm_components=1):
    """Helper to build a minimal TRecsConfig with specified n_rm_components."""
    from casa_sim.config import TRecsConfig, TRecsSpectralConfig, TRecsPolarizationConfig
    return TRecsConfig(
        catalog_paths={'agn': '/nonexistent.dat'},
        flux_floor_jy=1e-4,
        flux_floor_col='I1400',
        field_size_arcsec=3600.0,
        tile_center_deg=[2.5, 2.5],
        seed=42,
        spectral=TRecsSpectralConfig(mode='trecs_sed', ref_freq='1.4GHz'),
        polarization=TRecsPolarizationConfig(
            pol_fraction_source='trecs',
            rm_dist={'kind': 'uniform', 'low': -100.0, 'high': 100.0},
            chi0_dist={'kind': 'uniform', 'low': 0.0, 'high': 3.14159},
            n_rm_components=n_rm_components,
        ),
    )


def test_validate_n_rm_components_zero_raises(monkeypatch):
    monkeypatch.setattr('os.path.exists', lambda p: True)
    cfg, _ = _load_modified({
        'sky_model.mode': 't_recs',
        'sky_model.stokes': 'IQUV',
    })
    cfg.sky_model.trecs = _make_trecs_cfg(n_rm_components=0)
    with pytest.raises(ConfigError, match='invalid_n_rm_components'):
        validate_config(cfg)


def test_validate_n_rm_components_invalid_dict_raises(monkeypatch):
    monkeypatch.setattr('os.path.exists', lambda p: True)
    cfg, _ = _load_modified({
        'sky_model.mode': 't_recs',
        'sky_model.stokes': 'IQUV',
    })
    cfg.sky_model.trecs = _make_trecs_cfg(n_rm_components={'kind': 'uniform', 'low': 1, 'high': 3})
    with pytest.raises(ConfigError, match='invalid_n_rm_components_dist'):
        validate_config(cfg)


def test_validate_mtmfs_requires_nterms():
    cfg, _ = _load_yaml_str(_BASE_YAML)
    cfg.imaging.enabled = True
    cfg.imaging.deconvolver = 'mtmfs'
    cfg.imaging.nterms = None
    with pytest.raises(ConfigError, match='missing_nterms'):
        validate_config(cfg)


def test_validate_simplenoise_requires_value():
    cfg, _ = _load_yaml_str(_BASE_YAML)
    cfg.corruption.noise.enabled = True
    cfg.corruption.noise.mode = 'simplenoise'
    cfg.corruption.noise.value = None
    with pytest.raises(ConfigError, match='missing_noise_value'):
        validate_config(cfg)


def test_validate_observe_call_undefined_field():
    import yaml
    raw = yaml.safe_load(_BASE_YAML)
    raw['observation']['observe_calls'][0]['field'] = 'nonexistent_field'
    cfg = _parse_simconfig_from_raw(raw)
    with pytest.raises(ConfigError, match='undefined_field_reference'):
        validate_config(cfg)


def test_validate_observe_call_undefined_spw():
    import yaml
    raw = yaml.safe_load(_BASE_YAML)
    raw['observation']['observe_calls'][0]['spw'] = 'nonexistent_spw'
    cfg = _parse_simconfig_from_raw(raw)
    with pytest.raises(ConfigError, match='undefined_spw_reference'):
        validate_config(cfg)


def test_validate_spectral_line_flux_profile_length_mismatch():
    from casa_sim.config import SpectralLineConfig
    cfg, _ = _load_yaml_str(_BASE_YAML)
    cfg.sky_model.spectral_lines = [
        SpectralLineConfig(
            name='HI', stokes='I',
            channels=[0, 1], flux_profile=[1.0],  # length mismatch
            mode='point', direction='J2000 0h0m0s +0d0m0s'
        )
    ]
    with pytest.raises(ConfigError, match='flux_profile_length_mismatch'):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# derive_imaging_params
# ---------------------------------------------------------------------------

def test_derive_skips_when_both_specified():
    cfg, _ = _load_modified({
        'prediction.cell': '1arcsec',
        'prediction.imsize': 128,
    })
    result = derive_imaging_params(cfg)
    # Should return the user-specified values, not compute derived ones
    assert result.prediction.cell == '1arcsec'
    assert result.prediction.imsize == 128
    assert result._derived_cell is None
    assert result._derived_imsize is None


def test_derive_computes_cell_and_imsize_for_canned_vla():
    import yaml
    raw = yaml.safe_load(_BASE_YAML)
    raw['prediction']['cell'] = None
    raw['prediction']['imsize'] = None
    cfg = _parse_simconfig_from_raw(raw)
    result = derive_imaging_params(cfg)
    assert result._derived_cell is not None
    assert result._derived_imsize is not None
    assert 'arcsec' in result._derived_cell
    assert result._derived_imsize > 0
    # imsize must be a power of 2
    n = result._derived_imsize
    assert n & (n - 1) == 0, f"imsize {n} is not a power of 2"


def test_derive_cell_partial_override_imsize_only():
    """If only cell is specified, imsize should be derived (and vice versa)."""
    import yaml
    raw = yaml.safe_load(_BASE_YAML)
    raw['prediction']['cell'] = '2arcsec'
    raw['prediction']['imsize'] = None
    cfg = _parse_simconfig_from_raw(raw)
    result = derive_imaging_params(cfg)
    assert result.prediction.cell == '2arcsec'
    assert result._derived_imsize is not None
    assert result._derived_cell is None


# ---------------------------------------------------------------------------
# expand_sweep_from_raw
# ---------------------------------------------------------------------------

_SWEEP_YAML = textwrap.dedent("""
    name: sweep_test
    observatory:
      mode: canned
      feeds: "perfect R L"
      canned:
        telescope: VLA
    observation:
      epoch: "UTC 2020/10/4/00:00:00"
      integration_time: "300s"
      use_hourangle: true
      fields:
        - name: src1
          direction: "J2000 19h59m28.5s +40d40m00.0s"
      spws:
        - name: LBand
          freq: "1.0GHz"
          deltafreq: "0.05GHz"
          nchan: 4
          stokes: "RR LL"
      observe_calls:
        - field: src1
          spw: LBand
          start_time: "-0.5h"
          stop_time: "+0.5h"
    sky_model:
      stokes: I
      mode: component_list
      cl_path: dummy.cl
    prediction:
      gridder: standard
      predictor: ft_dft
      cell: 2arcsec
      imsize: 64
      normtype: flatsky
    corruption:
      seed: 42
      noise:
        enabled: true
        mode: simplenoise
        value: "0.01Jy"
      gains:
        enabled: false
        mode: fbm
    imaging:
      enabled: false
    sweep:
      axes:
        - parameter: corruption.noise.value
          values: ["0.01Jy", "0.05Jy", "0.10Jy"]
        - parameter: corruption.seed
          values: [1, 2]
""")


def test_sweep_cartesian_product_count():
    import yaml
    raw = yaml.safe_load(_SWEEP_YAML)
    cfg = _parse_simconfig_from_raw(raw)
    result = expand_sweep_from_raw(raw, cfg)
    # 3 noise values × 2 seeds = 6 configs
    assert len(result) == 6


def test_sweep_overridden_values_are_correct():
    import yaml
    raw = yaml.safe_load(_SWEEP_YAML)
    cfg = _parse_simconfig_from_raw(raw)
    result = expand_sweep_from_raw(raw, cfg)
    noise_values = [c.corruption.noise.value for c in result]
    seeds = [c.corruption.seed for c in result]
    # Each noise value should appear twice (once per seed)
    assert noise_values.count('0.01Jy') == 2
    assert noise_values.count('0.05Jy') == 2
    assert noise_values.count('0.10Jy') == 2
    # Each seed should appear 3 times (once per noise value)
    assert seeds.count(1) == 3
    assert seeds.count(2) == 3


def test_sweep_no_sweep_block_returns_single_config():
    cfg, raw = _load_yaml_str(_BASE_YAML)
    result = expand_sweep_from_raw(raw, cfg)
    assert len(result) == 1
    assert result[0] is cfg


# ---------------------------------------------------------------------------
# apply_override
# ---------------------------------------------------------------------------

def test_apply_override_changes_value():
    import yaml
    raw = yaml.safe_load(_BASE_YAML)
    updated = apply_override(raw, 'corruption.seed', 99)
    assert updated['corruption']['seed'] == 99
    # Original not mutated
    assert raw['corruption']['seed'] == 42


def test_apply_override_nested_leaf():
    import yaml
    raw = yaml.safe_load(_BASE_YAML)
    updated = apply_override(raw, 'prediction.cell', '5arcsec')
    assert updated['prediction']['cell'] == '5arcsec'


def test_apply_override_invalid_intermediate_key():
    import yaml
    raw = yaml.safe_load(_BASE_YAML)
    with pytest.raises(KeyError):
        apply_override(raw, 'nonexistent.key', 'value')


def test_apply_override_invalid_leaf_key():
    import yaml
    raw = yaml.safe_load(_BASE_YAML)
    with pytest.raises(KeyError):
        apply_override(raw, 'corruption.nonexistent_leaf', 'value')


# ---------------------------------------------------------------------------
# load_config round-trip: all test YAMLs parse without error
# ---------------------------------------------------------------------------

import glob as _glob

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), 'configs')
_config_files = _glob.glob(os.path.join(_CONFIGS_DIR, '*.yaml'))


@pytest.mark.parametrize("config_path", _config_files,
                          ids=[os.path.basename(p) for p in _config_files])
def test_config_file_parses(config_path):
    """Every YAML in tests/configs/ must parse without raising."""
    cfg = load_config(config_path)
    assert cfg.name is not None
    assert cfg.observation is not None
    assert cfg.sky_model is not None
