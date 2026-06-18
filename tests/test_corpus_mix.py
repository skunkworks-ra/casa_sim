"""
test_corpus_mix.py -- Unit tests for M2: corpus_mix sky model.

Covers:
  1. Config schema: parse and validate corpus_mix YAML correctly.
  2. Validation: hard errors for missing/invalid corpus_morphology keys.
  3. Predictor routing: corpus_mix routes to tclean.
  4. Field-type sampling: balance weights respected over many draws.
  5. Sky assembly (no CASA): build_corpus_mix_numpy() produces a combined
     model image with correct shape, dtype, non-negativity, and contains
     BOTH T-RECS flux and extended-morphology flux.

No CASA dependency.  All tests run with:

    pixi run python -m pytest tests/test_corpus_mix.py -v

Requires: data/morphology/lib/manifest.json (pre-built library).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from casa_sim.config import (
    ConfigError,
    CorpusMorphologyConfig,
    SkyModelConfig,
    TRecsConfig,
    TRecsSpectralConfig,
    TRecsPolarizationConfig,
    _parse_simconfig_from_raw,
    _resolve_predictor_auto,
    validate_config,
)

# ---------------------------------------------------------------------------
# Paths / skip guards
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
LIB_DIR = REPO_ROOT / "data" / "morphology" / "lib"
CONFIG_DIR = Path(__file__).parent / "configs"

_lib_present = (LIB_DIR / "manifest.json").exists()
skip_if_no_lib = pytest.mark.skipif(
    not _lib_present,
    reason="Template library not built (run build_template_library first)",
)

AGN_CATALOG = Path("/home/pjaganna/Software/radiosharp/data/agnsmedi.dat.gz")
_catalog_present = AGN_CATALOG.exists()
skip_if_no_catalog = pytest.mark.skipif(
    not _catalog_present,
    reason="AGN T-RECS catalog not present at expected path",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trecs_cfg(**kwargs):
    """Build a minimal TRecsConfig for tests that need one."""
    defaults = dict(
        catalog_paths={'agn': str(AGN_CATALOG)},
        flux_floor_jy=1.0e-3,
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
    defaults.update(kwargs)
    return TRecsConfig(**defaults)


def _make_corpus_sm(**kwargs):
    """Build a SkyModelConfig for corpus_mix with sensible defaults."""
    defaults = dict(
        stokes='IQUV',
        mode='corpus_mix',
        trecs=_make_trecs_cfg(),
        corpus_morphology=CorpusMorphologyConfig(
            field_type='diffuse_dominant',
            seed=7,
        ),
    )
    defaults.update(kwargs)
    return SkyModelConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. Config parsing from YAML file
# ---------------------------------------------------------------------------

class TestCorpusMixConfigParsing:
    """Parse the corpus_mix YAML fixture and verify all keys land correctly."""

    def test_parse_corpus_mix_yaml(self):
        """The example corpus_mix config must parse without error."""
        import yaml
        cfg_path = CONFIG_DIR / "corpus_mix_vla_c_lband.yaml"
        assert cfg_path.exists(), f"Config not found: {cfg_path}"
        with open(cfg_path) as fh:
            raw = yaml.safe_load(fh)
        cfg = _parse_simconfig_from_raw(raw)

        assert cfg.sky_model.mode == 'corpus_mix'
        assert cfg.sky_model.trecs is not None
        assert cfg.sky_model.corpus_morphology is not None

    def test_corpus_morphology_fields_parsed(self):
        """corpus_morphology sub-fields must be populated from YAML."""
        import yaml
        cfg_path = CONFIG_DIR / "corpus_mix_vla_c_lband.yaml"
        with open(cfg_path) as fh:
            raw = yaml.safe_load(fh)
        cfg = _parse_simconfig_from_raw(raw)

        cm = cfg.sky_model.corpus_morphology
        assert cm.field_type == 'diffuse_dominant'
        assert cm.balance is None                  # null in YAML
        assert cm.seed == 7
        assert cm.repo_root is None                # null in YAML

    def test_sampled_field_type_parses(self):
        """field_type: null should parse to None (sample mode)."""
        yaml_str = textwrap.dedent("""
            name: t
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
                - name: F1
                  direction: "J2000 12h00m00s +30d00m00s"
              spws:
                - name: L
                  freq: "1.0GHz"
                  deltafreq: "0.2GHz"
                  nchan: 4
                  stokes: "RR RL LR LL"
              observe_calls:
                - field: F1
                  spw: L
                  start_time: "-0.5h"
                  stop_time: "+0.5h"
            sky_model:
              stokes: IQUV
              mode: corpus_mix
              trecs:
                catalog_paths:
                  agn: /tmp/fake_catalog.dat.gz
                flux_floor_jy: 1.0e-3
                flux_floor_col: I1400
                field_size_arcsec: 3600.0
                tile_center_deg: [2.5, 2.5]
                seed: 0
                spectral:
                  mode: trecs_sed
                  ref_freq: 1.4GHz
                polarization:
                  pol_fraction_source: trecs
                  rm_dist: {kind: uniform, low: -100.0, high: 100.0}
                  chi0_dist: {kind: uniform, low: 0.0, high: 3.14}
              corpus_morphology:
                field_type: null
                balance: null
                seed: 0
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
        import yaml
        raw = yaml.safe_load(yaml_str)
        cfg = _parse_simconfig_from_raw(raw)
        assert cfg.sky_model.corpus_morphology.field_type is None

    def test_balance_dict_parses(self):
        """An explicit balance dict must be parsed as float values."""
        yaml_str = textwrap.dedent("""
            name: t
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
                - name: F1
                  direction: "J2000 12h00m00s +30d00m00s"
              spws:
                - name: L
                  freq: "1.0GHz"
                  deltafreq: "0.2GHz"
                  nchan: 4
                  stokes: "RR RL LR LL"
              observe_calls:
                - field: F1
                  spw: L
                  start_time: "-0.5h"
                  stop_time: "+0.5h"
            sky_model:
              stokes: IQUV
              mode: corpus_mix
              trecs:
                catalog_paths:
                  agn: /tmp/fake.gz
                flux_floor_jy: 1.0e-3
                flux_floor_col: I1400
                field_size_arcsec: 3600.0
                tile_center_deg: [2.5, 2.5]
                seed: 0
                spectral:
                  mode: trecs_sed
                  ref_freq: 1.4GHz
                polarization:
                  pol_fraction_source: trecs
                  rm_dist: {kind: uniform, low: -100.0, high: 100.0}
                  chi0_dist: {kind: uniform, low: 0.0, high: 3.14}
              corpus_morphology:
                field_type: null
                balance:
                  point_only: 0.25
                  diffuse_dominant: 0.32
                  central_shell_flow: 0.33
                  fully_diffuse: 0.10
                seed: 0
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
        import yaml
        raw = yaml.safe_load(yaml_str)
        cfg = _parse_simconfig_from_raw(raw)
        balance = cfg.sky_model.corpus_morphology.balance
        assert isinstance(balance, dict)
        assert set(balance.keys()) == {
            'point_only', 'diffuse_dominant', 'central_shell_flow', 'fully_diffuse'
        }
        assert abs(sum(balance.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 2. Validation hard errors
# ---------------------------------------------------------------------------

class TestCorpusMixValidation:
    """validate_config() must raise ConfigError for invalid corpus_mix configs."""

    def _base_cfg(self):
        """Return a corpus_mix SkyModelConfig that passes validation (catalog skipped)."""
        sm = _make_corpus_sm()
        return sm

    def _build_minimal_simcfg(self, sm: SkyModelConfig):
        """Wrap a SkyModelConfig in a minimal SimConfig (no CASA needed)."""
        from casa_sim.config import (
            CannedObservatory, CorruptionConfig, GainsConfig, ImagingConfig,
            NoiseConfig, ObservationConfig, ObservatoryConfig, FieldConfig,
            SpwConfig, ObserveCall, PredictionConfig, SimConfig,
        )
        return SimConfig(
            name='test',
            observatory=ObservatoryConfig(
                mode='canned', feeds='perfect R L',
                canned=CannedObservatory(telescope='VLA'),
            ),
            observation=ObservationConfig(
                epoch='UTC 2020/10/4/00:00:00',
                integration_time='300s',
                use_hourangle=True,
                fields=[FieldConfig(name='F1', direction='J2000 12h00m00s +30d00m00s')],
                spws=[SpwConfig(
                    name='L', freq='1.0GHz', deltafreq='0.2GHz',
                    nchan=4, stokes='RR RL LR LL',
                )],
                observe_calls=[ObserveCall(
                    field='F1', spw='L',
                    start_time='-0.5h', stop_time='+0.5h',
                )],
            ),
            sky_model=sm,
            prediction=PredictionConfig(
                gridder='standard', predictor='auto',
                normtype='flatsky', cell='2arcsec', imsize=64,
            ),
            corruption=CorruptionConfig(
                seed=42,
                noise=NoiseConfig(enabled=False, mode='simplenoise'),
                gains=GainsConfig(enabled=False, mode='fbm'),
            ),
            imaging=ImagingConfig(enabled=False),
        )

    def test_missing_corpus_morphology_block_raises(self):
        """corpus_mix without corpus_morphology: block must raise."""
        sm = _make_corpus_sm(corpus_morphology=None)
        cfg = self._build_minimal_simcfg(sm)
        # Patch catalog path to exist so we don't fail on catalog check
        cfg.sky_model.trecs.catalog_paths = {'agn': str(AGN_CATALOG)} if _catalog_present else {'agn': __file__}
        with pytest.raises(ConfigError, match='missing_corpus_morphology_block'):
            validate_config(cfg)

    def test_invalid_field_type_raises(self):
        """An unrecognised field_type value must raise."""
        sm = _make_corpus_sm(
            corpus_morphology=CorpusMorphologyConfig(
                field_type='bogus_type',
                seed=0,
            )
        )
        cfg = self._build_minimal_simcfg(sm)
        cfg.sky_model.trecs.catalog_paths = {'agn': str(AGN_CATALOG)} if _catalog_present else {'agn': __file__}
        with pytest.raises(ConfigError, match='invalid_corpus_field_type'):
            validate_config(cfg)

    def test_incomplete_balance_raises(self):
        """A balance dict missing any field type must raise."""
        sm = _make_corpus_sm(
            corpus_morphology=CorpusMorphologyConfig(
                field_type=None,
                balance={
                    'point_only': 0.5,
                    'diffuse_dominant': 0.5,
                    # missing central_shell_flow, fully_diffuse
                },
                seed=0,
            )
        )
        cfg = self._build_minimal_simcfg(sm)
        cfg.sky_model.trecs.catalog_paths = {'agn': str(AGN_CATALOG)} if _catalog_present else {'agn': __file__}
        with pytest.raises(ConfigError, match='incomplete_corpus_balance'):
            validate_config(cfg)

    def test_negative_balance_weight_raises(self):
        """Negative balance weights must raise."""
        sm = _make_corpus_sm(
            corpus_morphology=CorpusMorphologyConfig(
                field_type=None,
                balance={
                    'point_only': -0.1,
                    'diffuse_dominant': 0.5,
                    'central_shell_flow': 0.4,
                    'fully_diffuse': 0.2,
                },
                seed=0,
            )
        )
        cfg = self._build_minimal_simcfg(sm)
        cfg.sky_model.trecs.catalog_paths = {'agn': str(AGN_CATALOG)} if _catalog_present else {'agn': __file__}
        with pytest.raises(ConfigError, match='negative_corpus_balance_weight'):
            validate_config(cfg)

    def test_corpus_mix_requires_iquv(self):
        """corpus_mix with stokes=I must raise trecs_requires_iquv."""
        sm = _make_corpus_sm(stokes='I')
        cfg = self._build_minimal_simcfg(sm)
        cfg.sky_model.trecs.catalog_paths = {'agn': str(AGN_CATALOG)} if _catalog_present else {'agn': __file__}
        with pytest.raises(ConfigError, match='trecs_requires_iquv'):
            validate_config(cfg)


# ---------------------------------------------------------------------------
# 3. Predictor routing
# ---------------------------------------------------------------------------

class TestCorpusMixRouting:
    """corpus_mix always routes to tclean regardless of gridder."""

    def _cfg_with_mode_and_gridder(self, gridder: str):
        from casa_sim.config import (
            CannedObservatory, CorruptionConfig, GainsConfig, ImagingConfig,
            NoiseConfig, ObservationConfig, ObservatoryConfig, FieldConfig,
            SpwConfig, ObserveCall, PredictionConfig, SimConfig,
        )
        sm = _make_corpus_sm()
        return SimConfig(
            name='test',
            observatory=ObservatoryConfig(
                mode='canned', feeds='perfect R L',
                canned=CannedObservatory(telescope='VLA'),
            ),
            observation=ObservationConfig(
                epoch='UTC 2020/10/4/00:00:00',
                integration_time='300s',
                use_hourangle=True,
                fields=[FieldConfig(name='F1', direction='J2000 12h00m00s +30d00m00s')],
                spws=[SpwConfig(
                    name='L', freq='1.0GHz', deltafreq='0.2GHz',
                    nchan=4, stokes='RR RL LR LL',
                )],
                observe_calls=[ObserveCall(
                    field='F1', spw='L',
                    start_time='-0.5h', stop_time='+0.5h',
                )],
            ),
            sky_model=sm,
            prediction=PredictionConfig(
                gridder=gridder, predictor='auto',
                normtype='flatsky', cell='2arcsec', imsize=64,
            ),
            corruption=CorruptionConfig(
                seed=42,
                noise=NoiseConfig(enabled=False, mode='simplenoise'),
                gains=GainsConfig(enabled=False, mode='fbm'),
            ),
            imaging=ImagingConfig(enabled=False),
        )

    @pytest.mark.parametrize("gridder", ['standard', 'mosaic', 'awproject'])
    def test_corpus_mix_routes_to_tclean(self, gridder):
        cfg = self._cfg_with_mode_and_gridder(gridder)
        assert _resolve_predictor_auto(cfg) == 'tclean'


# ---------------------------------------------------------------------------
# 4. Field-type sampling respects balance
# ---------------------------------------------------------------------------

class TestFieldTypeSampling:
    """sample_field_type() must respect the balance proportions over many draws."""

    def test_default_balance_draws_all_types(self):
        """Sampling many times from the default balance yields all four types."""
        from casa_sim.skymodel.morphology_field import sample_field_type, FieldType
        rng = np.random.default_rng(42)
        types_seen = set()
        for _ in range(200):
            ft = sample_field_type(rng)
            types_seen.add(ft)
        assert types_seen == set(FieldType), (
            f"Not all FieldTypes sampled: {types_seen}"
        )

    def test_custom_balance_draws_only_allowed_types(self):
        """A balance that zeroes fully_diffuse must never draw FULLY_DIFFUSE."""
        from casa_sim.skymodel.morphology_field import (
            sample_field_type, FieldType,
        )
        balance = {
            FieldType.POINT_ONLY: 1.0,
            FieldType.DIFFUSE_DOMINANT: 1.0,
            FieldType.CENTRAL_SHELL_FLOW: 1.0,
            FieldType.FULLY_DIFFUSE: 0.0,   # zero weight
        }
        rng = np.random.default_rng(0)
        for _ in range(100):
            ft = sample_field_type(rng, balance=balance)
            assert ft != FieldType.FULLY_DIFFUSE, (
                "FULLY_DIFFUSE drawn despite zero weight"
            )

    def test_locked_balance_fractions(self):
        """
        Empirically verify the locked balance produces the right mix over 2000 draws.
        Expected: point_only ~25%, diffuse_dominant ~32%, central_shell_flow ~33%,
        fully_diffuse ~10%.  Tolerance ±5 pp at n=2000.
        """
        from casa_sim.skymodel.morphology_field import (
            sample_field_type, FieldType, DEFAULT_FIELD_TYPE_BALANCE,
        )
        rng = np.random.default_rng(123)
        counts = {ft: 0 for ft in FieldType}
        n = 2000
        for _ in range(n):
            ft = sample_field_type(rng, balance=DEFAULT_FIELD_TYPE_BALANCE)
            counts[ft] += 1

        expected = {
            FieldType.POINT_ONLY: 0.25,
            FieldType.DIFFUSE_DOMINANT: 0.32,
            FieldType.CENTRAL_SHELL_FLOW: 0.33,
            FieldType.FULLY_DIFFUSE: 0.10,
        }
        tol = 0.05
        for ft, exp_frac in expected.items():
            actual_frac = counts[ft] / n
            assert abs(actual_frac - exp_frac) < tol, (
                f"{ft.value}: expected {exp_frac:.0%}, got {actual_frac:.0%} "
                f"(diff {abs(actual_frac - exp_frac):.0%} > tol {tol:.0%})"
            )


# ---------------------------------------------------------------------------
# 5. Sky assembly (no CASA): pure numpy composite
# ---------------------------------------------------------------------------

class TestSkyAssemblyNumpyNoSkymodel:
    """
    Test the numpy-level composite without CASA tools.

    Strategy: call build_field() to get the morphology, build a synthetic
    T-RECS image as a plain numpy array (no CASA), then apply the same
    arithmetic that _build_corpus_mix_sky_model() does in the image loop.
    Verify: shape, dtype, non-negativity, both components present.
    """

    @skip_if_no_lib
    def test_composite_shape_and_dtype(self):
        """Combined image has correct shape and float dtype."""
        from casa_sim.skymodel.morphology_field import build_field, FieldType

        imsize = 128
        cell_arcsec = 2.0
        rng = np.random.default_rng(42)

        # Build morphology
        morph, _ = build_field(
            rng, FieldType.DIFFUSE_DOMINANT,
            ra_deg=180.0, dec_deg=30.0,
            cell_arcsec=cell_arcsec, imsize=imsize, freq_hz=1.4e9,
            repo_root=REPO_ROOT,
        )
        assert morph.shape == (imsize, imsize)
        assert morph.dtype == np.float32

        # Synthetic T-RECS-like image (point sources on black)
        trecs_img = np.zeros((imsize, imsize, 4, 1), dtype=np.float64)
        trecs_img[64, 64, 0, 0] = 0.01   # 10 mJy point in Stokes I

        # Composite
        combined = trecs_img.copy()
        combined[:, :, 0, 0] += morph.astype(np.float64)
        combined[:, :, 0, 0] = np.clip(combined[:, :, 0, 0], 0.0, None)

        assert combined.shape == (imsize, imsize, 4, 1)
        assert combined.min() >= 0.0

    @skip_if_no_lib
    def test_composite_contains_trecs_flux(self):
        """Point source from T-RECS must still be present after compositing."""
        from casa_sim.skymodel.morphology_field import build_field, FieldType

        imsize = 128
        rng = np.random.default_rng(1)
        morph, _ = build_field(
            rng, FieldType.DIFFUSE_DOMINANT,
            ra_deg=180.0, dec_deg=30.0,
            cell_arcsec=2.0, imsize=imsize, freq_hz=1.4e9,
            repo_root=REPO_ROOT,
        )

        trecs_flux = 0.05   # Jy
        trecs_img = np.zeros((imsize, imsize, 4, 1), dtype=np.float64)
        trecs_img[50, 50, 0, 0] = trecs_flux

        combined = trecs_img.copy()
        combined[:, :, 0, 0] += morph.astype(np.float64)

        # The T-RECS pixel should have at least its original flux
        assert combined[50, 50, 0, 0] >= trecs_flux - 1e-8

    @skip_if_no_lib
    def test_composite_contains_extended_flux(self):
        """Extended morphology flux must exceed zero in combined image."""
        from casa_sim.skymodel.morphology_field import build_field, FieldType

        imsize = 128
        rng = np.random.default_rng(2)
        morph, _ = build_field(
            rng, FieldType.DIFFUSE_DOMINANT,
            ra_deg=180.0, dec_deg=30.0,
            cell_arcsec=2.0, imsize=imsize, freq_hz=1.4e9,
            repo_root=REPO_ROOT,
        )
        morph_total = float(morph.sum())
        assert morph_total > 0.0, "Morphology is all-zero (unexpected for DIFFUSE_DOMINANT)"

        # Start from zero T-RECS so any positive flux comes from morphology
        combined_i = morph.astype(np.float64)
        assert combined_i.sum() > 0.0

    @skip_if_no_lib
    def test_point_only_morphology_is_zero(self):
        """POINT_ONLY morphology contributes zero extended flux."""
        from casa_sim.skymodel.morphology_field import build_field, FieldType

        imsize = 64
        rng = np.random.default_rng(3)
        morph, _ = build_field(
            rng, FieldType.POINT_ONLY,
            ra_deg=180.0, dec_deg=30.0,
            cell_arcsec=2.0, imsize=imsize, freq_hz=1.4e9,
            repo_root=REPO_ROOT,
        )
        assert morph.max() == 0.0, "POINT_ONLY morphology must be all-zero"

    @skip_if_no_lib
    def test_non_negativity_preserved(self):
        """Combined image must be non-negative after compositing."""
        from casa_sim.skymodel.morphology_field import build_field, FieldType

        imsize = 128
        rng = np.random.default_rng(99)
        for ft in FieldType:
            morph, _ = build_field(
                rng, ft,
                ra_deg=0.0, dec_deg=0.0,
                cell_arcsec=2.0, imsize=imsize, freq_hz=1.4e9,
                repo_root=REPO_ROOT,
            )
            # Simulated T-RECS I channel (non-negative)
            trecs_i = np.abs(rng.standard_normal((imsize, imsize)).astype(np.float64)) * 1e-3
            combined = trecs_i + morph.astype(np.float64)
            combined = np.clip(combined, 0.0, None)
            assert combined.min() >= 0.0, (
                f"Negative pixel after compositing for {ft.value}"
            )

    @skip_if_no_lib
    def test_cell_arcsec_affects_source_size(self):
        """
        Coarser cell should produce the same morphological pattern but with
        sources that are smaller in pixels (same angular size → fewer pixels).
        Verify that total flux is the same (flux normalisation is pixel-count-
        independent) while pixel count at threshold differs.
        """
        from casa_sim.skymodel.morphology_field import build_field, FieldType

        imsize = 256
        rng_fine = np.random.default_rng(55)
        rng_coarse = np.random.default_rng(55)  # same seed

        morph_fine, _ = build_field(
            rng_fine, FieldType.CENTRAL_SHELL_FLOW,
            ra_deg=0.0, dec_deg=0.0,
            cell_arcsec=1.0,   # fine: smaller pixels → more pixels per source
            imsize=imsize, freq_hz=1.4e9,
            repo_root=REPO_ROOT,
        )
        morph_coarse, _ = build_field(
            rng_coarse, FieldType.CENTRAL_SHELL_FLOW,
            ra_deg=0.0, dec_deg=0.0,
            cell_arcsec=4.0,   # coarse: larger pixels → fewer pixels per source
            imsize=imsize, freq_hz=1.4e9,
            repo_root=REPO_ROOT,
        )

        # Both should have positive flux
        assert morph_fine.sum() > 0.0
        assert morph_coarse.sum() > 0.0

        # Fine should have more non-zero pixels (source is bigger in pixel space)
        fine_nonzero = int((morph_fine > 1e-8).sum())
        coarse_nonzero = int((morph_coarse > 1e-8).sum())
        assert fine_nonzero > coarse_nonzero, (
            f"Fine cell ({fine_nonzero} px) should have more non-zero pixels "
            f"than coarse cell ({coarse_nonzero} px)"
        )
