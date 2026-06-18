"""
test_patchify.py -- Unit tests for M4: patchify + memmap writer.

Tests use the pre-existing smoketest_output FITS files (built by M3) without
requiring a live CASA run.  The smoketest FITS are at:
    {repo_root}/smoketest_output/smoketest_{dirty,model,psf}.fits

Covers:
  1. _fits_to_2d: collapses (FREQ, STOKES, Y, X) to 2D float32.
  2. _extract_patches: 512px → 16 patches of 128x128.
  3. _centre_crop_psf: output shape, peak=1.0, non-NaN.
  4. patchify_results: stack shapes/dtypes, dirty min<0, sky min>=0, psf peak≈1,
     field_id/config_idx lengths, manifest counts consistent.

Run with:
    pixi run python -m pytest tests/test_patchify.py -v

No CASA dependency.  Requires smoketest_output/ to exist (M3 artefacts).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths / skip guards
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SMOKE_DIR = REPO_ROOT / "smoketest_output"

_smoke_present = (
    (SMOKE_DIR / "smoketest_dirty.fits").exists()
    and (SMOKE_DIR / "smoketest_model.fits").exists()
    and (SMOKE_DIR / "smoketest_psf.fits").exists()
)
skip_if_no_smoke = pytest.mark.skipif(
    not _smoke_present,
    reason="smoketest_output FITS not present (run M3 smoke test first)",
)


# ---------------------------------------------------------------------------
# Minimal CorpusFieldResult stub (avoids importing CASA)
# ---------------------------------------------------------------------------

@dataclass
class _FakeSpec:
    field_idx: int = 0
    vla_config: str = "C"
    cell_arcsec: float = 2.44
    imsize: int = 512
    field_size_arcsec: float = 1249.28
    ha_start: str = "-1.0h"
    ha_stop: str = "+1.0h"
    integration_time: str = "60s"
    noise_jy: str = "0.001Jy"
    field_seed: int = 42
    field_type: Optional[str] = None


@dataclass
class _FakeResult:
    field_idx: int = 0
    spec: _FakeSpec = field(default_factory=_FakeSpec)
    dirty_fits: Optional[str] = None
    psf_fits: Optional[str] = None
    model_fits: Optional[str] = None
    elapsed_s: float = 0.0
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------

class TestFitsTo2D:
    """_fits_to_2d collapses 4D FITS to 2D float32."""

    @skip_if_no_smoke
    def test_dirty_shape(self):
        from casa_sim.patchify import _fits_to_2d
        plane = _fits_to_2d(str(SMOKE_DIR / "smoketest_dirty.fits"))
        assert plane.ndim == 2, f"Expected 2D, got {plane.ndim}D"
        # Dirty is imaged on the guard-band grid (768); patchify centre-crops it
        # to the 512 model size at pairing time.
        assert plane.shape == (768, 768), f"Expected (768,768), got {plane.shape}"

    @skip_if_no_smoke
    def test_dtype_float32(self):
        from casa_sim.patchify import _fits_to_2d
        plane = _fits_to_2d(str(SMOKE_DIR / "smoketest_dirty.fits"))
        assert plane.dtype == np.float32, f"Expected float32, got {plane.dtype}"

    @skip_if_no_smoke
    def test_dirty_has_negatives(self):
        """The dirty MFS-collapsed plane must have negative pixels (sidelobes/noise)."""
        from casa_sim.patchify import _fits_to_2d
        plane = _fits_to_2d(str(SMOKE_DIR / "smoketest_dirty.fits"))
        assert plane.min() < 0.0, (
            f"dirty plane has no negatives (min={plane.min():.4g}); "
            "FITS loading or channel-collapse is wrong."
        )

    @skip_if_no_smoke
    def test_psf_peak_is_one(self):
        """PSF FITS (before crop) should have peak ~1.0 (CASA normalises it)."""
        from casa_sim.patchify import _fits_to_2d
        plane = _fits_to_2d(str(SMOKE_DIR / "smoketest_psf.fits"))
        assert abs(plane.max() - 1.0) < 0.01, (
            f"PSF peak deviates from 1.0: {plane.max()}"
        )


class TestExtractPatches:
    """_extract_patches produces correct count and shape."""

    def test_512_to_16_patches(self):
        from casa_sim.patchify import _extract_patches
        img = np.random.default_rng(0).random((512, 512), dtype=np.float64)
        patches = _extract_patches(img, patch_size=128)
        assert patches.shape == (16, 128, 128), f"Expected (16,128,128), got {patches.shape}"

    def test_patches_dtype_float32(self):
        from casa_sim.patchify import _extract_patches
        img = np.ones((512, 512), dtype=np.float64)
        patches = _extract_patches(img, patch_size=128)
        assert patches.dtype == np.float32

    def test_patches_cover_all_pixels(self):
        """Sum of patches should match full image sum (no pixel dropped or duplicated)."""
        from casa_sim.patchify import _extract_patches
        img = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)
        patches = _extract_patches(img, patch_size=128)
        assert abs(patches.sum() - img.sum()) < 1.0, "Patch sum does not match image sum"


class TestCentreCropPsf:
    """_centre_crop_psf: shape, peak normalisation, no NaN."""

    def test_output_shape(self):
        from casa_sim.patchify import _centre_crop_psf
        psf = np.zeros((512, 512), dtype=np.float32)
        psf[256, 256] = 1.0
        out = _centre_crop_psf(psf, 128)
        assert out.shape == (128, 128), f"Expected (128,128), got {out.shape}"

    def test_peak_normalised(self):
        from casa_sim.patchify import _centre_crop_psf
        psf = np.zeros((512, 512), dtype=np.float32)
        psf[256, 256] = 3.7    # arbitrary peak > 1
        out = _centre_crop_psf(psf, 128)
        assert abs(out.max() - 1.0) < 1e-5, f"Peak not 1.0: {out.max()}"

    def test_no_nan(self):
        from casa_sim.patchify import _centre_crop_psf
        psf = np.random.default_rng(1).random((512, 512), dtype=np.float64).astype(np.float32)
        out = _centre_crop_psf(psf, 128)
        assert not np.isnan(out).any(), "NaN in PSF crop output"

    @skip_if_no_smoke
    def test_real_psf_from_fits(self):
        """Real PSF FITS → crop → peak=1.0, shape=(128,128), no NaN."""
        from casa_sim.patchify import _fits_to_2d, _centre_crop_psf
        psf_2d = _fits_to_2d(str(SMOKE_DIR / "smoketest_psf.fits"))
        out = _centre_crop_psf(psf_2d, 128)
        assert out.shape == (128, 128)
        assert abs(out.max() - 1.0) < 1e-4, f"PSF peak after crop: {out.max()}"
        assert not np.isnan(out).any()


# ---------------------------------------------------------------------------
# Integration test: patchify_results end-to-end
# ---------------------------------------------------------------------------

class TestPatchifyResults:
    """patchify_results writes correct stacks from real FITS."""

    @skip_if_no_smoke
    def test_stack_shapes_dtypes(self, tmp_path):
        from casa_sim.patchify import patchify_results

        result = _FakeResult(
            field_idx=0,
            spec=_FakeSpec(field_idx=0, vla_config="C"),
            dirty_fits=str(SMOKE_DIR / "smoketest_dirty.fits"),
            psf_fits=str(SMOKE_DIR / "smoketest_psf.fits"),
            model_fits=str(SMOKE_DIR / "smoketest_model.fits"),
            success=True,
        )

        stats = patchify_results([result], str(tmp_path), patch_size=128)

        dirty = np.load(stats.dirty_path, mmap_mode='r')
        sky = np.load(stats.sky_path, mmap_mode='r')
        psf = np.load(stats.psf_path, mmap_mode='r')
        fid = np.load(stats.field_id_path, mmap_mode='r')
        cidx = np.load(stats.config_idx_path, mmap_mode='r')

        # Shapes
        assert dirty.shape == (16, 128, 128), f"dirty shape: {dirty.shape}"
        assert sky.shape == (16, 128, 128), f"sky shape: {sky.shape}"
        assert psf.shape == (1, 128, 128), f"psf shape: {psf.shape}"
        assert fid.shape == (16,), f"field_id shape: {fid.shape}"
        assert cidx.shape == (1,), f"config_idx shape: {cidx.shape}"

        # Dtypes
        assert dirty.dtype == np.float32
        assert sky.dtype == np.float32
        assert psf.dtype == np.float32
        assert fid.dtype == np.int32
        assert cidx.dtype == np.int32

    @skip_if_no_smoke
    def test_dirty_has_negatives(self, tmp_path):
        """Stack dirty must retain negative values (sidelobes + noise)."""
        from casa_sim.patchify import patchify_results

        result = _FakeResult(
            field_idx=0,
            dirty_fits=str(SMOKE_DIR / "smoketest_dirty.fits"),
            psf_fits=str(SMOKE_DIR / "smoketest_psf.fits"),
            model_fits=str(SMOKE_DIR / "smoketest_model.fits"),
        )
        stats = patchify_results([result], str(tmp_path))
        dirty = np.load(stats.dirty_path)
        assert dirty.min() < 0.0, (
            f"dirty stack has no negatives (min={dirty.min():.4g}); "
            "clipping must have occurred — guardrail violated."
        )

    @skip_if_no_smoke
    def test_sky_non_negative(self, tmp_path):
        """Stack sky must be non-negative (model image guardrail)."""
        from casa_sim.patchify import patchify_results

        result = _FakeResult(
            field_idx=0,
            dirty_fits=str(SMOKE_DIR / "smoketest_dirty.fits"),
            psf_fits=str(SMOKE_DIR / "smoketest_psf.fits"),
            model_fits=str(SMOKE_DIR / "smoketest_model.fits"),
        )
        stats = patchify_results([result], str(tmp_path))
        sky = np.load(stats.sky_path)
        assert sky.min() >= 0.0, f"sky stack has negatives: min={sky.min()}"

    @skip_if_no_smoke
    def test_psf_peak_approx_one(self, tmp_path):
        """PSF stack peak must be ~1.0 per field."""
        from casa_sim.patchify import patchify_results

        result = _FakeResult(
            field_idx=0,
            dirty_fits=str(SMOKE_DIR / "smoketest_dirty.fits"),
            psf_fits=str(SMOKE_DIR / "smoketest_psf.fits"),
            model_fits=str(SMOKE_DIR / "smoketest_model.fits"),
        )
        stats = patchify_results([result], str(tmp_path))
        psf = np.load(stats.psf_path)
        assert abs(psf.max() - 1.0) < 1e-3, f"PSF stack peak: {psf.max()}"

    @skip_if_no_smoke
    def test_manifest_consistent(self, tmp_path):
        """manifest.json counts must be self-consistent."""
        from casa_sim.patchify import patchify_results

        result = _FakeResult(
            field_idx=7,    # non-zero index to verify field_id
            spec=_FakeSpec(field_idx=7, vla_config="B"),
            dirty_fits=str(SMOKE_DIR / "smoketest_dirty.fits"),
            psf_fits=str(SMOKE_DIR / "smoketest_psf.fits"),
            model_fits=str(SMOKE_DIR / "smoketest_model.fits"),
        )
        stats = patchify_results([result], str(tmp_path))

        with open(stats.manifest_path) as fh:
            manifest = json.load(fh)

        assert manifest["n_fields"] == 1
        assert manifest["n_patches"] == 16
        assert manifest["patches_per_field"] == 16
        assert manifest["patch_size"] == 128
        assert manifest["stride"] == 128
        assert len(manifest["fields"]) == 1
        assert manifest["fields"][0]["field_idx"] == 7

    @skip_if_no_smoke
    def test_field_id_values(self, tmp_path):
        """field_id array must contain the correct field index for every patch."""
        from casa_sim.patchify import patchify_results

        result = _FakeResult(
            field_idx=42,
            spec=_FakeSpec(field_idx=42),
            dirty_fits=str(SMOKE_DIR / "smoketest_dirty.fits"),
            psf_fits=str(SMOKE_DIR / "smoketest_psf.fits"),
            model_fits=str(SMOKE_DIR / "smoketest_model.fits"),
        )
        stats = patchify_results([result], str(tmp_path))
        fid = np.load(stats.field_id_path)
        assert np.all(fid == 42), f"field_id values wrong: unique={np.unique(fid)}"
