"""
test_morphology.py -- Unit tests for casa_sim morphology machinery.

Covers:
  - TemplateLibrary: manifest loading, kind filtering, sample interface
  - build_template_library: rebuild from tng_raw produces 318 templates / 4 kinds
  - build_field: all four FieldTypes, output shape / dtype / non-negative
  - Hann apodization: centre-to-edge falloff, zero edges
  - _resample_template: correct output shape, non-negative, unit-max

No CASA dependency.  All tests run under pytest-fast:

    pixi run pytest-fast tests/test_morphology.py -v

Requires: data/morphology/lib/manifest.json to exist (pre-built library).
If you need to regenerate the library first:

    pixi run python -c "
    from casa_sim.skymodel.morphology_templates import build_template_library
    build_template_library('.')
    "
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent  # casa_sim repo root
LIB_DIR = REPO_ROOT / "data" / "morphology" / "lib"
TNG_RAW_DIR = REPO_ROOT / "data" / "morphology" / "tng_raw"

_lib_present = (LIB_DIR / "manifest.json").exists()
_raw_present = TNG_RAW_DIR.is_dir() and any(TNG_RAW_DIR.glob("*.png"))

skip_if_no_lib = pytest.mark.skipif(
    not _lib_present,
    reason="Template library not built (run build_template_library first)",
)
skip_if_no_raw = pytest.mark.skipif(
    not _raw_present,
    reason="TNG raw PNGs not present in data/morphology/tng_raw/",
)


# ---------------------------------------------------------------------------
# TemplateLibrary tests
# ---------------------------------------------------------------------------

class TestTemplateLibrary:
    """Tests for the TemplateLibrary loader / sampler."""

    @skip_if_no_lib
    def test_load_counts_318(self):
        """Total template count must be 318 (matching the built manifest)."""
        from casa_sim.skymodel.morphology_templates import TemplateLibrary
        lib = TemplateLibrary(REPO_ROOT)
        assert len(lib) == 318, f"Expected 318, got {len(lib)}"

    @skip_if_no_lib
    def test_four_kinds_present(self):
        """Manifest must contain exactly the four expected kind tags."""
        with open(LIB_DIR / "manifest.json") as fh:
            manifest = json.load(fh)
        kinds = {m["kind"] for m in manifest}
        assert kinds == {"shock", "compact", "web", "filament"}, (
            f"Unexpected kinds: {kinds}"
        )

    @skip_if_no_lib
    def test_kind_counts(self):
        """Confirm individual kind counts (shock=48, compact=36, web=36, filament=198)."""
        with open(LIB_DIR / "manifest.json") as fh:
            manifest = json.load(fh)
        from collections import Counter
        counts = Counter(m["kind"] for m in manifest)
        assert counts["shock"] == 48,    f"shock: {counts['shock']}"
        assert counts["compact"] == 36,  f"compact: {counts['compact']}"
        assert counts["web"] == 36,      f"web: {counts['web']}"
        assert counts["filament"] == 198, f"filament: {counts['filament']}"

    @skip_if_no_lib
    def test_filter_by_kind(self):
        """kind= filter narrows the visible library."""
        from casa_sim.skymodel.morphology_templates import TemplateLibrary
        lib = TemplateLibrary(REPO_ROOT, kinds=["shock"])
        assert len(lib) == 48

    @skip_if_no_lib
    def test_sample_returns_float32(self):
        """sample() returns (arr, meta); arr is float32, unit-max normalised."""
        from casa_sim.skymodel.morphology_templates import TemplateLibrary
        rng = np.random.default_rng(0)
        lib = TemplateLibrary(REPO_ROOT)
        arr, meta = lib.sample(rng)
        assert arr.dtype == np.float32
        assert arr.max() <= 1.0 + 1e-5
        assert arr.min() >= 0.0

    @skip_if_no_lib
    def test_sample_kind_restricts(self):
        """sample(kind='shock') always returns a shock template."""
        from casa_sim.skymodel.morphology_templates import TemplateLibrary
        rng = np.random.default_rng(99)
        lib = TemplateLibrary(REPO_ROOT)
        for _ in range(10):
            _, meta = lib.sample(rng, kind="shock")
            assert meta["kind"] == "shock"

    @skip_if_no_lib
    def test_invalid_kind_raises(self):
        """Requesting an absent kind raises ValueError."""
        from casa_sim.skymodel.morphology_templates import TemplateLibrary
        with pytest.raises(ValueError):
            TemplateLibrary(REPO_ROOT, kinds=["tail"])

    def test_missing_lib_raises_filenotfound(self):
        """TemplateLibrary raises FileNotFoundError if manifest is absent."""
        from casa_sim.skymodel.morphology_templates import TemplateLibrary
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(FileNotFoundError):
                TemplateLibrary(d)


# ---------------------------------------------------------------------------
# build_template_library rebuild test
# ---------------------------------------------------------------------------

class TestBuildTemplateLibrary:
    """Rebuild the library from tng_raw and verify counts."""

    @skip_if_no_raw
    def test_rebuild_produces_318_templates(self):
        """Rebuilding in a temp dir produces exactly 318 .npy files and a manifest."""
        from casa_sim.skymodel.morphology_templates import build_template_library
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Provide the tng_raw PNGs by symlinking / copying
            raw_dst = tmp / "data" / "morphology" / "tng_raw"
            raw_dst.mkdir(parents=True)
            for f in TNG_RAW_DIR.glob("*.png"):
                shutil.copy(f, raw_dst / f.name)

            lib_dir = build_template_library(tmp, force=True, contact_sheet=False)
            manifest_path = lib_dir / "manifest.json"
            assert manifest_path.exists(), "manifest.json not written"

            with open(manifest_path) as fh:
                manifest = json.load(fh)

            assert len(manifest) == 318, (
                f"Expected 318 templates, got {len(manifest)}"
            )
            kinds = {m["kind"] for m in manifest}
            assert kinds == {"shock", "compact", "web", "filament"}

            npy_files = list(lib_dir.glob("*.npy"))
            assert len(npy_files) == 318, (
                f"Expected 318 .npy files, got {len(npy_files)}"
            )


# ---------------------------------------------------------------------------
# build_field tests
# ---------------------------------------------------------------------------

class TestBuildField:
    """Tests for all four FieldType builders via the build_field() API."""

    @skip_if_no_lib
    @pytest.mark.parametrize("ft_name", [
        "POINT_ONLY",
        "DIFFUSE_DOMINANT",
        "CENTRAL_SHELL_FLOW",
        "FULLY_DIFFUSE",
    ])
    def test_output_shape_and_dtype(self, ft_name):
        """build_field() returns (imsize, imsize) float32 non-negative array."""
        from casa_sim.skymodel.morphology_field import build_field, FieldType
        ft = FieldType[ft_name]
        rng = np.random.default_rng(42)
        image, meta = build_field(
            rng, ft,
            ra_deg=278.5,
            dec_deg=-2.1,
            cell_arcsec=1.8,
            imsize=128,      # small for speed
            freq_hz=1.4e9,
            repo_root=REPO_ROOT,
        )
        assert isinstance(image, np.ndarray), "Expected numpy array"
        assert image.shape == (128, 128), f"Wrong shape: {image.shape}"
        assert image.dtype == np.float32, f"Wrong dtype: {image.dtype}"
        assert image.min() >= 0.0, f"Negative pixels (min={image.min():.4g})"

    @skip_if_no_lib
    def test_point_only_is_zero(self):
        """POINT_ONLY must return an all-zero array (T-RECS adds the sources)."""
        from casa_sim.skymodel.morphology_field import build_field, FieldType
        rng = np.random.default_rng(0)
        image, _ = build_field(
            rng, FieldType.POINT_ONLY,
            ra_deg=278.5, dec_deg=-2.1,
            cell_arcsec=1.8, imsize=128, freq_hz=1.4e9,
            repo_root=REPO_ROOT,
        )
        assert image.max() == 0.0, "POINT_ONLY should be all-zero"

    @skip_if_no_lib
    def test_diffuse_has_positive_flux(self):
        """DIFFUSE_DOMINANT must have some positive flux."""
        from casa_sim.skymodel.morphology_field import build_field, FieldType
        rng = np.random.default_rng(7)
        image, _ = build_field(
            rng, FieldType.DIFFUSE_DOMINANT,
            ra_deg=278.5, dec_deg=-2.1,
            cell_arcsec=1.8, imsize=128, freq_hz=1.4e9,
            repo_root=REPO_ROOT,
        )
        assert image.sum() > 0.0

    @skip_if_no_lib
    def test_meta_keys_present(self):
        """meta dict must include the coordinate-contract keys."""
        from casa_sim.skymodel.morphology_field import build_field, FieldType
        rng = np.random.default_rng(1)
        _, meta = build_field(
            rng, FieldType.DIFFUSE_DOMINANT,
            ra_deg=10.0, dec_deg=20.0,
            cell_arcsec=2.0, imsize=64, freq_hz=3e9,
            repo_root=REPO_ROOT,
        )
        for key in ("field_type", "ra_deg", "dec_deg", "cell_arcsec", "imsize",
                    "freq_hz", "sources"):
            assert key in meta, f"Missing meta key: {key}"
        assert meta["ra_deg"] == 10.0
        assert meta["dec_deg"] == 20.0


# ---------------------------------------------------------------------------
# Apodization / Hann window tests
# ---------------------------------------------------------------------------

class TestHannWindow:
    """Tests for the Hann apodization window (numpy implementation)."""

    def test_centre_is_one(self):
        """Centre pixel of a Hann window must be 1.0 (for odd size)."""
        from casa_sim.skymodel.morphology_field import _hann_window_2d
        n = 101
        win = _hann_window_2d(n)
        c = n // 2
        assert abs(float(win[c, c]) - 1.0) < 1e-5, f"Centre value: {win[c,c]}"

    def test_edges_are_zero(self):
        """All four edges of the Hann window must be (near) zero."""
        from casa_sim.skymodel.morphology_field import _hann_window_2d
        n = 64
        win = _hann_window_2d(n)
        assert win[0, :].max() < 1e-6, "Top edge not zero"
        assert win[-1, :].max() < 1e-6, "Bottom edge not zero"
        assert win[:, 0].max() < 1e-6, "Left edge not zero"
        assert win[:, -1].max() < 1e-6, "Right edge not zero"

    def test_non_negative(self):
        """Hann window is everywhere non-negative."""
        from casa_sim.skymodel.morphology_field import _hann_window_2d
        win = _hann_window_2d(128)
        assert win.min() >= 0.0


# ---------------------------------------------------------------------------
# _resample_template tests
# ---------------------------------------------------------------------------

class TestResampleTemplate:
    """Tests for the numpy resampler."""

    def test_output_shape(self):
        """Output is always (target_px, target_px)."""
        from casa_sim.skymodel.morphology_field import _resample_template
        rng = np.random.default_rng(5)
        template = rng.random((220, 220)).astype(np.float32)
        out = _resample_template(template, 64)
        assert out.shape == (64, 64)

    def test_output_dtype(self):
        """Output dtype is float32."""
        from casa_sim.skymodel.morphology_field import _resample_template
        template = np.ones((100, 100), dtype=np.float64)
        out = _resample_template(template, 50)
        assert out.dtype == np.float32

    def test_unit_max(self):
        """Output is unit-max normalised (max <= 1)."""
        from casa_sim.skymodel.morphology_field import _resample_template
        rng = np.random.default_rng(3)
        template = (rng.random((200, 200)) * 5.0).astype(np.float32)
        out = _resample_template(template, 100)
        assert out.max() <= 1.0 + 1e-5
        assert out.min() >= 0.0

    def test_non_square_input(self):
        """Non-square templates are handled (aspect ratio preserved via padding)."""
        from casa_sim.skymodel.morphology_field import _resample_template
        template = np.ones((300, 512), dtype=np.float32)
        out = _resample_template(template, 64)
        assert out.shape == (64, 64)


# ---------------------------------------------------------------------------
# make_example_fields_png integration test
# ---------------------------------------------------------------------------

class TestMakeExampleFieldsPng:

    @skip_if_no_lib
    def test_png_written(self):
        """make_example_fields_png() writes a file at the specified path."""
        from casa_sim.skymodel.morphology_field import make_example_fields_png
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "example_fields.png"
            result = make_example_fields_png(
                out,
                repo_root=REPO_ROOT,
                rng=np.random.default_rng(0),
                imsize=128,   # small for speed
                cell_arcsec=1.8,
            )
            assert result.exists(), "PNG not written"
            assert result.stat().st_size > 1000, "PNG too small (likely empty)"
