"""
tools/smoketest_corpus.py -- M3 smoke test: one corpus field through CASA.

Runs a single C-config L-band field with:
    - HA track: -2h .. +2h, 60s integrations
    - simplenoise: 3 mJy (moderate; bright enough to show sidelobes clearly)
    - flux floor: 1 mJy AGN-only T-RECS
    - field_type: sampled from the locked balance

Then renders a triptych PNG:
    casa_sim/data/morphology/lib/smoketest_dirty_model_psf.png
    (model | dirty | psf)
    - model: asinh stretch, viridis
    - dirty: signed diverging colormap (RdBu_r) centred at zero, asinh stretch
    - psf:   log stretch, viridis

Usage (from repo root):
    pixi run python tools/smoketest_corpus.py

The script calls build_corpus_field() directly with a fixed spec so results
are deterministic (seed=42, C-config, -2h..+2h, 3 mJy).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup -- do this before any casa_sim imports
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoketest_corpus")

# ---------------------------------------------------------------------------
# Repo root resolution -- add casa_sim to sys.path if needed
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np

from casa_sim.corpus import (
    CorpusFieldSpec,
    build_corpus_field,
    _cell_arcsec_for_config,
)

# ---------------------------------------------------------------------------
# Fixed smoke-test parameters
# ---------------------------------------------------------------------------

SMOKETEST_SEED = 42
SMOKETEST_VLA_CONFIG = "C"
SMOKETEST_HA_START = "-2.000h"
SMOKETEST_HA_STOP = "+2.000h"
SMOKETEST_INTEGRATION = "60s"
SMOKETEST_NOISE = "0.003Jy"   # 3 mJy simplenoise
SMOKETEST_IMSIZE = 512
SMOKETEST_FIELD_TYPE = None    # sampled from balance

# Output directory alongside the MS and FITS files
WORK_DIR = str(_REPO_ROOT / "smoketest_output")

# Triptych output path (in data/morphology/lib per spec)
TRIPTYCH_DIR = _REPO_ROOT / "data" / "morphology" / "lib"
TRIPTYCH_PATH = str(TRIPTYCH_DIR / "smoketest_dirty_model_psf.png")


def _build_spec() -> CorpusFieldSpec:
    """Build the fixed smoke-test CorpusFieldSpec."""
    cell_arcsec = _cell_arcsec_for_config(SMOKETEST_VLA_CONFIG)
    field_size_arcsec = SMOKETEST_IMSIZE * cell_arcsec

    return CorpusFieldSpec(
        field_idx=0,
        field_seed=SMOKETEST_SEED,
        vla_config=SMOKETEST_VLA_CONFIG,
        cell_arcsec=cell_arcsec,
        imsize=SMOKETEST_IMSIZE,
        field_size_arcsec=field_size_arcsec,
        ha_start=SMOKETEST_HA_START,
        ha_stop=SMOKETEST_HA_STOP,
        integration_time=SMOKETEST_INTEGRATION,
        noise_jy=SMOKETEST_NOISE,
        field_type=SMOKETEST_FIELD_TYPE,
    )


def _asinh_stretch(arr: np.ndarray, a: float = 0.01) -> np.ndarray:
    """Asinh stretch: np.arcsinh(arr / a) / np.arcsinh(1.0 / a), normalised to [0,1].

    Preserves sign for signed arrays (used for dirty).
    For unsigned arrays (model) clip to [0, inf] first.
    """
    if arr.max() <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    peak = float(np.abs(arr).max())
    if peak == 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    normed = arr / peak
    stretched = np.arcsinh(normed / a)
    scale = float(np.arcsinh(1.0 / a))
    if scale == 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return (stretched / scale).astype(np.float32)


def _log_stretch(arr: np.ndarray, floor_frac: float = 1e-4) -> np.ndarray:
    """Log stretch: clip to [floor, max], log-normalise to [0, 1]."""
    arr = np.clip(arr, 0.0, None)
    peak = float(arr.max())
    if peak == 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    floor = peak * floor_frac
    arr = np.clip(arr, floor, peak)
    stretched = np.log10(arr / floor)
    log_range = np.log10(peak / floor)
    if log_range == 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return (stretched / log_range).astype(np.float32)


def _read_fits_stokes_i_chan0(fits_path: str) -> np.ndarray:
    """Read Stokes I, channel 0 from a FITS file.  Returns (ny, nx) float64 array."""
    from astropy.io import fits as pyfits

    with pyfits.open(fits_path) as hdul:
        data = hdul[0].data  # may be (nstokes, nchan, ny, nx) or (ny, nx) after dropdeg
        data = np.squeeze(data)   # remove degenerate axes
        # After squeeze: could be (ny, nx) or (nstokes, ny, nx) or (nchan, ny, nx)
        if data.ndim == 2:
            return data.astype(np.float64)
        if data.ndim == 3:
            return data[0].astype(np.float64)   # first plane (Stokes I or channel 0)
        if data.ndim == 4:
            return data[0, 0].astype(np.float64)   # stokes=0, chan=0
        raise ValueError(f"Unexpected FITS data shape after squeeze: {data.shape}")


def render_triptych(dirty_fits: str, model_fits: str, psf_fits: str, out_path: str) -> None:
    """Render a (model | dirty | psf) triptych PNG.

    - model: asinh stretch, viridis, non-negative
    - dirty: SIGNED diverging colormap (RdBu_r) centred at zero, asinh stretch
    - psf:   log stretch, viridis
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dirty = _read_fits_stokes_i_chan0(dirty_fits)
    model = _read_fits_stokes_i_chan0(model_fits)
    psf = _read_fits_stokes_i_chan0(psf_fits)

    log.info("model: shape=%s min=%.4g max=%.4g", model.shape, model.min(), model.max())
    log.info("dirty: shape=%s min=%.4g max=%.4g", dirty.shape, dirty.min(), dirty.max())
    log.info("psf:   shape=%s min=%.4g max=%.4g", psf.shape, psf.min(), psf.max())

    # Diagnostics
    dirty_has_negatives = bool(dirty.min() < 0.0)
    psf_peak_near_one = abs(psf.max() - 1.0) < 0.1
    log.info("dirty has negatives: %s  (min=%.4g)", dirty_has_negatives, dirty.min())
    log.info("PSF peak ≈ 1:        %s  (max=%.4g)", psf_peak_near_one, psf.max())

    # Stretch
    # Model has a large dynamic range (compact peak >> faint diffuse).  Tie the
    # asinh knee to a low percentile of the positive pixels so the faint
    # extended structure is visible rather than crushed into the linear regime.
    model_pos = model[model > 0.0]
    if model_pos.size:
        knee = float(np.percentile(model_pos, 50.0)) / float(model_pos.max())
        a_model = float(np.clip(knee, 1e-4, 1e-2))
    else:
        a_model = 0.01
    model_s = _asinh_stretch(np.clip(model, 0.0, None), a=a_model)
    dirty_s = _asinh_stretch(dirty)   # signed asinh, preserves sign
    psf_s = _log_stretch(psf)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax_model, ax_dirty, ax_psf = axes

    im0 = ax_model.imshow(model_s, origin="lower", cmap="viridis",
                          vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_model.set_title(f"Model (asinh)\nmax={model.max():.3g} Jy/px", fontsize=11)
    ax_model.axis("off")
    plt.colorbar(im0, ax=ax_model, fraction=0.046, pad=0.04, label="asinh stretch")

    # Dirty: symmetric diverging colormap; asinh-stretched value is signed [-1,1]
    im1 = ax_dirty.imshow(dirty_s, origin="lower", cmap="RdBu_r",
                          vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax_dirty.set_title(
        f"Dirty (signed asinh, RdBu_r)\n"
        f"min={dirty.min():.3g}  max={dirty.max():.3g} Jy/bm",
        fontsize=11
    )
    ax_dirty.axis("off")
    plt.colorbar(im1, ax=ax_dirty, fraction=0.046, pad=0.04, label="asinh stretch (signed)")

    im2 = ax_psf.imshow(psf_s, origin="lower", cmap="viridis",
                        vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_psf.set_title(f"PSF (log)\npeak={psf.max():.4g}", fontsize=11)
    ax_psf.axis("off")
    plt.colorbar(im2, ax=ax_psf, fraction=0.046, pad=0.04, label="log stretch")

    fig.suptitle(
        f"M3 smoke test — VLA {SMOKETEST_VLA_CONFIG}-config L-band "
        f"HA {SMOKETEST_HA_START}..{SMOKETEST_HA_STOP}  noise={SMOKETEST_NOISE}",
        fontsize=13
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Triptych saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("M3 corpus smoke test")
    log.info("  VLA config : %s", SMOKETEST_VLA_CONFIG)
    log.info("  HA track   : %s .. %s  (%s integration)", SMOKETEST_HA_START, SMOKETEST_HA_STOP, SMOKETEST_INTEGRATION)
    log.info("  Noise      : %s", SMOKETEST_NOISE)
    log.info("  Work dir   : %s", WORK_DIR)
    log.info("  Triptych   : %s", TRIPTYCH_PATH)
    log.info("=" * 60)

    spec = _build_spec()
    log.info("CorpusFieldSpec: cell=%.4g arcsec  field_size=%.1f arcsec",
             spec.cell_arcsec, spec.field_size_arcsec)

    t_start = time.perf_counter()
    result = build_corpus_field(spec, work_dir=WORK_DIR, name="smoketest")
    t_total = time.perf_counter() - t_start

    if not result.success:
        log.error("CASA run FAILED: %s", result.error)
        sys.exit(1)

    log.info("CASA run complete in %.1f s", t_total)
    log.info("  dirty : %s", result.dirty_fits)
    log.info("  psf   : %s", result.psf_fits)
    log.info("  model : %s", result.model_fits)

    if result.dirty_fits is None or result.psf_fits is None or result.model_fits is None:
        log.error("One or more output FITS files missing -- cannot render triptych")
        sys.exit(1)

    # Basic verification
    import numpy as np
    from astropy.io import fits as pyfits

    def _read(path):
        with pyfits.open(path) as h:
            return np.squeeze(h[0].data).astype(np.float64)

    dirty = _read(result.dirty_fits)
    psf = _read(result.psf_fits)

    dirty_has_neg = bool(dirty.min() < 0.0)
    psf_peak = float(np.squeeze(psf).max())

    log.info("Verification:")
    log.info("  dirty.min = %.4g  (has negatives: %s)", dirty.min(), dirty_has_neg)
    log.info("  dirty.max = %.4g", dirty.max())
    log.info("  psf.max   = %.4g  (should be ≈ 1.0)", psf_peak)

    if not dirty_has_neg:
        log.warning("WARNING: dirty image has no negative values -- sidelobes may be absent")
    if abs(psf_peak - 1.0) > 0.1:
        log.warning("WARNING: PSF peak = %.4g (expected ≈ 1.0)", psf_peak)

    # Render triptych
    render_triptych(result.dirty_fits, result.model_fits, result.psf_fits, TRIPTYCH_PATH)

    log.info("=" * 60)
    log.info("Smoke test PASSED")
    log.info("  dirty min<0 : %s", dirty_has_neg)
    log.info("  psf peak≈1  : %s  (%.4g)", abs(psf_peak - 1.0) < 0.1, psf_peak)
    log.info("  total time  : %.1f s", t_total)
    log.info("  triptych    : %s", TRIPTYCH_PATH)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
