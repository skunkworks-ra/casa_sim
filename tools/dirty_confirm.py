"""
tools/dirty_confirm.py -- End-to-end dirty-image confirmation: compact + diffuse.

Runs ONE fully-populated, flux-balanced DIFFUSE_DOMINANT corpus field through
CASA and renders a 3-panel dirty-confirm PNG.

Fixed parameters:
  - VLA C-config, L-band
  - HA track: -2h .. +2h, 60s integration (modest track, fast run)
  - simplenoise: 3 mJy (a few mJy; enough to show thermal noise, not bury signal)
  - flux_floor: 1e-5 Jy (10 uJy, full AGN+SFG T-RECS distribution; ~1794 sources)
  - field_type: DIFFUSE_DOMINANT (fixed -- confirms both compact and diffuse survive)
  - seed: 7  (different from smoketest seed=42 to get a fresh draw)

Output:
  casa_sim/data/morphology/lib/dirty_confirm.png

3-panel layout:
  [model (log stretch)]  |  [dirty (signed diverging, RdBu_r, no clip)]  |  [psf (log)]

Dirty guardrail: negatives MUST be present (PSF sidelobes).  If dirty.min() >= 0,
the script warns loudly -- that means the dirty was accidentally clipped somewhere.

Usage (from repo root):
    pixi run python tools/dirty_confirm.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dirty_confirm")

# ---------------------------------------------------------------------------
# Repo root
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
# Fixed parameters
# ---------------------------------------------------------------------------

CONFIRM_SEED = 7                  # distinct from smoketest seed=42
CONFIRM_VLA_CONFIG = "C"          # C-config at L-band gives ~2.44 arcsec/px
CONFIRM_HA_START = "-2.000h"
CONFIRM_HA_STOP = "+2.000h"
CONFIRM_INTEGRATION = "60s"
CONFIRM_NOISE = "0.003Jy"         # 3 mJy simplenoise
CONFIRM_IMSIZE = 512
CONFIRM_FIELD_TYPE = "diffuse_dominant"   # fixed -- this is the whole point

WORK_DIR = str(_REPO_ROOT / "smoketest_output")

CONFIRM_PNG_DIR = _REPO_ROOT / "data" / "morphology" / "lib"
CONFIRM_PNG_PATH = str(CONFIRM_PNG_DIR / "dirty_confirm.png")


# ---------------------------------------------------------------------------
# Spec builder
# ---------------------------------------------------------------------------

def _build_spec() -> CorpusFieldSpec:
    cell_arcsec = _cell_arcsec_for_config(CONFIRM_VLA_CONFIG)
    field_size_arcsec = CONFIRM_IMSIZE * cell_arcsec
    return CorpusFieldSpec(
        field_idx=0,
        field_seed=CONFIRM_SEED,
        vla_config=CONFIRM_VLA_CONFIG,
        cell_arcsec=cell_arcsec,
        imsize=CONFIRM_IMSIZE,
        field_size_arcsec=field_size_arcsec,
        ha_start=CONFIRM_HA_START,
        ha_stop=CONFIRM_HA_STOP,
        integration_time=CONFIRM_INTEGRATION,
        noise_jy=CONFIRM_NOISE,
        field_type=CONFIRM_FIELD_TYPE,
    )


# ---------------------------------------------------------------------------
# FITS reader
# ---------------------------------------------------------------------------

def _read_fits_2d(fits_path: str) -> np.ndarray:
    """Read first Stokes / channel plane from a FITS file.  Returns (ny, nx) float64."""
    from astropy.io import fits as pyfits
    with pyfits.open(fits_path) as hdul:
        data = hdul[0].data
        data = np.squeeze(data)
        if data.ndim == 2:
            return data.astype(np.float64)
        if data.ndim == 3:
            return data[0].astype(np.float64)
        if data.ndim == 4:
            return data[0, 0].astype(np.float64)
        raise ValueError(f"Unexpected FITS shape after squeeze: {data.shape}")


# ---------------------------------------------------------------------------
# Stretch utilities
# ---------------------------------------------------------------------------

def _log_stretch(arr: np.ndarray, floor_frac: float = 1e-4) -> np.ndarray:
    """Log stretch: clip to [floor, max], normalise to [0, 1]."""
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


def _asinh_stretch_signed(arr: np.ndarray, a: float = 0.01) -> np.ndarray:
    """Signed asinh stretch: maps arr to [-1, 1] via arcsinh(arr/peak/a).

    Preserves sign so PSF sidelobe negatives show in the diverging colormap.
    DOES NOT clip -- negatives survive.
    """
    peak = float(np.abs(arr).max())
    if peak == 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    normed = arr / peak
    stretched = np.arcsinh(normed / a)
    scale = float(np.arcsinh(1.0 / a))
    if scale == 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return (stretched / scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_confirm(
    dirty_fits: str,
    model_fits: str,
    psf_fits: str,
    out_path: str,
    n_sources: int,
    elapsed_s: float,
    noise_str: str,
) -> None:
    """Render 3-panel dirty_confirm.png.

    Panels: model (log, viridis) | dirty (signed asinh, RdBu_r, no clip) | psf (log, viridis)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = _read_fits_2d(model_fits)
    dirty = _read_fits_2d(dirty_fits)
    psf = _read_fits_2d(psf_fits)

    log.info("model: shape=%s  min=%.4g  max=%.4g Jy/px", model.shape, model.min(), model.max())
    log.info("dirty: shape=%s  min=%.4g  max=%.4g Jy/bm", dirty.shape, dirty.min(), dirty.max())
    log.info("psf:   shape=%s  min=%.4g  max=%.4g", psf.shape, psf.min(), psf.max())

    dirty_has_neg = bool(dirty.min() < 0.0)
    psf_peak = float(psf.max())

    if not dirty_has_neg:
        log.warning("GUARDRAIL FAIL: dirty.min()=%.4g >= 0 -- no PSF sidelobe negatives!", dirty.min())
    else:
        log.info("GUARDRAIL PASS: dirty has negatives (min=%.4g) -- PSF sidelobes present", dirty.min())

    if abs(psf_peak - 1.0) > 0.1:
        log.warning("PSF peak=%.4g (expected ~1.0)", psf_peak)
    else:
        log.info("PSF peak=%.4g (OK)", psf_peak)

    # Stretches
    model_s = _log_stretch(model)
    dirty_s = _asinh_stretch_signed(dirty)     # signed -- NO clip
    psf_s = _log_stretch(psf)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    ax_model, ax_dirty, ax_psf = axes

    # Panel 1: model (log stretch)
    im0 = ax_model.imshow(model_s, origin="lower", cmap="viridis",
                          vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_model.set_title(
        f"Model (log stretch)\n"
        f"max={model.max():.3g} Jy/px  n_src~{n_sources}",
        fontsize=11
    )
    ax_model.axis("off")
    plt.colorbar(im0, ax=ax_model, fraction=0.046, pad=0.04, label="log stretch [0,1]")

    # Panel 2: dirty (signed diverging, NO clip)
    im1 = ax_dirty.imshow(dirty_s, origin="lower", cmap="RdBu_r",
                          vmin=-1.0, vmax=1.0, interpolation="nearest")
    neg_flag = "YES (OK)" if dirty_has_neg else "NO (PROBLEM)"
    ax_dirty.set_title(
        f"Dirty (signed asinh, RdBu_r -- NO clip)\n"
        f"min={dirty.min():.3g}  max={dirty.max():.3g} Jy/bm  negatives: {neg_flag}",
        fontsize=11
    )
    ax_dirty.axis("off")
    plt.colorbar(im1, ax=ax_dirty, fraction=0.046, pad=0.04, label="signed asinh [-1,1]")

    # Panel 3: psf (log stretch)
    im2 = ax_psf.imshow(psf_s, origin="lower", cmap="viridis",
                        vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_psf.set_title(
        f"PSF (log stretch)\npeak={psf_peak:.4g}",
        fontsize=11
    )
    ax_psf.axis("off")
    plt.colorbar(im2, ax=ax_psf, fraction=0.046, pad=0.04, label="log stretch [0,1]")

    fig.suptitle(
        f"Dirty confirm -- VLA C-config L-band  HA -2h..+2h  60s int  noise={noise_str}\n"
        f"field_type=DIFFUSE_DOMINANT  seed={CONFIRM_SEED}  "
        f"flux_floor=10uJy (AGN+SFG)  CASA time={elapsed_s:.0f}s",
        fontsize=12
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Confirm PNG saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("dirty_confirm.py -- end-to-end dirty image confirmation")
    log.info("  VLA config  : %s", CONFIRM_VLA_CONFIG)
    log.info("  HA track    : %s .. %s  (%s integration)",
             CONFIRM_HA_START, CONFIRM_HA_STOP, CONFIRM_INTEGRATION)
    log.info("  Noise       : %s", CONFIRM_NOISE)
    log.info("  Field type  : %s (FIXED = DIFFUSE_DOMINANT)", CONFIRM_FIELD_TYPE)
    log.info("  Seed        : %d", CONFIRM_SEED)
    log.info("  Work dir    : %s", WORK_DIR)
    log.info("  Output PNG  : %s", CONFIRM_PNG_PATH)
    log.info("=" * 60)

    spec = _build_spec()
    log.info("CorpusFieldSpec: cell=%.4g arcsec  field_size=%.1f arcsec",
             spec.cell_arcsec, spec.field_size_arcsec)

    t_start = time.perf_counter()
    result = build_corpus_field(spec, work_dir=WORK_DIR, name="dirty_confirm")
    elapsed_s = time.perf_counter() - t_start

    if not result.success:
        log.error("CASA run FAILED after %.1f s: %s", elapsed_s, result.error)
        sys.exit(1)

    log.info("CASA run complete in %.1f s", elapsed_s)
    log.info("  dirty : %s", result.dirty_fits)
    log.info("  psf   : %s", result.psf_fits)
    log.info("  model : %s", result.model_fits)

    if result.dirty_fits is None or result.psf_fits is None or result.model_fits is None:
        log.error("One or more FITS outputs missing -- cannot render")
        sys.exit(1)

    # Count sources from the model FITS (count non-zero pixels as a proxy)
    from astropy.io import fits as pyfits
    with pyfits.open(result.model_fits) as h:
        model_data = np.squeeze(h[0].data).astype(np.float32)
    n_nonzero = int(np.count_nonzero(model_data > 0))
    log.info("Model nonzero pixels: %d (proxy for source count + extended flux)", n_nonzero)

    # Dirty guardrail
    with pyfits.open(result.dirty_fits) as h:
        dirty_data = np.squeeze(h[0].data).astype(np.float64)
    dirty_min = float(dirty_data.min())
    dirty_max = float(dirty_data.max())
    dirty_has_neg = dirty_min < 0.0

    log.info("=" * 60)
    log.info("VERIFICATION:")
    log.info("  CASA elapsed  : %.1f s", elapsed_s)
    log.info("  dirty.min     : %.4g  (negatives: %s)", dirty_min,
             "YES -- PSF sidelobes present" if dirty_has_neg else "NO -- PROBLEM")
    log.info("  dirty.max     : %.4g", dirty_max)
    if not dirty_has_neg:
        log.warning("  GUARDRAIL: dirty has NO negatives -- clipping may have occurred!")
    log.info("=" * 60)

    # Render
    render_confirm(
        dirty_fits=result.dirty_fits,
        model_fits=result.model_fits,
        psf_fits=result.psf_fits,
        out_path=CONFIRM_PNG_PATH,
        n_sources=n_nonzero,
        elapsed_s=elapsed_s,
        noise_str=CONFIRM_NOISE,
    )

    log.info("=" * 60)
    log.info("dirty_confirm COMPLETE")
    log.info("  CASA time    : %.1f s", elapsed_s)
    log.info("  dirty min<0  : %s  (%.4g)", dirty_has_neg, dirty_min)
    log.info("  PNG          : %s", CONFIRM_PNG_PATH)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
