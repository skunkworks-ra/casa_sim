"""
patchify.py -- M4: Patchify + memmap writer for the CASA corpus (M4).

Dices each 512px field's dirty + model into 128x128 non-overlapping patches
(stride=128 → 16 patches per field), crops the field PSF to 128x128, and
writes torch-ready numpy memmap stacks plus a manifest JSON to disk.

Public API
----------
patchify_results(results, out_dir, patch_size=128) -> PatchifyStats
    Patchify a list of CorpusFieldResult objects and write memmap stacks.

PatchifyStats
    Dataclass with counts, shapes, and paths for the written stacks.

FITS → 2D conversion
---------------------
FITS layout from CASA export: (FREQ, STOKES, Y, X) i.e. data[freq, stokes, y, x].
We take Stokes I (stokes index 0) and average across all frequency channels (MFS
collapse), producing a single (Y, X) float32 plane per field.

Guardrails
----------
- dirty: SIGNED and UNCLIPPED (negatives preserved).
- sky: clipped to >= 0 (model image has near-zero floating point negatives).
- psf: peak-normalised to 1.0 (PSF is already normalised by CASA but we enforce it).
- cond (normalised stats): NOT stored here (MAD-Clean derives from dirty patch).

Stack files
-----------
Written with np.save (not memmap write); loaded by MAD-Clean with
    np.load(path, mmap_mode='r')

    dirty.npy    -- (N, patch_size, patch_size) float32   signed, unclipped
    sky.npy      -- (N, patch_size, patch_size) float32   non-negative
    psf.npy      -- (F, patch_size, patch_size) float32   peak-normalised per field
    field_id.npy -- (N,) int32                             field index per patch
    config_idx.npy -- (F,) int32                           VLA config index per field

    manifest.json -- counts, config + cell per field, stride, patch_size

Where N = F * patches_per_field and F = number of successful fields.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, TYPE_CHECKING

import numpy as np
from astropy.io import fits

if TYPE_CHECKING:
    from .corpus import CorpusFieldResult

log = logging.getLogger(__name__)

# VLA config labels → integer index (alphabetic order, stable)
_VLA_CONFIG_ORDER = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# PSF crop/centre helper
# ---------------------------------------------------------------------------

def _centre_crop_psf(psf_2d: np.ndarray, patch_size: int) -> np.ndarray:
    """Crop or pad a 2D PSF to (patch_size, patch_size) centred on the peak.

    The PSF peak may not be exactly at the image centre (rounding); we find
    the peak pixel and extract a centred cutout. If the PSF is smaller than
    patch_size in any dimension, it is zero-padded.

    Returns
    -------
    float32 array of shape (patch_size, patch_size), peak-normalised to 1.0.
    """
    psf = psf_2d.astype(np.float32)

    # Find peak
    peak_val = psf.max()
    if peak_val <= 0.0:
        log.warning("[patchify] PSF peak <= 0 (%.4g); returning zeros", peak_val)
        return np.zeros((patch_size, patch_size), dtype=np.float32)

    peak_idx = np.unravel_index(psf.argmax(), psf.shape)
    cy, cx = int(peak_idx[0]), int(peak_idx[1])

    half = patch_size // 2
    ny, nx = psf.shape

    # Source extraction window in PSF coords (may extend outside)
    y0_src = cy - half
    y1_src = cy - half + patch_size
    x0_src = cx - half
    x1_src = cx - half + patch_size

    # Destination window in output coords
    y0_dst = max(0, -y0_src)
    y1_dst = y0_dst + (min(y1_src, ny) - max(y0_src, 0))
    x0_dst = max(0, -x0_src)
    x1_dst = x0_dst + (min(x1_src, nx) - max(x0_src, 0))

    # Clamp source to valid range
    y0_src_c = max(y0_src, 0)
    y1_src_c = min(y1_src, ny)
    x0_src_c = max(x0_src, 0)
    x1_src_c = min(x1_src, nx)

    out = np.zeros((patch_size, patch_size), dtype=np.float32)
    out[y0_dst:y1_dst, x0_dst:x1_dst] = psf[y0_src_c:y1_src_c, x0_src_c:x1_src_c]

    # Peak-normalise
    out /= peak_val
    return out


def _centre_crop(img_2d: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Centre-crop a 2D array to (target_h, target_w).

    The dirty/PSF are imaged on a guard-band grid larger than the model (source
    field) so aliased sidelobes do not wrap.  For training the dirty must align
    pixel-for-pixel with the model, so the central source field is cropped out.
    """
    h, w = img_2d.shape
    if (h, w) == (target_h, target_w):
        return img_2d
    if h < target_h or w < target_w:
        raise ValueError(
            f"_centre_crop: image {(h, w)} smaller than target {(target_h, target_w)}"
        )
    y0 = (h - target_h) // 2
    x0 = (w - target_w) // 2
    return img_2d[y0:y0 + target_h, x0:x0 + target_w].copy()


# ---------------------------------------------------------------------------
# FITS → 2D collapse (Stokes I, MFS average)
# ---------------------------------------------------------------------------

def _fits_to_2d(fits_path: str) -> np.ndarray:
    """Load a FITS image and collapse to 2D float32 (Stokes I, freq-averaged).

    CASA exports: (FREQ, STOKES, Y, X) with FITS convention.
    astropy.io.fits loads the array in Fortran/FITS order, so the Python
    array is (NAXIS4, NAXIS3, NAXIS2, NAXIS1) = (FREQ, STOKES, Y, X).
    We squeeze out degenerate dimensions and then take:
        - Stokes I: axis index where CTYPE3=STOKES and Stokes=I (index 0 after
          standard CASA export with dropdeg=True having been called with stokeslast=False).
        - Mean over frequency axis.

    Handles both 4D (FREQ, STOKES, Y, X) and 2D (Y, X) layouts.
    """
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header

    # 4D case: (FREQ, STOKES, Y, X)
    if data.ndim == 4:
        # Mean over frequency (axis 0), Stokes I (axis 1, index 0)
        plane = data.mean(axis=0)[0]   # (Y, X)
    elif data.ndim == 3:
        # (STOKES, Y, X) or (FREQ, Y, X) — take index 0
        plane = data[0]
    elif data.ndim == 2:
        plane = data
    else:
        raise ValueError(f"Unexpected FITS ndim={data.ndim} in {fits_path}")

    return plane.astype(np.float32)


# ---------------------------------------------------------------------------
# Grid patchify (non-overlapping, stride=patch_size)
# ---------------------------------------------------------------------------

def _extract_patches(img_2d: np.ndarray, patch_size: int) -> np.ndarray:
    """Extract non-overlapping patches in row-major grid order.

    Only complete patches are returned (trailing edge cropped).  For a
    512px image with patch_size=128: 4x4 = 16 patches.

    Returns
    -------
    (n_patches, patch_size, patch_size) float32
    """
    h, w = img_2d.shape
    patches = []
    y = 0
    while y + patch_size <= h:
        x = 0
        while x + patch_size <= w:
            patches.append(img_2d[y:y + patch_size, x:x + patch_size].copy())
            x += patch_size
        y += patch_size
    return np.stack(patches, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# PatchifyStats
# ---------------------------------------------------------------------------

@dataclass
class PatchifyStats:
    """Summary of a patchify_results() run."""
    n_fields: int = 0
    n_patches: int = 0
    patch_size: int = 128
    patches_per_field: int = 0

    out_dir: str = ""
    dirty_path: str = ""
    sky_path: str = ""
    psf_path: str = ""
    field_id_path: str = ""
    config_idx_path: str = ""
    manifest_path: str = ""

    # Per-field info for manifest
    field_entries: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def patchify_results(
    results: "Sequence[CorpusFieldResult]",
    out_dir: str,
    *,
    patch_size: int = 128,
) -> PatchifyStats:
    """Patchify a list of CorpusFieldResult objects and write memmap stacks.

    Only successful results with all three FITS files present are processed.
    Skips fields where any FITS file is missing with a warning.

    Parameters
    ----------
    results:
        List of CorpusFieldResult from run_corpus_batch or build_corpus_field.
    out_dir:
        Directory to write stacks and manifest into (created if absent).
    patch_size:
        Patch edge length in pixels (default 128).  Must divide imsize evenly.

    Returns
    -------
    PatchifyStats with counts, shapes, and paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_dir = os.path.abspath(out_dir)

    # Filter to successful fields with all FITS present
    valid = [
        r for r in results
        if r.success
        and r.dirty_fits and os.path.exists(r.dirty_fits)
        and r.psf_fits and os.path.exists(r.psf_fits)
        and r.model_fits and os.path.exists(r.model_fits)
    ]
    if not valid:
        raise ValueError("No valid (success + all FITS present) CorpusFieldResult entries.")

    # Determine patches per field from first field
    # Grid is set by the MODEL (source-field) size, not the dirty: the dirty/PSF
    # may be imaged on a larger guard-band grid and are centre-cropped to the
    # model size so patches stay aligned with the sky.
    first_sky = _fits_to_2d(valid[0].model_fits)
    h, w = first_sky.shape
    n_per_row = w // patch_size
    n_per_col = h // patch_size
    patches_per_field = n_per_row * n_per_col
    log.info(
        "[patchify] model %dx%d → %dx%d grid = %d patches/field",
        w, h, n_per_row, n_per_col, patches_per_field,
    )

    n_fields = len(valid)
    n_patches = n_fields * patches_per_field

    # Allocate in-memory arrays
    all_dirty = np.empty((n_patches, patch_size, patch_size), dtype=np.float32)
    all_sky = np.empty((n_patches, patch_size, patch_size), dtype=np.float32)
    all_psf = np.empty((n_fields, patch_size, patch_size), dtype=np.float32)
    all_field_id = np.empty(n_patches, dtype=np.int32)
    all_config_idx = np.empty(n_fields, dtype=np.int32)

    field_entries = []

    for f_idx, result in enumerate(valid):
        p_start = f_idx * patches_per_field
        p_end = p_start + patches_per_field

        log.info(
            "[patchify] field %d/%d  idx=%d  config=%s",
            f_idx + 1, n_fields, result.field_idx, result.spec.vla_config,
        )

        # Load and collapse to 2D
        sky_2d = _fits_to_2d(result.model_fits)            # may have tiny negatives
        # dirty is imaged on a guard-band grid; crop its central source field to
        # align with the model before patchifying.
        dirty_2d = _centre_crop(_fits_to_2d(result.dirty_fits), *sky_2d.shape)
        psf_2d = _fits_to_2d(result.psf_fits)              # signed (sidelobes)

        # Guardrails
        sky_2d = np.clip(sky_2d, 0.0, None)               # sky must be non-negative
        # dirty: no clipping -- negatives preserved
        # psf: centre-crop, peak-normalise
        psf_crop = _centre_crop_psf(psf_2d, patch_size)

        # Extract patches
        dirty_patches = _extract_patches(dirty_2d, patch_size)   # (P, H, W)
        sky_patches = _extract_patches(sky_2d, patch_size)       # (P, H, W)

        if dirty_patches.shape[0] != patches_per_field:
            raise RuntimeError(
                f"Field {result.field_idx}: expected {patches_per_field} patches, "
                f"got {dirty_patches.shape[0]}.  Image shape: {dirty_2d.shape}"
            )

        all_dirty[p_start:p_end] = dirty_patches
        all_sky[p_start:p_end] = sky_patches
        all_psf[f_idx] = psf_crop
        all_field_id[p_start:p_end] = result.field_idx
        all_config_idx[f_idx] = (
            _VLA_CONFIG_ORDER.index(result.spec.vla_config.upper())
            if result.spec.vla_config.upper() in _VLA_CONFIG_ORDER
            else -1
        )

        field_entries.append({
            "field_idx": result.field_idx,
            "vla_config": result.spec.vla_config,
            "config_idx": int(all_config_idx[f_idx]),
            "cell_arcsec": result.spec.cell_arcsec,
            "imsize": result.spec.imsize,
            "n_patches": patches_per_field,
            "patch_start": p_start,
            "patch_end": p_end,
        })

    # Write stacks
    dirty_path = os.path.join(out_dir, "dirty.npy")
    sky_path = os.path.join(out_dir, "sky.npy")
    psf_path = os.path.join(out_dir, "psf.npy")
    field_id_path = os.path.join(out_dir, "field_id.npy")
    config_idx_path = os.path.join(out_dir, "config_idx.npy")
    manifest_path = os.path.join(out_dir, "manifest.json")

    np.save(dirty_path, all_dirty)
    np.save(sky_path, all_sky)
    np.save(psf_path, all_psf)
    np.save(field_id_path, all_field_id)
    np.save(config_idx_path, all_config_idx)

    manifest = {
        "n_fields": n_fields,
        "n_patches": n_patches,
        "patches_per_field": patches_per_field,
        "patch_size": patch_size,
        "stride": patch_size,
        "fields": field_entries,
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    log.info(
        "[patchify] wrote %d patches from %d fields to %s",
        n_patches, n_fields, out_dir,
    )

    stats = PatchifyStats(
        n_fields=n_fields,
        n_patches=n_patches,
        patch_size=patch_size,
        patches_per_field=patches_per_field,
        out_dir=out_dir,
        dirty_path=dirty_path,
        sky_path=sky_path,
        psf_path=psf_path,
        field_id_path=field_id_path,
        config_idx_path=config_idx_path,
        manifest_path=manifest_path,
        field_entries=field_entries,
    )
    return stats
