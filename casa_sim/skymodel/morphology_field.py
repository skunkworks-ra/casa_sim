"""casa_sim.skymodel.morphology_field
=====================================
Extended-source sky-model builder for the CASA-simulated corpus.

Builds the **extended-source** component of a 512 x 512 field image in
Jy/pixel.  T-RECS compact sources are intentionally NOT rendered here --
they are composited on the casa_sim side via the existing ``t_recs`` mode.
casa_sim's responsibility is a clean FITS sky model with correct WCS; the
FITS -> CASA image conversion (importfits) and compact compositing live in
the casa_sim simulation pipeline.

Coordination invariant
-----------------------
The four scalar parameters ``ra_deg``, ``dec_deg``, ``cell_arcsec``, and
``imsize`` are a **shared coordinate contract** between this module and the
casa_sim per-field SimConfig.  They must match exactly; any mismatch will
produce a pixel-grid misalignment in the composited model image.

FITS format for casa_sim ingestion
------------------------------------
``image_native`` mode in casa_sim passes the image path directly to
``sm.predict(imagename=...)`` which requires a **CASA ``.im`` image**, not
a FITS file.  The FITS produced here is therefore an intermediate product;
the casa_sim batch driver calls ``importfits`` to convert it before
running the simulation.

Required FITS header keywords (consumed by importfits / CASA coordsys):
  - NAXIS=4, NAXIS1=imsize, NAXIS2=imsize, NAXIS3=1 (Stokes), NAXIS4=1 (freq)
  - CTYPE1='RA---SIN', CDELT1=-cell_deg, CRPIX1=imsize/2+1, CRVAL1=ra_deg
  - CTYPE2='DEC--SIN', CDELT2=+cell_deg, CRPIX2=imsize/2+1, CRVAL2=dec_deg
  - CTYPE3='STOKES',   CDELT3=1,         CRPIX3=1,           CRVAL3=1 (I)
  - CTYPE4='FREQ',     CDELT4=0,         CRPIX4=1,           CRVAL4=freq_hz
  - BUNIT='Jy/pixel'
  - EQUINOX=2000.0

All values are float64 in the header; the data array is float32 broadcast
to shape (1, 1, imsize, imsize) matching the NAXIS4/3/2/1 order (FITS
axis ordering is reversed vs NumPy).

Field taxonomy
--------------
Fields are classified by a ``FieldType`` enum.  Use ``build_field()`` as the
primary entry point; it dispatches per type.  The balance is controlled by
``DEFAULT_FIELD_TYPE_BALANCE`` (a module-level constant -- override it or
pass ``balance=`` to ``sample_field_type``).

Four types:

  POINT_ONLY          -- All-zero extended image; casa_sim adds T-RECS points.
  DIFFUSE_DOMINANT    -- Several large finite apodized sources from web/filament
                         kinds (diffuse fills much of the field).
  CENTRAL_SHELL_FLOW  -- One central object from the shock kind near field center,
                         finite apodized.
  FULLY_DIFFUSE       -- Faint full-field diffuse background from web/filament
                         templates resampled to fill the whole field; no
                         internal seams (tiled/mirrored to exceed field size).

Extended kinds used:
  "shock"    -- CENTRAL_SHELL_FLOW only
  "web"      -- DIFFUSE_DOMINANT and FULLY_DIFFUSE
  "filament" -- DIFFUSE_DOMINANT and FULLY_DIFFUSE

Compact kind dropped:
  "compact" (box_orange) is NOT used in any extended field type.  Points
  come exclusively from T-RECS on the casa_sim side.

Apodization:
  Each finite placed template is multiplied by a separable Hann window before
  compositing.  This fades the template to zero at its footprint edges,
  eliminating visible square tile boundaries.  The FULLY_DIFFUSE background
  layer is a single large seamless tile and is NOT apodized (no footprint edge).

Compute: numpy-native (no torch)
----------------------------------
All field-generation math runs on numpy arrays.  Resampling uses
``scipy.ndimage.zoom`` (order=1, bilinear equivalent).  This is intentional:
casa_sim is a numpy/CASA package; torch lives only in MAD-Clean's data loader.

Public API
----------
``FieldType``   -- enum of four field types.

``DEFAULT_FIELD_TYPE_BALANCE`` -- dict mapping FieldType -> relative weight.
  Single overridable constant.  Normalised internally at sample time.
  Default: {POINT_ONLY: 0.25, DIFFUSE_DOMINANT: 0.32,
            CENTRAL_SHELL_FLOW: 0.33, FULLY_DIFFUSE: 0.10}.

``sample_field_type(rng, balance=DEFAULT_FIELD_TYPE_BALANCE) -> FieldType``

``build_field(rng, field_type, *, ra_deg, dec_deg, cell_arcsec, imsize,
              freq_hz, ..., repo_root) -> (image_np, meta)``
  Returns the extended Jy/pixel image as a float32 numpy array.

``write_model_fits(image, path, *, ra_deg, dec_deg, cell_arcsec, freq_hz)``
  Accepts a numpy array.
"""

from __future__ import annotations

import enum
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.ndimage import zoom as _ndimage_zoom

__all__ = [
    "FieldType",
    "DEFAULT_FIELD_TYPE_BALANCE",
    "sample_field_type",
    "build_field",
    "write_model_fits",
    "make_example_fields_png",
]


# ---------------------------------------------------------------------------
# FieldType enum
# ---------------------------------------------------------------------------

class FieldType(enum.Enum):
    """Classification of simulated fields by extended-emission character.

    POINT_ONLY          All-zero extended image.  casa_sim adds T-RECS points.
    DIFFUSE_DOMINANT    Several large finite apodized diffuse sources (web/filament).
    CENTRAL_SHELL_FLOW  One central shell/shock near field centre (shock kind).
    FULLY_DIFFUSE       Seamless full-field diffuse background (web/filament tile).
    """

    POINT_ONLY = "point_only"
    DIFFUSE_DOMINANT = "diffuse_dominant"
    CENTRAL_SHELL_FLOW = "central_shell_flow"
    FULLY_DIFFUSE = "fully_diffuse"


# ---------------------------------------------------------------------------
# Field-type balance
#
# This is the single canonical balance constant.  Override it globally or
# pass balance= to sample_field_type().  Weights are relative (normalised
# internally).  Final mixture tuning deferred until CASA dirty images confirm
# sensible SNR and morphology diversity.
# FLAG: revisit after first visual review of simulated corpus.
# ---------------------------------------------------------------------------

DEFAULT_FIELD_TYPE_BALANCE: dict[FieldType, float] = {
    FieldType.POINT_ONLY: 0.25,
    FieldType.DIFFUSE_DOMINANT: 0.32,
    FieldType.CENTRAL_SHELL_FLOW: 0.33,
    FieldType.FULLY_DIFFUSE: 0.10,
}


# ---------------------------------------------------------------------------
# Per-type parameter defaults
# ---------------------------------------------------------------------------

# Flux per finite extended source [Jy].  Log-uniform over this range.
#
# Calibration (2026-06-17): verified against T-RECS C-config field with AGN+SFG
# at 10 uJy floor.  Bright point sources peak at ~0.011 Jy/px; the target is
# that the extended peak surface brightness stays BELOW the bright compact sources
# so both are clearly visible.  With these values, 30-seed sweeps give:
#   DIFFUSE_DOMINANT   peak median ~0.0003 Jy/px, max ~0.0014 Jy/px
#   CENTRAL_SHELL_FLOW peak median ~0.0005 Jy/px, max ~0.0043 Jy/px
# Both well below the AGN/SFG bright-point peak (~0.011 Jy/px), giving ~8x
# headroom.  Points clearly rise above diffuse in all draws.
# Knob: lower _FLUX_RANGE_JY_DIFFUSE[1] or _FLUX_RANGE_JY_SHOCK[1] to suppress
# further; raise to make extended brighter.  Do not exceed 0.5 Jy on either
# upper bound without re-checking the peak ratio against a C-config T-RECS render.
_FLUX_RANGE_JY_DIFFUSE: tuple[float, float] = (0.01, 0.25)
_FLUX_RANGE_JY_SHOCK: tuple[float, float] = (0.02, 0.40)

# Angular diameter of each finite source [arcsec].
# At 1.8 arcsec/px for a 512 px field (921.6 arcsec), these cover
# 10%-65% of the field.
_SIZE_RANGE_ARCSEC_DIFFUSE: tuple[float, float] = (80.0, 600.0)
_SIZE_RANGE_ARCSEC_SHOCK: tuple[float, float] = (60.0, 350.0)

# Number of finite sources for DIFFUSE_DOMINANT.
_N_DIFFUSE_DOMINANT: int = 4

# Fully-diffuse total flux [Jy/field].  Spread over the entire 512x512 field,
# so per-pixel brightness is tiny (~5e-7 Jy/px at 0.1 Jy total), well below
# point sources.  This is intentional: FULLY_DIFFUSE is a faint background layer.
# Knob: raise to make diffuse background more prominent; keep under 0.2 Jy to
# avoid burying point sources.
_FULLY_DIFFUSE_FLUX_JY: float = 0.1

# Kinds used per field type.  compact (box_orange) is intentionally excluded
# from all extended field types -- points come from T-RECS on the casa_sim side.
_DIFFUSE_KINDS: tuple[str, ...] = ("web", "filament")
_DIFFUSE_WEIGHTS: tuple[float, ...] = (1.0, 2.0)  # filaments slightly preferred

# Placement margin as a fraction of imsize.
_PLACEMENT_MARGIN_FRAC: float = 0.05

# Minimum source size in pixels.
_MIN_SIZE_PX: int = 4


# ---------------------------------------------------------------------------
# Template resampling (numpy / scipy -- no torch)
# ---------------------------------------------------------------------------

def _resample_template(template: np.ndarray, target_px: int) -> np.ndarray:
    """Resample a (H, W) float32 numpy template to (target_px, target_px).

    Uses scipy.ndimage.zoom with order=1 (bilinear-equivalent).  The template
    is first rescaled so its longer axis equals target_px, then centre-padded
    to exactly (target_px, target_px).  Output is unit-max normalised and
    non-negative, dtype float32.

    Note: scipy.ndimage.zoom order=1 and torch.nn.functional.interpolate
    bilinear produce slightly different pixel values (different anti-aliasing
    conventions).  The visual output and flux-conservation behaviour are
    equivalent for this use case.
    """
    H, W = template.shape
    scale = target_px / max(H, W)
    new_H = round(H * scale)
    new_W = round(W * scale)

    zoom_h = new_H / H
    zoom_w = new_W / W
    resampled = _ndimage_zoom(
        template.astype(np.float32), (zoom_h, zoom_w), order=1, prefilter=False
    )

    # Centre-pad to (target_px, target_px)
    out = np.zeros((target_px, target_px), dtype=np.float32)
    h_off = (target_px - resampled.shape[0]) // 2
    w_off = (target_px - resampled.shape[1]) // 2
    h_end = h_off + resampled.shape[0]
    w_end = w_off + resampled.shape[1]
    h_end_c = min(h_end, target_px)
    w_end_c = min(w_end, target_px)
    out[h_off:h_end_c, w_off:w_end_c] = resampled[: h_end_c - h_off, : w_end_c - w_off]

    # Non-negativity + unit-max normalise
    out = np.clip(out, 0.0, None)
    m = out.max()
    if m > 0.0:
        out = out / m
    return out.astype(np.float32)


def _random_augment(template: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random flip + 90-degree rotation to a (H, W) float32 numpy array."""
    if rng.random() > 0.5:
        template = np.fliplr(template)
    if rng.random() > 0.5:
        template = np.flipud(template)
    k = int(rng.integers(0, 4))
    if k:
        template = np.rot90(template, k=k)
    return np.ascontiguousarray(template)


# ---------------------------------------------------------------------------
# Hann apodization window (numpy)
# ---------------------------------------------------------------------------

def _hann_window_1d(n: int) -> np.ndarray:
    """Return a 1-D Hann window of length n as float32."""
    # numpy equivalent of torch.hann_window(n, periodic=False)
    # Hann: 0.5 * (1 - cos(2*pi*k/(n-1)))  for k = 0..n-1
    k = np.arange(n, dtype=np.float32)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * k / (n - 1))).astype(np.float32)


def _hann_window_2d(size_px: int) -> np.ndarray:
    """Build a separable 2-D Hann (raised-cosine) window of shape (size_px, size_px).

    The window is 1.0 at the centre and 0.0 at all four edges.  Multiplication
    by this window apodizes a template tile so that it fades smoothly to zero
    at its footprint boundary, eliminating hard square edges when composited.

    Returns float32 numpy array.
    """
    w1d = _hann_window_1d(size_px)
    return np.outer(w1d, w1d).astype(np.float32)


def _apodize(template: np.ndarray) -> np.ndarray:
    """Apply a Hann window to a (N, N) float32 numpy array.  Returns float32."""
    H, W = template.shape
    size = min(H, W)
    win = _hann_window_2d(size)
    if H != W:
        full = np.zeros((H, W), dtype=np.float32)
        rh = (H - size) // 2
        rw = (W - size) // 2
        full[rh:rh + size, rw:rw + size] = win
        win = full
    return (template * win).astype(np.float32)


# ---------------------------------------------------------------------------
# Finite source placer (shared by DIFFUSE_DOMINANT and CENTRAL_SHELL_FLOW)
# ---------------------------------------------------------------------------

def _place_finite_source(
    image: np.ndarray,
    template: np.ndarray,
    flux_jy: float,
    rng: np.random.Generator,
    imsize: int,
    margin_px: int,
    centre: bool = False,
    centre_jitter_frac: float = 0.1,
) -> dict | None:
    """Place one apodized finite source into ``image`` (in-place).

    Parameters
    ----------
    image:
        (imsize, imsize) float32 numpy array accumulation array.  Modified in-place.
    template:
        Resampled, augmented (size_px, size_px) float32 numpy array morphology template.
        Will be apodized (Hann-windowed) and flux-normalised before placement.
    flux_jy:
        Target total flux in Jy.
    rng:
        NumPy Generator (used for placement randomness only).
    imsize:
        Image side length in pixels.
    margin_px:
        Minimum distance from image edge for source centre.
    centre:
        If True, place near field centre with a small jitter.
    centre_jitter_frac:
        When ``centre=True``, the centre is randomly displaced by up to this
        fraction of imsize in each axis.

    Returns
    -------
    dict with placement metadata, or None if the template is degenerate (all-zero).
    """
    size_px = template.shape[0]

    # Apodize: fade to zero at footprint edges
    template = _apodize(template)

    # Ensure non-negative after apodization (should be already, but guard)
    template = np.clip(template, 0.0, None)

    # Flux-normalise
    total = float(template.sum())
    if total <= 0.0:
        return None
    template = template * (flux_jy / total)

    # Determine placement centre
    half = size_px // 2
    if centre:
        jitter_px = max(1, int(imsize * centre_jitter_frac))
        row_c = imsize // 2 + int(rng.integers(-jitter_px, jitter_px + 1))
        col_c = imsize // 2 + int(rng.integers(-jitter_px, jitter_px + 1))
    else:
        lo = margin_px + half
        hi = imsize - margin_px - half
        if lo >= hi:
            row_c = imsize // 2
            col_c = imsize // 2
        else:
            row_c = int(rng.integers(lo, hi))
            col_c = int(rng.integers(lo, hi))

    # Bounding box
    r0 = row_c - half
    r1 = r0 + size_px
    c0 = col_c - half
    c1 = c0 + size_px

    # Clamp to image bounds
    tr0 = max(0, -r0)
    tc0 = max(0, -c0)
    ir0 = max(0, r0)
    ic0 = max(0, c0)
    ir1 = min(imsize, r1)
    ic1 = min(imsize, c1)
    tr1 = tr0 + (ir1 - ir0)
    tc1 = tc0 + (ic1 - ic0)

    if ir1 > ir0 and ic1 > ic0:
        image[ir0:ir1, ic0:ic1] += template[tr0:tr1, tc0:tc1]

    return {
        "row_px": row_c,
        "col_px": col_c,
        "flux_jy": float(flux_jy),
        "size_px": int(size_px),
    }


# ---------------------------------------------------------------------------
# FieldType builders
# ---------------------------------------------------------------------------

def _build_point_only(
    rng: np.random.Generator,
    imsize: int,
) -> tuple[np.ndarray, dict]:
    """Return an all-zero image (extended layer only).  T-RECS adds points."""
    image = np.zeros((imsize, imsize), dtype=np.float32)
    meta: dict = {"sources": [], "field_type": FieldType.POINT_ONLY.value}
    return image, meta


def _build_diffuse_dominant(
    rng: np.random.Generator,
    lib,
    imsize: int,
    cell_arcsec: float,
    n_sources: int = _N_DIFFUSE_DOMINANT,
    flux_range_jy: tuple[float, float] = _FLUX_RANGE_JY_DIFFUSE,
    size_range_arcsec: tuple[float, float] = _SIZE_RANGE_ARCSEC_DIFFUSE,
) -> tuple[np.ndarray, dict]:
    """Several large apodized diffuse sources (web/filament kinds)."""
    image = np.zeros((imsize, imsize), dtype=np.float32)
    sources_meta: list[dict] = []
    margin_px = max(int(math.ceil(imsize * _PLACEMENT_MARGIN_FRAC)), _MIN_SIZE_PX)

    kinds_arr = list(_DIFFUSE_KINDS)
    weights_arr = np.asarray(_DIFFUSE_WEIGHTS, dtype=np.float64)
    weights_arr = weights_arr / weights_arr.sum()

    s_min, s_max = float(flux_range_jy[0]), float(flux_range_jy[1])
    theta_min, theta_max = float(size_range_arcsec[0]), float(size_range_arcsec[1])

    for _ in range(n_sources):
        kind = kinds_arr[int(rng.choice(len(kinds_arr), p=weights_arr))]
        log_flux = rng.uniform(math.log(s_min), math.log(s_max))
        flux_jy = math.exp(log_flux)

        log_theta = rng.uniform(math.log(theta_min), math.log(theta_max))
        size_arcsec = math.exp(log_theta)
        size_px = max(_MIN_SIZE_PX, int(round(size_arcsec / cell_arcsec)))
        max_size_px = imsize - 2 * margin_px
        size_px = min(size_px, max(max_size_px, _MIN_SIZE_PX))

        template_np, tmeta = lib.sample(rng, kind=kind)
        template = _resample_template(template_np, size_px)
        template = _random_augment(template, rng)

        result = _place_finite_source(
            image, template, flux_jy, rng, imsize, margin_px, centre=False
        )
        if result is None:
            sources_meta.append({
                "kind": kind,
                "template_id": tmeta.get("id"),
                "note": "degenerate_template_skipped",
            })
        else:
            sources_meta.append({
                "kind": kind,
                "template_id": tmeta.get("id"),
                "size_arcsec": float(size_arcsec),
                **result,
            })

    image = np.clip(image, 0.0, None)
    meta = {
        "field_type": FieldType.DIFFUSE_DOMINANT.value,
        "sources": sources_meta,
    }
    return image, meta


def _build_central_shell_flow(
    rng: np.random.Generator,
    lib,
    imsize: int,
    cell_arcsec: float,
    flux_range_jy: tuple[float, float] = _FLUX_RANGE_JY_SHOCK,
    size_range_arcsec: tuple[float, float] = _SIZE_RANGE_ARCSEC_SHOCK,
) -> tuple[np.ndarray, dict]:
    """One central shock/shell source, apodized, near field centre."""
    image = np.zeros((imsize, imsize), dtype=np.float32)
    margin_px = max(int(math.ceil(imsize * _PLACEMENT_MARGIN_FRAC)), _MIN_SIZE_PX)

    log_flux = rng.uniform(math.log(flux_range_jy[0]), math.log(flux_range_jy[1]))
    flux_jy = math.exp(log_flux)

    log_theta = rng.uniform(math.log(size_range_arcsec[0]), math.log(size_range_arcsec[1]))
    size_arcsec = math.exp(log_theta)
    size_px = max(_MIN_SIZE_PX, int(round(size_arcsec / cell_arcsec)))
    max_size_px = imsize - 2 * margin_px
    size_px = min(size_px, max(max_size_px, _MIN_SIZE_PX))

    template_np, tmeta = lib.sample(rng, kind="shock")
    template = _resample_template(template_np, size_px)
    template = _random_augment(template, rng)

    result = _place_finite_source(
        image, template, flux_jy, rng, imsize, margin_px, centre=True
    )

    image = np.clip(image, 0.0, None)
    if result is None:
        sources = [{"kind": "shock", "template_id": tmeta.get("id"),
                    "note": "degenerate_template_skipped"}]
    else:
        sources = [{
            "kind": "shock",
            "template_id": tmeta.get("id"),
            "size_arcsec": float(size_arcsec),
            **result,
        }]

    meta = {
        "field_type": FieldType.CENTRAL_SHELL_FLOW.value,
        "sources": sources,
    }
    return image, meta


def _build_fully_diffuse(
    rng: np.random.Generator,
    lib,
    imsize: int,
    total_flux_jy: float = _FULLY_DIFFUSE_FLUX_JY,
) -> tuple[np.ndarray, dict]:
    """Faint seamless full-field diffuse background.

    A single large template from the web or filament kind is resampled to
    at least (imsize x imsize) by zooming up via scipy.ndimage.zoom.  To avoid
    any internal seams, no tiling is used -- instead the template is upscaled
    (zoom >= 1).  If the aspect ratio of the source template differs from 1:1,
    the shorter axis is upscaled to cover the field (no gaps).  The output is
    then cropped or padded to exactly imsize x imsize.

    This layer is NOT apodized: it is a seamless background by design.
    """
    image = np.zeros((imsize, imsize), dtype=np.float32)

    kind = _DIFFUSE_KINDS[int(rng.choice(len(_DIFFUSE_KINDS)))]
    template_np, tmeta = lib.sample(rng, kind=kind)
    template_np = _random_augment(template_np, rng)

    H, W = template_np.shape
    scale = max(imsize / H, imsize / W) * (1.0 + rng.uniform(0.0, 0.2))
    new_H = round(H * scale)
    new_W = round(W * scale)

    zoom_h = new_H / H
    zoom_w = new_W / W
    zoomed = _ndimage_zoom(
        template_np.astype(np.float32), (zoom_h, zoom_w), order=1, prefilter=False
    )
    zoomed = np.clip(zoomed, 0.0, None)

    zH, zW = zoomed.shape

    # Random crop to exactly imsize x imsize
    if zH > imsize:
        r0 = int(rng.integers(0, zH - imsize + 1))
    else:
        r0 = 0
    if zW > imsize:
        c0 = int(rng.integers(0, zW - imsize + 1))
    else:
        c0 = 0
    crop = zoomed[r0:r0 + imsize, c0:c0 + imsize]

    # Pad if zoom produced slightly under-sized output (rare)
    if crop.shape != (imsize, imsize):
        padded = np.zeros((imsize, imsize), dtype=np.float32)
        ch, cw = crop.shape
        padded[:ch, :cw] = crop
        crop = padded

    # Normalise to total_flux_jy
    total = float(crop.sum())
    if total > 0.0:
        crop = crop * (total_flux_jy / total)
    # If degenerate (all-zero), leave as zeros

    image = image + crop
    image = np.clip(image, 0.0, None)

    meta = {
        "field_type": FieldType.FULLY_DIFFUSE.value,
        "sources": [{
            "kind": kind,
            "template_id": tmeta.get("id"),
            "flux_jy": float(total_flux_jy),
            "size_px": imsize,
            "note": "full_field_diffuse_background",
        }],
    }
    return image, meta


# ---------------------------------------------------------------------------
# Public field-type sampler
# ---------------------------------------------------------------------------

def sample_field_type(
    rng: np.random.Generator,
    balance: Mapping[FieldType, float] | None = None,
) -> FieldType:
    """Sample a FieldType according to the given balance weights.

    Parameters
    ----------
    rng:
        NumPy random generator.
    balance:
        Mapping from FieldType to relative weight.  Weights need not sum to 1;
        they are normalised internally.  If None, uses DEFAULT_FIELD_TYPE_BALANCE.

    Returns
    -------
    FieldType
    """
    if balance is None:
        balance = DEFAULT_FIELD_TYPE_BALANCE
    types = list(balance.keys())
    weights = np.asarray([balance[t] for t in types], dtype=np.float64)
    weights = weights / weights.sum()
    idx = int(rng.choice(len(types), p=weights))
    return types[idx]


# ---------------------------------------------------------------------------
# Primary entry point: build_field
# ---------------------------------------------------------------------------

def build_field(
    rng: np.random.Generator,
    field_type: FieldType,
    *,
    ra_deg: float,
    dec_deg: float,
    cell_arcsec: float,
    imsize: int,
    freq_hz: float,
    repo_root: str | Path | None = None,
    # Per-type overrides (all optional)
    n_diffuse: int = _N_DIFFUSE_DOMINANT,
    flux_range_jy_diffuse: tuple[float, float] = _FLUX_RANGE_JY_DIFFUSE,
    size_range_arcsec_diffuse: tuple[float, float] = _SIZE_RANGE_ARCSEC_DIFFUSE,
    flux_range_jy_shock: tuple[float, float] = _FLUX_RANGE_JY_SHOCK,
    size_range_arcsec_shock: tuple[float, float] = _SIZE_RANGE_ARCSEC_SHOCK,
    fully_diffuse_flux_jy: float = _FULLY_DIFFUSE_FLUX_JY,
) -> tuple[np.ndarray, dict]:
    """Build a (imsize, imsize) float32 extended-source sky model in Jy/pixel.

    This is the primary public API.  T-RECS points are NOT rendered here;
    they are added on the casa_sim side.

    Parameters
    ----------
    rng:
        NumPy random generator.  Seed for reproducibility.
    field_type:
        FieldType enum value controlling which extended-emission pattern is
        rendered.
    ra_deg, dec_deg:
        Field centre in decimal degrees (ICRS/J2000).  Must match the
        casa_sim SimConfig.
    cell_arcsec:
        Pixel scale in arcseconds per pixel.  Must match the SimConfig cell.
    imsize:
        Image side length (pixels).  Field is (imsize x imsize).
    freq_hz:
        Representative frequency in Hz (for the FITS header; single channel).
    repo_root:
        Repo root containing ``data/morphology/lib/``.  If None, inferred from
        this file's location (correct for in-tree installs on any machine).
    n_diffuse:
        Number of sources for DIFFUSE_DOMINANT.
    flux_range_jy_diffuse:
        (S_min, S_max) log-uniform flux range for diffuse finite sources [Jy].
        FLAG: tune after seeing CASA dirty-image noise levels.
    size_range_arcsec_diffuse:
        (theta_min, theta_max) angular size range for diffuse finite sources.
    flux_range_jy_shock:
        Flux range for the CENTRAL_SHELL_FLOW shock source [Jy].
    size_range_arcsec_shock:
        Angular size range for the CENTRAL_SHELL_FLOW shock source.
    fully_diffuse_flux_jy:
        Total flux [Jy] for the FULLY_DIFFUSE background.  Intentionally
        faint; tune after CASA dirty images.

    Returns
    -------
    image : numpy.ndarray, shape (imsize, imsize), dtype float32
        Extended sky model in Jy/pixel.  Non-negative.
        POINT_ONLY returns all-zeros.
    meta : dict
        Bookkeeping dict.  Always contains:
          "field_type", "ra_deg", "dec_deg", "cell_arcsec", "imsize",
          "freq_hz", "sources" (list of per-source dicts).
    """
    from .morphology_templates import TemplateLibrary

    if repo_root is None:
        # This file lives at casa_sim/skymodel/morphology_field.py
        # Repo root is three levels up: skymodel/ -> casa_sim/ -> repo/
        repo_root = Path(__file__).parent.parent.parent

    # POINT_ONLY does not need the library
    if field_type == FieldType.POINT_ONLY:
        image, inner_meta = _build_point_only(rng, imsize)
    else:
        lib = TemplateLibrary(
            repo_root,
            kinds=_kinds_for_type(field_type),
        )
        if field_type == FieldType.DIFFUSE_DOMINANT:
            image, inner_meta = _build_diffuse_dominant(
                rng, lib, imsize, cell_arcsec,
                n_sources=n_diffuse,
                flux_range_jy=flux_range_jy_diffuse,
                size_range_arcsec=size_range_arcsec_diffuse,
            )
        elif field_type == FieldType.CENTRAL_SHELL_FLOW:
            image, inner_meta = _build_central_shell_flow(
                rng, lib, imsize, cell_arcsec,
                flux_range_jy=flux_range_jy_shock,
                size_range_arcsec=size_range_arcsec_shock,
            )
        elif field_type == FieldType.FULLY_DIFFUSE:
            image, inner_meta = _build_fully_diffuse(
                rng, lib, imsize,
                total_flux_jy=fully_diffuse_flux_jy,
            )
        else:
            raise ValueError(f"Unknown FieldType: {field_type!r}")

    meta = {
        "field_type": field_type.value,
        "ra_deg": float(ra_deg),
        "dec_deg": float(dec_deg),
        "cell_arcsec": float(cell_arcsec),
        "imsize": int(imsize),
        "freq_hz": float(freq_hz),
        "sources": inner_meta.get("sources", []),
    }
    return image, meta


def _kinds_for_type(field_type: FieldType) -> list[str]:
    """Return the template kinds needed for a given FieldType."""
    if field_type == FieldType.POINT_ONLY:
        return []
    if field_type == FieldType.CENTRAL_SHELL_FLOW:
        return ["shock"]
    # DIFFUSE_DOMINANT and FULLY_DIFFUSE
    return list(_DIFFUSE_KINDS)


# ---------------------------------------------------------------------------
# FITS writer
# ---------------------------------------------------------------------------

def write_model_fits(
    image: np.ndarray,
    path: str | Path,
    *,
    ra_deg: float,
    dec_deg: float,
    cell_arcsec: float,
    freq_hz: float,
) -> Path:
    """Write a sky-model image to a CASA-ingestable FITS file.

    The output FITS has shape (1, 1, imsize, imsize) in FITS axis order
    (FREQ, STOKES, DEC, RA), which corresponds to NAXIS4=1 (freq),
    NAXIS3=1 (Stokes I), NAXIS2=imsize (DEC), NAXIS1=imsize (RA).

    FITS header conventions follow what CASA importfits expects:

      - NAXIS=4
      - CTYPE1='RA---SIN'   CDELT1=-cell_deg  (negative: RA increases right
                                                in sky, decreasing col index)
      - CTYPE2='DEC--SIN'   CDELT2=+cell_deg
      - CTYPE3='STOKES'     CRVAL3=1           (I)
      - CTYPE4='FREQ'       CRVAL4=freq_hz     CDELT4=0 (single channel)
      - CRPIX1=imsize/2+1   CRPIX2=imsize/2+1 (FITS 1-indexed field centre)
      - BUNIT='Jy/pixel'
      - EQUINOX=2000.0

    Parameters
    ----------
    image:
        (imsize, imsize) float32 or float64 sky model in Jy/pixel.
        Must be non-negative (truth sky).
    path:
        Output FITS path.  Parent directory must exist.
    ra_deg, dec_deg:
        Field centre in decimal degrees (J2000).
    cell_arcsec:
        Pixel scale in arcseconds per pixel.
    freq_hz:
        Reference frequency in Hz (single channel).

    Returns
    -------
    Path of the written FITS file.
    """
    from astropy.io import fits as _fits

    image_np = np.asarray(image, dtype=np.float32)
    imsize = image_np.shape[0]
    if image_np.shape != (imsize, imsize):
        raise ValueError(
            f"image must be square (imsize x imsize), got {image_np.shape}"
        )

    cell_deg = cell_arcsec / 3600.0
    crpix = float(imsize) / 2.0 + 1.0  # FITS 1-indexed centre pixel

    data = image_np[np.newaxis, np.newaxis, :, :]

    hdu = _fits.PrimaryHDU(data)
    h = hdu.header

    h["NAXIS"] = 4
    h["NAXIS1"] = imsize
    h["NAXIS2"] = imsize
    h["NAXIS3"] = 1
    h["NAXIS4"] = 1

    h["CTYPE1"] = "RA---SIN"
    h["CRVAL1"] = float(ra_deg)
    h["CRPIX1"] = crpix
    h["CDELT1"] = -cell_deg
    h["CUNIT1"] = "deg"

    h["CTYPE2"] = "DEC--SIN"
    h["CRVAL2"] = float(dec_deg)
    h["CRPIX2"] = crpix
    h["CDELT2"] = +cell_deg
    h["CUNIT2"] = "deg"

    h["CTYPE3"] = "STOKES"
    h["CRVAL3"] = 1.0
    h["CRPIX3"] = 1.0
    h["CDELT3"] = 1.0
    h["CUNIT3"] = ""

    h["CTYPE4"] = "FREQ"
    h["CRVAL4"] = float(freq_hz)
    h["CRPIX4"] = 1.0
    h["CDELT4"] = 0.0
    h["CUNIT4"] = "Hz"

    h["BUNIT"] = "Jy/pixel"
    h["EQUINOX"] = 2000.0
    h["RADESYS"] = "FK5"
    h["SPECSYS"] = "TOPOCENT"

    out = Path(path)
    hdu.writeto(str(out), overwrite=True)
    return out


# ---------------------------------------------------------------------------
# Example fields montage (2x2, one per FieldType)
# ---------------------------------------------------------------------------

def make_example_fields_png(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
    rng: np.random.Generator | None = None,
    imsize: int = 512,
    cell_arcsec: float = 1.8,
    ra_deg: float = 278.5,
    dec_deg: float = -2.1,
    freq_hz: float = 1.4e9,
) -> Path:
    """Build one example field per FieldType and write a 2x2 montage PNG.

    Uses an asinh stretch and inferno colormap.  Each panel is labelled with
    its FieldType name so a human can validate the taxonomy and confirm that
    hard square edges are absent.

    Returns the output path.
    """
    from PIL import Image as _PIL_Image
    from PIL import ImageDraw, ImageFont
    import matplotlib as _mpl

    if rng is None:
        rng = np.random.default_rng(42)
    if repo_root is None:
        repo_root = Path(__file__).parent.parent.parent

    cmap = _mpl.colormaps["inferno"]

    panel_size = 512
    gap = 8
    label_h = 22
    types_ordered = [
        FieldType.POINT_ONLY,
        FieldType.DIFFUSE_DOMINANT,
        FieldType.CENTRAL_SHELL_FLOW,
        FieldType.FULLY_DIFFUSE,
    ]

    # Build each panel image
    panels: list[np.ndarray] = []
    for ft in types_ordered:
        image_np, _ = build_field(
            rng,
            ft,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            cell_arcsec=cell_arcsec,
            imsize=imsize,
            freq_hz=freq_hz,
            repo_root=repo_root,
        )

        # Asinh stretch normalised to [0, 1]
        vmax = float(image_np.max())
        if vmax > 0.0:
            stretched = np.arcsinh(image_np / vmax * 10.0) / np.arcsinh(10.0)
        else:
            stretched = np.zeros_like(image_np)

        # Inferno colormap -> RGB
        rgba = (cmap(stretched) * 255).astype(np.uint8)
        rgb = rgba[:, :, :3]

        # Resize to panel_size if imsize differs
        if imsize != panel_size:
            pil = _PIL_Image.fromarray(rgb, mode="RGB")
            pil = pil.resize((panel_size, panel_size), _PIL_Image.LANCZOS)
            rgb = np.array(pil)

        panels.append(rgb)

    # Compose 2x2 montage
    total_w = 2 * panel_size + gap
    total_h = 2 * (panel_size + label_h) + gap
    canvas = _PIL_Image.new("RGB", (total_w, total_h), color=(10, 10, 10))

    positions = [
        (0, 0),
        (panel_size + gap, 0),
        (0, panel_size + label_h + gap),
        (panel_size + gap, panel_size + label_h + gap),
    ]
    labels = [ft.value for ft in types_ordered]

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for (x, y), rgb, label in zip(positions, panels, labels):
        panel_img = _PIL_Image.fromarray(rgb, mode="RGB")
        canvas.paste(panel_img, (x, y + label_h))
        draw.text((x + 4, y + 4), label, fill=(220, 200, 80), font=font)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out))
    return out
