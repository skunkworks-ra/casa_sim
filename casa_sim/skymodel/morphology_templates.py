"""casa_sim.skymodel.morphology_templates
=========================================
Build-once, reuse-forever library of normalised scalar morphology templates
extracted from four pre-rendered IllustrisTNG media PNGs.

Templates are relative surface-brightness morphology proxies (colormapped
renders, not physical maps).  Flux is imposed downstream; these arrays carry
morphology only.

On-disk assets
--------------
Source PNGs (NOT in git -- rsync from data store):
    data/morphology/tng_raw/*.png

Built library (NOT in git -- regenerate via build_template_library()):
    data/morphology/lib/<id>.npy   float32 arrays, unit-max normalised
    data/morphology/lib/manifest.json

Usage
-----
    from casa_sim.skymodel.morphology_templates import build_template_library, TemplateLibrary

    # Build once:
    build_template_library(repo_root)

    # Use at sky-build time:
    lib = TemplateLibrary(repo_root)
    arr, meta = lib.sample(rng, kind="shock")

Kind taxonomy (4 distinct kinds, 318 templates total)
------------------------------------------------------
    "shock"    -- TNG_Shocks panels (48):    shock fronts / shells / edges
    "compact"  -- box_orange tiles (36):     compact halos, Mach-number bright knots
    "web"      -- box_blue tiles (36):       diffuse cosmic web strands
    "filament" -- gas_density tiles (198):   filamentary cosmic web, dense knots

    NOTE: "compact" (box_orange) is present in the library but is retired from
    all extended field types in morphology_field.py.  T-RECS handles compact
    sources.  The library is kept as-is; retirement is enforced in the field
    builders, not here.

Donor summary (measured pixel dimensions)
------------------------------------------
TNG_Shocks_6times8_2k.png         1500 x 2000 px  --  6-col x 8-row montage
                                    panels: 240 x 240 px each (48 total)
                                    gutters: 5-px white borders + 5-px dark
                                    background: near-black (lum ~= 11)
                                    colormap: red-on-black (single warm channel)
                                    kind: "shock"
                                    ANNOTATION (1): text block top-left and scale
                                    bar bottom-right -- present in every panel.
                                    ANNOTATION (2): faint gray virial-radius circle
                                    overlaid on all panels; center ~= (118,110) px,
                                    radius ~= 120 px in the 240x240 panel.
                                    Fix: bottom corners zeroed (50 px x 80 px each
                                    side); virial circle zeroed by color-gating
                                    (balance = |R-B|+|R-G| < 40 within 8 px of
                                    circle arc).  Zero collateral damage on warm
                                    shock emission confirmed across all 48 panels.

TNG50_gas_density_z3_slice_5k.png  5000 x 3125 px  --  full-frame cosmic web
                                    background: dark (lum ~= 51 at p5)
                                    colormap: purple-to-orange (palette P->RGB)
                                    extract as overlapping 512 x 512 tiles
                                    no annotations -- clean
                                    kind: "filament"

boxComposite_TNG100-1_..._2000.png 2000 x 2000 px  --  full-frame two-colour
                                    background: dark (~16, 18, 29)
                                    orange channel: R - B > 0  -> kind "compact"
                                    blue channel:   B - R > 0  -> kind "web"
                                    extract each as overlapping 512 x 512 tiles
                                    no annotations -- clean

TNG100_jellyfish_gallery_3200.png  -- DROPPED (zero templates produced)
                                    Every panel has a dashed virial circle running
                                    through the galaxy tail -- not salvageable.
                                    Decision: produce zero jellyfish/tail templates.

TNG50_galaxies_Halpha_starlight_z2.png -- DROPPED (zero templates produced)
                                    The left-half H-alpha panel is a 5x5 grid of
                                    sub-images (378x474 px each), separated by
                                    white annotation bars.  Every sub-image has
                                    fixed-position text blocks at local rows 14-27
                                    (top) and 318-357 (bottom).  The clean galaxy
                                    zone is only ~290 px tall -- too narrow for
                                    512x512 tiles (yields ~1 tile per sub-image,
                                    25 total) and the text cannot be safely
                                    separated from the galaxy emission without
                                    shrinking below our minimum tile size.
                                    Decision: produce zero halpha templates; disk
                                    morphology must be re-sourced.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

__all__ = [
    "build_template_library",
    "TemplateLibrary",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TNG_RAW = Path("data") / "morphology" / "tng_raw"
_LIB = Path("data") / "morphology" / "lib"

# Inset fraction applied when cropping montage panels to remove annotation margins.
_SHOCK_INSET: float = 0.04    # shocks: small inset, annotation removed by corner/arc masking

# Shock annotation mask zones (applied to the 240 x 240 panel, before inset crop).
# Text block occupies bottom-left corner; scale bar occupies bottom-right corner.
# Both are zeroed in the FULL panel (before inset crop) to avoid any residual.
# Bottom 50 px = 21% of 240 px -- annotation confirmed to span rows 195..239.
_SHOCK_ANNOT_BOTTOM_PX: int = 50   # zero bottom N rows of each panel
_SHOCK_ANNOT_CORNER_PX: int = 80   # zero left/right N columns of the bottom strip

# Virial-circle removal for shock panels.
# Each panel has a faint gray/white virial-radius circle overlaid.  Measured
# across all 48 panels: center consistently near (118, 110) px in 240x240 space,
# radius 111-125 px (median ~120).  The circle pixels have balanced RGB channels
# (|R-B|+|R-G| < 40) while shock emission is warm (R >> B).  Zeroing gray pixels
# within 8 px of the circle arc removes the virial circle with zero collateral
# damage to shock emission -- verified across all 48 panels.
_SHOCK_VIRIAL_CX: float = 118.0   # circle centre x (col) in 240x240 panel
_SHOCK_VIRIAL_CY: float = 110.0   # circle centre y (row) in 240x240 panel
_SHOCK_VIRIAL_R: float = 120.0    # circle radius in pixels
_SHOCK_VIRIAL_BAND: float = 8.0   # half-width of annular mask in pixels
_SHOCK_VIRIAL_BALANCE_MAX: float = 40.0  # max |R-B|+|R-G| to call a pixel "gray"
_SHOCK_VIRIAL_LUM_MIN: float = 16.0      # min luminance to call a pixel non-background

# Tile config for full-frame donors.
_TILE_SIZE: int = 512
_TILE_STRIDE: int = 256   # 50% overlap -> more templates from large frames


# ---------------------------------------------------------------------------
# Background subtraction helpers
# ---------------------------------------------------------------------------

def _modal_floor(arr: np.ndarray, pct: float = 2.0) -> float:
    """Return the low-percentile pixel value as the background floor estimate."""
    return float(np.percentile(arr, pct))


def _bg_subtract_clip(arr: np.ndarray, pct: float = 2.0) -> np.ndarray:
    """Subtract background floor and clip to >= 0.  Returns float32."""
    floor = _modal_floor(arr, pct)
    return np.clip(arr - floor, 0.0, None).astype(np.float32)


def _unit_max(arr: np.ndarray) -> np.ndarray:
    """Normalise to [0, 1] by dividing by maximum.  Returns float32.
    If the array is all-zero, returns zeros (no-op)."""
    m = arr.max()
    if m > 0:
        return (arr / m).astype(np.float32)
    return arr.astype(np.float32)


def _to_luminance(rgb: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) uint8 RGB to float64 luminance (H, W)."""
    return rgb.astype(np.float64).mean(axis=2)


# ---------------------------------------------------------------------------
# Donor 1: TNG Shocks -- 6 x 8 montage, black background
# ---------------------------------------------------------------------------

def _extract_shocks(raw_dir: Path) -> list[tuple[np.ndarray, dict]]:
    """
    Slice 48 shock-front panels from TNG_Shocks_6times8_2k.png.

    Measured grid:
        Image: 1500 (W) x 2000 (H) px
        White gutter bands at multiples of 250 px (5-px white + 5-px dark border)
        Panel interiors: rows 5-245, 255-495, ...  cols 5-245, 255-495, ...
        Each panel interior: 240 x 240 px
        Inset crop: _SHOCK_INSET from each edge of the 240 x 240 interior

    Annotation removal (two steps):
        (1) Bottom corners: text block (bottom-left) and scale bar (bottom-right).
            The bottom 50 px of the left 80 px and right 80 px are zeroed in the
            luminance map BEFORE the inset crop.
        (2) Virial circle: a faint gray/white circle arc is present in all panels
            at radius ~120 px, centred near (118, 110) in the 240x240 panel.
            Pixels within 8 px of this circle arc that have balanced RGB channels
            (|R-B|+|R-G| < 40, i.e. not warm-colored shock emission) are zeroed.
            Verified: zero collateral damage on warm shock emission across all 48
            panels.
    """
    img = Image.open(raw_dir / "TNG_Shocks_6times8_2k.png").convert("RGB")
    arr = np.array(img)  # (2000, 1500, 3)

    # Panel interior boundaries (measured: 5-pixel white + 5-pixel dark border)
    row_starts = [5, 255, 505, 755, 1005, 1255, 1505, 1755]
    col_starts = [5, 255, 505, 755, 1005, 1255]
    panel_h = 240
    panel_w = 240

    inset_h = int(math.ceil(panel_h * _SHOCK_INSET))
    inset_w = int(math.ceil(panel_w * _SHOCK_INSET))

    # Precompute virial-circle distance map (same for every panel: same size, same circle).
    yy, xx = np.mgrid[0:panel_h, 0:panel_w]
    _dist_from_virial = np.abs(
        np.sqrt((xx - _SHOCK_VIRIAL_CX) ** 2 + (yy - _SHOCK_VIRIAL_CY) ** 2)
        - _SHOCK_VIRIAL_R
    )
    _near_virial = _dist_from_virial < _SHOCK_VIRIAL_BAND

    templates = []
    idx = 0
    for ri, rs in enumerate(row_starts):
        for ci, cs in enumerate(col_starts):
            panel = arr[rs : rs + panel_h, cs : cs + panel_w, :]  # (240,240,3) uint8
            R_ch = panel[:, :, 0].astype(np.float64)
            G_ch = panel[:, :, 1].astype(np.float64)
            B_ch = panel[:, :, 2].astype(np.float64)

            # Convert to luminance (float64).
            lum = (R_ch + G_ch + B_ch) / 3.0

            # Step 1: zero bottom corners (text block + scale bar).
            lum[-_SHOCK_ANNOT_BOTTOM_PX:, :_SHOCK_ANNOT_CORNER_PX] = 0.0   # bottom-left
            lum[-_SHOCK_ANNOT_BOTTOM_PX:, -_SHOCK_ANNOT_CORNER_PX:] = 0.0  # bottom-right

            # Step 2: zero virial circle (gray arc, not warm shock emission).
            balance = np.abs(R_ch - B_ch) + np.abs(R_ch - G_ch)
            is_gray = balance < _SHOCK_VIRIAL_BALANCE_MAX
            is_virial = _near_virial & is_gray & (lum > _SHOCK_VIRIAL_LUM_MIN)
            lum[is_virial] = 0.0

            # Apply inset crop (removes any remaining gutter/edge artefacts).
            cropped_lum = lum[inset_h : panel_h - inset_h, inset_w : panel_w - inset_w]
            scalar = _unit_max(_bg_subtract_clip(cropped_lum))
            meta = {
                "id": f"shock_{idx:03d}",
                "source": "TNG_Shocks_6times8_2k.png",
                "kind": "shock",
                "grid_row": ri,
                "grid_col": ci,
                "panel_px": [panel_h, panel_w],
                "cropped_px": list(scalar.shape),
            }
            templates.append((scalar, meta))
            idx += 1
    return templates


# ---------------------------------------------------------------------------
# Donor 2: Gas density slice -- full frame, overlapping tiles
# ---------------------------------------------------------------------------

def _extract_gas_density(raw_dir: Path) -> list[tuple[np.ndarray, dict]]:
    """
    Tile TNG50_gas_density_z3_slice_5k.png into overlapping 512 x 512 crops.

    Image: 5000 (W) x 3125 (H) px.  Colormap: purple-to-orange (filaments/knots).
    No annotations.  Background: dark (p5 ~= 51 lum).
    Kind: "filament".
    """
    img = Image.open(raw_dir / "TNG50_gas_density_z3_slice_5k.png").convert("RGB")
    arr = np.array(img)  # (3125, 5000, 3)
    H, W = arr.shape[:2]
    lum = _to_luminance(arr)

    templates = []
    idx = 0
    row = 0
    while row + _TILE_SIZE <= H:
        col = 0
        while col + _TILE_SIZE <= W:
            tile = lum[row : row + _TILE_SIZE, col : col + _TILE_SIZE]
            scalar = _unit_max(_bg_subtract_clip(tile))
            meta = {
                "id": f"gas_{idx:04d}",
                "source": "TNG50_gas_density_z3_slice_5k.png",
                "kind": "filament",
                "tile_row": row,
                "tile_col": col,
                "tile_px": [_TILE_SIZE, _TILE_SIZE],
                "cropped_px": list(scalar.shape),
            }
            templates.append((scalar, meta))
            idx += 1
            col += _TILE_STRIDE
        row += _TILE_STRIDE
    return templates


# ---------------------------------------------------------------------------
# Donor 3: Box composite -- channel split into orange (compact) + blue (web)
# ---------------------------------------------------------------------------

def _extract_box_composite(raw_dir: Path) -> list[tuple[np.ndarray, dict]]:
    """
    Extract two channel-split views from the 2000 x 2000 two-colour composite.

    Background: dark (~16, 18, 29 RGB).
    Orange channel:  clip(R - B, 0)  -- compact halos / shocked gas  (kind="compact")
    Blue channel:    clip(B - R, 0)  -- diffuse cosmic web             (kind="web")
    Each is tiled into overlapping 512 x 512 crops.

    NOTE: This split is an approximation.  The image is a colormapped composite,
    not a true two-channel scientific map.  Orange fraction is sparse (~10% nonzero),
    blue is dense (~88% nonzero); both are useful morphology donors.
    """
    img = Image.open(
        raw_dir / "boxComposite_TNG100-1_gas-shocks_machnum_dm-coldens_2000.png"
    )
    arr = np.array(img)[:, :, :3].astype(np.float64)  # (2000, 2000, 3) ignore alpha
    H, W = arr.shape[:2]

    # Channel split: warm_excess = R - B (orange halos); cool_excess = B - R (blue web)
    warm = arr[:, :, 0] - arr[:, :, 2]
    cool = arr[:, :, 2] - arr[:, :, 0]
    orange_map = np.clip(warm, 0.0, None)
    blue_map = np.clip(cool, 0.0, None)

    templates = []
    for ch_name, ch_map, kind in [
        ("orange", orange_map, "compact"),
        ("blue", blue_map, "web"),
    ]:
        idx = 0
        row = 0
        while row + _TILE_SIZE <= H:
            col = 0
            while col + _TILE_SIZE <= W:
                tile = ch_map[row : row + _TILE_SIZE, col : col + _TILE_SIZE]
                scalar = _unit_max(_bg_subtract_clip(tile, pct=2.0))
                meta = {
                    "id": f"box_{ch_name}_{idx:04d}",
                    "source": "boxComposite_TNG100-1_gas-shocks_machnum_dm-coldens_2000.png",
                    "kind": kind,
                    "channel": ch_name,
                    "tile_row": row,
                    "tile_col": col,
                    "tile_px": [_TILE_SIZE, _TILE_SIZE],
                    "cropped_px": list(scalar.shape),
                }
                templates.append((scalar, meta))
                idx += 1
                col += _TILE_STRIDE
            row += _TILE_STRIDE
    return templates


# ---------------------------------------------------------------------------
# Donor 4: Jellyfish galaxy gallery -- DROPPED
# ---------------------------------------------------------------------------

def _extract_jellyfish(raw_dir: Path) -> list[tuple[np.ndarray, dict]]:
    """
    DROPPED -- returns an empty list.  Zero jellyfish/tail templates are produced.

    Background (retained for future re-sourcing decisions):
        TNG100_jellyfish_gallery_3200.png is a 5-row x 8-col montage of jellyfish
        galaxies.  Every panel has a dashed virial-radius circle running through
        the galaxy tail.  Unlike the TNG_Shocks virial circle (gray arc on a red
        image, cleanly separable by color), the jellyfish virial circle is
        indistinguishable from real tail emission in luminance space.  An 18%
        inset (72 px) was previously attempted but the circle radius spans the
        full panel interior -- the arc still appears in the cropped region.

        Decision: produce zero templates from this donor.  The "tail" kind tag is
        intentionally absent from the built library.
    """
    return []


# ---------------------------------------------------------------------------
# Donor 5: Halpha + starlight -- DROPPED
# ---------------------------------------------------------------------------

def _extract_halpha(raw_dir: Path) -> list[tuple[np.ndarray, dict]]:
    """
    DROPPED -- returns an empty list.  Zero halpha/disk templates are produced.

    Background (retained for future re-sourcing decisions):
        TNG50_galaxies_Halpha_starlight_z2.png is a 5x5 grid of galaxy sub-images
        (378x474 px each) separated by white annotation bars.  Every sub-image has
        fixed-position text blocks at local rows 14-27 (top) and 318-357 (bottom).
        The clean galaxy zone is only ~290 px tall -- too narrow for 512x512 tiles
        (yields ~1 tile per sub-image, 25 total).  The text blocks cannot be safely
        isolated from galaxy emission without shrinking below the minimum tile size.

        Decision: produce zero templates from this donor.  The orchestrator should
        re-source disk/spiral morphology from a cleaner image before enabling this
        kind in training.  The "disk" kind tag is intentionally absent from the
        built library when this function returns empty.
    """
    return []


# ---------------------------------------------------------------------------
# Public build function
# ---------------------------------------------------------------------------

def build_template_library(
    repo_root: str | Path,
    force: bool = False,
    contact_sheet: bool = True,
) -> Path:
    """Slice donor PNGs into scalar templates and write them to disk.

    Templates are written to ``<repo_root>/data/morphology/lib/`` as
    individual ``<id>.npy`` float32 files plus a ``manifest.json``.

    Parameters
    ----------
    repo_root:      Repo root directory (the one containing ``data/``).
    force:          Rebuild even if the library already exists.
    contact_sheet:  Write a contact-sheet PNG for visual review.

    Returns
    -------
    Path to the lib directory.
    """
    root = Path(repo_root)
    raw_dir = root / _TNG_RAW
    lib_dir = root / _LIB
    manifest_path = lib_dir / "manifest.json"

    if manifest_path.exists() and not force:
        return lib_dir

    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"TNG raw PNG directory not found: {raw_dir}\n"
            "rsync the source PNGs there before building the library."
        )

    lib_dir.mkdir(parents=True, exist_ok=True)

    all_templates: list[tuple[np.ndarray, dict]] = []

    extractors = [
        _extract_shocks,
        _extract_gas_density,
        _extract_box_composite,
        _extract_jellyfish,
        _extract_halpha,
    ]
    for extractor in extractors:
        templates = extractor(raw_dir)
        all_templates.extend(templates)

    manifest = []
    for arr, meta in all_templates:
        npy_path = lib_dir / f"{meta['id']}.npy"
        np.save(npy_path, arr)
        manifest.append(
            {
                **meta,
                "npy_path": str(npy_path.relative_to(root)),
                "dtype": "float32",
            }
        )

    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    if contact_sheet:
        _write_contact_sheet(all_templates, lib_dir / "contact_sheet.png")

    _write_postage_review(all_templates, lib_dir / "postage_review.png")

    return lib_dir


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------

def _write_contact_sheet(
    templates: list[tuple[np.ndarray, dict]],
    out_path: Path,
    n_cols: int = 8,
    n_rows: int = 8,
    thumb_size: int = 128,
    font_height: int = 10,
) -> None:
    """Write an 8 x 8 contact sheet of representative templates."""
    rng = np.random.default_rng(42)

    # Sample up to n_cols * n_rows templates, stratified by kind
    by_kind: dict[str, list[int]] = {}
    for i, (_, meta) in enumerate(templates):
        k = meta["kind"]
        by_kind.setdefault(k, []).append(i)

    # Shuffle each kind and interleave
    selected: list[int] = []
    kinds_order = sorted(by_kind.keys())
    per_kind_lists = []
    for k in kinds_order:
        idxs = by_kind[k].copy()
        rng.shuffle(idxs)
        per_kind_lists.append(idxs)

    n_total = n_cols * n_rows
    # Round-robin across kinds
    pointer = {k: 0 for k in kinds_order}
    while len(selected) < n_total:
        added = False
        for k, lst in zip(kinds_order, per_kind_lists):
            if pointer[k] < len(lst) and len(selected) < n_total:
                selected.append(lst[pointer[k]])
                pointer[k] += 1
                added = True
        if not added:
            break

    # Pad with random picks if needed
    all_idxs = list(range(len(templates)))
    rng.shuffle(all_idxs)
    extra = [i for i in all_idxs if i not in set(selected)]
    selected.extend(extra[: n_total - len(selected)])

    label_h = font_height + 2
    cell_h = thumb_size + label_h
    cell_w = thumb_size

    sheet_h = n_rows * cell_h
    sheet_w = n_cols * cell_w
    sheet = np.ones((sheet_h, sheet_w), dtype=np.float32)

    # Fill cells
    for i, idx in enumerate(selected[: n_total]):
        row = i // n_cols
        col = i % n_cols
        arr, meta = templates[idx]
        # Resize thumbnail
        pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        pil_thumb = pil.resize((thumb_size, thumb_size), Image.LANCZOS)
        thumb = np.array(pil_thumb).astype(np.float32) / 255.0

        r0 = row * cell_h + label_h
        c0 = col * cell_w
        sheet[r0 : r0 + thumb_size, c0 : c0 + thumb_size] = thumb

        # Label row: draw kind text via tiny pixel font approximation
        # (PIL ImageDraw is the cleanest path here)
        label = meta["kind"][:8]  # trim to fit
        _draw_label(sheet, label, r0 - label_h, c0, cell_w, label_h)

    # Convert to uint8 and save
    sheet_uint8 = (sheet * 255).clip(0, 255).astype(np.uint8)
    out_img = Image.fromarray(sheet_uint8, mode="L")
    out_img.save(str(out_path))


def _draw_label(
    sheet: np.ndarray,
    text: str,
    row: int,
    col: int,
    width: int,
    height: int,
) -> None:
    """Draw a tiny greyscale text label using PIL ImageDraw."""
    from PIL import Image as _Image
    from PIL import ImageDraw, ImageFont

    label_img = _Image.new("L", (width, height), color=200)
    draw = ImageDraw.Draw(label_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((1, 1), text, fill=20, font=font)
    label_arr = np.array(label_img).astype(np.float32) / 255.0

    r0, r1 = max(0, row), min(sheet.shape[0], row + height)
    c0, c1 = max(0, col), min(sheet.shape[1], col + width)
    lr = r1 - r0
    lc = c1 - c0
    if lr > 0 and lc > 0:
        sheet[r0:r1, c0:c1] = label_arr[:lr, :lc]


# ---------------------------------------------------------------------------
# Per-kind postage review montage
# ---------------------------------------------------------------------------

def _write_postage_review(
    templates: list[tuple[np.ndarray, dict]],
    out_path: Path,
    n_per_kind: int = 10,
    stamp_size: int = 220,
) -> None:
    """Write a per-kind postage montage at near-native resolution for human sign-off.

    Layout: one row per kind (shock / compact / web / filament), up to n_per_kind
    stamps per row, inferno colormap, evenly spaced sample across the kind's pool.

    Parameters
    ----------
    templates:    Full list of (array, meta) pairs from the build run.
    out_path:     Output PNG path (``data/morphology/lib/postage_review.png``).
    n_per_kind:   Number of stamps per kind row (default 10).
    stamp_size:   Each stamp is resized to this square size in pixels (default 220).
    """
    from PIL import Image as _Image
    from PIL import ImageDraw, ImageFont

    try:
        import matplotlib as _mpl
        _inferno = _mpl.colormaps["inferno"]
        def _apply_inferno(arr_f32: np.ndarray) -> np.ndarray:
            """Apply inferno colormap to float32 [0,1] array, return uint8 (H,W,3)."""
            rgba = (_inferno(arr_f32) * 255).astype(np.uint8)
            return rgba[:, :, :3]
    except ImportError:
        # Fallback: greyscale if matplotlib not available
        def _apply_inferno(arr_f32: np.ndarray) -> np.ndarray:  # type: ignore[misc]
            grey = (arr_f32 * 255).clip(0, 255).astype(np.uint8)
            return np.stack([grey, grey, grey], axis=2)

    # Ordered kind list for rows
    review_kinds = ["shock", "compact", "web", "filament"]

    label_h = 18
    gap = 4
    cell = stamp_size

    n_rows = sum(1 for k in review_kinds
                 if any(m["kind"] == k for _, m in templates))
    n_cols = n_per_kind

    sheet_w = n_cols * cell + (n_cols - 1) * gap
    sheet_h = n_rows * (cell + label_h) + (n_rows - 1) * gap

    sheet_img = _Image.new("RGB", (sheet_w, sheet_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(sheet_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    row_idx = 0
    for kind in review_kinds:
        kind_items = [(a, m) for a, m in templates if m["kind"] == kind]
        if not kind_items:
            continue

        # Evenly spaced selection across the kind pool
        n = min(n_per_kind, len(kind_items))
        step = max(1, len(kind_items) // n)
        picked = [kind_items[i * step] for i in range(n)]

        y0_label = row_idx * (cell + label_h + gap)
        y0_stamp = y0_label + label_h

        for col_idx, (arr, meta) in enumerate(picked):
            x0 = col_idx * (cell + gap)

            # Apply inferno colormap
            coloured = _apply_inferno(arr)
            pil = _Image.fromarray(coloured, mode="RGB")
            pil_sized = pil.resize((cell, cell), _Image.LANCZOS)
            sheet_img.paste(pil_sized, (x0, y0_stamp))

            # Label above stamp: kind + id
            label_text = f"{meta['kind']}  {meta['id']}"
            draw.text((x0 + 2, y0_label + 2), label_text,
                      fill=(220, 200, 80), font=font)

        row_idx += 1

    sheet_img.save(str(out_path))


# ---------------------------------------------------------------------------
# TemplateLibrary -- loader / sampler
# ---------------------------------------------------------------------------

class TemplateLibrary:
    """Library of scalar morphology templates loaded from disk.

    Template files are NOT in git.  Build them first with
    :func:`build_template_library`.

    Parameters
    ----------
    repo_root:  Repo root directory (the one containing ``data/``).
    kinds:      If given, restrict to templates with these ``kind`` tags.
                Valid tags: ``"shock"``, ``"compact"``, ``"web"``, ``"filament"``.
                Note: ``"tail"`` and ``"disk"`` are absent -- those donors were
                dropped.  See ``_extract_jellyfish`` and ``_extract_halpha`` for
                re-sourcing guidance.
    """

    def __init__(
        self,
        repo_root: str | Path,
        kinds: Sequence[str] | None = None,
    ):
        root = Path(repo_root)
        lib_dir = root / _LIB
        manifest_path = lib_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Template library not found at {lib_dir}.\n"
                "Run build_template_library(repo_root) first."
            )

        with open(manifest_path) as fh:
            manifest = json.load(fh)

        # Filter by kind if requested
        if kinds is not None:
            kind_set = set(kinds)
            manifest = [m for m in manifest if m["kind"] in kind_set]

        if not manifest:
            raise ValueError(
                f"No templates found for kinds={kinds}. "
                "Check that build_template_library() has been run."
            )

        # Lazy-load: store paths and metadata; load arrays on demand.
        self._root = root
        self._manifest: list[dict] = manifest
        self._arrays: list[np.ndarray | None] = [None] * len(manifest)

    def __len__(self) -> int:
        return len(self._manifest)

    def __getitem__(self, i: int) -> tuple[np.ndarray, dict]:
        if self._arrays[i] is None:
            npy_path = self._root / self._manifest[i]["npy_path"]
            self._arrays[i] = np.load(str(npy_path))
        return self._arrays[i].copy(), dict(self._manifest[i])

    def sample(
        self,
        rng: np.random.Generator,
        kind: str | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Return a randomly selected template array and its metadata.

        Parameters
        ----------
        rng:   NumPy random generator.
        kind:  If given, restrict sampling to templates of this morphology kind.

        Returns
        -------
        (arr, meta)  --  float32 array unit-max normalised; dict with id/source/kind.
        """
        if kind is not None:
            idxs = [i for i, m in enumerate(self._manifest) if m["kind"] == kind]
            if not idxs:
                raise ValueError(f"No templates with kind={kind!r}")
            idx = idxs[int(rng.integers(0, len(idxs)))]
        else:
            idx = int(rng.integers(0, len(self._manifest)))
        return self[idx]


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_template_library(
    repo_root: str | Path,
    kinds: Sequence[str] | None = None,
) -> TemplateLibrary:
    """Load the template library from ``<repo_root>/data/morphology/lib/``.

    Raises :class:`FileNotFoundError` if the library has not been built yet.
    """
    return TemplateLibrary(repo_root, kinds=kinds)
