"""
tools/full_dist_model_render.py -- No-CASA model render for full T-RECS distribution.

Step 1 (fast, no CASA):
  - Load AGN + SFG catalogs with 10 uJy flux floor
  - Paint compact sources onto a 512x512 image at C-config cell (2.44 arcsec/px)
  - Build morphology via build_field()
  - Count sources
  - Render log-stretch PNG: data/morphology/lib/full_dist_model.png

Usage (from repo root):
    pixi run python tools/full_dist_model_render.py
"""
from __future__ import annotations

import gzip
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("full_dist_model")

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Parameters (must match corpus.py C-config smoke field)
# ---------------------------------------------------------------------------
VLA_CONFIG    = "C"
IMSIZE        = 512
FLUX_FLOOR_JY = 1e-5          # 10 uJy
TILE_CENTER   = (0.0, 0.0)    # catalog Euclidean coords are centred at 0,0
SEED          = 42

AGN_PATH = "/home/pjaganna/Software/radiosharp/data/agnsmedi.dat.gz"
SFG_PATH = "/home/pjaganna/Software/radiosharp/data/sfgsmedi.dat.gz"
FLOOR_COL = "I1400"  # mJy in catalog

OUT_PNG = _REPO_ROOT / "data" / "morphology" / "lib" / "full_dist_model.png"

# T-RECS fixed-width column specs (same as trecs.py)
_AGN_I1400_SPAN  = (112, 124)   # I1400 mJy, xcoord at (466,476), ycoord at (477,487)
_AGN_XCOORD_SPAN = (466, 476)
_AGN_YCOORD_SPAN = (477, 487)

_SFG_I1400_SPAN  = (98, 108)    # I1400 mJy in SFG
_SFG_XCOORD_SPAN = (379, 389)
_SFG_YCOORD_SPAN = (390, 400)


def _cell_arcsec_for_c() -> float:
    from casa_sim.corpus import _cell_arcsec_for_config
    return _cell_arcsec_for_config(VLA_CONFIG)


def _load_pop(path: str, i1400_span, x_span, y_span, label: str):
    """Load xcoord, ycoord, I1400_mJy from a fixed-width T-RECS catalog."""
    t0 = time.perf_counter()
    opener = gzip.open if path.endswith('.gz') else open
    xs, ys, fluxes = [], [], []
    with opener(path, 'rt') as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            try:
                xs.append(float(line[x_span[0]:x_span[1]].strip()))
                ys.append(float(line[y_span[0]:y_span[1]].strip()))
                fluxes.append(float(line[i1400_span[0]:i1400_span[1]].strip()))
            except (ValueError, IndexError):
                continue
    elapsed = time.perf_counter() - t0
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    fluxes = np.array(fluxes, dtype=np.float64)
    log.info("%s: %d sources loaded in %.1fs", label, len(xs), elapsed)
    return xs, ys, fluxes, elapsed


def _apply_cuts(xs, ys, flux_mjy, floor_jy, center, half_deg, label):
    """Apply flux floor + field spatial cut."""
    flux_jy = flux_mjy * 1e-3
    floor_mask = flux_jy >= floor_jy
    cx, cy = center
    spatial_mask = (np.abs(xs - cx) <= half_deg) & (np.abs(ys - cy) <= half_deg)
    mask = floor_mask & spatial_mask
    log.info("%s: %d survive flux floor %.0e Jy; %d in field",
             label, floor_mask.sum(), floor_jy, mask.sum())
    return xs[mask], ys[mask], flux_jy[mask]


def _paint_compact(xs, ys, flux_jy, cell_deg, imsize, img, label):
    """Paint compact sources as point pixels onto img (imsize x imsize)."""
    cx, cy = TILE_CENTER
    dx = xs - cx
    dy = ys - cy
    px = np.clip(np.round(imsize // 2 - dx / cell_deg).astype(int), 0, imsize - 1)
    py = np.clip(np.round(imsize // 2 + dy / cell_deg).astype(int), 0, imsize - 1)
    np.add.at(img, (px, py), flux_jy)
    log.info("%s: painted %d point sources", label, len(xs))
    return len(xs)


def _log_stretch(arr: np.ndarray, floor_frac: float = 1e-5) -> np.ndarray:
    arr = np.clip(arr, 0.0, None)
    peak = float(arr.max())
    if peak == 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    floor = peak * floor_frac
    arr = np.clip(arr, floor, peak)
    stretched = np.log10(arr / floor)
    log_range = np.log10(peak / floor)
    return (stretched / log_range).astype(np.float32)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell_arcsec = _cell_arcsec_for_c()
    cell_deg = cell_arcsec / 3600.0
    field_size_arcsec = IMSIZE * cell_arcsec
    half_deg = field_size_arcsec / 3600.0 / 2.0

    log.info("C-config: cell=%.4f arcsec, field=%.1f arcsec (%.4f deg half-width)",
             cell_arcsec, field_size_arcsec, half_deg)

    # ---- Load catalogs ----
    agn_xs, agn_ys, agn_flux_mjy, t_agn = _load_pop(
        AGN_PATH, _AGN_I1400_SPAN, _AGN_XCOORD_SPAN, _AGN_YCOORD_SPAN, "AGN")
    sfg_xs, sfg_ys, sfg_flux_mjy, t_sfg = _load_pop(
        SFG_PATH, _SFG_I1400_SPAN, _SFG_XCOORD_SPAN, _SFG_YCOORD_SPAN, "SFG")

    # ---- Apply cuts ----
    agn_xs_f, agn_ys_f, agn_jy = _apply_cuts(
        agn_xs, agn_ys, agn_flux_mjy, FLUX_FLOOR_JY, TILE_CENTER, half_deg, "AGN")
    sfg_xs_f, sfg_ys_f, sfg_jy = _apply_cuts(
        sfg_xs, sfg_ys, sfg_flux_mjy, FLUX_FLOOR_JY, TILE_CENTER, half_deg, "SFG")

    # ---- Paint compact sources ----
    img = np.zeros((IMSIZE, IMSIZE), dtype=np.float64)
    n_agn = _paint_compact(agn_xs_f, agn_ys_f, agn_jy, cell_deg, IMSIZE, img, "AGN")
    n_sfg = _paint_compact(sfg_xs_f, sfg_ys_f, sfg_jy, cell_deg, IMSIZE, img, "SFG")
    n_compact = n_agn + n_sfg

    log.info("Total compact sources painted: %d (AGN=%d, SFG=%d)", n_compact, n_agn, n_sfg)

    peak_compact = float(img.max())
    total_compact = float(img.sum())
    log.info("Compact: peak=%.4g Jy/px, total=%.4g Jy", peak_compact, total_compact)

    # ---- Extended morphology ----
    rng_morph = np.random.default_rng(SEED)
    from casa_sim.skymodel.morphology_field import build_field, FieldType
    morph_image, morph_meta = build_field(
        rng_morph,
        FieldType.DIFFUSE_DOMINANT,   # most visually interesting for render
        ra_deg=180.0, dec_deg=30.0,
        cell_arcsec=cell_arcsec,
        imsize=IMSIZE,
        freq_hz=1.4e9,
        repo_root=str(_REPO_ROOT),
    )
    ext_peak  = float(morph_image.max())
    ext_total = float(morph_image.sum())
    log.info("Extended: peak=%.4g Jy/px, total=%.4g Jy", ext_peak, ext_total)
    ratio = ext_peak / peak_compact if peak_compact > 0 else float("inf")
    log.info("Ratio extended_peak / compact_peak = %.3f  (target <1; compact should win)", ratio)

    # ---- Composite ----
    composite = img + morph_image.astype(np.float64)
    composite_peak = float(composite.max())
    log.info("Composite: peak=%.4g Jy/px, total=%.4g Jy", composite_peak, float(composite.sum()))

    # ---- Render: 2-panel figure ----
    #
    # Left panel:  composite image (log stretch) — full field overview.
    # Right panel: horizontal brightness profile through the image row with the
    #              highest peak, plotted in linear scale.  Both the compact-only
    #              and extended-only profiles are overlaid so the reader can see
    #              that bright point spikes sit above the diffuse floor.
    #
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # -- Left: composite image, log stretch tuned to show BOTH points and diffuse --
    #
    # Standard log stretch (floor_frac=1e-5) maps the diffuse (5e-5 Jy/px) to
    # ~0.17 in the [0,1] stretch -- visible but faint relative to the points.
    # We use a shallower floor_frac (1e-3) so the diffuse layer sits at mid-scale
    # while the bright points saturate the top of the colormap.  Points are still
    # clearly distinguishable as bright spots above the diffuse wash.
    stretch_composite = _log_stretch(composite, floor_frac=1e-3)
    ax0 = axes[0]
    ax0.imshow(stretch_composite.T, origin="lower", cmap="viridis",
               vmin=0.0, vmax=1.0, interpolation="nearest")
    ax0.set_title(
        f"Composite: compact + morphology (log stretch, floor=0.1% peak)\n"
        f"AGN={n_agn}  SFG={n_sfg}  total={n_compact} compact\n"
        f"floor={FLUX_FLOOR_JY*1e6:.0f} uJy  cell={cell_arcsec:.2f}\"/px\n"
        f"ext_total={ext_total:.3g} Jy  ext_peak={ext_peak:.3g} Jy/px\n"
        f"compact_peak={peak_compact:.3g} Jy/px  ratio={ratio:.4f}",
        fontsize=9
    )
    ax0.axis("off")

    # Choose the cut row: the row with the highest extended peak that also has
    # compact sources within +/-30 rows.  This guarantees the profile slice
    # shows BOTH diffuse emission AND point-source spikes in the same cut.
    # Strategy: rank rows by extended brightness; pick the highest-ranked row
    # that has at least one compact pixel within +-30 rows.
    ext_row_brightness = np.max(morph_image, axis=1)          # peak per row
    sorted_ext_rows = np.argsort(ext_row_brightness)[::-1]    # brightest first
    compact_rows_with_flux = set(np.where(np.max(img, axis=1) > 0)[0])

    cut_row = int(sorted_ext_rows[0])  # fallback: pure extended peak row
    for r in sorted_ext_rows[:50]:     # search top-50 extended rows
        # check within +/-30 rows for any compact pixel
        near = range(max(0, int(r) - 30), min(IMSIZE, int(r) + 31))
        if any(nr in compact_rows_with_flux for nr in near):
            cut_row = int(r)
            break

    # Row with the most compact flux (for the profile overlay)
    compact_row_peak = int(np.argmax(np.max(img, axis=1)))

    # Draw a horizontal guide line on the image at the chosen cut row
    ax0.axhline(y=cut_row, color="red", linewidth=0.8, alpha=0.7, linestyle="--")

    # -- Right: horizontal brightness profiles --
    ax1 = axes[1]
    x_px = np.arange(IMSIZE)

    profile_compact = img[:, cut_row]
    profile_ext     = morph_image[:, cut_row].astype(np.float64)
    profile_total   = composite[:, cut_row]

    # Compact peak row profile for reference (shows the brightest point source)
    profile_compact_peak_row = img[:, compact_row_peak]

    ax1.plot(x_px, profile_total,           color="white",  lw=1.2,
             label=f"composite (row {cut_row})")
    ax1.plot(x_px, profile_compact,         color="#4caf50", lw=1.0, alpha=0.9,
             label=f"compact (row {cut_row})")
    ax1.plot(x_px, profile_compact_peak_row, color="#a5d6a7", lw=0.8, alpha=0.6,
             linestyle="--", label=f"compact peak row ({compact_row_peak})")
    ax1.plot(x_px, profile_ext,             color="#2196f3", lw=1.0, alpha=0.8,
             label=f"extended (row {cut_row})")

    # Shade the diffuse envelope so it is visually obvious at this scale
    ax1.fill_between(x_px, 0, profile_ext, alpha=0.25, color="#2196f3")

    # Annotate the ratio at the extended peak pixel in this row
    ext_peak_this_row = float(profile_ext.max())
    cpt_peak_global   = float(img.max())
    row_ratio = ext_peak_this_row / cpt_peak_global if cpt_peak_global > 0 else 0.0

    ax1.set_xlabel("pixel (x)", fontsize=9)
    ax1.set_ylabel("flux [Jy/px]", fontsize=9)
    ax1.set_title(
        f"Brightness cut (row {cut_row})\n"
        f"compact peak (global) = {cpt_peak_global:.3g} Jy/px\n"
        f"extended peak (this row) = {ext_peak_this_row:.3g} Jy/px  |  "
        f"ratio = {row_ratio:.4f}",
        fontsize=9
    )
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_facecolor("#111111")
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#444444")
    ax1.yaxis.label.set_color("white")
    ax1.xaxis.label.set_color("white")
    ax1.title.set_color("white")

    fig.patch.set_facecolor("#1a1a1a")
    for ax in axes:
        ax.tick_params(colors="white")

    fig.suptitle(
        f"Full T-RECS distribution — VLA C-config  512x512  {cell_arcsec:.2f}\"/px\n"
        f"Catalog load: AGN {t_agn:.1f}s  SFG {t_sfg:.1f}s  |  "
        f"ext_peak/compact_peak = {ratio:.3f}",
        fontsize=11, color="white"
    )
    plt.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Model render saved: %s", OUT_PNG)

    # ---- Summary ----
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  Floor               : %.0f uJy", FLUX_FLOOR_JY * 1e6)
    log.info("  Populations         : AGN + SFG")
    log.info("  AGN in field        : %d", n_agn)
    log.info("  SFG in field        : %d", n_sfg)
    log.info("  Total compact       : %d", n_compact)
    log.info("  Compact peak        : %.4g Jy/px", peak_compact)
    log.info("  Extended total      : %.4g Jy", ext_total)
    log.info("  Extended peak       : %.4g Jy/px", ext_peak)
    log.info("  ext_peak/cpt_peak   : %.3f  (want << 1)", ratio)
    log.info("  AGN load time       : %.1f s", t_agn)
    log.info("  SFG load time       : %.1f s", t_sfg)
    log.info("  Render saved        : %s", OUT_PNG)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
