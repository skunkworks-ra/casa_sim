from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from .utils import _C_LIGHT, _get_chan_freqs_from_csys, _get_stokes_indices_from_csys

log = logging.getLogger(__name__)

# Catalog sampled frequencies in Hz (shared by both AGN and SFG formats)
_TRECS_FREQS_HZ = np.array([
    150e6, 160e6, 220e6, 300e6, 410e6, 560e6, 780e6, 1000e6,
    1400e6, 1900e6, 2700e6, 3000e6, 3600e6, 5000e6, 6700e6,
    9200e6, 12500e6, 20000e6,
])

_AGN_I_COLS = ['I150','I160','I220','I300','I410','I560','I780','I1000',
               'I1400','I1900','I2700','I3000','I3600','I5000','I6700',
               'I9200','I12500','I20000']
_AGN_P_COLS = ['P150','P160','P220','P300','P410','P560','P780','P1000',
               'P1400','P1900','P2700','P3000','P3600','P5000','P6700',
               'P9200','P12500','P20000']

_AGN_COLSPECS = [
    (0,  7),    # Lum1400
    (8,  20),   # I150
    (21, 33),   # I160
    (34, 46),   # I220
    (47, 59),   # I300
    (60, 72),   # I410
    (73, 85),   # I560
    (86, 98),   # I780
    (99, 111),  # I1000
    (112,124),  # I1400
    (125,137),  # I1900
    (138,150),  # I2700
    (151,163),  # I3000
    (164,176),  # I3600
    (177,189),  # I5000
    (190,202),  # I6700
    (203,215),  # I9200
    (216,228),  # I12500
    (229,241),  # I20000
    (242,253),  # P150
    (254,265),  # P160
    (266,277),  # P220
    (278,289),  # P300
    (290,301),  # P410
    (302,313),  # P560
    (314,325),  # P780
    (326,337),  # P1000
    (338,349),  # P1400
    (350,361),  # P1900
    (362,373),  # P2700
    (374,385),  # P3000
    (386,397),  # P3600
    (398,409),  # P5000
    (410,421),  # P6700
    (422,433),  # P9200
    (434,445),  # P12500
    (446,457),  # P20000
    (458,465),  # logMh
    (466,476),  # xcoord
    (477,487),  # ycoord
    (488,498),  # GLAT
    (499,509),  # GLON
    (510,518),  # z
    (519,536),  # physSize
    (537,552),  # angle
    (553,571),  # Size
    (572,581),  # Rs
    (582,583),  # PopFlag
]
_AGN_COLNAMES = [
    'Lum1400',
    'I150','I160','I220','I300','I410','I560','I780','I1000',
    'I1400','I1900','I2700','I3000','I3600','I5000','I6700','I9200','I12500','I20000',
    'P150','P160','P220','P300','P410','P560','P780','P1000',
    'P1400','P1900','P2700','P3000','P3600','P5000','P6700','P9200','P12500','P20000',
    'logMh','xcoord','ycoord','GLAT','GLON','z','physSize','angle','Size','Rs','PopFlag',
]

_SFG_COLSPECS = [
    (0,  9),    # logSFR
    (10, 20),   # I150
    (21, 31),   # I160
    (32, 42),   # I220
    (43, 53),   # I300
    (54, 64),   # I410
    (65, 75),   # I560
    (76, 86),   # I780
    (87, 97),   # I1000
    (98, 108),  # I1400
    (109,119),  # I1900
    (120,130),  # I2700
    (131,141),  # I3000
    (142,152),  # I3600
    (153,163),  # I5000
    (164,174),  # I6700
    (175,185),  # I9200
    (186,196),  # I12500
    (197,207),  # I20000
    (208,216),  # P150
    (217,225),  # P160
    (226,234),  # P220
    (235,243),  # P300
    (244,252),  # P410
    (253,261),  # P560
    (262,270),  # P780
    (271,279),  # P1000
    (280,288),  # P1400
    (289,297),  # P1900
    (298,306),  # P2700
    (307,315),  # P3000
    (316,324),  # P3600
    (325,333),  # P5000
    (334,342),  # P6700
    (343,351),  # P9200
    (352,360),  # P12500
    (361,369),  # P20000
    (370,378),  # logMh
    (379,389),  # xcoord
    (390,400),  # ycoord
    (401,411),  # GLAT
    (412,422),  # GLON
    (423,431),  # z
    (432,450),  # Size
    (451,461),  # e1
    (462,472),  # e2
    (473,474),  # PopFlag
]
_SFG_COLNAMES = [
    'logSFR',
    'I150','I160','I220','I300','I410','I560','I780','I1000',
    'I1400','I1900','I2700','I3000','I3600','I5000','I6700','I9200','I12500','I20000',
    'P150','P160','P220','P300','P410','P560','P780','P1000',
    'P1400','P1900','P2700','P3000','P3600','P5000','P6700','P9200','P12500','P20000',
    'logMh','xcoord','ycoord','GLAT','GLON','z','Size','e1','e2','PopFlag',
]


def _draw_n_components(n_rm_components_cfg, n_src: int, rng) -> np.ndarray:
    if isinstance(n_rm_components_cfg, int):
        return np.full(n_src, n_rm_components_cfg, dtype=int)
    lam = float(n_rm_components_cfg['lam'])
    return np.maximum(1, rng.poisson(lam, size=n_src))


def _build_trecs_sky_model(cfg, trecs_cfg, out_path: str, ia, qa,
                            msname: str = None, tb=None) -> None:
    """
    Build a full-Stokes IQUV CASA image from T-RECS catalog data.
    """
    from ..config import _parse_freq_to_hz
    from .core import _make_empty_image, _read_ms_chan_freqs

    trecs = trecs_cfg
    rng = np.random.default_rng(trecs.seed)

    all_frames = []
    for pop, path in trecs.catalog_paths.items():
        log.info("[skymodel/trecs] Loading %s catalog: %s", pop.upper(), path)
        df = _load_trecs_catalog(
            path, pop,
            flux_floor_jy=trecs.flux_floor_jy,
            flux_floor_col=trecs.flux_floor_col,
            field_half_deg=trecs.field_size_arcsec / 3600.0 / 2.0,
            tile_center_deg=trecs.tile_center_deg,
        )
        n_loaded = len(df['xcoord'])
        log.info("[skymodel/trecs] %s: %d sources loaded", pop.upper(), n_loaded)

        floor_col = trecs.flux_floor_col
        flux_jy = df[floor_col] * 1e-3
        flux_mask = flux_jy >= trecs.flux_floor_jy
        df = {k: v[flux_mask] for k, v in df.items()}
        log.info("[skymodel/trecs] %s: %d sources above flux floor %.1e Jy at %s",
                 pop.upper(), len(df['xcoord']), trecs.flux_floor_jy, floor_col)

        half_deg = trecs.field_size_arcsec / 3600.0 / 2.0
        cx, cy = trecs.tile_center_deg
        dx = df['xcoord'] - cx
        dy = df['ycoord'] - cy
        spatial_mask = (np.abs(dx) <= half_deg) & (np.abs(dy) <= half_deg)
        df = {k: v[spatial_mask] for k, v in df.items()}
        df['_dx_deg'] = dx[spatial_mask]
        df['_dy_deg'] = dy[spatial_mask]
        df['_pop'] = pop
        n_sel = len(df['xcoord'])
        log.info("[skymodel/trecs] %s: %d sources in field (%.0f arcsec square)",
                 pop.upper(), n_sel, trecs.field_size_arcsec)
        all_frames.append(df)

    if not all_frames or all(len(df['xcoord']) == 0 for df in all_frames):
        raise RuntimeError(
            "T-RECS: no sources survive flux floor + field cut. "
            "Lower flux_floor_jy or increase field_size_arcsec."
        )

    _make_empty_image(cfg, ia, qa, out_path, msname, tb)
    log.info("[skymodel/trecs] Empty image created: %s", out_path)

    ia.open(out_path)
    csys = ia.coordsys()
    shp = ia.shape()   # [nx, ny, 4, nchan]
    nx, ny, nstokes, nchan = shp

    stokes_idx = _get_stokes_indices_from_csys(csys)
    i_idx = stokes_idx.get('I', 0)
    q_idx = stokes_idx.get('Q', 1)
    u_idx = stokes_idx.get('U', 2)

    img_data = ia.getchunk()
    ia.close()

    if msname is not None and tb is not None:
        chan_freqs_hz = _read_ms_chan_freqs(msname, tb)
    else:
        chan_freqs_hz = _get_chan_freqs_from_csys(csys, nchan)

    ref_freq_hz = _parse_freq_to_hz(trecs.spectral.ref_freq)
    lam_sq_ref = (_C_LIGHT / ref_freq_hz) ** 2
    lam_sq_chans = (_C_LIGHT / chan_freqs_hz) ** 2

    cell_str = cfg.effective_cell
    cell_deg = float(qa.convert(qa.quantity(cell_str), 'deg')['value'])

    for df in all_frames:
        n_src = len(df['xcoord'])
        if n_src == 0:
            continue

        dx_deg = df['_dx_deg']
        dy_deg = df['_dy_deg']
        px_arr = np.clip(
            np.round(nx // 2 - dx_deg / cell_deg).astype(int), 0, nx - 1
        )
        py_arr = np.clip(
            np.round(ny // 2 + dy_deg / cell_deg).astype(int), 0, ny - 1
        )

        I_cat_jy = np.column_stack([df[c] * 1e-3 for c in _AGN_I_COLS])
        P_cat_jy = np.column_stack([df[c] * 1e-3 for c in _AGN_P_COLS])

        if trecs.polarization.pol_fraction_source == 'trecs':
            I_ref = _interp_one_freq(I_cat_jy, ref_freq_hz)
            P_ref = _interp_one_freq(P_cat_jy, ref_freq_hz)
            p_ref = np.where(I_ref > 1e-30, P_ref / I_ref, 0.0)
        else:
            p_ref = _sample_dist(trecs.polarization.pol_fraction_dist, n_src, rng)

        if trecs.polarization.pol_spidx_dist is not None:
            pol_spidx = _sample_dist(trecs.polarization.pol_spidx_dist, n_src, rng)
        else:
            pol_spidx = np.zeros(n_src)

        I_chans = _interp_sed_to_channels(I_cat_jy, chan_freqs_hz)

        nu_ratio = chan_freqs_hz[np.newaxis, :] / ref_freq_hz
        p_chan = p_ref[:, np.newaxis] * nu_ratio ** pol_spidx[:, np.newaxis]

        delta_lam_sq = lam_sq_chans[np.newaxis, :] - lam_sq_ref

        n_comps_arr = _draw_n_components(trecs.polarization.n_rm_components, n_src, rng)
        max_comps = int(n_comps_arr.max())

        P_complex = np.zeros((n_src, nchan), dtype=np.complex128)
        for k in range(max_comps):
            active = n_comps_arr > k
            RM_k   = _sample_dist(trecs.polarization.rm_dist,   n_src, rng)
            chi0_k = _sample_dist(trecs.polarization.chi0_dist, n_src, rng)
            p_k_chan = p_chan / n_comps_arr[:, np.newaxis]
            chi_k = chi0_k[:, np.newaxis] + RM_k[:, np.newaxis] * delta_lam_sq
            P_complex += active[:, np.newaxis] * p_k_chan * I_chans * np.exp(2j * chi_k)

        Q_chans = P_complex.real
        U_chans = P_complex.imag

        for k in range(nchan):
            np.add.at(img_data[:, :, i_idx, k], (px_arr, py_arr), I_chans[:, k])
            np.add.at(img_data[:, :, q_idx, k], (px_arr, py_arr), Q_chans[:, k])
            np.add.at(img_data[:, :, u_idx, k], (px_arr, py_arr), U_chans[:, k])

        log.info("[skymodel/trecs] %s: painted %d sources onto image", df['_pop'].upper(), n_src)

    ia.open(out_path)
    ia.putchunk(img_data)
    ia.close()
    log.info("[skymodel/trecs] Sky model image written: %s", out_path)


def _load_trecs_catalog(
    path: str,
    pop: str,
    *,
    flux_floor_jy: Optional[float] = None,
    flux_floor_col: Optional[str] = None,
    field_half_deg: Optional[float] = None,
    tile_center_deg: Optional[Sequence[float]] = None,
) -> dict:
    """Load a T-RECS catalog from a fixed-width .dat or .dat.gz file.

    The SFG catalog is multi-GB; reading it whole into memory OOMs the machine.
    When the flux-floor and field-cut selection parameters are supplied, they are
    applied *during* the streaming read: only the three filter columns are parsed
    per line, and the full row is materialised solely for sources that survive
    both cuts.  This collapses peak memory from "entire catalog" to "selected
    sources" while reproducing the downstream masks exactly.
    """
    import gzip

    colspecs = _AGN_COLSPECS if pop.lower() == 'agn' else _SFG_COLSPECS
    colnames = _AGN_COLNAMES if pop.lower() == 'agn' else _SFG_COLNAMES

    do_filter = (flux_floor_jy is not None and flux_floor_col is not None
                 and field_half_deg is not None and tile_center_deg is not None)
    if do_filter:
        flux_idx = colnames.index(flux_floor_col)
        x_idx = colnames.index('xcoord')
        y_idx = colnames.index('ycoord')
        cx, cy = tile_center_deg

    def _parse(idx, line):
        start, end = colspecs[idx]
        try:
            return float(line[start:end].strip())
        except ValueError:
            return np.nan

    opener = gzip.open if path.endswith('.gz') else open

    rows = []
    with opener(path, 'rt') as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            if do_filter:
                # Cheap pre-filter: parse only the selection columns first.
                flux_jy = _parse(flux_idx, line) * 1e-3
                if not (flux_jy >= flux_floor_jy):
                    continue
                if (abs(_parse(x_idx, line) - cx) > field_half_deg
                        or abs(_parse(y_idx, line) - cy) > field_half_deg):
                    continue
            rows.append([_parse(i, line) for i in range(len(colspecs))])

    if not rows:
        raise RuntimeError(f"T-RECS catalog is empty or unreadable: {path}")

    arr = np.array(rows, dtype=np.float64)
    return {name: arr[:, i] for i, name in enumerate(colnames)}


def _interp_sed_to_channels(I_cat_jy: np.ndarray, chan_freqs_hz: np.ndarray) -> np.ndarray:
    """
    Log-log interpolate SED from catalog frequencies to MS channel grid.

    Args:
        I_cat_jy:    [n_src, 18] flux density in Jy at _TRECS_FREQS_HZ
        chan_freqs_hz: [nchan] channel center frequencies in Hz

    Returns:
        I_chans: [n_src, nchan] in Jy
    """
    log_cat = np.log10(_TRECS_FREQS_HZ)
    log_chan = np.log10(chan_freqs_hz)
    tiny = 1e-30

    n_src = I_cat_jy.shape[0]
    nchan = len(chan_freqs_hz)
    I_chans = np.zeros((n_src, nchan), dtype=np.float64)

    log_I = np.log10(np.maximum(I_cat_jy, tiny))

    for i in range(n_src):
        I_chans[i] = 10.0 ** np.interp(log_chan, log_cat, log_I[i])

    return I_chans


def _interp_one_freq(I_cat_jy: np.ndarray, freq_hz: float) -> np.ndarray:
    """Interpolate each source SED to a single frequency. Returns [n_src]."""
    log_cat = np.log10(_TRECS_FREQS_HZ)
    log_freq = np.log10(freq_hz)
    tiny = 1e-30
    log_I = np.log10(np.maximum(I_cat_jy, tiny))
    return 10.0 ** np.array([np.interp(log_freq, log_cat, log_I[i])
                              for i in range(I_cat_jy.shape[0])])


def _sample_dist(dist_cfg: dict, n: int, rng) -> np.ndarray:
    """Draw n samples from a distribution config dict."""
    if dist_cfg is None:
        return np.zeros(n)
    kind = dist_cfg.get('kind', 'uniform')
    if kind == 'uniform':
        return rng.uniform(float(dist_cfg['low']), float(dist_cfg['high']), size=n)
    if kind == 'normal':
        return rng.normal(float(dist_cfg['mean']), float(dist_cfg['std']), size=n)
    raise ValueError(f"Unknown distribution kind '{kind}'. Supported: uniform, normal")
