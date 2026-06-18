"""
corpus.py -- Batch driver for the CASA corpus simulation pipeline (M3).

Builds one or many corpus fields end-to-end through CASA:
    per-field sample  ->  build SimConfig  ->  run_single  ->  export FITS

Public API
----------
CorpusFieldSpec
    Dataclass holding all sampled parameters for one field.  Reproducible via
    field_seed.

sample_field_spec(global_rng, field_idx, ...) -> CorpusFieldSpec
    Draw one field's hyper-parameters from the corpus sampling distribution.

build_corpus_field(spec, work_dir, ...) -> CorpusFieldResult
    Build the SimConfig from a CorpusFieldSpec and call run_single.  Exports
    dirty + PSF + model to FITS (signed; no clipping).

CorpusFieldResult
    Dataclass with output FITS paths and timing.

Config-aware cell/FoV coupling (locked design decision)
---------------------------------------------------------
cell is Nyquist-sampled for each VLA config at L-band centre (1.5 GHz):

    lambda_1p5GHz = c / 1.5e9  ~ 0.200 m
    VLA baselines (max): A~36.4 km, B~11.1 km, C~3.4 km, D~1.0 km

    cell = lambda / (5 * B_max):
        A  ~ 0.23 arcsec/px
        B  ~ 0.75 arcsec/px
        C  ~ 2.44 arcsec/px   (≈ trecs config template: 2arcsec)
        D  ~ 8.26 arcsec/px

imsize is always 512 (fixed corpus size).  trecs.field_size_arcsec is
derived as imsize * cell so the T-RECS cut and morphology grid are always
consistent (M2 flag resolved here).

Track and noise sampling
-------------------------
track_ha_range: uniform from (snapshot 0h..0h) to (full ±4h).
                Represented as (ha_start, ha_stop) in hours.
integration_time: 60s for smoke-test speed; user may override.
simplenoise: drawn uniform in log space over [0.1, 10.0] mJy/bm.

Stokes
------
Stokes I only (sky_model.stokes = "I", SPW stokes = "RR LL") to keep the
corpus simple and fast.  T-RECS compact requires "IQUV" normally, but the
corpus_mix mode composites into Stokes I only -- so we use a reduced stokes
config.

Wait -- T-RECS requires IQUV per validation rule.  We use stokes="IQUV" for
sky_model and "RR RL LR LL" for the SPW (matching the working template).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VLA config geometry at L-band
# ---------------------------------------------------------------------------

# B_max in metres for each VLA config (approximate; used for cell derivation).
# Source: VLA observational status summary (NRAO).
_VLA_CONFIG_BMAX_M: dict[str, float] = {
    "A": 36_400.0,
    "B": 11_100.0,
    "C":  3_400.0,
    "D":  1_030.0,
}

_C_LIGHT = 2.99792458e8   # m/s
_LBAND_CENTRE_HZ = 1.5e9  # representative L-band centre


def _resolve_vla_cfg(vla_config: str) -> str:
    """Resolve a VLA antenna .cfg path via CASA's data tree (no hardcoded path).

    Uses casatools.ctsys.resolve so the simmos files are found wherever the
    active CASA data installation lives.
    """
    from casatools import ctsys
    rel = f"alma/simmos/vla.{vla_config.lower()}.cfg"
    path = ctsys.resolve(rel)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"VLA cfg file not resolvable: {rel}. "
            "Ensure CASA simmos data is installed (casaconfig / casadata)."
        )
    return path


def _trecs_catalog_paths() -> dict:
    """Return AGN/SFG T-RECS catalog paths from CASA_SIM_TRECS_DIR.

    Defaults to <repo>/data/trecs (the fetch-trecs default destination), so the
    corpus is portable: no absolute user paths baked in.
    """
    default_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "data", "trecs")
    trecs_dir = os.environ.get("CASA_SIM_TRECS_DIR", default_dir)
    return {
        "agn": os.path.join(trecs_dir, "agnsmedi.dat.gz"),
        "sfg": os.path.join(trecs_dir, "sfgsmedi.dat.gz"),
    }


def _cell_arcsec_for_config(vla_config: str) -> float:
    """Return Nyquist cell size in arcsec for a given VLA config at L-band centre.

    cell = lambda / (5 * B_max)  -- factor 5 gives ~5 pixels per fringe.
    """
    import astropy.units as u
    b_max = _VLA_CONFIG_BMAX_M[vla_config.upper()]
    lam = _C_LIGHT / _LBAND_CENTRE_HZ
    cell_rad = lam / (5.0 * b_max)
    return float((cell_rad * u.rad).to(u.arcsec).value)


# ---------------------------------------------------------------------------
# CorpusFieldSpec
# ---------------------------------------------------------------------------

@dataclass
class CorpusFieldSpec:
    """All sampled parameters for one corpus field.  Reproducible via field_seed."""

    field_idx: int          # global field index (used to name outputs)
    field_seed: int         # RNG seed for this field (morphology + T-RECS)

    vla_config: str         # A | B | C | D
    cell_arcsec: float      # derived from vla_config; stored for transparency
    imsize: int             # model + prediction grid size (e.g. 512)
    field_size_arcsec: float   # = imsize * cell_arcsec  (T-RECS + morphology FoV)

    ha_start: str           # CASA hour-angle string, e.g. "-2h"
    ha_stop: str            # e.g. "+2h"
    integration_time: str   # e.g. "60s"

    noise_jy: str           # simplenoise value, e.g. "0.003Jy"

    field_type: Optional[str] = None   # None → sampled by corpus_morphology per the balance


# ---------------------------------------------------------------------------
# CorpusFieldResult
# ---------------------------------------------------------------------------

@dataclass
class CorpusFieldResult:
    """Output paths and timing for one completed corpus field."""

    field_idx: int
    spec: CorpusFieldSpec

    dirty_fits: Optional[str] = None    # signed dirty (residual from niter=0 tclean)
    psf_fits: Optional[str] = None      # PSF
    model_fits: Optional[str] = None    # sky model image (Jy/pixel, non-negative)

    elapsed_s: float = 0.0
    success: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_field_spec(
    global_rng: np.random.Generator,
    field_idx: int,
    *,
    vla_configs: Sequence[str] = ("A", "B", "C", "D"),
    imsize: int = 512,
    ha_range_h: tuple[float, float] = (0.0, 4.0),   # min and max |HA| half-width in hours
    noise_jy_range: tuple[float, float] = (1e-4, 1e-2),   # log-uniform simplenoise range [Jy]
    integration_time: str = "60s",
    field_type: Optional[str] = None,   # None → sampled by corpus_morphology per balance
) -> CorpusFieldSpec:
    """Draw one field's hyper-parameters from the corpus sampling distribution.

    Parameters
    ----------
    global_rng:
        Master numpy Generator.  Advances state for each sample.
    field_idx:
        Integer index for this field (used in naming).
    vla_configs:
        Pool of VLA config labels to sample from (uniform).
    imsize:
        Fixed image size in pixels.
    ha_range_h:
        (min_half_width, max_half_width) in hours.  The HA track is symmetric:
        ha_start = -hw, ha_stop = +hw, where hw is drawn uniform in this range.
        A minimum of 0.0 gives a snapshot (ha_start = ha_stop = 0).
    noise_jy_range:
        (lo, hi) for log-uniform draw of simplenoise value in Jy.
    integration_time:
        Fixed integration time per dump (60s keeps smoke tests fast).
    field_type:
        If not None, fix the field type (string matching FieldType enum values).
        If None, the corpus_morphology block samples per balance.
    """
    # Derive a per-field seed so each field is independently reproducible.
    field_seed = int(global_rng.integers(0, 2**31 - 1))

    # VLA config
    cfg_idx = int(global_rng.integers(0, len(vla_configs)))
    vla_config = vla_configs[cfg_idx].upper()

    # Cell and field size
    cell_arcsec = _cell_arcsec_for_config(vla_config)
    field_size_arcsec = imsize * cell_arcsec

    # HA track: uniform half-width in [ha_range_h[0], ha_range_h[1]] hours
    ha_hw = float(global_rng.uniform(ha_range_h[0], ha_range_h[1]))
    ha_start = f"-{ha_hw:.3f}h"
    ha_stop = f"+{ha_hw:.3f}h"

    # Noise: log-uniform in Jy
    log_noise = float(global_rng.uniform(
        np.log10(noise_jy_range[0]),
        np.log10(noise_jy_range[1])
    ))
    noise_jy = float(10.0 ** log_noise)
    noise_str = f"{noise_jy:.6g}Jy"

    return CorpusFieldSpec(
        field_idx=field_idx,
        field_seed=field_seed,
        vla_config=vla_config,
        cell_arcsec=cell_arcsec,
        imsize=imsize,
        field_size_arcsec=field_size_arcsec,
        ha_start=ha_start,
        ha_stop=ha_stop,
        integration_time=integration_time,
        noise_jy=noise_str,
        field_type=field_type,
    )


# ---------------------------------------------------------------------------
# SimConfig builder
# ---------------------------------------------------------------------------

def _build_simconfig(spec: CorpusFieldSpec, work_dir: str, name: str) -> object:
    """Build a validated SimConfig from a CorpusFieldSpec.

    The config targets:
    - L-band 1-2 GHz, 8 channels (broad coverage, moderate data volume)
    - IQUV stokes (required by T-RECS corpus_mix validation)
    - simplenoise corruption
    - niter=0 tclean (dirty + PSF only; export_fits=True)
    """
    from .config import (
        CannedObservatory, CorpusMorphologyConfig, CorruptionConfig,
        FieldConfig, GainsConfig, ImagingConfig, NoiseConfig,
        ObservationConfig, ObservatoryConfig, ObserveCall,
        PredictionConfig, SimConfig, SkyModelConfig, SpwConfig,
        TRecsConfig, TRecsPolarizationConfig, TRecsSpectralConfig,
        validate_config, derive_imaging_params,
    )

    cfg_file = _resolve_vla_cfg(spec.vla_config)

    cell_str = f"{spec.cell_arcsec:.4g}arcsec"

    # Output prefix lives in work_dir
    sim_name = os.path.join(work_dir, name)

    observatory = ObservatoryConfig(
        mode="canned",
        feeds="perfect R L",
        canned=CannedObservatory(
            telescope="VLA",
            cfg_file=cfg_file,
        ),
    )

    observation = ObservationConfig(
        epoch="UTC 2026/06/15/03:00:00",
        integration_time=spec.integration_time,
        use_hourangle=True,
        fields=[FieldConfig(
            name="F1",
            direction="J2000 12h00m00s +30d00m00s",
        )],
        spws=[SpwConfig(
            name="L_band",
            freq="1.0GHz",
            deltafreq="125MHz",      # 8 channels x 125 MHz = 1 GHz bandwidth
            nchan=8,
            stokes="RR RL LR LL",   # full-pol to satisfy IQUV requirement
        )],
        observe_calls=[ObserveCall(
            field="F1",
            spw="L_band",
            start_time=spec.ha_start,
            stop_time=spec.ha_stop,
        )],
    )

    trecs = TRecsConfig(
        catalog_paths=_trecs_catalog_paths(),
        flux_floor_jy=1e-5,         # 10 uJy floor: full T-RECS distribution (AGN+SFG)
        flux_floor_col="I1400",
        field_size_arcsec=spec.field_size_arcsec,   # coupled to imsize * cell
        tile_center_deg=[0.0, 0.0],  # catalog Euclidean coords span -2.5..+2.5 deg
        seed=spec.field_seed,
        spectral=TRecsSpectralConfig(
            mode="trecs_sed",
            ref_freq="1.4GHz",
        ),
        polarization=TRecsPolarizationConfig(
            pol_fraction_source="trecs",
            pol_spidx_dist={"kind": "uniform", "low": -1.0, "high": 0.0},
            rm_dist={"kind": "uniform", "low": -100.0, "high": 100.0},
            chi0_dist={"kind": "uniform", "low": 0.0, "high": 3.141592653589793},
        ),
    )

    corpus_morphology = CorpusMorphologyConfig(
        field_type=spec.field_type,   # None → sampled from balance
        balance=None,                 # use DEFAULT_FIELD_TYPE_BALANCE
        seed=spec.field_seed,
    )

    sky_model = SkyModelConfig(
        stokes="IQUV",
        mode="corpus_mix",
        trecs=trecs,
        corpus_morphology=corpus_morphology,
    )

    prediction = PredictionConfig(
        gridder="standard",
        predictor="auto",
        normtype="flatsky",
        cell=cell_str,
        imsize=spec.imsize,
    )

    corruption = CorruptionConfig(
        seed=spec.field_seed,
        noise=NoiseConfig(
            enabled=True,
            mode="simplenoise",
            value=spec.noise_jy,
        ),
        gains=GainsConfig(enabled=False, mode="fbm"),
    )

    imaging = ImagingConfig(
        enabled=True,
        deconvolver="hogbom",
        nterms=None,
        pbcor=False,
        niter=0,          # dirty only
        export_fits=True, # write dirty.fits and psf.fits via sanity._export_fits
        pblimit=-1.0,     # disable PB masking (keeps full signed dirty)
        stokes="I",       # Stokes I continuum image (sky is IQUV for prediction only)
        specmode="mfs",   # MFS continuum across the 8 channels (not a cube)
        imsize=768,       # guard band: image 768 with the 512 source field centred
    )

    cfg = SimConfig(
        name=sim_name,
        observatory=observatory,
        observation=observation,
        sky_model=sky_model,
        prediction=prediction,
        corruption=corruption,
        imaging=imaging,
    )

    validate_config(cfg)
    cfg = derive_imaging_params(cfg)
    return cfg


# ---------------------------------------------------------------------------
# FITS export for the model image
# ---------------------------------------------------------------------------

def _export_model_fits(cfg, model_name: str) -> Optional[str]:
    """Export the sky model CASA image to FITS alongside the dirty/PSF.

    The corpus_mix sky model image is {cfg.name}_skymodel_corpus.im.
    cfg.name may contain directory components (e.g. ./smoketest or /abs/path).
    Returns the absolute FITS path, or None if the CASA image is not found.
    """
    from casatasks import exportfits

    # corpus_mix produces {name}_skymodel_corpus.im (see skymodel/core.py)
    # cfg.name is the sim_name, which when cwd=work_dir is just the base name prefix.
    casa_img = f"{cfg.name}_skymodel_corpus.im"
    fits_path = f"{cfg.name}_model.fits"
    # Resolve to absolute path (cwd is work_dir at call time)
    fits_path_abs = os.path.abspath(fits_path)

    if not os.path.exists(casa_img):
        log.warning("[corpus] model image not found at %s — skipping model export", casa_img)
        return None

    exportfits(imagename=casa_img, fitsimage=fits_path_abs,
               overwrite=True, dropdeg=True, stokeslast=False)
    log.info("[corpus] model exported: %s -> %s", casa_img, fits_path_abs)
    return fits_path_abs


# ---------------------------------------------------------------------------
# Main field builder
# ---------------------------------------------------------------------------

def build_corpus_field(
    spec: CorpusFieldSpec,
    work_dir: str,
    *,
    name: Optional[str] = None,
) -> CorpusFieldResult:
    """Build a single corpus field end-to-end through CASA.

    Steps:
      1. Build SimConfig from spec (cell/FoV coupled; corpus_mix sky model).
      2. Call run_single (stages 1-8): observatory -> observation -> sky ->
         predict -> corrupt -> sanity_image (niter=0) -> write_outputs.
      3. Export model FITS (dirty+PSF already exported via imaging.export_fits).

    The dirty image is the niter=0 tclean residual: SIGNED, sidelobes present,
    thermal noise present.  It is written by sanity._export_fits to
    {name}_dirty.fits.  Do NOT clip it.

    Parameters
    ----------
    spec:
        CorpusFieldSpec from sample_field_spec().
    work_dir:
        Directory in which all CASA outputs are written.  Created if absent.
    name:
        Output name prefix (default: corpus_field_{field_idx:04d}).

    Returns
    -------
    CorpusFieldResult with FITS paths populated on success.
    """
    from .simulate import run_single

    os.makedirs(work_dir, exist_ok=True)

    if name is None:
        name = f"corpus_field_{spec.field_idx:04d}"

    result = CorpusFieldResult(field_idx=spec.field_idx, spec=spec)
    t0 = time.perf_counter()

    # Change cwd to work_dir so CASA writes all files there
    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)

        cfg = _build_simconfig(spec, work_dir=".", name=name)

        log.info(
            "[corpus] field %04d | config=%s cell=%.3farcsec "
            "imsize=%d field_size=%.1farcsec ha=%s..%s noise=%s",
            spec.field_idx, spec.vla_config, spec.cell_arcsec, spec.imsize,
            spec.field_size_arcsec, spec.ha_start, spec.ha_stop, spec.noise_jy,
        )

        run_single(cfg)

        # Collect FITS outputs
        dirty_candidate = f"{name}_dirty.fits"
        psf_candidate = f"{name}_psf.fits"
        model_candidate = _export_model_fits(cfg, name)

        result.dirty_fits = os.path.join(work_dir, dirty_candidate) if os.path.exists(dirty_candidate) else None
        result.psf_fits = os.path.join(work_dir, psf_candidate) if os.path.exists(psf_candidate) else None
        # model_candidate may be a relative path (written while cwd=work_dir); resolve it
        if model_candidate is not None:
            model_abs = os.path.join(work_dir, os.path.basename(model_candidate))
            result.model_fits = model_abs if os.path.exists(model_abs) else None
        else:
            result.model_fits = None

        if result.dirty_fits is None:
            log.warning("[corpus] dirty FITS not found at %s", dirty_candidate)
        if result.psf_fits is None:
            log.warning("[corpus] PSF FITS not found at %s", psf_candidate)

        result.success = True

    except Exception as exc:
        result.error = str(exc)
        log.error("[corpus] field %04d FAILED: %s", spec.field_idx, exc, exc_info=True)

    finally:
        os.chdir(original_cwd)
        result.elapsed_s = time.perf_counter() - t0

    log.info(
        "[corpus] field %04d done in %.1fs | success=%s dirty=%s psf=%s model=%s",
        spec.field_idx, result.elapsed_s, result.success,
        result.dirty_fits, result.psf_fits, result.model_fits,
    )
    return result


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def run_corpus_batch(
    n_fields: int,
    work_dir: str,
    *,
    seed: int = 0,
    vla_configs: Sequence[str] = ("A", "B", "C", "D"),
    imsize: int = 512,
    ha_range_h: tuple[float, float] = (0.0, 4.0),
    noise_jy_range: tuple[float, float] = (1e-4, 1e-2),
    integration_time: str = "60s",
    field_type: Optional[str] = None,
    patchify_out_dir: Optional[str] = None,
    patch_size: int = 128,
) -> list[CorpusFieldResult]:
    """Run the corpus batch driver for n_fields fields sequentially.

    Parameters
    ----------
    n_fields:
        How many fields to simulate.
    work_dir:
        Root output directory.  Each field writes into work_dir directly
        (prefixed by field index).
    seed:
        Master RNG seed for reproducible sampling.
    vla_configs:
        Pool of VLA config labels (sampled uniformly).
    imsize:
        Fixed image size in pixels (always 512 for the corpus).
    ha_range_h:
        (min, max) HA half-width in hours.
    noise_jy_range:
        (lo, hi) log-uniform simplenoise range in Jy.
    integration_time:
        CASA integration time per dump.
    field_type:
        Fixed field type override (None -> sample from balance per field).
    patchify_out_dir:
        If given, patchify all successful fields and write memmap stacks here.
        If None, patchify is skipped.
    patch_size:
        Patch edge length in pixels (default 128).  Only used when
        patchify_out_dir is set.

    Returns
    -------
    List of CorpusFieldResult, one per field.
    """
    global_rng = np.random.default_rng(seed)
    results = []

    for idx in range(n_fields):
        spec = sample_field_spec(
            global_rng,
            field_idx=idx,
            vla_configs=vla_configs,
            imsize=imsize,
            ha_range_h=ha_range_h,
            noise_jy_range=noise_jy_range,
            integration_time=integration_time,
            field_type=field_type,
        )
        result = build_corpus_field(spec, work_dir=work_dir)
        results.append(result)

    n_ok = sum(r.success for r in results)
    log.info("[corpus] batch complete: %d/%d fields succeeded", n_ok, n_fields)

    if patchify_out_dir is not None:
        from .patchify import patchify_results
        log.info("[corpus] patchifying %d successful fields → %s", n_ok, patchify_out_dir)
        patchify_results(results, patchify_out_dir, patch_size=patch_size)

    return results
