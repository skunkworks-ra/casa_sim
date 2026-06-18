from __future__ import annotations

import logging
import os
from typing import Any

from .dataclasses import ConfigError, SimConfig

log = logging.getLogger(__name__)


def _is_valid_path(value: Any) -> bool:
    """Return True if value is a non-empty string (path existence not checked here)."""
    return isinstance(value, str) and len(value.strip()) > 0


def validate_config(cfg: SimConfig) -> None:
    """
    Enforce all validation rules from Section 7.
    Raises ConfigError on hard errors.
    Logs warnings via logging module.
    """
    from .routing import _resolve_predictor_auto

    sm = cfg.sky_model
    pred = cfg.prediction
    obs = cfg.observation
    corr = cfg.corruption
    img = cfg.imaging

    # ---- Predictor / gridder compatibility --------------------------------

    effective_predictor = _resolve_predictor_auto(cfg) if pred.predictor == 'auto' else pred.predictor

    if (effective_predictor == 'ft_dft'
            and pred.gridder in ('mosaic', 'awproject')):
        raise ConfigError(
            "incompatible_predictor_gridder",
            "prediction.predictor",
            "ft_dft predictor incompatible with mosaic/awproject gridder"
        )

    if (effective_predictor == 'sm_predict'
            and pred.gridder in ('mosaic', 'awproject')):
        log.warning("[validation] prediction.predictor: sm_predict demoted to tclean "
                    "for gridder=%s", pred.gridder)

    # ---- Sky model --------------------------------------------------------

    if sm.mode == 'image_extrapolate' and not sm.ref_freq:
        raise ConfigError(
            "missing_ref_freq",
            "sky_model.ref_freq",
            "ref_freq required for image_extrapolate mode"
        )

    if (sm.mode == 'image_extrapolate'
            and sm.alpha_mode == 'map'
            and not _is_valid_path(sm.alpha_value)):
        raise ConfigError(
            "invalid_alpha_map_path",
            "sky_model.alpha_value",
            "alpha_value must be a valid image path for alpha_mode=map"
        )

    # ---- Sources (inline component list) -----------------------------------

    if sm.sources and sm.cl_path:
        raise ConfigError(
            "sources_and_cl_path_exclusive",
            "sky_model",
            "sources and cl_path are mutually exclusive — use one or the other"
        )

    if sm.sources:
        if sm.mode != 'component_list':
            raise ConfigError(
                "sources_require_component_list_mode",
                "sky_model.mode",
                "sources require mode: component_list"
            )
        for i, src in enumerate(sm.sources):
            if len(src.flux) not in (1, 4):
                raise ConfigError(
                    "invalid_flux_length",
                    f"sky_model.sources[{i}].flux",
                    f"flux must have 1 (I) or 4 (IQUV) elements, got {len(src.flux)}"
                )
            if len(src.spectral_index) not in (1, 2):
                raise ConfigError(
                    "invalid_spectral_index_length",
                    f"sky_model.sources[{i}].spectral_index",
                    f"spectral_index must have 1 (alpha) or 2 (alpha, beta) elements, "
                    f"got {len(src.spectral_index)}"
                )
            if src.frac_pol is not None and len(src.flux) == 4:
                raise ConfigError(
                    "frac_pol_with_iquv_flux",
                    f"sky_model.sources[{i}]",
                    "frac_pol cannot be used with explicit IQUV flux — use one or the other"
                )
            if src.frac_pol is not None and src.chi is None:
                raise ConfigError(
                    "frac_pol_requires_chi",
                    f"sky_model.sources[{i}]",
                    "chi (EVPA in degrees) is required when frac_pol is specified"
                )
            if src.shape in ('gaussian', 'disk'):
                if not src.major or not src.minor:
                    raise ConfigError(
                        "missing_shape_params",
                        f"sky_model.sources[{i}]",
                        f"major and minor required for shape={src.shape}"
                    )

    # ---- Faraday ----------------------------------------------------------

    if sm.faraday and sm.faraday.enabled:
        if sm.stokes != 'IQUV':
            raise ConfigError(
                "faraday_requires_iquv",
                "sky_model.stokes",
                "Faraday rotation requires stokes: IQUV"
            )
        if sm.faraday.rm_mode == 'map' and not _is_valid_path(sm.faraday.rm_value):
            raise ConfigError(
                "invalid_rm_map_path",
                "sky_model.faraday.rm_value",
                "rm_value must be a valid image path for rm_mode=map"
            )
        if not sm.faraday.ref_freq:
            raise ConfigError(
                "missing_faraday_ref_freq",
                "sky_model.faraday.ref_freq",
                "faraday.ref_freq required when faraday.enabled is true"
            )

    # ---- T-RECS -----------------------------------------------------------

    _TRECS_MODES = ('t_recs', 'corpus_mix')

    if sm.mode in _TRECS_MODES:
        if sm.trecs is None:
            raise ConfigError("missing_trecs_block", "sky_model.trecs",
                              f"sky_model.mode={sm.mode} requires a trecs: block")
        if sm.stokes != 'IQUV':
            raise ConfigError("trecs_requires_iquv", "sky_model.stokes",
                              f"{sm.mode} mode requires stokes: IQUV")
        for pop, path in sm.trecs.catalog_paths.items():
            if not _is_valid_path(path):
                raise ConfigError("missing_trecs_catalog", f"sky_model.trecs.catalog_paths.{pop}",
                                  f"catalog path for '{pop}' is empty")
            if not os.path.exists(path):
                raise ConfigError("trecs_catalog_not_found", f"sky_model.trecs.catalog_paths.{pop}",
                                  f"catalog file not found: {path}")
            if 'sfg' in pop.lower():
                log.warning(
                    "[validation] T-RECS SFG catalog requested (%s). "
                    "The medium SFG catalog has ~28 million sources and will take "
                    "several minutes to load and filter. Consider using AGN only "
                    "for initial testing.", path
                )
        if sm.trecs.flux_floor_jy <= 0:
            raise ConfigError("invalid_flux_floor", "sky_model.trecs.flux_floor_jy",
                              "flux_floor_jy must be positive")
        if sm.trecs.field_size_arcsec <= 0:
            raise ConfigError("invalid_field_size", "sky_model.trecs.field_size_arcsec",
                              "field_size_arcsec must be positive")
        if sm.trecs.polarization.rm_dist is None:
            raise ConfigError("missing_rm_dist", "sky_model.trecs.polarization.rm_dist",
                              "rm_dist is required for t_recs polarization")
        if sm.trecs.polarization.chi0_dist is None:
            raise ConfigError("missing_chi0_dist", "sky_model.trecs.polarization.chi0_dist",
                              "chi0_dist is required for t_recs polarization")
        n_rmc = sm.trecs.polarization.n_rm_components
        if isinstance(n_rmc, int):
            if n_rmc < 1:
                raise ConfigError("invalid_n_rm_components",
                                  "sky_model.trecs.polarization.n_rm_components",
                                  "n_rm_components must be >= 1")
        elif isinstance(n_rmc, dict):
            if n_rmc.get('kind') != 'poisson' or 'lam' not in n_rmc:
                raise ConfigError("invalid_n_rm_components_dist",
                                  "sky_model.trecs.polarization.n_rm_components",
                                  "n_rm_components dict must be {kind: poisson, lam: <float>}")
        else:
            raise ConfigError("invalid_n_rm_components_type",
                              "sky_model.trecs.polarization.n_rm_components",
                              "n_rm_components must be int or {kind: poisson, lam: float}")

    # ---- corpus_mix -specific checks (morphology block) -------------------

    if sm.mode == 'corpus_mix':
        if sm.corpus_morphology is None:
            raise ConfigError(
                "missing_corpus_morphology_block",
                "sky_model.corpus_morphology",
                "sky_model.mode=corpus_mix requires a corpus_morphology: block"
            )
        cm = sm.corpus_morphology
        _VALID_FIELD_TYPES = {
            'point_only', 'diffuse_dominant',
            'central_shell_flow', 'fully_diffuse',
        }
        if cm.field_type is not None and cm.field_type not in _VALID_FIELD_TYPES:
            raise ConfigError(
                "invalid_corpus_field_type",
                "sky_model.corpus_morphology.field_type",
                f"field_type must be one of {sorted(_VALID_FIELD_TYPES)} or null; "
                f"got '{cm.field_type}'"
            )
        if cm.balance is not None:
            missing = _VALID_FIELD_TYPES - set(cm.balance.keys())
            if missing:
                raise ConfigError(
                    "incomplete_corpus_balance",
                    "sky_model.corpus_morphology.balance",
                    f"balance dict missing keys: {sorted(missing)}. "
                    "All four field types must be present."
                )
            if any(v < 0.0 for v in cm.balance.values()):
                raise ConfigError(
                    "negative_corpus_balance_weight",
                    "sky_model.corpus_morphology.balance",
                    "All balance weights must be non-negative."
                )

    # ---- Spectral lines ---------------------------------------------------

    for i, line in enumerate(sm.spectral_lines):
        if line.mode == 'image' and not line.image_path:
            raise ConfigError(
                "missing_line_image_path",
                f"sky_model.spectral_lines[{i}].image_path",
                "image_path required for spectral line image mode"
            )
        if len(line.flux_profile) != len(line.channels):
            raise ConfigError(
                "flux_profile_length_mismatch",
                f"sky_model.spectral_lines[{i}].flux_profile",
                f"flux_profile length ({len(line.flux_profile)}) must match "
                f"channels length ({len(line.channels)})"
            )

    # ---- Observe calls reference valid field/spw names --------------------

    field_names = {f.name for f in obs.fields}
    spw_names = {s.name for s in obs.spws}

    for i, call in enumerate(obs.observe_calls):
        if call.field not in field_names:
            raise ConfigError(
                "undefined_field_reference",
                f"observation.observe_calls[{i}].field",
                f"observe_call references undefined field name '{call.field}'"
            )
        if call.spw not in spw_names:
            raise ConfigError(
                "undefined_spw_reference",
                f"observation.observe_calls[{i}].spw",
                f"observe_call references undefined spw name '{call.spw}'"
            )

    # ---- Imaging ----------------------------------------------------------

    if img.enabled and img.deconvolver == 'mtmfs' and img.nterms is None:
        raise ConfigError(
            "missing_nterms",
            "imaging.nterms",
            "nterms required for mtmfs deconvolver"
        )

    # ---- Corruption -------------------------------------------------------

    if corr.noise.enabled and corr.noise.mode == 'simplenoise' and not corr.noise.value:
        raise ConfigError(
            "missing_noise_value",
            "corruption.noise.value",
            "noise value required for simplenoise mode"
        )

    # ---- Warnings (log, do not raise) -------------------------------------

    if sm.stokes == 'IQUV':
        for s in obs.spws:
            n_prods = len(s.stokes.strip().split())
            if n_prods < 4:
                log.warning("[validation] sky_model.stokes=IQUV but SPW '%s' has "
                            "only %d correlation product(s): %s",
                            s.name, n_prods, s.stokes)

    if (pred.cell is None or pred.imsize is None) and effective_predictor == 'ft_dft':
        log.warning("[validation] prediction.cell/imsize not needed for ft_dft, ignoring")

    if corr.seed is None:
        log.warning("[validation] corruption.seed=null — results will not be reproducible")
