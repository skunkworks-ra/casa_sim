"""
test_bug_regression.py — regression tests for the 6 silent-failure bugs.

Mark strategy
-------------
Tests marked @pytest.mark.casa require casatools/casatasks and a real MS.
Run them locally with pixi:

    pixi run pytest tests/test_bug_regression.py -m casa

CI runs only the unmarked tests (no CASA dependency):

    pytest tests/test_bug_regression.py -m "not casa"

Bug inventory:
  1. skymodel._apply_faraday_rotation -- ia not closed before fromshape
  2. observation._resolve_freqresolution -- freqresolution = deltafreq/nchan instead of deltafreq
  3. predict._predict_tclean -- imagename 'sim_predict' not namespaced; collides in sweeps
  4. predict._predict_tclean -- missing stokes= arg; silently drops Q,U,V in full-pol runs
  5. sanity._estimate_rm -- hardcoded {I:0,Q:1,U:2,V:3} instead of csys query
  6. config.validate_config -- simplenoise with value=None silently passes validation
"""

from __future__ import annotations

import os
import shutil
import textwrap
import tempfile

import numpy as np
import pytest

casa = pytest.mark.casa  # shorthand for marking CASA-dependent tests

try:
    from conftest import CONFIGS_DIR, DATA_DIR, make_point_cl, make_polarized_cl, write_config
except ImportError:
    pass   # CI skips casa-marked tests; conftest not needed there


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_sim(config_path: str):
    """Load, validate, and run a simulation. Returns (msname, cfg)."""
    from casa_sim.config import load_config_with_sweep
    from casa_sim.simulate import run_single
    cfg, _, _ = load_config_with_sweep(config_path)
    run_single(cfg)
    return cfg.name + '.ms', cfg


# ---------------------------------------------------------------------------
# Bug 1 -- ia.close() missing before ia.fromshape() in _apply_faraday_rotation
# ---------------------------------------------------------------------------

@casa
def test_bug1_faraday_image_non_zero(workdir):
    """
    _apply_faraday_rotation must close ia before calling fromshape.
    Without the fix the output image is written while ia is still attached to
    the input image, silently producing a zero or stale output.

    Assert: the faraday image exists AND the Q plane has non-zero values that
    vary across channels (i.e., rotation actually happened).
    """
    cl_path = str(workdir / 'pol_src.cl')
    cfg_path = str(workdir / 'faraday.yaml')
    make_polarized_cl(cl_path)
    write_config(
        os.path.join(CONFIGS_DIR, 'minimal_iquv_faraday.yaml'),
        cfg_path, cl_path,
    )

    msname, cfg = run_sim(cfg_path)

    faraday_im = f"{cfg.name}_skymodel_faraday.im"
    assert os.path.exists(faraday_im), f"Faraday image not written: {faraday_im}"

    from casatools import image
    ia = image()
    ia.open(faraday_im)
    data = ia.getchunk()   # [nx, ny, nstokes, nchan]
    ia.close()

    # Stokes axis: find Q index
    from casa_sim.skymodel import _get_stokes_indices_from_csys
    ia.open(faraday_im)
    csys = ia.coordsys()
    ia.close()
    stokes_idx = _get_stokes_indices_from_csys(csys)
    q_idx = stokes_idx['Q']

    q_plane = data[:, :, q_idx, :]   # [nx, ny, nchan]
    # Peak Q pixel over all channels
    q_peak_per_chan = np.max(np.abs(q_plane), axis=(0, 1))

    assert np.any(q_peak_per_chan > 0), "Q plane is all zeros — fromshape bug still present"
    assert not np.allclose(q_peak_per_chan, q_peak_per_chan[0]), (
        "Q peak is identical across all channels — Faraday rotation not applied"
    )


# ---------------------------------------------------------------------------
# Bug 2 -- freqresolution = deltafreq/nchan instead of deltafreq
# ---------------------------------------------------------------------------

@casa
def test_bug2_freqresolution_matches_channel_width(workdir):
    """
    sm.setspwindow freqresolution should equal channel width (deltafreq),
    not deltafreq divided by nchan.  Read the value back from the MS
    SPECTRAL_WINDOW table and compare.

    Config: deltafreq=0.05GHz (50 MHz per channel), nchan=4.
    Correct freqresolution: 50 MHz.
    Buggy freqresolution: 50 MHz / 4 = 12.5 MHz.
    """
    cl_path = str(workdir / 'pt_src.cl')
    cfg_path = str(workdir / 'stokes_i.yaml')
    make_point_cl(cl_path)
    write_config(
        os.path.join(CONFIGS_DIR, 'minimal_stokes_i.yaml'),
        cfg_path, cl_path,
    )

    msname, cfg = run_sim(cfg_path)

    from casatools import table
    tb = table()
    tb.open(msname + '/SPECTRAL_WINDOW')
    resolution = tb.getcol('RESOLUTION')   # shape [nchan, nspw]
    chan_width  = tb.getcol('CHAN_WIDTH')   # shape [nchan, nspw]
    tb.close()

    # All channels, first SPW
    res_hz = resolution[:, 0]
    cw_hz  = chan_width[:, 0]

    np.testing.assert_allclose(
        res_hz, cw_hz, rtol=1e-4,
        err_msg=(
            f"RESOLUTION {res_hz[0]/1e6:.3f} MHz != CHAN_WIDTH {cw_hz[0]/1e6:.3f} MHz. "
            "freqresolution is being divided by nchan."
        ),
    )


# ---------------------------------------------------------------------------
# Bug 3 -- tclean predict imagename not namespaced; sweep runs collide
# ---------------------------------------------------------------------------

@casa
def test_bug3_predict_scratch_namespaced(workdir):
    """
    _predict_tclean must use cfg.name as prefix for its scratch images so that
    parallel or sequential sweep runs don't delete each other's files.

    Assert: after a tclean-predict run, {cfg.name}_predict.* exists and the
    legacy 'sim_predict.*' pattern does NOT exist.
    """
    cl_path = str(workdir / 'pt_src.cl')
    cfg_path = str(workdir / 'stokes_i_tclean.yaml')
    make_point_cl(cl_path)

    # Build a config that routes through tclean (image_native mode)
    import yaml
    with open(os.path.join(CONFIGS_DIR, 'minimal_stokes_i.yaml')) as fh:
        raw = fh.read().replace('{CL_PATH}', cl_path)
    cfg_dict = yaml.safe_load(raw)
    # Switch to image_native + tclean predictor by using component_list but forcing predictor
    cfg_dict['prediction']['predictor'] = 'tclean'
    with open(cfg_path, 'w') as fh:
        yaml.dump(cfg_dict, fh)

    # Run with tclean predictor
    from casa_sim.config import load_config_with_sweep
    from casa_sim.simulate import run_single
    cfg, _, _ = load_config_with_sweep(cfg_path)
    run_single(cfg)

    predict_name = f"{cfg.name}_predict"
    legacy_pattern = list(workdir.glob('sim_predict.*'))
    namespaced_pattern = list(workdir.glob(f'{predict_name}.*'))

    assert not legacy_pattern, (
        f"Legacy 'sim_predict.*' files found: {legacy_pattern}. Namespace bug still present."
    )
    assert namespaced_pattern, (
        f"Expected '{predict_name}.*' files but found none."
    )


# ---------------------------------------------------------------------------
# Bug 4 -- tclean predict missing stokes= drops Q,U,V silently
# ---------------------------------------------------------------------------

@casa
def test_bug4_full_pol_tclean_predict_has_all_stokes(workdir):
    """
    _predict_tclean must pass stokes=cfg.sky_model.stokes to tclean.
    Without it, tclean defaults to Stokes I only; MODEL_DATA (and therefore DATA
    after the copy) has zero cross-hand correlations for an IQUV source.

    Assert: after prediction, DATA RL and LR correlations have non-zero mean amplitude.
    """
    cl_path = str(workdir / 'pol_src.cl')
    cfg_path = str(workdir / 'iquv_tclean.yaml')
    make_polarized_cl(cl_path)

    import yaml
    with open(os.path.join(CONFIGS_DIR, 'minimal_iquv_faraday.yaml')) as fh:
        raw = fh.read().replace('{CL_PATH}', cl_path)
    cfg_dict = yaml.safe_load(raw)
    # Disable faraday so the sky model is just a plain IQUV image (no rotation)
    cfg_dict['sky_model']['faraday']['enabled'] = False
    cfg_dict['prediction']['predictor'] = 'tclean'
    cfg_dict['imaging']['enabled'] = False
    with open(cfg_path, 'w') as fh:
        yaml.dump(cfg_dict, fh)

    msname, cfg = run_sim(cfg_path)

    from casatools import table
    tb = table()
    tb.open(msname)
    data = tb.getcol('DATA')   # shape [npol, nchan, nrows]
    tb.close()

    # Correlation order for "RR RL LR LL": indices 1=RL, 2=LR are cross-hands
    npol = data.shape[0]
    assert npol == 4, f"Expected 4 correlations, got {npol}"

    rl_amp = np.mean(np.abs(data[1, :, :]))
    lr_amp = np.mean(np.abs(data[2, :, :]))

    assert rl_amp > 1e-6, (
        f"RL mean amplitude is {rl_amp:.2e} — tclean predict dropped cross-hand correlations"
    )
    assert lr_amp > 1e-6, (
        f"LR mean amplitude is {lr_amp:.2e} — tclean predict dropped cross-hand correlations"
    )


# ---------------------------------------------------------------------------
# Bug 5 -- _estimate_rm uses hardcoded Stokes index instead of csys query
# ---------------------------------------------------------------------------

@casa
def test_bug5_rm_estimate_within_tolerance(workdir):
    """
    _estimate_rm must read Q and U plane indices from the image coordinate system.
    Hardcoded indices work only if tclean happens to write IQUV in that exact order.

    Assert: the recovered RM from the sanity image is within 15 rad/m^2 of
    the injected value of 50 rad/m^2.
    """
    cl_path = str(workdir / 'pol_src.cl')
    cfg_path = str(workdir / 'faraday_rm.yaml')
    make_polarized_cl(cl_path)
    write_config(
        os.path.join(CONFIGS_DIR, 'minimal_iquv_faraday.yaml'),
        cfg_path, cl_path,
    )

    msname, cfg = run_sim(cfg_path)

    import json
    sanity_json = f"{cfg.name}_sanity.json"
    assert os.path.exists(sanity_json), f"Sanity JSON not written: {sanity_json}"

    with open(sanity_json) as fh:
        metrics = json.load(fh)

    assert 'rm_estimate_rad_per_m2' in metrics, (
        "rm_estimate_rad_per_m2 key missing from sanity metrics"
    )

    rm_recovered = metrics['rm_estimate_rad_per_m2']
    rm_injected  = 50.0
    assert abs(rm_recovered - rm_injected) < 15.0, (
        f"RM recovery poor: injected={rm_injected}, recovered={rm_recovered:.2f}. "
        "Stokes index bug may still be present."
    )


# ---------------------------------------------------------------------------
# Bug 6 -- simplenoise with value=None passes validation silently
# ---------------------------------------------------------------------------

def test_bug6_simplenoise_requires_value():
    """
    validate_config must raise ConfigError when noise.mode='simplenoise'
    and noise.value is not set.  No simulation needed -- config-only check.
    """
    import yaml
    from casa_sim.config import ConfigError, load_config, validate_config

    bad_yaml = textwrap.dedent("""
        name: bad_noise
        observatory:
          mode: canned
          feeds: "perfect R L"
          canned:
            telescope: VLA
        observation:
          epoch: "UTC 2020/10/4/00:00:00"
          integration_time: "300s"
          use_hourangle: true
          fields:
            - name: src1
              direction: "J2000 19h59m28.5s +40d40m00.0s"
          spws:
            - name: LBand
              freq: "1.0GHz"
              deltafreq: "0.05GHz"
              nchan: 4
              stokes: "RR LL"
          observe_calls:
            - field: src1
              spw: LBand
              start_time: "-0.5h"
              stop_time: "+0.5h"
        sky_model:
          stokes: I
          mode: component_list
          cl_path: dummy.cl
        prediction:
          gridder: standard
          predictor: ft_dft
          cell: 2arcsec
          imsize: 64
          normtype: flatsky
        corruption:
          seed: 42
          noise:
            enabled: true
            mode: simplenoise
        imaging:
          enabled: false
    """)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as fh:
        fh.write(bad_yaml)
        tmp_path = fh.name

    try:
        with pytest.raises(ConfigError, match="missing_noise_value"):
            cfg = load_config(tmp_path)
            from casa_sim.config import validate_config
            validate_config(cfg)
    finally:
        os.unlink(tmp_path)
