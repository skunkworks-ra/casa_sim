"""
test_tclean_spw_isolation.py

Confirms that tclean(spw='0', savemodel='modelcolumn') only writes MODEL_DATA
rows belonging to SPW 0 and leaves all other SPW rows untouched.

This is the behavioral guarantee that the per-SPW prediction loop in
_predict_tclean relies on. If this test fails, the loop implementation
needs incremental=True and a pre-zeroed MODEL_DATA instead.
"""

from __future__ import annotations

import os
import textwrap

import numpy as np
import pytest

casa = pytest.mark.casa

_SENTINEL = 99.0 + 0j   # written to SPW 1 MODEL_DATA before the test tclean call

_TWO_SPW_YAML = textwrap.dedent("""
    name: test_2spw
    observatory:
      mode: canned
      feeds: "perfect R L"
      canned:
        telescope: VLA
        cfg_file: null
        antlist: [W01, W02, W03]
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
        - name: SBand
          freq: "3.0GHz"
          deltafreq: "0.05GHz"
          nchan: 4
          stokes: "RR LL"
      observe_calls:
        - field: src1
          spw: LBand
          start_time: "-0.5h"
          stop_time: "+0.5h"
        - field: src1
          spw: SBand
          start_time: "-0.5h"
          stop_time: "+0.5h"
    sky_model:
      stokes: I
      mode: component_list
      cl_path: {CL_PATH}
    prediction:
      gridder: standard
      predictor: tclean
      cell: 2arcsec
      imsize: 64
      normtype: flatsky
    corruption:
      seed: 42
      noise:
        enabled: false
        mode: simplenoise
    imaging:
      enabled: false
""")


@casa
def test_tclean_spw_selection_does_not_overwrite_other_spws(workdir):
    """
    tclean(spw='0', savemodel='modelcolumn') must leave MODEL_DATA rows
    for SPW 1 exactly as they were before the call.
    """
    from casatools import table as tbtool, image as iatool
    from casatasks import tclean

    # ---- Build a 2-SPW MS via the full pipeline ----
    from conftest import make_point_cl
    cl_path = str(workdir / 'src.cl')
    make_point_cl(cl_path)

    cfg_path = str(workdir / 'two_spw.yaml')
    with open(cfg_path, 'w') as fh:
        fh.write(_TWO_SPW_YAML.replace('{CL_PATH}', cl_path))

    from casa_sim.config import load_config_with_sweep
    from casa_sim.simulate import run_single
    cfg, _, _ = load_config_with_sweep(cfg_path)
    run_single(cfg)

    msname = cfg.name + '.ms'
    assert os.path.exists(msname), f"MS not created: {msname}"

    # ---- Overwrite MODEL_DATA for SPW 1 rows with sentinel ----
    tb = tbtool()
    tb.open(msname, nomodify=False)

    ddid_col  = tb.getcol('DATA_DESC_ID')  # one entry per MS row; 0=SPW0, 1=SPW1
    spw1_rows = np.where(ddid_col == 1)[0]
    assert len(spw1_rows) > 0, "No SPW 1 rows found in MS"

    # Read current shape from one SPW 1 row to allocate sentinel correctly
    sample = tb.getcell('MODEL_DATA', int(spw1_rows[0]))  # [ncorr, nchan]
    sentinel_chunk = np.full(sample.shape, _SENTINEL, dtype=np.complex64)
    for row in spw1_rows:
        tb.putcell('MODEL_DATA', int(row), sentinel_chunk)

    tb.close()

    # ---- Run tclean on SPW 0 only ----
    sky_model_path = cfg.name + '_skymodel_fromcl.im'
    assert os.path.exists(sky_model_path), f"Sky model image not found: {sky_model_path}"

    scratch = str(workdir / 'spw0_predict')
    os.system(f'rm -rf {scratch}.*')
    tclean(
        vis=msname,
        startmodel=sky_model_path,
        imagename=scratch,
        savemodel='modelcolumn',
        spw='0',
        imsize=64,
        cell='2arcsec',
        specmode='cube',
        interpolation='nearest',
        nchan=-1,
        stokes='I',
        gridder='standard',
        normtype='flatsky',
        calcres=False,
        calcpsf=False,
        niter=0,
    )

    # ---- Assert SPW 1 MODEL_DATA rows are still the sentinel ----
    tb.open(msname)
    for row in spw1_rows:
        chunk = tb.getcell('MODEL_DATA', int(row))
        np.testing.assert_array_equal(
            chunk,
            np.full(chunk.shape, _SENTINEL, dtype=np.complex64),
            err_msg=f"MODEL_DATA row {row} (SPW 1) was modified by tclean(spw='0')"
        )
    tb.close()
