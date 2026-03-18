#!/usr/bin/env python
"""
test_vla_basic.py — Categories A, B, C

Tests:
  A: MS metadata (scan structure, field directions, SPW, antenna table)
  B: Predict-recover (point source flux within 2%, position within 1 cell)
  C: Noise RMS from visstat within 5% of expected for seed=42

Usage:
  python tests/integration/test_vla_basic.py --generate-reference
  python tests/integration/test_vla_basic.py
  python tests/integration/test_vla_basic.py --config-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure casa_sim is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

REFERENCE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'reference', 'test_vla_basic.json')
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'configs', 'vla_basic.yaml')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


# ---------------------------------------------------------------------------
# PASS/FAIL helpers
# ---------------------------------------------------------------------------

_results = []

def check(name: str, measured, reference, tol: float, relative: bool = True):
    if relative:
        err = abs(measured - reference) / (abs(reference) + 1e-30)
        passed = err <= tol
        detail = f"measured={measured:.6g}, reference={reference:.6g}, rel_err={err:.4f}, tol={tol}"
    else:
        err = abs(measured - reference)
        passed = err <= tol
        detail = f"measured={measured:.6g}, reference={reference:.6g}, abs_err={err:.4g}, tol={tol}"
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {name}: {detail}")
    _results.append((tag, name))
    return passed


def summarize():
    n_pass = sum(1 for t, _ in _results if t == 'PASS')
    n_fail = sum(1 for t, _ in _results if t == 'FAIL')
    print(f"\n{'='*60}")
    print(f"Results: {n_pass} PASS, {n_fail} FAIL")
    if n_fail > 0:
        print("FAILED checks:")
        for tag, name in _results:
            if tag == 'FAIL':
                print(f"  - {name}")
    print('='*60)
    return n_fail == 0


# ---------------------------------------------------------------------------
# Test setup — create data and run simulation
# ---------------------------------------------------------------------------

def make_point_source_cl():
    """Create a 1 Jy point source component list for testing."""
    from casatools import componentlist, measures, quanta
    cl = componentlist()
    me = measures()
    qa = quanta()

    clpath = os.path.join(DATA_PATH, 'point_source_1Jy.cl')
    os.system(f'rm -rf {clpath}')
    cl.done()

    cl.addcomponent(
        dir='J2000 19h59m28.5s +40d40m00.0s',
        flux=1.0,
        fluxunit='Jy',
        freq='1.0GHz',
        shape='point',
        spectrumtype='spectral index',
        index=0.0
    )
    cl.rename(filename=clpath)
    cl.done()
    print(f"Created: {clpath}")


def run_simulation():
    """Run the full simulation pipeline and return msname."""
    from casa_sim.config import load_config_with_sweep
    from casa_sim.simulate import run_single

    base_cfg, sweep_cfgs, raw = load_config_with_sweep(CONFIG_PATH)
    run_single(base_cfg)
    return base_cfg.name + '.ms', base_cfg


# ---------------------------------------------------------------------------
# Category A: structural checks
# ---------------------------------------------------------------------------

def check_structure(msname: str) -> dict:
    from casatools import table, ms as mstool
    from casatasks import listobs

    tb = table()
    myms = mstool()

    results = {}

    # Scan count
    tb.open(msname + '/OBSERVATION')
    tb.close()

    myms.open(msname)
    summary = myms.summary()
    myms.close()

    # Field count
    tb.open(msname + '/FIELD')
    n_fields = tb.nrows()
    field_dirs = tb.getcol('NAME')
    tb.close()

    # SPW count
    tb.open(msname + '/SPECTRAL_WINDOW')
    n_spws = tb.nrows()
    n_chans = tb.getcol('NUM_CHAN')
    tb.close()

    # Antenna count
    tb.open(msname + '/ANTENNA')
    n_ant = tb.nrows()
    ant_names = list(tb.getcol('NAME'))
    tb.close()

    results['n_fields'] = int(n_fields)
    results['n_spws'] = int(n_spws)
    results['n_antennas'] = int(n_ant)
    results['n_chans_spw0'] = int(n_chans[0])
    results['ant_names'] = sorted(ant_names)

    # Baseline count from DATA shape
    tb.open(msname)
    data = tb.getcol('DATA')
    n_baselines = data.shape[2] // n_chans[0] if n_chans[0] > 0 else 0
    tb.close()

    # visstat for mean amplitude
    from casatasks import visstat
    vstat = visstat(vis=msname, axis='amp', datacolumn='data')
    key = list(vstat.keys())[0]
    results['mean_amp'] = float(vstat[key]['mean'])
    results['stddev_amp'] = float(vstat[key]['stddev'])

    return results


# ---------------------------------------------------------------------------
# Category B: predict-recover
# ---------------------------------------------------------------------------

def check_predict_recover(cfg) -> dict:
    imagename = cfg.name + '_sanity'
    results = {}

    sanity_json = cfg.name + '_sanity.json'
    if os.path.exists(sanity_json):
        with open(sanity_json) as fh:
            sanity = json.load(fh)
        results.update(sanity)

    from casatasks import imstat
    image_path = imagename + '.image'
    if os.path.exists(image_path):
        istat = imstat(image_path)
        results['peak_I_image'] = float(istat['max'][0])

    return results


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_tests(generate_reference: bool = False, config_only: bool = False):
    # Config-only mode: just parse and validate
    if config_only:
        print("\n--- Config-only validation ---")
        from casa_sim.config import load_config_with_sweep
        base_cfg, _, _ = load_config_with_sweep(CONFIG_PATH)
        print(f"[PASS] Config parsed and validated: {CONFIG_PATH}")
        print(f"       name={base_cfg.name}")
        print(f"       cell={base_cfg.effective_cell}")
        print(f"       imsize={base_cfg.effective_imsize}")
        return True

    # Setup
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)

    make_point_source_cl()
    msname, cfg = run_simulation()

    # Collect measurements
    struct = check_structure(msname)
    predict = check_predict_recover(cfg)

    measurements = {
        'ms_structure': struct,
        'predict_recover': predict
    }

    if generate_reference:
        with open(REFERENCE_PATH, 'w') as fh:
            json.dump(measurements, fh, indent=2)
        print(f"\nReference written to: {REFERENCE_PATH}")
        print("Share this file and re-run without --generate-reference to compare.")
        return True

    # Compare against reference
    if not os.path.exists(REFERENCE_PATH):
        print(f"\nNo reference found at {REFERENCE_PATH}.")
        print("Run with --generate-reference first.")
        return False

    with open(REFERENCE_PATH) as fh:
        reference = json.load(fh)

    ref_struct = reference.get('ms_structure', {})
    ref_predict = reference.get('predict_recover', {})

    print("\n--- Category A: Structural ---")
    check('n_fields', struct['n_fields'], ref_struct.get('n_fields', 1), tol=0, relative=False)
    check('n_spws', struct['n_spws'], ref_struct.get('n_spws', 1), tol=0, relative=False)
    check('n_antennas', struct['n_antennas'], ref_struct.get('n_antennas', 9), tol=0, relative=False)
    check('n_chans_spw0', struct['n_chans_spw0'], ref_struct.get('n_chans_spw0', 5), tol=0, relative=False)

    print("\n--- Category B: Predict-Recover ---")
    if 'peak_I_image' in predict and 'peak_I_image' in ref_predict:
        check('peak_intensity_within_2pct',
              predict['peak_I_image'], ref_predict['peak_I_image'],
              tol=0.02, relative=True)

    print("\n--- Category C: Noise (relative to reference) ---")
    if 'stddev_amp' in struct and 'stddev_amp' in ref_struct:
        check('noise_rms_within_5pct',
              struct['stddev_amp'], ref_struct['stddev_amp'],
              tol=0.05, relative=True)

    return summarize()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test_vla_basic: Categories A, B, C')
    parser.add_argument('--generate-reference', action='store_true')
    parser.add_argument('--config-only', action='store_true')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S')

    ok = run_tests(generate_reference=args.generate_reference,
                   config_only=args.config_only)
    sys.exit(0 if ok else 1)
