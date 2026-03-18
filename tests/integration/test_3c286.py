#!/usr/bin/env python
"""
test_3c286.py — 3C 286 full-Stokes reference test

Simulates a realistic VLA L-band observation of 3C 286 using Perley & Butler
(2013) flux and polarization parameters, then images back and checks fidelity.

Tests:
  A: MS structure (correlations RR/RL/LR/LL, field count, SPW, antennas)
  B: Predict-recover (Stokes I peak within 2%, Q/U peak within 5%)
  C: Noise RMS within 5% of reference (seed=42, 0.05 Jy simplenoise)

3C 286 parameters (Perley & Butler 2013, L-band ~1.4 GHz):
  Position:  J2000 13h31m08.2899s +30d30m32.959s
  Stokes I:  14.9 Jy
  Spectral index: -0.61
  Lin. pol. fraction: 11.3%
  EVPA: 66 deg  ->  Q = I*p*cos(2*chi) = -1.13 Jy, U = I*p*sin(2*chi) = +1.25 Jy
  Stokes V:  ~0 Jy

Usage:
  /path/to/python tests/integration/test_3c286.py --generate-reference
  /path/to/python tests/integration/test_3c286.py
  /path/to/python tests/integration/test_3c286.py --config-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

# Ensure casa_sim root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

REFERENCE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'reference', 'test_3c286.json')
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'configs', 'vla_3c286.yaml')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# ---------------------------------------------------------------------------
# 3C 286 physical constants (Perley & Butler 2013)
# ---------------------------------------------------------------------------

_3C286_I_JY   = 14.9        # Stokes I at 1.4 GHz [Jy]
_3C286_SPIX   = -0.61       # spectral index
_3C286_POL    = 0.113       # linear polarization fraction
_3C286_EVPA_DEG = 66.0      # EVPA [degrees]

# Derived: Q = I*p*cos(2*chi), U = I*p*sin(2*chi)
_chi_rad = math.radians(2 * _3C286_EVPA_DEG)
_3C286_Q_JY = _3C286_I_JY * _3C286_POL * math.cos(_chi_rad)   # ~-1.13 Jy
_3C286_U_JY = _3C286_I_JY * _3C286_POL * math.sin(_chi_rad)   # ~+1.25 Jy
_3C286_V_JY = 0.0


# ---------------------------------------------------------------------------
# PASS/FAIL helpers
# ---------------------------------------------------------------------------

_results = []


def check(name: str, measured, reference, tol: float, relative: bool = True):
    if relative:
        err = abs(measured - reference) / (abs(reference) + 1e-30)
        passed = err <= tol
        detail = (f"measured={measured:.6g}, reference={reference:.6g}, "
                  f"rel_err={err:.4f}, tol={tol}")
    else:
        err = abs(measured - reference)
        passed = err <= tol
        detail = (f"measured={measured:.6g}, reference={reference:.6g}, "
                  f"abs_err={err:.4g}, tol={tol}")
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
# Test setup
# ---------------------------------------------------------------------------

def make_3c286_cl():
    """
    Create a realistic 3C 286 L-band component list.

    Uses Perley & Butler (2013) values:
      I=14.9 Jy, spectral_index=-0.61
      Q=-1.13 Jy, U=+1.25 Jy, V=0 Jy at 1.4 GHz
    """
    from casatools import componentlist

    cl = componentlist()
    clpath = os.path.join(DATA_PATH, '3c286_lband.cl')
    os.system(f'rm -rf {clpath}')
    cl.done()

    cl.addcomponent(
        dir='J2000 13h31m08.2899s +30d30m32.959s',
        flux=[_3C286_I_JY, _3C286_Q_JY, _3C286_U_JY, _3C286_V_JY],
        fluxunit='Jy',
        freq='1.4GHz',
        shape='point',
        spectrumtype='spectral index',
        index=_3C286_SPIX,
    )
    cl.rename(filename=clpath)
    cl.done()
    print(f"Created: {clpath}")
    print(f"  I={_3C286_I_JY:.2f} Jy  Q={_3C286_Q_JY:.3f} Jy  "
          f"U={_3C286_U_JY:.3f} Jy  V={_3C286_V_JY:.1f} Jy  "
          f"spix={_3C286_SPIX}")


def run_simulation():
    """Run full pipeline and return (msname, cfg)."""
    from casa_sim.config import load_config_with_sweep
    from casa_sim.simulate import run_single

    base_cfg, sweep_cfgs, raw = load_config_with_sweep(CONFIG_PATH)
    run_single(base_cfg)
    return base_cfg.name + '.ms', base_cfg


# ---------------------------------------------------------------------------
# Category A: structural checks
# ---------------------------------------------------------------------------

def check_structure(msname: str) -> dict:
    from casatools import table
    from casatasks import visstat

    tb = table()
    results = {}

    tb.open(msname + '/FIELD')
    results['n_fields'] = int(tb.nrows())
    tb.close()

    tb.open(msname + '/SPECTRAL_WINDOW')
    results['n_spws'] = int(tb.nrows())
    results['n_chans_spw0'] = int(tb.getcol('NUM_CHAN')[0])
    tb.close()

    tb.open(msname + '/ANTENNA')
    results['n_antennas'] = int(tb.nrows())
    tb.close()

    # Correlation types: expect RR=5, RL=6, LR=7, LL=8 (CASA codes)
    tb.open(msname + '/POLARIZATION')
    corr_types = [int(x) for x in tb.getcol('CORR_TYPE').flatten()]
    tb.close()
    results['corr_types'] = corr_types

    vstat = visstat(vis=msname, axis='amp', datacolumn='data')
    key = list(vstat.keys())[0]
    results['mean_amp'] = float(vstat[key]['mean'])
    results['stddev_amp'] = float(vstat[key]['stddev'])

    return results


# ---------------------------------------------------------------------------
# Category B: predict-recover (Stokes I, Q, U)
# ---------------------------------------------------------------------------

def check_predict_recover(cfg) -> dict:
    """Read peak intensities per Stokes from sanity imaging output."""
    results = {}

    sanity_json = cfg.name + '_sanity.json'
    if os.path.exists(sanity_json):
        with open(sanity_json) as fh:
            sanity = json.load(fh)
        # peak_intensity is a dict: {'I': val, 'Q': val, 'U': val, 'V': val}
        peaks = sanity.get('peak_intensity', {})
        for stokes in ('I', 'Q', 'U', 'V'):
            if stokes in peaks:
                results[f'peak_{stokes}'] = float(peaks[stokes])

    return results


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_tests(generate_reference: bool = False, config_only: bool = False):
    if config_only:
        print("\n--- Config-only validation ---")
        from casa_sim.config import load_config_with_sweep
        base_cfg, _, _ = load_config_with_sweep(CONFIG_PATH)
        print(f"[PASS] Config parsed: {CONFIG_PATH}")
        print(f"       name={base_cfg.name}")
        print(f"       stokes={base_cfg.sky_model.stokes}")
        print(f"       cell={base_cfg.effective_cell}  imsize={base_cfg.effective_imsize}")
        return True

    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)

    make_3c286_cl()
    msname, cfg = run_simulation()

    struct = check_structure(msname)
    predict = check_predict_recover(cfg)

    measurements = {
        'ms_structure': struct,
        'predict_recover': predict,
    }

    if generate_reference:
        with open(REFERENCE_PATH, 'w') as fh:
            json.dump(measurements, fh, indent=2)
        print(f"\nReference written to: {REFERENCE_PATH}")
        return True

    if not os.path.exists(REFERENCE_PATH):
        print(f"\nNo reference at {REFERENCE_PATH}. Run --generate-reference first.")
        return False

    with open(REFERENCE_PATH) as fh:
        reference = json.load(fh)

    ref_struct = reference.get('ms_structure', {})
    ref_predict = reference.get('predict_recover', {})

    print("\n--- Category A: Structural ---")
    check('n_fields',     struct['n_fields'],     ref_struct.get('n_fields', 1),   tol=0, relative=False)
    check('n_spws',       struct['n_spws'],        ref_struct.get('n_spws', 1),     tol=0, relative=False)
    check('n_antennas',   struct['n_antennas'],    ref_struct.get('n_antennas', 12), tol=0, relative=False)
    check('n_chans_spw0', struct['n_chans_spw0'],  ref_struct.get('n_chans_spw0', 5), tol=0, relative=False)
    # All 4 correlations present (RR=5, RL=6, LR=7, LL=8)
    expected_corr = ref_struct.get('corr_types', [5, 6, 7, 8])
    corr_match = sorted(struct['corr_types']) == sorted(expected_corr)
    tag = "PASS" if corr_match else "FAIL"
    print(f"[{tag}] corr_types: measured={struct['corr_types']} expected={expected_corr}")
    _results.append((tag, 'corr_types'))

    print("\n--- Category B: Predict-Recover (Stokes I, Q, U) ---")
    for stokes, tol in [('I', 0.02), ('Q', 0.05), ('U', 0.05)]:
        key = f'peak_{stokes}'
        if key in predict and key in ref_predict:
            check(f'peak_{stokes}_within_tol',
                  predict[key], ref_predict[key], tol=tol, relative=True)

    print("\n--- Category C: Noise RMS ---")
    if 'stddev_amp' in struct and 'stddev_amp' in ref_struct:
        check('noise_rms_within_5pct',
              struct['stddev_amp'], ref_struct['stddev_amp'],
              tol=0.05, relative=True)

    return summarize()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test_3c286: full-Stokes VLA reference test')
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
