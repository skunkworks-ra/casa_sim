#!/usr/bin/env python
"""
test_faraday.py — Category D

Tests:
  D: Simulate known RM. Recover via lambda^2 fit at source peak.
     Pass: recovered RM within 5% of input RM (SNR > 10 regime).

Usage:
  python tests/integration/test_faraday.py --generate-reference
  python tests/integration/test_faraday.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

REFERENCE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'reference', 'test_faraday.json')
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'configs', 'faraday.yaml')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

INPUT_RM = 50.0   # rad/m^2 — must match faraday.yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_results = []

def check(name, measured, reference, tol, relative=True):
    if relative:
        err = abs(measured - reference) / (abs(reference) + 1e-30)
        passed = err <= tol
        detail = f"measured={measured:.4g}, reference={reference:.4g}, rel_err={err:.4f}, tol={tol}"
    else:
        err = abs(measured - reference)
        passed = err <= tol
        detail = f"measured={measured:.4g}, reference={reference:.4g}, abs_err={err:.4g}, tol={tol}"
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {name}: {detail}")
    _results.append((tag, name))
    return passed


def summarize():
    n_pass = sum(1 for t, _ in _results if t == 'PASS')
    n_fail = sum(1 for t, _ in _results if t == 'FAIL')
    print(f"\n{'='*60}")
    print(f"Results: {n_pass} PASS, {n_fail} FAIL")
    return n_fail == 0


# ---------------------------------------------------------------------------
# Data creation
# ---------------------------------------------------------------------------

def make_polarized_cl():
    """Create a polarized point source: I=1 Q=0.3 U=0.1 V=0.0 Jy."""
    from casatools import componentlist
    cl = componentlist()

    clpath = os.path.join(DATA_PATH, 'polarized_point_source.cl')
    os.system(f'rm -rf {clpath}')
    cl.done()

    # I component
    cl.addcomponent(
        dir='J2000 19h59m28.5s +40d40m00.0s',
        flux=1.0,
        fluxunit='Jy',
        freq='1.0GHz',
        shape='point',
        spectrumtype='spectral index',
        index=0.0,
        polarization='Stokes'
    )
    # The CASA componentlist addcomponent with polarization fractions:
    # Set Stokes directly via flux vector [I, Q, U, V]
    # Re-add with proper Stokes vector
    cl.done()

    # Use flux as [I, Q, U, V] vector
    cl.addcomponent(
        dir='J2000 19h59m28.5s +40d40m00.0s',
        flux=[1.0, 0.3, 0.1, 0.0],
        fluxunit='Jy',
        freq='1.0GHz',
        shape='point',
        spectrumtype='spectral index',
        index=0.0
    )
    cl.rename(filename=clpath)
    cl.done()
    print(f"Created: {clpath}")


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_tests(generate_reference=False):
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)

    make_polarized_cl()

    from casa_sim.config import load_config_with_sweep
    from casa_sim.simulate import run_single

    base_cfg, _, _ = load_config_with_sweep(CONFIG_PATH)
    metrics = run_single(base_cfg)

    rm_recovered = metrics.get('rm_estimate_rad_per_m2', None)
    measurements = {'rm_recovered': rm_recovered, 'rm_input': INPUT_RM}

    if generate_reference:
        with open(REFERENCE_PATH, 'w') as fh:
            json.dump(measurements, fh, indent=2)
        print(f"Reference written: {REFERENCE_PATH}")
        return True

    print("\n--- Category D: Faraday RM Recovery ---")
    print(f"Input RM:     {INPUT_RM:.2f} rad/m^2")
    print(f"Recovered RM: {rm_recovered:.2f} rad/m^2" if rm_recovered else "Recovered RM: N/A")

    if rm_recovered is not None:
        check('rm_within_5pct', rm_recovered, INPUT_RM, tol=0.05, relative=True)
    else:
        print("[FAIL] rm_estimate not present in metrics")
        _results.append(('FAIL', 'rm_estimate_present'))

    return summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-reference', action='store_true')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    ok = run_tests(generate_reference=args.generate_reference)
    sys.exit(0 if ok else 1)
