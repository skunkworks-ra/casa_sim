#!/usr/bin/env python
"""
test_sweep.py — Category E

Tests:
  E: 3 MSes produced, correct naming, sweep index JSON complete,
     noise RMS scales monotonically with noise.value.

Usage:
  python tests/integration/test_sweep.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'configs', 'sweep_noise.yaml')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
EXPECTED_N = 3   # 3 noise values in sweep_noise.yaml


_results = []

def check(name, condition, detail=''):
    tag = "PASS" if condition else "FAIL"
    print(f"[{tag}] {name}" + (f": {detail}" if detail else ""))
    _results.append((tag, name))
    return condition


def summarize():
    n_pass = sum(1 for t, _ in _results if t == 'PASS')
    n_fail = sum(1 for t, _ in _results if t == 'FAIL')
    print(f"\n{'='*60}")
    print(f"Results: {n_pass} PASS, {n_fail} FAIL")
    return n_fail == 0


def ensure_cl():
    clpath = os.path.join(DATA_PATH, 'point_source_1Jy.cl')
    if not os.path.exists(clpath):
        from casatools import componentlist
        cl = componentlist()
        cl.done()
        cl.addcomponent(
            dir='J2000 19h59m28.5s +40d40m00.0s',
            flux=1.0, fluxunit='Jy', freq='1.0GHz',
            shape='point', spectrumtype='spectral index', index=0.0
        )
        cl.rename(filename=clpath)
        cl.done()


def run_tests(generate_reference: bool = False):
    os.makedirs(DATA_PATH, exist_ok=True)
    ensure_cl()

    from casa_sim.config import load_config_with_sweep
    from casa_sim.sweep import run_sweep
    from casa_sim.simulate import run_single

    base_cfg, _, raw = load_config_with_sweep(CONFIG_PATH)
    results = run_sweep(base_cfg, raw, run_single)

    print("\n--- Category E: Sweep ---")

    # E1: Correct MS count
    check('ms_count', len(results) == EXPECTED_N,
          f"expected {EXPECTED_N}, got {len(results)}")

    # E2: All MS names distinct
    ms_names = [r['ms_name'] for r in results]
    check('ms_names_distinct', len(set(ms_names)) == len(ms_names),
          f"names: {ms_names}")

    # E3: Sweep index JSON written
    index_path = f"{base_cfg.name}_sweep_index.json"
    check('sweep_index_exists', os.path.exists(index_path), index_path)

    if os.path.exists(index_path):
        with open(index_path) as fh:
            index = json.load(fh)
        check('sweep_index_complete', len(index) == EXPECTED_N,
              f"expected {EXPECTED_N} entries, got {len(index)}")

    # E4: Noise RMS scales monotonically with noise.value
    rms_values = []
    for r in results:
        rms = r.get('metrics', {}).get('residual_rms', [])
        if rms:
            rms_values.append(float(rms[0]) if isinstance(rms[0], (int, float)) else 0.0)

    if len(rms_values) == EXPECTED_N:
        monotonic = all(rms_values[i] < rms_values[i+1]
                        for i in range(len(rms_values)-1))
        check('rms_monotonic_with_noise', monotonic,
              f"rms values: {[f'{v:.4e}' for v in rms_values]}")
    else:
        print(f"[SKIP] rms_monotonic: insufficient rms data ({len(rms_values)} values)")

    return summarize()


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    ok = run_tests()
    sys.exit(0 if ok else 1)
