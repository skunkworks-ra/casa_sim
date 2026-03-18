#!/usr/bin/env python
"""
run_all.py — Run all integration tests in sequence.

Usage:
  python tests/run_all.py
  python tests/run_all.py --generate-reference
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TESTS = [
    'tests/integration/test_vla_basic.py',
    'tests/integration/test_faraday.py',
    'tests/integration/test_sweep.py',
]


def load_and_run(path: str, generate_reference: bool) -> bool:
    spec = importlib.util.spec_from_file_location('test_module', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Auto-generate reference if missing
    ref_missing = (hasattr(mod, 'REFERENCE_PATH')
                   and not os.path.exists(mod.REFERENCE_PATH))
    if ref_missing and not generate_reference:
        print(f"[INFO] No reference found — generating reference for {os.path.basename(path)}")
        mod.run_tests(generate_reference=True)

    return mod.run_tests(generate_reference=generate_reference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-reference', action='store_true')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    results = {}
    for test_rel in TESTS:
        test_path = os.path.join(root, test_rel)
        test_name = os.path.basename(test_path)
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        try:
            ok = load_and_run(test_path, args.generate_reference)
            results[test_name] = 'PASS' if ok else 'FAIL'
        except Exception as e:
            print(f"[ERROR] {test_name}: {e}")
            results[test_name] = 'ERROR'

    print(f"\n{'='*60}")
    print("Test Suite Summary")
    print('='*60)
    for name, status in results.items():
        print(f"  {status:6s}  {name}")

    n_fail = sum(1 for s in results.values() if s != 'PASS')
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
