"""
conftest.py — shared pytest fixtures for casa_sim integration tests.

All tests that simulate an MS receive the `workdir` fixture, which:
  - creates a temporary directory via pytest's tmp_path
  - changes cwd into it so CASA writes files there
  - restores cwd on teardown

CASA tools are process-global; each test gets fresh instances from casatools.
"""

from __future__ import annotations

import os
import sys
import shutil
import textwrap

import pytest

# Make sure casa_sim is importable when running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TESTS_DIR, 'data')
CONFIGS_DIR = os.path.join(TESTS_DIR, 'configs')


@pytest.fixture
def workdir(tmp_path):
    """Change into a fresh temp directory for the duration of a test."""
    original = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original)


def make_point_cl(cl_path: str) -> None:
    """Write a 1 Jy flat-spectrum Stokes-I point source component list."""
    from casatools import componentlist
    cl = componentlist()
    os.system(f'rm -rf {cl_path}')
    cl.done()
    cl.addcomponent(
        dir='J2000 19h59m28.5s +40d40m00.0s',
        flux=1.0, fluxunit='Jy',
        freq='1.0GHz',
        shape='point',
        spectrumtype='spectral index',
        index=0.0,
    )
    cl.rename(filename=cl_path)
    cl.done()


def make_polarized_cl(cl_path: str) -> None:
    """Write a polarized point source: I=1 Jy, Q=0.2 Jy, U=0.1 Jy, V=0 Jy."""
    from casatools import componentlist
    cl = componentlist()
    os.system(f'rm -rf {cl_path}')
    cl.done()
    cl.addcomponent(
        dir='J2000 19h59m28.5s +40d40m00.0s',
        flux=[1.0, 0.2, 0.1, 0.0], fluxunit='Jy',
        freq='1.0GHz',
        shape='point',
        spectrumtype='spectral index',
        index=0.0,
    )
    cl.rename(filename=cl_path)
    cl.done()


def write_config(template_path: str, dest_path: str, cl_path: str) -> None:
    """Copy a config template, substituting {CL_PATH} with the actual path."""
    with open(template_path) as fh:
        text = fh.read()
    text = text.replace('{CL_PATH}', cl_path)
    with open(dest_path, 'w') as fh:
        fh.write(text)
