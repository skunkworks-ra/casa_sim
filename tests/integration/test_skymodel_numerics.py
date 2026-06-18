"""
test_skymodel_numerics.py — Design-level tests for casa_sim/skymodel/

Covers (no CASA):
  - _sample_dist: uniform, normal, None input
  - _interp_sed_to_channels: values at catalog frequencies recover catalog values

Covers (CASA required, @pytest.mark.casa):
  - _apply_faraday_rotation: Q/U match RM * (lambda^2 - lambda0^2) per channel
  - _apply_spectral_extrapolation: flux at each channel matches power law
  - build_component_list with frac_pol: derives correct Q and U from I, p, chi
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

casa = pytest.mark.casa

# ---------------------------------------------------------------------------
# No-CASA: _sample_dist
# ---------------------------------------------------------------------------

from casa_sim.skymodel.trecs import _sample_dist, _TRECS_FREQS_HZ, _interp_sed_to_channels, _draw_n_components


def test_sample_dist_uniform_shape_and_bounds():
    rng = np.random.default_rng(0)
    samples = _sample_dist({'kind': 'uniform', 'low': -10.0, 'high': 10.0}, 1000, rng)
    assert samples.shape == (1000,)
    assert samples.min() >= -10.0
    assert samples.max() <= 10.0


def test_sample_dist_normal_shape_and_moments():
    rng = np.random.default_rng(0)
    samples = _sample_dist({'kind': 'normal', 'mean': 5.0, 'std': 2.0}, 10000, rng)
    assert samples.shape == (10000,)
    assert abs(samples.mean() - 5.0) < 0.1
    assert abs(samples.std() - 2.0) < 0.1


def test_sample_dist_none_returns_zeros():
    rng = np.random.default_rng(0)
    samples = _sample_dist(None, 50, rng)
    assert samples.shape == (50,)
    assert np.all(samples == 0.0)


def test_sample_dist_unknown_kind_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Unknown distribution kind"):
        _sample_dist({'kind': 'poisson', 'lam': 1.0}, 10, rng)


# ---------------------------------------------------------------------------
# No-CASA: _interp_sed_to_channels
# ---------------------------------------------------------------------------

def test_interp_sed_recovers_catalog_values():
    """
    Interpolating at catalog frequencies should recover the catalog fluxes
    to within the precision of log-log interpolation at the support points.
    """
    rng = np.random.default_rng(42)
    n_src = 5
    # Random positive flux densities at the 18 catalog frequencies
    I_cat_jy = rng.uniform(0.001, 10.0, size=(n_src, 18))

    # Interpolate at the catalog frequencies themselves
    I_interp = _interp_sed_to_channels(I_cat_jy, _TRECS_FREQS_HZ)

    np.testing.assert_allclose(
        I_interp, I_cat_jy, rtol=1e-5,
        err_msg="Interpolation at catalog support frequencies should recover exact values"
    )


def test_interp_sed_output_shape():
    rng = np.random.default_rng(0)
    n_src, nchan = 10, 32
    I_cat = rng.uniform(0.01, 1.0, size=(n_src, 18))
    chan_freqs = np.linspace(1e9, 2e9, nchan)
    result = _interp_sed_to_channels(I_cat, chan_freqs)
    assert result.shape == (n_src, nchan)


def test_interp_sed_monotone_for_power_law_source():
    """
    A source with a steep negative spectral index should produce strictly
    decreasing flux with increasing frequency.
    """
    # Build a perfect power-law SED: S(nu) = S0 * (nu/nu0)^alpha
    alpha = -0.8
    nu0 = _TRECS_FREQS_HZ[9]   # 1400 MHz
    S0 = 1.0
    I_cat_jy = np.array([[S0 * (nu / nu0) ** alpha for nu in _TRECS_FREQS_HZ]])

    chan_freqs = np.linspace(500e6, 10e9, 50)
    result = _interp_sed_to_channels(I_cat_jy, chan_freqs)[0]

    assert np.all(np.diff(result) < 0), "Power-law source with alpha<0 must have decreasing flux"


# ---------------------------------------------------------------------------
# CASA: _apply_faraday_rotation formula check
# ---------------------------------------------------------------------------

@casa
def test_faraday_rotation_formula(workdir):
    """
    Inject I=1, Q=1, U=0 at all channels. Apply known RM.
    Verify Q(chan) = cos(2*angle), U(chan) = sin(2*angle)
    where angle = RM * (lam^2 - lam0^2).
    """
    from casatools import image as iatool, quanta
    from casa_sim.skymodel.faraday import _apply_faraday_rotation
    from casa_sim.skymodel.utils import _get_stokes_indices_from_csys, _C_LIGHT
    from casa_sim.config import FaradayConfig, SkyModelConfig

    qa = quanta()
    ia = iatool()

    # Build a tiny [1,1,4,8] image with a real spectral axis
    nchan = 8
    ref_freq_hz = 1.0e9    # 1 GHz
    chan_width_hz = 0.1e9  # 100 MHz

    in_path = str(workdir / 'test_in.im')
    os.system(f'rm -rf {in_path}')
    ia.fromshape(in_path, [1, 1, 4, nchan], overwrite=True)
    cs = ia.coordsys()
    cs.setunits(['rad', 'rad', '', 'Hz'])
    cs.setincrement([-1e-5, 1e-5], 'direction')
    cs.setreferencevalue([0.0, 0.0], type='direction')
    cs.setreferencevalue(f'{ref_freq_hz}Hz', 'spectral')
    cs.setreferencepixel([0], 'spectral')
    cs.setincrement(f'{chan_width_hz}Hz', 'spectral')
    # Set Stokes axis to IQUV
    cs.setstokes('I Q U V')
    ia.setcoordsys(cs.torecord())

    # Fill: I=1, Q=1, U=0, V=0 at all channels
    pix = ia.getchunk()   # [1, 1, 4, nchan]
    pix[:, :, 0, :] = 1.0   # I
    pix[:, :, 1, :] = 1.0   # Q
    pix[:, :, 2, :] = 0.0   # U
    pix[:, :, 3, :] = 0.0   # V
    ia.putchunk(pix)
    ia.close()

    # Build a mock SkyModelConfig with Faraday enabled
    RM = 50.0   # rad/m^2
    ref_freq_str = '1.0GHz'
    faraday = FaradayConfig(enabled=True, rm_mode='global',
                            rm_value=RM, ref_freq=ref_freq_str)
    sm_cfg = SkyModelConfig(stokes='IQUV', mode='component_list', faraday=faraday)

    out_path = str(workdir / 'test_out.im')
    ia_inst = iatool()
    _apply_faraday_rotation(in_path, sm_cfg, out_path, ia_inst, qa)

    # Read result
    ia_inst.open(out_path)
    result = ia_inst.getchunk()   # [1, 1, 4, nchan]
    csys_out = ia_inst.coordsys()
    ia_inst.close()

    stokes_idx = _get_stokes_indices_from_csys(csys_out)
    q_idx = stokes_idx['Q']
    u_idx = stokes_idx['U']

    # Reconstruct channel frequencies
    ref_val = csys_out.referencevalue()['numeric']
    inc = csys_out.increment()['numeric']
    axis_types = csys_out.axiscoordinatetypes()
    spec_axis = axis_types.index('Spectral')
    freq0 = ref_val[spec_axis]
    dfreq = inc[spec_axis]
    chan_freqs = np.array([freq0 + i * dfreq for i in range(nchan)])

    lam0_sq = (_C_LIGHT / ref_freq_hz) ** 2

    for k in range(nchan):
        lam_sq = (_C_LIGHT / chan_freqs[k]) ** 2
        angle = RM * (lam_sq - lam0_sq)
        expected_Q = math.cos(2.0 * angle)
        expected_U = math.sin(2.0 * angle)

        np.testing.assert_allclose(
            result[0, 0, q_idx, k], expected_Q, atol=1e-6,
            err_msg=f"Q[chan={k}]: expected cos(2*angle)={expected_Q:.6f}"
        )
        np.testing.assert_allclose(
            result[0, 0, u_idx, k], expected_U, atol=1e-6,
            err_msg=f"U[chan={k}]: expected sin(2*angle)={expected_U:.6f}"
        )


# ---------------------------------------------------------------------------
# CASA: _apply_spectral_extrapolation power-law check
# ---------------------------------------------------------------------------

@casa
def test_spectral_extrapolation_power_law(workdir):
    """
    Fill an image with 1 Jy at every channel at a reference pixel.
    Extrapolate with alpha=-0.7.
    Verify each channel flux matches (nu/nu_ref)^(-0.7).
    """
    from casatools import image as iatool, quanta
    from casa_sim.skymodel.spectral import _apply_spectral_extrapolation
    from casa_sim.config import SkyModelConfig

    qa = quanta()
    ia = iatool()

    nchan = 10
    ref_freq_hz = 1.4e9
    chan_width_hz = 0.1e9
    alpha = -0.7
    ref_freq_str = '1.4GHz'

    in_path = str(workdir / 'extrap_in.im')
    os.system(f'rm -rf {in_path}')
    ia.fromshape(in_path, [4, 4, 1, nchan], overwrite=True)
    cs = ia.coordsys()
    cs.setunits(['rad', 'rad', '', 'Hz'])
    cs.setincrement([-1e-5, 1e-5], 'direction')
    cs.setreferencevalue([0.0, 0.0], type='direction')
    cs.setreferencevalue(f'{ref_freq_hz}Hz', 'spectral')
    cs.setreferencepixel([0], 'spectral')
    cs.setincrement(f'{chan_width_hz}Hz', 'spectral')
    ia.setcoordsys(cs.torecord())
    # Fill with uniform 1 Jy
    pix = ia.getchunk()
    pix[:] = 1.0
    ia.putchunk(pix)
    ia.close()

    sm_cfg = SkyModelConfig(
        stokes='I', mode='image_extrapolate',
        image_path=in_path, ref_freq=ref_freq_str,
        alpha_mode='global', alpha_value=alpha
    )

    out_path = str(workdir / 'extrap_out.im')
    ia_inst = iatool()
    _apply_spectral_extrapolation(in_path, sm_cfg, out_path, ia_inst, qa)

    ia_inst.open(out_path)
    result = ia_inst.getchunk()   # [4, 4, 1, nchan]
    csys_out = ia_inst.coordsys()
    ia_inst.close()

    # Reconstruct channel frequencies
    ref_val = csys_out.referencevalue()['numeric']
    inc = csys_out.increment()['numeric']
    axis_types = csys_out.axiscoordinatetypes()
    spec_axis = axis_types.index('Spectral')
    freq0 = ref_val[spec_axis]
    dfreq = inc[spec_axis]
    chan_freqs = np.array([freq0 + i * dfreq for i in range(nchan)])

    for k in range(nchan):
        expected = (chan_freqs[k] / ref_freq_hz) ** alpha
        # Check one pixel (center-ish)
        np.testing.assert_allclose(
            result[2, 2, 0, k], expected, rtol=1e-5,
            err_msg=f"chan {k}: expected {expected:.6f}, got {result[2, 2, 0, k]:.6f}"
        )


# ---------------------------------------------------------------------------
# CASA: build_component_list frac_pol derives correct Q and U
# ---------------------------------------------------------------------------

@casa
def test_build_component_list_frac_pol_derives_qu(workdir):
    """
    A source with frac_pol=p and chi degrees must produce
    Q = I * p * cos(2*chi_rad) and U = I * p * sin(2*chi_rad).
    """
    from casatools import componentlist
    from casa_sim.config import SourceDef
    from casa_sim.skymodel.component_list import build_component_list

    I = 2.0
    p = 0.15
    chi_deg = 30.0
    chi_rad = math.radians(chi_deg)
    expected_Q = I * p * math.cos(2.0 * chi_rad)
    expected_U = I * p * math.sin(2.0 * chi_rad)

    sources = [SourceDef(
        name='src',
        direction='J2000 19h59m28.5s +40d40m00.0s',
        flux=[I],
        ref_freq='1.4GHz',
        frac_pol=p,
        chi=chi_deg,
    )]

    cl_path = str(workdir / 'test_frac_pol.cl')
    cl = componentlist()
    build_component_list(sources, cl_path, cl)

    cl.open(cl_path)
    comp = cl.getcomponent(0)
    cl.done()

    flux = comp['flux']['value']   # [I, Q, U, V] in Jy
    np.testing.assert_allclose(flux[0], I, rtol=1e-6, err_msg="Stokes I")
    np.testing.assert_allclose(flux[1], expected_Q, atol=1e-6, err_msg="Stokes Q")
    np.testing.assert_allclose(flux[2], expected_U, atol=1e-6, err_msg="Stokes U")
    np.testing.assert_allclose(flux[3], 0.0,        atol=1e-6, err_msg="Stokes V")


@casa
def test_build_component_list_explicit_iquv(workdir):
    """Explicit [I,Q,U,V] flux is passed through unchanged."""
    from casatools import componentlist
    from casa_sim.config import SourceDef
    from casa_sim.skymodel.component_list import build_component_list

    sources = [SourceDef(
        name='src',
        direction='J2000 19h59m28.5s +40d40m00.0s',
        flux=[3.0, 0.5, 0.2, 0.0],
        ref_freq='1.4GHz',
    )]
    cl_path = str(workdir / 'test_iquv.cl')
    cl = componentlist()
    build_component_list(sources, cl_path, cl)

    cl.open(cl_path)
    comp = cl.getcomponent(0)
    cl.done()

    flux = comp['flux']['value']
    np.testing.assert_allclose(flux, [3.0, 0.5, 0.2, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# No-CASA: _draw_n_components
# ---------------------------------------------------------------------------

def test_draw_n_components_poisson_min_1():
    rng = np.random.default_rng(0)
    result = _draw_n_components({'kind': 'poisson', 'lam': 0.1}, n_src=1000, rng=rng)
    assert np.all(result >= 1)
    assert result.dtype == int


def test_multi_component_produces_depolarization():
    """
    With many RM components spread across a wide RM range, the fractional
    polarization |P|/I must be lower on average than with a single component.
    This is the whole point of Faraday complexity.
    """
    C_LIGHT = 299792458.0
    n_src = 200
    nchan = 64
    chan_freqs_hz = np.linspace(1.0e9, 2.0e9, nchan)
    ref_freq_hz = 1.5e9
    lam_sq_ref   = (C_LIGHT / ref_freq_hz) ** 2
    lam_sq_chans = (C_LIGHT / chan_freqs_hz) ** 2
    delta_lam_sq = lam_sq_chans[np.newaxis, :] - lam_sq_ref

    I_chans = np.ones((n_src, nchan))
    p_chan  = 0.1 * np.ones((n_src, nchan))
    dist_rm   = {'kind': 'uniform', 'low': -200.0, 'high': 200.0}
    dist_chi0 = {'kind': 'uniform', 'low': 0.0,    'high': np.pi}

    def run(n_comps_cfg, seed):
        rng = np.random.default_rng(seed)
        n_comps_arr = _draw_n_components(n_comps_cfg, n_src, rng)
        max_comps = int(n_comps_arr.max())
        P_complex = np.zeros((n_src, nchan), dtype=np.complex128)
        for k in range(max_comps):
            active = n_comps_arr > k
            RM_k   = _sample_dist(dist_rm,   n_src, rng)
            chi0_k = _sample_dist(dist_chi0, n_src, rng)
            p_k_chan = p_chan / n_comps_arr[:, np.newaxis]
            chi_k = chi0_k[:, np.newaxis] + RM_k[:, np.newaxis] * delta_lam_sq
            P_complex += active[:, np.newaxis] * p_k_chan * I_chans * np.exp(2j * chi_k)
        return np.abs(P_complex)

    frac_pol_1 = run(1,  seed=7).mean()
    frac_pol_5 = run(5,  seed=7).mean()

    assert frac_pol_5 < frac_pol_1, (
        f"5-component RM should depolarize: mean |P|={frac_pol_5:.4f} vs 1-component {frac_pol_1:.4f}"
    )


def test_active_mask_sources_with_one_component_unaffected_by_second_loop_iter():
    """
    Sources with n_comps=1 must contribute zero to the k=1 loop iteration.
    Split n_src into half n=1 and half n=2. The n=1 sources should have
    identical P_complex after the full loop as they would from a single pass.
    """
    C_LIGHT = 299792458.0
    nchan = 16
    chan_freqs_hz = np.linspace(1.0e9, 2.0e9, nchan)
    ref_freq_hz = 1.5e9
    lam_sq_chans = (C_LIGHT / chan_freqs_hz) ** 2
    delta_lam_sq = lam_sq_chans[np.newaxis, :] - (C_LIGHT / ref_freq_hz) ** 2

    n_src = 10
    I_chans = np.ones((n_src, nchan))
    p_chan  = 0.1 * np.ones((n_src, nchan))

    # Force exactly: first 5 sources get 1 component, last 5 get 2
    n_comps_arr = np.array([1]*5 + [2]*5, dtype=int)

    rng = np.random.default_rng(0)
    dist_rm   = {'kind': 'uniform', 'low': -100.0, 'high': 100.0}
    dist_chi0 = {'kind': 'uniform', 'low': 0.0,    'high': np.pi}

    P_complex = np.zeros((n_src, nchan), dtype=np.complex128)
    for k in range(2):
        active = n_comps_arr > k
        RM_k   = _sample_dist(dist_rm,   n_src, rng)
        chi0_k = _sample_dist(dist_chi0, n_src, rng)
        p_k_chan = p_chan / n_comps_arr[:, np.newaxis]
        chi_k = chi0_k[:, np.newaxis] + RM_k[:, np.newaxis] * delta_lam_sq
        P_complex += active[:, np.newaxis] * p_k_chan * I_chans * np.exp(2j * chi_k)

    # n=1 sources: |P|/I must equal p=0.1 exactly (single component, no cancellation)
    frac_pol_n1 = np.abs(P_complex[:5]) / I_chans[:5]
    np.testing.assert_allclose(frac_pol_n1, 0.1, rtol=1e-10,
        err_msg="n=1 sources must have |P|/I == p regardless of second loop iteration"
    )
