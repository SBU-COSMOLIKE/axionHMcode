"""
functions for the cold matter halo denity profile
"""

import numpy as np
import scipy 
from scipy import optimize, interpolate
from cosmology.basic_cosmology import func_rho_comp_0
from cosmology.overdensities import func_D_z_norm, func_r_vir, func_Delta_vir
from cosmology.variance import func_nu

# Global cache for the interpolator
_concentration_interpolator_cache = None
_zguess_interpolator_cache = None

#find z_formation given by Mead 2020 eq. 21
def func_z_formation(M, k, PS, cosmo_dic, Omega_0, f = 0.01):
    """
    k is in units of h/Mpc, PS in (Mpc/h)^3 and M in solar_mass/h
    returns the fromation redshift of a halo which is defined as
    in HMCode2020: https://arxiv.org/abs/2009.01858 in eq. 21
    """
    # VM-SPEEDUP BEGINS
    #
    # Quantity. The halo formation redshift z_f of HMcode-2020 (Mead et al.,
    # arXiv:2009.01858, eq. 21; Dome et al., arXiv:2409.11469, eq. 50),
    # defined as the solution of
    #
    #   D(z_f)/D(z) * sigma(f*M, z) = delta_c(z) ,       f = 0.01,
    #
    # where D is the linear growth factor, sigma(f*M, z) the mass variance
    # at the evaluation redshift (the power spectrum passed in is already at
    # redshift z), and delta_c the critical collapse threshold. Writing
    # D_norm(z) = D(z)/D(0), the equation is equivalent to
    #
    #   D_norm(z_f) = t ,   with  t = D_norm(z) * delta_c(z) / sigma(f*M) .
    #
    # Upstream evaluation. scipy.optimize.brentq on z_f in [z, 100]. Each
    # iteration of the bracketing search re-evaluated func_D_z_norm(z_f) —
    # which re-integrates the growth factor and its constant normalization
    # D(0) from scratch — and the objective's nu(f*M) factor re-integrated
    # sigma over the whole k grid. Because this function sits at the bottom
    # of the cut-mass -> concentration -> central-density solver cascade, the
    # measured totals were ~1.1e5 growth integrations and ~1e5 variance
    # integrations per redshift (fork README, appendix section A.1).
    #
    # Fork evaluation. D_norm is smooth and strictly decreasing on [0, 100],
    # so the root-find is a monotone table inversion: compute the target t
    # once and read z_f = D_norm^{-1}(t) from the tabulated growth
    # (fast_tables.z_of_D_norm, linear interpolation on nodes computed with
    # the unmodified upstream quadrature). sigma(f*M) comes from the
    # per-evaluation sigma table (fast_tables.sigma_M_fast), whose nodes are
    # likewise computed with the unmodified upstream function.
    #
    # Clamp semantics. Upstream only ran brentq when the objective changed
    # sign across [z, 100] and returned z otherwise; that branch implements
    # the minimum-concentration clamp of HMcode-2020 (z_f < z would give
    # c < B, which is truncated to c = B). The fork reproduces the branch
    # with the same endpoint sign products, evaluated exactly on the
    # tabulated values: scalar path, a positive product returns z; array
    # path, the inversion is used only where the product is negative. The
    # branch decision itself is never interpolated.
    #
    # Accuracy. Bounded by the fork validation gate, max |dB/B| = 1.6e-5 on
    # the final boost (dev_scripts/fork_validate.py); table node counts scale
    # with the boost theory's accuracy_boost option.
    from cosmology import fast_tables
    from cosmology.overdensities import func_delta_c
    z = cosmo_dic['z']
    Omega_m_0 = cosmo_dic['Omega_m_0']
    Omega_w_0 = cosmo_dic['Omega_w_0']
    Omega_ax_0 = cosmo_dic['Omega_ax_0']
    delta_c = func_delta_c(z, Omega_ax_0, Omega_m_0, Omega_w_0,
                           cosmo_dic['G_a'], cosmo_dic['version'])
    D_norm_z = fast_tables.D_norm_fast(z, Omega_m_0, Omega_w_0)
    D_norm_100 = fast_tables.D_norm_fast(100., Omega_m_0, Omega_w_0)
    if isinstance(M, (int, float)) == True:
        sigma = fast_tables.sigma_M_fast(f * M, k, PS, Omega_0, cosmo_dic)
        target = D_norm_z * delta_c / sigma
        if (D_norm_z - target) * (D_norm_100 - target) > 0.:
            return z
        return float(fast_tables.z_of_D_norm(target, Omega_m_0, Omega_w_0))
    else:
        sigma = fast_tables.sigma_M_fast(f * np.asarray(M, dtype=np.float64),
                                         k, PS, Omega_0, cosmo_dic)
        target = D_norm_z * delta_c / sigma
        z_f = fast_tables.z_of_D_norm(target, Omega_m_0, Omega_w_0)
        no_root = (D_norm_z - target) * (D_norm_100 - target) >= 0.
        return np.where(no_root, z, z_f)
    # VM-SPEEDUP ENDS



def func_conc_param(M, k, PS, cosmo_dic, Omega_0, c_min, axion_dic=None):
    """
    k is in units of h/Mpc, PS in (Mpc/h)^3 and M in solar_mass/h
    NOTE: Omega_0 must match with chosen PS
    returns the concentration parameter as defined in
    https://arxiv.org/abs/2009.01858 in eq. 20
    """
    # VM-SPEEDUP BEGINS
    #
    # Quantity. The Bullock-style concentration of HMcode-2020 (Mead et al.,
    # arXiv:2009.01858, eq. 20; Dome et al., arXiv:2409.11469, eq. 49),
    #
    #   c(M, z) = B * (1 + z_f(M)) / (1 + z) ,
    #
    # optionally rescaled by the Dentler et al. (arXiv:2111.01199, eq. 33)
    # gamma correction for mixed dark matter (the `factor` below).
    #
    # Change. Exact memoization for scalar masses; no formula is altered.
    # The soliton central-density root solver (axion_density_profile.
    # func_central_density_param) evaluates its objective ~30 times per halo
    # mass, and every evaluation reaches this function — through NFW_profile
    # and func_delta_char — with identical arguments, because the solver's
    # unknown (the soliton central density) does not enter c(M). The memo
    # returns the previously computed value instead of re-running the
    # z_f lookup and the gamma correction.
    #
    # Exactness. A memoization is exact by construction when the key
    # captures every input the result depends on. The key holds the mass,
    # Omega_0, c_min, the identity of the power-spectrum array (id(PS) — the
    # arrays are built once per evaluation and never mutated), and the
    # axion_dic M_cut state that gates the gamma correction; the redshift
    # and cosmology are fixed within the memo's lifetime because the memo
    # lives in cosmo_dic ('_vm_' key, one (cosmology, z) evaluation). Cached
    # and uncached paths return byte-identical values; the fork validation
    # measured identical boost deviations before and after the memoization
    # round (dev_scripts/fork_validate.py).
    if isinstance(M, (int, float)):
        if isinstance(axion_dic, dict):
            mcut_state = axion_dic.get('M_cut', None)
        else:
            mcut_state = axion_dic  # None or 'ignore'
        memo = cosmo_dic.setdefault('_vm_conc_memo', {})
        memo_key = (float(M), round(float(Omega_0), 12), float(c_min),
                    id(PS), mcut_state)
        if memo_key in memo:
            return memo[memo_key]
    # VM-SPEEDUP ENDS
    B = c_min
    if 'gamma_1' in cosmo_dic and 'gamma_2' in cosmo_dic:
        # Implementing 2111.01199 Eq. (33) for mixed dark matter
        #Only apply correction for cold DM halos > M_cut
        if axion_dic == 'ignore':
            #print('Assuming axions affect arbitrarily low-mass cold halos')
            correction_array = np.ones_like(M)
        else:
            #cold_halo_cut = axion_dic['M_cut'] * cosmo_dic['omega_d_0'] / cosmo_dic['omega_ax_0']
            correction_array = (M > axion_dic['M_cut'])

        gamma_1 = cosmo_dic['gamma_1']
        gamma_2 = cosmo_dic['gamma_2']
        f_ax = cosmo_dic['omega_ax_0'] / (cosmo_dic['omega_ax_0']+cosmo_dic['omega_d_0'])
        M0 = 1.6e10 * (cosmo_dic['m_ax']/1e-22)**(-4/3) * cosmo_dic['h'] # to convert to Msun/h
        factor = 1 + (f_ax * (((1+gamma_1*M0/M)**(-gamma_2)) - 1.) * correction_array)
        #print('Gamma correction =', factor)
        B *= factor

    conc_param = B * (1 + func_z_formation(M, k, PS, cosmo_dic, Omega_0) )/(1+cosmo_dic['z'])

    # VM-SPEEDUP BEGINS (store scalar result in the memo built above)
    if isinstance(M, (int, float)):
        memo[memo_key] = conc_param
    # VM-SPEEDUP ENDS
    return conc_param


#function for the normalisation factor in NFW profile
def func_for_norm_factor(x):
    """
    normalisation function for the NFW profile
    """
    return (- x/(1+x)) + np.log(1+x)

#density profile in k space (fourietrafo)
def func_dens_profile_kspace(M, k, PS, cosmo_dic, hmcode_dic, Omega_0, c_min, eta_given = False, axion_dic=None):
    """
    k, k units of h/Mpc, M in solar_mass/h and PS, PS in (Mpc/h)^3 
    NOTE: Omega_0 must match with chosen PS
    returns Fourier trafo of NFW profile (dimensionless) at k as given eq 18 in https://arxiv.org/abs/2209.13445
    """
    z = cosmo_dic['z']
    Omega_m_0 = cosmo_dic['Omega_m_0']
    Omega_w_0 = cosmo_dic['Omega_w_0']
    Omega_ax_0 = cosmo_dic['Omega_ax_0']
    #eta is a halo shape parameter introduced my Mead in https://arxiv.org/abs/2009.01858 in Tab2
    if eta_given == True:
        eta = np.array([hmcode_dic['eta']])
        nu = np.atleast_1d(func_nu(M, k, PS, Omega_ax_0, Omega_0, Omega_m_0, Omega_w_0, z, cosmo_dic['G_a'], cosmo_dic['version']))
    else:
        eta = np.array([0.]) 
        nu = np.array([1.])

    R_vir = np.atleast_1d(func_r_vir(cosmo_dic['z'], M, Omega_ax_0, Omega_0, 
                                     cosmo_dic['Omega_m_0'], cosmo_dic['Omega_w_0'], cosmo_dic['G_a'], cosmo_dic['version']))
    concentration = np.atleast_1d(func_conc_param(M, k, PS, cosmo_dic, Omega_0, c_min, axion_dic=axion_dic))
    
    k_scaled = k * nu[:, None]**eta[:, None]
    k_R_vir = R_vir[:, None] * k_scaled
    a = (R_vir[:, None] / concentration[:, None]) * k_scaled
    
    def sin_integral(x):
        return scipy.special.sici(x)[0]
    def cos_integral(x):
        return scipy.special.sici(x)[1]
    
    summand1 = np.cos(a) * (cos_integral(a+k_R_vir) - cos_integral(a))
    summand2 = np.sin(a) * (sin_integral(a+k_R_vir) - sin_integral(a))
    summand3 = - np.sin(k_R_vir) / (a+k_R_vir)
    
    dens_profile_kspace = 1. / func_for_norm_factor(concentration)[:, None] * (summand1 + summand2 + summand3)
    return dens_profile_kspace 

#delta_char for the NFW profile
def func_delta_char(M, k, PS, cosmo_dic, hmcode_dic, Omega_0, c_min, eta_given = False, axion_dic=None): 
    """
    k units of h/Mpc, M in solar_mass/h and PS in (Mpc/h)^3 
    returns NFW profile in h^2 * M_sun/Mpc^3 at k in h/Mpc
    as given in  eq 17 in https://arxiv.org/abs/2209.13445
    """
    z = cosmo_dic['z']
    Omega_m_0 = cosmo_dic['Omega_m_0']
    Omega_w_0 = cosmo_dic['Omega_w_0']
    Omega_ax_0 = cosmo_dic['Omega_ax_0']
    #eta is a halo shape parameter introduced my Mead in https://arxiv.org/abs/2009.01858 in Tab2
    if eta_given == True:
        eta = hmcode_dic['eta']
        nu = func_nu(M, k, PS, Omega_ax_0, Omega_0, Omega_m_0, Omega_w_0, z, cosmo_dic['G_a'], cosmo_dic['version'])
    else:
        eta = np.array([0.]) 
        nu = 1.   
    Delta_vir = func_Delta_vir(cosmo_dic['z'], Omega_ax_0, cosmo_dic['Omega_m_0'], cosmo_dic['Omega_w_0'], cosmo_dic['G_a'], cosmo_dic['version'])
    
    concentration = func_conc_param(M, k, PS, cosmo_dic, Omega_0, c_min, axion_dic=axion_dic) / (nu**eta)
    delta_char = func_rho_comp_0(Omega_0) * Delta_vir * concentration **3 / (3. * func_for_norm_factor(concentration))
    return delta_char


#density profile in real space
def NFW_profile(M, r, k, PS, cosmo_dic, hmcode_dic, Omega_0, c_min, eta_given = False, axion_dic=None):
    """
    r in units of Mpc/h, k in h/Mpc, M in solar_mass/h PS in (Mpc/h)^3
    returns NFW denisty profile in units of solar_mass/Mpc^3*h^2 at radius r
    """
    # VM-SPEEDUP BEGINS
    #
    # Quantity. The real-space NFW profile (Vogt et al., arXiv:2209.13445,
    # eq. 17),
    #
    #   rho_NFW(r) = delta_char / [ (r/r_s) (1 + r/r_s)^2 ] ,
    #
    # with r_s = R_vir / c and delta_char fixed by the virial mass.
    #
    # Change. Exact memoization of the full radial profile array; no formula
    # is altered. Inside the soliton central-density solver the total axion
    # profile is max(soliton, NFW-shaped part), and only the soliton
    # amplitude depends on the solver's unknown — the NFW part is rebuilt on
    # an identical, deterministically constructed geomspace r grid at every
    # one of the ~30 objective evaluations per halo mass. The memo returns
    # the stored array for repeated (mass, grid) requests.
    #
    # Exactness. The key fingerprints every input the array depends on: the
    # mass, the r grid (length and both endpoints identify a geomspace grid
    # uniquely), Omega_0, c_min, the eta_given switch, the axion_dic M_cut
    # state, and the identity of the power-spectrum array (id(PS); built
    # once per evaluation, never mutated). The memo lives in cosmo_dic
    # ('_vm_' key), so its lifetime is one (cosmology, redshift) evaluation.
    # Cached and uncached paths return byte-identical arrays; the fork
    # validation measured identical boost deviations before and after the
    # memoization round (dev_scripts/fork_validate.py).
    _memo = None
    if isinstance(M, (int, float)) and np.ndim(r) == 1 and len(r) > 8:
        if isinstance(axion_dic, dict):
            _mcut_state = axion_dic.get('M_cut', None)
        else:
            _mcut_state = axion_dic
        _memo = cosmo_dic.setdefault('_vm_nfw_memo', {})
        _memo_key = (float(M), len(r), float(r[0]), float(r[-1]),
                     round(float(Omega_0), 12), float(c_min), bool(eta_given),
                     _mcut_state, id(PS))
        if _memo_key in _memo:
            return _memo[_memo_key]
    # VM-SPEEDUP ENDS
    z = cosmo_dic['z']
    Omega_m_0 = cosmo_dic['Omega_m_0']
    Omega_w_0 = cosmo_dic['Omega_w_0']
    Omega_ax_0 = cosmo_dic['Omega_ax_0']
    #eta is a halo shape parameter introduced my Mead in https://arxiv.org/abs/2009.01858 in Tab2
    if eta_given == True:
        eta = hmcode_dic['eta']
        nu = func_nu(M, k, PS, Omega_ax_0, cosmo_dic, Omega_0, Omega_m_0, Omega_w_0, z, cosmo_dic['G_a'], cosmo_dic['version'])
    else:
        eta = np.array([0.]) 
        nu = 1.
        
    concentration = func_conc_param(M, k, PS, cosmo_dic, Omega_0, c_min, axion_dic=axion_dic) / (nu**eta)
    normalisation = func_delta_char(M, k, PS, cosmo_dic, hmcode_dic, Omega_0, c_min, eta_given = eta_given, axion_dic=axion_dic)
    r_s = func_r_vir(cosmo_dic['z'], M, Omega_ax_0, Omega_0, cosmo_dic['Omega_m_0'], cosmo_dic['Omega_w_0'], 
                     cosmo_dic['G_a'], cosmo_dic['version']) / concentration
    
    NFW_func = 1 /((r/r_s) * (1+r/r_s)**2)

    # VM-SPEEDUP BEGINS (store in the memo built above)
    _result = normalisation * NFW_func
    if _memo is not None:
        _memo[_memo_key] = _result
    return _result
    # VM-SPEEDUP ENDS