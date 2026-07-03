"""functions for the density profile of axions"""

import numpy as np
from scipy import optimize, integrate
from astropy import constants as const
from cosmology.overdensities import func_r_vir
from .HMcode_params import HMCode_param_dic
from .cold_density_profile import NFW_profile

def getRhoCrit():
    """ Returns critical comoving density of the universe

    Return value is in units of [solar masses/(cMpc)^3*h^2]

    :return: critical comoving density
    :rtype: float"""
    G = 6.674*10**(-29)/(3.086*10**16)**3 # in (cMpc)^3/(kg*s^2)
    H_z = 100/(3.086*10**19) # in 1/s
    solar_mass = 2*10**30 # in kg
    return 3*H_z**2/(8*np.pi*G)/solar_mass

def getKJeans(z, OMEGA_M, little_h, m_a):
    # in cMpc^-1
    return 66.5*(1+z)**(-1/4)*(OMEGA_M*little_h**2/0.12)**(1/4)*(m_a/10**(-22))**(1/2)

def getLJeans(z, OMEGA_M, little_h, m_a):
    # assume spherical basis functions (not plane waves)
    kJeq = getKJeans(z, OMEGA_M, little_h, m_a)
    return np.pi/kJeq # cMpc

def getMJeq(z, OMEGA_M, little_h, m_a):
    # returns mass in M_sun/h
    rhocrit = getRhoCrit() # solar masses/(cMpc)^3*h^2
    lJeq = getLJeans(z, OMEGA_M, little_h, m_a) # cMpc
    lJeq = lJeq*little_h # cMpc/h
    MJeq = 4/3*np.pi*lJeq**3*rhocrit*OMEGA_M # M_sun/h
    return MJeq # M_sun/h

def MaxofMc(M_c, beta1, beta2, z, OMEGA_M, c_frac, little_h, m_a, version, M_cut, no_cut = False):
    # expects M_c in M_sun/h
    if version == 'basic':
        OMEGA_F = OMEGA_M*(1-c_frac)
        OMEGA_C = OMEGA_M*c_frac
        if no_cut == True:
            Max = OMEGA_F/OMEGA_C*M_c
        else:
            Max = OMEGA_F/OMEGA_C*M_c[M_c >=M_cut] # in M_sun/h
        return Max
    else:
        MJeq = getMJeq(z, OMEGA_M, little_h, m_a) # M_sun/h
        OMEGA_F = OMEGA_M*(1-c_frac)
        OMEGA_C = OMEGA_M*c_frac
        Max = (1 + (M_c/MJeq)**(-beta1))**(-beta2)*OMEGA_F/OMEGA_C*M_c
        return Max # in M_sun/h

def func_core_radius(M, cosmo_dic):
    """
    M in solar_mass/h, cold halo mass
    computes the core radius of the soliton given
    as in https://arxiv.org/abs/2007.08256 eq. 8
    returns r in Mpc/h
    """
    grav_const = 1.3271244e+20 #const.G.to('m**3/(Msun*s**2)').value
    M_tot = (1 + cosmo_dic['omega_ax_0']/cosmo_dic['omega_db_0']) * M/cosmo_dic['h'] # in solar_mass
    r_vir = func_r_vir(cosmo_dic['z'], (1 + cosmo_dic['omega_ax_0']/cosmo_dic['omega_db_0']) * M, cosmo_dic['Omega_ax_0'], cosmo_dic['Omega_m_0'], 
                       cosmo_dic['Omega_m_0'],  cosmo_dic['Omega_w_0'], cosmo_dic['G_a'], cosmo_dic['version']) / cosmo_dic['h'] * 3.086e+22 # in m
    v_vir = np.sqrt(grav_const*M_tot/r_vir) # in m/s
    # print(r_vir)
    
    h_bar = 1.0545718176461565e-34 #const.hbar.value
    m_ax = cosmo_dic['m_ax'] * 1.78266269594644e-36 # in kg
    r_core = 2 * np.pi * h_bar / (7.5 * m_ax * v_vir) # in m

    return r_core / (3.086e+22) * cosmo_dic['h'] * (1+cosmo_dic['z'])**(-1./2.) # in cMpc/h



def func_rho_soliton(r, M, cosmo_dic, rho_central_param):
    """
    soliton profile for axions as in https://arxiv.org/abs/1407.7762 eq.3
    but with core radius as in func_core_radius
    r in Mpc/h, M_vir in solar_mass/h and m_ax in eV
    the rho_central_param scales the central density, this is needed
    for the complete axion density profile, see eq. 47 https://arxiv.org/abs/2209.13445
    returns the soliton denity profile in solar_mass/pc^3 * h^2
    """
    m_ax = cosmo_dic['m_ax']
    z = cosmo_dic['z']
    A = (1+z) * 0.019 * (m_ax/1e-22)**(-2)
    x_c = func_core_radius(M, cosmo_dic) * 1e3 / cosmo_dic['h'] #in the formula we need units kpc
    r_in_formula = r * 1e3 / cosmo_dic['h']  #in the formula we need units kpc

    if isinstance(M, (int, float)) == True:
        return A * rho_central_param / ( x_c**4 * (1 + 0.091 * (r_in_formula/x_c)**2)**8 )\
               * cosmo_dic['h']**2 * 1e18 #transform from solar_mass/pc^3 to solar_mass/pc^3 * h^2
    else:
        return A * rho_central_param / ( np.outer(x_c, np.ones(len(r))) **4 * (1 + 0.091 * np.outer(1/x_c, r_in_formula)**2)**8 ) \
               * cosmo_dic['h']**2 * 1e18 #transform from solar_mass/pc^3 to solar_mass/pc^3 * h^2hubble units

    


def func_dens_profile_ax(r_arr, M, cosmo_dic, power_spec_dic, rho_central_param, hmcode_dic, concentration_param=False, eta_given=False, axion_dic=None):
    """
    r_arr in Mpc/h, M in solar_mass/h
    returns the axion density profile
    with a solition core and a NFW profile in the
    outer region and with the free patameter
    rho_central_param. This free parameter is set such that
    we get the correct mass of the soliton halo,
    see func_central_density_param
    the density profile has units solar_mass/Mpc^3 * h^2
    """
    # define the concentraion param for the cold matter profile
    if concentration_param == True:
        c_min = hmcode_dic['c_min']
    else:
        c_min = 4.
    #distinguish whether M is an array or a scalar
    if isinstance(M, (int, float)) == True:
        #there is no axion halo, if the cold halo is below a cut-off
        if rho_central_param == 0:
            if isinstance(r_arr, (int, float)) == True:
                return 0.0
            else:
                return np.zeros(len(r_arr))
        else:
            NFW = cosmo_dic['omega_ax_0']/cosmo_dic['omega_db_0'] * \
                          NFW_profile(M, r_arr, power_spec_dic['k'], power_spec_dic['power_cold'], cosmo_dic, 
                                      hmcode_dic, cosmo_dic['Omega_db_0'], 
                                      c_min, eta_given = eta_given, axion_dic=axion_dic)
            soliton = func_rho_soliton(r_arr, M, cosmo_dic, rho_central_param)

            idx_arr = np.argwhere(np.diff(np.sign(NFW - soliton))).flatten() #found the intersection points
            if len(idx_arr)<=0:
                return soliton
            else:
                return np.where(r_arr > r_arr[idx_arr[-1]], NFW, soliton)

    else:
        return_arr = []
        for idx, m in enumerate(M):
            if rho_central_param[idx] == 0:
                if isinstance(r_arr, (int, float)) == True:
                    return_arr.append(0.0)
                else:
                    return_arr.append(np.zeros(len(r_arr)))
            else:
                NFW = cosmo_dic['omega_ax_0']/cosmo_dic['omega_db_0'] * \
                              NFW_profile(m, r_arr, power_spec_dic['k'], power_spec_dic['power_cold'], cosmo_dic, 
                                          hmcode_dic, cosmo_dic['Omega_db_0'], 
                                          c_min, eta_given = eta_given, axion_dic=axion_dic)
                soliton = func_rho_soliton(r_arr, m, cosmo_dic, rho_central_param[idx])

                idx_arr = np.argwhere(np.diff(np.sign(NFW - soliton))).flatten() #found the intersection points
                if len(idx_arr)<=0:
                    return_arr.append(soliton)
                else:
                    return_arr.append(np.where(r_arr > r_arr[idx_arr[-1]], NFW, soliton))
        return return_arr
                
        
def func_ax_halo_mass(M, cosmo_dic, power_spec_dic, rho_central_param, hmcode_dic, concentration_param=False, eta_given=False, axion_dic=None):
    """
    M in solar_mass/h
    The free parameter rho_central_param is set such that
    we get the correct mass of the soliton halo,
    see func_central_density_param
    returns the axion halo mass by integrating the halo
    density profile in units of solar_mass/h
    """
    # VM-SPEEDUP BEGINS
    #
    # Quantity. The axion halo mass by integration of the composed
    # soliton + NFW axion profile out to the virial radius,
    #
    #   M_ax = 4 pi int_0^{r_vir} rho_ax(r) r^2 dr .
    #
    # Upstream evaluation. rho_ax was sampled on a 2000-point geomspace
    # grid in [1e-15, r_vir] and integrated with scipy.integrate.simpson.
    # This function is the objective of the soliton central-density root
    # solver (func_central_density_param), which evaluates it ~30 times per
    # halo mass, so simpson's per-call bookkeeping (~60 us of validation,
    # slicing and masked divisions per call) recurs ~3000 times per
    # redshift and dominates the solver's cost.
    #
    # Fork evaluation. The identical grid with precomputed quadrature
    # weights (fast_tables.geom_simpson_grid): the integral becomes the
    # plain dot product 4 pi sum_i w_i rho_ax(r_i) r_i^2. The weights
    # implement the canonical unevenly-spaced composite Simpson rule
    # (parabola pairs, trapezoid on the leftover final interval) — textbook
    # mathematics with no scipy call in the hot loop and hence no
    # dependence on scipy's per-version last-interval policy.
    #
    # The grid must be upstream's, not a better quadrature's. The profile
    # composition inside func_dens_profile_ax selects the soliton -> NFW
    # crossover from sign changes of the sampled difference, so the
    # composed profile depends on where the nodes fall. A Gauss-Legendre
    # rewrite in log radius (measured before this design was adopted) is
    # spectrally accurate on each smooth branch, yet flips the crossover
    # detection for grazing halos as the node count changes, moving the
    # final boost non-monotonically at the 1e-3 level — a behavior change
    # relative to upstream, not a quadrature error. Keeping upstream's
    # geomspace nodes keeps the composition, the crossover snapping, and
    # the solver's rejection behavior bit-identical; only the summation
    # rule applied to the identical samples changes.
    #
    # Accuracy. The rule difference from scipy's simpson is confined to the
    # outermost radial interval (trapezoid here vs scipy >= 1.11's
    # last-interval parabola) and is orders of magnitude below the fork
    # validation gate max |dB/B| <= 1e-4 (dev_scripts/fork_validate.py).
    # The node count is pinned at upstream's 2000 and excluded from the
    # accuracy_boost scaling: doubling it re-snaps the crossover and flips
    # marginal rejections, moving the boost by up to ~7e-2 at high k
    # (measured) — model behavior tied to the released grid, not
    # quadrature convergence (see fast_tables.geom_simpson_grid).
    from cosmology import fast_tables
    #distinguish whether M is an array or a scalar
    if isinstance(M, (int, float)) == True:
        r_vir = func_r_vir(cosmo_dic['z'], M, cosmo_dic['Omega_ax_0'], cosmo_dic['Omega_db_0'], cosmo_dic['Omega_m_0'],
                           cosmo_dic['Omega_w_0'], cosmo_dic['G_a'], cosmo_dic['version'])
        r_arr, w_arr = fast_tables.geom_simpson_grid(1e-15, r_vir, cosmo_dic)
        integrand = func_dens_profile_ax(r_arr, M, cosmo_dic, power_spec_dic, rho_central_param, hmcode_dic,
                                         concentration_param=concentration_param, eta_given=eta_given, axion_dic=axion_dic) * r_arr**2
        return 4 * np.pi * np.dot(w_arr, integrand)
    else:
        integral = np.zeros(len(M))
        for i in range(len(M)):
            upper_bound = func_r_vir(cosmo_dic['z'], M[i], cosmo_dic['Omega_ax_0'],
                                     cosmo_dic['Omega_db_0'], cosmo_dic['Omega_m_0'],
                                     cosmo_dic['Omega_w_0'], cosmo_dic['G_a'], cosmo_dic['version'])
            r_arr, w_arr = fast_tables.geom_simpson_grid(1e-15, upper_bound, cosmo_dic)
            integral[i] = 4 * np.pi * np.dot(w_arr, func_dens_profile_ax(r_arr, M[i], cosmo_dic, power_spec_dic, rho_central_param[i], hmcode_dic,
                                                                         concentration_param=concentration_param, eta_given=eta_given, axion_dic=axion_dic)*r_arr**2)

        return integral
    # VM-SPEEDUP ENDS (upstream body, kept as the reference implementation)
    if isinstance(M, (int, float)) == True:
        r_vir = func_r_vir(cosmo_dic['z'], M, cosmo_dic['Omega_ax_0'], cosmo_dic['Omega_db_0'], cosmo_dic['Omega_m_0'],
                           cosmo_dic['Omega_w_0'], cosmo_dic['G_a'], cosmo_dic['version'])
        r_arr = np.geomspace(1e-15, r_vir, num=2000)
        integrand = func_dens_profile_ax(r_arr, M, cosmo_dic, power_spec_dic, rho_central_param, hmcode_dic,
                                         concentration_param=concentration_param, eta_given=eta_given, axion_dic=axion_dic) * r_arr**2
        return 4 * np.pi * integrate.simpson(y=integrand, x = r_arr)
    else:
        integral = np.zeros(len(M))
        for i in range(len(M)):
            upper_bound = func_r_vir(cosmo_dic['z'], M[i], cosmo_dic['Omega_ax_0'],
                                     cosmo_dic['Omega_db_0'], cosmo_dic['Omega_m_0'],
                                     cosmo_dic['Omega_w_0'], cosmo_dic['G_a'], cosmo_dic['version'])
            R_int = np.geomspace(1e-15, upper_bound, num=2000)
            integral[i] = 4 * np.pi * integrate.simpson(y=func_dens_profile_ax(R_int, M[i], cosmo_dic, power_spec_dic, rho_central_param[i], hmcode_dic,
                                                                               concentration_param=concentration_param, eta_given=eta_given, axion_dic=axion_dic)*R_int**2, x=R_int)

        return integral
        

def func_central_density_param(M, cosmo_dic, power_spec_dic, concentration_param=False, eta_given=False, axion_dic=None):
    """
    M in solar_mass/h
    The central density of the soliton profile
    has to be change in such a way that the total
    mass of the axion halo matches the target value,
    ie M_ax_halo = MaxofMc(Mc)
    """
    #distinguish whether M is an array or a scalar
    hmcode_dic = HMCode_param_dic(cosmo_dic, power_spec_dic['k'], power_spec_dic['power_cold'])
    c_frac = 1 - cosmo_dic['omega_ax_0']/cosmo_dic['omega_m_0'] # 1 - ax_frac = 1 - f
    if isinstance(M, (int, float)) == True:
        r_c = func_core_radius(M, cosmo_dic) 
        
        #need a gues to find the correct central_dens_param:
        #guess is set via Omega_ax/Omega_cold * M = int_0_rvir \rho *r^2 dr
        #so we need the soliton and NFW part
        def integrand_ax(x):
            return func_dens_profile_ax(x, M, cosmo_dic, power_spec_dic, 1., concentration_param=concentration_param, eta_given=eta_given, axion_dic=axion_dic)*x**2
        #integral_soliton = integrate.quad(integrand_ax, 0, r_c)[0] # 
        
        r_arr = np.geomspace(1e-15 , r_c, 1000)
        integrand_cold = NFW_profile(M, r_arr, power_spec_dic['k'], power_spec_dic['cold'], cosmo_dic, hmcode_dic, cosmo_dic['Omega_db_0'], 
                                        hmcode_dic['c_min'], eta_given = eta_given, axion_dic=axion_dic) \
                            *r_arr**2
        integral_NFW = integrate.simps(y=integrand_cold, x = r_arr)
        integral_NFW = MaxofMc(integral_NFW, axion_dic['beta1'], axion_dic['beta2'], cosmo_dic['z'], cosmo_dic['omega_m_0'], 
                               c_frac, cosmo_dic['h'], cosmo_dic['m_ax'], cosmo_dic['version'], axion_dic['M_cut'], no_cut = True)

        guess = integral_NFW / integral_soliton
        
        #find the central density parameter
        def func_find_root(dens):
            return func_ax_halo_mass(M, cosmo_dic, power_spec_dic, dens, hmcode_dic, concentration_param=concentration_param, eta_given=eta_given, axion_dic=axion_dic) - MaxofMc(M, axion_dic['beta1'], axion_dic['beta2'], cosmo_dic['z'], cosmo_dic['omega_m_0'], c_frac, cosmo_dic['h'], cosmo_dic['m_ax'], cosmo_dic['version'], axion_dic['M_cut'])
        
        
        dens_param = optimize.root(func_find_root, x0 = guess).x
        #sometimes the solution is not really a solution,
        #so set than the central density paameter to zero, ie no solution can be found
        if np.abs(guess - dens_param) > 100.:
            return 0.
        else:
            return float(dens_param)
            
    else:
        dens_param_arr = []
        r_c = func_core_radius(M, cosmo_dic)
        for idx, m in enumerate(M):

            #need a gues to find the correct central_dens_param:
            #guess is set via Omega_ax/Omega_cold * M = int_0_rvir \rho *r^2 dr
            #so we need the soliton and NFW part
            def integrand_ax(x):
                return func_dens_profile_ax(x, m, cosmo_dic, power_spec_dic, 1., hmcode_dic, concentration_param=concentration_param, eta_given=eta_given, axion_dic=axion_dic)*x**2

            integral_soliton = integrate.quad(integrand_ax, 0, r_c[idx])[0]

            r_arr = np.geomspace(1e-15 , r_c[idx], 1000)
            integrand_cold = NFW_profile(m, r_arr, power_spec_dic['k'], power_spec_dic['power_cold'], cosmo_dic, hmcode_dic, cosmo_dic['Omega_db_0'], 
                                            hmcode_dic['c_min'], eta_given = eta_given, axion_dic=axion_dic)*r_arr**2 

            integral_NFW = integrate.simpson(y=integrand_cold, x = r_arr)
            integral_NFW = MaxofMc(integral_NFW, axion_dic['beta1'], axion_dic['beta2'], cosmo_dic['z'], cosmo_dic['omega_m_0'], 
                                   c_frac, cosmo_dic['h'], cosmo_dic['m_ax'], cosmo_dic['version'], axion_dic['M_cut'], no_cut = True)
            guess = integral_NFW / integral_soliton
            
            #find the central density parameter
            def func_find_root(dens):
                return func_ax_halo_mass(m, cosmo_dic, power_spec_dic, dens, hmcode_dic, concentration_param=concentration_param, eta_given=eta_given, axion_dic=axion_dic) - MaxofMc(m, axion_dic['beta1'], axion_dic['beta2'], cosmo_dic['z'], cosmo_dic['omega_m_0'], c_frac, cosmo_dic['h'], cosmo_dic['m_ax'], cosmo_dic['version'], axion_dic['M_cut'])
            dens_param = optimize.root(func_find_root, x0 = guess).x
            
            #sometimes the solution is not really a solution,
            #so set than the central density paameter to zero, ie so solution can be found
            if np.abs(guess - dens_param) > 100:
                dens_param_arr.append(0.)
            else:
                dens_param_arr.append(float(dens_param))
                
        return dens_param_arr



def func_dens_profile_ax_kspace(k, M, cosmo_dic, power_spec_dic, central_dens_param, hmcode_dic, concentration_param=False, eta_given=False, axion_dic=None):
    """
    k in units of h/Mpc and M in solar_mass/h
    The free parameter central_dens_param is set such that
    we get the correct mass of the soliton halo,
    see func_central_density_param
    return kspace denisty profile for the axion halo
    the normalised density profile is demensionles
    """
    #the kspace density profile is defined via
    # \rho(k) = 4*\pi* int_0^r_vir \rho(r) * r^2 * sin(kr)/kr dr
    M_ax = func_ax_halo_mass(M, cosmo_dic, power_spec_dic, central_dens_param, hmcode_dic, concentration_param=concentration_param, eta_given=eta_given, axion_dic=axion_dic)
    r_vir = func_r_vir(cosmo_dic['z'], M, cosmo_dic['Omega_ax_0'], cosmo_dic['Omega_db_0'],
                       cosmo_dic['Omega_m_0'], cosmo_dic['Omega_w_0'], cosmo_dic['G_a'], cosmo_dic['version'])

    # VM-SPEEDUP BEGINS
    #
    # Quantity. The normalized k-space axion halo profile
    #
    #   u(k, M) = (4 pi / M_ax) int_0^{r_vir} rho_ax(r) r^2 sin(kr)/(kr) dr,
    #
    # the spherical Bessel j0 transform of the composed soliton + NFW
    # profile, needed by the one-halo and cross terms of the halo model.
    #
    # Upstream evaluation. For each halo mass, the kernel sin(kr)/(kr) was
    # materialized as a dense (n_k x 2000) outer-product array, multiplied
    # by the profile, and reduced with scipy.integrate.simpson along r —
    # an O(n_k * n_r) direct Hankel transform per mass, dominated by the
    # ~2e6 transcendental evaluations of the kernel (measured ~0.38 s per
    # redshift, ~45% of the fork's round-2 cost).
    #
    # Fork evaluation. FFTLog on the identical samples (fast_tables.
    # fftlog_j0_grid / fftlog_j0_eval; the algorithm of cosmolike's cfftlog
    # in cosmo2D.c with the Bessel order pinned to zero, so no per-k kernel
    # loop exists at all): the log-uniform radial grid is exactly FFTLog's
    # native format, and one forward + one inverse FFT per mass yield the
    # transform at every k simultaneously in O(n_r log n_r). The radial
    # samples, and hence the profile composition and its crossover
    # snapping, are bit-identical to upstream's (the grid is pinned for
    # the reasons documented at func_ax_halo_mass); only the reduction of
    # those samples changes. The truncation edge at r_vir is handled by a
    # continuous constant extension through the zero padding plus the
    # analytic subtraction of its tail (closed form in Si/Ci — the same
    # scipy.special.sici upstream uses for the cold NFW profile), and the
    # padded length is a power of two both for FFT efficiency and to push
    # the reciprocal k grid ~36 e-folds below 1/r_vir, covering every
    # target k in one transform.
    #
    # Accuracy. Measured against upstream's dense-Simpson rule on the same
    # samples: max |dI|/I(k->0) <= 2e-5 over k in [1e-4, 15] h/cMpc
    # (dev_scripts prototype scan; insensitive to the bias, window, and
    # padded length). End-to-end agreement is bounded by the fork
    # validation gate max |dB/B| <= 1e-4 (dev_scripts/fork_validate.py).
    from cosmology import fast_tables
    k_arr = np.asarray(k, dtype=np.float64)
    if isinstance(M, (int, float)) == True:
        tab = fast_tables.fftlog_j0_grid(1e-15, r_vir, cosmo_dic)
        rho_arr = func_dens_profile_ax(tab[0], M, cosmo_dic, power_spec_dic, central_dens_param, hmcode_dic,
                                       concentration_param=concentration_param, eta_given=eta_given, axion_dic=axion_dic)
        return list(4 * np.pi * fast_tables.fftlog_j0_eval(tab, rho_arr, k_arr) / M_ax)
    else:
        dens_profile_kspace_arr = []
        for idx, m in enumerate(M):
            if M_ax[idx] == 0:
                dens_profile_kspace_arr.append(list(np.zeros(len(k))))
            else:
                tab = fast_tables.fftlog_j0_grid(1e-15, r_vir[idx], cosmo_dic)
                rho_arr = func_dens_profile_ax(tab[0], m, cosmo_dic, power_spec_dic, central_dens_param[idx], hmcode_dic,
                                               concentration_param=concentration_param, eta_given=eta_given, axion_dic=axion_dic)
                dens_profile_kspace_arr.append(
                    list(4 * np.pi * fast_tables.fftlog_j0_eval(tab, rho_arr, k_arr) / M_ax[idx]))
        return dens_profile_kspace_arr
    # VM-SPEEDUP ENDS (upstream body, kept as the reference implementation)
    if isinstance(M, (int, float)) == True:
        r_arr = np.geomspace(1e-15, r_vir, num=2000)
        dens_profile_arr = func_dens_profile_ax(r_arr, M, cosmo_dic, power_spec_dic, central_dens_param, hmcode_dic,
                                                concentration_param=concentration_param, eta_given=eta_given, axion_dic=axion_dic) \
                           * r_arr**2 * np.sin(np.outer(k, r_arr)) / np.outer(k, r_arr)
        return list(4 * np.pi * integrate.simpson(y=dens_profile_arr, x = r_arr, axis=-1) / M_ax)

    else:
        dens_profile_kspace_arr = []
        for idx, m in enumerate(M):
            if  M_ax[idx] == 0:
                dens_profile_kspace_arr.append(list(np.zeros(len(k))))
            else:
                r_arr = np.geomspace(1e-15, r_vir[idx], num=2000)
                dens_profile_arr = func_dens_profile_ax(r_arr, m, cosmo_dic, power_spec_dic, central_dens_param[idx], hmcode_dic,
                                                        concentration_param=concentration_param, eta_given=eta_given, axion_dic=axion_dic) \
                                   * r_arr**2 * np.sin(np.outer(k, r_arr)) / np.outer(k, r_arr)
                dens_kspace = list(4 * np.pi * integrate.simpson(y=dens_profile_arr, x = r_arr, axis=-1) / M_ax[idx] )
                dens_profile_kspace_arr.append(dens_kspace)

        return dens_profile_kspace_arr
