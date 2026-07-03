# VM-SPEEDUP BEGINS (whole file)
"""Lookup tables for quantities axionHMcode recomputes from scratch at every
call (see the fork README appendix for the full mathematical documentation).

Motivation. Profiling one halo-model evaluation (one cosmology, one redshift)
shows that ~85% of the runtime is spent in the axion parameter stage, whose
nested root solvers re-derive three smooth one-variable functions at the
innermost point of every iteration: the linear growth factor D(z) is
re-integrated ~1.1e5 times (each call also re-integrates its constant
normalization D(0)), the mass variance sigma(M) is re-integrated over the
whole k grid ~1e5 times, and the HMcode-2020 integrated growth G(z) is
evaluated with an adaptive double quadrature. All three are ideal
interpolation targets: smooth, monotone or near-power-law, and needed over
fixed, known ranges. This module builds each of them once and interpolates.

The three tables:

1. Growth factor. Upstream (overdensities.func_D_z_unnorm) evaluates
       D(z) = (5/2) Omega_m E(z) int_z^{100} dx (1+x)/E(x)^3
   with a 2000-point trapezoid rule per call. `growth_tables` evaluates that
   same upstream function on a dense linear grid in z over [0, 100] (the
   interval upstream's brentq searched) and stores D(z)/D(0).
   `D_norm_fast` interpolates it linearly; `z_of_D_norm` inverts it, which
   is well defined because D(z)/D(0) is strictly decreasing in z.

2. Integrated growth. `G_integral_fast` evaluates HMcode-2020 eq. A5,
       G(z) = (5/2) Omega_m int_z^{z_max} dx E(x)/(1+x)
                            int_x^{z_max} dy (1+y)/E(y)^3 ,
   by observing that the inner integral I(x) and the outer integral are both
   cumulative from z_max downward: on one dense node grid, two reversed
   cumulative trapezoid passes produce G at every node simultaneously.
   This replaces scipy.integrate.dblquad (~0.15 s per call).

3. Mass variance. Upstream (variance.func_sigma_M) evaluates
       sigma^2(M) = 1/(2 pi^2) int dln k P(k) W^2(k R(M)) k^3 ,
   with W the spherical top-hat window and R(M) = (3M / 4 pi rho_bar)^(1/3),
   as a full k-grid trapezoid integral per call. `sigma_M_fast` evaluates
   that same upstream function at log-spaced mass nodes and interpolates
   log sigma linearly in log10 M (sigma(M) is close to a power law in M, so
   the log-log interpolant is nearly exact); outside the tabulated mass
   range it falls back to the exact upstream evaluation.

Design rules:
- Table nodes are computed with the original, unmodified upstream functions,
  so the tables reproduce upstream numerics at the nodes exactly; between
  nodes the interpolation error is bounded by the fork validation gate
  (max |dB/B| = 1.6e-5 on the boost, primitives <= 1.6e-6; see
  dev_scripts/fork_validate.py in the AxiECAMB boost package).
- Node counts are scaled by `set_accuracy_boost` (wired to the boost
  theory's yaml `accuracy_boost` option); running the same setup at 1 and 2
  and comparing boost grids is the convergence test for every table here.
- Per-evaluation state lives inside cosmo_dic (keys prefixed '_vm_'), so
  those caches have exactly the lifetime of one (cosmology, redshift)
  evaluation and multiprocessing forks behave trivially. Only the growth and
  G tables, which depend on (Omega_m_0, Omega_w_0) alone, use a small
  module-level cache keyed on the cosmology and the accuracy boost.
"""

import numpy as np

from .overdensities import func_D_z_unnorm
from .basic_cosmology import func_E_z
from cosmology import variance as _variance

_GROWTH_NODES = 4001          # z in [0, 100], linear (brentq search range)
_G_NODES = 20000              # y in [z_min, 10000] for the eq. A5 integrals
_SIGMA_NODES = 321            # log10 M in [3, 19.5]
_SIGMA_LOG10M_MIN = 3.0
_SIGMA_LOG10M_MAX = 19.5
_R_GRID_NODES = 2000          # radial nodes of the soliton mass integral
                              # (upstream's geomspace point count; the grid
                              # must match upstream exactly, see
                              # geom_simpson_grid)
_FFTLOG_PAD = 4096            # padded FFT length for the profile transform:
                              # a power of two (FFT-friendly small primes)
                              # whose extra ~36 e-folds of back padding also
                              # push the reciprocal k grid far below 1/r_vir
_FFTLOG_BIAS = 0.9            # Mellin bias q, inside the j0 strip (0, 2)
_FFTLOG_WINDOW = 0.25         # tapered fraction of Fourier modes (the
                              # c_window of cosmolike's cfftlog)

# global node-count multiplier, set from the boost theory's yaml
# `accuracy_boost` option; 1.0 = the counts the fork was validated with.
# Running the same yaml at 1 and 2 and comparing boost grids is the
# convergence check for every table in this module.
_ACCURACY_BOOST = 1.0


def set_accuracy_boost(boost):
  """Scale all table node counts by `boost` (> 0); tables rebuild lazily."""
  global _ACCURACY_BOOST
  boost = float(boost)
  if not boost > 0:
    raise ValueError(f"accuracy_boost must be > 0, got {boost!r}")
  _ACCURACY_BOOST = boost


def _nodes(base):
  return max(51, int(round(base * _ACCURACY_BOOST)))


_growth_cache = {}


def _growth_key(Omega_m_0, Omega_w_0):
  return (round(float(Omega_m_0), 12), round(float(Omega_w_0), 12),
          round(_ACCURACY_BOOST, 6))


def growth_tables(Omega_m_0, Omega_w_0):
  """(z_grid, D_norm(z_grid)) with D_norm(0) = 1, nodes from upstream
  func_D_z_unnorm (same 2000-point trapz quadrature per node)."""
  key = _growth_key(Omega_m_0, Omega_w_0)
  tab = _growth_cache.get(key)
  if tab is None:
    zg = np.linspace(0.0, 100.0, _nodes(_GROWTH_NODES))
    D = np.array([func_D_z_unnorm(z, Omega_m_0, Omega_w_0) for z in zg])
    tab = (zg, D / D[0])
    _growth_cache[key] = tab
  return tab


def D_norm_fast(z, Omega_m_0, Omega_w_0):
  """Interpolated D(z)/D(0); scalar or array z in [0, 100]."""
  zg, Dn = growth_tables(Omega_m_0, Omega_w_0)
  return np.interp(z, zg, Dn)


def z_of_D_norm(target, Omega_m_0, Omega_w_0):
  """Inverse of the monotone-decreasing D_norm on z in [0, 100]."""
  zg, Dn = growth_tables(Omega_m_0, Omega_w_0)
  # np.interp needs ascending x: D_norm decreases with z
  return np.interp(target, Dn[::-1], zg[::-1])


def G_integral_fast(z, Omega_m_0, Omega_w_0):
  """HMcode-2020 eq. A5 integrated growth
      G(z) = (5 Omega_m / 2) int_z^10000 dx E(x)/(1+x) int_x^10000 dy (1+y)/E(y)^3
  evaluated from one dense table per cosmology (replaces scipy.dblquad;
  agreement with the upstream dblquad validated in the fork test suite)."""
  key = ("G",) + _growth_key(Omega_m_0, Omega_w_0)
  tab = _growth_cache.get(key)
  if tab is None:
    y = np.geomspace(1e-4, 10000.0, _nodes(_G_NODES)) - 1e-4  # dense near 0, reaches 0
    Ey = func_E_z(y, Omega_m_0, Omega_w_0)
    inner_integrand = (1.0 + y) / Ey**3
    # I(x) = int_x^10000 dy inner_integrand: reverse cumulative trapezoid
    seg = 0.5 * (inner_integrand[1:] + inner_integrand[:-1]) * np.diff(y)
    I_of_x = np.concatenate((np.cumsum(seg[::-1])[::-1], [0.0]))
    outer_integrand = Ey / (1.0 + y) * I_of_x
    seg2 = 0.5 * (outer_integrand[1:] + outer_integrand[:-1]) * np.diff(y)
    G_of_z = 2.5 * Omega_m_0 * np.concatenate((np.cumsum(seg2[::-1])[::-1],
                                               [0.0]))
    tab = (y, G_of_z)
    _growth_cache[key] = tab
  yg, Gg = tab
  return float(np.interp(z, yg, Gg)) if np.isscalar(z) else np.interp(z, yg, Gg)


def geom_simpson_grid(r_min, r_max, cosmo_dic):
  """Upstream's radial grid together with precomputed composite-Simpson
  weights, so that int f(r) dr ~= sum_i w_i f(r_i) is a plain dot product.

  Grid. r = np.geomspace(r_min, r_max, n) with n = _nodes(_R_GRID_NODES) —
  at accuracy boost 1 this is the identical call upstream makes, so the
  nodes are bit-identical to upstream's. That is not an optimization detail
  but a correctness requirement: the axion profile composition
  (halo_model.axion_density_profile.func_dens_profile_ax) selects the
  soliton -> NFW crossover by sign changes of the sampled difference, so
  the composed profile is a function of the node positions. Integrating on
  any other grid (measured with Gauss-Legendre nodes) flips the crossover
  detection for grazing halos and changes the boost at the 1e-3 level —
  a behavior change relative to upstream, not a quadrature error.

  Weights. Composite Simpson's rule for unevenly spaced samples: for each
  pair of adjacent intervals (h0, h1) the unique parabola through the three
  points integrates to

      (h0+h1)/6 * [ (2 - h1/h0) f_i
                    + (h0+h1)^2/(h0 h1) f_{i+1}
                    + (2 - h0/h1) f_{i+2} ] ,

  which reduces to the textbook (h/3)(f_i + 4 f_{i+1} + f_{i+2}) for
  h0 = h1. With n = 2000 points there are 1999 intervals; the pairs cover
  the first 1998 and the final interval is closed with a trapezoid. Both
  pieces are canonical mathematics, so the weights carry no library
  version dependence — deliberately unlike scipy's simpson, whose
  even-point-count last-interval policy is a per-version choice (changed
  in scipy 1.11, compatibility keyword removed in 1.14). The difference
  from scipy's current last-interval parabola is confined to the outermost
  interval and is orders of magnitude below the fork validation gate.

  The node count is pinned at upstream's 2000 and deliberately excluded
  from the set_accuracy_boost scaling: because the crossover snapping and
  the solver's rejection heuristic are tied to the node positions, changing
  this grid re-composes grazing halos discretely (measured: up to
  ~7e-2 in max |dB/B| at k > 8 h/cMpc when doubling it). That sensitivity
  is a property of the released, calibrated model — the Dome et al.
  calibration ran with this grid — so the grid is treated as part of the
  model definition, like its fitted constants, and not as a numerical
  convergence parameter of this fork.

  The weight vector is cached per (n, r_min, r_max) in cosmo_dic ('_vm_'
  convention, one (cosmology, redshift) evaluation lifetime).
  """
  n = _R_GRID_NODES
  cache = cosmo_dic.setdefault("_vm_simw_cache", {})
  key = (n, float(r_min), float(r_max))
  tab = cache.get(key)
  if tab is None:
    r = np.geomspace(r_min, r_max, num=n)
    h = np.diff(r)
    w = np.zeros(n)
    # parabola pairs over intervals (0,1), (2,3), ...
    m = (n - 1) - (n - 1) % 2          # number of paired intervals
    h0, h1 = h[0:m:2], h[1:m:2]
    hs = h0 + h1
    w[0:m:2] += hs / 6.0 * (2.0 - h1 / h0)
    w[1:m:2] += hs / 6.0 * hs * hs / (h0 * h1)
    w[2:m + 1:2] += hs / 6.0 * (2.0 - h0 / h1)
    if m < n - 1:                      # leftover interval: trapezoid
      w[-2] += 0.5 * h[-1]
      w[-1] += 0.5 * h[-1]
    tab = (r, w)
    cache[key] = tab
  return tab


def _mellin_j0(s):
  """Mellin transform of the spherical Bessel function j0,
      int_0^inf x^(s-1) j0(x) dx = 2^(s-2) sqrt(pi) Gamma(s/2) / Gamma((3-s)/2),
  valid on the strip 0 < Re s < 2; evaluated via loggamma for complex s."""
  from scipy.special import loggamma
  return np.exp((s - 2.0) * np.log(2.0) + 0.5 * np.log(np.pi)
                + loggamma(s / 2.0) - loggamma((3.0 - s) / 2.0))


def fftlog_j0_grid(r_min, r_max, cosmo_dic):
  """Everything k-independent for the FFTLog evaluation of

      I(k) = int_{r_min}^{r_max} rho(r) r^2 j0(k r) dr
           = int dr/r F(r) j0(k r),        F(r) = rho(r) r^3,

  on the pinned upstream radial grid r = geomspace(r_min, r_max, n) (the
  same nodes as geom_simpson_grid, so the profile composition is
  bit-identical to upstream's; see that docstring for why the grid is
  pinned). FFTLog (Hamilton 2000): on a log-uniform grid the biased samples
  F(r_n) r_n^(-q) are Fourier-decomposed into power laws r^(q + i eta_m),
  each of which transforms analytically through the Mellin kernel of j0, so
  one forward and one inverse FFT yield I(k) on the reciprocal log-uniform
  k grid — for every k at once. This is the same machinery as cosmolike's
  cfftlog (cosmo2D.c) with the Bessel order fixed to zero, which removes
  the per-multipole kernel loop entirely.

  Edge treatment (validated in the round-3 prototype): the profile is
  truncated at r_max where F is at its maximum, and a bare Fourier
  representation integrates that jump O(dlnr) differently from Simpson
  (~5e-3). The padding is therefore filled with the constant F(r_max) — a
  continuous extension whose biased amplitude decays by e^(-36 q) ~ 1e-14
  long before the periodic wrap — and the extension's exact contribution is
  subtracted analytically,

      int_{r_max}^inf F(r_max) j0(k r) dr/r
          = F(r_max) [ sin(a)/a - Ci(a) ],   a = k r_max,

  with Ci from scipy.special.sici (the same special function upstream uses
  for the cold NFW k-space profile). Together with cubic interpolation of
  I onto the target k (the k grid is uniform in ln k, and linear
  interpolation of the oscillatory tail was the dominant residual), the
  measured agreement with dense-Simpson-on-the-same-samples is
  max |dI|/I(0) <= 5e-6 over k in [1e-4, 15] h/cMpc, insensitive to the
  bias, window, and padded length (dev_scripts prototype scan).

  Returns (r, ext_bias, kernel, lnk0, dlnr): the radial nodes, the biased
  constant-extension row, the combined Mellin x phase x taper kernel, and
  the reciprocal-grid geometry. Cached per grid in cosmo_dic ('_vm_'
  convention). The Fourier-mode count is n_fft//2 + 1 with
  n_fft = _FFTLOG_PAD (its loggamma evaluation is the dominant setup cost,
  paid once per (mass, redshift) grid).
  """
  n = _R_GRID_NODES
  n_fft = _FFTLOG_PAD
  q = _FFTLOG_BIAS
  cache = cosmo_dic.setdefault("_vm_fftlog_cache", {})
  key = (n, n_fft, float(r_min), float(r_max))
  tab = cache.get(key)
  if tab is None:
    r = np.geomspace(r_min, r_max, num=n)
    dlnr = np.log(r[-1] / r[0]) / (n - 1)
    r_full = r[0] * np.exp(np.arange(n_fft) * dlnr)
    data_bias = r**(-q)
    ext_bias = r_full[n:]**(-q)
    m = np.arange(n_fft // 2 + 1)
    eta = 2.0 * np.pi * m / (n_fft * dlnr)
    lnk0 = -np.log(r_full[-1])            # k_j = e^{lnk0 + j dlnr}
    kernel = (_mellin_j0(q + 1j * eta)
              * np.exp(-1j * eta * (lnk0 + np.log(r[0]))))
    m_cut = int((1.0 - _FFTLOG_WINDOW) * (len(m) - 1))
    hi = m > m_cut
    taper = np.ones(len(m))
    taper[hi] = 0.5 * (1.0 + np.cos(np.pi * (m[hi] - m_cut)
                                    / (len(m) - 1 - m_cut)))
    kernel = np.conj(kernel * taper)      # conj once here, not per call
    tab = (r, data_bias, ext_bias, kernel, lnk0, dlnr)
    cache[key] = tab
  return tab


def fftlog_j0_eval(tab, rho, k_targets):
  """I(k_targets) = int rho(r) r^2 j0(k r) dr from the grid machinery of
  fftlog_j0_grid and the profile samples rho(r). Two FFTs plus the analytic
  edge tail and a cubic interpolation in ln k (uniform grid, closed-form
  4-point Lagrange weights).

  The transforms go through scipy.fft (pocketfft), measured ~1.5x faster
  than numpy.fft on this length; pyfftw/FFTW with reused plans would gain
  another ~2x on the transforms themselves, but the whole FFT share is
  ~3 ms per redshift (~1% of the evaluation), which does not justify the
  dependency (benchmark in the strategy notes, 2026-07-03)."""
  from scipy.special import sici
  from scipy import fft as _sfft
  r, data_bias, ext_bias, kernel, lnk0, dlnr = tab
  n = len(r)
  n_fft = n + len(ext_bias)
  q = _FFTLOG_BIAS
  F = rho * r**3
  a = np.empty(n_fft)
  a[:n] = F * data_bias
  a[n:] = F[-1] * ext_bias                # continuous constant extension
  # the target sum is Re sum_m c_m K_m e^{-2 pi i m j / N} = irfft of
  # conj(c K) up to the 1/N convention; the cached kernel is conj(K), so
  # only the rfft output needs conjugating here
  b = np.conj(_sfft.rfft(a)) * kernel
  I_grid = _sfft.irfft(b, n=n_fft)
  # interpolation window around the targets (slice before the k^-q unbias
  # and the sici tail so both run on ~a few hundred points, not 4096)
  lnk_t = np.log(k_targets)
  j0f = (lnk_t - lnk0) / dlnr             # fractional index on the k grid
  jlo = max(int(np.floor(j0f.min())) - 2, 1)
  jhi = min(int(np.ceil(j0f.max())) + 3, n_fft - 2)
  js = np.arange(jlo - 1, jhi + 2)
  k_win = np.exp(lnk0 + js * dlnr)
  I_win = I_grid[js] * k_win**(-q)
  aa = k_win * r[-1]
  si, ci = sici(aa)
  I_win -= F[-1] * (np.sin(aa) / aa - ci)
  # 4-point Lagrange (cubic) interpolation on the uniform-ln k window
  jf = j0f - (jlo - 1)                    # position within the window
  i1 = np.clip(jf.astype(int), 1, len(js) - 3)
  t = jf - i1
  w0 = -t * (t - 1.0) * (t - 2.0) / 6.0
  w1 = (t * t - 1.0) * (t - 2.0) / 2.0
  w2 = -t * (t + 1.0) * (t - 2.0) / 2.0
  w3 = t * (t * t - 1.0) / 6.0
  return (w0 * I_win[i1 - 1] + w1 * I_win[i1] + w2 * I_win[i1 + 1]
          + w3 * I_win[i1 + 2])


def _sigma_fingerprint(PS, Omega_0):
  PS = np.asarray(PS)
  return (id(PS), len(PS), float(PS[0]), float(PS[-1]),
          round(float(Omega_0), 12), round(_ACCURACY_BOOST, 6))


def sigma_M_fast(M, k, PS, Omega_0, cosmo_dic):
  """sigma(M) from a per-evaluation log-log interpolation table whose nodes
  are computed with upstream func_sigma_M; falls back to the direct upstream
  evaluation outside the tabulated mass range."""
  cache = cosmo_dic.setdefault("_vm_sigma_cache", {})
  key = _sigma_fingerprint(PS, Omega_0)
  tab = cache.get(key)
  if tab is None:
    lgM = np.linspace(_SIGMA_LOG10M_MIN, _SIGMA_LOG10M_MAX,
                      _nodes(_SIGMA_NODES))
    sig = np.array([_variance.func_sigma_M(10.0**x, k, PS, Omega_0)
                    for x in lgM])
    tab = (lgM, np.log(sig), PS)   # keep PS referenced so id() stays valid
    cache[key] = tab
  lgM, lnsig, _ = tab
  lg = np.log10(M)
  out = np.exp(np.interp(lg, lgM, lnsig))
  # out-of-range guard: defer to the exact upstream function
  if np.isscalar(lg) or np.ndim(lg) == 0:
    if lg < _SIGMA_LOG10M_MIN or lg > _SIGMA_LOG10M_MAX:
      return _variance.func_sigma_M(float(M), k, PS, Omega_0)
    return float(out)
  bad = (lg < _SIGMA_LOG10M_MIN) | (lg > _SIGMA_LOG10M_MAX)
  if np.any(bad):
    out = np.asarray(out)
    out[bad] = [_variance.func_sigma_M(float(m), k, PS, Omega_0)
                for m in np.asarray(M)[bad]]
  return out
# VM-SPEEDUP ENDS
