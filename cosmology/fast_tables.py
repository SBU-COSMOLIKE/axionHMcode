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
