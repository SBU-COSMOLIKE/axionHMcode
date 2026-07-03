# axionHMcode

`axionHMcode` is a code to compute the non-linear matter power spectrum in a mixed dark matter cosmology with an ultra-light axion (ULA) component of the dark matter as described in [Vogt et al. (2022)](https://arxiv.org/abs/2209.13445). A very accurate halo model for &Lambda;CDM and massive neutrino cosmologies is given in [Mead et al. (2020)](https://arxiv.org/abs/2009.01858), referred to as `HMCode-2020`. Since the `axionHMcode` model is inspired by their code and uses some of their fitting parameters, our model is named after `HMcode`. More recently, the code was updated and calibrated to simulations in [Dome et al. (2024)](https://arxiv.org/abs/2409.11469).

## Theory

The model computes the non-linear power spectrum by using the fully expanded power spectrum

![codesketch](eq_halo_model.png)


The cold part can be computed as usually with the standard halo model (see [Massara et al. (2014)](https://arxiv.org/abs/1410.6813) or [Mead et al. (2020)](https://arxiv.org/abs/2009.01858)). In contrast, the cross and axion parts have to take into account the non clustering of axions on small scales due to free-streaming. This is done by splitting the axion overdensity into a clustered and linear component. For details see [Massara et al. (2014)](https://arxiv.org/abs/1410.6813) where the same full treatment was used for massive neutrinos, but can be translated to any other warm/hot, i.e. free streaming/(partially) non-clustering, matter component. 

## How Does the Code Work?

The code expects an input file called `input_file.txt` which contains information about the cosmology in the style of a Python dictionary. Alternatively, one can also provide the corresponding dictionary directly. Have a look into `load_cosmology.py` to see what information has to be stored in this dictionary. At the least, the linear matter power spectrum for the target cosmology broken down into different components such as, axions, CDM and baryons saved in a dictionary is needed (see example notebook for the structure). If the linear matter power spectrum is not available, the built-in wrapper to `axionCAMB` can be used. In other words, if you have a working `axionCAMB` executable, the code can directly compute the linear power spectrum for the cosmology as specified by `input_file.txt`.

An example Python file is given in `example_file.py`. To run the file you have to change the `input_file_path` and the `axionCAMB_exe_path` fields (provide the complete path). If the paths are not correct the code will produce an error message. Besides the non-linear total matter power spectrum, the example file also computes the non-linear power spectrum in a &Lambda;CDM cosmology where the axion density is transformed into CDM density. Both power spectra are saved to a file, whose name can be set by the user via the `datafile_path` variable. The units of the wavenumber and the power spectra are $h/\mathrm{cMpc}$ and $\left(\mathrm{cMpc}/h\right)^3$ respectively. The code also produces a plot of the ratio between the MDM and &Lambda;CDM linear and non-linear power spectra.

Moreover, for consistency the example notebook makes a comparison to `HMcode-2020` in the &Lambda;CDM case. A comparison is made using the implementation in `CAMB` and the Python package `hmcode`.


## HMCode-2020 Parameters

The `axionHMcode` can also use the parameters from `HMCode-2020` in [Mead et al. (2020)](https://arxiv.org/abs/2009.01858) which improves the predictions in the case of a CDM cosmology with massive neutrinos. The parameters can be switched on by setting the corresponding variables to `True` or `False` in the function for the non-linear power spectrum (make sure the parameters are consistent in the different functions). The parameters are the smoothing parameters, `alpha`, the halo bloating term, `eta_given`, the one-halo damping on large scales, `one_halo_damping`, and the two-halo damping on large scales, `two_halo_damping`.

Since these parameters are not calibrated to mixed axion - cold dark matter cosmologies, we suggest to not use them in the base version.


## Code Updates including calibration from [Dome et al. (2024)](https://arxiv.org/abs/2409.11469)

[Dome et al. (2024)](https://arxiv.org/abs/2409.11469) presented an improved version of `axionHMcode`, which introduced new parameters calibrated to MDM simulations.

Moreover, several updates were made to speed up the code. The changes are summarised in the following:

1. The axion mass - cold mass relation, $M_a(M_c)$, is now modelled as a broken power law (see Eq. 51 in [Dome et al. (2024)](https://arxiv.org/abs/2409.11469)) which ensures the old relation of $M_a(M_c) = (\Omega_a/\Omega_c) M_c$ is satisfied above a defined cut-off mass. This new relation is inspired by simulations and is calibrated by them in the redshift range $1 < z < 8$ and for axion fractions of $0.01 < f_{\mathrm{ax}} = \Omega_{\mathrm{ax}} / \Omega_{\mathrm{m}} < 0.3$ around the pivot axion mass $m_a=10^{-24.5}$ eV.
2. [Dome et al. (2024)](https://arxiv.org/abs/2409.11469) introduced new smoothing parameters $\alpha$ for the cold-cold power spectrum and the cross-power spectrum which depend on the axion mass and axion density. The exact form of these parameters was calibrated to the MDM simulations in the redshift range $1 < z < 8$ and for axion fractions of $0.01 < f_{\mathrm{ax}} = \Omega_{\mathrm{ax}} / \Omega_{\mathrm{m}} < 0.3$ around the pivot axion mass $m_a=10^{-24.5}$ eV.
3. The two-halo term is now calculated by the linear power spectrum only. The difference between this and the total two-halo term is minor and the speed up is increased. There is still the option to use the full two-halo term by setting the parameter `full_2h = True`, but note that the calibrations were performed with `full_2h = False`. 
4. A bug in the cold density profile when using the halo bloating term $\eta$ was corrected.
5. The relations for the critical density threshold, $\delta_c$, and the virial overdensity, $\Delta_{\mathrm{vir}}$, are now calculated as in `HMCode-2020` (see [Mead et al. (2020)](https://arxiv.org/abs/2009.01858), Eq. A1 and A2). This ensures that `axionHMcode` and `HMCode-2020` agree in the case of a &Lambda;CDM cosmology.
6. The minimum concentration is now $B=5.196$ as found in [Mead et al. (2020)](https://arxiv.org/abs/2009.01858).
7. Alex Laguë implemented Numba in this updated version to increase the speed of the code.
8. Alex Laguë and Keir Rogers also included the optional parameters `alpha_1`, `alpha_2`, `gamma_1`, `gamma_2` defined in [Dentler er al. (2021)](https://arxiv.org/abs/2111.01199), Eq. (36). To use them, just include them in your dictionary of cosmological parameters (the `cosmo_dic` file) before running the `params` and `power spectra` calculation using e.g. `cosmo_dic['alpha_1'] = X`. They are not yet included in the input file.

Both the base version and calibrated version from [Dome et al. (2024)](https://arxiv.org/abs/2409.11469) can be used. To distinguish, set the parameter `version` in the input file to `basic` or `dome`, respectively. 

When using the calibrated version from [Dome et al. (2024)](https://arxiv.org/abs/2409.11469), one has to take into account the redshift range, the axion fraction as well as the axion mass for which the model was calibrated. Therefore, we only recommend to use the `dome` version when $1 < z < 8$, $0.01 < f_{\mathrm{ax}} = \Omega_{\mathrm{ax}} / \Omega_{\mathrm{m}} < 0.3$ and for axion masses that are close to the pivot mass $m_a=10^{-24.5}$ eV. In addition, in the `dome` version, we recommend to use `alpha = True` and `concentration_param = True`.

## Contact data

If you find any bugs or have any questions about the code, please send me a message via GitHub or open an issue.

## Fork note (SBU-COSMOLIKE): VM-SPEEDUP

This fork adds physics-preserving performance fixes to the upstream code, all
marked with greppable `# VM-SPEEDUP` fences (`grep -rn "VM-SPEEDUP" .` lists
every change site):

- `cosmology/fast_tables.py` (new): lazily built lookup tables for the growth
  factor D(z), the HMcode-2020 eq. A5 integrated growth G(z), and sigma(M).
  Table nodes are computed with the original upstream functions.
- `halo_model/cold_density_profile.py`: `func_z_formation` inverts the
  tabulated growth instead of running a brentq whose objective re-integrated
  the growth (including its constant normalization) and sigma(f*M) at every
  iteration; exact clamp/branch semantics preserved. `func_conc_param` and
  `NFW_profile` memoize exact repeated evaluations inside the central-density
  root solver.
- `cosmology/overdensities.py`: `func_D_z_unnorm_int` evaluates the same
  double integral from a dense cumulative-trapezoid table instead of
  scipy.dblquad (~0.15 s per call).
- `halo_model/axion_density_profile.py`: `func_ax_halo_mass` (the objective
  of the soliton central-density solver) integrates the composed profile on
  upstream's identical 2000-point geomspace grid, but with precomputed
  canonical composite-Simpson weights (`fast_tables.geom_simpson_grid`)
  instead of a scipy.integrate.simpson call — the integral becomes a dot
  product, removing ~60 us of per-call bookkeeping from the solver's
  ~3000 objective evaluations per redshift.
- `cosmology/fast_tables.py` also exposes `set_accuracy_boost(x)`: a single
  multiplier on the table node counts (default 1.0 = the counts validated
  below). The AxiECAMB boost theory forwards its yaml `accuracy_boost`
  option here; running the same setup at 1 and 2 is the convergence check.
  The radial 2000-point grid is deliberately excluded from the scaling (see
  appendix section A.7).

Validation against unmodified upstream (commit a85ba26): nonlinear boost
grids B(k,z) agree to max |dB/B| = 1.6e-5 across dome/basic, z = 0/2, three
cosmologies including the LCDM limit; growth-table primitives agree with the
original quadratures to <= 1.6e-6. Measured single-redshift speedups: 4.9x
(dome, z=0: 3.5 -> 0.73 s), 4.0x (basic, z=0), 2.4x (dome, z=2), 15-22x
(LCDM path); the remaining cost is dominated by the soliton central-density
root solver structure (`optimize.root` re-evaluating the full profile per
iteration) and the (M,k) profile Fourier transform, untouched here because
solver replacement changes the guess-based no-solution rejection behavior.

The complete mathematical and implementation documentation of every change,
including the measured performance anatomy that motivated it and the
validation protocol, is in the appendix below.

## Appendix (SBU-COSMOLIKE): mathematics and implementation of the VM-SPEEDUP changes

This appendix documents each change in full: the quantity involved, how the
unmodified upstream code evaluated it, what the fork does instead, why the
result is either bit-identical or bounded, and the measurements. Every change
site in the source carries a `# VM-SPEEDUP` fence with a condensed version of
the same information. All profiling numbers were measured on one halo-model
evaluation (one cosmology, one redshift, $n_k \simeq 212$, 100-point halo-mass
grid) with cProfile; the profiling script is
`axionhmcode_boost/dev_scripts/profile_axionhmcode.py` in the AxiECAMB
repository.

### A.1 Where the time went in upstream

One dome-version evaluation at $z=0$ cost ~3.5 s, split as:

| stage | dome | basic |
|---|---|---|
| `HMCode_param_dic` (cold HMcode-2020 parameters) | 0.00 s | 0.00 s |
| `func_axion_param_dic` (cut mass, soliton central density, clustered fraction) | 2.35 s | 1.22 s |
| `func_full_halo_model_ax` (the halo-model integrals themselves) | 0.43 s | 0.33 s |

The halo-model sums are not the problem; ~85% of the runtime is the axion
parameter stage, and the mechanism is structural. That stage solves implicit
equations with root finders nested inside root finders:

```
func_cut_mass_axion_halo:            brentq over log10 M
  -> func_jeans_virial_ratio(M_trial)
    -> func_halo_jeans_kscale(M_trial)
      -> func_conc_param(M_trial)        [trial masses are off any grid]
        -> func_z_formation:             a second brentq, over z_f in [z, 100]
          -> every iteration re-integrates the growth factor D(z_f)
             (including its constant normalization D(0)) and re-integrates
             sigma(f*M) over the whole k grid
```

and the same pattern repeats inside the soliton central-density solver
(`axion_density_profile.func_central_density_param`). The innermost
quantities are smooth one-variable functions, but upstream re-derived them
from quadratures at every innermost iteration. Measured per redshift (dome,
$z=0$): 109,334 growth-factor integrations, 99,055 full $\sigma(M)$
integrations, 6,451 `brentq` calls, 2.7 million Python function calls in
total. Fortran HMcode-2020 evaluates the same class of physics in
milliseconds because it tabulates $\sigma(M)$ and the growth once per
redshift and gets the formation redshift by inverse interpolation; the fork
applies exactly that strategy to the axion stage, without changing any
formula.

### A.2 The growth-factor table (`cosmology/fast_tables.py: growth_tables`, `D_norm_fast`, `z_of_D_norm`)

The linear growth factor used throughout the code is (upstream
`overdensities.func_D_z_unnorm`, a standard matter-era integral
representation)

$$D(z) = \frac{5\,\Omega_m}{2}\, E(z) \int_z^{100} \frac{1+x}{E(x)^3}\, dx,
\qquad E(z) = \frac{H(z)}{H_0},$$

which upstream evaluates with a 2000-point trapezoid rule per call, and
normalizes per call as $D_{\rm norm}(z) = D(z)/D(0)$ — recomputing the
constant $D(0)$ every time.

The fork evaluates the same upstream function once at 4001 linearly spaced
nodes $z \in [0, 100]$ (the interval upstream's `brentq` searched), stores
$D_{\rm norm}$ at the nodes, and interpolates linearly (`np.interp`).
Because the nodes are computed with the unmodified upstream quadrature, the
table reproduces upstream numerics at the nodes exactly; between nodes the
interpolation error of a smooth function on a $\Delta z = 0.025$ grid is
negligible against the halo-model calibration accuracy (measured:
$\le 1.6\times10^{-6}$ relative on the validation points). $D_{\rm norm}(z)$
is strictly decreasing, so it is also invertible by the same interpolation
run backwards (`z_of_D_norm`), which is what eliminates the inner `brentq`
(section A.5). The table is cached per $(\Omega_{m,0}, \Omega_{w,0})$ — it
depends on nothing else — so its cost is paid once per cosmology, not per
redshift.

### A.3 The integrated growth (`fast_tables.G_integral_fast`, replacing `scipy.integrate.dblquad`)

HMcode-2020 (arXiv:2009.01858, eq. A5) needs the integrated growth

$$G(z) = \frac{5\,\Omega_m}{2} \int_z^{z_{\max}} dx\, \frac{E(x)}{1+x}
\int_x^{z_{\max}} dy\, \frac{1+y}{E(y)^3}, \qquad z_{\max} = 10^4,$$

which upstream computed with `scipy.integrate.dblquad` at ~0.15 s per call,
once per (cosmology, redshift) evaluation.

The key observation is that both integrals run from a moving lower limit to
the same fixed upper limit, so the whole function $G(z)$ — not just one value
— is obtained from a single dense grid by two cumulative passes. Define the
inner integral $I(x) = \int_x^{z_{\max}} (1+y)\,E(y)^{-3}\, dy$. On nodes
$y_0 < y_1 < \dots < y_N$, the reversed cumulative trapezoid sum

$$I(y_i) = \sum_{j \ge i} \tfrac12\,[g(y_j) + g(y_{j+1})]\,(y_{j+1} - y_j),
\qquad g(y) = \frac{1+y}{E(y)^3},$$

gives $I$ at every node in one vectorized pass; a second identical pass over
$E(x)(1+x)^{-1} I(x)$ gives $G$ at every node. The fork uses 20,000
geometrically spaced nodes (dense near $z=0$ where the integrand varies
fastest, reaching $z=0$ exactly by an offset), builds the table once per
cosmology, and interpolates. Measured agreement with the upstream `dblquad`
value: $\le 1.6\times10^{-6}$ relative at $z \in \{0, 1, 5, 9.8\}$. The
upstream code remains in `overdensities.func_D_z_unnorm_int` after the
return statement as the reference implementation.

### A.4 The mass-variance table (`fast_tables.sigma_M_fast`)

The mass variance is (upstream `variance.func_sigma_M`/`func_sigma_r`)

$$\sigma^2(M) = \frac{1}{2\pi^2} \int d\ln k\; P(k)\, W^2\!\big(k R(M)\big)\, k^3,
\qquad R(M) = \Big(\frac{3M}{4\pi\bar\rho}\Big)^{1/3},$$

with $W(x) = 3(\sin x - x\cos x)/x^3$ the spherical top-hat window; upstream
integrates over the full k grid per call.

The fork evaluates that same upstream function at 321 nodes uniform in
$\log_{10} M \in [3, 19.5]$ and interpolates $\ln \sigma$ linearly in
$\log_{10} M$. The log-log choice is what makes 321 nodes sufficient:
$\sigma(M)$ is close to a power law in $M$, so its log-log graph is close to
a straight line and piecewise-linear interpolation is nearly exact. Requests
outside the tabulated range fall back to the exact upstream evaluation, so
the table can never extrapolate. The cache lives inside `cosmo_dic` under a
`'_vm_'` key and is fingerprinted by the identity and endpoints of the power
spectrum array, so its lifetime is exactly one (cosmology, redshift)
evaluation — there is no possibility of a stale table leaking across
likelihood points, and `multiprocessing` fork workers inherit nothing they
should not.

### A.5 Formation redshift by monotone inversion (`halo_model/cold_density_profile.py: func_z_formation`)

HMcode-2020 defines the halo formation redshift (arXiv:2009.01858 eq. 21;
Dome et al. arXiv:2409.11469 eq. 50) as the solution $z_f$ of

$$\frac{D(z_f)}{D(z)}\,\sigma(f M, z) = \delta_c(z), \qquad f = 0.01 .$$

Equivalently, with $t \equiv D_{\rm norm}(z)\,\delta_c(z)/\sigma(fM)$:
$D_{\rm norm}(z_f) = t$. Upstream solved this with `brentq` on
$z_f \in [z, 100]$, paying the full growth and variance quadratures at every
iteration (section A.1); the fork computes $t$ once (with $\sigma$ from the
A.4 table) and reads $z_f = D_{\rm norm}^{-1}(t)$ from the A.2 table.

The subtle part is not the inversion but the clamp. Upstream only ran
`brentq` when its objective changed sign across $[z, 100]$ and returned $z$
otherwise; that branch implements HMcode's minimum-concentration rule (a
formal $z_f < z$ would give $c < B$, which the model truncates at $c = B$,
$B = 5.196$). The fork reproduces the branch through the same endpoint sign
products, evaluated exactly on the tabulated values — scalar path: positive
product returns $z$; array path: the inversion is used only where the
product is negative. The branch decision is a comparison, never an
interpolation, so no halo can land on the wrong side of the clamp because of
table smoothing.

### A.6 Exact memoization (`func_conc_param`, `NFW_profile`)

The soliton central-density solver
(`axion_density_profile.func_central_density_param`) finds, for each halo
mass, the soliton amplitude such that the integrated axion halo profile
matches the target axion halo mass. `scipy.optimize.root` evaluates the
objective ~30 times per halo mass, and each evaluation rebuilds the full
density profile — but the solver's unknown only scales the soliton term; the
NFW-shaped part and the concentration $c(M) = B\,(1+z_f)/(1+z)$ do not
depend on it at all. Upstream therefore recomputed identical quantities ~30
times per mass, 100 masses per redshift.

The fork memoizes the two quantities exactly:

- `func_conc_param` (scalar masses): keyed on the mass, $\Omega_0$,
  `c_min`, the identity of the power-spectrum array (`id(PS)` is stable
  because the arrays are constructed once per evaluation and never mutated),
  and the `axion_dic` M_cut state that gates the Dentler gamma correction.
- `NFW_profile` (scalar mass, radial grid): additionally fingerprints the r
  grid by its length and both endpoints — which identifies a geomspace grid
  uniquely — plus the `eta_given` switch.

A memoization is exact by construction when the key captures every input the
result depends on; the remaining inputs (redshift, cosmology) are constant
within the memo's lifetime because both memos live in `cosmo_dic` under
`'_vm_'` keys, i.e. for one (cosmology, redshift) evaluation. The proof that
nothing was missed is empirical and strict: the fork validation measured
bit-identical boost deviations before and after the memoization round — the
memos changed the runtime and nothing else.

### A.7 The soliton mass integral: precomputed weights on upstream's grid (`func_ax_halo_mass`, `fast_tables.geom_simpson_grid`)

The soliton central-density solver finds, per halo mass, the soliton
amplitude $\rho_c$ such that the integrated axion profile matches the target
axion halo mass; its objective is

$$M_{\rm ax}(\rho_c) = 4\pi \int_0^{r_{\rm vir}} \rho_{\rm ax}(r;\rho_c)\, r^2\, dr,$$

which upstream evaluated by sampling $\rho_{\rm ax}$ on a 2000-point
geomspace grid and calling `scipy.integrate.simpson`. With ~30 objective
evaluations per mass and 100 masses per redshift, simpson's per-call
bookkeeping (~60 us of argument validation, slice construction and masked
divisions) recurs ~3000 times per redshift and dominated the solver's cost.

Quadrature by a weight vector. Any fixed-node quadrature rule is a linear
functional of the samples, $\int f \approx \sum_i w_i f(r_i)$ with weights
depending only on the grid. The fork precomputes $w$ once per grid
(`fast_tables.geom_simpson_grid`, cached in `cosmo_dic`) and evaluates each
integral as a dot product. The weights implement the canonical composite
Simpson rule for unevenly spaced samples — for each pair of adjacent
intervals $(h_0, h_1)$ the unique parabola through the three points
integrates to

$$\frac{h_0+h_1}{6}\left[\Big(2-\frac{h_1}{h_0}\Big) f_i
+ \frac{(h_0+h_1)^2}{h_0 h_1} f_{i+1}
+ \Big(2-\frac{h_0}{h_1}\Big) f_{i+2}\right],$$

with the leftover final interval closed by a trapezoid. Both pieces are
textbook mathematics: no scipy call remains in the hot loop, so the result
does not depend on scipy's per-version policy for even point counts (the
default handling changed in scipy 1.11 and the compatibility keyword was
removed in 1.14; that policy dependence is why cloning scipy's exact
weights was rejected). The difference from scipy's current rule is confined
to the outermost interval and is orders of magnitude below the gate.

Why the grid must be upstream's, not a better quadrature's. A
Gauss-Legendre rewrite in $\ln r$ was implemented and measured first: it is
spectrally accurate on each smooth branch of the profile and reproduced
upstream to $6\times10^{-6}$ in some cases — but non-monotonically in the
node count, with excursions to $|\Delta B/B| \sim 10^{-2}$ at
$k \gtrsim 8\,h/{\rm cMpc}$. The mechanism: `func_dens_profile_ax` composes
the profile as soliton inside, NFW outside, with the crossover radius
detected from sign changes of the sampled difference. The composed profile
is therefore a function of the node positions; for halos where the two
branches barely cross, moving the nodes flips the detection, discretely
changing the profile, the solved $\rho_c$, and the boost. That is a
behavior change relative to upstream, not a quadrature error, so the fork
keeps upstream's exact `np.geomspace(1e-15, r_vir, 2000)` nodes —
bit-identical composition, crossover snapping, and rejection behavior — and
changes only the summation rule applied to the identical samples.

For the same reason the 2000-point radial grid is pinned and excluded from
the `accuracy_boost` scaling: doubling it re-snaps the crossovers and flips
marginal rejections, moving the dome boost by up to
$|\Delta B/B| \sim 7\times10^{-2}$ at high $k$ (measured). That sensitivity
is a property of the released, calibrated model — the Dome et al.
calibration ran with this grid — so it is treated as part of the model
definition, like the fitted constants, and not as a numerical convergence
parameter of this fork.

The guess-construction integrals of the solver
(`integrate.quad`/`integrate.simpson` in `func_central_density_param`) are
untouched: they run once per mass (negligible cost) and feed the
$|{\rm guess} - \rho_c| > 100$ rejection heuristic, which must stay
bit-identical to upstream.

### A.8 The accuracy knob (`fast_tables.set_accuracy_boost`)

All table node counts in A.2-A.4 are scaled by a single multiplier, set from
the AxiECAMB boost theory's yaml option `accuracy_boost` (modeled on CAMB's
`AccuracyBoost` and cosmolike's `Ntable.high_def_integration`). The default
1.0 reproduces the validated node counts exactly. Rerunning the same setup
at `accuracy_boost: 1` and `2` and comparing boost grids measures the total
internal-discretization error in one shot; measurement (dome, $z=0$): max
$|\Delta B/B| = 0.7\!-\!1.1\times10^{-3}$ between the two, dominated by the
halo-mass integration grid (also doubled by the option), with the tables of
this fork contributing $\lesssim 2\times10^{-6}$. The radial 2000-point
grid of A.7 is deliberately excluded from the scaling (see A.7).

### A.9 Validation protocol and results

The gate for every change: max $|\Delta B/B| \le 10^{-4}$ on the final
nonlinear boost $B(k,z) = P_{\rm NL}/P_{\rm L}$ against unmodified upstream
(commit a85ba26), same transfer inputs, across dome/basic $\times$
$z \in \{0, 2\}$ $\times$ three cosmologies (a Gaughan-like axion cosmology,
the upstream input-file cosmology, and the LCDM limit). The driver
(`axionhmcode_boost/dev_scripts/fork_validate.py` in the AxiECAMB repo)
builds one set of transfers, runs upstream and fork in separate child
processes, and compares the grids; a primitive-level check compares the A.2
and A.3 tables against the original quadratures directly. The gate is what
caught the Gauss-Legendre crossover flips of A.7 before they shipped.

Results (all rounds, including A.7): max $|\Delta B/B| = 1.6\times10^{-5}$
over all cases (deviations carried by the table interpolations — they were
unchanged by the memoization round, proving A.6 exact); growth/G primitives
$\le 1.6\times10^{-6}$; the AxiECAMB boost pytest suite passes against the
fork. Measured warm single-redshift times (same machine as A.1):

| case | upstream | fork | speedup |
|---|---|---|---|
| dome, z=0 | 3.5 s | 0.73 s | 4.9x |
| basic, z=0 | 2.0 s | 0.49 s | 4.0x |
| dome, z=2 | 1.8 s | 0.73 s | 2.4x |
| basic, z=2 | 1.0 s | 0.48 s | 2.1x |
| LCDM limit | 0.1 s | 0.006 s | 15-22x |

Per likelihood evaluation in the AxiECAMB boost pipeline (~50 redshifts),
the dome cost drops from ~135 s (upstream) to ~36 s, before the boost
theory's `processes` fork-parallelism divides the wall time (default:
one worker per `OMP_NUM_THREADS` core).

### A.10 What was deliberately not changed

- The soliton central-density solver still uses `scipy.optimize.root` with
  upstream's initial guess and its rejection heuristic (solutions with
  $|{\rm guess} - \rho_c| > 100$ are declared unphysical and set to zero).
  Replacing the solver would change which halos are assigned a soliton —
  behavior, not precision — and needs a physics decision, not an
  optimization. The same reasoning pinned the radial grid in A.7.
- The (M,k) profile Fourier transform in the halo-model sums
  (`func_dens_profile_kspace_ax` and the `integrate.simpson` calls in
  `PS_nonlin_*`) is untouched (~0.4 s per redshift). Its integrands are
  oscillatory in $k r$ and its grids feed no branch decisions, so a
  weight-vector treatment in the style of A.7 is a possible next step, to
  be revalidated through the A.9 gate.
- The input k grid is never thinned or resampled: the boost is evaluated on
  the transfer grid it receives (halving that grid was measured to corrupt
  the boost by 11% through the $\sigma(M)$ integrals).
