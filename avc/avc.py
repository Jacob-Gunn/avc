"""
Created on Tue Nov 18 15:30:56 2025

Author: Dr Jacob Gunn
"""
import numpy as np
from scipy.integrate import simpson




'''Fundamental physics constants'''

h = 6.626e-34                                                                  #Planck's constant [Js]
hbar = h/(2*np.pi)                                                             #Reduced Planck's constant [Js]
q = 1.6e-19                                                                    #Proton charge [C] or [J/eV]
heV = h / q                                                                    #Planck's constant [eVs]
Au = 1.66e-27                                                                  #Atomic unit ie m_proton [kg]
kb = 8.617e-5                                                                  #Boltzmann constant in [eV/K]
kbsi = 1.36e-23                                                                #Boltzmann constant [J/K]
Td = 1e-17                                                                     #1Td in Vcm^2
me = 9e-31                                                                     #Electron mass [kg]
eps0 = 8.85e-12                                                                #Permittivity of free space [F/m]

G0 = np.array([0.007, 0.07, 0.18, 0.33,0.52,0.72,
      0.92,1.2,1.4,1.6,1.8,2.1,2.3,2.5,2.6]) * 1e-3                             #Total decay widths of O_2^- autoionisation states starting from v = 4, in eV


def N0(T,p,unit = 'm'):
    '''Returns the number density of an ideal gas with temperature T and pressure p
    [T] = K
    [p] = atm'''
    if unit == 'm':
          pref = 2.48e25 #m^3
    if unit == 'cm':
          pref = 2.48e25 * 1e-6 #cm^-3
    else:
          print('Unrecognised units. Use cm or m')
          return 0
    return  pref * 300 / T * p











'''Distributions and cross sections'''
def mbd(T,E = None,v = None,m = None):
    '''Returns the value of the Maxwell Boltzmann distribution
    T is in K
    E is in eV'''
    if v is None:
        return 2*np.sqrt(E/np.pi) * (kb*T)**(-3/2) * np.exp(-E/(kb*T))
    if E is None:
        return np.sqrt(m/(2*np.pi*kbsi*T)*np.exp(-m*v**2/(2*kbsi*T)))
    
    
def mbdrel(vrel, T1,T2,m1,m2):
    '''Returns the value of the Maxwell Boltzmann distribution of the scalar relative velocity for two particles with temperatures T1, T2 and masses
    m1 and m2
    T is in K
    E is in eV'''
    mu = (m1*m2)/(m1 + m2)
    Teff = (m1*T2 + m2*T1)/(m1+m2)
    return 4*np.pi * (mu/(2*np.pi*kbsi*Teff))**(3/2)*np.exp(-mu*vrel**2/(2*kbsi*Teff)) * vrel**2

def sigBW(e, Gamma0, Gamma, eps_k):
    '''Breit-Wigner resonant scattering cross section as function of collisional energy e. This is the cross section for the resonant process i + j -> k 
    where k is a single sharply defined resonance with total width Gamma, partial width Gamma0, and peak energy eps_k.
    
    Gamma0, Gamma, eps_j, and e are all in units of electron volts
    gs is array-like, length 3, [g_i, g_j, g_k]
    p1 is a numerical fudge factor for fitting magnitude'''
    gs = [2.0, 3.0, 4.0]
    g_i, g_j, g_k = iter(gs) #Statistical weights of ion, electron


    lam = h / np.sqrt(2 * me * e * q)
    prefactor = np.pi * lam**2 * (g_k / (g_i * g_j))
    denom = (e - eps_k)**2 + (Gamma / 2.0)**2
    
    p1 = 3e-6 

    return p1 * prefactor * (Gamma0 * Gamma) / denom






























'''Functions for calculating rate constants'''

def Tion(mi, mm, EN,q = 1.6e-19,v = 1e10):
    '''Calculates the contribution to the average energy of ions in the low field regime from external E, assuming mi<<mm and 
    assuming a constant rate of collisions, according to Eqn (10) of Physica 101A (1980) 265-274
    mi is the ion mass, in kg
    mm is the molecular mass, kg
    EN is the reduced electric field strength in Td
    q is the ion charge, C
    v is the collisional frequency in Hz which for N = 6e25m^-3 '''
    
    N0 = 6e25
    EN = EN*1e-21 # convert Td to Vm^2
    return  (q*EN*N0/(v*mi))**2*mm/(3*kbsi)




def k(sig,T1,m1,m2,E1 = 1e-3,E2 = 1e5,N = 1000, Case = 2, v1 = 1e-10, v2 = 1e5,T2 = 300):
    '''Calculates the rate constant for a reaction given a particular temperature T, cross section sig
    sig is the cross section, should be a function returning a scalar in units of cm^-2

    Case 1 assumes that the interaction is a two-body process between two non-identical particles
    which are both Maxwell-Boltzmann distributed at temperature T1

    Case 2 assumes that the collision is between two non-identical particles, with m1 != m2,
    T1 != T2, and that the cross section is given in terms of the relative velocity.
    
    T is the temperature of the energy distribution of the incoming particles, in K
    m1,m2 are the masses of the initial state particles, in kg
    E1,E2 are the lower and upper limits of integration, in eV, provide for Case 1
    N is the number of points for integration
    v1, v2 are the lower and upper limits of integration, in ms^-1, provide for Case 2
    
    returns the rate constant, in cm^3 s^-1'''

    '''First calculate the reduced mass'''
    mu = m1*m2/(m1+m2)

    '''Case 1'''
    if Case == 1:
        '''Now generate the integrand'''
        elist = np.logspace(np.log10(E1),np.log10(E2),int(N)) #energy list in eV
        #intlist= [sig(e)*mbd(T,e)*np.sqrt(e*1.6e-19) for e in elist]
        intlist= [sig(e)*mbd(T1,e)*np.sqrt(e*1.6e-19*2/mu) for e in elist]
        #print(intlist)

        '''And integrate'''
        return simpson(x = elist, y = intlist)
    if Case == 2:
        '''Create a list of velocities, used for both integrands'''
        vlist = np.logspace(np.log10(v1),np.log10(v2),int(N)) #energy list in eV
        intlist = mbdrel(vlist,T1,T2,m1,m2) * sig(np.abs(vlist)) * np.abs(vlist)
        return simpson(x = vlist, y = intlist)
        





















'''Auxilliary functions, fitting etc'''
def fit(x, y, func, p0=[], bounds=[], acc=100):
    """
    Fit y(x) to a user-specified model function using a combination of
    global and local optimisation.

    Parameters
    ----------
    x : array-like, shape (L,)
        1D array of independent variable values.
    y : array-like, shape (L,)
        1D array of dependent variable values (data to be fitted).
    func : callable
        Model function of the form func(x, p1, p2, ..., pN) returning
        an array-like of the same shape as x.

        Example:
            def model(x, a, b, c):
                return a * np.exp(-b * x) + c
    p0 : sequence of float, optional
        Initial guess for the parameters (length N). If provided, this
        is used as an additional starting point for a local least-squares
        refinement, but *never* as the only parameter set tried.
    bounds : sequence, optional
        Two-dimensional array-like `[[p1L, p2L, ..., pNL],
                                     [p1U, p2U, ..., pNU]]`
        giving lower and upper bounds for each parameter. If empty, all
        parameters are assumed unbounded.

        Notes:
            - If provided, the length of bounds[0] and bounds[1] must
              match the number of parameters N.
    acc : int or sequence of float, optional
        Numerical effort / accuracy knob controlling the *global* search:

            - If scalar: used to scale the number of global iterations and
              population size for the global optimiser.
            - If array-like: per-parameter effort; the mean value is used
              as the global effort scale.

        Larger values increase the thoroughness (and cost) of the global
        search. Typical values: 50–300.

    Returns
    -------
    params : ndarray, shape (N,)
        Best-fit parameter vector that minimises the sum of squared
        residuals ||y - func(x, *params)||^2 over the provided data.

    Notes
    -----
    - This function *always* performs a global search over the parameter
      space (within the given bounds) using a population-based method
      before any local refinement. Therefore there is no mode in which
      it can simply return the initial guess p0 because of small or
      vanishing derivatives.
    - If SciPy is not available, a basic random-search fallback is used
      that still explores many parameter combinations within the bounds.
    """
    import numpy as np

    # Try to import SciPy; fall back gracefully if not available
    try:
        from scipy.optimize import differential_evolution, least_squares
        HAVE_SCIPY = True
    except Exception:
        HAVE_SCIPY = False

    # ------------------------
    # Sanitise and validate input
    # ------------------------
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape; got {x.shape} and {y.shape}"
        )

    # Determine number of parameters N
    import inspect

    N = None
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        # Expect first argument to be x, remaining to be parameters.
        # If there's a *args, we can't infer N from the signature.
        has_var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL
                                 for p in params)
        if not has_var_positional:
            N = max(len(params) - 1, 0)
    except (TypeError, ValueError):
        # Builtins or C-implemented callables may not have signatures
        N = None

    # If N could not be inferred (or is zero), infer from p0 or bounds
    if (N is None or N == 0):
        if p0:
            N = len(p0)
        elif bounds:
            if len(bounds) != 2:
                raise ValueError("bounds must be of the form [[...],[...]]")
            N = len(bounds[0])
        else:
            raise ValueError(
                "Could not infer number of parameters from func; please provide p0 or bounds."
            )

    # If p0 is given, ensure it has correct length
    if p0:
        p0 = np.asarray(p0, dtype=float).ravel()
        if p0.size != N:
            raise ValueError(
                f"p0 must have length {N}, got {p0.size}"
            )
    else:
        p0 = None

    # Handle bounds
    if bounds:
        if len(bounds) != 2:
            raise ValueError("bounds must be a 2D array-like [[lower...],[upper...]]")
        lower = np.asarray(bounds[0], dtype=float).ravel()
        upper = np.asarray(bounds[1], dtype=float).ravel()
        if lower.size != N or upper.size != N:
            raise ValueError(
                f"bounds must have length {N} per row; got {lower.size} and {upper.size}"
            )
        if np.any(upper < lower):
            raise ValueError("Each upper bound must be >= the corresponding lower bound.")
    else:
        # Unbounded parameters
        lower = np.full(N, -np.inf)
        upper = np.full(N, np.inf)

    # Interpret acc
    if np.isscalar(acc):
        effort = float(acc)
    else:
        acc_arr = np.asarray(acc, dtype=float).ravel()
        if acc_arr.size == 0:
            effort = 100.0
        else:
            effort = float(np.mean(acc_arr))

    # Clip effort to a sensible range
    effort = max(10.0, effort)

    # ------------------------
    # Define residual and objective
    # ------------------------
    def residuals(theta):
        """Vector of residuals y_model - y_data for given parameters."""
        y_model = func(x, *theta)
        y_model = np.asarray(y_model, dtype=float).ravel()
        if y_model.shape != y.shape:
            raise ValueError(
                "func(x, *params) must return an array of the same shape as x and y."
            )
        return y_model - y

    def objective(theta):
        """Scalar objective: sum of squared residuals."""
        r = residuals(theta)
        return float(np.dot(r, r))

    # ------------------------
    # Global optimisation step
    # ------------------------
    best_theta = None
    best_cost = np.inf

    if HAVE_SCIPY:
        # Differential evolution for robust global search
        # SciPy's differential_evolution uses:
        #   population size = popsize * N
        # so we use effort to set popsize and maxiter.
        popsize = max(5, int(effort // 20) + 5)
        maxiter = max(50, int(effort))

        de_bounds = list(zip(lower, upper))

        result_de = differential_evolution(
            objective,
            de_bounds,
            maxiter=maxiter,
            popsize=popsize,
            polish=False,  # we'll polish ourselves
            updating='deferred',
            workers=1,      # keep deterministic within a single process
        )

        best_theta = result_de.x
        best_cost = result_de.fun

        # Optional: local refinement from DE optimum
        try:
            lsq_res = least_squares(
                residuals, best_theta, bounds=(lower, upper), method='trf'
            )
            theta_polished = lsq_res.x
            cost_polished = objective(theta_polished)
            if cost_polished < best_cost:
                best_theta, best_cost = theta_polished, cost_polished
        except Exception:
            # If polishing fails, just keep the DE result
            pass

        # Optional: also try local refinement from p0 (if provided)
        if p0 is not None:
            try:
                lsq_res_p0 = least_squares(
                    residuals, p0, bounds=(lower, upper), method='trf'
                )
                theta_p0 = lsq_res_p0.x
                cost_p0 = objective(theta_p0)
                if cost_p0 < best_cost:
                    best_theta, best_cost = theta_p0, cost_p0
            except Exception:
                # Ignore if local fit from p0 fails
                pass

    else:
        # ------------------------
        # Fallback: pure NumPy random search
        # ------------------------
        # Still guarantees that *many* parameter combinations are tried.
        rng = np.random.default_rng()
        num_samples = int(50 * effort)  # e.g. ~5000 for effort=100
        num_samples = max(num_samples, 1000)

        # For infinite bounds, fall back to a wide Gaussian around p0 or 0.
        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)

        # If everything is unbounded and no p0 is given, we can't scale sensibly.
        if not np.any(finite_lower | finite_upper) and p0 is None:
            raise ValueError(
                "In the no-SciPy fallback, at least some finite bounds or a p0 "
                "must be provided."
            )

        # Build sampling means and scales
        if p0 is not None:
            mu = p0
        else:
            # Use midpoints for bounded params, zeros otherwise
            mu = np.zeros(N)
            mid = 0.5 * (lower + upper)
            mid[~np.isfinite(mid)] = 0.0
            mu = mid

        # Scale: fraction of (upper-lower) where finite, else relative to |mu| or 1
        scale = np.ones(N)
        span = upper - lower
        span[~np.isfinite(span)] = 0.0

        finite_span = np.isfinite(span) & (span > 0)
        scale[finite_span] = span[finite_span] / 4.0  # cover most of bounds

        no_span = ~finite_span
        scale[no_span] = np.maximum(np.abs(mu[no_span]), 1.0)

        for _ in range(num_samples):
            # Propose a new theta
            theta = mu + rng.normal(size=N) * scale

            # Respect finite bounds
            theta = np.where(finite_lower, np.maximum(theta, lower), theta)
            theta = np.where(finite_upper, np.minimum(theta, upper), theta)

            cost = objective(theta)
            if cost < best_cost:
                best_theta, best_cost = theta, cost

        # In fallback mode, we do not do gradient-based polishing; this is
        # still a global-ish search with many parameter combinations tried.

    if best_theta is None:
        raise RuntimeError("Fitting failed to find any valid parameter set.")

    return np.asarray(best_theta, dtype=float)

    

