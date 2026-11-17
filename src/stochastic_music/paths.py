"""
Path simulation layer.

Responsibility: generate raw stochastic paths as plain numpy arrays.
This module knows NOTHING about music, MIDI, pitch, or rhythm - just math.

Every process implements the same interface:

    simulate(T, dt, seed=None) -> (times, values)

so the downstream layers (clock, pitch mapping, MIDI writing) never need
to know which model produced the numbers.

Driving shocks
--------------
Internally every process is written as a function of its *main Brownian
shocks* (one standard normal per time step) plus a random generator for
anything else it needs (variance shocks, jump arrivals, time changes).
That split is what makes correlated voices possible for ANY process:
`simulate_correlated` draws the main shocks of all voices jointly, with
the requested correlation, and hands one row to each process.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import gamma as gamma_fn

import numpy as np

Latent = dict[str, np.ndarray]


def time_grid(T: float, dt: float) -> np.ndarray:
    """Evenly spaced times covering [0, T] with a step as close to dt as possible."""
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    num_steps = max(1, int(round(T / dt)))
    return np.linspace(0.0, T, num_steps + 1)


def _cumulate(x0: float, increments: np.ndarray) -> np.ndarray:
    """Path starting at x0 whose successive differences are `increments`."""
    values = np.empty(len(increments) + 1)
    values[0] = x0
    values[1:] = x0 + np.cumsum(increments)
    return values


class PathSimulator(ABC):
    """Base interface for all stochastic path generators."""

    def simulate(
        self, T: float, dt: float, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate a path over [0, T] with time step dt.

        Returns
        -------
        times : np.ndarray, shape (num_steps + 1,)
        values : np.ndarray, shape (num_steps + 1,)
        """
        times, values, _ = self.simulate_with_latent(T, dt, seed)
        return times, values

    def simulate_with_latent(
        self, T: float, dt: float, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, Latent]:
        """
        Same as simulate(), but also returns the hidden series driving the
        path, if the model has any (e.g. {"variance": ...} for Heston).
        """
        rng = np.random.default_rng(seed)
        times = time_grid(T, dt)
        shocks = rng.standard_normal(len(times) - 1)
        values, latent = self.simulate_driven(times, shocks, rng)
        return times, values, latent

    def simulate_driven(
        self, times: np.ndarray, shocks: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, Latent]:
        """
        Simulate on a given time grid from externally supplied main shocks.

        Parameters
        ----------
        times : np.ndarray, shape (n + 1,)
            Evenly spaced time grid (see time_grid).
        shocks : np.ndarray, shape (n,)
            Standard normal draws driving the path's main Brownian motion.
        rng : np.random.Generator
            Source for any additional randomness the model needs.
        """
        if len(shocks) != len(times) - 1:
            raise ValueError("shocks must have exactly one entry per time step.")
        dt = float(times[-1] - times[0]) / (len(times) - 1)
        result = self._evolve(dt, np.asarray(shocks, dtype=float), rng)
        if isinstance(result, tuple):
            return result
        return result, {}

    @abstractmethod
    def _evolve(self, dt: float, shocks: np.ndarray, rng: np.random.Generator):
        """Return the path values (len(shocks) + 1,), or (values, latent)."""
        raise NotImplementedError


class WienerProcess(PathSimulator):
    """
    Simple Wiener process (scaled Brownian motion):
        dX = sigma * dW

    Parameters
    ----------
    sigma : float
        Volatility / diffusion coefficient.
    x0 : float
        Starting value.
    """

    def __init__(self, sigma: float = 1.0, x0: float = 0.0):
        self.sigma = sigma
        self.x0 = x0

    def _evolve(self, dt, shocks, rng):
        return _cumulate(self.x0, self.sigma * np.sqrt(dt) * shocks)


class BrownianMotion(PathSimulator):
    """
    Brownian motion with drift (arithmetic Brownian motion):
        dX = mu * dt + sigma * dW

    The drift gives the path a systematic upward (mu > 0) or downward
    (mu < 0) tendency on top of the random wandering: a melody that
    trends steadily higher or lower instead of meandering around its
    starting point. Euler-Maruyama is exact here (increments are exactly
    Gaussian whatever the step size).

    Parameters
    ----------
    mu : float
        Drift per unit time.
    sigma : float
        Volatility / diffusion coefficient.
    x0 : float
        Starting value.
    """

    def __init__(self, mu: float = 0.1, sigma: float = 1.0, x0: float = 0.0):
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0

    def _evolve(self, dt, shocks, rng):
        return _cumulate(self.x0, self.mu * dt + self.sigma * np.sqrt(dt) * shocks)


class CIRProcess(PathSimulator):
    """
    Cox-Ingersoll-Ross (CIR) mean-reverting process:
        dX = kappa * (theta - X) * dt + xi * sqrt(X) * dW

    Reverts toward theta at speed kappa, with a volatility that shrinks
    to zero as X approaches zero, so the path clings to its floor.

    Discretized with Euler-Maruyama plus full truncation: both the
    sqrt(X) term and the stored value are clipped at zero each step,
    since discretization error can otherwise push a step negative.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run mean.
    xi : float
        Volatility coefficient.
    x0 : float
        Starting value. Must be >= 0.
    """

    def __init__(self, kappa: float = 2.0, theta: float = 1.0, xi: float = 0.5, x0: float = 1.0):
        if x0 < 0:
            raise ValueError(f"x0 must be >= 0 for CIR, got {x0}")
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.x0 = x0

    def _evolve(self, dt, shocks, rng):
        values = np.empty(len(shocks) + 1)
        values[0] = self.x0
        sqrt_dt = np.sqrt(dt)
        for i, z in enumerate(shocks):
            x_prev = max(values[i], 0.0)
            drift = self.kappa * (self.theta - x_prev) * dt
            diffusion = self.xi * np.sqrt(x_prev) * sqrt_dt * z
            values[i + 1] = max(x_prev + drift + diffusion, 0.0)  # truncate the stored value too
        return values


class OrnsteinUhlenbeckProcess(PathSimulator):
    """
    Ornstein-Uhlenbeck (OU) mean-reverting process:
        dX = kappa * (theta - X) * dt + sigma * dW

    Mean-reverting like CIR, but with CONSTANT volatility and no floor.
    CIR goes quiet near zero; OU wiggles just as much everywhere, so it
    sounds like symmetric noise around a tonal centre.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run mean.
    sigma : float
        Volatility (constant).
    x0 : float
        Starting value. Only its distance to theta is audible.
    """

    def __init__(self, kappa: float = 2.0, theta: float = 0.0, sigma: float = 1.0, x0: float = 0.0):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.x0 = x0

    def _evolve(self, dt, shocks, rng):
        values = np.empty(len(shocks) + 1)
        values[0] = self.x0
        noise = self.sigma * np.sqrt(dt) * shocks
        for i in range(len(shocks)):
            values[i + 1] = values[i] + self.kappa * (self.theta - values[i]) * dt + noise[i]
        return values


class GBMProcess(PathSimulator):
    """
    Geometric Brownian motion (the Black-Scholes benchmark):
        dS = mu * S * dt + sigma * S * dW

    Simulated exactly through its log-space solution:
        S_t = x0 * exp((mu - sigma^2 / 2) * t + sigma * W_t)

    The noise is PROPORTIONAL to the current level, so the path gets more
    agitated as it climbs and calmer as it falls - an asymmetry additive
    Brownian motion cannot produce.

    Parameters
    ----------
    mu : float
        Drift rate.
    sigma : float
        Volatility coefficient.
    x0 : float
        Starting value. Must be > 0. A pure scale factor.
    """

    def __init__(self, mu: float = 0.05, sigma: float = 0.3, x0: float = 1.0):
        if x0 <= 0:
            raise ValueError(f"x0 must be > 0 for GBM, got {x0}")
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0

    def _evolve(self, dt, shocks, rng):
        log_increments = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * shocks
        return np.exp(_cumulate(np.log(self.x0), log_increments))


class HestonProcess(PathSimulator):
    """
    Stochastic-volatility process whose value CAN go negative, driven by
    a CIR (or rough CIR) variance process.

        dX = sqrt(v) * dW^X                            <- the audible path
        dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW^v    <- the variance

    with Corr(dW^X, dW^v) = rho.

    This is the *additive* form, not textbook multiplicative Heston
    (dS = mu*S*dt + sqrt(v)*S*dW, which stays positive like a stock
    price). X wanders freely above and below its starting point while
    the variance modulates how agitated that wandering is: v controls
    the *energy* of the melody, not its level - calm stretches
    alternating with busy ones.

    Rough volatility
    ----------------
    With rough=True the variance follows a rough Heston dynamic: the CIR
    increments are convolved with a fractional (Volterra) kernel
    (t - s)^(H - 1/2) / Gamma(H + 1/2), H being the Hurst exponent:

        v_t = v0 + (1/Gamma(a)) * integral_0^t (t-s)^(a-1)
                 * [ kappa*(theta - v_s) ds + xi*sqrt(v_s) dW^v_s ],
        a = H + 1/2.

    H = 0.5 recovers the standard CIR variance (the kernel becomes 1).
    H < 0.5 is the "rough" regime seen in real volatility data (H around
    0.1): a far more jagged, bursty variance.

    The rough scheme is a Volterra Euler discretization, O(n^2) because
    every step depends on the whole history. To bound the cost, the path
    is simulated on a coarser internal grid of at most `max_rough_steps`
    points and interpolated back. Price and variance share that grid, so
    their shocks stay correlated by rho.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed of the variance.
    theta : float
        Long-run mean variance.
    xi : float
        Vol-of-vol.
    rho : float
        Correlation between the path and variance shocks, in [-1, 1].
    v0 : float
        Initial variance. Must be >= 0.
    x0 : float
        Initial value of the audible path.
    rough : bool
        Use the rough (fractional) variance dynamic.
    hurst : float
        Hurst exponent H in (0, 1). Only used when rough=True.
    max_rough_steps : int
        Cap on the internal grid size of the rough scheme.
    """

    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.2,
        xi: float = 0.4,
        rho: float = -0.7,
        v0: float = 0.2,
        x0: float = 0.0,
        rough: bool = False,
        hurst: float = 0.1,
        max_rough_steps: int = 3000,
    ):
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {rho}")
        if rough and not 0.0 < hurst < 1.0:
            raise ValueError(f"hurst must be in (0, 1), got {hurst}")
        if v0 < 0:
            raise ValueError(f"v0 must be >= 0, got {v0}")
        if max_rough_steps < 1:
            raise ValueError("max_rough_steps must be >= 1")

        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0
        self.x0 = x0
        self.rough = rough
        self.hurst = hurst
        self.max_rough_steps = max_rough_steps

    def _variance_shocks(self, z_x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Variance shocks with correlation rho to the path shocks z_x."""
        z_ind = rng.standard_normal(len(z_x))
        return self.rho * z_x + np.sqrt(1.0 - self.rho**2) * z_ind

    def _variance_standard(self, dt: float, z_v: np.ndarray) -> np.ndarray:
        """Standard CIR variance: Euler-Maruyama with full truncation."""
        v = np.empty(len(z_v) + 1)
        v[0] = self.v0
        sqrt_dt = np.sqrt(dt)
        for i, z in enumerate(z_v):
            v_prev = max(v[i], 0.0)
            drift = self.kappa * (self.theta - v_prev) * dt
            diffusion = self.xi * np.sqrt(v_prev) * sqrt_dt * z
            v[i + 1] = max(v_prev + drift + diffusion, 0.0)
        return v

    def _variance_rough(self, dt: float, z_v: np.ndarray) -> np.ndarray:
        """
        Rough variance via a Volterra Euler scheme:

        v_{i+1} = v0 + (1/Gamma(a)) * sum_{j<=i} (t_{i+1} - t_j)^(a-1)
                        * [ kappa*(theta - v_j)*dt + xi*sqrt(v_j)*sqrt(dt)*z_j ]
        """
        n = len(z_v)
        alpha = self.hurst + 0.5
        sqrt_dt = np.sqrt(dt)

        v = np.empty(n + 1)
        v[0] = self.v0
        increments = np.zeros(n)
        # kernel[k] = weight of an increment generated k steps before the current one
        kernel = (np.arange(1, n + 1) * dt) ** (alpha - 1.0) / gamma_fn(alpha)

        for i in range(n):
            v_prev = max(v[i], 0.0)
            increments[i] = (
                self.kappa * (self.theta - v_prev) * dt
                + self.xi * np.sqrt(v_prev) * sqrt_dt * z_v[i]
            )
            v[i + 1] = max(self.v0 + np.dot(kernel[i::-1], increments[: i + 1]), 0.0)
        return v

    def _evolve(self, dt, shocks, rng):
        n = len(shocks)

        if not self.rough:
            v = self._variance_standard(dt, self._variance_shocks(shocks, rng))
            values = _cumulate(self.x0, np.sqrt(v[:-1]) * np.sqrt(dt) * shocks)
            return values, {"variance": v}

        # Rough case: merge `stride` fine steps into one coarse step. Summing
        # the fine shocks (and rescaling) gives the coarse shocks, so the path
        # still follows the supplied Brownian motion - which is what keeps
        # correlated voices correlated.
        stride = int(np.ceil(n / self.max_rough_steps))
        coarse_n = int(np.ceil(n / stride))
        padding = rng.standard_normal(coarse_n * stride - n)
        z_x = np.concatenate([shocks, padding]).reshape(coarse_n, stride).sum(axis=1)
        z_x /= np.sqrt(stride)
        coarse_dt = dt * stride

        v_coarse = self._variance_rough(coarse_dt, self._variance_shocks(z_x, rng))
        x_coarse = _cumulate(self.x0, np.sqrt(v_coarse[:-1]) * np.sqrt(coarse_dt) * z_x)

        fine_times = np.arange(n + 1) * dt
        coarse_times = np.arange(coarse_n + 1) * coarse_dt
        values = np.interp(fine_times, coarse_times, x_coarse)
        v = np.interp(fine_times, coarse_times, v_coarse)
        return values, {"variance": v}

    def simulate_with_variance(self, T: float, dt: float, seed: int | None = None):
        """Same as simulate(), but returns (times, values, variance)."""
        times, values, latent = self.simulate_with_latent(T, dt, seed)
        return times, values, latent["variance"]


class SABRProcess(PathSimulator):
    """
    SABR (Stochastic Alpha Beta Rho) model, simulated as a plain Monte
    Carlo path (not through its closed-form smile approximation):

        dF     = sigma * F^beta * dW^F
        dsigma = alpha * sigma * dW^sigma
        Corr(dW^F, dW^sigma) = rho

    F is the audible path. sigma is a driftless stochastic volatility:
    unlike Heston's variance it has no mean reversion, so the energy of
    the melody can drift away for good. beta sets how the noise scales
    with the level: beta=1 is lognormal-like (as in GBM), beta=0 is
    normal-like (level-independent), values in between interpolate.

    Euler-Maruyama, with F and sigma floored at zero each step (F^beta
    needs F >= 0, and discretization error can push it below).

    Parameters
    ----------
    alpha : float
        Vol-of-vol.
    beta : float
        CEV exponent on F, in [0, 1].
    rho : float
        Correlation between the F and sigma shocks, in [-1, 1].
    sigma0 : float
        Initial volatility. Must be >= 0.
    f0 : float
        Initial value of F. Must be >= 0.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.5,
        rho: float = -0.3,
        sigma0: float = 1.0,
        f0: float = 5.0,
    ):
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {rho}")
        if sigma0 < 0:
            raise ValueError(f"sigma0 must be >= 0, got {sigma0}")
        if f0 < 0:
            raise ValueError(f"f0 must be >= 0, got {f0}")

        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.sigma0 = sigma0
        self.f0 = f0

    def _evolve(self, dt, shocks, rng):
        n = len(shocks)
        z_ind = rng.standard_normal(n)
        z_sigma = self.rho * shocks + np.sqrt(1.0 - self.rho**2) * z_ind
        sqrt_dt = np.sqrt(dt)

        f = np.empty(n + 1)
        sig = np.empty(n + 1)
        f[0] = self.f0
        sig[0] = self.sigma0
        for i in range(n):
            f[i + 1] = max(f[i] + sig[i] * f[i] ** self.beta * sqrt_dt * shocks[i], 0.0)
            sig[i + 1] = max(sig[i] + self.alpha * sig[i] * sqrt_dt * z_sigma[i], 0.0)
        return f, {"volatility": sig}


class MertonJumpDiffusion(PathSimulator):
    """
    Merton jump-diffusion: Brownian motion with drift plus Poisson-driven
    jumps.

        dX = mu * dt + sigma * dW + dJ

    where jumps arrive at rate lambda_jump and have sizes
    N(jump_mean, jump_std^2).

    Kept ADDITIVE (jumps and diffusion add directly to X), consistent
    with BrownianMotion and HestonProcess, instead of the textbook
    multiplicative form on top of GBM.

    Every other process here is continuous, so two neighbouring notes can
    only differ by however far the path wandered in between. Jumps are
    real discontinuities: sudden melodic leaps no continuous process can
    produce.

    Parameters
    ----------
    mu : float
        Drift per unit time (continuous part).
    sigma : float
        Volatility of the continuous part.
    x0 : float
        Starting value.
    lambda_jump : float
        Expected number of jumps per unit time. Must be >= 0.
    jump_mean : float
        Mean jump size.
    jump_std : float
        Standard deviation of the jump size. Must be >= 0.
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        x0: float = 0.0,
        lambda_jump: float = 0.5,
        jump_mean: float = 0.0,
        jump_std: float = 3.0,
    ):
        if lambda_jump < 0:
            raise ValueError(f"lambda_jump must be >= 0, got {lambda_jump}")
        if jump_std < 0:
            raise ValueError(f"jump_std must be >= 0, got {jump_std}")

        self.mu = mu
        self.sigma = sigma
        self.x0 = x0
        self.lambda_jump = lambda_jump
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def _evolve(self, dt, shocks, rng):
        n = len(shocks)
        # Compound Poisson part: a jump COUNT per step, then that many sizes,
        # summed back into the step they belong to.
        counts = rng.poisson(self.lambda_jump * dt, size=n)
        sizes = rng.normal(self.jump_mean, self.jump_std, size=int(counts.sum()))
        jumps = np.zeros(n)
        np.add.at(jumps, np.repeat(np.arange(n), counts), sizes)

        increments = self.mu * dt + self.sigma * np.sqrt(dt) * shocks + jumps
        return _cumulate(self.x0, increments)


class VarianceGammaProcess(PathSimulator):
    """
    Variance Gamma (VG) process: a pure-jump process with no Brownian
    component, built by running Brownian motion with drift on a random
    Gamma clock:

        X_t = x0 + mu * G_t + sigma * W(G_t)

    G_t is a Gamma process with mean rate 1 and variance rate nu.
    Instead of ordinary time, the Brownian motion runs on a "business
    time" that speeds up and slows down: large nu gives bursts of big
    moves separated by quiet stretches.

    Each step is exact in distribution: draw dG ~ Gamma(dt/nu, nu), then
    a Gaussian increment with mean mu*dG and variance sigma^2*dG.

    At one note per beat the infinite jump activity is not audible as
    such; what stands out is the heavier tails (the occasional much
    larger move), more so as nu grows.

    Parameters
    ----------
    mu : float
        Drift of the time-changed Brownian motion.
    sigma : float
        Volatility of the time-changed Brownian motion.
    nu : float
        Variance rate of the Gamma clock. Must be > 0. nu -> 0 recovers
        Brownian motion with drift.
    x0 : float
        Starting value.
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0, nu: float = 0.3, x0: float = 0.0):
        if nu <= 0:
            raise ValueError(f"nu must be > 0, got {nu}")
        self.mu = mu
        self.sigma = sigma
        self.nu = nu
        self.x0 = x0

    def _evolve(self, dt, shocks, rng):
        # Gamma clock increments: shape dt/nu, scale nu -> mean dt, variance nu*dt
        dg = rng.gamma(shape=dt / self.nu, scale=self.nu, size=len(shocks))
        return _cumulate(self.x0, self.mu * dg + self.sigma * np.sqrt(dg) * shocks)


# --------------------------------------------------------------------------
# Correlated voices
# --------------------------------------------------------------------------


def as_correlation_matrix(correlation, n: int) -> np.ndarray:
    """
    Turn `correlation` into a validated n x n correlation matrix.

    Accepts None (independent voices), a single number (the same
    correlation between every pair of voices) or a full matrix.
    """
    if correlation is None:
        return np.eye(n)

    corr = np.asarray(correlation, dtype=float)
    if corr.ndim == 0:
        corr = np.full((n, n), float(corr))
        np.fill_diagonal(corr, 1.0)

    if corr.shape != (n, n):
        raise ValueError(f"Correlation matrix must have shape ({n}, {n}), got {corr.shape}.")
    if not np.allclose(corr, corr.T, atol=1e-8):
        raise ValueError("Correlation matrix must be symmetric.")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
        raise ValueError("Correlation matrix must have 1.0 on the diagonal.")
    smallest = float(np.min(np.linalg.eigvalsh(corr)))
    if smallest < -1e-8:
        raise ValueError(
            "Correlation matrix must be positive semi-definite "
            f"(smallest eigenvalue: {smallest:.4f}). With one shared correlation "
            f"between {n} voices, it cannot go below {-1.0 / max(n - 1, 1):.3f}."
        )
    return corr


def correlated_shocks(
    correlation, num_voices: int, num_steps: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Standard normal shocks of shape (num_voices, num_steps) whose rows
    have the requested correlation.

    Uses a Cholesky factor, so voice i only depends on the raw draws of
    voices 0..i: adding a voice leaves the earlier ones unchanged.
    """
    corr = as_correlation_matrix(correlation, num_voices)
    try:
        factor = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # singular matrix (e.g. a correlation of exactly 1): use an eigen factor
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    return factor @ rng.standard_normal((num_voices, num_steps))


def simulate_correlated(
    processes: Sequence[PathSimulator],
    T: float,
    dt: float,
    correlation=None,
    seed: int | np.random.SeedSequence | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[Latent]]:
    """
    Simulate several processes whose main Brownian shocks are correlated.

    The processes can be of any kind and need not be the same. The
    correlation applies to the main shocks: two Wiener voices at 0.9
    move almost in parallel; at -0.9 they move in contrary motion. For
    models with extra randomness of their own (jumps, stochastic
    volatility, random clocks) the paths end up less correlated than the
    shocks.

    Returns
    -------
    times : np.ndarray, shape (num_steps + 1,)
    values : list of np.ndarray, one path per process
    latents : list of dict, the hidden series of each process (may be empty)
    """
    if len(processes) == 0:
        raise ValueError("At least one process is required.")

    seed_seq = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    children = seed_seq.spawn(len(processes) + 1)

    times = time_grid(T, dt)
    shocks = correlated_shocks(
        correlation, len(processes), len(times) - 1, np.random.default_rng(children[0])
    )

    values, latents = [], []
    for process, row, child in zip(processes, shocks, children[1:], strict=True):
        path, latent = process.simulate_driven(times, row, np.random.default_rng(child))
        values.append(path)
        latents.append(latent)
    return times, values, latents


class MultiVoiceSimulator:
    """
    N correlated voices, each driven by its own process.

    Keeps the simulate(T, dt, seed) signature of a single process, but
    returns values of shape (num_voices, num_steps + 1).

    Parameters
    ----------
    processes : sequence of PathSimulator
        One process per voice (any mix of models).
    correlation : None, float or array of shape (N, N)
        Correlation between the voices' main shocks. See
        as_correlation_matrix.
    """

    def __init__(self, processes: Sequence[PathSimulator], correlation=None):
        self.processes = list(processes)
        self.num_voices = len(self.processes)
        self.correlation_matrix = as_correlation_matrix(correlation, self.num_voices)

    def simulate(
        self, T: float, dt: float, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        times, values, _ = simulate_correlated(self.processes, T, dt, self.correlation_matrix, seed)
        return times, np.vstack(values)
