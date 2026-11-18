"""
Catalog of the available processes and their parameters.

One place describing, for every process, a human-readable label, a
and the sensible range of each parameter. The web app builds its controls from this, so adding a
process to the app means adding one entry here.

`audible=False` marks parameters that cannot be heard through
RealizedRangeBinner because they only shift or rescale the path (see
pitch_mapping). They still exist on the classes; the app hides them.
"""

from dataclasses import dataclass

from .paths import (
    BrownianBridge,
    BrownianMotion,
    CIRProcess,
    FractionalBrownianMotion,
    GBMProcess,
    HestonProcess,
    MertonJumpDiffusion,
    OrnsteinUhlenbeckProcess,
    PathSimulator,
    SABRProcess,
    VarianceGammaProcess,
    WienerProcess,
)


@dataclass(frozen=True)
class Param:
    """One constructor argument of a process, with its slider range."""

    name: str
    label: str
    default: float | bool
    low: float = 0.0
    high: float = 1.0
    step: float = 0.01
    audible: bool = True
    requires: str | None = None  # only relevant when this boolean parameter is on

    @property
    def is_flag(self) -> bool:
        return isinstance(self.default, bool)


@dataclass(frozen=True)
class ProcessInfo:
    """A process class with its label and parameters."""

    key: str
    label: str
    cls: type[PathSimulator]
    params: tuple[Param, ...]

    def build(self, **overrides) -> PathSimulator:
        """Instantiate the process with catalog defaults, overridden as given."""
        kwargs = {param.name: param.default for param in self.params}
        kwargs.update(overrides)
        return self.cls(**kwargs)


_X0 = Param("x0", "x0 (start)", 0.0, -10.0, 10.0, 0.1, audible=False)

_ENTRIES = [
    ProcessInfo(
        "wiener",
        "Wiener process",
        WienerProcess,
        (Param("sigma", "sigma (volatility)", 1.0, 0.05, 10.0, 0.05, audible=False), _X0),
    ),
    ProcessInfo(
        "bm",
        "Brownian motion (with drift)",
        BrownianMotion,
        (
            Param("mu", "mu (drift)", 0.1, -20.0, 20.0, 0.1),
            Param("sigma", "sigma (volatility)", 1.0, 0.05, 10.0, 0.05),
            _X0,
        ),
    ),
    ProcessInfo(
        "fbm",
        "Fractional Brownian motion",
        FractionalBrownianMotion,
        (
            Param("hurst", "H (Hurst exponent)", 0.7, 0.05, 0.95, 0.01),
            Param("sigma", "sigma (scale)", 1.0, 0.05, 10.0, 0.05, audible=False),
            _X0,
        ),
    ),
    ProcessInfo(
        "bridge",
        "Brownian bridge",
        BrownianBridge,
        (
            Param("x_end", "x_end (end value)", 0.0, -20.0, 20.0, 0.1),
            Param("sigma", "sigma (volatility)", 1.0, 0.05, 10.0, 0.05),
            _X0,
        ),
    ),
    ProcessInfo(
        "gbm",
        "Geometric Brownian motion",
        GBMProcess,
        (
            Param("mu", "mu (drift)", 0.05, -2.0, 2.0, 0.01),
            Param("sigma", "sigma (volatility)", 0.3, 0.05, 3.0, 0.01),
            Param("x0", "x0 (start)", 1.0, 0.1, 10.0, 0.1, audible=False),
        ),
    ),
    ProcessInfo(
        "ou",
        "Ornstein-Uhlenbeck",
        OrnsteinUhlenbeckProcess,
        (
            Param("kappa", "kappa (pull speed)", 2.0, 0.05, 50.0, 0.05),
            Param("theta", "theta (centre)", 0.0, -30.0, 30.0, 0.05),
            Param("sigma", "sigma (volatility)", 1.0, 0.05, 10.0, 0.05),
            Param("x0", "x0 (start)", 0.0, -10.0, 10.0, 0.1),
        ),
    ),
    ProcessInfo(
        "cir",
        "CIR",
        CIRProcess,
        (
            Param("kappa", "kappa (pull speed)", 2.0, 0.05, 50.0, 0.05),
            Param("theta", "theta (centre)", 1.0, 0.0, 30.0, 0.05),
            Param("xi", "xi (volatility)", 0.5, 0.05, 30.0, 0.05),
            Param("x0", "x0 (start)", 1.0, 0.0, 30.0, 0.05),
        ),
    ),
    ProcessInfo(
        "heston",
        "Heston (stochastic volatility)",
        HestonProcess,
        (
            Param("kappa", "kappa (variance pull speed)", 2.0, 0.05, 50.0, 0.05),
            Param("theta", "theta (long-run variance)", 0.2, 0.0, 30.0, 0.05),
            Param("xi", "xi (vol of vol)", 0.4, 0.05, 30.0, 0.05),
            Param("rho", "rho (path/variance correlation)", -0.7, -0.99, 0.99, 0.01),
            Param("v0", "v0 (initial variance)", 0.2, 0.0, 30.0, 0.05),
            _X0,
            Param("rough", "Rough volatility", False),
            Param("hurst", "H (Hurst exponent)", 0.1, 0.01, 0.99, 0.01, requires="rough"),
        ),
    ),
    ProcessInfo(
        "sabr",
        "SABR (stochastic volatility)",
        SABRProcess,
        (
            Param("alpha", "alpha (vol of vol)", 0.3, 0.0, 5.0, 0.01),
            Param("beta", "beta (level dependence)", 0.5, 0.0, 1.0, 0.01),
            Param("rho", "rho (path/volatility correlation)", -0.3, -0.99, 0.99, 0.01),
            Param("sigma0", "sigma0 (initial volatility)", 1.0, 0.0, 10.0, 0.05),
            Param("f0", "f0 (start)", 5.0, 0.0, 30.0, 0.1),
        ),
    ),
    ProcessInfo(
        "merton",
        "Merton jump-diffusion",
        MertonJumpDiffusion,
        (
            Param("mu", "mu (drift)", 0.0, -20.0, 20.0, 0.1),
            Param("sigma", "sigma (volatility)", 1.0, 0.05, 10.0, 0.05),
            _X0,
            Param("lambda_jump", "lambda (jumps per beat)", 0.5, 0.0, 10.0, 0.05),
            Param("jump_mean", "jump mean", 0.0, -20.0, 20.0, 0.1),
            Param("jump_std", "jump std", 3.0, 0.0, 30.0, 0.1),
        ),
    ),
    ProcessInfo(
        "vg",
        "Variance Gamma",
        VarianceGammaProcess,
        (
            Param("mu", "mu (drift)", 0.0, -20.0, 20.0, 0.1),
            Param("sigma", "sigma (volatility)", 1.0, 0.05, 10.0, 0.05),
            Param("nu", "nu (clock irregularity)", 0.3, 0.01, 5.0, 0.01),
            _X0,
        ),
    ),
]

PROCESSES: dict[str, ProcessInfo] = {entry.key: entry for entry in _ENTRIES}
