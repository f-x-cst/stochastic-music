import numpy as np
import pytest

import stochastic_music as sm
from stochastic_music.paths import as_correlation_matrix, correlated_shocks

ALL_PROCESSES = [info.build() for info in sm.PROCESSES.values()] + [
    sm.HestonProcess(rough=True, hurst=0.1)
]


@pytest.mark.parametrize("process", ALL_PROCESSES, ids=lambda p: type(p).__name__)
def test_shape_and_finite(process):
    times, values = process.simulate(T=8, dt=0.01, seed=0)
    assert times.shape == values.shape == (801,)
    assert times[0] == 0 and times[-1] == pytest.approx(8)
    assert np.isfinite(values).all()


@pytest.mark.parametrize("process", ALL_PROCESSES, ids=lambda p: type(p).__name__)
def test_seed_is_reproducible(process):
    _, a = process.simulate(T=4, dt=0.01, seed=123)
    _, b = process.simulate(T=4, dt=0.01, seed=123)
    _, c = process.simulate(T=4, dt=0.01, seed=124)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_wiener_increment_variance():
    sigma, dt = 2.0, 0.01
    _, values = sm.WienerProcess(sigma=sigma).simulate(T=1000, dt=dt, seed=1)
    assert np.var(np.diff(values)) == pytest.approx(sigma**2 * dt, rel=0.03)


def test_brownian_drift():
    _, values = sm.BrownianMotion(mu=3.0, sigma=0.1).simulate(T=100, dt=0.01, seed=1)
    assert np.mean(np.diff(values)) / 0.01 == pytest.approx(3.0, rel=0.05)


def test_gbm_stays_positive():
    for seed in range(10):
        _, values = sm.GBMProcess(mu=-1.0, sigma=2.0).simulate(T=50, dt=0.01, seed=seed)
        assert (values > 0).all()


def test_cir_and_sabr_stay_non_negative():
    for seed in range(10):
        _, cir = sm.CIRProcess(kappa=0.5, theta=0.1, xi=3.0, x0=0.1).simulate(16, 0.01, seed)
        _, f = sm.SABRProcess(alpha=2.0, beta=0.0, f0=0.5).simulate(16, 0.01, seed)
        assert (cir >= 0).all() and (f >= 0).all()


def test_ou_reverts_to_theta():
    _, values = sm.OrnsteinUhlenbeckProcess(kappa=5, theta=10, sigma=1, x0=0).simulate(
        200, 0.01, seed=2
    )
    assert np.mean(values[2000:]) == pytest.approx(10, abs=0.2)


def test_merton_produces_jumps():
    _, values = sm.MertonJumpDiffusion(lambda_jump=2, jump_std=5).simulate(100, 0.01, seed=1)
    diffusion_scale = 1.0 * np.sqrt(0.01)
    assert np.abs(np.diff(values)).max() > 20 * diffusion_scale


def test_rough_heston_with_h_half_matches_standard_statistics():
    # H = 0.5 makes the Volterra kernel equal to 1, i.e. plain CIR variance.
    standard = sm.HestonProcess(rough=False)
    rough = sm.HestonProcess(rough=True, hurst=0.5)
    mean_std = np.mean([standard.simulate_with_variance(8, 0.01, s)[2].mean() for s in range(30)])
    mean_rough = np.mean([rough.simulate_with_variance(8, 0.01, s)[2].mean() for s in range(30)])
    assert mean_rough == pytest.approx(mean_std, rel=0.15)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cls": sm.HestonProcess, "rho": 1.5},
        {"cls": sm.HestonProcess, "rough": True, "hurst": 1.2},
        {"cls": sm.GBMProcess, "x0": 0.0},
        {"cls": sm.SABRProcess, "beta": 2.0},
        {"cls": sm.VarianceGammaProcess, "nu": 0.0},
        {"cls": sm.MertonJumpDiffusion, "lambda_jump": -1.0},
        {"cls": sm.CIRProcess, "x0": -1.0},
    ],
)
def test_invalid_parameters_raise(kwargs):
    cls = kwargs.pop("cls")
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_correlation_matrix_validation():
    assert as_correlation_matrix(None, 3).tolist() == np.eye(3).tolist()
    assert as_correlation_matrix(0.4, 3)[0, 1] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        as_correlation_matrix(-0.9, 3)  # below -1/(n-1): not positive semi-definite
    with pytest.raises(ValueError):
        as_correlation_matrix(np.eye(2), 3)


def test_correlated_shocks_have_requested_correlation():
    shocks = correlated_shocks(0.7, 3, 200_000, np.random.default_rng(0))
    measured = np.corrcoef(shocks)
    assert measured[0, 1] == pytest.approx(0.7, abs=0.01)
    assert measured[1, 2] == pytest.approx(0.7, abs=0.01)


def test_perfect_correlation_is_supported():
    shocks = correlated_shocks(1.0, 2, 1000, np.random.default_rng(0))
    np.testing.assert_allclose(shocks[0], shocks[1], atol=1e-9)


def test_correlation_works_across_different_processes():
    processes = [sm.WienerProcess(), sm.OrnsteinUhlenbeckProcess(kappa=0.1)]
    _, values, _ = sm.simulate_correlated(processes, T=200, dt=0.01, correlation=-0.8, seed=0)
    increments = [np.diff(v) for v in values]
    assert np.corrcoef(increments)[0, 1] == pytest.approx(-0.8, abs=0.05)


def test_multivoice_simulator_shape():
    sim = sm.MultiVoiceSimulator([sm.WienerProcess(), sm.HestonProcess(), sm.GBMProcess()], 0.3)
    times, values = sim.simulate(T=4, dt=0.01, seed=0)
    assert values.shape == (3, len(times))


def test_fbm_with_h_half_is_the_wiener_process():
    _, fbm = sm.FractionalBrownianMotion(hurst=0.5, sigma=2.0).simulate(T=8, dt=0.01, seed=3)
    _, wiener = sm.WienerProcess(sigma=2.0).simulate(T=8, dt=0.01, seed=3)
    np.testing.assert_allclose(fbm, wiener, atol=1e-10)


@pytest.mark.parametrize("hurst", [0.2, 0.5, 0.8])
def test_fbm_increment_statistics(hurst):
    dt, sigma = 0.01, 1.5
    _, values = sm.FractionalBrownianMotion(hurst=hurst, sigma=sigma).simulate(
        T=2000, dt=dt, seed=0
    )
    increments = np.diff(values)
    assert np.var(increments) == pytest.approx(sigma**2 * dt ** (2 * hurst), rel=0.05)
    # lag-1 correlation of fractional Gaussian noise: 2^(2H-1) - 1
    lag1 = np.corrcoef(increments[:-1], increments[1:])[0, 1]
    assert lag1 == pytest.approx(2 ** (2 * hurst - 1) - 1, abs=0.02)


def test_brownian_bridge_is_pinned_at_both_ends():
    for x_end in (0.0, 4.0):
        bridge = sm.BrownianBridge(sigma=1.0, x0=1.0, x_end=x_end)
        _, values = bridge.simulate(T=16, dt=0.01, seed=0)
        assert values[0] == pytest.approx(1.0) and values[-1] == pytest.approx(x_end)


def test_brownian_bridge_middle_variance():
    # Var(B_{T/2}) = sigma^2 * T / 4 for a bridge over [0, T]
    middles = [
        sm.BrownianBridge(sigma=1.0).simulate(T=4, dt=0.01, seed=s)[1][200] for s in range(2000)
    ]
    assert np.var(middles) == pytest.approx(1.0, rel=0.1)
