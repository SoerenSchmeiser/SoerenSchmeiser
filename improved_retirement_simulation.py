from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationConfig:
    n_sims_calibration: int = 120
    n_sims_validation: int = 500
    horizon_years: int = 30
    success_threshold: float = 0.90
    inflation: float = 0.02
    initial_wealth: float = 1_000_000.0
    random_seed: int = 42
    n_seed_runs: int = 10

    # Inner Monte Carlo for PoS estimation
    n_inner: int = 100

    # Probability-Based Guardrails (close to original threshold style)
    pbg_lower: float = 0.75
    pbg_upper: float = 0.95
    pbg_cut: float = 0.10      # cut by 10% when PoS <= pbg_lower
    pbg_raise: float = 0.10    # raise by 10% when PoS >= pbg_upper
    cap_multiplier: float = 1.2


@dataclass
class AssetClass:
    name: str
    exp_return: float
    volatility: float


ASSETS = [
    AssetClass("Equity", 0.06, 0.18),
    AssetClass("Bonds", 0.02, 0.06),
]
CORR = np.array([[1.0, 0.2], [0.2, 1.0]])


def covariance_matrix() -> np.ndarray:
    vols = np.array([a.volatility for a in ASSETS], dtype=float)
    return np.outer(vols, vols) * CORR


def build_portfolio(stock_ratio: float = 0.6) -> np.ndarray:
    if not (0.0 <= stock_ratio <= 1.0):
        raise ValueError("stock_ratio must be between 0 and 1.")
    return np.array([stock_ratio, 1.0 - stock_ratio], dtype=float)


def simulate_asset_returns(
    n_sims: int,
    years: int,
    rng: np.random.Generator,
    mu_normal: np.ndarray | None = None,
    mu_crash: np.ndarray | None = None,
    crash_prob: float = 0.10,
    df: int = 5,
) -> np.ndarray:
    """Simulate annual arithmetic returns with fat tails and clustered crashes."""
    if df <= 2:
        raise ValueError("df must be > 2 for finite variance scaling.")

    if mu_normal is None:
        mu_normal = np.array([0.06, 0.02], dtype=float)
    if mu_crash is None:
        mu_crash = np.array([-0.25, -0.05], dtype=float)

    cov = covariance_matrix()
    L = np.linalg.cholesky(cov)

    # Scale t shocks to unit variance
    t_scale = np.sqrt(df / (df - 2))

    out = np.zeros((n_sims, years, 2), dtype=float)

    # Vectorized regime handling across simulations for speed
    in_crash = np.zeros(n_sims, dtype=bool)
    crash_years_left = np.zeros(n_sims, dtype=np.int16)

    for t in range(years):
        # Start new crashes only for paths not already in crash regime
        start_mask = (~in_crash) & (rng.random(n_sims) < crash_prob)
        if np.any(start_mask):
            crash_years_left[start_mask] = rng.integers(2, 4, size=np.sum(start_mask))
            in_crash[start_mask] = True

        # Path-dependent mean vector by regime
        mu_t = np.where(in_crash[:, None], mu_crash, mu_normal)

        # Correlated t-shocks for all paths in one draw
        z = rng.standard_t(df, size=(n_sims, 2)) / t_scale
        out[:, t, :] = mu_t + z @ L.T

        # Avoid impossible arithmetic returns below -100%
        out[:, t, :] = np.maximum(out[:, t, :], -0.99)

        # Advance crash regime
        crash_years_left[in_crash] -= 1
        ended = in_crash & (crash_years_left <= 0)
        in_crash[ended] = False

    return out


def portfolio_returns(asset_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return asset_returns @ weights


def estimate_pos(
    wealth: float,
    withdrawal: float,
    years: int,
    weights: np.ndarray,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> float:
    """Estimate probability of success from current state via inner MC."""
    if wealth <= 0:
        return 0.0
    if years <= 0:
        return 1.0

    # Vectorized inner simulation for speed and reduced noise
    asset_ret = simulate_asset_returns(config.n_inner, years, rng)
    port_ret = portfolio_returns(asset_ret, weights)

    w = np.full(config.n_inner, wealth, dtype=float)
    wd = np.full(config.n_inner, withdrawal, dtype=float)

    alive = np.ones(config.n_inner, dtype=bool)
    for t in range(years):
        w[alive] -= wd[alive]
        alive &= w > 0
        if not np.any(alive):
            return 0.0

        w[alive] *= (1.0 + port_ret[alive, t])
        wd[alive] *= (1.0 + config.inflation)

    return float(np.mean(w > 0))


def pbg_adjustment(pos: float, config: SimulationConfig) -> float:
    """Classic threshold-based PBG adjustment."""
    if pos <= config.pbg_lower:
        return 1.0 - config.pbg_cut
    if pos >= config.pbg_upper:
        return 1.0 + config.pbg_raise
    return 1.0


def run_pbg(
    path_returns: np.ndarray,
    weights: np.ndarray,
    config: SimulationConfig,
    rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_sims, years = path_returns.shape
    terminal = np.zeros(n_sims, dtype=float)
    success = np.ones(n_sims, dtype=bool)
    final_withdrawals = np.zeros(n_sims, dtype=float)

    initial_wd = config.initial_wealth * rate

    for s in range(n_sims):
        wealth = config.initial_wealth
        wd = initial_wd

        for t in range(years):
            # Re-estimate PoS using independent inner RNG draws
            pos = estimate_pos(wealth, wd, years - t, weights, config, rng)

            # Threshold-based PBG adjustment (close to original strategy)
            wd *= pbg_adjustment(pos, config)

            # Inflation-indexed cap
            cap = initial_wd * config.cap_multiplier * ((1.0 + config.inflation) ** t)
            wd = min(wd, cap)

            # Withdrawal then return
            wealth -= wd
            if wealth <= 0:
                success[s] = False
                wealth = 0.0
                break

            wealth *= (1.0 + path_returns[s, t])
            wd *= (1.0 + config.inflation)

        terminal[s] = wealth
        final_withdrawals[s] = wd

    return terminal, success, final_withdrawals


def success_rate(
    path_returns: np.ndarray,
    weights: np.ndarray,
    config: SimulationConfig,
    rate: float,
    rng: np.random.Generator,
) -> float:
    _, success, _ = run_pbg(path_returns, weights, config, rate, rng)
    return float(success.mean())


def find_swr(
    path_returns: np.ndarray,
    weights: np.ndarray,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> float:
    low, high = 0.0, 0.08
    for _ in range(12):
        mid = (low + high) / 2.0
        sr = success_rate(path_returns, weights, config, mid, rng)
        if sr >= config.success_threshold:
            low = mid
        else:
            high = mid
    return low


def run_for_seed(
    config: SimulationConfig,
    seed: int,
    weights: np.ndarray,
) -> tuple[float, float, float, float, float, float, float]:
    # Use independent RNG streams to avoid coupling across calibration/search/evaluation
    seed_seq = np.random.SeedSequence(seed)
    (
        rng_cal_paths,
        rng_val_paths,
        rng_swr_search,
        rng_cal_eval,
        rng_val_eval,
    ) = [np.random.default_rng(s) for s in seed_seq.spawn(5)]

    # Split data to reduce in-sample bias
    cal_asset_ret = simulate_asset_returns(config.n_sims_calibration, config.horizon_years, rng_cal_paths)
    val_asset_ret = simulate_asset_returns(config.n_sims_validation, config.horizon_years, rng_val_paths)

    cal_port_ret = portfolio_returns(cal_asset_ret, weights)
    val_port_ret = portfolio_returns(val_asset_ret, weights)

    swr = find_swr(cal_port_ret, weights, config, rng_swr_search)

    cal_terminal, cal_success, cal_final_wd = run_pbg(cal_port_ret, weights, config, swr, rng_cal_eval)
    val_terminal, val_success, val_final_wd = run_pbg(val_port_ret, weights, config, swr, rng_val_eval)
    return (
        swr,
        float(cal_success.mean()),
        float(np.median(cal_terminal)),
        float(np.median(cal_final_wd)),
        float(val_success.mean()),
        float(np.median(val_terminal)),
        float(np.median(val_final_wd)),
    )


def main() -> None:
    config = SimulationConfig()
    weights = build_portfolio(0.6)

    # Generate 10 random seeds from a master seed so the seed list itself is reproducible
    seed_rng = np.random.default_rng(config.random_seed)
    seed_values = seed_rng.integers(0, 2**32 - 1, size=config.n_seed_runs, dtype=np.uint32)

    print(f"=== Multi-Seed Run ({config.n_seed_runs} random seeds) ===")
    for i, seed in enumerate(seed_values, start=1):
        swr, cal_sr, cal_med_wealth, cal_med_final_wd, val_sr, val_med_wealth, val_med_final_wd = run_for_seed(
            config, int(seed), weights
        )
        print(
            f"Run {i:02d} | Seed {int(seed):10d} | "
            f"SWR {swr:5.2%} | Cal SR {cal_sr:6.2%} | Cal Median End Wealth {cal_med_wealth:>10,.0f} | "
            f"Cal Median Final WD {cal_med_final_wd:>9,.0f}"
        )
        print(
            f"         Validation -> SR {val_sr:6.2%} | Median End Wealth {val_med_wealth:>10,.0f} | "
            f"Median Final WD {val_med_final_wd:>9,.0f}"
        )


if __name__ == "__main__":
    main()
