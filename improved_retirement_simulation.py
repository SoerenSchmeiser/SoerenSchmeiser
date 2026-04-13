from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationConfig:
    n_sims_calibration: int = 400
    n_sims_validation: int = 2000
    horizon_years: int = 30
    success_threshold: float = 0.90
    inflation: float = 0.02
    initial_wealth: float = 1_000_000.0
    random_seed: int = 42

    # Inner Monte Carlo for PoS estimation
    n_inner: int = 300

    # Guardrail / spending policy
    pos_target: float = 0.90
    pos_deadband: float = 0.03  # no change if PoS in [target-deadband, target+deadband]
    max_adjustment: float = 0.10  # max +/- 10% annual policy adjustment
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
    for s in range(n_sims):
        in_crash = False
        crash_years_left = 0
        for t in range(years):
            if (not in_crash) and (rng.random() < crash_prob):
                in_crash = True
                crash_years_left = int(rng.integers(2, 4))  # 2-3 years

            mu = mu_crash if in_crash else mu_normal

            z = rng.standard_t(df, size=2) / t_scale
            r = mu + L @ z

            # Avoid impossible arithmetic returns below -100%
            out[s, t] = np.maximum(r, -0.99)

            if in_crash:
                crash_years_left -= 1
                if crash_years_left == 0:
                    in_crash = False

    return out


def portfolio_returns(asset_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("sty,y->st", asset_returns, weights)


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


def policy_adjustment(pos: float, config: SimulationConfig) -> float:
    """Continuous adjustment around target PoS with deadband."""
    lo = config.pos_target - config.pos_deadband
    hi = config.pos_target + config.pos_deadband
    if lo <= pos <= hi:
        return 1.0

    # Linear response mapped to +/- max_adjustment
    diff = pos - config.pos_target
    scale = diff / max(config.pos_target, 1.0 - config.pos_target)
    adj = np.clip(scale, -1.0, 1.0) * config.max_adjustment
    return 1.0 + float(adj)


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

            # Dynamic guardrail adjustment
            wd *= policy_adjustment(pos, config)

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
    for _ in range(16):
        mid = (low + high) / 2.0
        sr = success_rate(path_returns, weights, config, mid, rng)
        if sr >= config.success_threshold:
            low = mid
        else:
            high = mid
    return low


def summarize(name: str, swr: float, terminal: np.ndarray, success: np.ndarray, final_wd: np.ndarray) -> None:
    print(f"\n=== {name} ===")
    print(f"SWR (initial rate): {swr:.2%}")
    print(f"Success Rate:       {success.mean():.2%}")
    print(f"Median End Wealth:  {np.median(terminal):,.0f}")
    print(f"Median Final WD:    {np.median(final_wd):,.0f}")


def main() -> None:
    config = SimulationConfig()
    rng = np.random.default_rng(config.random_seed)

    weights = build_portfolio(0.6)

    # Split data to reduce in-sample bias
    cal_asset_ret = simulate_asset_returns(config.n_sims_calibration, config.horizon_years, rng)
    val_asset_ret = simulate_asset_returns(config.n_sims_validation, config.horizon_years, rng)

    cal_port_ret = portfolio_returns(cal_asset_ret, weights)
    val_port_ret = portfolio_returns(val_asset_ret, weights)

    swr = find_swr(cal_port_ret, weights, config, rng)

    cal_terminal, cal_success, cal_final_wd = run_pbg(cal_port_ret, weights, config, swr, rng)
    val_terminal, val_success, val_final_wd = run_pbg(val_port_ret, weights, config, swr, rng)

    summarize("Calibration", swr, cal_terminal, cal_success, cal_final_wd)
    summarize("Validation", swr, val_terminal, val_success, val_final_wd)


if __name__ == "__main__":
    main()
