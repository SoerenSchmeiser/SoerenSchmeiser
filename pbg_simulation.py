from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

@dataclass
class SimulationConfig:
    n_sims_calibration: int = 200
    n_sims_validation: int = 400
    horizon_years: int = 30
    success_threshold: float = 0.95
    inflation: float = 0.02
    initial_wealth: float = 1_000_000.0
    random_seed: int | None = None
    n_seed_runs: int = 8

    # Nested Monte Carlo für PoS
    n_inner: int = 300

    # PBG
    pbg_lower: float = 0.75
    pbg_upper: float = 0.95
    pbg_cut: float = 0.05
    pbg_raise: float = 0.05
    cap_multiplier: float = 1.2

    # Marktmodell
    crash_prob: float = 0.03
    crash_min_years: int = 1
    crash_max_years: int = 2
    t_df: int = 8


# ============================================================
# ASSETS
# ============================================================

@dataclass
class AssetClass:
    name: str
    exp_return: float
    volatility: float
    crash_return: float


ASSETS = [
    AssetClass("Equity", 0.06, 0.15, -0.22),
    AssetClass("Bonds", 0.02, 0.01, 0.00),
]

CORR = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)


def covariance_matrix() -> np.ndarray:
    vols = np.array([a.volatility for a in ASSETS], dtype=float)
    return np.outer(vols, vols) * CORR


def build_portfolio(stock_ratio: float = 0.6) -> np.ndarray:
    if not (0.0 <= stock_ratio <= 1.0):
        raise ValueError("stock_ratio must be between 0 and 1.")
    return np.array([stock_ratio, 1.0 - stock_ratio], dtype=float)


# ============================================================
# MARKTMODELL
# ============================================================

def simulate_asset_returns(
    n_sims: int,
    years: int,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> np.ndarray:
    mu_normal = np.array([a.exp_return for a in ASSETS], dtype=float)
    mu_crash = np.array([a.crash_return for a in ASSETS], dtype=float)

    cov = covariance_matrix()
    L = np.linalg.cholesky(cov)

    df = config.t_df
    if df <= 2:
        raise ValueError("t_df must be > 2 for finite variance.")

    t_scale = np.sqrt(df / (df - 2))
    out = np.zeros((n_sims, years, len(ASSETS)), dtype=float)

    in_crash = np.zeros(n_sims, dtype=bool)
    crash_years_left = np.zeros(n_sims, dtype=int)

    for t in range(years):
        start_mask = (~in_crash) & (rng.random(n_sims) < config.crash_prob)

        if np.any(start_mask):
            crash_years_left[start_mask] = rng.integers(
                config.crash_min_years,
                config.crash_max_years + 1,
                size=np.sum(start_mask),
            )
            in_crash[start_mask] = True

        mu_t = np.where(in_crash[:, None], mu_crash, mu_normal)

        z = rng.standard_t(df, size=(n_sims, len(ASSETS))) / t_scale
        out[:, t, :] = mu_t + z @ L.T
        out[:, t, :] = np.maximum(out[:, t, :], -0.99)

        crash_years_left[in_crash] -= 1
        ended = in_crash & (crash_years_left <= 0)
        in_crash[ended] = False

    return out


def portfolio_returns(asset_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return asset_returns @ weights


# ============================================================
# PoS
# ============================================================

def estimate_pos(
    wealth: float,
    withdrawal: float,
    years: int,
    weights: np.ndarray,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> float:
    if wealth <= 0:
        return 0.0
    if years <= 0:
        return 1.0

    asset_ret = simulate_asset_returns(config.n_inner, years, rng, config)
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


# ============================================================
# PBG
# ============================================================

def pbg_adjustment(pos: float, config: SimulationConfig) -> float:
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
            pos = estimate_pos(wealth, wd, years - t, weights, config, rng)

            wd *= pbg_adjustment(pos, config)

            cap = initial_wd * config.cap_multiplier * ((1.0 + config.inflation) ** t)
            wd = min(wd, cap)

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


# ============================================================
# SWR
# ============================================================

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
    low, high = 0.00, 0.06

    for _ in range(20):
        mid = (low + high) / 2.0
        sr = success_rate(path_returns, weights, config, mid, rng)

        if sr >= config.success_threshold:
            low = mid
        else:
            high = mid

    return low


# ============================================================
# ECHTE PBG-PFADVISUALISIERUNG
# ============================================================

def simulate_pbg_paths(
    n_paths: int,
    weights: np.ndarray,
    config: SimulationConfig,
    rate: float,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simuliert wenige einzelne Pfade mit echter PBG-Logik.
    Rückgabe:
      wealth_paths: (n_paths, years+1)
      withdrawal_paths: (n_paths, years)
      success: (n_paths,)
    """
    years = config.horizon_years
    wealth_paths = np.zeros((n_paths, years + 1), dtype=float)
    withdrawal_paths = np.zeros((n_paths, years), dtype=float)
    success = np.ones(n_paths, dtype=bool)

    seed_seq = np.random.SeedSequence(rng_seed)
    rng_market, rng_policy = [np.random.default_rng(s) for s in seed_seq.spawn(2)]

    asset_ret = simulate_asset_returns(n_paths, years, rng_market, config)
    port_ret = portfolio_returns(asset_ret, weights)

    initial_wd = config.initial_wealth * rate
    wealth_paths[:, 0] = config.initial_wealth

    for s in range(n_paths):
        wealth = config.initial_wealth
        wd = initial_wd

        for t in range(years):
            pos = estimate_pos(wealth, wd, years - t, weights, config, rng_policy)

            wd *= pbg_adjustment(pos, config)

            cap = initial_wd * config.cap_multiplier * ((1.0 + config.inflation) ** t)
            wd = min(wd, cap)

            withdrawal_paths[s, t] = wd

            wealth -= wd
            if wealth <= 0:
                success[s] = False
                wealth = 0.0
                wealth_paths[s, t + 1] = 0.0
                if t + 1 < years:
                    wealth_paths[s, t + 2 :] = 0.0
                    withdrawal_paths[s, t + 1 :] = 0.0
                break

            wealth *= (1.0 + port_ret[s, t])
            wealth = max(0.0, wealth)
            wealth_paths[s, t + 1] = wealth

            wd *= (1.0 + config.inflation)

    return wealth_paths, withdrawal_paths, success


# ============================================================
# PLOTS
# ============================================================

def plot_pbg_paths(
    wealth_paths: np.ndarray,
    withdrawal_paths: np.ndarray,
    success: np.ndarray,
    config: SimulationConfig,
    rate: float,
    run_number: int,
) -> None:
    years = config.horizon_years
    x_wealth = np.arange(years + 1)
    x_withdrawal = np.arange(1, years + 1)
    x_rate = np.arange(1, years + 1)

    median_wealth = np.median(wealth_paths, axis=0)
    p10_wealth = np.percentile(wealth_paths, 10, axis=0)
    p90_wealth = np.percentile(wealth_paths, 90, axis=0)

    terminal = wealth_paths[:, -1]
    worst_idx = np.argmin(terminal)
    best_idx = np.argmax(terminal)

    median_wd = np.median(withdrawal_paths, axis=0)
    p10_wd = np.percentile(withdrawal_paths, 10, axis=0)
    p90_wd = np.percentile(withdrawal_paths, 90, axis=0)

    # Entnahmerate je Jahr = Entnahme / Vermögen zu Jahresbeginn
    wealth_start_year = wealth_paths[:, :-1]
    withdrawal_rate_paths = np.divide(
        withdrawal_paths,
        wealth_start_year,
        out=np.zeros_like(withdrawal_paths),
        where=wealth_start_year > 0,
    )
    withdrawal_rate_paths = np.clip(withdrawal_rate_paths, 0.0, 0.3)
    median_rate = np.median(withdrawal_rate_paths, axis=0)
    p10_rate = np.percentile(withdrawal_rate_paths, 10, axis=0)
    p90_rate = np.percentile(withdrawal_rate_paths, 90, axis=0)

    plt.figure(figsize=(11, 13))
    try:
        plt.gcf().canvas.manager.set_window_title(f"Run {run_number}")
    except Exception:
        pass

    ax = plt.subplot(3, 1, 1)
    n_plot = min(wealth_paths.shape[0], 20)

    for i in range(n_plot):
        ax.plot(x_wealth, wealth_paths[i], color="gray", alpha=0.20, linewidth=1)

    ax.fill_between(x_wealth, p10_wealth, p90_wealth, alpha=0.18, label="P10–P90")
    ax.plot(x_wealth, median_wealth, linewidth=2.5, label="Median")
    ax.plot(x_wealth, wealth_paths[worst_idx], linewidth=2.0, label="Worst Case")
    ax.plot(x_wealth, wealth_paths[best_idx], linewidth=2.0, label="Best Case")
    ax.axhline(0, linewidth=1)

    ax.set_title(
        f"Run {run_number} | Echte PBG-Vermögenspfade | "
        f"SWR = {rate:.2%} | Success in Visualisierung = {success.mean():.2%}"
    )
    ax.set_xlabel("Jahre")
    ax.set_ylabel("Vermögen")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = plt.subplot(3, 1, 2)

    for i in range(n_plot):
        ax.plot(x_withdrawal, withdrawal_paths[i], color="gray", alpha=0.20, linewidth=1)

    ax.fill_between(x_withdrawal, p10_wd, p90_wd, alpha=0.18, label="P10–P90")
    ax.plot(x_withdrawal, median_wd, linewidth=2.5, label="Median")
    ax.plot(x_withdrawal, withdrawal_paths[worst_idx], linewidth=2.0, label="Worst Case")
    ax.plot(x_withdrawal, withdrawal_paths[best_idx], linewidth=2.0, label="Best Case")

    ax.set_title("Echte PBG-Entnahmepfade")
    ax.set_xlabel("Jahre")
    ax.set_ylabel("Entnahme")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = plt.subplot(3, 1, 3)

    for i in range(n_plot):
        ax.plot(x_rate, withdrawal_rate_paths[i], color="gray", alpha=0.20, linewidth=1)

    ax.fill_between(x_rate, p10_rate, p90_rate, alpha=0.18, label="P10–P90")
    ax.plot(x_rate, median_rate, linewidth=2.5, label="Median")
    ax.plot(x_rate, withdrawal_rate_paths[worst_idx], linewidth=2.0, label="Worst Case")
    ax.plot(x_rate, withdrawal_rate_paths[best_idx], linewidth=2.0, label="Best Case")

    ax.set_title("Entnahmerate (Entnahme / Vermögen)")
    ax.set_xlabel("Jahre")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 0.3)
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.subplots_adjust(
        left=0.124,
        bottom=0.06,
        right=0.912,
        top=0.883,
        wspace=0.5,
        hspace=0.517,
    )
    # Kein plt.show() hier


def plot_overlay(
    all_wealth: list[np.ndarray],
    all_withdrawals: list[np.ndarray],
    swrs: list[float],
    config: SimulationConfig,
) -> None:
    years = config.horizon_years
    x_wealth = np.arange(years + 1)
    x_withdrawal = np.arange(1, years + 1)
    x_rate = np.arange(1, years + 1)

    plt.figure(figsize=(12, 11))
    try:
        plt.gcf().canvas.manager.set_window_title("Overlay")
    except Exception:
        pass

    ax = plt.subplot(3, 1, 1)
    for i, wealth_paths in enumerate(all_wealth):
        median_wealth = np.median(wealth_paths, axis=0)
        ax.plot(x_wealth, median_wealth, linewidth=2, label=f"Run {i+1} | {swrs[i]:.2%}")

    ax.set_title("Median Wealth Paths (Overlay aller Runs)")
    ax.set_xlabel("Jahre")
    ax.set_ylabel("Vermögen")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)

    ax = plt.subplot(3, 1, 2)
    for i, withdrawal_paths in enumerate(all_withdrawals):
        median_wd = np.median(withdrawal_paths, axis=0)
        ax.plot(x_withdrawal, median_wd, linewidth=2, label=f"Run {i+1} | {swrs[i]:.2%}")

    ax.set_title("Median Withdrawal Paths (Overlay aller Runs)")
    ax.set_xlabel("Jahre")
    ax.set_ylabel("Entnahme")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)

    ax = plt.subplot(3, 1, 3)
    for i, (wealth_paths, withdrawal_paths) in enumerate(zip(all_wealth, all_withdrawals)):
        wealth_start_year = wealth_paths[:, :-1]
        rate_paths = np.divide(
            withdrawal_paths,
            wealth_start_year,
            out=np.zeros_like(withdrawal_paths),
            where=wealth_start_year > 0,
        )
        rate_paths = np.clip(rate_paths, 0.0, 0.2)
        median_rate = np.median(rate_paths, axis=0)
        ax.plot(x_rate, median_rate, linewidth=2, label=f"Run {i+1} | {swrs[i]:.2%}")

    ax.set_title("Median Entnahmerate (Overlay aller Runs)")
    ax.set_xlabel("Jahre")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 0.2)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)

    plt.subplots_adjust(
        left=0.124,
        bottom=0.06,
        right=0.912,
        top=0.883,
        wspace=0.5,
        hspace=0.517,
    )
    # Kein plt.show() hier


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    config = SimulationConfig()
    weights = build_portfolio(0.8)

    if config.random_seed is None:
        config.random_seed = int(time.time() * 1000) % (2**32 - 1)

    print(f"Seed used: {config.random_seed}\n")

    master_rng = np.random.default_rng(config.random_seed)
    seeds = master_rng.integers(0, 2**32 - 1, size=config.n_seed_runs)

    results = []
    all_wealth = []
    all_withdrawals = []
    swrs = []

    print(f"=== MULTI RUN VISUALIZATION ({config.n_seed_runs} RUNS) ===\n")

    for i, seed in enumerate(seeds, 1):
        seed_seq = np.random.SeedSequence(int(seed))
        rng_cal, rng_val, rng_swr, rng_cal_eval, rng_val_eval, rng_vis = [
            np.random.default_rng(s) for s in seed_seq.spawn(6)
        ]

        # ---------------- Calibration ----------------
        cal_asset = simulate_asset_returns(
            config.n_sims_calibration, config.horizon_years, rng_cal, config
        )
        cal_port = portfolio_returns(cal_asset, weights)

        swr = find_swr(cal_port, weights, config, rng_swr)

        cal_terminal, cal_success, cal_final_wd = run_pbg(
            cal_port, weights, config, swr, rng_cal_eval
        )

        # ---------------- Validation ----------------
        val_asset = simulate_asset_returns(
            config.n_sims_validation, config.horizon_years, rng_val, config
        )
        val_port = portfolio_returns(val_asset, weights)

        val_terminal, val_success, val_final_wd = run_pbg(
            val_port, weights, config, swr, rng_val_eval
        )

        results.append([
            swr,
            cal_success.mean(),
            np.median(cal_terminal),
            np.median(cal_final_wd),
            val_success.mean(),
            np.median(val_terminal),
            np.median(val_final_wd),
        ])

        print(
            f"Run {i:02d} | SWR {swr:5.2%} | "
            f"Cal SR {cal_success.mean():6.2%} | "
            f"Cal Median End Wealth {np.median(cal_terminal):>10,.0f} | "
            f"Cal Median Final WD {np.median(cal_final_wd):>9,.0f}"
        )
        print(
            f"         Validation -> SR {val_success.mean():6.2%} | "
            f"Median End Wealth {np.median(val_terminal):>10,.0f} | "
            f"Median Final WD {np.median(val_final_wd):>9,.0f}\n"
        )

        # ---------------- Visualisierung ----------------
        wealth_paths, withdrawal_paths, vis_success = simulate_pbg_paths(
            n_paths=12,
            weights=weights,
            config=config,
            rate=swr,
            rng_seed=int(rng_vis.integers(0, 2**32 - 1)),
        )

        all_wealth.append(wealth_paths)
        all_withdrawals.append(withdrawal_paths)
        swrs.append(swr)

        plot_pbg_paths(
            wealth_paths=wealth_paths,
            withdrawal_paths=withdrawal_paths,
            success=vis_success,
            config=config,
            rate=swr,
            run_number=i,
        )

    results = np.array(results, dtype=float)

    print("=== AVERAGE OVER ALL RUNS ===\n")
    print(
        f"Avg SWR: {results[:,0].mean():.2%}\n"
        f"Avg Cal SR: {results[:,1].mean():.2%} | "
        f"Avg Val SR: {results[:,4].mean():.2%}\n"
        f"Avg Cal Median End Wealth: {results[:,2].mean():,.0f} | "
        f"Avg Val Median End Wealth: {results[:,5].mean():,.0f}\n"
        f"Avg Cal Median Final WD: {results[:,3].mean():,.0f} | "
        f"Avg Val Median Final WD: {results[:,6].mean():,.0f}"
    )

    print("\n=== VARIABILITY (ACROSS RUNS) ===\n")

    labels = [
        "SWR",
        "Cal SR",
        "Cal Median End Wealth",
        "Cal Median Final WD",
        "Val SR",
        "Val Median End Wealth",
        "Val Median Final WD",
    ]

    for i, label in enumerate(labels):
        col = results[:, i]
        min_v = col.min()
        max_v = col.max()
        range_v = max_v - min_v
        std_v = col.std()

        if i in [0, 1, 4]:
            print(
                f"{label}:\n"
                f"  Min: {min_v:.2%} | Max: {max_v:.2%} | "
                f"Range: {range_v:.2%} | Std: {std_v:.2%}\n"
            )
        else:
            print(
                f"{label}:\n"
                f"  Min: {min_v:,.0f} | Max: {max_v:,.0f} | "
                f"Range: {range_v:,.0f} | Std: {std_v:,.0f}\n"
            )

    plot_overlay(all_wealth, all_withdrawals, swrs, config)

    # Alle Figures gesammelt am Ende anzeigen
    plt.show()


if __name__ == "__main__":
    main()
