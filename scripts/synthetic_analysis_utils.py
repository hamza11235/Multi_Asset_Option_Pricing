from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def basket_values(prices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(prices, dtype=float) @ np.asarray(weights, dtype=float)


def basket_call_payoff(prices: np.ndarray, weights: np.ndarray, strike: float) -> np.ndarray:
    return np.maximum(basket_values(prices, weights) - strike, 0.0)


def equicorrelation_matrix(n_assets: int, rho: float) -> np.ndarray:
    matrix = np.full((n_assets, n_assets), rho, dtype=float)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def build_transition_matrix(p01_daily: float, p10_daily: float) -> np.ndarray:
    return np.array(
        [
            [1.0 - p01_daily, p01_daily],
            [p10_daily, 1.0 - p10_daily],
        ],
        dtype=float,
    )


def average_off_diagonal(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=float)
    mask = ~np.eye(values.shape[0], dtype=bool)
    return float(values[mask].mean())


def empirical_return_correlation(paths: np.ndarray) -> np.ndarray:
    log_returns = np.log(paths[:, 1:, :] / paths[:, :-1, :]).reshape(-1, paths.shape[-1])
    return np.corrcoef(log_returns, rowvar=False)


def simulate_constant_paths(
    spot: np.ndarray,
    rate: float,
    div_yield: np.ndarray,
    vol: np.ndarray,
    maturity: float,
    n_steps: int,
    n_paths: int,
    corr: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spot = np.asarray(spot, dtype=float)
    div_yield = np.asarray(div_yield, dtype=float)
    vol = np.asarray(vol, dtype=float)
    dt = maturity / n_steps
    drift = (rate - div_yield - 0.5 * vol**2) * dt
    diffusion = vol * np.sqrt(dt)
    chol = np.linalg.cholesky(corr)

    paths = np.empty((n_paths, n_steps + 1, spot.size), dtype=float)
    paths[:, 0, :] = spot

    for step in range(n_steps):
        draws = rng.standard_normal((n_paths, spot.size)) @ chol.T
        paths[:, step + 1, :] = paths[:, step, :] * np.exp(drift + diffusion * draws)

    return paths


def simulate_regime_switching_paths(
    spot: np.ndarray,
    rate: float,
    div_yield: np.ndarray,
    vol: np.ndarray,
    maturity: float,
    n_steps: int,
    n_paths: int,
    corr_calm: np.ndarray,
    corr_stress: np.ndarray,
    transition_matrix: np.ndarray,
    start_regime: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    spot = np.asarray(spot, dtype=float)
    div_yield = np.asarray(div_yield, dtype=float)
    vol = np.asarray(vol, dtype=float)
    dt = maturity / n_steps
    drift = (rate - div_yield - 0.5 * vol**2) * dt
    diffusion = vol * np.sqrt(dt)
    chol_by_regime = {
        0: np.linalg.cholesky(corr_calm),
        1: np.linalg.cholesky(corr_stress),
    }

    paths = np.empty((n_paths, n_steps + 1, spot.size), dtype=float)
    regimes = np.empty((n_paths, n_steps), dtype=np.int8)
    paths[:, 0, :] = spot
    current_regimes = np.full(n_paths, start_regime, dtype=np.int8)

    for step in range(n_steps):
        regimes[:, step] = current_regimes
        base_draws = rng.standard_normal((n_paths, spot.size))
        correlated_draws = correlate_draws(base_draws, current_regimes, chol_by_regime)
        paths[:, step + 1, :] = paths[:, step, :] * np.exp(drift + diffusion * correlated_draws)
        if step < n_steps - 1:
            current_regimes = advance_regimes(current_regimes, transition_matrix, rng)

    return paths, regimes


def monte_carlo_price_summary(
    discounted_payoffs: np.ndarray,
    model_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n_paths = discounted_payoffs.size
    price = float(discounted_payoffs.mean())
    payoff_std = float(discounted_payoffs.std(ddof=1))
    std_error = payoff_std / np.sqrt(n_paths)
    summary = {
        "model": model_name,
        "price": price,
        "std_error": std_error,
        "ci_low": price - 1.96 * std_error,
        "ci_high": price + 1.96 * std_error,
        "discounted_payoff_std": payoff_std,
        "n_paths": int(n_paths),
    }
    if extra:
        summary.update(extra)
    return summary


def constant_terminal_factors(
    spot: np.ndarray,
    rate: float,
    div_yield: np.ndarray,
    vol: np.ndarray,
    tau: float,
    chol_constant: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if tau <= 0.0:
        return np.ones((n_samples, len(spot)))
    drift = (rate - div_yield - 0.5 * vol**2) * tau
    diffusion = vol * np.sqrt(tau)
    shocks = rng.standard_normal((n_samples, len(spot))) @ chol_constant.T
    return np.exp(drift + diffusion * shocks)


def correlate_draws(
    base_draws: np.ndarray,
    regime_states: np.ndarray,
    chol_by_regime: dict[int, np.ndarray],
) -> np.ndarray:
    correlated = np.empty_like(base_draws)
    for regime_value, chol in chol_by_regime.items():
        mask = regime_states == regime_value
        if np.any(mask):
            correlated[mask] = base_draws[mask] @ chol.T
    return correlated


def advance_regimes(
    current_regimes: np.ndarray,
    transition_matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    prob_to_stress = transition_matrix[current_regimes, 1]
    draws = rng.random(current_regimes.size)
    return (draws < prob_to_stress).astype(np.int8)


def regime_terminal_factors(
    spot: np.ndarray,
    rate: float,
    div_yield: np.ndarray,
    vol: np.ndarray,
    hedge_dt: float,
    steps_remaining: int,
    chol_calm: np.ndarray,
    chol_stress: np.ndarray,
    transition_hedge: np.ndarray,
    start_regime: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if steps_remaining <= 0:
        return np.ones((n_samples, len(spot)))

    drift = (rate - div_yield - 0.5 * vol**2) * hedge_dt
    diffusion = vol * np.sqrt(hedge_dt)
    chol_by_regime = {0: chol_calm, 1: chol_stress}
    current_regimes = np.full(n_samples, start_regime, dtype=np.int8)
    log_factors = np.zeros((n_samples, len(spot)))

    for step in range(steps_remaining):
        base_draws = rng.standard_normal((n_samples, len(spot)))
        correlated_draws = correlate_draws(base_draws, current_regimes, chol_by_regime)
        log_factors += drift + diffusion * correlated_draws
        if step < steps_remaining - 1:
            current_regimes = advance_regimes(current_regimes, transition_hedge, rng)

    return np.exp(log_factors)


def maturity_delta(spot: np.ndarray, weights: np.ndarray, strike: float) -> np.ndarray:
    basket_level = float(weights @ spot)
    if basket_level > strike:
        return weights.copy()
    if np.isclose(basket_level, strike):
        return 0.5 * weights
    return np.zeros_like(spot)


def price_and_delta_from_terminal_factors(
    spot: np.ndarray,
    terminal_factors: np.ndarray,
    weights: np.ndarray,
    strike: float,
    rate: float,
    tau: float,
    bump_fraction: float,
) -> tuple[float, np.ndarray]:
    terminal_prices = terminal_factors * spot
    discount = np.exp(-rate * tau)
    payoff = basket_call_payoff(terminal_prices, weights, strike)
    price = float(discount * payoff.mean())

    bump_sizes = bump_fraction * np.maximum(spot, 1.0)
    deltas = np.empty_like(spot)
    for asset_idx in range(len(spot)):
        up_spot = spot[asset_idx] + bump_sizes[asset_idx]
        down_spot = max(spot[asset_idx] - bump_sizes[asset_idx], 1e-8)
        denominator = up_spot - down_spot

        up_terminal = terminal_prices.copy()
        down_terminal = terminal_prices.copy()
        up_terminal[:, asset_idx] = terminal_factors[:, asset_idx] * up_spot
        down_terminal[:, asset_idx] = terminal_factors[:, asset_idx] * down_spot

        up_payoff = basket_call_payoff(up_terminal, weights, strike)
        down_payoff = basket_call_payoff(down_terminal, weights, strike)
        deltas[asset_idx] = discount * (up_payoff.mean() - down_payoff.mean()) / denominator

    return price, deltas


def constant_model_price_and_delta(
    spot: np.ndarray,
    tau: float,
    weights: np.ndarray,
    strike: float,
    rate: float,
    div_yield: np.ndarray,
    vol: np.ndarray,
    chol_constant: np.ndarray,
    n_samples: int,
    bump_fraction: float,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    if tau <= 0.0:
        payoff = float(basket_call_payoff(spot[None, :], weights, strike)[0])
        return payoff, maturity_delta(spot, weights, strike)
    terminal_factors = constant_terminal_factors(
        spot=spot,
        rate=rate,
        div_yield=div_yield,
        vol=vol,
        tau=tau,
        chol_constant=chol_constant,
        n_samples=n_samples,
        rng=rng,
    )
    return price_and_delta_from_terminal_factors(spot, terminal_factors, weights, strike, rate, tau, bump_fraction)


def regime_model_price_and_delta(
    spot: np.ndarray,
    steps_remaining: int,
    current_regime: int,
    weights: np.ndarray,
    strike: float,
    rate: float,
    div_yield: np.ndarray,
    vol: np.ndarray,
    hedge_dt: float,
    chol_calm: np.ndarray,
    chol_stress: np.ndarray,
    transition_hedge: np.ndarray,
    n_samples: int,
    bump_fraction: float,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    tau = steps_remaining * hedge_dt
    if steps_remaining <= 0:
        payoff = float(basket_call_payoff(spot[None, :], weights, strike)[0])
        return payoff, maturity_delta(spot, weights, strike)
    terminal_factors = regime_terminal_factors(
        spot=spot,
        rate=rate,
        div_yield=div_yield,
        vol=vol,
        hedge_dt=hedge_dt,
        steps_remaining=steps_remaining,
        chol_calm=chol_calm,
        chol_stress=chol_stress,
        transition_hedge=transition_hedge,
        start_regime=current_regime,
        n_samples=n_samples,
        rng=rng,
    )
    return price_and_delta_from_terminal_factors(spot, terminal_factors, weights, strike, rate, tau, bump_fraction)


def initial_hedge_from_model(
    hedge_model: str,
    pricing_inputs: dict[str, Any],
    base_seed: int,
) -> tuple[float, np.ndarray]:
    if hedge_model == "constant":
        return constant_model_price_and_delta(
            spot=pricing_inputs["spot"],
            tau=pricing_inputs["maturity"],
            weights=pricing_inputs["weights"],
            strike=pricing_inputs["strike"],
            rate=pricing_inputs["rate"],
            div_yield=pricing_inputs["div_yield"],
            vol=pricing_inputs["vol"],
            chol_constant=pricing_inputs["chol_constant"],
            n_samples=pricing_inputs["initial_price_mc_paths"],
            bump_fraction=pricing_inputs["bump_fraction"],
            rng=np.random.default_rng(base_seed + 1),
        )
    if hedge_model == "regime":
        return regime_model_price_and_delta(
            spot=pricing_inputs["spot"],
            steps_remaining=pricing_inputs["hedge_steps"],
            current_regime=pricing_inputs["start_regime"],
            weights=pricing_inputs["weights"],
            strike=pricing_inputs["strike"],
            rate=pricing_inputs["rate"],
            div_yield=pricing_inputs["div_yield"],
            vol=pricing_inputs["vol"],
            hedge_dt=pricing_inputs["hedge_dt"],
            chol_calm=pricing_inputs["chol_calm"],
            chol_stress=pricing_inputs["chol_stress"],
            transition_hedge=pricing_inputs["transition_hedge"],
            n_samples=pricing_inputs["initial_price_mc_paths"],
            bump_fraction=pricing_inputs["bump_fraction"],
            rng=np.random.default_rng(base_seed + 2),
        )
    raise ValueError(f"Unknown hedge model: {hedge_model}")


def pnl_summary(pnl: np.ndarray) -> dict[str, float]:
    return {
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl, ddof=1)),
        "median_pnl": float(np.median(pnl)),
        "q05_pnl": float(np.quantile(pnl, 0.05)),
        "q01_pnl": float(np.quantile(pnl, 0.01)),
        "min_pnl": float(np.min(pnl)),
        "max_pnl": float(np.max(pnl)),
    }


def evaluate_hedger(
    true_paths: np.ndarray,
    true_regimes: np.ndarray,
    hedge_model: str,
    pricing_inputs: dict[str, Any],
    base_seed: int,
    funding_price: float | None = None,
    oracle_regime: bool = True,
    precomputed_initial: tuple[float, np.ndarray] | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    time_start = time.perf_counter()
    if precomputed_initial is None:
        model_price, initial_delta = initial_hedge_from_model(hedge_model, pricing_inputs, base_seed)
    else:
        model_price, initial_delta = precomputed_initial
    effective_funding_price = model_price if funding_price is None else funding_price

    terminal_payoff = basket_call_payoff(true_paths[:, -1, :], pricing_inputs["weights"], pricing_inputs["strike"])
    n_paths = true_paths.shape[0]
    cash_account = np.full(n_paths, effective_funding_price) - true_paths[:, 0, :] @ initial_delta
    delta_holdings = np.tile(initial_delta, (n_paths, 1))

    for step in range(pricing_inputs["hedge_steps"] - 1):
        next_spots = true_paths[:, step + 1, :]
        cash_account *= np.exp(pricing_inputs["rate"] * pricing_inputs["hedge_dt"])
        portfolio_before = cash_account + np.sum(delta_holdings * next_spots, axis=1)
        remaining_steps = pricing_inputs["hedge_steps"] - (step + 1)
        new_deltas = np.empty_like(delta_holdings)

        for path_idx in range(n_paths):
            state_rng = np.random.default_rng(base_seed + 10000 * (step + 1) + path_idx)
            if hedge_model == "constant":
                _, state_delta = constant_model_price_and_delta(
                    spot=next_spots[path_idx],
                    tau=remaining_steps * pricing_inputs["hedge_dt"],
                    weights=pricing_inputs["weights"],
                    strike=pricing_inputs["strike"],
                    rate=pricing_inputs["rate"],
                    div_yield=pricing_inputs["div_yield"],
                    vol=pricing_inputs["vol"],
                    chol_constant=pricing_inputs["chol_constant"],
                    n_samples=pricing_inputs["delta_mc_paths"],
                    bump_fraction=pricing_inputs["bump_fraction"],
                    rng=state_rng,
                )
            else:
                current_regime = int(true_regimes[path_idx, step + 1]) if oracle_regime else pricing_inputs["start_regime"]
                _, state_delta = regime_model_price_and_delta(
                    spot=next_spots[path_idx],
                    steps_remaining=remaining_steps,
                    current_regime=current_regime,
                    weights=pricing_inputs["weights"],
                    strike=pricing_inputs["strike"],
                    rate=pricing_inputs["rate"],
                    div_yield=pricing_inputs["div_yield"],
                    vol=pricing_inputs["vol"],
                    hedge_dt=pricing_inputs["hedge_dt"],
                    chol_calm=pricing_inputs["chol_calm"],
                    chol_stress=pricing_inputs["chol_stress"],
                    transition_hedge=pricing_inputs["transition_hedge"],
                    n_samples=pricing_inputs["delta_mc_paths"],
                    bump_fraction=pricing_inputs["bump_fraction"],
                    rng=state_rng,
                )
            new_deltas[path_idx] = state_delta

        cash_account = portfolio_before - np.sum(new_deltas * next_spots, axis=1)
        delta_holdings = new_deltas

    cash_account *= np.exp(pricing_inputs["rate"] * pricing_inputs["hedge_dt"])
    pnl = cash_account + np.sum(delta_holdings * true_paths[:, -1, :], axis=1) - terminal_payoff

    summary = {
        "runtime_seconds": float(time.perf_counter() - time_start),
        "model_initial_price": float(model_price),
        "funding_initial_price": float(effective_funding_price),
        "premium_adjustment": float(model_price - effective_funding_price),
        **pnl_summary(pnl),
    }
    return summary, pnl, initial_delta


def unhedged_short_pnl(
    terminal_payoff: np.ndarray,
    rate: float,
    maturity: float,
    funding_price: float,
) -> np.ndarray:
    return funding_price * np.exp(rate * maturity) - terminal_payoff


def adjust_pnl_to_common_premium(
    pnl: np.ndarray,
    from_price: float,
    to_price: float,
    rate: float,
    maturity: float,
) -> np.ndarray:
    return pnl - (from_price - to_price) * np.exp(rate * maturity)


def summary_frame_from_results(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(results).T
    frame.index.name = "strategy"
    return frame.reset_index()
