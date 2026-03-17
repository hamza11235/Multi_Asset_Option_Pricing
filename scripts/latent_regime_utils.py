from __future__ import annotations

import time
from typing import Any

import numpy as np

from synthetic_analysis_utils import (
    adjust_pnl_to_common_premium,
    basket_call_payoff,
    build_transition_matrix,
    constant_model_price_and_delta,
    evaluate_hedger,
    pnl_summary,
    regime_model_price_and_delta,
    simulate_regime_switching_paths,
    unhedged_short_pnl,
)


def gaussian_logpdf_rows(
    values: np.ndarray, mean: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    centered = values - mean
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite.")
    inv_cov = np.linalg.inv(covariance)
    quad_form = np.einsum("...i,ij,...j->...", centered, inv_cov, centered)
    dimension = mean.size
    return -0.5 * (dimension * np.log(2.0 * np.pi) + logdet + quad_form)


def state_log_return_moments(
    rate: float,
    div_yield: np.ndarray,
    vol: np.ndarray,
    corr: np.ndarray,
    hedge_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean = (rate - div_yield - 0.5 * vol**2) * hedge_dt
    covariance = np.diag(vol) @ corr @ np.diag(vol) * hedge_dt
    return mean, covariance


def filtered_stress_probabilities(
    true_paths: np.ndarray,
    rate: float,
    div_yield: np.ndarray,
    vol: np.ndarray,
    corr_calm: np.ndarray,
    corr_stress: np.ndarray,
    transition_hedge: np.ndarray,
    initial_stress_probability: float,
    hedge_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_paths, n_time_points, n_assets = true_paths.shape
    hedge_steps = n_time_points - 1
    log_returns = np.log(true_paths[:, 1:, :] / true_paths[:, :-1, :])

    mean_calm, cov_calm = state_log_return_moments(
        rate, div_yield, vol, corr_calm, hedge_dt
    )
    mean_stress, cov_stress = state_log_return_moments(
        rate, div_yield, vol, corr_stress, hedge_dt
    )

    priors = np.empty((n_paths, hedge_steps), dtype=float)
    posteriors = np.empty((n_paths, hedge_steps), dtype=float)

    current_prior = np.tile(
        np.array([1.0 - initial_stress_probability, initial_stress_probability], dtype=float),
        (n_paths, 1),
    )

    eps = 1e-12

    for step in range(hedge_steps):
        priors[:, step] = current_prior[:, 1]

        log_like_calm = gaussian_logpdf_rows(log_returns[:, step, :], mean_calm, cov_calm)
        log_like_stress = gaussian_logpdf_rows(
            log_returns[:, step, :], mean_stress, cov_stress
        )

        log_post_calm = np.log(np.clip(current_prior[:, 0], eps, 1.0)) + log_like_calm
        log_post_stress = np.log(np.clip(current_prior[:, 1], eps, 1.0)) + log_like_stress

        max_log = np.maximum(log_post_calm, log_post_stress)
        post_calm = np.exp(log_post_calm - max_log)
        post_stress = np.exp(log_post_stress - max_log)
        normalizer = post_calm + post_stress

        posterior = np.column_stack((post_calm / normalizer, post_stress / normalizer))
        posteriors[:, step] = posterior[:, 1]

        if step < hedge_steps - 1:
            current_prior = posterior @ transition_hedge

    return priors, posteriors


def filtered_model_price_and_delta(
    stress_probability: float,
    spot: np.ndarray,
    steps_remaining: int,
    pricing_inputs: dict[str, Any],
    n_samples: int,
    bump_fraction: float,
    base_seed: int,
) -> tuple[float, np.ndarray]:
    stress_probability = float(np.clip(stress_probability, 0.0, 1.0))
    if stress_probability <= 0.0:
        return regime_model_price_and_delta(
            spot=spot,
            steps_remaining=steps_remaining,
            current_regime=0,
            weights=pricing_inputs["weights"],
            strike=pricing_inputs["strike"],
            rate=pricing_inputs["rate"],
            div_yield=pricing_inputs["div_yield"],
            vol=pricing_inputs["vol"],
            hedge_dt=pricing_inputs["hedge_dt"],
            chol_calm=pricing_inputs["chol_calm"],
            chol_stress=pricing_inputs["chol_stress"],
            transition_hedge=pricing_inputs["transition_hedge"],
            n_samples=n_samples,
            bump_fraction=bump_fraction,
            rng=np.random.default_rng(base_seed + 1),
        )
    if stress_probability >= 1.0:
        return regime_model_price_and_delta(
            spot=spot,
            steps_remaining=steps_remaining,
            current_regime=1,
            weights=pricing_inputs["weights"],
            strike=pricing_inputs["strike"],
            rate=pricing_inputs["rate"],
            div_yield=pricing_inputs["div_yield"],
            vol=pricing_inputs["vol"],
            hedge_dt=pricing_inputs["hedge_dt"],
            chol_calm=pricing_inputs["chol_calm"],
            chol_stress=pricing_inputs["chol_stress"],
            transition_hedge=pricing_inputs["transition_hedge"],
            n_samples=n_samples,
            bump_fraction=bump_fraction,
            rng=np.random.default_rng(base_seed + 2),
        )

    price_calm, delta_calm = regime_model_price_and_delta(
        spot=spot,
        steps_remaining=steps_remaining,
        current_regime=0,
        weights=pricing_inputs["weights"],
        strike=pricing_inputs["strike"],
        rate=pricing_inputs["rate"],
        div_yield=pricing_inputs["div_yield"],
        vol=pricing_inputs["vol"],
        hedge_dt=pricing_inputs["hedge_dt"],
        chol_calm=pricing_inputs["chol_calm"],
        chol_stress=pricing_inputs["chol_stress"],
        transition_hedge=pricing_inputs["transition_hedge"],
        n_samples=n_samples,
        bump_fraction=bump_fraction,
        rng=np.random.default_rng(base_seed + 3),
    )
    price_stress, delta_stress = regime_model_price_and_delta(
        spot=spot,
        steps_remaining=steps_remaining,
        current_regime=1,
        weights=pricing_inputs["weights"],
        strike=pricing_inputs["strike"],
        rate=pricing_inputs["rate"],
        div_yield=pricing_inputs["div_yield"],
        vol=pricing_inputs["vol"],
        hedge_dt=pricing_inputs["hedge_dt"],
        chol_calm=pricing_inputs["chol_calm"],
        chol_stress=pricing_inputs["chol_stress"],
        transition_hedge=pricing_inputs["transition_hedge"],
        n_samples=n_samples,
        bump_fraction=bump_fraction,
        rng=np.random.default_rng(base_seed + 4),
    )

    calm_weight = 1.0 - stress_probability
    return (
        calm_weight * price_calm + stress_probability * price_stress,
        calm_weight * delta_calm + stress_probability * delta_stress,
    )


def evaluate_filtered_hedger(
    true_paths: np.ndarray,
    filtered_stress_priors: np.ndarray,
    pricing_inputs: dict[str, Any],
    base_seed: int,
    funding_price: float | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    time_start = time.perf_counter()
    initial_probability = float(filtered_stress_priors[0, 0])
    model_price, initial_delta = filtered_model_price_and_delta(
        stress_probability=initial_probability,
        spot=pricing_inputs["spot"],
        steps_remaining=pricing_inputs["hedge_steps"],
        pricing_inputs=pricing_inputs,
        n_samples=pricing_inputs["initial_price_mc_paths"],
        bump_fraction=pricing_inputs["bump_fraction"],
        base_seed=base_seed,
    )
    effective_funding_price = model_price if funding_price is None else funding_price

    terminal_payoff = basket_call_payoff(
        true_paths[:, -1, :], pricing_inputs["weights"], pricing_inputs["strike"]
    )
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
            _, state_delta = filtered_model_price_and_delta(
                stress_probability=float(filtered_stress_priors[path_idx, step + 1]),
                spot=next_spots[path_idx],
                steps_remaining=remaining_steps,
                pricing_inputs=pricing_inputs,
                n_samples=pricing_inputs["delta_mc_paths"],
                bump_fraction=pricing_inputs["bump_fraction"],
                base_seed=base_seed + 10000 * (step + 1) + path_idx,
            )
            new_deltas[path_idx] = state_delta

        cash_account = portfolio_before - np.sum(new_deltas * next_spots, axis=1)
        delta_holdings = new_deltas

    cash_account *= np.exp(pricing_inputs["rate"] * pricing_inputs["hedge_dt"])
    pnl = (
        cash_account
        + np.sum(delta_holdings * true_paths[:, -1, :], axis=1)
        - terminal_payoff
    )

    summary = {
        "runtime_seconds": float(time.perf_counter() - time_start),
        "model_initial_price": float(model_price),
        "funding_initial_price": float(effective_funding_price),
        "premium_adjustment": float(model_price - effective_funding_price),
        **pnl_summary(pnl),
    }
    return summary, pnl, initial_delta


def filter_diagnostics(
    filtered_stress_priors: np.ndarray, true_regimes: np.ndarray
) -> dict[str, float]:
    prior = filtered_stress_priors.reshape(-1)
    truth = true_regimes.reshape(-1).astype(float)
    hard_pred = (prior >= 0.5).astype(float)
    return {
        "mean_prior_stress": float(prior.mean()),
        "mean_prior_given_true_stress": float(prior[truth == 1.0].mean()),
        "mean_prior_given_true_calm": float(prior[truth == 0.0].mean()),
        "hard_classification_accuracy": float((hard_pred == truth).mean()),
        "brier_score": float(np.mean((prior - truth) ** 2)),
    }


def scenario_results(
    true_paths: np.ndarray,
    true_regimes: np.ndarray,
    filtered_priors: np.ndarray,
    pricing_inputs: dict[str, Any],
    base_seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    oracle_initial = regime_model_price_and_delta(
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
        rng=np.random.default_rng(base_seed + 1),
    )
    filtered_initial = filtered_model_price_and_delta(
        stress_probability=float(filtered_priors[0, 0]),
        spot=pricing_inputs["spot"],
        steps_remaining=pricing_inputs["hedge_steps"],
        pricing_inputs=pricing_inputs,
        n_samples=pricing_inputs["initial_price_mc_paths"],
        bump_fraction=pricing_inputs["bump_fraction"],
        base_seed=base_seed + 2,
    )
    constant_initial = constant_model_price_and_delta(
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
        rng=np.random.default_rng(base_seed + 3),
    )

    funding_price = float(oracle_initial[0])
    unhedged_pnl = unhedged_short_pnl(
        terminal_payoff=np.maximum(
            true_paths[:, -1, :] @ pricing_inputs["weights"] - pricing_inputs["strike"], 0.0
        ),
        rate=pricing_inputs["rate"],
        maturity=pricing_inputs["maturity"],
        funding_price=funding_price,
    )

    constant_summary, constant_pnl, constant_delta = evaluate_hedger(
        true_paths=true_paths,
        true_regimes=true_regimes,
        hedge_model="constant",
        pricing_inputs=pricing_inputs,
        base_seed=base_seed + 10,
        funding_price=funding_price,
        precomputed_initial=constant_initial,
    )
    oracle_summary, oracle_pnl, oracle_delta = evaluate_hedger(
        true_paths=true_paths,
        true_regimes=true_regimes,
        hedge_model="regime",
        pricing_inputs=pricing_inputs,
        base_seed=base_seed + 20,
        funding_price=funding_price,
        oracle_regime=True,
        precomputed_initial=oracle_initial,
    )
    filtered_summary, filtered_pnl, filtered_delta = evaluate_filtered_hedger(
        true_paths=true_paths,
        filtered_stress_priors=filtered_priors,
        pricing_inputs=pricing_inputs,
        base_seed=base_seed + 30,
        funding_price=funding_price,
    )

    results = {
        "Unhedged short option": {
            "runtime_seconds": 0.0,
            "model_initial_price": funding_price,
            "funding_initial_price": funding_price,
            "premium_adjustment": 0.0,
            **pnl_summary(unhedged_pnl),
        },
        "Constant-correlation hedge": constant_summary,
        "Oracle regime hedge": oracle_summary,
        "Filtered regime hedge": filtered_summary,
    }

    pnls = {
        "Unhedged short option": unhedged_pnl,
        "Constant-correlation hedge": constant_pnl,
        "Oracle regime hedge": oracle_pnl,
        "Filtered regime hedge": filtered_pnl,
    }
    deltas = {
        "Constant-correlation hedge": constant_delta,
        "Oracle regime hedge": oracle_delta,
        "Filtered regime hedge": filtered_delta,
    }
    return results, pnls, deltas
