# Basket Option Pricing and Hedging Under Correlation Model Risk

This project studies how correlation assumptions affect the pricing and hedging of a European basket option in a multi-asset Black-Scholes setting.

The core idea is to compare:

- a baseline model with constant correlation,
- an extended model with regime-switching correlation,
- hedging performance when the true world has changing correlation but the hedger assumes a simpler constant-correlation model.

## Project Goal

The goal is to build a clean Monte Carlo framework for:

1. pricing a European basket call,
2. estimating multi-asset deltas,
3. running a discrete-time delta-hedging experiment,
4. quantifying correlation model risk in both price and hedged P&L.

## Instrument

We consider a basket of `N` equities with prices `S_1(t), ..., S_N(t)` and weights `w_1, ..., w_N`.

The basket value is:

```math
B(t) = \sum_{i=1}^N w_i S_i(t)
```

The main payoff is a European basket call:

```math
\left(B(T) - K\right)^+ = \max\left(\sum_{i=1}^N w_i S_i(T) - K, 0\right)
```

## Models

### 1. Constant-Correlation Multi-Asset Black-Scholes

Under the risk-neutral measure, each asset follows a geometric Brownian motion with constant volatility and correlated Brownian shocks. The basket option price is estimated by Monte Carlo simulation of the joint terminal distribution.

### 2. Regime-Switching Correlation Model

To capture stress behavior, the project introduces two correlation regimes:

- calm regime: lower average correlation,
- stress regime: higher average correlation.

A two-state Markov chain drives switching between the regimes. Conditional on the current regime, shocks are simulated using the corresponding correlation matrix.

## Main Tasks

- simulate correlated terminal asset prices under the constant-correlation model,
- price the basket option via Monte Carlo,
- report Monte Carlo error bars and convergence diagnostics,
- estimate multi-asset deltas using bump-and-revalue finite differences with common random numbers,
- simulate hedging paths under regime-switching correlation,
- evaluate hedging error when the hedger uses a misspecified constant-correlation model,
- summarize pricing sensitivity and hedging risk across model settings.

## Hedging Experiment

The key hedging setup is:

- true world: regime-switching correlation model,
- hedger model: constant-correlation model,
- strategy: discrete-time delta hedge using the underlying assets and a cash account.

The main output is the terminal hedged P&L distribution, with summary statistics such as:

- mean,
- standard deviation,
- tail quantiles,
- comparison of correctly specified vs misspecified hedging assumptions.

## Planned Deliverables

- a Jupyter notebook implementing pricing, delta estimation, and hedging simulation,
- figures showing price sensitivity to correlation and regime parameters,
- Monte Carlo convergence plots and confidence intervals,
- hedged P&L comparisons under correct and misspecified models,
- a short write-up discussing pricing impact, hedging implications, and practical limitations.

## Possible Second Stage

After validating the pipeline on synthetic data, the project may optionally use real equity return data to estimate:

- historical volatilities,
- historical correlations,
- calm vs stress correlation regimes.

This stage would be used for hypothetical basket option experiments rather than full market calibration.

## Expected Outcome

The expected result is that constant-correlation Black-Scholes provides a useful baseline, but correlation regime shifts can materially affect both option value and hedging error, especially in stress-like scenarios.

## Status

This repository is currently in the planning stage. The README is an initial summary and will be refined as the implementation develops.
