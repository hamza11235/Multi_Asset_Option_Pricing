# Basket Option Pricing and Hedging Under Correlation Model Risk

This repository studies how correlation misspecification affects the pricing and delta hedging of a European basket call in a multi-asset Black-Scholes setting.

The central question is whether a constant-correlation model remains adequate when the true world exhibits calm and stress correlation regimes, and how much of any regime-based hedging advantage survives once the regime is hidden from the hedger.

## Contract and State Variables

Let \(S_i(t)\) denote the price of asset \(i\), and let the basket be

$$
B_t = \sum_{i=1}^N w_i S_i(t).
$$

The derivative is a European basket call with payoff

$$
g(S_T) = (B_T - K)^+.
$$

Throughout the synthetic analysis, the strike is usually set at-the-money so that \(K = B_0\).

## Correlation Models

### Constant-Correlation Benchmark

Under the risk-neutral measure,

$$
\frac{dS_i(t)}{S_i(t)} = (r - q_i)\,dt + \sigma_i\,dW_i(t),
$$

with

$$
dW_i(t)\,dW_j(t) = \rho_{ij}\,dt.
$$

The constant-correlation model uses a single correlation matrix \(\rho\) for the full horizon.

### Regime-Switching Correlation Model

The regime model keeps the same marginals but lets correlation depend on a two-state Markov chain \(X_t \in \{0,1\}\):

$$
dW_i(t)\,dW_j(t) = \rho_{ij}^{(X_t)}\,dt,
$$

where \(X_t = 0\) denotes the calm state and \(X_t = 1\) denotes the stress state. The transition law is

$$
P(X_{t+\Delta}=1 \mid X_t=0)=p_{01},
\qquad
P(X_{t+\Delta}=0 \mid X_t=1)=p_{10}.
$$

The synthetic experiments specify \(p_{01}\), \(p_{10}\), \(\rho^{(0)}\), and \(\rho^{(1)}\) directly, which allows the true dependence structure to be controlled cleanly.

## Pricing and Hedging

The time-0 option value is computed by Monte Carlo under the risk-neutral measure:

$$
V_0 = \mathbb{E}^{\mathbb{Q}}\left[e^{-rT}(B_T-K)^+\right].
$$

For the hedging experiments, the seller shorts one option and dynamically trades the underlying assets. Terminal P&L is

$$
\Pi_T = C_T + \sum_{i=1}^N \Delta_i(T^-) S_i(T) - (B_T-K)^+,
$$

where \(C_T\) is the terminal cash account and \(\Delta_i\) denotes the stock holding in asset \(i\). Deltas are estimated by bump-and-revalue Monte Carlo.

The synthetic notebooks compare:

- an unhedged short option
- a constant-correlation hedge
- an oracle regime hedge that knows the true simulated state

All constant-versus-regime hedge comparisons are reported on a common premium basis so that the comparison isolates hedge quality rather than initial price differences.

## Latent-State Extension

The follow-up notebook makes the regime unobserved to the hedger. The true world still evolves under calm/stress correlation regimes, but the hedger only sees past multivariate returns and updates a posterior stress probability

$$
\pi_t = P(X_t = 1 \mid \mathcal{F}^{\text{returns}}_t).
$$

This produces a third hedge:

- constant-correlation hedge
- oracle regime hedge
- filtered regime hedge

The filtered hedge uses posterior-weighted regime deltas instead of the true hidden state. This turns the oracle benchmark into a realistic signal-extraction problem.

## Main Findings

The final outputs support the following conclusions.

- Pricing differences between constant and regime-switching correlation are usually modest in the baseline synthetic setting.
- Hedging is the economically important margin: both dynamic hedges reduce P&L dispersion sharply relative to an unhedged short option.
- In the cleaned synthetic baseline, the oracle regime hedge is not uniformly better than the constant hedge on every tail metric; the baseline comparison is mixed once both are put on the same premium basis.
- Under harsher synthetic stress scenarios, regime-aware hedging becomes more valuable, especially in downside-tail metrics.
- In the latent-state follow-up, the filtered hedge does not match the oracle benchmark, but it still preserves a meaningful share of the oracle downside-tail improvement in the stronger stress case.

Representative output tables:

- [baseline_pricing_summary.csv](outputs/synthetic/baseline_pricing_summary.csv)
- [baseline_hedging_summary_clean.csv](outputs/synthetic/baseline_hedging_summary_clean.csv)
- [structural_stress_gap_clean.csv](outputs/synthetic/structural_stress_gap_clean.csv)
- [dimension_homogeneous_gap_clean.csv](outputs/synthetic/dimension_homogeneous_gap_clean.csv)
- [n10_parameter_gap_clean.csv](outputs/synthetic/n10_parameter_gap_clean.csv)
- [latent_filtered_hedging_summary.csv](outputs/latent_regime/latent_filtered_hedging_summary.csv)
- [latent_filtered_filter_diagnostics.csv](outputs/latent_regime/latent_filtered_filter_diagnostics.csv)
- [latent_filtered_recovery.csv](outputs/latent_regime/latent_filtered_recovery.csv)

## Repository Structure

The tracked repository is organized into three top-level folders:

- [scripts](scripts): reusable Python utilities for simulation, pricing, hedging, filtering, and plot generation
- [notebooks](notebooks): the main analysis sequence
- [outputs](outputs): saved CSV artifacts and exported figures

The notebook sequence is:

1. [01_setup_and_pricing.ipynb](notebooks/01_setup_and_pricing.ipynb)
2. [02_delta_hedging_backtest.ipynb](notebooks/02_delta_hedging_backtest.ipynb)
3. [03_stress_sensitivity.ipynb](notebooks/03_stress_sensitivity.ipynb)
4. [04_dimension_and_robustness.ipynb](notebooks/04_dimension_and_robustness.ipynb)
5. [05_filtered_regime_hedging.ipynb](notebooks/05_filtered_regime_hedging.ipynb)

Saved outputs are grouped into:

- [outputs/synthetic](outputs/synthetic)
- [outputs/latent_regime](outputs/latent_regime)
- [outputs/plots](outputs/plots)

## Reproducibility

Use Python 3.10+ and install dependencies from [requirements.txt](requirements.txt):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

The notebooks locate the repository root by checking for `README.md`, `notebooks`, and `scripts`, so they can run from either a normal clone or an extracted GitHub ZIP archive.
