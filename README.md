# Basket Option Pricing and Hedging Under Correlation Model Risk

This repo studies how correlation assumptions affect the pricing and hedging of a European basket call in a multi-asset Black-Scholes framework.

The project has two parts:

- a cleaned synthetic analysis pipeline with four notebooks
- a latent-regime follow-up where the true correlation regime is hidden from the hedger and must be filtered from returns

## Main Conclusion

The synthetic results support a cautious version of the original thesis:

- pricing differences between constant and regime-switching correlation are usually modest
- hedging matters far more than the time-0 price difference
- regime switching can improve hedge quality under harsher stress settings, but the effect is moderate and not universal once comparisons are put on a common premium basis

The latent-regime follow-up sharpens the implementation question:

- the oracle regime hedge is an upper benchmark because it knows the hidden state
- the filtered hedge only sees past returns and updates a stress probability
- in the stronger latent-stress case, the filtered hedge preserves a meaningful share of the oracle downside-tail improvement

## Repository Structure

### Synthetic Analysis

The main synthetic workflow is:

1. [01_setup_and_pricing.ipynb](notebooks/01_setup_and_pricing.ipynb)
   - builds the baseline synthetic market
   - simulates constant and regime-switching paths
   - prices the basket option
   - reports convergence and correlation diagnostics

2. [02_delta_hedging_backtest.ipynb](notebooks/02_delta_hedging_backtest.ipynb)
   - runs the baseline hedging experiment from scratch
   - compares unhedged, constant-correlation hedge, and regime-switching hedge
   - uses a common premium basis for clean constant-vs-regime comparison

3. [03_stress_sensitivity.ipynb](notebooks/03_stress_sensitivity.ipynb)
   - stress benchmark matrix
   - structural stress scenarios
   - sensitivity to harsher dependence environments

4. [04_dimension_and_robustness.ipynb](notebooks/04_dimension_and_robustness.ipynb)
   - homogeneous dimension scaling
   - fixed-`N=10` robustness analysis
   - parameter sweeps for stress severity, stress share, moneyness, and related inputs

Synthetic outputs are written to [data/synthetic_consolidated](data/synthetic_consolidated).

### Latent-Regime Follow-Up

The latent-state extension is under [latent_regime_followup](latent_regime_followup):

1. [01_filtered_regime_hedging.ipynb](latent_regime_followup/notebooks/01_filtered_regime_hedging.ipynb)
   - keeps the true world regime-switching
   - treats the regime as hidden from the hedger
   - filters a stress probability from observed multivariate returns
   - compares constant-correlation, oracle regime, and filtered regime hedging
   - averages the headline results over repeated true-world simulations

Latent-regime outputs are written to [latent_regime_followup/data](latent_regime_followup/data).

## Reproducibility

The repo is now set up so the notebooks can run from either:

- a normal `git clone`
- an extracted GitHub ZIP archive

To reproduce locally:

1. Use Python `3.10+`.
2. Install dependencies from [requirements.txt](requirements.txt).
3. Open the notebooks from inside the extracted repository folder.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Notes:

- The synthetic notebooks are fully self-contained once dependencies are installed.
- The latent-regime follow-up is also fully synthetic and does not require external data.

## Key Files

- [synthetic_analysis_utils.py](scripts/synthetic_analysis_utils.py): shared synthetic simulation and hedging helpers
- [latent_regime_utils.py](scripts/latent_regime_utils.py): hidden-state filtering and filtered-hedge helpers
- [baseline_pricing_summary.csv](data/synthetic_consolidated/baseline_pricing_summary.csv): baseline synthetic price comparison
- [baseline_hedging_summary_clean.csv](data/synthetic_consolidated/baseline_hedging_summary_clean.csv): baseline synthetic hedge summary on a common premium basis
- [dimension_homogeneous_gap_clean.csv](data/synthetic_consolidated/dimension_homogeneous_gap_clean.csv): clean homogeneous-`N` gap table
- [n10_parameter_gap_clean.csv](data/synthetic_consolidated/n10_parameter_gap_clean.csv): clean `N=10` robustness gap table
- [latent_filtered_hedging_summary.csv](latent_regime_followup/data/latent_filtered_hedging_summary.csv): averaged filtered-hedge comparison
- [latent_filtered_filter_diagnostics.csv](latent_regime_followup/data/latent_filtered_filter_diagnostics.csv): latent-state filter diagnostics
- [latent_filtered_recovery.csv](latent_regime_followup/data/latent_filtered_recovery.csv): fraction of oracle tail improvement recovered by filtering

## Notes

- The regime hedge in the synthetic notebooks is an oracle benchmark: it knows the current simulated regime at rebalance.
- The cleaned synthetic analysis emphasizes dispersion and 5% tails more than 1% tails, because the latter are noisier with moderate Monte Carlo sample sizes.
- The latent-regime notebook is the natural follow-up to that oracle benchmark: it asks how much of the oracle advantage survives when the regime is hidden and must be inferred.
- The repo intentionally keeps one synthetic narrative and one synthetic latent-state follow-up. Older exploratory branches and the dropped semiconductor extension were removed from the tracked repo.
