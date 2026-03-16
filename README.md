# Basket Option Pricing and Hedging Under Correlation Model Risk

This repo studies how correlation assumptions affect the pricing and hedging of a European basket call in a multi-asset Black-Scholes framework.

The project has two parts:

- a cleaned synthetic analysis pipeline with four notebooks
- a real-data extension using a semiconductor basket and hypothetical options

## Main Conclusion

The synthetic results support a cautious version of the original thesis:

- pricing differences between constant and regime-switching correlation are usually modest
- hedging matters far more than the time-0 price difference
- regime switching can improve hedge quality under harsher stress settings, but the effect is moderate and not universal once comparisons are put on a common premium basis

The real-data semiconductor extension shows a similar theme:

- calm and stress correlations are empirically different
- but a constant-correlation approximation can still be fairly competitive for a diversified basket

## Repository Structure

### Synthetic Analysis

The main synthetic workflow is:

1. [01_setup_and_pricing.ipynb](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/notebooks/01_setup_and_pricing.ipynb)
   - builds the baseline synthetic market
   - simulates constant and regime-switching paths
   - prices the basket option
   - reports convergence and correlation diagnostics

2. [02_delta_hedging_backtest.ipynb](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/notebooks/02_delta_hedging_backtest.ipynb)
   - runs the baseline hedging experiment from scratch
   - compares unhedged, constant-correlation hedge, and regime-switching hedge
   - uses a common premium basis for clean constant-vs-regime comparison

3. [03_stress_sensitivity.ipynb](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/notebooks/03_stress_sensitivity.ipynb)
   - stress benchmark matrix
   - structural stress scenarios
   - sensitivity to harsher dependence environments

4. [04_dimension_and_robustness.ipynb](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/notebooks/04_dimension_and_robustness.ipynb)
   - homogeneous dimension scaling
   - fixed-`N=10` robustness analysis
   - parameter sweeps for stress severity, stress share, moneyness, and related inputs

Synthetic outputs are written to [data/synthetic_consolidated](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/data/synthetic_consolidated).

### Real-Data Extension

The real-data branch is under [real_simulation](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/real_simulation):

1. [01_real_data_setup_and_calibration.ipynb](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/real_simulation/notebooks/01_real_data_setup_and_calibration.ipynb)
   - downloads semiconductor stock data
   - estimates vols, constant correlation, calm/stress correlation matrices, and regime transitions

2. [02_real_data_basket_option_pricing.ipynb](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/real_simulation/notebooks/02_real_data_basket_option_pricing.ipynb)
   - prices a hypothetical basket call using the calibrated inputs

3. [03_real_data_delta_hedging.ipynb](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/real_simulation/notebooks/03_real_data_delta_hedging.ipynb)
   - runs the simulated hedging comparison on the calibrated semiconductor basket

Real-data outputs are written to [real_simulation/data](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/real_simulation/data).

## Key Files

- [synthetic_analysis_utils.py](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/scripts/synthetic_analysis_utils.py): shared synthetic simulation and hedging helpers
- [baseline_pricing_summary.csv](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/data/synthetic_consolidated/baseline_pricing_summary.csv): baseline synthetic price comparison
- [baseline_hedging_summary_clean.csv](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/data/synthetic_consolidated/baseline_hedging_summary_clean.csv): baseline synthetic hedge summary on a common premium basis
- [dimension_homogeneous_gap_clean.csv](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/data/synthetic_consolidated/dimension_homogeneous_gap_clean.csv): clean homogeneous-`N` gap table
- [n10_parameter_gap_clean.csv](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/data/synthetic_consolidated/n10_parameter_gap_clean.csv): clean `N=10` robustness gap table
- [calibration_summary.csv](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/real_simulation/data/calibration_summary.csv): real-data calibration summary
- [real_basket_hedging_clean_gap.csv](/Users/hamzaahmed/Multi-Asset%20Option%20Pricing/real_simulation/data/real_basket_hedging_clean_gap.csv): real-data hedge comparison on a clean basis

## Notes

- The regime hedge in the synthetic notebooks is an oracle benchmark: it knows the current simulated regime at rebalance.
- The cleaned synthetic analysis emphasizes dispersion and 5% tails more than 1% tails, because the latter are noisier with moderate Monte Carlo sample sizes.
- The repo intentionally keeps one synthetic narrative and one real-data extension. Older exploratory branches and duplicate notebook paths were removed from the tracked repo.
