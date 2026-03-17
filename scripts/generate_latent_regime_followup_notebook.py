from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path("/Users/hamzaahmed/Multi-Asset Option Pricing")
NOTEBOOK_PATH = (
    ROOT
    / "latent_regime_followup"
    / "notebooks"
    / "01_filtered_regime_hedging.ipynb"
)


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


cells = [
    md_cell(
        """
        # Latent Regime Follow-Up - Filtered Hedging

        This notebook extends the synthetic hedging study by making the regime **latent**.

        The true world still switches between calm and stress correlation regimes, but the hedger no longer observes the true state directly. Instead, it only sees past multivariate returns and updates a posterior stress probability through a hidden Markov filter.

        We compare four positions:

        - unhedged short option
        - constant-correlation hedge
        - oracle regime hedge
        - filtered regime hedge

        The filtered hedge answers the practical question the original synthetic notebook could not:

        **How much of the oracle regime-hedge advantage survives when the state is hidden rather than observed?**
        """
    ),
    code_cell(
        """
        from pathlib import Path
        import sys

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import Markdown, display

        plt.style.use("seaborn-v0_8-whitegrid")
        pd.set_option("display.max_columns", 80)
        pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


        def find_project_root() -> Path:
            current = Path.cwd().resolve()
            for candidate in [current, *current.parents]:
                if (
                    (candidate / "README.md").exists()
                    and (candidate / "notebooks").is_dir()
                    and (candidate / "scripts").is_dir()
                ):
                    return candidate
            raise FileNotFoundError("Could not locate the project root. Run from inside the repository.")


        PROJECT_ROOT = find_project_root()
        FOLLOWUP_DIR = PROJECT_ROOT / "latent_regime_followup"
        DATA_DIR = FOLLOWUP_DIR / "data"
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

        from synthetic_analysis_utils import (
            average_off_diagonal,
            build_transition_matrix,
            equicorrelation_matrix,
            simulate_regime_switching_paths,
            summary_frame_from_results,
        )
        from latent_regime_utils import (
            filter_diagnostics,
            filtered_stress_probabilities,
            scenario_results,
        )
        """
    ),
    code_cell(
        """
        spot = np.array([100.0, 95.0, 110.0])
        weights = np.array([0.40, 0.35, 0.25])
        vol = np.array([0.20, 0.25, 0.22])
        div_yield = np.zeros_like(spot)

        rate = 0.03
        maturity = 1.0
        strike = float(weights @ spot)
        bump_fraction = 0.01
        n_true_world_paths = 220
        delta_mc_paths = 1500
        initial_price_mc_paths = 15000
        n_repeats = 3
        start_regime = 0

        scenario_specs = [
            {
                "scenario": "Baseline",
                "description": "Original synthetic benchmark",
                "rho_constant": 0.35,
                "rho_calm": 0.20,
                "rho_stress": 0.75,
                "p01_daily": 0.03,
                "p10_daily": 0.12,
                "hedge_steps": 12,
            },
            {
                "scenario": "Latent stress",
                "description": "Monthly high-stress benchmark with 50% stress share and symmetric switching",
                "rho_constant": 0.35,
                "rho_calm": 0.20,
                "rho_stress": 0.98,
                "p01_daily": 0.06,
                "p10_daily": 0.06,
                "hedge_steps": 12,
            },
        ]

        scenario_config = pd.DataFrame(scenario_specs)
        scenario_config["stress_share_stationary"] = (
            scenario_config["p01_daily"] / (scenario_config["p01_daily"] + scenario_config["p10_daily"])
        )
        display(Markdown("## Scenario setup"))
        display(Markdown(f"Averaging over **{n_repeats}** independent true-world repeats per scenario."))
        display(scenario_config)
        """
    ),
    code_cell(
        """
        raw_summary_frames = []
        raw_filter_rows = []
        recovery_rows = []
        pathwise_rows = []
        sample_path_rows = []

        for idx, spec in enumerate(scenario_specs):
            hedge_steps = int(spec["hedge_steps"])
            hedge_dt = maturity / hedge_steps

            corr_constant = equicorrelation_matrix(len(spot), spec["rho_constant"])
            corr_calm = equicorrelation_matrix(len(spot), spec["rho_calm"])
            corr_stress = equicorrelation_matrix(len(spot), spec["rho_stress"])

            transition_daily = build_transition_matrix(spec["p01_daily"], spec["p10_daily"])
            transition_hedge = np.linalg.matrix_power(transition_daily, 252 // hedge_steps)
            pricing_inputs = {
                "spot": spot,
                "weights": weights,
                "strike": strike,
                "rate": rate,
                "div_yield": div_yield,
                "vol": vol,
                "maturity": maturity,
                "hedge_steps": hedge_steps,
                "hedge_dt": hedge_dt,
                "delta_mc_paths": delta_mc_paths,
                "initial_price_mc_paths": initial_price_mc_paths,
                "bump_fraction": bump_fraction,
                "chol_constant": np.linalg.cholesky(corr_constant),
                "chol_calm": np.linalg.cholesky(corr_calm),
                "chol_stress": np.linalg.cholesky(corr_stress),
                "transition_hedge": transition_hedge,
                "start_regime": start_regime,
            }

            for repeat in range(n_repeats):
                true_paths, true_regimes = simulate_regime_switching_paths(
                    spot=spot,
                    rate=rate,
                    div_yield=div_yield,
                    vol=vol,
                    maturity=maturity,
                    n_steps=hedge_steps,
                    n_paths=n_true_world_paths,
                    corr_calm=corr_calm,
                    corr_stress=corr_stress,
                    transition_matrix=transition_hedge,
                    start_regime=start_regime,
                    seed=2026032100 + 1000 * idx + repeat,
                )

                filtered_priors, filtered_posteriors = filtered_stress_probabilities(
                    true_paths=true_paths,
                    rate=rate,
                    div_yield=div_yield,
                    vol=vol,
                    corr_calm=corr_calm,
                    corr_stress=corr_stress,
                    transition_hedge=transition_hedge,
                    initial_stress_probability=float(start_regime),
                    hedge_dt=hedge_dt,
                )

                results, pnls, deltas = scenario_results(
                    true_paths=true_paths,
                    true_regimes=true_regimes,
                    filtered_priors=filtered_priors,
                    pricing_inputs=pricing_inputs,
                    base_seed=2026032500 + 1000 * idx + repeat,
                )

                frame = summary_frame_from_results(results)
                frame["scenario"] = spec["scenario"]
                frame["description"] = spec["description"]
                frame["repeat"] = repeat
                frame["avg_stress_fraction"] = float(true_regimes.mean())
                frame["avg_prior_stress_probability"] = float(filtered_priors.mean())
                frame["avg_posterior_stress_probability"] = float(filtered_posteriors.mean())
                raw_summary_frames.append(frame)

                diag = filter_diagnostics(filtered_priors, true_regimes)
                diag.update(
                    {
                        "scenario": spec["scenario"],
                        "description": spec["description"],
                        "repeat": repeat,
                        "rho_constant": spec["rho_constant"],
                        "rho_calm": spec["rho_calm"],
                        "rho_stress": spec["rho_stress"],
                        "avg_offdiag_constant": average_off_diagonal(corr_constant),
                        "avg_offdiag_calm": average_off_diagonal(corr_calm),
                        "avg_offdiag_stress": average_off_diagonal(corr_stress),
                    }
                )
                raw_filter_rows.append(diag)

                if repeat == 0:
                    pathwise = pd.DataFrame(
                        {
                            "scenario": spec["scenario"],
                            "repeat": repeat,
                            "path_id": np.arange(n_true_world_paths),
                            "stress_fraction": true_regimes.mean(axis=1),
                            "avg_prior_stress_probability": filtered_priors.mean(axis=1),
                            "avg_posterior_stress_probability": filtered_posteriors.mean(axis=1),
                            "Unhedged short option": pnls["Unhedged short option"],
                            "Constant-correlation hedge": pnls["Constant-correlation hedge"],
                            "Oracle regime hedge": pnls["Oracle regime hedge"],
                            "Filtered regime hedge": pnls["Filtered regime hedge"],
                        }
                    )
                    pathwise_rows.append(pathwise)

                    sample_ids = (
                        pd.Series(true_regimes.mean(axis=1))
                        .sort_values(ascending=False)
                        .head(3)
                        .index
                        .tolist()
                    )
                    for path_id in sample_ids:
                        for step in range(hedge_steps):
                            sample_path_rows.append(
                                {
                                    "scenario": spec["scenario"],
                                    "repeat": repeat,
                                    "path_id": int(path_id),
                                    "step": step,
                                    "true_regime": int(true_regimes[path_id, step]),
                                    "prior_stress_probability": float(filtered_priors[path_id, step]),
                                    "posterior_stress_probability": float(filtered_posteriors[path_id, step]),
                                }
                            )

        summary_raw_table = pd.concat(raw_summary_frames, ignore_index=True)
        summary_table = (
            summary_raw_table.drop(columns=["repeat"])
            .groupby(["scenario", "description", "strategy"], as_index=False)
            .mean(numeric_only=True)
        )

        filter_diagnostics_raw_table = pd.DataFrame(raw_filter_rows)
        filter_diagnostics_table = (
            filter_diagnostics_raw_table.drop(columns=["repeat"])
            .groupby(
                [
                    "scenario",
                    "description",
                    "rho_constant",
                    "rho_calm",
                    "rho_stress",
                    "avg_offdiag_constant",
                    "avg_offdiag_calm",
                    "avg_offdiag_stress",
                ],
                as_index=False,
            )
            .mean(numeric_only=True)
        )

        for spec in scenario_specs:
            scenario_slice = summary_table.loc[summary_table["scenario"] == spec["scenario"]].set_index("strategy")
            std_const = scenario_slice.loc["Constant-correlation hedge", "std_pnl"]
            std_oracle = scenario_slice.loc["Oracle regime hedge", "std_pnl"]
            std_filtered = scenario_slice.loc["Filtered regime hedge", "std_pnl"]
            q05_const = scenario_slice.loc["Constant-correlation hedge", "q05_pnl"]
            q05_oracle = scenario_slice.loc["Oracle regime hedge", "q05_pnl"]
            q05_filtered = scenario_slice.loc["Filtered regime hedge", "q05_pnl"]
            q01_const = scenario_slice.loc["Constant-correlation hedge", "q01_pnl"]
            q01_oracle = scenario_slice.loc["Oracle regime hedge", "q01_pnl"]
            q01_filtered = scenario_slice.loc["Filtered regime hedge", "q01_pnl"]

            recovery_rows.append(
                {
                    "scenario": spec["scenario"],
                    "std_advantage_oracle_vs_constant": std_const - std_oracle,
                    "std_advantage_filtered_vs_constant": std_const - std_filtered,
                    "std_recovery_fraction": (
                        (std_const - std_filtered) / (std_const - std_oracle)
                        if (std_const - std_oracle) > 1e-10
                        else np.nan
                    ),
                    "q05_advantage_oracle_vs_constant": q05_oracle - q05_const,
                    "q05_advantage_filtered_vs_constant": q05_filtered - q05_const,
                    "q05_recovery_fraction": (
                        (q05_filtered - q05_const) / (q05_oracle - q05_const)
                        if (q05_oracle - q05_const) > 1e-10
                        else np.nan
                    ),
                    "q01_advantage_oracle_vs_constant": q01_oracle - q01_const,
                    "q01_advantage_filtered_vs_constant": q01_filtered - q01_const,
                    "q01_recovery_fraction": (
                        (q01_filtered - q01_const) / (q01_oracle - q01_const)
                        if (q01_oracle - q01_const) > 1e-10
                        else np.nan
                    ),
                }
            )

        recovery_table = pd.DataFrame(recovery_rows)
        pathwise_table = pd.concat(pathwise_rows, ignore_index=True)
        sample_paths_table = pd.DataFrame(sample_path_rows)

        display(Markdown("## Hedging summary"))
        display(
            summary_table[
                [
                    "scenario",
                    "strategy",
                    "model_initial_price",
                    "funding_initial_price",
                    "std_pnl",
                    "q05_pnl",
                    "q01_pnl",
                    "avg_stress_fraction",
                ]
            ]
        )

        display(Markdown("## Filter diagnostics"))
        display(filter_diagnostics_table)

        display(Markdown("## Recovery of oracle advantage"))
        display(recovery_table)
        """
    ),
    code_cell(
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=False)

        sns.barplot(
            data=summary_table.loc[summary_table["strategy"] != "Unhedged short option"],
            x="scenario",
            y="std_pnl",
            hue="strategy",
            ax=axes[0],
        )
        axes[0].set_title("Hedged P&L dispersion")
        axes[0].set_ylabel("Std of terminal P&L")

        sns.barplot(
            data=summary_table.loc[summary_table["strategy"] != "Unhedged short option"],
            x="scenario",
            y="q05_pnl",
            hue="strategy",
            ax=axes[1],
        )
        axes[1].set_title("Hedged 5% tail")
        axes[1].set_ylabel("5% quantile of terminal P&L")

        sns.barplot(
            data=summary_table.loc[summary_table["strategy"] != "Unhedged short option"],
            x="scenario",
            y="q01_pnl",
            hue="strategy",
            ax=axes[2],
        )
        axes[2].set_title("Hedged 1% tail")
        axes[2].set_ylabel("1% quantile of terminal P&L")

        for ax in axes:
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axes[2].legend(loc="best")
        plt.tight_layout()
        plt.show()
        """
    ),
    code_cell(
        """
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)

        for axis, scenario_name in zip(axes, [spec["scenario"] for spec in scenario_specs]):
            subset = sample_paths_table.loc[sample_paths_table["scenario"] == scenario_name]
            for path_id in subset["path_id"].unique():
                path_df = subset.loc[subset["path_id"] == path_id]
                axis.plot(
                    path_df["step"],
                    path_df["prior_stress_probability"],
                    marker="o",
                    linewidth=1.6,
                    label=f"path {path_id} prior",
                )
                axis.step(
                    path_df["step"],
                    path_df["true_regime"],
                    where="mid",
                    linestyle="--",
                    linewidth=1.0,
                    color="black",
                    alpha=0.5,
                )
            axis.set_title(f"{scenario_name}: filtered prior vs true regime")
            axis.set_xlabel("Hedge interval")
            axis.set_ylabel("Stress probability")
            axis.set_ylim(-0.05, 1.05)

        axes[0].legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.show()
        """
    ),
    code_cell(
        """
        summary_raw_path = DATA_DIR / "latent_filtered_hedging_raw.csv"
        summary_path = DATA_DIR / "latent_filtered_hedging_summary.csv"
        diagnostics_raw_path = DATA_DIR / "latent_filtered_filter_diagnostics_raw.csv"
        diagnostics_path = DATA_DIR / "latent_filtered_filter_diagnostics.csv"
        recovery_path = DATA_DIR / "latent_filtered_recovery.csv"
        pathwise_path = DATA_DIR / "latent_filtered_pathwise_pnl.csv"
        sample_paths_path = DATA_DIR / "latent_filtered_sample_paths.csv"

        summary_raw_table.to_csv(summary_raw_path, index=False)
        summary_table.to_csv(summary_path, index=False)
        filter_diagnostics_raw_table.to_csv(diagnostics_raw_path, index=False)
        filter_diagnostics_table.to_csv(diagnostics_path, index=False)
        recovery_table.to_csv(recovery_path, index=False)
        pathwise_table.to_csv(pathwise_path, index=False)
        sample_paths_table.to_csv(sample_paths_path, index=False)

        print(f"Saved raw hedging summary to {summary_raw_path}")
        print(f"Saved hedging summary to {summary_path}")
        print(f"Saved raw filter diagnostics to {diagnostics_raw_path}")
        print(f"Saved filter diagnostics to {diagnostics_path}")
        print(f"Saved oracle-recovery table to {recovery_path}")
        print(f"Saved pathwise P&L to {pathwise_path}")
        print(f"Saved sample path probabilities to {sample_paths_path}")
        """
    ),
    md_cell(
        """
        ## Reading the result

        The key interpretation is:

        - `Constant-correlation hedge`: ignores the regime entirely
        - `Oracle regime hedge`: knows the true hidden state
        - `Filtered regime hedge`: only sees past returns and updates a posterior stress probability

        The filtered hedge is the realistic version of the regime model. If it lands between constant and oracle, then the project gets a clean decomposition:

        - `constant -> oracle`: total value of the regime model
        - `filtered -> oracle`: cost of latent-state uncertainty
        - `constant -> filtered`: implementable gain from filtering the hidden state

        In the stronger latent-stress scenario, the main question is whether the filtered hedge preserves most of the oracle improvement once the true state is hidden.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {NOTEBOOK_PATH}")
