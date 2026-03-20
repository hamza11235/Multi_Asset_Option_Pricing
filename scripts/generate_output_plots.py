"""Generate a compact set of final plot artifacts for the cleaned repo.

The notebooks remain the primary analysis medium, but these PNGs provide a
lightweight summary inside ``outputs/plots`` for quick inspection and for
including the final repo outputs in a single folder tree.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def find_project_root() -> Path:
    """Locate the repository root from the current working directory."""
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (
            (candidate / "README.md").exists()
            and (candidate / "notebooks").is_dir()
            and (candidate / "scripts").is_dir()
        ):
            return candidate
    raise FileNotFoundError("Could not locate the project root.")


def main() -> None:
    """Generate the tracked summary plots under ``outputs/plots``."""
    root = find_project_root()
    output_dir = root / "outputs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    synthetic_dir = root / "outputs" / "synthetic"
    latent_dir = root / "outputs" / "latent_regime"

    baseline = pd.read_csv(synthetic_dir / "baseline_hedging_summary_clean.csv")
    dimension = pd.read_csv(synthetic_dir / "dimension_homogeneous_gap_clean.csv")
    stress_share = pd.read_csv(synthetic_dir / "n10_parameter_gap_clean.csv")
    latent = pd.read_csv(latent_dir / "latent_filtered_hedging_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    baseline_plot = baseline[["strategy", "std_pnl", "q05_pnl", "q01_pnl"]].copy()
    short_labels = {
        "Unhedged short option (regime premium basis)": "Unhedged",
        "Constant-correlation hedge (regime premium basis)": "Constant",
        "Regime-switching hedge (oracle, regime premium basis)": "Oracle regime",
    }
    baseline_plot["strategy"] = baseline_plot["strategy"].map(short_labels)
    for axis, metric, title in zip(
        axes,
        ["std_pnl", "q05_pnl", "q01_pnl"],
        ["Std of P&L", "5% tail", "1% tail"],
    ):
        sns.barplot(
            data=baseline_plot,
            x="strategy",
            y=metric,
            hue="strategy",
            legend=False,
            ax=axis,
            palette="deep",
        )
        axis.set_title(f"Synthetic baseline: {title}")
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_baseline_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    for axis, metric, title, ylabel in zip(
        axes,
        [
            "regime_price_minus_constant",
            "std_gap_constant_minus_regime",
            "q05_gap_regime_minus_constant",
            "q01_gap_regime_minus_constant",
        ],
        [
            "Regime price premium vs N",
            "Dispersion gap vs N",
            "5% tail gap vs N",
            "1% tail gap vs N",
        ],
        [
            "Regime minus constant",
            "Constant minus regime",
            "Regime minus constant",
            "Regime minus constant",
        ],
    ):
        axis.plot(dimension["n_assets"], dimension[metric], marker="o", linewidth=2.0)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Number of assets")
        axis.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_dimension_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    stress_plot = stress_share.loc[stress_share["scenario_group"] == "stress_share"].copy()
    stress_plot["scenario_value"] = stress_plot["scenario_value"].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric, title, ylabel in zip(
        axes,
        ["regime_price_minus_constant", "q05_gap_regime_minus_constant", "q01_gap_regime_minus_constant"],
        ["Price premium vs stress share", "5% tail gap vs stress share", "1% tail gap vs stress share"],
        ["Regime minus constant", "Regime minus constant", "Regime minus constant"],
    ):
        axis.plot(stress_plot["scenario_value"], stress_plot[metric], marker="o", linewidth=2.0)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Stress share")
        axis.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_stress_share_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    latent_plot = latent.loc[latent["strategy"] != "Unhedged short option"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric, title in zip(
        axes,
        ["std_pnl", "q05_pnl", "q01_pnl"],
        ["Filtered follow-up: std", "Filtered follow-up: 5% tail", "Filtered follow-up: 1% tail"],
    ):
        sns.barplot(data=latent_plot, x="scenario", y=metric, hue="strategy", ax=axis, palette="deep")
        axis.set_title(title)
        axis.set_xlabel("")
        if axis is not axes[-1]:
            axis.get_legend().remove()
    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "latent_filtered_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
