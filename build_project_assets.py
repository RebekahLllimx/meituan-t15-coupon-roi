"""Build the committed figures used by the README and technical report."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

from project_pipeline import ProjectConfig, build_demo_population, run_project


OUTPUT = Path(__file__).resolve().parent / "docs" / "assets"


def build_architecture() -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(13, 5.4))
    figure.patch.set_facecolor("#f7f8fa")
    axis.set_facecolor("#f7f8fa")
    axis.set_xlim(0, 13)
    axis.set_ylim(0, 5.4)
    axis.axis("off")

    boxes = [
        (0.35, 2.0, 2.05, 1.35, "Source-schema\nspreadsheets", "Schema validation\nScene filtering", "#fff4e8"),
        (2.9, 2.0, 2.05, 1.35, "Feature engineering", "User state\nFunnel calibration", "#edf6ff"),
        (5.45, 2.0, 2.05, 1.35, "Decision simulator", "Coupon response\nBudget and reward", "#eefaf4"),
        (8.0, 2.0, 2.05, 1.35, "Shared LinUCB", "Context-action features\nPartial-feedback update", "#f1efff"),
        (10.55, 2.0, 2.05, 1.35, "Decision products", "Streamlit demo\nBenchmark and report", "#fff0f4"),
    ]
    for x, y, width, height, title, subtitle, color in boxes:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.2, edgecolor="#c9ced6", facecolor=color,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + 0.86, title, ha="center", va="center", fontsize=11, weight="bold", color="#20252b")
        axis.text(x + width / 2, y + 0.38, subtitle, ha="center", va="center", fontsize=8.7, color="#56606a", linespacing=1.35)
    for left in [2.4, 4.95, 7.5, 10.05]:
        axis.add_patch(FancyArrowPatch((left, 2.68), (left + 0.48, 2.68), arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color="#7a8490"))
    axis.text(6.5, 4.65, "Meituan T15 Coupon Strategy - End-to-End Architecture", ha="center", fontsize=17, weight="bold", color="#20252b")
    axis.text(6.5, 4.18, "One pipeline connects competition data, business simulation, online allocation, and presentation.", ha="center", fontsize=10.5, color="#606a73")
    axis.text(6.5, 0.85, "Shared interfaces connect calibrated user state, budget-aware allocation, and decision analysis.", ha="center", fontsize=10, color="#3d4750")
    path = OUTPUT / "system_architecture.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return path


def build_benchmark() -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    users, parameters = build_demo_population(500, 42)
    histories = []
    for seed in range(3):
        run = run_project(
            ProjectConfig(
                users=500,
                steps=20_000,
                budget=120_000.0,
                alpha=0.25,
                feedback_batch_size=16,
                seed=seed,
            ),
            users=users,
            parameters=parameters,
        )
        frame = run.history.copy()
        frame["seed"] = seed
        histories.append(frame)
    history = pd.concat(histories, ignore_index=True)
    rolling = (
        history.set_index("step")
        .groupby("seed")[["expected_reward", "oracle_expected_reward", "expected_regret"]]
        .rolling(1_000, min_periods=100)
        .mean()
        .reset_index()
    )
    mean = rolling.groupby("step").mean(numeric_only=True)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    figure.patch.set_facecolor("white")
    axes[0].plot(mean.index, mean["expected_reward"], label="Shared LinUCB", color="#f05a28", linewidth=2.2)
    axes[0].plot(mean.index, mean["oracle_expected_reward"], label="Contextual oracle", color="#334e68", linestyle="--", linewidth=1.8)
    axes[0].axhline(run.baselines["best_fixed"], label="Best fixed action", color="#2a9d8f", linestyle=":", linewidth=1.8)
    axes[0].set_title("Expected reward during learning", loc="left", weight="bold")
    axes[0].set_xlabel("Policy updates")
    axes[0].set_ylabel("Incremental net reward")
    axes[0].legend(frameon=False)

    axes[1].plot(mean.index, mean["expected_regret"], color="#7b2cbf", linewidth=2.2)
    axes[1].set_title("Expected regret declines", loc="left", weight="bold")
    axes[1].set_xlabel("Policy updates")
    axes[1].set_ylabel("Expected regret")
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Reference benchmark - 500 users, 20,000 updates, 3 policy seeds", fontsize=14, weight="bold", y=1.02)
    figure.tight_layout()
    path = OUTPUT / "benchmark_results.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    for asset in (build_architecture(), build_benchmark()):
        print(asset)
