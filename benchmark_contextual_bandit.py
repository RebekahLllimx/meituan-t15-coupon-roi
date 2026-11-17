"""Reproduce convergence and policy baselines for the contextual bandit."""

import argparse
import pandas as pd

from project_pipeline import ProjectConfig, build_demo_population, compute_policy_baselines, run_project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--window", type=int, default=1_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--budget", type=float, default=120_000)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--feedback-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.window > args.steps:
        raise ValueError("window cannot exceed steps")

    users, parameters = build_demo_population(args.users, 42)
    baselines = compute_policy_baselines(users, parameters)
    print("Environment baselines")
    print(f"  random action: {baselines['random']:.6f}")
    print(f"  best fixed action: {baselines['best_fixed']:.6f}")
    print(f"  contextual oracle: {baselines['oracle']:.6f}")

    rows = []
    for seed in range(args.seeds):
        run = run_project(
            ProjectConfig(
                users=args.users,
                steps=args.steps,
                budget=args.budget,
                alpha=args.alpha,
                feedback_batch_size=args.feedback_batch_size,
                seed=seed,
            ),
            users=users,
            parameters=parameters,
        )
        history = run.history
        first_regret = history.expected_regret.iloc[: args.window].mean()
        last_regret = history.expected_regret.iloc[-args.window :].mean()
        last_expected = history.expected_reward.iloc[-args.window :].mean()
        last_oracle = history.oracle_expected_reward.iloc[-args.window :].mean()
        rows.append({
            "seed": seed,
            "first_window_regret": first_regret,
            "last_window_regret": last_regret,
            "regret_reduction_pct": 100 * (1 - last_regret / first_regret),
            "last_window_expected_reward": last_expected,
            "oracle_capture_pct": 100 * last_expected / last_oracle,
            "coupon_cost": history.cost.sum(),
        })

    results = pd.DataFrame(rows)
    print("\nPer-seed results")
    print(results.to_string(index=False))
    print("\nMean")
    print(results.mean(numeric_only=True).to_string())
    print("\nStandard deviation")
    print(results.std(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
