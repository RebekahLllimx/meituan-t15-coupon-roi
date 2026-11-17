"""Reproduce convergence and policy baselines for the contextual bandit."""

import argparse
import tempfile

import numpy as np
import pandas as pd

from contextual_bandit import CouponBanditEnvironment, run_contextual_bandit
from feature_engineering import FeatureEngineering
from generate_synthetic_demo import generate


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

    with tempfile.TemporaryDirectory() as directory:
        generate(directory, seed=42)
        features = FeatureEngineering()
        if not features.load_data(directory):
            raise RuntimeError("failed to load generated fixture")
        users, parameters = features.build_features_and_params(
            n_users=args.users, random_state=42, use_auto_distribution_selector=False
        )
        environment = CouponBanditEnvironment(users, parameters, seed=0)
        reward_matrix = np.array([
            [environment.expected_reward(index, action) for action in environment.actions]
            for index in range(len(users))
        ])
        print("Environment baselines")
        print(f"  random action: {reward_matrix.mean():.6f}")
        print(f"  best fixed action: {reward_matrix.mean(axis=0).max():.6f}")
        print(f"  contextual oracle: {reward_matrix.max(axis=1).mean():.6f}")

        rows = []
        for seed in range(args.seeds):
            history = run_contextual_bandit(
                users,
                parameters,
                steps=args.steps,
                budget=args.budget,
                alpha=args.alpha,
                seed=seed,
                policy_type="shared",
                feedback_batch_size=args.feedback_batch_size,
            )
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
