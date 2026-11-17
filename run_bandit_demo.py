"""Run the sequential LinUCB experiment on deterministic public fixtures."""

import argparse
import tempfile

from contextual_bandit import run_contextual_bandit
from feature_engineering import FeatureEngineering
from generate_synthetic_demo import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--budget", type=float, default=3000.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--cost-penalty", type=float, default=0.25)
    parser.add_argument("--feedback-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        generate(directory, seed=args.seed)
        features = FeatureEngineering()
        if not features.load_data(directory):
            raise RuntimeError("failed to load generated fixture")
        users, parameters = features.build_features_and_params(
            n_users=200, random_state=args.seed, use_auto_distribution_selector=False
        )
        history = run_contextual_bandit(
            users, parameters, steps=args.steps, budget=args.budget,
            alpha=args.alpha, seed=args.seed, cost_penalty=args.cost_penalty,
            feedback_batch_size=args.feedback_batch_size,
        )

    print(history.tail(10).to_string(index=False))
    print(f"\nTotal reward: {history.reward.sum():.2f}")
    print(f"Total coupon cost: {history.cost.sum():.2f}")
    print(f"Cumulative expected regret: {history.expected_regret.sum():.2f}")


if __name__ == "__main__":
    main()
