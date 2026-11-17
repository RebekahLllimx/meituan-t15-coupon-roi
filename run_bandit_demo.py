"""Run the sequential LinUCB experiment on deterministic public fixtures."""

import argparse

from contextual_bandit import CouponBanditEnvironment
from project_pipeline import ProjectConfig, format_action, run_project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--budget", type=float, default=3000.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--cost-penalty", type=float, default=0.25)
    parser.add_argument("--feedback-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run = run_project(ProjectConfig(
        users=200,
        steps=args.steps,
        budget=args.budget,
        alpha=args.alpha,
        cost_penalty=args.cost_penalty,
        feedback_batch_size=args.feedback_batch_size,
        seed=args.seed,
    ))
    history = run.history.copy()
    environment = CouponBanditEnvironment(
        run.users,
        run.parameters,
        seed=args.seed,
        cost_penalty=args.cost_penalty,
        feedback_batch_size=args.feedback_batch_size,
    )
    history["action"] = history["action"].map(
        lambda action: format_action(action, environment)
    )

    print(history.tail(10).to_string(index=False))
    print(f"\nTotal reward: {history.reward.sum():.2f}")
    print(f"Total coupon cost: {history.cost.sum():.2f}")
    print(f"Cumulative expected regret: {history.expected_regret.sum():.2f}")


if __name__ == "__main__":
    main()
