"""Project-level orchestration for the Meituan T15 coupon strategy.

The domain modules remain intentionally separate: feature engineering prepares
the campaign state, the simulator defines the decision environment, and the
bandit implements online allocation.  This module connects them into the one
pipeline used by the dashboard, command-line demo, benchmark, and tests.
"""

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Dict, Optional

import numpy as np
import pandas as pd

from contextual_bandit import CouponBanditEnvironment, run_contextual_bandit
from feature_engineering import FeatureEngineering
from generate_synthetic_demo import generate


@dataclass(frozen=True)
class ProjectConfig:
    users: int = 300
    steps: int = 10_000
    budget: float = 60_000.0
    alpha: float = 0.25
    feedback_batch_size: int = 16
    cost_penalty: float = 0.25
    seed: int = 42
    policy_type: str = "shared"

    def validate(self) -> None:
        if self.users < 1 or self.steps < 1:
            raise ValueError("users and steps must be positive")
        if self.budget < 0:
            raise ValueError("budget must be non-negative")
        if self.alpha < 0 or self.feedback_batch_size < 1:
            raise ValueError("alpha must be non-negative and feedback batch size positive")
        if self.policy_type not in {"shared", "disjoint"}:
            raise ValueError("policy_type must be shared or disjoint")


@dataclass
class ProjectRun:
    config: ProjectConfig
    users: pd.DataFrame
    parameters: Dict
    history: pd.DataFrame
    baselines: Dict[str, float]

    def summary(self, window: Optional[int] = None) -> Dict[str, float]:
        if window is None:
            window = min(1_000, max(100, self.config.steps // 5))
        window = min(int(window), len(self.history))
        first = self.history.head(window)
        final = self.history.tail(window)
        expected = float(final["expected_reward"].mean())
        oracle = float(final["oracle_expected_reward"].mean())
        first_regret = float(first["expected_regret"].mean())
        final_regret = float(final["expected_regret"].mean())
        return {
            "window": float(window),
            "final_expected_reward": expected,
            "final_oracle_reward": oracle,
            "oracle_capture_pct": 100.0 * expected / oracle if oracle else 0.0,
            "first_regret": first_regret,
            "final_regret": final_regret,
            "regret_reduction_pct": (
                100.0 * (1.0 - final_regret / first_regret) if first_regret else 0.0
            ),
            "coupon_cost": float(self.history["cost"].sum()),
            "remaining_budget": float(self.history["remaining_budget"].iloc[-1]),
        }


def build_population(data_directory: str | Path, users: int, seed: int):
    """Load source-schema spreadsheets and build campaign state."""
    features = FeatureEngineering()
    if not features.load_data(str(data_directory)):
        raise RuntimeError("the source-schema spreadsheets could not be loaded")
    return features.build_features_and_params(
        n_users=int(users),
        random_state=int(seed),
        use_auto_distribution_selector=False,
    )


def build_demo_population(users: int, seed: int):
    """Create the public fixture and pass it through the production-style adapter."""
    with tempfile.TemporaryDirectory() as directory:
        generate(directory, seed=int(seed))
        return build_population(directory, users=int(users), seed=int(seed))


def compute_policy_baselines(users: pd.DataFrame, parameters: Dict) -> Dict[str, float]:
    """Calculate random, fixed-action, and contextual-oracle references."""
    environment = CouponBanditEnvironment(users, parameters, seed=0)
    rewards = np.array([
        [environment.expected_reward(index, action) for action in environment.actions]
        for index in range(len(users))
    ])
    return {
        "random": float(rewards.mean()),
        "best_fixed": float(rewards.mean(axis=0).max()),
        "oracle": float(rewards.max(axis=1).mean()),
    }


def format_action(action, environment: CouponBanditEnvironment) -> str:
    """Return an English display label for a source-schema coupon action."""
    if action is None or pd.isna(action):
        return "No coupon"
    metadata = environment.action_metadata(action)
    properties = []
    if metadata["free"]:
        properties.append("free")
    if metadata["expand"]:
        properties.append("expanded-value")
    suffix = f" ({', '.join(properties)})" if properties else ""
    return f"CNY {metadata['coupon_value']:g} coupon{suffix}"


def run_project(
    config: ProjectConfig,
    data_directory: Optional[str | Path] = None,
    users: Optional[pd.DataFrame] = None,
    parameters: Optional[Dict] = None,
) -> ProjectRun:
    """Execute the complete data-to-decision project pipeline."""
    config.validate()
    if (users is None) != (parameters is None):
        raise ValueError("users and parameters must be provided together")
    if users is None and parameters is None:
        if data_directory is None:
            users, parameters = build_demo_population(config.users, config.seed)
        else:
            users, parameters = build_population(data_directory, config.users, config.seed)
    else:
        users = users.copy()
    history = run_contextual_bandit(
        users,
        parameters,
        steps=config.steps,
        budget=config.budget,
        alpha=config.alpha,
        seed=config.seed,
        cost_penalty=config.cost_penalty,
        policy_type=config.policy_type,
        feedback_batch_size=config.feedback_batch_size,
    )
    return ProjectRun(
        config=config,
        users=users,
        parameters=parameters,
        history=history,
        baselines=compute_policy_baselines(users, parameters),
    )
