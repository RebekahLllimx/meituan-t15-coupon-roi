"""Sequential contextual-bandit experiment for coupon allocation.

The learner receives feedback only for the action it selected.  The environment
is semi-synthetic and calibrated by ``SimulationEngine``; consequently this
module demonstrates a real bandit learning loop, not production effectiveness.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from simulation_engine import SimulationEngine


@dataclass(frozen=True)
class BanditStep:
    step: int
    user_id: object
    action: Optional[str]
    reward: float
    cost: float
    purchased: bool
    expected_regret: float
    remaining_budget: float


class DisjointLinUCB:
    """One ridge-regression model per action with an upper-confidence bound."""

    def __init__(self, actions: Iterable[Optional[str]], context_dim: int, alpha: float = 0.5, ridge: float = 1.0):
        self.actions = tuple(actions)
        if not self.actions:
            raise ValueError("at least one action is required")
        if context_dim < 1 or alpha < 0 or ridge <= 0:
            raise ValueError("invalid LinUCB configuration")
        self.alpha = float(alpha)
        self._a = {action: np.eye(context_dim) * float(ridge) for action in self.actions}
        self._b = {action: np.zeros(context_dim) for action in self.actions}
        self.observations = {action: 0 for action in self.actions}

    def score(self, context: np.ndarray, action: Optional[str]) -> float:
        inverse = np.linalg.inv(self._a[action])
        theta = inverse @ self._b[action]
        uncertainty = np.sqrt(max(0.0, float(context @ inverse @ context)))
        return float(theta @ context + self.alpha * uncertainty)

    def select(self, context: np.ndarray, feasible_actions: Iterable[Optional[str]]) -> Optional[str]:
        feasible = tuple(feasible_actions)
        if not feasible:
            raise ValueError("no feasible action is available")
        # Stable action order makes equal-score runs reproducible.
        return max(feasible, key=lambda action: (self.score(context, action), -self.actions.index(action)))

    def update(self, context: np.ndarray, action: Optional[str], reward: float) -> None:
        self._a[action] += np.outer(context, context)
        self._b[action] += float(reward) * context
        self.observations[action] += 1


class CouponBanditEnvironment:
    """Sample partial, stochastic feedback from the existing funnel model."""

    def __init__(self, users: pd.DataFrame, global_params: Dict, seed: int = 0, cost_penalty: float = 1.0):
        if users.empty or users["user_id"].duplicated().any():
            raise ValueError("users must be non-empty with unique user_id values")
        self.users = users.reset_index(drop=True).copy()
        self.engine = SimulationEngine(self.users, global_params)
        self.actions = (None, *tuple(global_params.get("candidate_coupons", [])))
        if len(self.actions) < 2:
            raise ValueError("the environment requires at least one coupon action")
        self.rng = np.random.default_rng(seed)
        self.cost_penalty = float(cost_penalty)
        self._context_matrix = self._build_context_matrix()

    def _build_context_matrix(self) -> np.ndarray:
        sensitivity = self.users["S_alpha_post"] / (
            self.users["S_alpha_post"] + self.users["S_beta_post"]
        )
        raw = np.column_stack([
            np.log1p(self.users["V"].astype(float)),
            np.log1p(self.users["A"].astype(float)),
            sensitivity.astype(float),
        ])
        scale = raw.std(axis=0)
        scale[scale < 1e-9] = 1.0
        normalized = (raw - raw.mean(axis=0)) / scale
        return np.column_stack([np.ones(len(raw)), normalized])

    @property
    def context_dim(self) -> int:
        return int(self._context_matrix.shape[1])

    def context(self, user_index: int) -> np.ndarray:
        return self._context_matrix[int(user_index)].copy()

    def coupon_value(self, action: Optional[str]) -> float:
        return float(self.engine._parse_coupon_name(action)["coupon_value"])

    def expected_reward(self, user_index: int, action: Optional[str]) -> float:
        user = self.users.iloc[int(user_index)]
        response = self.engine.funnel_response(user, coupon=action)
        return float(response["gtv_with_coupon"] - self.cost_penalty * response["cost"])

    def step(self, user_index: int, action: Optional[str]):
        """Return only the selected action's realized reward and cost."""
        user = self.users.iloc[int(user_index)]
        response = self.engine.funnel_response(user, coupon=action)
        purchased = bool(self.rng.random() < response["prob_with_coupon"])
        coupon_cost = self.coupon_value(action) if purchased else 0.0
        realized_gtv = (response["effective_V"] + response["uplift"]) if purchased else 0.0
        reward = float(realized_gtv - self.cost_penalty * coupon_cost)
        return reward, float(coupon_cost), purchased


def run_contextual_bandit(
    users: pd.DataFrame,
    global_params: Dict,
    steps: int = 500,
    budget: float = 3000.0,
    alpha: float = 0.5,
    seed: int = 0,
    cost_penalty: float = 1.0,
) -> pd.DataFrame:
    """Run a budget-aware LinUCB policy against partial simulated feedback."""
    if steps < 1 or budget < 0:
        raise ValueError("steps must be positive and budget must be non-negative")
    environment = CouponBanditEnvironment(users, global_params, seed=seed, cost_penalty=cost_penalty)
    policy = DisjointLinUCB(environment.actions, environment.context_dim, alpha=alpha)
    user_rng = np.random.default_rng(seed + 1)
    remaining_budget = float(budget)
    records: List[BanditStep] = []

    for step_number in range(int(steps)):
        user_index = int(user_rng.integers(0, len(environment.users)))
        context = environment.context(user_index)
        feasible = [
            action for action in environment.actions
            if action is None or environment.coupon_value(action) <= remaining_budget
        ]
        action = policy.select(context, feasible)
        expected = {candidate: environment.expected_reward(user_index, candidate) for candidate in feasible}
        reward, cost, purchased = environment.step(user_index, action)
        remaining_budget = max(0.0, remaining_budget - cost)
        policy.update(context, action, reward)
        regret = max(expected.values()) - expected[action]
        records.append(BanditStep(
            step=step_number + 1,
            user_id=environment.users.iloc[user_index]["user_id"],
            action=action,
            reward=reward,
            cost=cost,
            purchased=purchased,
            expected_regret=float(max(0.0, regret)),
            remaining_budget=remaining_budget,
        ))

    return pd.DataFrame([record.__dict__ for record in records])
