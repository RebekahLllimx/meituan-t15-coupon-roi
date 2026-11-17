"""Sequential contextual-bandit experiment for coupon allocation.

The learner receives feedback only for the action it selected.  The environment
is semi-synthetic and calibrated by ``SimulationEngine``; consequently this
module demonstrates a real bandit learning loop, not production effectiveness.
"""

import copy
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
    purchase_rate: float
    expected_reward: float
    oracle_expected_reward: float
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
        self._a_inverse = {
            action: np.eye(context_dim) / float(ridge) for action in self.actions
        }
        self._b = {action: np.zeros(context_dim) for action in self.actions}
        self.observations = {action: 0 for action in self.actions}

    def score(self, context: np.ndarray, action: Optional[str]) -> float:
        inverse = self._a_inverse[action]
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
        inverse = self._a_inverse[action]
        inverse_context = inverse @ context
        denominator = 1.0 + float(context @ inverse_context)
        self._a_inverse[action] = inverse - np.outer(inverse_context, inverse_context) / denominator
        self._b[action] += float(reward) * context
        self.observations[action] += 1


class SharedLinUCB:
    """A single linear reward model over context-action feature vectors."""

    def __init__(self, feature_dim: int, alpha: float = 0.25, ridge: float = 1.0):
        if feature_dim < 1 or alpha < 0 or ridge <= 0:
            raise ValueError("invalid LinUCB configuration")
        self.alpha = float(alpha)
        self._a = np.eye(feature_dim) * float(ridge)
        self._a_inverse = np.eye(feature_dim) / float(ridge)
        self._b = np.zeros(feature_dim)
        self.observations = 0

    def score(self, features: np.ndarray) -> float:
        theta = self._a_inverse @ self._b
        uncertainty = np.sqrt(max(0.0, float(features @ self._a_inverse @ features)))
        return float(theta @ features + self.alpha * uncertainty)

    def select(self, action_features: Dict[Optional[str], np.ndarray]) -> Optional[str]:
        return max(action_features, key=lambda action: self.score(action_features[action]))

    def update(self, features: np.ndarray, reward: float) -> None:
        self._a += np.outer(features, features)
        inverse_features = self._a_inverse @ features
        denominator = 1.0 + float(features @ inverse_features)
        self._a_inverse -= np.outer(inverse_features, inverse_features) / denominator
        self._b += float(reward) * features
        self.observations += 1


class CouponBanditEnvironment:
    """Sample partial, stochastic feedback from the existing funnel model."""

    def __init__(
        self,
        users: pd.DataFrame,
        global_params: Dict,
        seed: int = 0,
        cost_penalty: float = 0.25,
        reward_mode: str = "incremental_net",
        feedback_batch_size: int = 16,
    ):
        if users.empty or users["user_id"].duplicated().any():
            raise ValueError("users must be non-empty with unique user_id values")
        self.users = users.reset_index(drop=True).copy()
        environment_params = copy.deepcopy(global_params)
        environment_params["heterogeneous_coupon_fit"] = True
        environment_params.setdefault("coupon_fit_sigma", 0.08)
        self.engine = SimulationEngine(self.users, environment_params)
        self.actions = (None, *tuple(global_params.get("candidate_coupons", [])))
        if len(self.actions) < 2:
            raise ValueError("the environment requires at least one coupon action")
        self.rng = np.random.default_rng(seed)
        self.cost_penalty = float(cost_penalty)
        if reward_mode not in {"incremental_net", "absolute_net"}:
            raise ValueError("reward_mode must be incremental_net or absolute_net")
        self.reward_mode = reward_mode
        if int(feedback_batch_size) < 1:
            raise ValueError("feedback_batch_size must be at least 1")
        self.feedback_batch_size = int(feedback_batch_size)
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
        value, activity, sensitivity = normalized.T
        # The simulator is deliberately nonlinear. Quadratic and interaction
        # terms let a linear bandit approximate threshold and sensitivity effects
        # without hiding the learning process behind a neural network.
        return np.column_stack([
            np.ones(len(raw)),
            value,
            activity,
            sensitivity,
            value * sensitivity,
            activity * sensitivity,
            value * activity,
            value ** 2,
            activity ** 2,
            sensitivity ** 2,
        ])

    @property
    def context_dim(self) -> int:
        return int(self._context_matrix.shape[1])

    def context(self, user_index: int) -> np.ndarray:
        return self._context_matrix[int(user_index)].copy()

    def action_context(self, user_index: int, action: Optional[str]) -> np.ndarray:
        user_context = self.context(user_index)
        metadata = self.engine._parse_coupon_name(action)
        reference_value = max(1.0, float(self.engine.reference_means["value"]))
        action_features = np.array([
            float(metadata["coupon_value"]) / reference_value,
            float(metadata["free"]),
            float(metadata["expand"]),
            float(metadata["uplift"]) / reference_value,
            float(metadata["attraction"]) - 1.0,
        ])
        # Share statistical strength across coupons through their properties and
        # interactions with the three original normalized user variables.
        interactions = np.outer(user_context[1:4], action_features).ravel()
        return np.concatenate([user_context, action_features, interactions])

    @property
    def shared_feature_dim(self) -> int:
        return int(len(self.action_context(0, self.actions[0])))

    def coupon_value(self, action: Optional[str]) -> float:
        return float(self.engine._parse_coupon_name(action)["coupon_value"])

    def expected_reward(self, user_index: int, action: Optional[str]) -> float:
        user = self.users.iloc[int(user_index)]
        response = self.engine.funnel_response(user, coupon=action)
        reward = response["gtv_with_coupon"] - self.cost_penalty * response["cost"]
        if self.reward_mode == "incremental_net":
            reward -= response["gtv_no_coupon"]
        return float(reward)

    def step(self, user_index: int, action: Optional[str]):
        """Return only the selected action's realized reward and cost."""
        user = self.users.iloc[int(user_index)]
        response = self.engine.funnel_response(user, coupon=action)
        purchases = self.rng.random(self.feedback_batch_size) < response["prob_with_coupon"]
        purchase_rate = float(purchases.mean())
        coupon_cost = self.coupon_value(action) * purchase_rate
        realized_gtv = (response["effective_V"] + response["uplift"]) * purchase_rate
        reward = float(realized_gtv - self.cost_penalty * coupon_cost)
        if self.reward_mode == "incremental_net":
            # An action-independent baseline reduces variance and aligns learning
            # with incremental subsidy value. It does not reveal another coupon's
            # realized outcome to the policy.
            reward -= float(response["gtv_no_coupon"])
        return reward, float(coupon_cost), purchase_rate


def run_contextual_bandit(
    users: pd.DataFrame,
    global_params: Dict,
    steps: int = 500,
    budget: float = 3000.0,
    alpha: float = 0.5,
    seed: int = 0,
    cost_penalty: float = 0.25,
    reward_scale: Optional[float] = None,
    reward_mode: str = "incremental_net",
    policy_type: str = "shared",
    feedback_batch_size: int = 16,
) -> pd.DataFrame:
    """Run a budget-aware LinUCB policy against partial simulated feedback."""
    if steps < 1 or budget < 0:
        raise ValueError("steps must be positive and budget must be non-negative")
    environment = CouponBanditEnvironment(
        users, global_params, seed=seed, cost_penalty=cost_penalty,
        reward_mode=reward_mode, feedback_batch_size=feedback_batch_size,
    )
    if policy_type == "shared":
        policy = SharedLinUCB(environment.shared_feature_dim, alpha=alpha)
    elif policy_type == "disjoint":
        policy = DisjointLinUCB(environment.actions, environment.context_dim, alpha=alpha)
    else:
        raise ValueError("policy_type must be shared or disjoint")
    if reward_scale is None:
        reward_scale = max(1.0, float(global_params["reference_means"]["value"]))
    if float(reward_scale) <= 0:
        raise ValueError("reward_scale must be positive")
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
        if policy_type == "shared":
            feasible_features = {
                action: environment.action_context(user_index, action) for action in feasible
            }
            action = policy.select(feasible_features)
            learning_features = feasible_features[action]
        else:
            action = policy.select(context, feasible)
            learning_features = context
        expected = {candidate: environment.expected_reward(user_index, candidate) for candidate in feasible}
        reward, cost, purchase_rate = environment.step(user_index, action)
        remaining_budget = max(0.0, remaining_budget - cost)
        # Normalize noisy monetary rewards for numerically meaningful confidence
        # bounds while keeping raw currency values in the evaluation history.
        if policy_type == "shared":
            policy.update(learning_features, reward / float(reward_scale))
        else:
            policy.update(learning_features, action, reward / float(reward_scale))
        regret = max(expected.values()) - expected[action]
        records.append(BanditStep(
            step=step_number + 1,
            user_id=environment.users.iloc[user_index]["user_id"],
            action=action,
            reward=reward,
            cost=cost,
            purchase_rate=purchase_rate,
            expected_reward=float(expected[action]),
            oracle_expected_reward=float(max(expected.values())),
            expected_regret=float(max(0.0, regret)),
            remaining_budget=remaining_budget,
        ))

    return pd.DataFrame([record.__dict__ for record in records])
