import tempfile
import unittest

import numpy as np

from contextual_bandit import CouponBanditEnvironment, DisjointLinUCB, SharedLinUCB, run_contextual_bandit
from feature_engineering import FeatureEngineering
from generate_synthetic_demo import generate


def build_fixture(seed=21, n_users=24):
    directory = tempfile.TemporaryDirectory()
    generate(directory.name, seed=seed)
    features = FeatureEngineering()
    if not features.load_data(directory.name):
        directory.cleanup()
        raise RuntimeError("fixture failed to load")
    users, parameters = features.build_features_and_params(
        n_users=n_users, random_state=seed, use_auto_distribution_selector=False
    )
    return directory, users, parameters


class ContextualBanditTest(unittest.TestCase):
    def test_policy_updates_only_selected_action(self):
        policy = DisjointLinUCB([None, "CNY5"], context_dim=2, alpha=0.2)
        context = np.array([1.0, 0.5])
        policy.update(context, "CNY5", reward=3.0)
        self.assertEqual(policy.observations["CNY5"], 1)
        self.assertEqual(policy.observations[None], 0)
        np.testing.assert_allclose(
            policy._a_inverse["CNY5"],
            np.linalg.inv(policy._a["CNY5"]),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_environment_is_reproducible_but_stochastic(self):
        directory, users, parameters = build_fixture()
        try:
            first = CouponBanditEnvironment(users, parameters, seed=8)
            second = CouponBanditEnvironment(users, parameters, seed=8)
            outcomes_first = [first.step(0, first.actions[1]) for _ in range(20)]
            outcomes_second = [second.step(0, second.actions[1]) for _ in range(20)]
            self.assertEqual(outcomes_first, outcomes_second)
            self.assertGreater(len(set(outcomes_first)), 1)
        finally:
            directory.cleanup()

    def test_shared_policy_updates_incremental_inverse(self):
        policy = SharedLinUCB(feature_dim=3, alpha=0.2)
        features = np.array([1.0, -0.5, 0.25])
        policy.update(features, reward=0.7)
        self.assertEqual(policy.observations, 1)
        np.testing.assert_allclose(
            policy._a_inverse, np.linalg.inv(policy._a), rtol=1e-10, atol=1e-10
        )

    def test_run_uses_partial_feedback_and_respects_budget(self):
        directory, users, parameters = build_fixture(seed=27)
        try:
            history = run_contextual_bandit(
                users, parameters, steps=60, budget=30, alpha=0.4, seed=27
            )
            self.assertEqual(len(history), 60)
            self.assertTrue((history["remaining_budget"] >= -1e-9).all())
            self.assertLessEqual(history["cost"].sum(), 30.0 + 1e-9)
            self.assertTrue((history["expected_regret"] >= 0).all())
        finally:
            directory.cleanup()

    def test_personalization_scenario_has_multiple_optimal_actions(self):
        directory, users, parameters = build_fixture(seed=31, n_users=200)
        try:
            environment = CouponBanditEnvironment(users, parameters, seed=31)
            optimal_actions = set()
            for user_index in range(len(users)):
                rewards = {
                    action: environment.expected_reward(user_index, action)
                    for action in environment.actions
                }
                optimal_actions.add(max(rewards, key=rewards.get))
            self.assertGreater(len(optimal_actions), 1)
        finally:
            directory.cleanup()

    def test_invalid_policy_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid LinUCB configuration"):
            SharedLinUCB(feature_dim=0)
        with self.assertRaisesRegex(ValueError, "at least one action"):
            DisjointLinUCB([], context_dim=2)


if __name__ == "__main__":
    unittest.main()
