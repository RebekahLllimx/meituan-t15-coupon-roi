import tempfile
import unittest

from contextual_bandit import CouponBanditEnvironment, DisjointLinUCB, run_contextual_bandit
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
        policy = DisjointLinUCB([None, "5元券"], context_dim=2, alpha=0.2)
        context = __import__("numpy").array([1.0, 0.5])
        policy.update(context, "5元券", reward=3.0)
        self.assertEqual(policy.observations["5元券"], 1)
        self.assertEqual(policy.observations[None], 0)

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


if __name__ == "__main__":
    unittest.main()
