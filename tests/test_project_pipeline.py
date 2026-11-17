import re
import unittest

from contextual_bandit import CouponBanditEnvironment
from project_pipeline import ProjectConfig, format_action, run_project


class ProjectPipelineTest(unittest.TestCase):
    def test_complete_public_pipeline_produces_decisions_and_summary(self):
        config = ProjectConfig(
            users=40,
            steps=80,
            budget=500.0,
            alpha=0.25,
            feedback_batch_size=4,
            seed=11,
        )
        run = run_project(config)

        self.assertEqual(len(run.users), 40)
        self.assertEqual(len(run.history), 80)
        self.assertEqual(set(run.baselines), {"random", "best_fixed", "oracle"})
        self.assertTrue((run.history["remaining_budget"] >= 0).all())
        self.assertLessEqual(run.history["cost"].sum(), config.budget + 1e-9)
        summary = run.summary(window=20)
        self.assertGreater(summary["final_oracle_reward"], 0)
        self.assertGreaterEqual(summary["oracle_capture_pct"], 0)

    def test_invalid_project_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "users and steps"):
            run_project(ProjectConfig(users=0, steps=10))

    def test_public_action_labels_are_english(self):
        run = run_project(ProjectConfig(users=20, steps=10, seed=7))
        environment = CouponBanditEnvironment(run.users, run.parameters, seed=7)
        labels = [format_action(action, environment) for action in environment.actions]

        self.assertEqual(labels[0], "No coupon")
        self.assertTrue(all(not re.search(r"[\u4e00-\u9fff]", label) for label in labels))


if __name__ == "__main__":
    unittest.main()
