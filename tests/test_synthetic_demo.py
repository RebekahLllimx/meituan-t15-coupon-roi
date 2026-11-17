import tempfile
import unittest
from pathlib import Path

import pandas as pd

from feature_engineering import FeatureEngineering
from generate_synthetic_demo import FILES, generate
from optimization import Optimization
from simulation_engine import SimulationEngine
from result_cache import ResultCache


class SyntheticDemoTest(unittest.TestCase):
    def test_generator_runs_core_flow_without_real_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            counts = generate(directory, seed=42)
            self.assertEqual(set(counts), set(FILES))
            self.assertTrue(all((Path(directory) / name).exists() for name in FILES))
            with self.assertRaises(FileExistsError):
                generate(directory, seed=42)

            features = FeatureEngineering()
            self.assertTrue(features.load_data(directory))
            users, parameters = features.build_features_and_params(
                n_users=20, random_state=42, use_auto_distribution_selector=False
            )
            self.assertEqual(len(users), 20)
            self.assertGreater(parameters["funnel"]["scene_sessions"], 0)
            baseline = SimulationEngine(users, parameters).calculate_expected_metrics(
                {user_id: None for user_id in users["user_id"]}
            )
            self.assertIn("gtv_no_coupon", baseline)
            result = Optimization(users, parameters, order_df=features.order_df).optimize(
                budget=100, epochs=1, show_progress=False
            )
            self.assertIn("allocation", result)
            selected = result["allocation_df"]
            self.assertLessEqual(float(selected["cost"].sum()), 100.0 + 1e-9)
            self.assertEqual(selected["user_id"].nunique(), len(selected))
            cap = max(1, int(len(users) * result["constraints"]["coupon_cap_share"]))
            self.assertTrue(all(count <= cap for count in result["coupon_usage"].values()))

    def test_generator_is_deterministic_for_the_same_seed(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            generate(first, seed=17)
            generate(second, seed=17)
            for filename in FILES:
                pd.testing.assert_frame_equal(
                    pd.read_excel(Path(first) / filename),
                    pd.read_excel(Path(second) / filename),
                    check_dtype=True,
                )

    def test_zero_budget_produces_no_allocations(self):
        with tempfile.TemporaryDirectory() as directory:
            generate(directory, seed=9)
            features = FeatureEngineering()
            self.assertTrue(features.load_data(directory))
            users, parameters = features.build_features_and_params(
                n_users=16, random_state=9, use_auto_distribution_selector=False
            )
            result = Optimization(users, parameters).optimize(
                budget=0, epochs=1, show_progress=False
            )
            self.assertTrue(result["allocation_df"].empty)
            self.assertEqual(result["optimized_metrics"]["total_cost"], 0.0)

    def test_progress_reporting_does_not_depend_on_streamlit(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            generate(directory, seed=11)
            features = FeatureEngineering()
            self.assertTrue(features.load_data(directory))
            users, parameters = features.build_features_and_params(
                n_users=8, random_state=11, use_auto_distribution_selector=False
            )
            Optimization(users, parameters, progress_callback=events.append).optimize(
                budget=25, epochs=1, show_progress=True
            )
        self.assertTrue(events)
        self.assertEqual(events[-1].stage, "allocation")
        self.assertEqual(events[-1].fraction, 1.0)

    def test_optimizer_rejects_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            generate(directory, seed=13)
            features = FeatureEngineering()
            self.assertTrue(features.load_data(directory))
            users, parameters = features.build_features_and_params(
                n_users=8, random_state=13, use_auto_distribution_selector=False
            )

            with self.assertRaisesRegex(ValueError, "user_id values must be unique"):
                Optimization(pd.concat([users, users.iloc[[0]]]), parameters)
            with self.assertRaisesRegex(ValueError, "coupon_cap_share"):
                Optimization(users, parameters, coupon_cap_share=0)
            with self.assertRaisesRegex(ValueError, "alpha_ucb"):
                Optimization(users, parameters, alpha_ucb=-0.1)
            with self.assertRaisesRegex(ValueError, "budget"):
                Optimization(users, parameters).optimize(-1, show_progress=False)

    def test_result_cache_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ResultCache(directory)
            with self.assertRaisesRegex(ValueError, "run_id"):
                cache.load("../../outside")


if __name__ == "__main__":
    unittest.main()
