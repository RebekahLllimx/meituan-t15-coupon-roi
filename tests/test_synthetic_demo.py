import tempfile
import unittest
from pathlib import Path

from feature_engineering import FeatureEngineering
from generate_synthetic_demo import FILES, generate
from optimization import Optimization
from simulation_engine import SimulationEngine


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


if __name__ == "__main__":
    unittest.main()
