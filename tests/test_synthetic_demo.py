import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import FeatureEngineering
from generate_synthetic_demo import FILES, generate
from simulation_engine import SimulationEngine


class SyntheticDemoTest(unittest.TestCase):
    def test_generator_runs_feature_and_simulation_flow(self):
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

            engine = SimulationEngine(users, parameters)
            responses = [
                engine.funnel_response(users.iloc[0], action)
                for action in (None, *parameters["candidate_coupons"])
            ]
            self.assertTrue(all(np.isfinite(item["gtv_with_coupon"]) for item in responses))
            self.assertTrue(all(0 <= item["prob_with_coupon"] <= 1 for item in responses))

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

    def test_missing_inputs_fail_without_creating_data(self):
        with tempfile.TemporaryDirectory() as directory:
            features = FeatureEngineering()
            self.assertFalse(features.load_data(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
