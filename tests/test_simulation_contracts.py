import copy
import unittest
from unittest.mock import patch

import numpy as np

from contextual_bandit import CouponBanditEnvironment
from feature_engineering import FeatureEngineering
from project_pipeline import build_demo_population
from simulation_engine import SimulationEngine


class SimulationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.users, cls.parameters = build_demo_population(12, 42)

    def test_nonexpanded_fallback_label_is_not_expanded(self):
        engine = SimulationEngine(self.users, self.parameters)
        self.assertEqual(engine.coupon_metadata("99元免费非膨胀")["expand"], 0)
        self.assertEqual(engine.coupon_metadata("99元免费膨胀")["expand"], 1)

    def test_empty_coupon_catalog_fallback_has_consistent_flags(self):
        features = FeatureEngineering()
        import pandas as pd
        features.scene_orders = pd.DataFrame({"美补金额": [0]})
        self.assertTrue(all(value["expand"] == 0 for value in features._build_coupon_catalog().values()))

    def test_explicit_plain_response_scenario_is_respected(self):
        parameters = copy.deepcopy(self.parameters)
        parameters["heterogeneous_coupon_fit"] = False
        env = CouponBanditEnvironment(self.users, parameters)
        self.assertFalse(env.engine.global_params["heterogeneous_coupon_fit"])

    def test_mixture_sampling_advances_the_supplied_random_generator(self):
        from sklearn.mixture import GaussianMixture
        values = np.concatenate([np.linspace(1,2,30),np.linspace(6,8,30)])
        gmm = GaussianMixture(n_components=2,random_state=42).fit(values.reshape(-1,1))
        spec = {"generator":"mixture","model":"gmm","gmm_model":gmm,
                "support":"positive","lower":0.1,"upper":10.0}
        features = FeatureEngineering()
        first_rng, second_rng = np.random.default_rng(4), np.random.default_rng(4)
        first = [features._sample_from_spec(spec,first_rng,size=1)[0] for _ in range(10)]
        second = [features._sample_from_spec(spec,second_rng,size=1)[0] for _ in range(10)]
        np.testing.assert_array_equal(first,second)
        self.assertGreater(np.std(first),0.01)

    def test_stochastic_replay_uses_effective_basket_and_realized_discounted_delta(self):
        parameters = copy.deepcopy(self.parameters)
        parameters["scene_constraints"]["time_slots"] = ["15:00"]
        engine = SimulationEngine(self.users, parameters, simulation_days=1)
        user = self.users.iloc[0]
        response = engine.funnel_response(user, engine.global_params["candidate_coupons"][0])
        response.update(prob_no_coupon=0.0, prob_with_coupon=1.0,
                        effective_V=100.0, uplift=5.0, coupon_value=3.0,
                        causal_discount_weight=0.5, delta_gtv_causal=999.0)
        with patch.object(engine, "funnel_response", return_value=response):
            result = engine._aggregate_user_metrics(user, "coupon", stochastic=True,
                                                    rng=np.random.default_rng(4))
        self.assertEqual(result["gtv_with_coupon"], 105)
        self.assertEqual(result["total_cost"], 3)
        self.assertEqual(result["delta_gtv_causal"], 52.5)


if __name__ == "__main__":
    unittest.main()
