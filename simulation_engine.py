import math
import re

import numpy as np


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _safe_logit(p):
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return math.log(p / (1 - p))


def compute_causal_discount(beta, mode="linear", params=None):
    beta = float(np.clip(beta, 0.0, 0.995))
    mode = str(mode or "linear").lower()
    params = params or {}

    if mode == "none":
        weight = 1.0
    elif mode == "linear":
        weight = 1.0 - beta
    elif mode == "quadratic":
        weight = (1.0 - beta) ** 2
    elif mode == "piecewise":
        if beta < 0.15:
            weight = 1.0 - 0.5 * beta
        elif beta < 0.35:
            weight = 1.0 - beta
        else:
            weight = max(0.05, (1.0 - beta) ** 2)
    elif mode == "logistic":
        k = float(params.get("k", 12.0))
        b0 = float(params.get("b0", 0.25))
        weight = 1.0 / (1.0 + math.exp(k * (beta - b0)))
    else:
        weight = 1.0 - beta
        mode = "linear"

    return float(np.clip(weight, 0.0, 1.0))


class SimulationEngine:
    def __init__(self, user_features, global_params, simulation_days=7):
        self.user_features = user_features.copy()
        self.global_params = global_params
        self.simulation_days = simulation_days
        self.time_slots = global_params['scene_constraints']['time_slots']
        self.funnel = global_params['funnel']
        self.reference_means = global_params['reference_means']
        self.coupon_catalog = global_params['coupon_catalog']
        self.time_profile = global_params['time_profile']
        self.causal_discount_mode = str(global_params.get("causal_discount_mode", "linear"))
        self.causal_discount_params = global_params.get("causal_discount_params", {}) or {}
        self.coupon_response_mode = str(global_params.get("coupon_response_mode", "p3_only"))
        self.coupon_response_params = global_params.get("coupon_response_params", {}) or {}

        self.p1_intercept = _safe_logit(self.funnel['p1'])
        self.p2_intercept = _safe_logit(self.funnel['p2'])
        self.p3_intercept = _safe_logit(self.funnel['p3'])

    def coupon_metadata(self, coupon):
        """Return normalized coupon properties for one source-schema label."""
        if coupon in self.coupon_catalog:
            return self.coupon_catalog[coupon]

        if not coupon:
            return {'coupon_value': 0.0, 'free': 0.0, 'expand': 0.0, 'uplift': 0.0, 'attraction': 1.0}

        match = re.search(r'(\d+)\s*元', str(coupon))
        coupon_value = float(match.group(1)) if match else 0.0
        return {
            'coupon_value': coupon_value,
            'free': 1.0 if '免费' in str(coupon) else 0.0,
            'expand': 1.0 if '膨胀' in str(coupon) and '非膨胀' not in str(coupon) else 0.0,
            'uplift': max(1.0, coupon_value * 0.45),
            'attraction': 1.0
        }

    def _posterior_sensitivity(self, user, sample=False, rng=None):
        alpha = float(user['S_alpha_post'])
        beta = float(user['S_beta_post'])
        if sample and rng is not None:
            return float(rng.beta(alpha, beta))
        return float(alpha / (alpha + beta))

    def _time_factor(self, time_slot):
        return float(self.time_profile.get(time_slot, 1.0))

    def _apply_coupon_stage_response(self, p1, p2, p3_base, V, delta, coupon=None):
        mode = str(self.coupon_response_mode or "p3_only")
        params = self.coupon_response_params or {}

        if not coupon:
            p1_coupon = float(np.clip(p1, 1e-4, 0.995))
            p2_coupon = float(np.clip(p2, 1e-4, 0.995))
            p3_coupon = float(np.clip(p3_base, 1e-4, 0.995))
            return {
                "p1_coupon": p1_coupon,
                "p2_coupon": p2_coupon,
                "p3_coupon": p3_coupon,
                "effective_V": float(V),
                "delta_p1": 0.0,
                "delta_p2": 0.0,
                "delta_p3": 0.0,
                "delta": 0.0,
                "stage_response_mode": mode,
            }

        if mode == "p2_p3":
            p2_scale = float(params.get("p2_scale", 0.35))
            delta_p1 = 0.0
            delta_p2 = p2_scale * delta * (1.0 - p2)
            delta_p3 = delta
            effective_V = float(V)
        elif mode == "p1_p2_p3_v":
            p1_scale = float(params.get("p1_scale", 0.20))
            p2_scale = float(params.get("p2_scale", 0.40))
            p3_scale = float(params.get("p3_scale", 1.00))
            v_scale = float(params.get("v_scale", 0.50))
            delta_p1 = p1_scale * delta * (1.0 - p1)
            delta_p2 = p2_scale * delta * (1.0 - p2)
            delta_p3 = p3_scale * delta
            effective_V = float(V * (1.0 + v_scale * max(delta, 0.0)))
        else:
            delta_p1 = 0.0
            delta_p2 = 0.0
            delta_p3 = delta
            effective_V = float(V)
            mode = "p3_only"

        p1_coupon = float(np.clip(p1 + delta_p1, 1e-4, 0.995))
        p2_coupon = float(np.clip(p2 + delta_p2, 1e-4, 0.995))
        p3_coupon = float(np.clip(p3_base + delta_p3, 1e-4, 0.995))

        return {
            "p1_coupon": p1_coupon,
            "p2_coupon": p2_coupon,
            "p3_coupon": p3_coupon,
            "effective_V": float(max(effective_V, 1e-3)),
            "delta_p1": float(delta_p1),
            "delta_p2": float(delta_p2),
            "delta_p3": float(delta_p3),
            "delta": float(delta_p3),
            "stage_response_mode": mode,
        }

    def funnel_response(self, user, coupon=None, time_slot="15:00", sample_bayesian=False, rng=None):
        coupon_meta = self.coupon_metadata(coupon)
        V = float(user['V'])
        A = float(user['A'])
        S = self._posterior_sensitivity(user, sample=sample_bayesian, rng=rng)

        z_v = np.log(max(V, 1e-3) / max(self.reference_means['value'], 1e-3))
        z_a = np.log1p(A) - np.log1p(max(self.reference_means['activity'], 1e-3))
        z_s = S - self.reference_means['sensitivity']
        z_t = np.log(max(self._time_factor(time_slot), 1e-3))

        p1 = _sigmoid(self.p1_intercept + 0.45 * z_v + 0.18 * z_a + 0.30 * z_t)
        p2 = _sigmoid(self.p2_intercept + 0.72 * z_a + 0.10 * z_v)
        p3_base = _sigmoid(self.p3_intercept + 0.22 * z_v - 1.15 * z_s)

        coupon_value = coupon_meta['coupon_value']
        delta = 0.0
        if coupon:
            coupon_fit = 1.0
            if self.global_params.get("heterogeneous_coupon_fit", False):
                # Scenario assumption for personalization: users respond most to
                # a coupon-to-basket ratio near a sensitivity-dependent target.
                # This is not estimated as a causal effect from the source data.
                actual_ratio = coupon_value / max(V, 1.0)
                target_ratio = float(np.clip(0.06 + 0.22 * S, 0.05, 0.30))
                fit_sigma = float(self.global_params.get("coupon_fit_sigma", 0.08))
                coupon_fit = math.exp(-0.5 * ((actual_ratio - target_ratio) / fit_sigma) ** 2)
            delta = (
                self.global_params.get('delta_scale', 1.0)
                * S
                * np.log1p(coupon_value / max(V, 1.0))
                * (1 - p3_base)
                * (1 + 0.10 * coupon_meta['free'] + 0.08 * coupon_meta['expand'])
                * coupon_meta.get('attraction', 1.0)
                * coupon_fit
            )
        stage_response = self._apply_coupon_stage_response(
            p1=float(p1),
            p2=float(p2),
            p3_base=float(p3_base),
            V=V,
            delta=float(delta),
            coupon=coupon,
        )
        p1_coupon = stage_response["p1_coupon"]
        p2_coupon = stage_response["p2_coupon"]
        p3_coupon = stage_response["p3_coupon"]
        effective_V = stage_response["effective_V"]

        beta = float(np.clip(p1 * p2 * p3_base, 1e-5, 0.995))
        prob_no_coupon = float(np.clip(p1 * p2 * p3_base, 0.0, 0.995))
        prob_with_coupon = float(np.clip(p1_coupon * p2_coupon * p3_coupon, 0.0, 0.995))
        uplift = float(coupon_meta.get('uplift', 0.0))
        gtv_no_coupon = prob_no_coupon * V
        gtv_with_coupon = prob_with_coupon * (effective_V + uplift)
        delta_gtv = gtv_with_coupon - gtv_no_coupon
        # Historical compatibility key: model-adjusted incremental GTV after
        # discounting natural conversion. This is not a causal estimate.
        causal_discount_weight = compute_causal_discount(
            beta=beta,
            mode=self.causal_discount_mode,
            params=self.causal_discount_params,
        )
        delta_gtv_causal = delta_gtv * causal_discount_weight
        expected_cost = prob_with_coupon * coupon_value

        return {
            'p1': float(p1),
            'p2': float(p2),
            'p3_base': float(p3_base),
            'p1_coupon': float(p1_coupon),
            'p2_coupon': float(p2_coupon),
            'p3_coupon': float(p3_coupon),
            'beta': beta,
            'delta': float(stage_response['delta']),
            'delta_p1': float(stage_response['delta_p1']),
            'delta_p2': float(stage_response['delta_p2']),
            'delta_p3': float(stage_response['delta_p3']),
            'prob_no_coupon': prob_no_coupon,
            'prob_with_coupon': prob_with_coupon,
            'gtv_no_coupon': float(gtv_no_coupon),
            'gtv_with_coupon': float(gtv_with_coupon),
            'delta_gtv': float(delta_gtv),
            'delta_gtv_causal': float(delta_gtv_causal),
            'cost': float(expected_cost),
            'uplift': uplift,
            'coupon_value': coupon_value,
            'effective_V': float(effective_V),
            'stage_response_mode': stage_response['stage_response_mode'],
            'causal_discount_mode': str(self.causal_discount_mode),
            'causal_discount_weight': float(causal_discount_weight),
            'coupon_fit': float(coupon_fit if coupon else 1.0),
        }

    def _aggregate_user_metrics(self, user, coupon=None, stochastic=False, rng=None):
        total = {
            'gtv_no_coupon': 0.0,
            'gtv_with_coupon': 0.0,
            'delta_gtv': 0.0,
            'delta_gtv_causal': 0.0,
            'total_cost': 0.0,
            'expected_orders_no_coupon': 0.0,
            'expected_orders_with_coupon': 0.0,
            'avg_p1': [],
            'avg_p2': [],
            'avg_p3': [],
            'avg_p3_base': [],
            'avg_p3_coupon': [],
            'avg_p1_coupon': [],
            'avg_p2_coupon': [],
            'avg_beta': [],
            'avg_causal_discount_weight': [],
            'avg_delta_p1': [],
            'avg_delta_p2': [],
            'avg_delta_p3': [],
            'avg_delta': []
        }

        for _ in range(self.simulation_days):
            for slot in self.time_slots:
                response = self.funnel_response(user, coupon=coupon, time_slot=slot, sample_bayesian=stochastic, rng=rng)
                total['avg_p1'].append(response['p1'])
                total['avg_p2'].append(response['p2'])
                total['avg_p3_base'].append(response['p3_base'])
                total['avg_p3_coupon'].append(response['p3_coupon'])
                total['avg_p1_coupon'].append(response['p1_coupon'])
                total['avg_p2_coupon'].append(response['p2_coupon'])
                total['avg_p3'].append(response['p3_coupon'] if coupon else response['p3_base'])
                total['avg_beta'].append(response['beta'])
                total['avg_causal_discount_weight'].append(response['causal_discount_weight'])
                total['avg_delta_p1'].append(response['delta_p1'])
                total['avg_delta_p2'].append(response['delta_p2'])
                total['avg_delta_p3'].append(response['delta_p3'])
                total['avg_delta'].append(response['delta'])

                if stochastic and rng is not None:
                    order_no = float(rng.random() < response['prob_no_coupon'])
                    order_yes = float(rng.random() < response['prob_with_coupon'])
                    total['gtv_no_coupon'] += order_no * float(user['V'])
                    total['gtv_with_coupon'] += order_yes * (response['effective_V'] + response['uplift'])
                    total['total_cost'] += order_yes * response['coupon_value']
                    total['expected_orders_no_coupon'] += order_no
                    total['expected_orders_with_coupon'] += order_yes
                else:
                    total['gtv_no_coupon'] += response['gtv_no_coupon']
                    total['gtv_with_coupon'] += response['gtv_with_coupon']
                    total['total_cost'] += response['cost']
                    total['expected_orders_no_coupon'] += response['prob_no_coupon']
                    total['expected_orders_with_coupon'] += response['prob_with_coupon']

                total['delta_gtv'] = total['gtv_with_coupon'] - total['gtv_no_coupon']
                if stochastic:
                    realized_delta = (order_yes * (response['effective_V'] + response['uplift'])
                                      - order_no * float(user['V']))
                    total['delta_gtv_causal'] += realized_delta * response['causal_discount_weight']
                else:
                    total['delta_gtv_causal'] += response['delta_gtv_causal']

        total['avg_p1'] = float(np.mean(total['avg_p1'])) if total['avg_p1'] else 0.0
        total['avg_p2'] = float(np.mean(total['avg_p2'])) if total['avg_p2'] else 0.0
        total['avg_p3'] = float(np.mean(total['avg_p3'])) if total['avg_p3'] else 0.0
        total['avg_p3_base'] = float(np.mean(total['avg_p3_base'])) if total['avg_p3_base'] else 0.0
        total['avg_p3_coupon'] = float(np.mean(total['avg_p3_coupon'])) if total['avg_p3_coupon'] else 0.0
        total['avg_p1_coupon'] = float(np.mean(total['avg_p1_coupon'])) if total['avg_p1_coupon'] else 0.0
        total['avg_p2_coupon'] = float(np.mean(total['avg_p2_coupon'])) if total['avg_p2_coupon'] else 0.0
        total['avg_beta'] = float(np.mean(total['avg_beta'])) if total['avg_beta'] else 0.0
        total['avg_causal_discount_weight'] = float(np.mean(total['avg_causal_discount_weight'])) if total['avg_causal_discount_weight'] else 0.0
        total['avg_delta_p1'] = float(np.mean(total['avg_delta_p1'])) if total['avg_delta_p1'] else 0.0
        total['avg_delta_p2'] = float(np.mean(total['avg_delta_p2'])) if total['avg_delta_p2'] else 0.0
        total['avg_delta_p3'] = float(np.mean(total['avg_delta_p3'])) if total['avg_delta_p3'] else 0.0
        total['avg_delta'] = float(np.mean(total['avg_delta'])) if total['avg_delta'] else 0.0
        total['causal_discount_mode'] = str(self.causal_discount_mode)
        total['coupon_response_mode'] = str(self.coupon_response_mode)
        return total

    def calculate_expected_metrics(self, coupon_strategy):
        total = {
            'gtv_no_coupon': 0.0,
            'gtv_with_coupon': 0.0,
            'delta_gtv': 0.0,
            'delta_gtv_causal': 0.0,
            'total_cost': 0.0,
            'orders_no_coupon': 0.0,
            'orders_with_coupon': 0.0,
            'avg_p1': [],
            'avg_p2': [],
            'avg_p3': [],
            'avg_p3_base': [],
            'avg_p3_coupon': [],
            'avg_p1_coupon': [],
            'avg_p2_coupon': [],
            'avg_beta': [],
            'avg_causal_discount_weight': [],
            'avg_delta_p1': [],
            'avg_delta_p2': [],
            'avg_delta_p3': [],
            'avg_delta': []
        }

        if coupon_strategy is None:
            coupon_strategy = {user_id: None for user_id in self.user_features['user_id']}

        target_ids = set(coupon_strategy.keys())
        for _, user in self.user_features.iterrows():
            if user['user_id'] not in target_ids:
                continue
            coupon = coupon_strategy.get(user['user_id'])
            metrics = self._aggregate_user_metrics(user, coupon=coupon, stochastic=False)
            total['gtv_no_coupon'] += metrics['gtv_no_coupon']
            total['gtv_with_coupon'] += metrics['gtv_with_coupon']
            total['delta_gtv_causal'] += metrics['delta_gtv_causal']
            total['total_cost'] += metrics['total_cost']
            total['orders_no_coupon'] += metrics['expected_orders_no_coupon']
            total['orders_with_coupon'] += metrics['expected_orders_with_coupon']
            total['avg_p1'].append(metrics['avg_p1'])
            total['avg_p2'].append(metrics['avg_p2'])
            total['avg_p3'].append(metrics['avg_p3'])
            total['avg_p3_base'].append(metrics['avg_p3_base'])
            total['avg_p3_coupon'].append(metrics['avg_p3_coupon'])
            total['avg_p1_coupon'].append(metrics['avg_p1_coupon'])
            total['avg_p2_coupon'].append(metrics['avg_p2_coupon'])
            total['avg_beta'].append(metrics['avg_beta'])
            total['avg_causal_discount_weight'].append(metrics['avg_causal_discount_weight'])
            total['avg_delta_p1'].append(metrics['avg_delta_p1'])
            total['avg_delta_p2'].append(metrics['avg_delta_p2'])
            total['avg_delta_p3'].append(metrics['avg_delta_p3'])
            total['avg_delta'].append(metrics['avg_delta'])

        total['delta_gtv'] = total['gtv_with_coupon'] - total['gtv_no_coupon']
        roi = total['delta_gtv_causal'] / total['total_cost'] if total['total_cost'] > 0 else 0.0
        delta_percentage = (total['delta_gtv_causal'] / total['gtv_no_coupon']) * 100 if total['gtv_no_coupon'] > 0 else 0.0

        return {
            'gtv_no_coupon': float(total['gtv_no_coupon']),
            'gtv_with_coupon': float(total['gtv_with_coupon']),
            'delta_gtv': float(total['delta_gtv']),
            'delta_gtv_causal': float(total['delta_gtv_causal']),
            'total_cost': float(total['total_cost']),
            'roi': float(roi),
            'delta_percentage': float(delta_percentage),
            'orders_no_coupon': float(total['orders_no_coupon']),
            'orders_with_coupon': float(total['orders_with_coupon']),
            'avg_p1': float(np.mean(total['avg_p1'])) if total['avg_p1'] else 0.0,
            'avg_p2': float(np.mean(total['avg_p2'])) if total['avg_p2'] else 0.0,
            'avg_p3': float(np.mean(total['avg_p3'])) if total['avg_p3'] else 0.0,
            'avg_p3_base': float(np.mean(total['avg_p3_base'])) if total['avg_p3_base'] else 0.0,
            'avg_p3_coupon': float(np.mean(total['avg_p3_coupon'])) if total['avg_p3_coupon'] else 0.0,
            'avg_p1_coupon': float(np.mean(total['avg_p1_coupon'])) if total['avg_p1_coupon'] else 0.0,
            'avg_p2_coupon': float(np.mean(total['avg_p2_coupon'])) if total['avg_p2_coupon'] else 0.0,
            'avg_beta': float(np.mean(total['avg_beta'])) if total['avg_beta'] else 0.0,
            'avg_causal_discount_weight': float(np.mean(total['avg_causal_discount_weight'])) if total['avg_causal_discount_weight'] else 0.0,
            'avg_delta_p1': float(np.mean(total['avg_delta_p1'])) if total['avg_delta_p1'] else 0.0,
            'avg_delta_p2': float(np.mean(total['avg_delta_p2'])) if total['avg_delta_p2'] else 0.0,
            'avg_delta_p3': float(np.mean(total['avg_delta_p3'])) if total['avg_delta_p3'] else 0.0,
            'avg_delta': float(np.mean(total['avg_delta'])) if total['avg_delta'] else 0.0,
            'causal_discount_mode': str(self.causal_discount_mode),
            'coupon_response_mode': str(self.coupon_response_mode),
        }

    def calculate_simulated_metrics(self, coupon_strategy, seed=None):
        rng = np.random.default_rng(seed)
        total = {
            'gtv_no_coupon': 0.0,
            'gtv_with_coupon': 0.0,
            'delta_gtv_causal': 0.0,
            'total_cost': 0.0,
            'orders_no_coupon': 0.0,
            'orders_with_coupon': 0.0
        }

        if coupon_strategy is None:
            coupon_strategy = {user_id: None for user_id in self.user_features['user_id']}

        target_ids = set(coupon_strategy.keys())
        for _, user in self.user_features.iterrows():
            if user['user_id'] not in target_ids:
                continue
            coupon = coupon_strategy.get(user['user_id'])
            metrics = self._aggregate_user_metrics(user, coupon=coupon, stochastic=True, rng=rng)
            total['gtv_no_coupon'] += metrics['gtv_no_coupon']
            total['gtv_with_coupon'] += metrics['gtv_with_coupon']
            total['delta_gtv_causal'] += metrics['delta_gtv_causal']
            total['total_cost'] += metrics['total_cost']
            total['orders_no_coupon'] += metrics['expected_orders_no_coupon']
            total['orders_with_coupon'] += metrics['expected_orders_with_coupon']

        total['delta_gtv'] = total['gtv_with_coupon'] - total['gtv_no_coupon']
        roi = total['delta_gtv_causal'] / total['total_cost'] if total['total_cost'] > 0 else 0.0
        delta_percentage = (total['delta_gtv_causal'] / total['gtv_no_coupon']) * 100 if total['gtv_no_coupon'] > 0 else 0.0

        return {
            'gtv_no_coupon': float(total['gtv_no_coupon']),
            'gtv_with_coupon': float(total['gtv_with_coupon']),
            'delta_gtv': float(total['delta_gtv']),
            'delta_gtv_causal': float(total['delta_gtv_causal']),
            'total_cost': float(total['total_cost']),
            'roi': float(roi),
            'delta_percentage': float(delta_percentage),
            'orders_no_coupon': float(total['orders_no_coupon']),
            'orders_with_coupon': float(total['orders_with_coupon']),
            'seed': int(seed if seed is not None else 0)
        }

    def evaluate_single_coupon(self, coupon):
        strategy = {user_id: coupon for user_id in self.user_features['user_id']}
        return self.calculate_expected_metrics(strategy)

    def user_coupon_metrics(self, user, coupon):
        return self._aggregate_user_metrics(user, coupon=coupon, stochastic=False)

    def monte_carlo_runs(self, coupon_strategy, n_runs=50, seed=2026):
        rows = []
        for i in range(n_runs):
            run_seed = int(seed + i)
            metrics = self.calculate_simulated_metrics(coupon_strategy, seed=run_seed)
            rows.append({
                'run': i + 1,
                'seed': run_seed,
                'delta_gtv_causal': metrics['delta_gtv_causal'],
                'delta_gtv': metrics['delta_gtv'],
                'roi': metrics['roi'],
                'total_cost': metrics['total_cost']
            })
        return rows
