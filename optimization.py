import copy
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from simulation_engine import SimulationEngine


@dataclass(frozen=True)
class ProgressEvent:
    """Framework-neutral progress message emitted by long-running operations."""

    stage: str
    completed: int
    total: int
    message: str

    @property
    def fraction(self):
        return 1.0 if self.total <= 0 else min(1.0, self.completed / self.total)


class Optimization:
    def __init__(
        self,
        user_features,
        global_params,
        order_df=None,
        coupon_cap_share=0.35,
        alpha_ucb=0.25,
        progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
    ):
        self.base_user_features = user_features.copy()
        self.user_features = user_features.copy()
        self.global_params = global_params
        self.order_df = order_df
        self.simulation_engine = SimulationEngine(self.user_features, global_params)
        self.candidate_coupons = list(global_params.get('candidate_coupons', []))
        self.coupon_cap_share = coupon_cap_share
        self.alpha_ucb = alpha_ucb
        self.progress_callback = progress_callback
        self._validate_configuration()

    def _validate_configuration(self):
        """Reject invalid constraints before an allocation run starts."""
        required_columns = {
            'user_id', 'V', 'A', 'S_alpha_post', 'S_beta_post',
            'exposure', 'successes', 'failures'
        }
        missing = required_columns.difference(self.user_features.columns)
        if missing:
            raise ValueError(f"user_features is missing required columns: {sorted(missing)}")
        if self.user_features['user_id'].duplicated().any():
            raise ValueError("user_id values must be unique")
        if not 0 < float(self.coupon_cap_share) <= 1:
            raise ValueError("coupon_cap_share must be in (0, 1]")
        if float(self.alpha_ucb) < 0:
            raise ValueError("alpha_ucb must be non-negative")
        if not self.candidate_coupons:
            raise ValueError("global_params must define at least one candidate coupon")

    def _report_progress(self, stage, completed, total, message):
        if self.progress_callback is not None:
            self.progress_callback(ProgressEvent(stage, completed, total, message))

    def reset_state(self):
        self.user_features = self.base_user_features.copy()
        self.simulation_engine = SimulationEngine(self.user_features, self.global_params)

    def _state_vector(self, user):
        return {
            'V': float(user['V']),
            'A': float(user['A']),
            'S': float(user['S_alpha_post'] / (user['S_alpha_post'] + user['S_beta_post']))
        }

    def _uncertainty(self, user):
        return 1 / np.sqrt(1 + float(user['exposure']))

    def precompute_metrics(self, show_progress=True):
        metrics = {}
        total_tasks = len(self.user_features) * len(self.candidate_coupons)
        done = 0
        if show_progress:
            self._report_progress("precompute", 0, total_tasks, "Precomputing user-coupon outcomes")

        for row in self.user_features.itertuples():
            user_id = row.user_id
            user_metrics = {None: {'value': 0.0, 'cost': 0.0, 'roi': 0.0, 'uncertainty': self._uncertainty(pd.Series(row._asdict()))}}
            user_df = pd.Series(row._asdict())

            for coupon in self.candidate_coupons:
                response = self.simulation_engine.user_coupon_metrics(user_df, coupon)
                uncertainty = self._uncertainty(user_df)
                value = response['delta_gtv_causal']
                cost = response['total_cost']
                roi = value / cost if cost > 0 else 0.0
                user_metrics[coupon] = {
                    'value': float(value),
                    'cost': float(cost),
                    'roi': float(roi),
                    'value_ucb': float(value + self.alpha_ucb * uncertainty),
                    'uncertainty': float(uncertainty),
                    'avg_beta': float(response['avg_beta']),
                    'avg_delta': float(response['avg_delta']),
                    'avg_p1': float(response['avg_p1']),
                    'avg_p2': float(response['avg_p2']),
                    'avg_p3': float(response['avg_p3'])
                }
                done += 1
                if show_progress and total_tasks > 0 and (done % max(1, total_tasks // 25) == 0 or done == total_tasks):
                    self._report_progress("precompute", done, total_tasks, "Precomputing user-coupon outcomes")

            metrics[user_id] = user_metrics

        if show_progress:
            self._report_progress("precompute", total_tasks, total_tasks, "Precomputation complete")
        return metrics

    def _greedy_allocate(self, budget, metrics, use_ucb=True):
        allocation = {user_id: None for user_id in metrics.keys()}
        remaining_budget = float(budget)
        cap = max(1, int(len(metrics) * self.coupon_cap_share))
        coupon_usage = {coupon: 0 for coupon in self.candidate_coupons}

        candidates = []
        for _, user in self.user_features.iterrows():
            user_id = user['user_id']
            for coupon in self.candidate_coupons:
                item = metrics[user_id][coupon]
                value_for_rank = item['value_ucb'] if use_ucb else item['value']
                score = value_for_rank / max(item['cost'], 1e-6)
                candidates.append({
                    'user_id': user_id,
                    'coupon': coupon,
                    'value': item['value'],
                    'value_ucb': item['value_ucb'],
                    'cost': item['cost'],
                    'roi': item['roi'],
                    'score': score,
                    'uncertainty': item['uncertainty']
                })

        candidate_df = pd.DataFrame(candidates)
        candidate_df = candidate_df[candidate_df['cost'] > 0].sort_values(
            by=['score', 'value_ucb', 'roi'], ascending=False
        )

        selected_rows = []
        for row in candidate_df.itertuples():
            if allocation[row.user_id] is not None:
                continue
            if coupon_usage[row.coupon] >= cap:
                continue
            if row.cost > remaining_budget:
                continue
            if row.value_ucb <= 0:
                continue

            allocation[row.user_id] = row.coupon
            coupon_usage[row.coupon] += 1
            remaining_budget -= row.cost
            selected_rows.append(row._asdict())

        return allocation, pd.DataFrame(selected_rows), coupon_usage

    def _update_simulated_beliefs(self, allocation_df):
        """Update synthetic sensitivity beliefs from model-implied responses.

        This is not an online bandit update: no observed production reward enters
        this method.  The explicit name prevents simulated iteration from being
        mistaken for evidence learned from deployed decisions.
        """
        if allocation_df is None or len(allocation_df) == 0:
            return

        update_map = allocation_df.set_index('user_id')[['coupon', 'value', 'cost']].to_dict(orient='index')
        for idx, row in self.user_features.iterrows():
            user_id = row['user_id']
            if user_id not in update_map:
                continue
            coupon = update_map[user_id]['coupon']
            response = self.simulation_engine.user_coupon_metrics(row, coupon)
            success = float(response['expected_orders_with_coupon'] / max(self.simulation_engine.simulation_days * len(self.simulation_engine.time_slots), 1))
            failure = max(0.0, 1.0 - success)
            self.user_features.at[idx, 'exposure'] = float(row['exposure']) + 1
            self.user_features.at[idx, 'successes'] = float(row['successes']) + success
            self.user_features.at[idx, 'failures'] = float(row['failures']) + failure
            self.user_features.at[idx, 'S_alpha_post'] = float(row['S_alpha_post']) + success
            self.user_features.at[idx, 'S_beta_post'] = float(row['S_beta_post']) + failure

        self.simulation_engine = SimulationEngine(self.user_features, self.global_params)

    def _uniform_budget_allocation(self, metrics, budget, coupon, rank_by='roi'):
        """
        Build a fair uniform-coupon baseline under the same budget/cap constraints.
        rank_by supports: roi / value / score(value/cost)
        """
        allocation = {user_id: None for user_id in metrics.keys()}
        remaining_budget = float(budget)
        cap = max(1, int(len(metrics) * self.coupon_cap_share))
        rows = []
        for user_id in metrics.keys():
            item = metrics[user_id].get(coupon, {})
            cost = float(item.get('cost', 0.0))
            value = float(item.get('value', 0.0))
            roi = float(item.get('roi', 0.0))
            score = value / max(cost, 1e-6) if cost > 0 else 0.0
            rows.append({
                'user_id': user_id,
                'coupon': coupon,
                'cost': cost,
                'value': value,
                'roi': roi,
                'score': score
            })
        df = pd.DataFrame(rows)
        df = df[df['cost'] > 0]
        if rank_by == 'value':
            df = df.sort_values(['value', 'roi'], ascending=False)
        elif rank_by == 'score':
            df = df.sort_values(['score', 'value'], ascending=False)
        else:
            df = df.sort_values(['roi', 'value'], ascending=False)

        usage = 0
        for row in df.itertuples():
            if usage >= cap:
                break
            if row.cost > remaining_budget:
                continue
            if row.value <= 0:
                continue
            allocation[row.user_id] = coupon
            usage += 1
            remaining_budget -= row.cost
        return allocation

    def run_iterative_ucb_allocation(self, budget, epochs=3, show_progress=True):
        """Repeat UCB-style ranking with model-implied belief updates."""
        if float(budget) < 0:
            raise ValueError("budget must be non-negative")
        if int(epochs) < 1:
            raise ValueError("epochs must be at least 1")
        history = []
        final_metrics = None
        final_allocation = None
        final_selected = None
        final_precomputed = None
        final_usage = None

        for epoch in range(epochs):
            if show_progress:
                self._report_progress(
                    "allocation", epoch, epochs, f"Running allocation round {epoch + 1}/{epochs}"
                )

            precomputed = self.precompute_metrics(show_progress=False)
            allocation, selected_df, coupon_usage = self._greedy_allocate(budget, precomputed, use_ucb=True)
            metrics = self.simulation_engine.calculate_expected_metrics(allocation)
            history.append({
                'epoch': epoch + 1,
                'delta_gtv_causal': metrics['delta_gtv_causal'],
                'roi': metrics['roi'],
                'total_cost': metrics['total_cost'],
                'covered_users': int((pd.Series(allocation).notna()).sum())
            })
            self._update_simulated_beliefs(selected_df)
            final_metrics = metrics
            final_allocation = allocation
            final_selected = selected_df
            final_precomputed = precomputed
            final_usage = coupon_usage

        if show_progress:
            self._report_progress("allocation", epochs, epochs, "Allocation rounds complete")

        return {
            'allocation': final_allocation,
            'selected_df': final_selected if final_selected is not None else pd.DataFrame(columns=['user_id', 'coupon']),
            'metrics': final_precomputed,
            'coupon_usage': final_usage or {},
            'history': pd.DataFrame(history),
            'optimized_metrics': final_metrics or self.simulation_engine.calculate_expected_metrics({}),
            'final_user_pool': self.user_features.copy()
        }

    def optimize(self, budget, epochs=3, show_progress=True):
        self.reset_state()
        result = self.run_iterative_ucb_allocation(budget, epochs=epochs, show_progress=show_progress)
        allocation = result['allocation']
        optimized_metrics = result['optimized_metrics']
        selected_df = result['selected_df'].copy()
        if len(selected_df) == 0:
            selected_df = pd.DataFrame(columns=['user_id', 'coupon', 'value', 'cost', 'roi', 'score', 'uncertainty'])

        strategies = {'不发券': self.simulation_engine.calculate_expected_metrics({user_id: None for user_id in self.user_features['user_id']})}
        # Fair baseline: compare under same budget/cap, not full-population uniform send
        precomputed = result.get('metrics') or self.precompute_metrics(show_progress=False)
        for coupon in self.candidate_coupons[: min(4, len(self.candidate_coupons))]:
            alloc_roi = self._uniform_budget_allocation(precomputed, budget=float(budget), coupon=coupon, rank_by='roi')
            alloc_gtv = self._uniform_budget_allocation(precomputed, budget=float(budget), coupon=coupon, rank_by='value')
            strategies[f'统一发{coupon}(同预算ROI)'] = self.simulation_engine.calculate_expected_metrics(alloc_roi)
            strategies[f'统一发{coupon}(同预算GTV)'] = self.simulation_engine.calculate_expected_metrics(alloc_gtv)
        strategies['迭代UCB分配'] = optimized_metrics

        return {
            'allocation': allocation,
            'allocation_df': selected_df,
            'optimized_metrics': optimized_metrics,
            'metrics': result['metrics'],
            'coupon_usage': result['coupon_usage'],
            'history': result['history'],
            'strategies': strategies,
            'constraints': {
                'coupon_cap_share': self.coupon_cap_share,
                'alpha_ucb': self.alpha_ucb,
                'epochs': epochs
            },
            'final_user_pool': result['final_user_pool']
        }

    def sensitivity_analysis(self, budget, epochs=2):
        scenarios = [
            ('基线', 1.0, 1.0, 1.0),
            ('高价值提升', 1.1, 1.0, 1.0),
            ('强券响应', 1.0, 1.15, 1.0),
            ('更保守场景', 1.0, 0.9, 0.9)
        ]
        rows = []
        for name, value_scale, delta_scale, funnel_scale in scenarios:
            modified_params = copy.deepcopy(self.global_params)
            modified_params['reference_means']['value'] *= value_scale
            modified_params['delta_scale'] *= delta_scale
            modified_params['funnel'] = {
                'p1': float(np.clip(modified_params['funnel']['p1'] * funnel_scale, 0.03, 0.95)),
                'p2': float(np.clip(modified_params['funnel']['p2'] * funnel_scale, 0.03, 0.95)),
                'p3': float(np.clip(modified_params['funnel']['p3'] * funnel_scale, 0.03, 0.95)),
                'hesitation': modified_params['funnel']['hesitation'],
                'impulse': modified_params['funnel']['impulse'],
                'scene_sessions': modified_params['funnel']['scene_sessions']
            }
            optimizer = Optimization(
                self.base_user_features.copy(),
                modified_params,
                self.order_df,
                coupon_cap_share=self.coupon_cap_share,
                alpha_ucb=self.alpha_ucb
            )
            result = optimizer.optimize(budget, epochs=epochs, show_progress=False)
            rows.append({
                'scenario': name,
                'delta_gtv_causal': result['optimized_metrics']['delta_gtv_causal'],
                'roi': result['optimized_metrics']['roi'],
                'covered_users': int((pd.Series(result['allocation']).notna()).sum())
            })
        return pd.DataFrame(rows)

    def budget_frontier(self, budget_list, epochs=2):
        rows = []
        for budget in budget_list:
            result = self.optimize(budget=float(budget), epochs=epochs, show_progress=False)
            covered = int((pd.Series(result['allocation']).notna()).sum())
            rows.append({
                'budget': float(budget),
                'delta_gtv_causal': result['optimized_metrics']['delta_gtv_causal'],
                'roi': result['optimized_metrics']['roi'],
                'covered_users': covered
            })
        return pd.DataFrame(rows)

    def sensitivity_heatmap(self, budget, beta_scales=None, delta_scales=None, epochs=2):
        if beta_scales is None:
            beta_scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        if delta_scales is None:
            delta_scales = [0.8, 0.9, 1.0, 1.1, 1.2]

        rows = []
        for beta_scale in beta_scales:
            for delta_scale in delta_scales:
                modified_params = copy.deepcopy(self.global_params)
                modified_params['delta_scale'] *= float(delta_scale)
                modified_params['funnel'] = {
                    'p1': modified_params['funnel']['p1'],
                    'p2': modified_params['funnel']['p2'],
                    'p3': float(np.clip(modified_params['funnel']['p3'] * beta_scale, 0.03, 0.95)),
                    'hesitation': modified_params['funnel']['hesitation'],
                    'impulse': modified_params['funnel']['impulse'],
                    'scene_sessions': modified_params['funnel']['scene_sessions']
                }
                optimizer = Optimization(
                    self.base_user_features.copy(),
                    modified_params,
                    self.order_df,
                    coupon_cap_share=self.coupon_cap_share,
                    alpha_ucb=self.alpha_ucb
                )
                result = optimizer.optimize(budget=budget, epochs=epochs, show_progress=False)
                rows.append({
                    'beta_scale': float(beta_scale),
                    'delta_scale': float(delta_scale),
                    'delta_gtv_causal': result['optimized_metrics']['delta_gtv_causal'],
                    'roi': result['optimized_metrics']['roi']
                })
        return pd.DataFrame(rows)
