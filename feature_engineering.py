import logging
import math
import os
import re

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from utils import load_excel_file


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


SCENE_BU_KEYWORDS = ['饮品', '甜点', '水果', '酒水茶饮']
SCENE_POI_KEYWORDS = ['饮品', '奶茶', '咖啡', '生日蛋糕', '西式点心', '果切店', '整果店', '酒水店']
SCENE_EXCLUDE_BU_KEYWORDS = ['美食', '餐饮', '超市便利', '休闲食品', '食材']
SCENE_EXCLUDE_POI_KEYWORDS = [
    '快餐简餐', '中式正餐', '小吃', '小吃快餐', '火锅', '川菜', '粤菜',
    '烤串烤肉', '特色菜', '全球美食', '大型超市/卖场', '小型超市', '便利店', '零食店', '菜市场'
]
SCENE_BEHAVIOR_KEYWORDS = [
    '奶茶', '咖啡', '酸奶', '果汁', '甜品', '甜点', '蛋糕', '饮品',
    '水果', '茉酸奶', '生日蛋糕', '果切', '面包', '冰淇淋'
]
TIME_SLOTS = ["14:00", "14:30", "15:00", "15:30", "16:00", "16:30"]


def _safe_logit(p):
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return math.log(p / (1 - p))


class FeatureEngineering:
    def __init__(self, use_api=False):
        self.use_api = use_api
        self.user_profile = None
        self.order_df = None
        self.behavior_seq = None
        self.scene_orders = None
        self.scene_profiles = None
        self.scene_constraints = None
        self.funnel_stats = None
        self.global_params = None
        self.generated_users = None
        self.calibration = None
        self.use_auto_distribution_selector = True

    def load_data(self, data_dir):
        """加载 raw 数据。"""
        candidate_dir = data_dir
        if os.path.isdir(os.path.join(data_dir, 'raw')):
            candidate_dir = os.path.join(data_dir, 'raw')

        self.user_profile = load_excel_file(os.path.join(candidate_dir, '用户画像表.xlsx'))
        self.order_df = load_excel_file(os.path.join(candidate_dir, '神券订单数据样例.xlsx'))
        self.behavior_seq = load_excel_file(os.path.join(candidate_dir, '用户行为序列.xlsx'))

        if self.user_profile is None or self.order_df is None or self.behavior_seq is None:
            return False

        logger.info("raw 数据加载成功: 用户%d, 订单%d, 行为%d", len(self.user_profile), len(self.order_df), len(self.behavior_seq))
        return True

    def build_features_and_params(self, n_users=1000, random_state=42, use_auto_distribution_selector=True):
        """主入口：基于 raw 数据构建半合成用户池与场景参数。"""
        logger.info("开始构建半合成仿真模型")
        self.use_auto_distribution_selector = bool(use_auto_distribution_selector)
        self.scene_constraints = self._build_scene_constraints()
        self.scene_orders = self._extract_scene_orders()
        self.scene_profiles = self.user_profile[self.user_profile['user_id'].isin(self.scene_orders['user_id'].unique())].copy()
        self.funnel_stats = self._extract_funnel_structure()
        self.global_params = self._fit_distributions_and_params()
        self.generated_users = self.generate_users(n=n_users, random_state=random_state)
        self.calibration = self._build_calibration_snapshot(self.generated_users)
        self.global_params['calibration'] = self.calibration
        logger.info("半合成仿真模型构建完成，生成用户数=%d", len(self.generated_users))
        return self.generated_users, self.global_params

    def _build_scene_constraints(self):
        return {
            'name': 'afternoon_tea',
            'hour_start': 14,
            'hour_end': 17,
            'time_slots': TIME_SLOTS,
            'bu_keywords': SCENE_BU_KEYWORDS,
            'poi_keywords': SCENE_POI_KEYWORDS,
            'exclude_bu_keywords': SCENE_EXCLUDE_BU_KEYWORDS,
            'exclude_poi_keywords': SCENE_EXCLUDE_POI_KEYWORDS,
            'behavior_keywords': SCENE_BEHAVIOR_KEYWORDS
        }

    def _extract_hour(self, value):
        if hasattr(value, 'hour'):
            return int(value.hour)
        if isinstance(value, str):
            try:
                return int(value.split(':')[0])
            except Exception:
                return 0
        return 0

    def _extract_scene_orders(self):
        orders = self.order_df.copy()
        orders['hour'] = orders['下单时间'].apply(self._extract_hour)
        afternoon_mask = (orders['hour'] >= self.scene_constraints['hour_start']) & (orders['hour'] < self.scene_constraints['hour_end'])
        bu_mask = orders['BU'].astype(str).str.contains('|'.join(self.scene_constraints['bu_keywords']), na=False)
        poi_mask = orders['POI分类'].astype(str).str.contains('|'.join(self.scene_constraints['poi_keywords']), na=False)
        exclude_bu_mask = orders['BU'].astype(str).str.contains('|'.join(self.scene_constraints['exclude_bu_keywords']), na=False)
        exclude_poi_mask = orders['POI分类'].astype(str).str.contains('|'.join(self.scene_constraints['exclude_poi_keywords']), na=False)
        amount_mask = orders['订单金额'].between(5, 120, inclusive='both')
        scene_orders = orders[afternoon_mask & (bu_mask | poi_mask) & (~exclude_bu_mask) & (~exclude_poi_mask) & amount_mask].copy()
        logger.info("下午茶场景订单=%d，场景用户=%d", len(scene_orders), scene_orders['user_id'].nunique())
        return scene_orders

    def _parse_behavior_timestamp(self, value):
        text = str(value)
        match = re.search(r'T-(\d+).*(\d{2}):(\d{2})', text)
        if match:
            day_offset = int(match.group(1))
            hour = int(match.group(2))
            minute = int(match.group(3))
            return pd.Timestamp('2026-01-01') - pd.Timedelta(days=day_offset) + pd.Timedelta(hours=hour, minutes=minute)
        return pd.NaT

    def _normalize_behavior_stage(self, behavior_type):
        stage_map = {
            '搜索': 'view',
            'search': 'view',
            '点击店铺': 'click',
            '点击并收藏': 'click',
            'click': 'click',
            'click&collect': 'click',
            'click&call&collect': 'click',
            '加入购物车': 'cart',
            'cart': 'cart',
            'click&cart': 'cart',
            'cart&click': 'cart',
            '点击并加购': 'cart',
            '下单': 'purchase',
            'order': 'purchase'
        }
        return stage_map.get(str(behavior_type), 'other')

    def _extract_funnel_structure(self):
        seq = self.behavior_seq.copy()
        seq['行为时间'] = seq['行为时间戳'].apply(self._parse_behavior_timestamp)
        seq['stage'] = seq['行为类型'].apply(self._normalize_behavior_stage)
        seq['scene_related'] = seq['具体内容'].astype(str).str.contains('|'.join(self.scene_constraints['behavior_keywords']), na=False)
        seq['hour'] = seq['行为时间'].dt.hour
        seq = seq[
            seq['scene_related']
            & seq['stage'].isin(['view', 'click', 'cart', 'purchase'])
            & seq['hour'].between(self.scene_constraints['hour_start'], self.scene_constraints['hour_end'] - 1, inclusive='both')
        ].copy()
        seq = seq.sort_values(['用户id', 'session_id', '行为时间'])

        session_rows = []
        hesitation_values = []
        impulse_flags = []
        for (_, session_id), group in seq.groupby(['用户id', 'session_id']):
            stages = group['stage'].tolist()
            if not stages:
                continue
            viewed = 1
            clicked = int(any(stage in ['click', 'cart', 'purchase'] for stage in stages))
            carted = int(any(stage in ['cart', 'purchase'] for stage in stages))
            purchased = int(any(stage == 'purchase' for stage in stages))
            session_rows.append({'viewed': viewed, 'clicked': clicked, 'carted': carted, 'purchased': purchased})
            hesitation_values.append(max(0, len(stages) - len(set(stages))))
            if purchased:
                purchase_idx = stages.index('purchase')
                impulse_flags.append(int(purchase_idx <= 1))

        session_df = pd.DataFrame(session_rows)
        if len(session_df) == 0:
            funnel = {'p1': 0.35, 'p2': 0.28, 'p3': 0.22}
        else:
            p1 = session_df['clicked'].sum() / max(1, session_df['viewed'].sum())
            p2 = session_df['carted'].sum() / max(1, session_df['clicked'].sum())
            p3 = session_df['purchased'].sum() / max(1, session_df['carted'].sum())
            funnel = {
                'p1': float(np.clip(p1, 0.05, 0.95)),
                'p2': float(np.clip(p2, 0.05, 0.95)),
                'p3': float(np.clip(p3, 0.05, 0.95))
            }

        hesitation = float(np.mean(hesitation_values)) if hesitation_values else 0.0
        impulse = float(np.mean(impulse_flags)) if impulse_flags else 0.25
        logger.info("Funnel结构提取完成: p1=%.3f p2=%.3f p3=%.3f", funnel['p1'], funnel['p2'], funnel['p3'])
        return {
            'p1': funnel['p1'],
            'p2': funnel['p2'],
            'p3': funnel['p3'],
            'hesitation': hesitation,
            'impulse': impulse,
            'scene_sessions': int(len(session_df))
        }

    def _fit_lognormal(self, values):
        values = np.asarray(values, dtype=float)
        values = values[values > 0]
        log_values = np.log(values)
        return {
            'mu': float(log_values.mean()),
            'sigma': float(max(log_values.std(ddof=0), 0.15))
        }

    def _fit_gamma(self, values):
        values = np.asarray(values, dtype=float)
        values = values[values > 0]
        mean = float(values.mean())
        var = float(max(values.var(ddof=0), 1e-4))
        shape = max((mean ** 2) / var, 0.5)
        scale = max(var / mean, 0.05)
        return {'shape': float(shape), 'scale': float(scale)}

    def _fit_beta(self, values):
        values = np.asarray(values, dtype=float)
        values = np.clip(values, 1e-3, 1 - 1e-3)
        mean = float(values.mean())
        var = float(max(values.var(ddof=0), 1e-4))
        concentration = max(mean * (1 - mean) / var - 1, 2.0)
        return {
            'alpha': float(max(mean * concentration, 1.1)),
            'beta': float(max((1 - mean) * concentration, 1.1))
        }

    def _detect_multimodal(self, values, bins=12):
        values = np.asarray(values, dtype=float)
        if len(values) < 120:
            return False
        hist, _ = np.histogram(values, bins=min(bins, max(6, int(np.sqrt(len(values))))))
        if hist.size < 3:
            return False
        peaks = sum(1 for i in range(1, len(hist) - 1) if hist[i] > hist[i - 1] and hist[i] > hist[i + 1])
        return peaks >= 2

    def _quantile_summary(self, values):
        values = np.asarray(values, dtype=float)
        qs = np.quantile(values, [0.25, 0.5, 0.75, 0.9])
        return {
            'q25': float(qs[0]),
            'q50': float(qs[1]),
            'q75': float(qs[2]),
            'q90': float(qs[3]),
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }

    def _distribution_shape_stats(self, values):
        values = np.asarray(values, dtype=float)
        centered = values - values.mean()
        std = float(np.std(values))
        if std < 1e-8:
            return {'skewness': 0.0, 'kurtosis': 0.0}
        z = centered / std
        skewness = float(np.mean(z ** 3))
        kurtosis = float(np.mean(z ** 4) - 3)
        return {'skewness': skewness, 'kurtosis': kurtosis}

    def _ks_distance(self, actual, simulated):
        actual = np.sort(np.asarray(actual, dtype=float))
        simulated = np.sort(np.asarray(simulated, dtype=float))
        pooled = np.sort(np.unique(np.concatenate([actual, simulated])))
        if len(pooled) == 0:
            return 0.0
        cdf_actual = np.searchsorted(actual, pooled, side='right') / max(len(actual), 1)
        cdf_sim = np.searchsorted(simulated, pooled, side='right') / max(len(simulated), 1)
        return float(np.max(np.abs(cdf_actual - cdf_sim)))

    def _sample_from_spec(self, spec, rng, size=1, bootstrap_source=None):
        generator = spec.get('generator', 'parametric')
        model = spec.get('model', 'empirical')
        support = spec.get('support', 'positive')

        if generator == 'empirical':
            base = np.asarray(bootstrap_source if bootstrap_source is not None else spec.get('values', [0.0]), dtype=float)
            sampled = rng.choice(base, size=size, replace=True)
            sampled = sampled * rng.uniform(0.9, 1.1, size=size)
        elif generator == 'rule_based':
            center = float(spec.get('center', 0.0))
            scale = float(spec.get('scale', max(center * 0.08, 0.05)))
            sampled = rng.normal(center, scale, size=size)
        elif generator == 'mixture' and model == 'gmm':
            gmm = spec.get('gmm_model')
            sampled = gmm.sample(size)[0].reshape(-1)
        elif model == 'lognormal':
            params = spec['params']
            sampled = rng.lognormal(params['mu'], params['sigma'], size=size)
        elif model == 'gamma':
            params = spec['params']
            sampled = rng.gamma(params['shape'], params['scale'], size=size)
        elif model == 'beta':
            params = spec['params']
            sampled = rng.beta(params['alpha'], params['beta'], size=size)
        else:
            base = np.asarray(bootstrap_source if bootstrap_source is not None else spec.get('values', [0.0]), dtype=float)
            sampled = rng.choice(base, size=size, replace=True)

        sampled = np.asarray(sampled, dtype=float)
        lower = float(spec.get('lower', 0.0 if support == 'positive' else 0.0))
        upper = float(spec.get('upper', 1.0 if support == 'unit_interval' else np.percentile(sampled, 99.5) if len(sampled) else 1.0))
        if support == 'positive':
            sampled = np.clip(sampled, max(lower, 1e-4), upper)
        elif support == 'unit_interval':
            sampled = np.clip(sampled, max(lower, 1e-3), min(upper, 1 - 1e-3))
        else:
            sampled = np.clip(sampled, lower, upper)
        return sampled

    def _evaluate_distribution_spec(self, values, spec, rng):
        simulated = self._sample_from_spec(spec, rng, size=max(800, len(values) * 4), bootstrap_source=values)
        actual_summary = self._quantile_summary(values)
        sim_summary = self._quantile_summary(simulated)
        quantile_errors = []
        for key in ['q25', 'q50', 'q75', 'q90']:
            denom = max(abs(actual_summary[key]), 1e-4)
            quantile_errors.append(abs(sim_summary[key] - actual_summary[key]) / denom)
        mean_error = abs(sim_summary['mean'] - actual_summary['mean']) / max(abs(actual_summary['mean']), 1e-4)
        std_error = abs(sim_summary['std'] - actual_summary['std']) / max(abs(actual_summary['std']), 1e-4)
        ks_distance = self._ks_distance(values, simulated)
        return {
            'ks_distance': float(ks_distance),
            'max_quantile_error': float(max(quantile_errors) if quantile_errors else 0.0),
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'accepted': bool(ks_distance <= 0.12 and max(quantile_errors) < 0.20)
        }

    def _bootstrap_stability(self, values, candidate_builder, rng, rounds=18):
        if len(values) < 120:
            return {'stable': False, 'param_cv': np.nan}
        param_vectors = []
        for _ in range(rounds):
            sample = rng.choice(values, size=len(values), replace=True)
            spec = candidate_builder(sample)
            params = spec.get('params')
            if spec.get('model') == 'gmm':
                gmm = spec.get('gmm_model')
                params = {
                    'weights_mean': float(np.mean(gmm.weights_)),
                    'means_gap': float(np.ptp(np.sort(gmm.means_.reshape(-1)))),
                    'cov_mean': float(np.mean(gmm.covariances_))
                }
            if not params:
                continue
            vector = np.asarray(list(params.values()), dtype=float)
            param_vectors.append(vector)
        if len(param_vectors) < 6:
            return {'stable': False, 'param_cv': np.nan}
        matrix = np.vstack(param_vectors)
        denom = np.maximum(np.abs(matrix.mean(axis=0)), 1e-4)
        cv = np.abs(matrix.std(axis=0) / denom)
        score = float(np.nanmax(cv))
        return {'stable': bool(score <= 0.30), 'param_cv': score}

    def _build_distribution_spec(self, values, variable_name, support='positive'):
        rng = np.random.default_rng(2026)
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if support == 'positive':
            values = values[values > 0]
        elif support == 'unit_interval':
            values = np.clip(values, 1e-3, 1 - 1e-3)

        if len(values) == 0:
            return {
                'variable': variable_name,
                'support': support,
                'generator': 'rule_based',
                'model': 'rule_based',
                'center': 0.5 if support == 'unit_interval' else 1.0,
                'scale': 0.1 if support == 'unit_interval' else 0.3,
                'lower': 1e-3 if support == 'unit_interval' else 0.1,
                'upper': 0.999 if support == 'unit_interval' else 5.0,
                'diagnostics': {'reason': 'empty_values'}
            }

        lower = float(np.quantile(values, 0.01))
        upper = float(np.quantile(values, 0.99))
        shape_stats = self._distribution_shape_stats(values)
        n = len(values)
        is_multimodal = self._detect_multimodal(values)

        def with_bounds(spec):
            spec['lower'] = 1e-3 if support == 'unit_interval' else max(1e-4, lower * 0.75)
            spec['upper'] = 0.999 if support == 'unit_interval' else max(upper * 1.25, np.max(values))
            spec['support'] = support
            spec['variable'] = variable_name
            spec['sample_size'] = int(n)
            spec['shape_stats'] = shape_stats
            return spec

        def empirical_spec(sample_values):
            return with_bounds({
                'generator': 'empirical',
                'model': 'empirical',
                'values': sample_values.tolist()
            })

        def rule_based_spec():
            return with_bounds({
                'generator': 'rule_based',
                'model': 'rule_based',
                'center': float(np.median(values)),
                'scale': float(max(np.std(values) * 0.15, 0.03 if support == 'unit_interval' else 0.3))
            })

        def lognormal_spec(sample_values):
            return with_bounds({
                'generator': 'parametric',
                'model': 'lognormal',
                'params': self._fit_lognormal(sample_values)
            })

        def gamma_spec(sample_values):
            return with_bounds({
                'generator': 'parametric',
                'model': 'gamma',
                'params': self._fit_gamma(sample_values)
            })

        def beta_spec(sample_values):
            return with_bounds({
                'generator': 'parametric',
                'model': 'beta',
                'params': self._fit_beta(sample_values)
            })

        def gmm_spec(sample_values):
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
            gmm.fit(sample_values.reshape(-1, 1))
            return with_bounds({
                'generator': 'mixture',
                'model': 'gmm',
                'gmm_model': gmm,
                'params': {
                    'weights_1': float(gmm.weights_[0]),
                    'weights_2': float(gmm.weights_[1]),
                    'mean_1': float(gmm.means_[0][0]),
                    'mean_2': float(gmm.means_[1][0])
                }
            })

        if not self.use_auto_distribution_selector:
            default_map = {
                'V': lognormal_spec,
                'A': gamma_spec,
                'S': beta_spec
            }
            spec = default_map.get(variable_name, empirical_spec)(values)
            spec['diagnostics'] = {
                'selected_by': 'manual_default',
                'n': int(n),
                'skewness': shape_stats['skewness'],
                'kurtosis': shape_stats['kurtosis'],
                'multimodal': bool(is_multimodal)
            }
            return spec

        if n < 100:
            spec = empirical_spec(values)
            spec['diagnostics'] = {
                'selected_by': 'sample_size_gate',
                'n': int(n),
                'reason': 'small_sample_empirical',
                'skewness': shape_stats['skewness'],
                'kurtosis': shape_stats['kurtosis'],
                'multimodal': bool(is_multimodal)
            }
            return spec

        candidate_builders = []
        if support == 'unit_interval':
            candidate_builders = [('beta', beta_spec)]
        elif is_multimodal and n >= 180:
            candidate_builders = [('gmm', gmm_spec), ('lognormal', lognormal_spec), ('gamma', gamma_spec)]
        elif shape_stats['skewness'] > 1.0:
            candidate_builders = [('lognormal', lognormal_spec), ('gamma', gamma_spec)]
        else:
            candidate_builders = [('gamma', gamma_spec), ('lognormal', lognormal_spec)]

        evaluations = []
        for name, builder in candidate_builders:
            try:
                spec = builder(values)
                quality = self._evaluate_distribution_spec(values, spec, rng)
                stability = self._bootstrap_stability(values, builder, rng) if spec['generator'] != 'empirical' else {'stable': True, 'param_cv': 0.0}
                evaluations.append((name, spec, quality, stability))
            except Exception as exc:
                logger.warning("%s 分布选择失败: %s", variable_name, exc)

        accepted = [
            (name, spec, quality, stability)
            for name, spec, quality, stability in evaluations
            if quality['accepted'] and stability['stable']
        ]
        if is_multimodal:
            accepted.sort(key=lambda item: (item[0] != 'gmm', item[2]['ks_distance'], item[2]['max_quantile_error']))
        else:
            accepted.sort(key=lambda item: (item[2]['ks_distance'], item[2]['max_quantile_error']))

        if accepted:
            name, spec, quality, stability = accepted[0]
        elif evaluations and is_multimodal and evaluations[0][0] == 'gmm' and evaluations[0][3]['stable']:
            name, spec, quality, stability = evaluations[0]
        elif evaluations:
            name, _, _, _ = evaluations[0]
            spec = empirical_spec(values)
            quality = {'ks_distance': np.nan, 'max_quantile_error': np.nan, 'mean_error': np.nan, 'std_error': np.nan, 'accepted': False}
            stability = {'stable': False, 'param_cv': np.nan}
        else:
            name = 'rule_based'
            spec = rule_based_spec()
            quality = {'ks_distance': np.nan, 'max_quantile_error': np.nan, 'mean_error': np.nan, 'std_error': np.nan, 'accepted': False}
            stability = {'stable': False, 'param_cv': np.nan}

        spec['diagnostics'] = {
            'selected_by': 'auto_selector',
            'selected_model': name if spec['generator'] != 'empirical' else 'empirical',
            'n': int(n),
            'skewness': float(shape_stats['skewness']),
            'kurtosis': float(shape_stats['kurtosis']),
            'multimodal': bool(is_multimodal),
            'ks_distance': float(quality['ks_distance']) if not np.isnan(quality['ks_distance']) else None,
            'max_quantile_error': float(quality['max_quantile_error']) if not np.isnan(quality['max_quantile_error']) else None,
            'stable': bool(stability['stable']),
            'param_cv': float(stability['param_cv']) if not np.isnan(stability['param_cv']) else None
        }
        return spec

    def _build_time_profile(self):
        scene_orders = self.scene_orders.copy()
        scene_orders['slot'] = scene_orders['下单时间'].apply(
            lambda x: f"{self._extract_hour(x):02d}:{30 if getattr(x, 'minute', 0) >= 30 else 0:02d}"
        )
        counts = scene_orders['slot'].value_counts().to_dict()
        weights = {}
        mean_count = np.mean([counts.get(slot, 0) for slot in TIME_SLOTS]) if len(TIME_SLOTS) > 0 else 1.0
        mean_count = max(mean_count, 1.0)
        for slot in TIME_SLOTS:
            weights[slot] = float(np.clip((counts.get(slot, 0) + 0.5) / mean_count, 0.45, 1.55))
        return weights

    def _build_coupon_catalog(self):
        coupon_orders = self.scene_orders[self.scene_orders['美补金额'] > 0].copy()
        if coupon_orders.empty:
            return {
                '3元免费非膨胀': {'coupon_value': 3.0, 'free': 1.0, 'expand': 1.0, 'uplift': 2.5, 'frequency': 1},
                '5元免费非膨胀': {'coupon_value': 5.0, 'free': 1.0, 'expand': 1.0, 'uplift': 3.5, 'frequency': 1},
                '8元免费非膨胀': {'coupon_value': 8.0, 'free': 1.0, 'expand': 1.0, 'uplift': 4.8, 'frequency': 1}
            }

        no_coupon_mean = float(self.scene_orders[self.scene_orders['美补金额'] <= 0]['订单金额'].mean())
        no_coupon_mean = no_coupon_mean if not np.isnan(no_coupon_mean) else float(self.scene_orders['订单金额'].mean())
        catalog = {}
        grouped = coupon_orders.groupby(['美补金额', '1-免费神券，2-付费神券', '1-非膨胀券，2-膨胀券'])
        for (amount, coupon_type, expand_type), group in grouped:
            type_name = '免费' if coupon_type == 1 else '付费'
            expand_name = '膨胀' if expand_type == 2 else '非膨胀'
            name = f"{int(amount)}元{type_name}{expand_name}"
            uplift = float(max(group['订单金额'].mean() - no_coupon_mean, amount * 0.35))
            catalog[name] = {
                'coupon_value': float(amount),
                'free': 1.0 if coupon_type == 1 else 0.0,
                'expand': 1.0 if expand_type == 2 else 0.0,
                'uplift': float(np.clip(uplift, 1.0, amount * 1.25 + 4)),
                'frequency': int(len(group)),
                'attraction': float(1 + 0.08 * np.log1p(len(group)) + 0.02 * amount + 0.04 * (coupon_type == 1) + 0.05 * (expand_type == 2))
            }

        sorted_items = sorted(catalog.items(), key=lambda x: x[1]['frequency'], reverse=True)[:6]
        return dict(sorted_items)

    def _fit_distributions_and_params(self):
        scene_user_counts = self.scene_orders.groupby('user_id').size().reindex(self.user_profile['user_id'], fill_value=0)
        positive_counts = scene_user_counts[scene_user_counts > 0]
        user_coupon_rate = self.scene_orders.groupby('user_id')['美补金额'].apply(lambda s: (s > 0).mean())

        value_spec = self._build_distribution_spec(self.scene_orders['订单金额'].values, variable_name='V', support='positive')
        activity_spec = self._build_distribution_spec(
            positive_counts.values if len(positive_counts) > 0 else np.array([1.0]),
            variable_name='A',
            support='positive'
        )
        sensitivity_spec = self._build_distribution_spec(
            user_coupon_rate.values if len(user_coupon_rate) > 0 else np.array([0.3, 0.4, 0.5]),
            variable_name='S',
            support='unit_interval'
        )
        value_dist = value_spec.get('params', self._fit_lognormal(self.scene_orders['订单金额'].values))
        activity_dist = activity_spec.get('params', self._fit_gamma(positive_counts.values if len(positive_counts) > 0 else np.array([1.0])))
        sensitivity_dist = sensitivity_spec.get('params', self._fit_beta(user_coupon_rate.values if len(user_coupon_rate) > 0 else np.array([0.3, 0.4, 0.5])))
        time_profile = self._build_time_profile()
        coupon_catalog = self._build_coupon_catalog()

        return {
            'scene_name': 'afternoon_tea',
            'value_dist': value_dist,
            'activity_dist': activity_dist,
            'sensitivity_dist': sensitivity_dist,
            'funnel': self.funnel_stats,
            'time_profile': time_profile,
            'coupon_catalog': coupon_catalog,
            'candidate_coupons': list(coupon_catalog.keys()),
            'delta_scale': 1.0,
            'scene_constraints': self.scene_constraints,
            'distribution_models': {
                'V': value_spec,
                'A': activity_spec,
                'S': sensitivity_spec
            },
            'reference_means': {
                'value': float(self.scene_orders['订单金额'].mean()),
                'activity': float(max(positive_counts.mean() if len(positive_counts) > 0 else 1.0, 1.0)),
                'sensitivity': float(user_coupon_rate.mean() if len(user_coupon_rate) > 0 else sensitivity_dist['alpha'] / (sensitivity_dist['alpha'] + sensitivity_dist['beta']))
            }
        }

    def _segment_from_profile(self, row):
        payment = str(row['历史实付分层（365天）'])
        member = str(row['会员状态'])
        if payment in ['高客单价', '超高客单价'] or member in ['L5', 'L6']:
            return 'high_value'
        if payment == '中客单价' or member in ['L3', 'L4']:
            return 'mid_value'
        return 'low_value'

    def _sample_user_parameters(self, profile_row, rng):
        ref = self.global_params['reference_means']
        value_spec = self.global_params.get('distribution_models', {}).get('V', {'model': 'lognormal', 'params': self.global_params['value_dist'], 'generator': 'parametric', 'support': 'positive'})
        activity_spec = self.global_params.get('distribution_models', {}).get('A', {'model': 'gamma', 'params': self.global_params['activity_dist'], 'generator': 'parametric', 'support': 'positive'})
        sensitivity_spec = self.global_params.get('distribution_models', {}).get('S', {'model': 'beta', 'params': self.global_params['sensitivity_dist'], 'generator': 'parametric', 'support': 'unit_interval'})
        sensitivity_params = self.global_params['sensitivity_dist']

        payment_value_factor = {
            '低客单价': 0.78, '中客单价': 0.95, '高客单价': 1.18, '超高客单价': 1.38
        }
        member_value_factor = {'L1': 0.90, 'L2': 0.95, 'L3': 1.0, 'L4': 1.08, 'L5': 1.16, 'L6': 1.24}
        lifecycle_activity_factor = {'新客': 0.82, '老客': 1.10, '流失召回': 0.72}
        spending_sensitivity_factor = {'低客单价': 1.26, '中客单价': 1.08, '高客单价': 0.88, '超高客单价': 0.74}

        payment = str(profile_row['历史实付分层（365天）'])
        member = str(profile_row['会员状态'])
        lifecycle = str(profile_row['用户生命周期阶段'])

        sampled_value = float(self._sample_from_spec(value_spec, rng, size=1)[0])
        value = sampled_value * payment_value_factor.get(payment, 1.0) * member_value_factor.get(member, 1.0)
        value = float(np.clip(value, 8.0, 220.0))

        sampled_activity = float(self._sample_from_spec(activity_spec, rng, size=1)[0])
        activity = sampled_activity * lifecycle_activity_factor.get(lifecycle, 1.0) * member_value_factor.get(member, 1.0)
        activity = float(np.clip(activity, 0.08, 6.0))

        global_mean_s = sensitivity_params['alpha'] / (sensitivity_params['alpha'] + sensitivity_params['beta'])
        mean_s = np.clip(global_mean_s * spending_sensitivity_factor.get(payment, 1.0), 0.05, 0.92)
        strength = sensitivity_params['alpha'] + sensitivity_params['beta']
        s_alpha = float(max(mean_s * strength, 1.1))
        s_beta = float(max((1 - mean_s) * strength, 1.1))
        sampled_s = float(self._sample_from_spec(
            {**sensitivity_spec, 'params': {'alpha': s_alpha, 'beta': s_beta}, 'generator': 'parametric', 'model': 'beta'},
            rng,
            size=1
        )[0])
        s_alpha_post = float(max(sampled_s * strength, 1.1))
        s_beta_post = float(max((1 - sampled_s) * strength, 1.1))

        return {
            'V': value,
            'A': activity,
            'S': sampled_s,
            'S_alpha': s_alpha,
            'S_beta': s_beta,
            'S_alpha_post': s_alpha_post,
            'S_beta_post': s_beta_post,
            'value_index': float(value / max(ref['value'], 1.0)),
            'activity_index': float(activity / max(ref['activity'], 1e-3))
        }

    def generate_users(self, n=1000, random_state=42):
        """基于 V/A/S 分布和画像辅助构造连续用户。"""
        rng = np.random.default_rng(random_state)
        profiles = self.user_profile.sample(n=n, replace=True, random_state=random_state).reset_index(drop=True)
        users = []
        for idx, row in profiles.iterrows():
            params = self._sample_user_parameters(row, rng)
            users.append({
                'user_id': f'SYN{idx + 1:05d}',
                'source_user_id': row['user_id'],
                '常驻城市': row['常驻城市'],
                '年龄': int(row['年龄']),
                '用户生命周期阶段': row['用户生命周期阶段'],
                '会员状态': row['会员状态'],
                '历史实付分层（365天）': row['历史实付分层（365天）'],
                'segment': self._segment_from_profile(row),
                'exposure': 0.0,
                'successes': 0.0,
                'failures': 0.0,
                **params
            })
        return pd.DataFrame(users)

    def _build_calibration_snapshot(self, users):
        posterior_mean_s = users['S'] if 'S' in users.columns else users['S_alpha_post'] / (users['S_alpha_post'] + users['S_beta_post'])
        return {
            'raw_scene_orders': int(len(self.scene_orders)),
            'raw_scene_users': int(self.scene_orders['user_id'].nunique()),
            'raw_avg_order_value': float(self.scene_orders['订单金额'].mean()),
            'raw_coupon_rate': float((self.scene_orders['美补金额'] > 0).mean()),
            'synthetic_users': int(len(users)),
            'synthetic_avg_value': float(users['V'].mean()),
            'synthetic_avg_activity': float(users['A'].mean()),
            'synthetic_avg_sensitivity': float(posterior_mean_s.mean())
        }
