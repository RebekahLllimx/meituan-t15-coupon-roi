import time
import os
import json
import numpy as np
import pandas as pd
import streamlit as st

from feature_engineering import FeatureEngineering
from llm_explainer import StrategyExplainer
from optimization import Optimization
from simulation_engine import SimulationEngine
from visualization import Visualization
from result_cache import ResultCache


viz = Visualization()
explainer = StrategyExplainer()
cache = ResultCache()


def _init_state():
    for key in [
        'feature_engineering', 'user_features', 'global_params', 'simulation_result',
        'optimization_result', 'sensitivity_result', 'budget_frontier', 'heatmap_result',
        'llm_explanations', 'run_cache_last_saved'
    ]:
        if key not in st.session_state:
            st.session_state[key] = None


def _monthly_business_impact_text(bandit_metrics, unified_metrics, daily_coupons=10000, days=30):
    """
    Convert strategy-level metrics into a business-friendly monthly impact sentence.
    IMPORTANT: scale by "expected coupon orders" (orders_with_coupon), NOT by covered_users.
    This avoids mismatch between unified-vs-bandit exposure bases.
    """
    def _safe(v):
        try:
            return float(v)
        except Exception:
            return 0.0

    bandit_cost = _safe(bandit_metrics.get("total_cost", 0.0))
    bandit_gtv = _safe(bandit_metrics.get("delta_gtv_causal", 0.0))
    bandit_orders_with_coupon = max(1.0, _safe(bandit_metrics.get("orders_with_coupon", 0.0)))
    uni_cost = _safe(unified_metrics.get("total_cost", 0.0))
    uni_gtv = _safe(unified_metrics.get("delta_gtv_causal", 0.0))
    uni_orders_with_coupon = max(1.0, _safe(unified_metrics.get("orders_with_coupon", 0.0)))

    # per-coupon-order average cost/gtv, then scale to assumed coupon volume
    bandit_cost_per = bandit_cost / bandit_orders_with_coupon
    bandit_gtv_per = bandit_gtv / bandit_orders_with_coupon
    uni_cost_per = uni_cost / uni_orders_with_coupon
    uni_gtv_per = uni_gtv / uni_orders_with_coupon

    monthly_cost_bandit = bandit_cost_per * daily_coupons * days
    monthly_gtv_bandit = bandit_gtv_per * daily_coupons * days
    monthly_cost_uni = uni_cost_per * daily_coupons * days
    monthly_gtv_uni = uni_gtv_per * daily_coupons * days

    saved_cost = monthly_cost_uni - monthly_cost_bandit
    extra_gtv = monthly_gtv_bandit - monthly_gtv_uni
    # 单位换成“万元”
    saved_cost_wan = saved_cost / 10000.0
    extra_gtv_wan = extra_gtv / 10000.0
    return (
        f"若每天 {int(daily_coupons):,} 次发券，相比统一发券，"
        f"每月可节省成本约 {saved_cost_wan:,.1f} 万元，同时额外带来 GTV 约 {extra_gtv_wan:,.1f} 万元"
    )


def _cache_sidebar():
    st.sidebar.divider()
    st.sidebar.header("结果缓存")
    metas = cache.list(limit=30)
    if not metas:
        st.sidebar.caption("暂无历史结果")
        return
    options = {f"{time.strftime('%m-%d %H:%M', time.localtime(m.created_at))} | {m.run_type} | {m.run_id[-8:]}": m.run_id for m in metas}
    selected_label = st.sidebar.selectbox("历史运行", list(options.keys()), index=0)
    run_id = options[selected_label]
    meta, artifacts = cache.load(run_id)
    with st.sidebar.expander("查看数值摘要", expanded=False):
        st.json(meta.summary)
        st.download_button(
            "下载摘要 JSON",
            data=cache.export_summary_json(run_id),
            file_name=f"{run_id}.summary.json",
            mime="application/json"
        )
    with st.sidebar.expander("查看参数", expanded=False):
        st.json(meta.params)

    st.sidebar.caption("提示：主页面也提供“结果层级视图”，可查看更完整对象")


def _render_object(name, obj):
    """Render nested objects in a compact, expandable way."""
    if obj is None:
        st.caption(f"{name}: None")
        return
    if isinstance(obj, pd.DataFrame):
        st.caption(f"{name}: DataFrame {obj.shape[0]}×{obj.shape[1]}")
        st.dataframe(obj, use_container_width=True)
        return
    if isinstance(obj, (dict, list)):
        st.caption(f"{name}: {type(obj).__name__}")
        st.json(obj)
        return
    st.caption(f"{name}: {type(obj).__name__}")
    st.write(obj)


def _results_hierarchy_panel():
    st.divider()
    st.header("结果层级视图")
    st.caption("用于快速查看已跑过的关键数值对象（session 当前结果 + 缓存历史 artifacts）")

    mode = st.radio("查看来源", ["当前 session", "缓存历史"], horizontal=True)
    if mode == "当前 session":
        cexp1, cexp2 = st.columns([1, 2])
        with cexp1:
            if st.button("导出当前 session 到本地", key="export_session_local"):
                out_dir, files = _export_current_session_results()
                st.success(f"已导出到 `{out_dir}`")
                st.caption(f"导出文件数：{len(files)}")
        with st.expander("仿真结果 simulation_result", expanded=False):
            _render_object("simulation_result", st.session_state.get("simulation_result"))
        with st.expander("优化结果 optimization_result", expanded=False):
            _render_object("optimization_result", st.session_state.get("optimization_result"))
        with st.expander("敏感性 sensitivity_result / heatmap_result", expanded=False):
            _render_object("sensitivity_result", st.session_state.get("sensitivity_result"))
            _render_object("heatmap_result", st.session_state.get("heatmap_result"))
        with st.expander("预算前沿 budget_frontier", expanded=False):
            _render_object("budget_frontier", st.session_state.get("budget_frontier"))
        with st.expander("LLM explanations", expanded=False):
            _render_object("llm_explanations", st.session_state.get("llm_explanations"))
        return

    metas = cache.list(limit=30)
    if not metas:
        st.info("暂无缓存历史结果")
        return
    label_to_id = {
        f"{time.strftime('%m-%d %H:%M', time.localtime(m.created_at))} | {m.run_type} | {m.run_id}": m.run_id
        for m in metas
    }
    selected = st.selectbox("选择一个缓存 run", list(label_to_id.keys()), index=0)
    run_id = label_to_id[selected]
    cexp3, cexp4 = st.columns([1, 2])
    with cexp3:
        if st.button("导出该缓存 run 到本地", key="export_cache_local"):
            out_dir, files = _export_cached_run_results(run_id=run_id)
            st.success(f"已导出到 `{out_dir}`")
            st.caption(f"导出文件数：{len(files)}")
    meta, artifacts = cache.load(run_id)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("摘要")
        st.json(meta.summary)
    with c2:
        st.subheader("参数")
        st.json(meta.params)
    st.subheader("Artifacts（完整对象）")
    if not artifacts:
        st.caption("该 run 未保存 artifacts")
        return
    for k in sorted(artifacts.keys()):
        with st.expander(k, expanded=False):
            _render_object(k, artifacts.get(k))


def _to_jsonable(obj):
    """Convert common objects to JSON-serializable structures."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def _save_dataframe_if_possible(base_dir, name, obj):
    if isinstance(obj, pd.DataFrame):
        csv_path = os.path.join(base_dir, f"{name}.csv")
        obj.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return csv_path
    return None


def _export_current_session_results(output_root="exports/results"):
    """
    Export current session result objects to local timestamped folder.
    Returns export directory path and exported file list.
    """
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    out_dir = os.path.join(output_root, f"session-{ts}")
    os.makedirs(out_dir, exist_ok=True)
    files = []

    payload = {
        "simulation_result": st.session_state.get("simulation_result"),
        "optimization_result": st.session_state.get("optimization_result"),
        "sensitivity_result": st.session_state.get("sensitivity_result"),
        "heatmap_result": st.session_state.get("heatmap_result"),
        "budget_frontier": st.session_state.get("budget_frontier"),
        "llm_explanations": st.session_state.get("llm_explanations"),
    }

    # Save JSON bundle
    json_path = os.path.join(out_dir, "session_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
    files.append(json_path)

    # Save key tabular artifacts as CSV if present
    if isinstance(payload["simulation_result"], dict):
        sim = payload["simulation_result"]
        for key in ["mc_df", "segment_lift_df"]:
            p = _save_dataframe_if_possible(out_dir, f"simulation_{key}", sim.get(key))
            if p:
                files.append(p)

    if isinstance(payload["optimization_result"], dict):
        opt = payload["optimization_result"]
        for key in ["allocation_df", "history", "final_user_pool"]:
            p = _save_dataframe_if_possible(out_dir, f"optimization_{key}", opt.get(key))
            if p:
                files.append(p)

    for key in ["sensitivity_result", "heatmap_result", "budget_frontier"]:
        p = _save_dataframe_if_possible(out_dir, key, payload.get(key))
        if p:
            files.append(p)

    return out_dir, files


def _export_cached_run_results(run_id, output_root="exports/results"):
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    out_dir = os.path.join(output_root, f"{run_id}-{ts}")
    os.makedirs(out_dir, exist_ok=True)
    files = []

    meta, artifacts = cache.load(run_id)
    meta_path = os.path.join(out_dir, "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": meta.run_id,
            "run_type": meta.run_type,
            "created_at": meta.created_at,
            "params": _to_jsonable(meta.params),
            "summary": _to_jsonable(meta.summary),
        }, f, ensure_ascii=False, indent=2)
    files.append(meta_path)

    art_json = os.path.join(out_dir, "artifacts.json")
    with open(art_json, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(artifacts), f, ensure_ascii=False, indent=2)
    files.append(art_json)

    if isinstance(artifacts, dict):
        for name, obj in artifacts.items():
            p = _save_dataframe_if_possible(out_dir, name, obj)
            if p:
                files.append(p)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    p2 = _save_dataframe_if_possible(out_dir, f"{name}_{k}", v)
                    if p2:
                        files.append(p2)
    return out_dir, files


def _posterior_mean_s(user_df):
    if 'S' in user_df.columns:
        return user_df['S']
    return user_df['S_alpha_post'] / (user_df['S_alpha_post'] + user_df['S_beta_post'])


def _safe_sensitivity_labels(values):
    clean_values = pd.Series(values).astype(float).clip(1e-4, 1 - 1e-4)
    ranked = clean_values.rank(method='first')
    try:
        n_bins = max(1, min(3, int(ranked.nunique())))
        labels = pd.qcut(ranked, q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        labels = pd.Series(np.zeros(len(clean_values), dtype=int), index=clean_values.index)
    label_map = {0: '低敏感', 1: '中敏感', 2: '高敏感'}
    return labels.map(label_map).fillna('中敏感')


def _build_distribution_selector_df(global_params):
    rows = []
    label_map = {'V': '用户价值 V', 'A': '活跃度 A', 'S': '价格敏感度 S'}
    for key, spec in global_params.get('distribution_models', {}).items():
        diagnostics = spec.get('diagnostics', {})
        rows.append({
            '变量': label_map.get(key, key),
            '生成器': spec.get('generator', '-'),
            '模型': spec.get('model', '-'),
            '样本量': diagnostics.get('n'),
            '偏度': round(diagnostics.get('skewness', 0.0), 3) if diagnostics.get('skewness') is not None else None,
            '多峰': '是' if diagnostics.get('multimodal') else '否',
            'KS距离': round(diagnostics.get('ks_distance', 0.0), 3) if diagnostics.get('ks_distance') is not None else None,
            '最大分位误差': round(diagnostics.get('max_quantile_error', 0.0), 3) if diagnostics.get('max_quantile_error') is not None else None,
            '稳定': '是' if diagnostics.get('stable') else '否'
        })
    return pd.DataFrame(rows)


def _build_beta_df(user_features, engine, time_slot='15:00'):
    rows = []
    for _, user in user_features.sample(min(len(user_features), 350), random_state=42).iterrows():
        response = engine.funnel_response(user, coupon=None, time_slot=time_slot)
        rows.append({
            'user_id': user['user_id'],
            'V': user['V'],
            'beta': response['beta'],
            '会员状态': user['会员状态'],
            '用户生命周期阶段': user['用户生命周期阶段']
        })
    return pd.DataFrame(rows)


def _build_delta_curve_df(global_params, reference_value):
    rows = []
    coupon_values = sorted({int(meta['coupon_value']) for meta in global_params['coupon_catalog'].values()})
    for sensitivity, label in [(0.2, 'S=0.2 低敏感'), (0.5, 'S=0.5 中敏感'), (0.8, 'S=0.8 高敏感')]:
        for coupon_value in coupon_values:
            p3 = global_params['funnel']['p3']
            delta = (
                global_params.get('delta_scale', 1.0)
                * sensitivity
                * np.log1p(coupon_value / max(reference_value, 1.0))
                * (1 - p3)
            )
            rows.append({
                'coupon_value': coupon_value,
                'delta': delta,
                '敏感度分组': label
            })
    return pd.DataFrame(rows)


def _build_stage_difference_df(user_features, engine):
    rows = []
    sample_df = user_features.groupby('segment', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 120), random_state=42)
    ).reset_index(drop=True)
    for segment, group in sample_df.groupby('segment'):
        p1_list, p2_list, p3_list = [], [], []
        for _, user in group.iterrows():
            response = engine.funnel_response(user, coupon=None, time_slot='15:00')
            p1_list.append(response['p1'])
            p2_list.append(response['p2'])
            p3_list.append(response['p3_base'])
        rows.extend([
            {'segment': segment, 'stage': 'p1', 'probability': np.mean(p1_list)},
            {'segment': segment, 'stage': 'p2', 'probability': np.mean(p2_list)},
            {'segment': segment, 'stage': 'p3', 'probability': np.mean(p3_list)}
        ])
    return pd.DataFrame(rows)


def _build_segment_lift_df(user_features, engine, coupon):
    sample_df = user_features.copy()
    sample_df['S_mean'] = _posterior_mean_s(sample_df)
    sample_df['敏感度分组'] = _safe_sensitivity_labels(sample_df['S_mean'])
    rows = []
    for label, group in sample_df.groupby('敏感度分组'):
        no_coupon, with_coupon = [], []
        for _, user in group.sample(min(len(group), 120), random_state=42).iterrows():
            base = engine.funnel_response(user, coupon=None, time_slot='15:00')
            treat = engine.funnel_response(user, coupon=coupon, time_slot='15:00')
            no_coupon.append(base['prob_no_coupon'])
            with_coupon.append(treat['prob_with_coupon'])
        rows.append({
            '敏感度分组': label,
            'prob_no_coupon': np.mean(no_coupon),
            'prob_with_coupon': np.mean(with_coupon),
            'lift': np.mean(with_coupon) - np.mean(no_coupon)
        })
    return pd.DataFrame(rows)


def _build_coverage_df(final_user_pool, allocation):
    coverage = final_user_pool[['user_id', 'segment']].copy()
    coverage['coupon'] = coverage['user_id'].map(allocation)
    coverage['是否发券'] = np.where(coverage['coupon'].notna(), '发券', '不发券')
    return coverage.groupby(['segment', '是否发券']).size().reset_index(name='count')


def _build_top_user_table(result, global_params):
    allocation_df = result['allocation_df'].copy()
    if allocation_df.empty:
        return allocation_df

    allocation_df = allocation_df.rename(columns={'value': 'delta_gtv_causal'})

    final_user_pool = result['final_user_pool'].copy()
    final_user_pool['S_mean'] = _posterior_mean_s(final_user_pool)
    merged = allocation_df.merge(
        final_user_pool[
            ['user_id', 'segment', 'V', 'A', 'S_mean', '会员状态', '用户生命周期阶段', '历史实付分层（365天）']
        ],
        on='user_id',
        how='left'
    )
    metric_rows = []
    for row in merged.itertuples():
        user_metrics = result['metrics'][row.user_id][row.coupon]
        metric_rows.append({
            'user_id': row.user_id,
            'beta': user_metrics['avg_beta'],
            'delta': user_metrics['avg_delta'],
            'p1': user_metrics['avg_p1'],
            'p2': user_metrics['avg_p2'],
            'p3': user_metrics['avg_p3'],
        })
    merged = merged.merge(pd.DataFrame(metric_rows), on='user_id', how='left')
    return merged.sort_values(['delta_gtv_causal', 'roi'], ascending=False)


def _build_strategy_overview_text(top_user_df):
    if top_user_df is None or top_user_df.empty:
        return "当前预算与约束下暂无有效发券用户，建议放宽预算或提高单券覆盖上限后重试。"
    avg_roi = float(top_user_df['roi'].mean())
    top_segment = top_user_df['segment'].value_counts().index[0]
    top_coupon = top_user_df['coupon'].value_counts().index[0]
    high_s_ratio = float((top_user_df['S_mean'] >= top_user_df['S_mean'].quantile(0.7)).mean())
    return (
        f"总体上，策略优先把券投向 `{top_segment}`，主力券型是 `{top_coupon}`。"
        f" 当前 Top 用户平均 ROI 约为 {avg_roi:.2f}，且高敏感用户占比约 {high_s_ratio:.0%}，"
        "说明模型在预算约束下更偏向“高潜力 + 高价格响应”人群。"
    )


def _build_scan_values(min_v, max_v, step):
    min_v = float(min_v)
    max_v = float(max_v)
    step = max(float(step), 1e-4)
    if max_v < min_v:
        min_v, max_v = max_v, min_v
    count = int(np.floor((max_v - min_v) / step)) + 1
    values = [round(min_v + i * step, 4) for i in range(max(count, 1))]
    if not values or values[-1] < round(max_v, 4):
        values.append(round(max_v, 4))
    # 去重并按顺序输出
    dedup = []
    for v in values:
        if not dedup or not np.isclose(v, dedup[-1]):
            dedup.append(v)
    return dedup


st.set_page_config(page_title='下午茶半合成仿真系统', layout='wide')
st.title('下午茶半合成仿真与策略优化系统')
st.caption('用数据定义结构，在结构化环境中推导补贴策略，并为业务提供可解释的发券建议。')
_init_state()
_cache_sidebar()
_results_hierarchy_panel()

st.sidebar.header('数据与建模配置')
data_dir = st.sidebar.text_input('原始数据目录', './data/raw')
n_users = st.sidebar.slider('生成用户数', min_value=500, max_value=2000, value=1000, step=100)
random_state = st.sidebar.number_input('随机种子', min_value=1, value=42, step=1)
use_llm_explainer = st.sidebar.toggle('启用大模型策略解释', value=True)
use_auto_distribution_selector = st.sidebar.toggle('启用自动分布选择器', value=True)

if st.sidebar.button('加载原始数据'):
    fe = FeatureEngineering(use_api=False)
    if fe.load_data(data_dir):
        st.session_state['feature_engineering'] = fe
        st.success('原始数据加载成功')
    else:
        st.error('原始数据加载失败，请检查目录')

if st.session_state['feature_engineering'] is not None:
    st.header('Step 1. 半合成用户生成')
    if st.button('构建 Funnel + 分布 + 用户生成器'):
        fe = st.session_state['feature_engineering']
        with st.spinner('正在根据原始数据构建半合成场景模型...'):
            user_features, global_params = fe.build_features_and_params(
                n_users=n_users,
                random_state=int(random_state),
                use_auto_distribution_selector=use_auto_distribution_selector
            )
        st.session_state['user_features'] = user_features
        st.session_state['global_params'] = global_params
        st.success('半合成场景模型构建完成')

if st.session_state['user_features'] is not None and st.session_state['global_params'] is not None:
    fe = st.session_state['feature_engineering']
    user_features = st.session_state['user_features']
    global_params = st.session_state['global_params']
    engine = SimulationEngine(user_features, global_params)

    st.header('Step 2. 数据概览与结构校准')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('原始用户数', len(fe.user_profile))
    
    # 检查scene_orders是否存在
    if fe.scene_orders is not None and not fe.scene_orders.empty:
        col2.metric('下午茶场景订单', len(fe.scene_orders))
        col3.metric('下午茶场景用户', fe.scene_orders['user_id'].nunique())
    else:
        col2.metric('下午茶场景订单', 0)
        col3.metric('下午茶场景用户', 0)
        
    col4.metric('生成用户数', len(user_features))

    st.pyplot(viz.plot_profile_distribution(fe.user_profile))
    
    # 检查scene_orders是否存在
    if fe.scene_orders is not None and not fe.scene_orders.empty:
        st.pyplot(viz.plot_scene_overview(fe.order_df, fe.scene_orders))
        st.pyplot(viz.plot_order_amount_analysis(fe.order_df, fe.scene_orders))
    
    row_a1, row_a2 = st.columns(2)
    with row_a1:
        st.pyplot(viz.plot_funnel(global_params['funnel']))
    with row_a2:
        st.pyplot(viz.plot_stage_difference(_build_stage_difference_df(user_features, engine)))
    st.pyplot(viz.plot_distribution_fit(user_features, global_params))
    st.pyplot(viz.plot_generated_user_map(user_features))
    st.pyplot(viz.plot_calibration(global_params['calibration']))
    selector_df = _build_distribution_selector_df(global_params)
    if not use_auto_distribution_selector:
        st.warning('当前未启用自动分布选择器：系统使用固定分布设定，下面结果仅作参考说明。')
    if not selector_df.empty:
        st.subheader('自动分布选择结果')
        st.dataframe(selector_df, use_container_width=True)

    funnel = global_params['funnel']
    st.info(
        f"当前 Funnel 结构：p1={funnel['p1']:.3f}, p2={funnel['p2']:.3f}, p3={funnel['p3']:.3f}。"
        " 行为序列只负责提供结构，beta 由 p1*p2*p3 推导，券只作用于 p3。"
    )

if st.session_state['user_features'] is not None and st.session_state['global_params'] is not None:
    st.header('Step 3. 用户仿真与机制展示')
    user_features = st.session_state['user_features']
    global_params = st.session_state['global_params']
    engine = SimulationEngine(user_features, global_params)

    coupon_options = ['不发券'] + list(global_params['candidate_coupons'])
    default_coupon_index = 1 if len(coupon_options) > 1 else 0
    selected_coupon = st.selectbox('选择仿真券型', coupon_options, index=default_coupon_index)
    sim_seed = st.number_input('仿真随机种子', min_value=1, value=2026, step=1)
    mc_runs = st.slider('Monte Carlo 轮数', min_value=20, max_value=120, value=50, step=10)
    if selected_coupon == '不发券':
        st.info('当前选择的是“不发券”基线策略，因此期望因果增量、分组转化增量和 Monte Carlo 增量分布接近 0 属于正常现象。')

    if st.button('运行 Funnel 仿真'):
        coupon_value = None if selected_coupon == '不发券' else selected_coupon
        strategy = {user_id: coupon_value for user_id in user_features['user_id']}
        expected_metrics = engine.calculate_expected_metrics(strategy)
        simulated_metrics = engine.calculate_simulated_metrics(strategy, seed=int(sim_seed))
        mc_df = pd.DataFrame(engine.monte_carlo_runs(strategy, n_runs=int(mc_runs), seed=int(sim_seed)))
        st.session_state['simulation_result'] = {
            'selected_coupon': selected_coupon,
            'expected_metrics': expected_metrics,
            'simulated_metrics': simulated_metrics,
            'mc_df': mc_df,
            'segment_lift_df': _build_segment_lift_df(user_features, engine, coupon_value)
        }
        cache.save(
            run_type="simulation",
            params={"coupon": selected_coupon, "seed": int(sim_seed), "mc_runs": int(mc_runs), "n_users": int(len(user_features))},
            summary={
                "selected_coupon": selected_coupon,
                "expected_metrics": {k: float(expected_metrics[k]) for k in ["delta_gtv_causal", "roi", "total_cost", "orders_with_coupon", "avg_beta"]},
                "simulated_metrics": {k: float(simulated_metrics[k]) for k in ["delta_gtv_causal", "roi", "total_cost", "orders_with_coupon"]},
            },
            artifacts={"simulation_result": st.session_state["simulation_result"]},
        )

    if st.session_state['simulation_result'] is not None:
        sim_result = st.session_state['simulation_result']
        expected_metrics = sim_result['expected_metrics']
        simulated_metrics = sim_result['simulated_metrics']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('解析因果增量GTV', f"{expected_metrics['delta_gtv_causal']:.2f}")
        c2.metric('解析 ROI', f"{expected_metrics['roi']:.2f}")
        c3.metric('随机因果增量GTV', f"{simulated_metrics['delta_gtv_causal']:.2f}")
        c4.metric('随机 ROI', f"{simulated_metrics['roi']:.2f}")
        st.pyplot(viz.plot_simulation_comparison(expected_metrics, simulated_metrics))
        st.pyplot(viz.plot_segment_lift(sim_result['segment_lift_df']))
        st.pyplot(viz.plot_monte_carlo_distribution(sim_result['mc_df']))
        st.dataframe(pd.DataFrame([
            {'metric': 'p1(view→click)', 'value': expected_metrics['avg_p1']},
            {'metric': 'p2(click→cart)', 'value': expected_metrics['avg_p2']},
            {'metric': 'p3_base(cart→purchase)', 'value': expected_metrics['avg_p3_base']},
            {'metric': 'p3_coupon(cart→purchase)', 'value': expected_metrics['avg_p3_coupon']},
            {'metric': 'beta', 'value': expected_metrics['avg_beta']},
            {'metric': 'orders_with_coupon', 'value': expected_metrics['orders_with_coupon']}
        ]), use_container_width=True)
        st.caption("说明：beta 基于 p3_base 计算；发券后 p3_coupon 会上升，因此 beta 与 p1*p2*p3_coupon 不应直接对齐。")

if st.session_state['user_features'] is not None and st.session_state['global_params'] is not None:
    st.header('Step 4. Bandit 优化与业务解释')
    user_features = st.session_state['user_features']
    global_params = st.session_state['global_params']
    fe = st.session_state['feature_engineering']

    control1, control2, control3 = st.columns(3)
    budget = control1.number_input('总预算', min_value=100, value=3000, step=100)
    coupon_cap_share = control2.slider('单券型覆盖上限', min_value=0.10, max_value=0.80, value=0.35, step=0.05)
    alpha_ucb = control3.slider('Bandit探索强度 α', min_value=0.0, max_value=2.0, value=0.25, step=0.05)
    epochs = st.slider('Bandit训练轮次', min_value=1, max_value=10, value=3, step=1)

    if st.button('求解最优策略'):
        opt = Optimization(
            user_features,
            global_params,
            fe.order_df,
            coupon_cap_share=coupon_cap_share,
            alpha_ucb=alpha_ucb
        )
        with st.spinner('正在进行 Contextual Bandit 优化...'):
            result = opt.optimize(budget=budget, epochs=epochs, show_progress=True)
        st.session_state['optimization_result'] = result
        st.session_state['llm_explanations'] = None
        # cache optimization summary + key artifacts
        optimized_metrics = result["optimized_metrics"].copy()
        covered_users = int((pd.Series(result["allocation"]).notna()).sum())
        optimized_metrics["covered_users"] = covered_users
        cache.save(
            run_type="optimization",
            params={
                "budget": float(budget),
                "epochs": int(epochs),
                "coupon_cap_share": float(coupon_cap_share),
                "alpha_ucb": float(alpha_ucb),
                "n_users": int(len(user_features)),
            },
            summary={
                "optimized_metrics": {k: float(optimized_metrics.get(k, 0.0)) for k in ["delta_gtv_causal", "roi", "total_cost", "covered_users"]},
                "constraints": result.get("constraints", {}),
                "coupon_usage": result.get("coupon_usage", {}),
            },
            artifacts={"optimization_result": result},
        )

    if st.session_state['optimization_result'] is not None:
        result = st.session_state['optimization_result']
        optimized_metrics = result['optimized_metrics']
        top_user_df = _build_top_user_table(result, global_params)
        coverage_df = _build_coverage_df(result['final_user_pool'], result['allocation'])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('因果增量GTV', f"{optimized_metrics['delta_gtv_causal']:.2f}")
        c2.metric('ROI', f"{optimized_metrics['roi']:.2f}")
        c3.metric('预算消耗', f"{optimized_metrics['total_cost']:.2f}")
        c4.metric('覆盖用户数', int((pd.Series(result['allocation']).notna()).sum()))
        st.caption(
            f"约束：单券型上限 {result['constraints']['coupon_cap_share']:.0%}，"
            f"探索强度 α={result['constraints']['alpha_ucb']:.2f}，训练轮次={result['constraints']['epochs']}"
        )
        st.pyplot(viz.plot_strategy_comparison(result['strategies']))
        st.pyplot(viz.plot_bandit_history(result['history']))
        st.pyplot(viz.plot_allocation_mix(result['allocation_df']))
        st.pyplot(viz.plot_coverage_by_segment(coverage_df))

        # ---- Bandit vs 基线 对比表（数值，不一定可视化）----
        strategy_rows = []
        strategies = result.get("strategies", {})
        # 选择一个“统一发券”做对比：允许切换对照口径，避免“对照选得太强/太弱”造成误解
        unified_candidates = {k: v for k, v in strategies.items() if k.startswith("统一发")}
        unified_name = None
        unified_metrics = None
        unified_mode = st.selectbox(
            "统一发券对照选择口径",
            ["ROI 最优（效率对照）", "增量GTV 最优（规模对照）", "手动指定券"],
            index=0
        )
        manual_unified = None
        if unified_mode == "手动指定券":
            manual_unified = st.selectbox("手动选择统一发券券型", sorted(list(unified_candidates.keys())) or ["统一发券(无)"])
        if unified_candidates:
            if unified_mode == "增量GTV 最优（规模对照）":
                unified_name, unified_metrics = max(unified_candidates.items(), key=lambda kv: float(kv[1].get("delta_gtv_causal", 0.0)))
            elif unified_mode == "手动指定券" and manual_unified in unified_candidates:
                unified_name, unified_metrics = manual_unified, unified_candidates[manual_unified]
            else:
                unified_name, unified_metrics = max(unified_candidates.items(), key=lambda kv: float(kv[1].get("roi", 0.0)))
        baseline_metrics = strategies.get("不发券", {})
        bandit_metrics = strategies.get("Bandit优化", result.get("optimized_metrics", {}))
        if unified_metrics is None:
            unified_name, unified_metrics = "统一发券(无)", {"delta_gtv_causal": 0.0, "roi": 0.0, "total_cost": 0.0, "orders_with_coupon": 0.0}

        # covered users for bandit from allocation
        bandit_covered_users = int((pd.Series(result["allocation"]).notna()).sum())
        bandit_metrics = dict(bandit_metrics)
        bandit_metrics["covered_users"] = bandit_covered_users
        unified_metrics = dict(unified_metrics)
        # 商业价值换算将按 orders_with_coupon 缩放，不再强行对齐 covered_users
        baseline_metrics = dict(baseline_metrics)
        baseline_metrics["covered_users"] = 0

        for name, metrics in [("基线(不发券)", baseline_metrics), (unified_name, unified_metrics), ("Bandit优化", bandit_metrics)]:
            strategy_rows.append({
                "策略": name,
                "因果增量GTV": float(metrics.get("delta_gtv_causal", 0.0)),
                "ROI": float(metrics.get("roi", 0.0)),
                "预算成本": float(metrics.get("total_cost", 0.0)),
                "覆盖用户数": int(metrics.get("covered_users", 0)),
            })
        st.subheader("Bandit vs 基线：数值对比表")
        st.dataframe(pd.DataFrame(strategy_rows), use_container_width=True)

        # ---- 商业价值换算（展示给用户，也可喂给 LLM）----
        daily_coupons = st.number_input("商业价值换算：每日发券次数", min_value=1000, value=10000, step=1000)
        impact_text = _monthly_business_impact_text(bandit_metrics, unified_metrics, daily_coupons=int(daily_coupons), days=30)
        st.info(impact_text)

        # ---- 拆解：每 10,000 次发券的成本与增量（用于解释“为什么是负的”）----
        def _per_coupon(metrics, key):
            denom = float(metrics.get("orders_with_coupon", 0.0)) if float(metrics.get("orders_with_coupon", 0.0)) > 0 else 0.0
            return float(metrics.get(key, 0.0)) / denom if denom > 0 else 0.0

        scale = int(daily_coupons)
        rows_explain = []
        for name, m in [(unified_name, unified_metrics), ("Bandit优化", bandit_metrics)]:
            cost_per = _per_coupon(m, "total_cost")
            gtv_per = _per_coupon(m, "delta_gtv_causal")
            rows_explain.append({
                "策略": name,
                "orders_with_coupon(期望)": float(m.get("orders_with_coupon", 0.0)),
                "每次发券成本": cost_per,
                "每次发券增量GTV": gtv_per,
                f"每 {scale:,} 次发券成本": cost_per * scale,
                f"每 {scale:,} 次发券增量GTV": gtv_per * scale,
                "ROI(原口径)": float(m.get("roi", 0.0)),
            })
        explain_df = pd.DataFrame(rows_explain)
        st.subheader("商业价值拆解（同口径：按发券次数）")
        st.dataframe(explain_df, use_container_width=True)
        delta_per_coupon = float(explain_df.loc[explain_df["策略"] == "Bandit优化", "每次发券增量GTV"].values[0]) - float(explain_df.loc[explain_df["策略"] == unified_name, "每次发券增量GTV"].values[0])
        if delta_per_coupon < 0:
            st.warning("当前显示为负，表示在“每次发券”口径下 Bandit 的增量GTV低于统一发券。常见原因：Bandit在预算约束下更偏向控成本/提ROI，或你选择的统一券型在模型里对增量更强。可切换上方对照口径验证。")
        if st.button('计算预算前沿', key='calc_budget_frontier'):
            opt_frontier = Optimization(
                user_features,
                global_params,
                fe.order_df,
                coupon_cap_share=coupon_cap_share,
                alpha_ucb=alpha_ucb
            )
            with st.spinner('正在计算预算前沿...'):
                budget_points = sorted(set([max(100, budget * x) for x in [0.4, 0.7, 1.0, 1.3, 1.6]]))
                st.session_state['budget_frontier'] = opt_frontier.budget_frontier(budget_points, epochs=max(1, min(epochs, 2)))
        if st.session_state['budget_frontier'] is not None:
            st.pyplot(viz.plot_budget_frontier(st.session_state['budget_frontier']))

        st.subheader('Top 用户策略表')
        show_cols = ['user_id', 'segment', 'coupon', 'delta_gtv_causal', 'roi', 'V', 'A', 'S_mean', '会员状态', '用户生命周期阶段']
        st.dataframe(top_user_df[show_cols].head(20), use_container_width=True)

        if use_llm_explainer:
            explain_count = st.slider('生成解释的用户数', min_value=3, max_value=10, value=5, step=1)
            if st.button('生成业务策略解释'):
                rows = top_user_df.head(explain_count).to_dict('records')
                with st.spinner('正在生成业务策略解释...'):
                    business_context = {
                        "impact_text": impact_text,
                        "daily_coupons": int(daily_coupons),
                        "unified_strategy": unified_name,
                        "bandit_vs_unified_monthly": impact_text,
                    }
                    st.session_state['llm_explanations'] = explainer.explain_batch(rows, business_context=business_context)

            if st.session_state['llm_explanations'] is not None:
                st.subheader('业务策略解释')
                st.markdown("**总体策略**")
                st.write(_build_strategy_overview_text(top_user_df.head(explain_count)))
                st.markdown("**分用户策略**")
                explanation_df = pd.DataFrame(st.session_state['llm_explanations'])
                merged_explanations = explanation_df.merge(top_user_df[['user_id', 'coupon', 'segment']].head(len(explanation_df)), on='user_id', how='left')
                for row in merged_explanations.itertuples():
                    st.markdown(f"**{row.user_id} | {row.segment} | {row.coupon}**")
                    st.write(row.explanation)

        if st.checkbox('显示详细发券分配'):
            st.dataframe(result['allocation_df'], use_container_width=True)

        # ---- 可选：A/B Test 回测设计（展示方案，不做真实回放）----
        with st.expander("A/B Test 回测设计", expanded=False):
            st.markdown("**实验目的**：验证 Bandit 个体化发券相对“统一发券/不发券”是否在真实线上带来更高的增量 GTV 与更优 ROI")
            st.markdown("**分组**")
            ab_df = pd.DataFrame([
                {"组别": "A（控制组）", "策略": "统一发券（选 ROI 最优统一券）或 不发券", "人群": "全量可触达用户随机抽样"},
                {"组别": "B（实验组）", "策略": "Bandit 个体化发券", "人群": "同口径随机抽样"},
            ])
            st.table(ab_df)
            st.markdown("**核心指标（建议同时看规模与效率）**")
            metric_df = pd.DataFrame([
                {"指标": "增量GTV（反事实口径）", "说明": "B组 - A组 的 GTV 差异（可叠加 CUPED/分层校正）"},
                {"指标": "补贴成本", "说明": "实际核销/补贴金额"},
                {"指标": "ROI", "说明": "增量GTV / 补贴成本"},
                {"指标": "转化率提升", "说明": "购买率、下单率、复购率等漏斗指标"},
            ])
            st.table(metric_df)
            st.markdown("**实验周期与注意事项**")
            st.write("建议至少覆盖 2 个完整周周期（工作日/周末），并做分层随机（会员等级/历史价值/城市）以降低方差；同时监控券滥用与串扰（同一用户多券触达）")

if st.session_state['user_features'] is not None and st.session_state['global_params'] is not None:
    st.header('Step 5. 敏感性分析与鲁棒性')
    budget_sa = st.number_input('敏感性分析预算', min_value=100, value=3000, step=100, key='sa_budget')
    st.caption('该步骤会多次重复跑优化器（不同参数组合），因此耗时通常显著高于单次求解最优策略。')
    st.markdown('**热力图搜索范围设置**')
    scan_col1, scan_col2, scan_col3 = st.columns(3)
    beta_min = scan_col1.number_input('beta_scale 最小值', min_value=0.5, max_value=1.5, value=0.8, step=0.05, key='beta_min')
    beta_max = scan_col2.number_input('beta_scale 最大值', min_value=0.5, max_value=1.5, value=1.2, step=0.05, key='beta_max')
    beta_step = scan_col3.number_input('beta_scale 步长', min_value=0.05, max_value=0.5, value=0.1, step=0.05, key='beta_step')
    scan_col4, scan_col5, scan_col6 = st.columns(3)
    delta_min = scan_col4.number_input('delta_scale 最小值', min_value=0.5, max_value=1.5, value=0.8, step=0.05, key='delta_min')
    delta_max = scan_col5.number_input('delta_scale 最大值', min_value=0.5, max_value=1.5, value=1.2, step=0.05, key='delta_max')
    delta_step = scan_col6.number_input('delta_scale 步长', min_value=0.05, max_value=0.5, value=0.1, step=0.05, key='delta_step')
    beta_scales = _build_scan_values(beta_min, beta_max, beta_step)
    delta_scales = _build_scan_values(delta_min, delta_max, delta_step)
    st.caption(
        f"当前热力图网格：beta_scale {beta_scales[0]:.2f}~{beta_scales[-1]:.2f}（{len(beta_scales)} 点），"
        f"delta_scale {delta_scales[0]:.2f}~{delta_scales[-1]:.2f}（{len(delta_scales)} 点）。"
    )
    if st.button('运行敏感性分析'):
        fe = st.session_state['feature_engineering']
        opt = Optimization(st.session_state['user_features'], st.session_state['global_params'], fe.order_df)
        with st.spinner('正在扫描关键参数...'):
            sensitivity_df = opt.sensitivity_analysis(budget=budget_sa, epochs=2)
            heatmap_df = opt.sensitivity_heatmap(
                budget=budget_sa,
                beta_scales=beta_scales,
                delta_scales=delta_scales,
                epochs=1
            )
        st.session_state['sensitivity_result'] = sensitivity_df
        st.session_state['heatmap_result'] = heatmap_df
        cache.save(
            run_type="sensitivity",
            params={
                "budget": float(budget_sa),
                "beta_scales": beta_scales,
                "delta_scales": delta_scales,
                "epochs_sa": 2,
                "epochs_heatmap": 1,
            },
            summary={
                "sensitivity": sensitivity_df.to_dict(orient="records"),
                "heatmap_best_roi": float(heatmap_df["roi"].max()) if len(heatmap_df) else 0.0,
                "heatmap_grid": {"beta_points": len(beta_scales), "delta_points": len(delta_scales)},
            },
            artifacts={"sensitivity_df": sensitivity_df, "heatmap_df": heatmap_df},
        )

    if st.session_state['sensitivity_result'] is not None:
        sensitivity_df = st.session_state['sensitivity_result']
        st.pyplot(viz.plot_sensitivity(sensitivity_df))
        if st.session_state['heatmap_result'] is not None:
            heatmap_df = st.session_state['heatmap_result']
            st.pyplot(viz.plot_sensitivity_heatmap(heatmap_df, value_col='roi'))
            best_row = heatmap_df.loc[heatmap_df['roi'].idxmax()]
            beta_min, beta_max = heatmap_df['beta_scale'].min(), heatmap_df['beta_scale'].max()
            delta_min, delta_max = heatmap_df['delta_scale'].min(), heatmap_df['delta_scale'].max()
            on_boundary = (
                np.isclose(best_row['beta_scale'], beta_min) or
                np.isclose(best_row['beta_scale'], beta_max) or
                np.isclose(best_row['delta_scale'], delta_min) or
                np.isclose(best_row['delta_scale'], delta_max)
            )
            if on_boundary:
                st.warning(
                    "热力图峰值落在边界，通常表示当前扫描范围可能还没覆盖真实最优区间。"
                    "这不一定是算法错误，更常见是参数网格太窄。建议扩大 beta/delta 扫描区间并提高网格密度复核。"
                )
            else:
                st.success("热力图最优点位于内部区域，当前参数扫描范围对最优点覆盖较充分。")
        st.dataframe(sensitivity_df, use_container_width=True)
