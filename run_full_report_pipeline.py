import argparse
import copy
import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm


def _posterior_mean_s(df: pd.DataFrame) -> pd.Series:
    if "S" in df.columns:
        return df["S"]
    return df["S_alpha_post"] / (df["S_alpha_post"] + df["S_beta_post"])


def _safe_sensitivity_labels(values: pd.Series) -> pd.Series:
    clean_values = pd.Series(values).astype(float).clip(1e-4, 1 - 1e-4)
    ranked = clean_values.rank(method="first")
    try:
        n_bins = max(1, min(3, int(ranked.nunique())))
        labels = pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        labels = pd.Series(np.zeros(len(clean_values), dtype=int), index=clean_values.index)
    return labels.map({0: "低敏感", 1: "中敏感", 2: "高敏感"}).fillna("中敏感")


def _build_stage_difference_df(user_features: pd.DataFrame, engine) -> pd.DataFrame:
    rows = []
    sample_df = user_features.groupby("segment", group_keys=False).apply(
        lambda x: x.sample(min(len(x), 120), random_state=42)
    ).reset_index(drop=True)
    for segment, group in sample_df.groupby("segment"):
        p1_list, p2_list, p3_list = [], [], []
        for _, user in group.iterrows():
            response = engine.funnel_response(user, coupon=None, time_slot="15:00")
            p1_list.append(response["p1"])
            p2_list.append(response["p2"])
            p3_list.append(response["p3_base"])
        rows.extend([
            {"segment": segment, "stage": "p1", "probability": np.mean(p1_list)},
            {"segment": segment, "stage": "p2", "probability": np.mean(p2_list)},
            {"segment": segment, "stage": "p3", "probability": np.mean(p3_list)},
        ])
    return pd.DataFrame(rows)


def _build_segment_lift_df(user_features: pd.DataFrame, engine, coupon: str) -> pd.DataFrame:
    sample_df = user_features.copy()
    sample_df["S_mean"] = _posterior_mean_s(sample_df)
    sample_df["敏感度分组"] = _safe_sensitivity_labels(sample_df["S_mean"])
    rows = []
    for label, group in sample_df.groupby("敏感度分组"):
        no_coupon, with_coupon = [], []
        for _, user in group.sample(min(len(group), 120), random_state=42).iterrows():
            base = engine.funnel_response(user, coupon=None, time_slot="15:00")
            treat = engine.funnel_response(user, coupon=coupon, time_slot="15:00")
            no_coupon.append(base["prob_no_coupon"])
            with_coupon.append(treat["prob_with_coupon"])
        rows.append({
            "敏感度分组": label,
            "prob_no_coupon": np.mean(no_coupon),
            "prob_with_coupon": np.mean(with_coupon),
            "lift": np.mean(with_coupon) - np.mean(no_coupon),
        })
    return pd.DataFrame(rows)


def _build_coverage_df(final_user_pool: pd.DataFrame, allocation: Dict[str, str]) -> pd.DataFrame:
    coverage = final_user_pool[["user_id", "segment"]].copy()
    coverage["coupon"] = coverage["user_id"].map(allocation)
    coverage["是否发券"] = np.where(coverage["coupon"].notna(), "发券", "不发券")
    return coverage.groupby(["segment", "是否发券"]).size().reset_index(name="count")


def _save_json(path: str, payload: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def _coupon_value_map(global_params: Dict) -> Dict[str, float]:
    catalog = global_params.get("coupon_catalog", {}) or {}
    return {name: float(meta.get("coupon_value", 0.0)) for name, meta in catalog.items()}


def _coupon_mix_info(allocation: Dict[str, str]):
    allocation_series = pd.Series(allocation)
    covered_series = allocation_series.dropna()
    covered_users = int(len(covered_series))
    if covered_users == 0:
        return "None", "{}", 0
    mix = covered_series.value_counts(normalize=True).round(6).to_dict()
    top_coupon = covered_series.value_counts().idxmax()
    return str(top_coupon), json.dumps(mix, ensure_ascii=False), covered_users


def _summarize_monte_carlo(mc_df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        s = pd.to_numeric(mc_df[metric], errors="coerce").dropna()
        if s.empty:
            rows.append({
                "metric": metric,
                "mean": 0.0,
                "std": 0.0,
                "p2_5": 0.0,
                "p10": 0.0,
                "median": 0.0,
                "p90": 0.0,
                "p97_5": 0.0,
                "min": 0.0,
                "max": 0.0,
            })
            continue
        rows.append({
            "metric": metric,
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "p2_5": float(s.quantile(0.025)),
            "p10": float(s.quantile(0.10)),
            "median": float(s.quantile(0.50)),
            "p90": float(s.quantile(0.90)),
            "p97_5": float(s.quantile(0.975)),
            "min": float(s.min()),
            "max": float(s.max()),
        })
    return pd.DataFrame(rows)


def _allocate_random(precomputed, candidate_coupons, budget, coupon_cap_share, seed=42):
    rng = np.random.default_rng(seed)
    remaining = float(budget)
    user_ids = list(precomputed.keys())
    allocation = {uid: None for uid in user_ids}
    cap = max(1, int(len(user_ids) * float(coupon_cap_share)))
    usage = {c: 0 for c in candidate_coupons}

    rows = []
    for uid in user_ids:
        for c in candidate_coupons:
            item = precomputed[uid].get(c, {})
            cost = float(item.get("cost", 0.0))
            value = float(item.get("value", 0.0))
            if cost <= 0:
                continue
            rows.append((uid, c, cost, value))
    rng.shuffle(rows)

    for uid, c, cost, value in rows:
        if allocation[uid] is not None:
            continue
        if usage[c] >= cap:
            continue
        if cost > remaining:
            continue
        if value <= 0:
            continue
        allocation[uid] = c
        usage[c] += 1
        remaining -= cost
    return allocation


def _allocate_heuristic_by_value(user_features, precomputed, coupons_sorted, budget, coupon_cap_share):
    remaining = float(budget)
    user_ids = user_features.sort_values("V", ascending=False)["user_id"].tolist()
    allocation = {uid: None for uid in precomputed.keys()}
    cap = max(1, int(len(user_ids) * float(coupon_cap_share)))
    usage = {c: 0 for c in coupons_sorted}

    if not coupons_sorted:
        return allocation
    low = coupons_sorted[0]
    high = coupons_sorted[-1]
    mid = coupons_sorted[len(coupons_sorted) // 2]

    n = len(user_ids)
    for i, uid in enumerate(user_ids):
        if uid not in precomputed:
            continue
        if i < n * 0.33:
            c = high
        elif i < n * 0.70:
            c = mid
        else:
            c = low
        item = precomputed[uid].get(c, {})
        cost = float(item.get("cost", 0.0))
        value = float(item.get("value", 0.0))
        if cost <= 0 or value <= 0:
            continue
        if usage.get(c, 0) >= cap:
            continue
        if cost > remaining:
            continue
        allocation[uid] = c
        usage[c] = usage.get(c, 0) + 1
        remaining -= cost
    return allocation


def run_pipeline(
    data_dir: str,
    output_root: str,
    n_users: int,
    random_state: int,
    budget: float,
    epochs: int,
    coupon_cap_share: float,
    alpha_ucb: float,
    mc_runs: int,
    use_auto_distribution_selector: bool,
    bandit_param_grid: List[Dict],
    heatmap_scales: Dict,
):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    from feature_engineering import FeatureEngineering
    from optimization import Optimization
    from simulation_engine import SimulationEngine
    from visualization import Visualization

    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    out_dir = os.path.join(output_root, f"full-report-{ts}")
    fig_dir = os.path.join(out_dir, "figures")
    table_dir = os.path.join(out_dir, "tables")
    meta_dir = os.path.join(out_dir, "meta")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    viz = Visualization()

    # 1) Build semi-synthetic environment
    print("Step 1: 构建半合成仿真环境...")
    fe = FeatureEngineering(use_api=False)
    ok = fe.load_data(data_dir)
    if not ok:
        raise RuntimeError(f"数据加载失败，请检查目录: {data_dir}")
    print("正在生成半合成用户...")
    user_features, global_params = fe.build_features_and_params(
        n_users=n_users,
        random_state=random_state,
        use_auto_distribution_selector=use_auto_distribution_selector,
    )
    engine = SimulationEngine(user_features, global_params)
    print(f"生成用户数: {len(user_features)}")

    # 2) Baseline simulation and coupon comparisons
    print("\nStep 2: 基准仿真和券对比...")
    baseline_strategy = {uid: None for uid in user_features["user_id"]}
    baseline_expected = engine.calculate_expected_metrics(baseline_strategy)
    simulation_rows = [{"strategy": "不发券", **baseline_expected}]

    coupon_list = list(global_params.get("candidate_coupons", []))
    print(f"正在评估 {len(coupon_list)} 种券型...")
    for coupon in tqdm(coupon_list, desc="评估券型"):
        s = {uid: coupon for uid in user_features["user_id"]}
        metrics = engine.calculate_expected_metrics(s)
        simulation_rows.append({"strategy": f"统一发{coupon}", **metrics})
    simulation_df = pd.DataFrame(simulation_rows)
    simulation_df.to_csv(os.path.join(table_dir, "simulation_strategy_comparison.csv"), index=False, encoding="utf-8-sig")

    selected_coupon = coupon_list[0] if coupon_list else None
    if selected_coupon:
        print(f"\n正在运行 Monte Carlo 仿真 ({mc_runs} 轮)...")
        sim_strategy = {uid: selected_coupon for uid in user_features["user_id"]}
        mc_df = pd.DataFrame(engine.monte_carlo_runs(sim_strategy, n_runs=mc_runs, seed=random_state))
        mc_df.to_csv(os.path.join(table_dir, "simulation_monte_carlo.csv"), index=False, encoding="utf-8-sig")
        print("正在计算分段提升...")
        segment_lift_df = _build_segment_lift_df(user_features, engine, selected_coupon)
        segment_lift_df.to_csv(os.path.join(table_dir, "simulation_segment_lift.csv"), index=False, encoding="utf-8-sig")
    else:
        mc_df = pd.DataFrame()
        segment_lift_df = pd.DataFrame()

    # 3) Optimization + fair baselines + Bandit 参数检验实验
    print("\nStep 3: 优化和参数检验...")
    bandit_runs = []
    opt_result = None
    print(f"正在运行 {len(bandit_param_grid)} 个 Bandit 配置...")
    for idx, spec in enumerate(tqdm(bandit_param_grid, desc="Bandit 配置")):
        opt = Optimization(
            user_features=user_features,
            global_params=global_params,
            order_df=fe.order_df,
            coupon_cap_share=float(spec.get("coupon_cap_share", coupon_cap_share)),
            alpha_ucb=float(spec.get("alpha_ucb", alpha_ucb)),
        )
        res = opt.optimize(budget=budget, epochs=int(spec.get("epochs", epochs)), show_progress=False)
        run_name = spec.get("name", f"bandit_{idx+1}")
        metrics = res["optimized_metrics"]
        bandit_runs.append({
            "run": run_name,
            "budget": float(budget),
            "epochs": int(spec.get("epochs", epochs)),
            "coupon_cap_share": float(spec.get("coupon_cap_share", coupon_cap_share)),
            "alpha_ucb": float(spec.get("alpha_ucb", alpha_ucb)),
            "delta_gtv_causal": float(metrics.get("delta_gtv_causal", 0.0)),
            "roi": float(metrics.get("roi", 0.0)),
            "total_cost": float(metrics.get("total_cost", 0.0)),
            "covered_users": int((pd.Series(res["allocation"]).notna()).sum()),
        })
        # keep the first run as the "main" artifact for figures/tables
        if opt_result is None:
            opt_result = res

    pd.DataFrame(bandit_runs).to_csv(os.path.join(table_dir, "bandit_param_sweep.csv"), index=False, encoding="utf-8-sig")

    pd.DataFrame(opt_result["history"]).to_csv(os.path.join(table_dir, "optimization_history.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(opt_result["allocation_df"]).to_csv(os.path.join(table_dir, "optimization_allocation.csv"), index=False, encoding="utf-8-sig")
    coverage_df = _build_coverage_df(opt_result["final_user_pool"], opt_result["allocation"])
    coverage_df.to_csv(os.path.join(table_dir, "optimization_coverage.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(opt_result["strategies"]).T.reset_index().rename(columns={"index": "strategy"}).to_csv(
        os.path.join(table_dir, "optimization_strategies.csv"), index=False, encoding="utf-8-sig"
    )

    # 3a) Bandit 主策略 Monte Carlo（固定 allocation，仅改变随机种子）
    bandit_allocation = opt_result["allocation"]
    top_coupon, coupon_mix_json, covered_users = _coupon_mix_info(bandit_allocation)
    bandit_mc_rows = []
    for i in range(mc_runs):
        run_seed = int(random_state + i)
        mc_metrics = engine.calculate_simulated_metrics(bandit_allocation, seed=run_seed)
        bandit_mc_rows.append({
            "run": i + 1,
            "seed": run_seed,
            "strategy": "Bandit_main",
            "delta_gtv_causal": float(mc_metrics.get("delta_gtv_causal", 0.0)),
            "delta_gtv": float(mc_metrics.get("delta_gtv", 0.0)),
            "roi": float(mc_metrics.get("roi", 0.0)),
            "total_cost": float(mc_metrics.get("total_cost", 0.0)),
            "covered_users": int(covered_users),
        })
    bandit_mc_df = pd.DataFrame(bandit_mc_rows)
    bandit_mc_df.to_csv(os.path.join(table_dir, "bandit_monte_carlo.csv"), index=False, encoding="utf-8-sig")
    _summarize_monte_carlo(
        bandit_mc_df,
        metrics=["delta_gtv_causal", "delta_gtv", "roi", "total_cost"]
    ).to_csv(os.path.join(table_dir, "bandit_monte_carlo_summary.csv"), index=False, encoding="utf-8-sig")

    # 3a-2) 新增模型敏感性输出（在 Bandit 主策略 allocation 下评估）
    causal_rows = []
    for mode in ["none", "linear", "quadratic", "piecewise", "logistic"]:
        params = {"k": 12, "b0": 0.25} if mode == "logistic" else {}
        gp = copy.deepcopy(global_params)
        gp["causal_discount_mode"] = mode
        gp["causal_discount_params"] = params
        sim = SimulationEngine(user_features, gp)
        m = sim.calculate_expected_metrics(bandit_allocation)
        causal_rows.append({
            "causal_discount_mode": mode,
            "causal_discount_params": json.dumps(params, ensure_ascii=False),
            "delta_gtv": float(m.get("delta_gtv", 0.0)),
            "delta_gtv_causal": float(m.get("delta_gtv_causal", 0.0)),
            "roi": float(m.get("roi", 0.0)),
            "total_cost": float(m.get("total_cost", 0.0)),
            "covered_users": int(covered_users),
            "avg_beta": float(m.get("avg_beta", 0.0)),
            "avg_causal_discount_weight": float(m.get("avg_causal_discount_weight", 0.0)),
            "top_coupon": top_coupon,
            "coupon_mix_json": coupon_mix_json,
        })
    pd.DataFrame(causal_rows).to_csv(
        os.path.join(table_dir, "causal_discount_sensitivity.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    stage_rows = []
    for mode in ["p3_only", "p2_p3", "p1_p2_p3_v"]:
        params = {}
        gp = copy.deepcopy(global_params)
        gp["coupon_response_mode"] = mode
        gp["coupon_response_params"] = params
        sim = SimulationEngine(user_features, gp)
        m = sim.calculate_expected_metrics(bandit_allocation)
        stage_rows.append({
            "coupon_response_mode": mode,
            "coupon_response_params": json.dumps(params, ensure_ascii=False),
            "delta_gtv": float(m.get("delta_gtv", 0.0)),
            "delta_gtv_causal": float(m.get("delta_gtv_causal", 0.0)),
            "roi": float(m.get("roi", 0.0)),
            "total_cost": float(m.get("total_cost", 0.0)),
            "covered_users": int(covered_users),
            "avg_p1": float(m.get("avg_p1", 0.0)),
            "avg_p2": float(m.get("avg_p2", 0.0)),
            "avg_p3_base": float(m.get("avg_p3_base", 0.0)),
            "avg_p1_coupon": float(m.get("avg_p1_coupon", 0.0)),
            "avg_p2_coupon": float(m.get("avg_p2_coupon", 0.0)),
            "avg_p3_coupon": float(m.get("avg_p3_coupon", 0.0)),
            "avg_delta_p1": float(m.get("avg_delta_p1", 0.0)),
            "avg_delta_p2": float(m.get("avg_delta_p2", 0.0)),
            "avg_delta_p3": float(m.get("avg_delta_p3", 0.0)),
            "top_coupon": top_coupon,
            "coupon_mix_json": coupon_mix_json,
        })
    pd.DataFrame(stage_rows).to_csv(
        os.path.join(table_dir, "coupon_stage_response_sensitivity.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 3b) Experiment suite (E1~E6) for report validation
    # E1 不发券；E2 随机发券；E3 统一最低面额；E4 统一最高面额；E5 规则启发式；E6 alpha=0 无探索Bandit
    print("\nStep 3b: 实验套件 (E1~E6)...")
    print("正在预计算用户-券指标...")
    precomputed = opt.precompute_metrics(show_progress=False)
    coupon_values = _coupon_value_map(global_params)
    candidate_coupons = list(global_params.get("candidate_coupons", []))
    coupons_sorted = sorted(candidate_coupons, key=lambda c: coupon_values.get(c, 0.0))
    lowest_coupon = coupons_sorted[0] if coupons_sorted else None
    highest_coupon = coupons_sorted[-1] if coupons_sorted else None

    allocations = {}
    allocations["E1_不发券(自然转化)"] = {uid: None for uid in user_features["user_id"]}
    print("正在计算 E2_随机发券...")
    allocations["E2_随机发券"] = _allocate_random(
        precomputed, candidate_coupons, budget=float(budget), coupon_cap_share=float(coupon_cap_share), seed=random_state
    )
    if lowest_coupon:
        print("正在计算 E3_统一发最低面额券...")
        allocations["E3_统一发最低面额券"] = opt._uniform_budget_allocation(
            precomputed, budget=float(budget), coupon=lowest_coupon, rank_by="roi"
        )
    else:
        allocations["E3_统一发最低面额券"] = allocations["E1_不发券(自然转化)"]
    if highest_coupon:
        print("正在计算 E4_统一发最高面额券...")
        allocations["E4_统一发最高面额券"] = opt._uniform_budget_allocation(
            precomputed, budget=float(budget), coupon=highest_coupon, rank_by="roi"
        )
    else:
        allocations["E4_统一发最高面额券"] = allocations["E1_不发券(自然转化)"]
    print("正在计算 E5_规则启发式...")
    allocations["E5_规则启发式(按历史客单价分层)"] = _allocate_heuristic_by_value(
        user_features, precomputed, coupons_sorted, budget=float(budget), coupon_cap_share=float(coupon_cap_share)
    )
    print("正在计算 E6_Bandit仅利用(alpha=0)...")
    opt_no_explore = Optimization(
        user_features=user_features,
        global_params=global_params,
        order_df=fe.order_df,
        coupon_cap_share=float(coupon_cap_share),
        alpha_ucb=0.0,
    )
    res_no_explore = opt_no_explore.optimize(budget=float(budget), epochs=int(epochs), show_progress=False)
    allocations["E6_Bandit仅利用(alpha=0)"] = res_no_explore["allocation"]

    # Optional: also keep Bandit main config as reference row
    allocations["B_Bandit(主配置)"] = opt_result["allocation"]

    suite_rows = []
    print("正在评估所有实验...")
    for exp_name, alloc in tqdm(allocations.items(), desc="评估实验"):
        metrics = engine.calculate_expected_metrics(alloc)
        covered = int((pd.Series(alloc).notna()).sum())
        suite_rows.append({
            "experiment": exp_name,
            "covered_users": covered,
            "total_cost": float(metrics.get("total_cost", 0.0)),
            "delta_gtv_causal": float(metrics.get("delta_gtv_causal", 0.0)),
            "roi": float(metrics.get("roi", 0.0)),
            "orders_with_coupon": float(metrics.get("orders_with_coupon", 0.0)),
            "gtv_no_coupon": float(metrics.get("gtv_no_coupon", 0.0)),
            "gtv_with_coupon": float(metrics.get("gtv_with_coupon", 0.0)),
        })
        pd.Series(alloc).rename("coupon").reset_index().rename(columns={"index": "user_id"}).to_csv(
            os.path.join(table_dir, f"allocation_{exp_name}.csv"),
            index=False,
            encoding="utf-8-sig"
        )
    pd.DataFrame(suite_rows).sort_values(["roi", "delta_gtv_causal"], ascending=False).to_csv(
        os.path.join(table_dir, "experiment_suite_E1_E6.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 4) Sensitivity / robustness
    print("\nStep 4: 敏感性分析...")
    print("正在运行敏感性分析...")
    sensitivity_df = opt.sensitivity_analysis(budget=budget, epochs=max(1, min(epochs, 2)))
    sensitivity_df.to_csv(os.path.join(table_dir, "sensitivity_analysis.csv"), index=False, encoding="utf-8-sig")
    
    beta_scales = heatmap_scales.get("beta_scales", [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    delta_scales = heatmap_scales.get("delta_scales", [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    print(f"正在生成热力图 ({len(beta_scales)}x{len(delta_scales)} 网格)...")
    heatmap_df = opt.sensitivity_heatmap(
        budget=budget,
        beta_scales=beta_scales,
        delta_scales=delta_scales,
        epochs=1,
    )
    heatmap_df.to_csv(os.path.join(table_dir, "sensitivity_heatmap_grid.csv"), index=False, encoding="utf-8-sig")
    
    print("正在计算预算前沿...")
    frontier_df = opt.budget_frontier(
        budget_list=sorted(set([max(100, budget * x) for x in [0.4, 0.7, 1.0, 1.3, 1.6]])),
        epochs=max(1, min(epochs, 2)),
    )
    frontier_df.to_csv(os.path.join(table_dir, "budget_frontier.csv"), index=False, encoding="utf-8-sig")

    # 5) Plot key figures for report
    print("\nStep 5: 生成图表...")
    figs = {
        "profile_distribution.png": viz.plot_profile_distribution(fe.user_profile, save=False),
        "scene_overview.png": viz.plot_scene_overview(fe.order_df, fe.scene_orders, save=False),
        "order_amount_analysis.png": viz.plot_order_amount_analysis(fe.order_df, fe.scene_orders, save=False),
        "funnel.png": viz.plot_funnel(global_params["funnel"], save=False),
        "distribution_fit.png": viz.plot_distribution_fit(user_features, global_params, save=False),
        "generated_user_map.png": viz.plot_generated_user_map(user_features, save=False),
        "calibration.png": viz.plot_calibration(global_params["calibration"], save=False),
        "stage_difference.png": viz.plot_stage_difference(_build_stage_difference_df(user_features, engine), save=False),
        "strategy_comparison.png": viz.plot_strategy_comparison(opt_result["strategies"], save=False),
        "bandit_history.png": viz.plot_bandit_history(opt_result["history"], save=False),
        "allocation_mix.png": viz.plot_allocation_mix(opt_result["allocation_df"], save=False),
        "coverage_by_segment.png": viz.plot_coverage_by_segment(coverage_df, save=False),
        "sensitivity.png": viz.plot_sensitivity(sensitivity_df, save=False),
        "sensitivity_heatmap_roi.png": viz.plot_sensitivity_heatmap(heatmap_df, value_col="roi", save=False),
        "budget_frontier.png": viz.plot_budget_frontier(frontier_df, save=False),
    }
    if not mc_df.empty:
        figs["monte_carlo_distribution.png"] = viz.plot_monte_carlo_distribution(mc_df, save=False)
    if not segment_lift_df.empty:
        figs["segment_lift.png"] = viz.plot_segment_lift(segment_lift_df, save=False)

    import matplotlib.pyplot as plt
    print(f"正在保存 {len(figs)} 个图表...")
    for filename, fig in tqdm(figs.items(), desc="保存图表"):
        if fig is None:
            continue
        fig.savefig(os.path.join(fig_dir, filename), dpi=320, bbox_inches="tight")
        plt.close(fig)

    # 6) A/B design template (non-visual deliverable)
    ab_design = {
        "objective": "验证 Bandit 个体化发券相对统一发券/不发券的增量GTV与ROI提升",
        "groups": [
            {"group": "A 控制组", "strategy": "统一发券（同预算口径）或不发券", "traffic": "随机抽样"},
            {"group": "B 实验组", "strategy": "Bandit 个体化发券", "traffic": "随机抽样"},
        ],
        "core_metrics": ["增量GTV", "补贴成本", "ROI", "转化率提升", "复购率"],
        "duration_recommendation": "至少2周，覆盖工作日/周末，建议分层随机（会员等级、历史客单、城市）",
    }
    _save_json(os.path.join(meta_dir, "ab_test_design.json"), ab_design)

    # 7) Experiment checklist and run summary
    experiment_checklist = [
        {"module": "主实验", "item": "半合成建模 + Funnel仿真 + Bandit优化", "output": "optimization_strategies.csv"},
        {"module": "对比实验", "item": "不发券 vs 统一发券(同预算ROI/GTV) vs Bandit", "output": "optimization_strategies.csv"},
        {"module": "对比实验", "item": "E1~E6 实验套件（随机/一刀切/大券/规则/无探索）", "output": "experiment_suite_E1_E6.csv"},
        {"module": "鲁棒性", "item": "敏感性分析 + 热力图网格", "output": "sensitivity_analysis.csv / sensitivity_heatmap_grid.csv"},
        {"module": "鲁棒性", "item": "因果折减函数敏感性", "output": "causal_discount_sensitivity.csv"},
        {"module": "鲁棒性", "item": "多阶段券响应敏感性", "output": "coupon_stage_response_sensitivity.csv"},
        {"module": "预算分析", "item": "预算前沿（增量与ROI）", "output": "budget_frontier.csv"},
        {"module": "随机性检验", "item": "Monte Carlo收益分布", "output": "simulation_monte_carlo.csv"},
        {"module": "随机性检验", "item": "Bandit主策略 Monte Carlo 分布与统计", "output": "bandit_monte_carlo.csv / bandit_monte_carlo_summary.csv"},
        {"module": "业务实验设计", "item": "A/B Test 回测方案", "output": "ab_test_design.json"},
    ]
    pd.DataFrame(experiment_checklist).to_csv(os.path.join(meta_dir, "experiment_checklist.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "output_dir": out_dir,
        "config": {
            "data_dir": data_dir,
            "n_users": n_users,
            "random_state": random_state,
            "budget": budget,
            "epochs": epochs,
            "coupon_cap_share": coupon_cap_share,
            "alpha_ucb": alpha_ucb,
            "mc_runs": mc_runs,
        },
        "headline_metrics": {
            "bandit_delta_gtv_causal": float(opt_result["optimized_metrics"]["delta_gtv_causal"]),
            "bandit_roi": float(opt_result["optimized_metrics"]["roi"]),
            "bandit_total_cost": float(opt_result["optimized_metrics"]["total_cost"]),
            "scene_orders": int(global_params["calibration"]["raw_scene_orders"]),
            "scene_users": int(global_params["calibration"]["raw_scene_users"]),
        },
    }
    _save_json(os.path.join(meta_dir, "run_summary.json"), summary)

    # 8) Extra hypothesis-check experiments for FULL runs (auto distribution selector on/off)
    # For fast mode we already run only one setup.
    if heatmap_scales.get("mode") == "full":
        print("\nStep 8: 分布选择器对比实验...")
        exp_rows = []
        for flag in tqdm([True, False], desc="分布选择器实验"):
            fe2 = FeatureEngineering(use_api=False)
            ok2 = fe2.load_data(data_dir)
            if not ok2:
                continue
            uf2, gp2 = fe2.build_features_and_params(
                n_users=n_users,
                random_state=random_state,
                use_auto_distribution_selector=flag,
            )
            opt2 = Optimization(
                user_features=uf2,
                global_params=gp2,
                order_df=fe2.order_df,
                coupon_cap_share=coupon_cap_share,
                alpha_ucb=alpha_ucb,
            )
            res2 = opt2.optimize(budget=budget, epochs=epochs, show_progress=False)
            m2 = res2["optimized_metrics"]
            exp_rows.append({
                "auto_distribution_selector": bool(flag),
                "delta_gtv_causal": float(m2.get("delta_gtv_causal", 0.0)),
                "roi": float(m2.get("roi", 0.0)),
                "total_cost": float(m2.get("total_cost", 0.0)),
                "scene_orders": int(gp2["calibration"]["raw_scene_orders"]),
                "scene_users": int(gp2["calibration"]["raw_scene_users"]),
            })
        if exp_rows:
            pd.DataFrame(exp_rows).to_csv(os.path.join(table_dir, "distribution_selector_ablation.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# 全自动实验产出\n\n"
            f"- 生成时间: {ts}\n"
            f"- 输出目录: `{out_dir}`\n\n"
            "## 目录说明\n"
            "- `figures/`: 报告可直接使用的图表\n"
            "- `tables/`: 分析与对比实验CSV结果\n"
            "- `meta/`: 运行摘要、实验清单、A/B测试设计\n"
        )

    print(f"\n完成！输出保存到: {out_dir}")
    print(f"\n生成的文件：")
    print(f"- 图表: {len(figs)} 个")
    print(f"- 表格: {len(os.listdir(table_dir))} 个")
    print(f"- 元数据: {len(os.listdir(meta_dir))} 个")


def main():
    parser = argparse.ArgumentParser(description="一键全自动运行报告实验并落盘")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--fast", action="store_true", help="快速模式：少量用户/粗网格/少轮次，适合快速出报告草稿")
    mode_group.add_argument("--full", action="store_true", help="完整版：更多用户/更密网格/更多对比实验，适合最终版")
    parser.add_argument("--data-dir", default="./data/raw", help="原始数据目录（可传 ./data 或 ./data/raw）")
    parser.add_argument("--output-root", default="./exports/auto_runs", help="自动运行输出根目录")
    parser.add_argument("--n-users", type=int, default=1000, help="生成半合成用户数")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--budget", type=float, default=3000, help="优化预算")
    parser.add_argument("--epochs", type=int, default=3, help="Bandit训练轮次")
    parser.add_argument("--coupon-cap-share", type=float, default=0.35, help="单券型覆盖上限")
    parser.add_argument("--alpha-ucb", type=float, default=0.25, help="Bandit探索强度")
    parser.add_argument("--mc-runs", type=int, default=50, help="Monte Carlo轮数")
    args = parser.parse_args()

    mode = "full" if args.full else "fast" if args.fast else "fast"
    if mode == "fast":
        n_users = min(args.n_users, 800)
        epochs = min(args.epochs, 2)
        mc_runs = args.mc_runs
        heatmap_scales = {"mode": "fast", "beta_scales": [0.8, 0.9, 1.0, 1.1, 1.2], "delta_scales": [0.8, 0.9, 1.0, 1.1, 1.2]}
        # 只跑一个 bandit 配置
        bandit_param_grid = [{"name": "bandit_main", "epochs": epochs, "coupon_cap_share": args.coupon_cap_share, "alpha_ucb": args.alpha_ucb}]
        use_auto_distribution_selector = True
    else:
        n_users = args.n_users
        epochs = args.epochs
        mc_runs = args.mc_runs
        heatmap_scales = {"mode": "full", "beta_scales": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], "delta_scales": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]}
        # 参数检验：不同 alpha/cap/epochs
        bandit_param_grid = [
            {"name": "bandit_main", "epochs": epochs, "coupon_cap_share": args.coupon_cap_share, "alpha_ucb": args.alpha_ucb},
            {"name": "alpha_low", "epochs": epochs, "coupon_cap_share": args.coupon_cap_share, "alpha_ucb": max(0.0, args.alpha_ucb * 0.5)},
            {"name": "alpha_high", "epochs": epochs, "coupon_cap_share": args.coupon_cap_share, "alpha_ucb": min(2.0, args.alpha_ucb * 1.8)},
            {"name": "cap_tight", "epochs": epochs, "coupon_cap_share": max(0.10, args.coupon_cap_share * 0.7), "alpha_ucb": args.alpha_ucb},
            {"name": "cap_loose", "epochs": epochs, "coupon_cap_share": min(0.80, args.coupon_cap_share * 1.3), "alpha_ucb": args.alpha_ucb},
            {"name": "epochs_plus", "epochs": min(10, epochs + 2), "coupon_cap_share": args.coupon_cap_share, "alpha_ucb": args.alpha_ucb},
        ]
        # 假设检验：自动分布选择器 vs 手工默认
        use_auto_distribution_selector = True

    run_pipeline(
        data_dir=args.data_dir,
        output_root=args.output_root,
        n_users=n_users,
        random_state=args.random_state,
        budget=args.budget,
        epochs=epochs,
        coupon_cap_share=args.coupon_cap_share,
        alpha_ucb=args.alpha_ucb,
        mc_runs=mc_runs,
        use_auto_distribution_selector=use_auto_distribution_selector,
        bandit_param_grid=bandit_param_grid,
        heatmap_scales=heatmap_scales,
    )


if __name__ == "__main__":
    main()
