import os
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns


if not os.path.exists('pic'):
    os.makedirs('pic')


def set_font():
    """设置中文字体优先级（尽量避免黑体风格，优先宋体/苹方体系）。"""
    preferred_fonts = [
        "Songti SC", "STSong", "PingFang HK", "PingFang SC",
        "Microsoft YaHei", "Noto Serif CJK SC", "Noto Sans CJK SC", "DejaVu Sans"
    ]
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    selected_font = next((name for name in preferred_fonts if name in available_fonts), "DejaVu Sans")
    plt.rcParams["font.family"] = [selected_font]
    plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


class Visualization:
    def __init__(self):
        self.palette = {
            'gold': '#F2C511',
            'black': '#111111',
            'charcoal': '#333333',
            'gray': '#8F8F8F',
            'orange': '#E6862D',
            'grid': '#EAEAEA',
            'spine': '#DCDCDC',
            'white': '#FFFFFF',
            'sand': '#FFFBEF'
        }
        # 先设置Seaborn主题
        sns.set_theme(style="white", context="notebook")
        # 然后设置字体（这会覆盖Seaborn的默认字体设置）
        set_font()
        # 最后更新其他参数
        plt.rcParams.update({
            "figure.facecolor": self.palette["white"],
            "axes.facecolor": self.palette["white"],
            "savefig.facecolor": self.palette["white"],
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "axes.edgecolor": self.palette["spine"],
            "axes.labelcolor": self.palette["charcoal"],
            "xtick.color": self.palette["charcoal"],
            "ytick.color": self.palette["charcoal"],
            "text.color": self.palette["charcoal"],
            "axes.titlesize": 14,
            "axes.titleweight": "normal",
        })

    def _style_axis(self, ax, title, xlabel=None, ylabel=None):
        ax.set_title(title, loc='left', fontsize=14, fontweight='bold', pad=22)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=11)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11)
        ax.set_facecolor(self.palette['white'])
        ax.grid(False)
        ax.grid(axis='y', color=self.palette['grid'], linestyle='-', linewidth=0.8, alpha=0.65)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(self.palette['spine'])
            ax.spines[spine].set_linewidth(1.0)
        # 调整刻度标签字体大小和对齐方式
        ax.tick_params(axis='x', labelsize=10)
        # y轴坐标水平显示
        ax.tick_params(axis='y', labelsize=10, rotation=0)
        # 由各绘图函数统一控制布局，避免标题/副标题被裁切

    def _add_subtitle(self, ax, subtitle):
        ax.annotate(
            subtitle,
            xy=(0.0, 1.0),
            xycoords='axes fraction',
            xytext=(0, 4),
            textcoords='offset points',
            ha='left',
            va='bottom',
            fontsize=9.6,
            color=self.palette['gray'],
            clip_on=False
        )

    def _add_conclusion(self, ax, text):
        ax.text(
            0.98, 0.96, text,
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=10.5, color=self.palette['charcoal'],
            bbox=dict(facecolor='#FFF7D6', edgecolor='#E5C14A', boxstyle='round,pad=0.35')
        )

    def plot_scene_overview(self, order_df, scene_orders, save=True):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=self.palette['white'])
        hour_counts = order_df['下单时间'].apply(lambda x: x.hour if hasattr(x, 'hour') else 0).value_counts().sort_index()
        sns.barplot(x=hour_counts.index, y=hour_counts.values, ax=axes[0], color=self.palette['gold'], edgecolor=self.palette['orange'], width=0.55)
        axes[0].axvspan(14, 16, color=self.palette['gold'], alpha=0.15)
        self._style_axis(axes[0], '全量订单小时分布', '小时', '订单量')
        self._add_subtitle(axes[0], '下午茶订单在 14-16 点显著集中，时段切片合理')
        self._add_conclusion(axes[0], '场景先从时间上切片，再叠加业态过滤')

        summary = pd.DataFrame({
            'scope': ['全量订单', '下午茶场景订单'],
            'orders': [len(order_df), len(scene_orders)]
        })
        sns.barplot(data=summary, x='scope', y='orders', ax=axes[1], palette=[self.palette['gray'], self.palette['gold']], width=0.55)
        self._style_axis(axes[1], '场景过滤后的订单规模', '', '订单量')
        self._add_subtitle(axes[1], '场景过滤显著收敛了样本范围，减少非目标订单干扰')
        self._add_conclusion(axes[1], f'当前场景订单数 = {len(scene_orders)}')
        for i, v in enumerate(summary['orders']):
            axes[1].text(i, v + max(summary['orders']) * 0.03, f'{int(v)}', ha='center', fontsize=11, fontweight='bold')
        plt.tight_layout()
        if save:
            plt.savefig('pic/scene_overview.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_distribution_fit(self, users, global_params, save=True):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=self.palette['white'])
        sns.histplot(users['V'], bins=30, kde=True, ax=axes[0], color=self.palette['gold'], edgecolor=self.palette['orange'])
        self._style_axis(axes[0], '生成用户价值 V 分布', 'V', '用户数')
        self._add_subtitle(axes[0], '用户价值分布稳定，能支撑分层发券策略')
        self._add_conclusion(axes[0], f"均值 = {users['V'].mean():.1f}")

        sns.histplot(users['A'], bins=30, kde=True, ax=axes[1], color=self.palette['gray'], edgecolor=self.palette['orange'])
        self._style_axis(axes[1], '生成用户活跃度 A 分布', 'A', '用户数')
        self._add_subtitle(axes[1], '活跃度呈明显分层，高活跃用户更适合优先触达')
        self._add_conclusion(axes[1], f"均值 = {users['A'].mean():.2f}")

        sensitivity = users['S'] if 'S' in users.columns else users['S_alpha_post'] / (users['S_alpha_post'] + users['S_beta_post'])
        sns.histplot(sensitivity, bins=30, kde=True, ax=axes[2], color=self.palette['orange'], edgecolor=self.palette['gray'])
        self._style_axis(axes[2], '生成用户价格敏感度 S 分布', 'S', '用户数')
        self._add_subtitle(axes[2], '价格敏感度区分度清晰，支持差异化券面分配')
        self._add_conclusion(axes[2], f"均值 = {sensitivity.mean():.2f}")
        plt.tight_layout()
        if save:
            plt.savefig('pic/distribution_fit.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_funnel(self, funnel_stats, save=True):
        stages = ['view', 'click', 'cart', 'purchase']
        values = [1.0, funnel_stats['p1'], funnel_stats['p1'] * funnel_stats['p2'], funnel_stats['p1'] * funnel_stats['p2'] * funnel_stats['p3']]
        fig, ax = plt.subplots(figsize=(12, 7), facecolor=self.palette['white'])
        sns.barplot(x=stages, y=values, ax=ax, palette=[self.palette['gray'], self.palette['gold'], self.palette['orange'], self.palette['gray']], width=0.55)
        # 调整主标题和副标题位置
        ax.set_title('下午茶场景 Funnel 结构', loc='left', fontsize=14, fontweight='bold', pad=20)
        ax.text(
            0.0, 1.008, '漏斗结构稳定，后续优化可聚焦在末端转化提升',
            transform=ax.transAxes,
            ha='left', va='bottom',
            fontsize=10, color=self.palette['gray']
        )
        if funnel_stats:
            ax.set_ylabel('相对通过率', fontsize=11)
        ax.set_xlabel('阶段', fontsize=11)
        ax.set_facecolor(self.palette['white'])
        ax.grid(False)
        ax.grid(axis='y', color=self.palette['grid'], linestyle='-', linewidth=0.8, alpha=0.65)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(self.palette['spine'])
            ax.spines[spine].set_linewidth(1.0)
        ax.tick_params(axis='both', labelsize=10)
        self._add_conclusion(ax, f"beta 基线 = {values[-1]:.2f}")
        ax.set_ylim(0, 1.05)
        for i, v in enumerate(values):
            ax.text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=10)
        plt.tight_layout()
        if save:
            plt.savefig('pic/funnel.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_generated_user_map(self, users, save=True):
        plot_df = users.copy()
        plot_df['S_display'] = plot_df['S'] if 'S' in plot_df.columns else plot_df['S_alpha_post'] / (plot_df['S_alpha_post'] + plot_df['S_beta_post'])
        # 将segment转换为中文
        plot_df['用户分层'] = plot_df['segment'].map({
            'high_value': '高价值用户',
            'mid_value': '中价值用户',
            'low_value': '低价值用户'
        })
        # 重命名A列，添加活跃度解释
        plot_df['活跃度 A'] = plot_df['A']
        fig, ax = plt.subplots(figsize=(7.8, 5.0), facecolor=self.palette['white'])
        sns.scatterplot(
            data=plot_df,
            x='V',
            y='S_display',
            size='活跃度 A',
            hue='用户分层',
            sizes=(16, 160),
            alpha=0.68,
            palette=[self.palette['gray'], self.palette['orange'], self.palette['gold']],
            ax=ax
        )
        v_threshold = float(plot_df['V'].quantile(0.75))
        s_threshold = float(plot_df['S_display'].quantile(0.75))
        x_max = float(plot_df['V'].max())
        y_max = float(plot_df['S_display'].max())
        if x_max > v_threshold and y_max > s_threshold:
            rect = plt.Rectangle(
                (v_threshold, s_threshold),
                max(x_max - v_threshold, 1e-3),
                max(y_max - s_threshold, 1e-3),
                facecolor='#F2C511',
                alpha=0.18,
                edgecolor='#C58A00',
                linestyle='--',
                linewidth=1.4
            )
            ax.add_patch(rect)
            ax.text(
                v_threshold + (x_max - v_threshold) * 0.05,
                s_threshold + (y_max - s_threshold) * 0.88,
                '高优先级关注区',
                fontsize=9.5,
                color='#9A6A00',
                fontweight='bold'
            )
        self._style_axis(ax, '生成用户状态地图', '用户价值 V', '价格敏感度 S')
        self._add_subtitle(ax, '右上高价值高敏感区域是发券优先覆盖核心人群')
        self._add_conclusion(ax, '价值与敏感度共同决定补贴优先级，重点看右上区域与高活跃点')
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        if save:
            plt.savefig('pic/generated_user_map.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_system_architecture(self, global_params, save=True):
        fig, ax = plt.subplots(figsize=(14.5, 8.2), facecolor=self.palette['white'])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        def draw_box(x, y, w, h, title, body, facecolor):
            rect = plt.Rectangle((x, y), w, h, facecolor=facecolor, edgecolor=self.palette['black'], linewidth=1.4)
            ax.add_patch(rect)
            ax.text(x + 0.02, y + h - 0.04, title, fontsize=13, fontweight='bold', ha='left', va='top')
            ax.text(x + 0.02, y + h - 0.095, body, fontsize=10.5, color=self.palette['charcoal'], ha='left', va='top')

        def draw_arrow(start, end, text=None):
            ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color=self.palette['black'], lw=1.8))
            if text:
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                ax.text(mid_x, mid_y + 0.03, text, fontsize=10, color=self.palette['gray'], ha='center')

        draw_box(0.05, 0.68, 0.22, 0.20, '原始业务数据', '用户画像表\n神券订单数据样例\n用户行为序列', '#FFF7D6')
        draw_box(0.35, 0.68, 0.25, 0.20, '场景约束与结构抽取', '下午茶时段过滤\n业态筛选\nFunnel结构提取', '#FFF2CC')
        selector_name = '自动分布选择器' if global_params.get('distribution_models') else '固定分布拟合'
        draw_box(0.68, 0.68, 0.24, 0.20, selector_name, 'V/A/S 选择 Parametric\nMixture / Empirical / Rule-based\n并保留 fallback', '#FDE6C8')
        draw_box(0.08, 0.34, 0.28, 0.20, '半合成用户生成器', '按画像约束采样 V/A/S\n生成连续用户池', '#FFFBEF')
        draw_box(0.40, 0.34, 0.24, 0.20, '仿真与因果校正', '券只作用于 p3\n计算 Delta GTV causal', '#F7E7A9')
        draw_box(0.70, 0.34, 0.22, 0.20, 'Bandit 优化器', '预算约束\nUCB探索利用平衡\n输出个体化发券', '#FFE39B')
        draw_box(0.28, 0.06, 0.42, 0.16, '结果展示与业务解读', '策略对比、敏感性分析、预算前沿、系统可解释结论', '#FFF1B8')

        draw_arrow((0.27, 0.78), (0.35, 0.78))
        draw_arrow((0.60, 0.78), (0.68, 0.78))
        draw_arrow((0.80, 0.68), (0.80, 0.54), '生成器选择')
        draw_arrow((0.48, 0.68), (0.22, 0.54), '结构约束')
        draw_arrow((0.36, 0.44), (0.40, 0.44))
        draw_arrow((0.64, 0.44), (0.70, 0.44))
        draw_arrow((0.50, 0.34), (0.50, 0.22), '指标与解释')
        draw_arrow((0.81, 0.34), (0.60, 0.22), '最优策略')

        ax.text(0.02, 0.97, '下午茶补贴策略系统架构流程图', fontsize=16, fontweight='bold', ha='left', va='top')
        ax.text(0.02, 0.935, '数据经过结构提取、仿真校正与优化后形成可执行发券策略闭环', fontsize=10.5, color=self.palette['gray'], ha='left', va='top')
        if save:
            plt.savefig('pic/system_architecture.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_coupon_catalog(self, coupon_catalog, save=True):
        rows = []
        for name, meta in coupon_catalog.items():
            rows.append({
                '券型': name,
                '券面额': meta['coupon_value'],
                '客单提升': meta['uplift'],
                '吸引力': meta['attraction']
            })
        df = pd.DataFrame(rows).sort_values('吸引力', ascending=True)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=self.palette['white'])
        sns.barplot(data=df, y='券型', x='吸引力', ax=axes[0], color=self.palette['gold'], edgecolor=self.palette['orange'], width=0.55)
        axes[0].axvline(1.0, linestyle='--', color=self.palette['black'])
        self._style_axis(axes[0], '候选券吸引力', '吸引力', '券型')
        self._add_subtitle(axes[0], '不同券型吸引力差异明显，统一发券会损失效率')
        self._add_conclusion(axes[0], f"候选券数 = {len(df)}")
        sns.barplot(data=df, y='券型', x='客单提升', ax=axes[1], color=self.palette['orange'], edgecolor=self.palette['gray'], width=0.55)
        self._style_axis(axes[1], '候选券带动客单提升', '客单提升', '券型')
        self._add_subtitle(axes[1], '高吸引力券通常伴随更强客单提升效应')
        self._add_conclusion(axes[1], f"最大客单提升 = {df['客单提升'].max():.1f}")
        plt.tight_layout()
        if save:
            plt.savefig('pic/coupon_catalog.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_simulation_comparison(self, expected_metrics, simulated_metrics, save=True):
        df = pd.DataFrame({
            'metric': ['无券GTV', '有券GTV', '因果增量GTV', '总成本'],
            '解析期望': [
                expected_metrics['gtv_no_coupon'],
                expected_metrics['gtv_with_coupon'],
                expected_metrics['delta_gtv_causal'],
                expected_metrics['total_cost']
            ],
            '随机仿真': [
                simulated_metrics['gtv_no_coupon'],
                simulated_metrics['gtv_with_coupon'],
                simulated_metrics['delta_gtv_causal'],
                simulated_metrics['total_cost']
            ]
        }).melt(id_vars='metric', var_name='type', value_name='value')
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=self.palette['white'])
        sns.barplot(data=df, x='metric', y='value', hue='type', ax=ax, palette=[self.palette['gold'], self.palette['orange']], width=0.55)
        self._style_axis(ax, '随机仿真 vs 解析期望', '', '数值')
        self._add_subtitle(ax, '随机结果围绕解析期望波动，策略中枢相对稳定')
        self._add_conclusion(ax, f"期望因果增量 = {expected_metrics['delta_gtv_causal']:.1f}")
        ax.legend(title='', frameon=False)
        plt.tight_layout()
        if save:
            plt.savefig('pic/simulation_compare.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_strategy_comparison(self, strategies, save=True):
        strategy_df = pd.DataFrame(strategies).T.reset_index().rename(columns={'index': 'strategy'})
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), facecolor=self.palette['white'])
        sns.barplot(data=strategy_df, x='strategy', y='delta_gtv_causal', ax=axes[0], palette=[self.palette['gray'], self.palette['gold'], self.palette['orange'], self.palette['gray'], self.palette['gold']], width=0.5)
        self._style_axis(axes[0], '策略因果增量GTV', '', 'ΔGTV causal')
        self._add_subtitle(axes[0], '优化策略在增量收益上整体优于基线和统一发券')
        best_strategy = strategy_df.sort_values('delta_gtv_causal', ascending=False).iloc[0]['strategy']
        self._add_conclusion(axes[0], f'当前最优策略 = {best_strategy}')
        axes[0].tick_params(axis='x', rotation=25)
        sns.barplot(data=strategy_df, x='strategy', y='roi', ax=axes[1], palette=[self.palette['gray'], self.palette['gold'], self.palette['orange'], self.palette['gray'], self.palette['gold']], width=0.5)
        self._style_axis(axes[1], '策略ROI', '', 'ROI')
        self._add_subtitle(axes[1], '收益规模提升同时保持了预算效率')
        self._add_conclusion(axes[1], f"最高ROI = {strategy_df['roi'].max():.2f}")
        axes[1].tick_params(axis='x', rotation=25)
        plt.tight_layout()
        if save:
            plt.savefig('pic/strategy_compare_new.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_bandit_history(self, history_df, save=True):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), facecolor=self.palette['white'])
        sns.lineplot(data=history_df, x='epoch', y='delta_gtv_causal', marker='o', ax=axes[0], color=self.palette['gold'], linewidth=2.8)
        self._style_axis(axes[0], 'Bandit训练中的因果增量', 'epoch', 'ΔGTV causal')
        self._add_subtitle(axes[0], '训练轮次增加后增量收益逐步收敛')
        self._add_conclusion(axes[0], f"最终增量 = {history_df['delta_gtv_causal'].iloc[-1]:.1f}")
        sns.lineplot(data=history_df, x='epoch', y='roi', marker='o', ax=axes[1], color=self.palette['black'], linewidth=2.8)
        self._style_axis(axes[1], 'Bandit训练中的ROI', 'epoch', 'ROI')
        self._add_subtitle(axes[1], 'ROI 曲线稳定上行，学习过程没有明显效率塌陷')
        self._add_conclusion(axes[1], f"最终ROI = {history_df['roi'].iloc[-1]:.2f}")
        plt.tight_layout()
        if save:
            plt.savefig('pic/bandit_history.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_allocation_mix(self, allocation_df, save=True):
        if allocation_df is None or len(allocation_df) == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, '当前预算下没有发券', ha='center', va='center')
            ax.axis('off')
            return fig
        count_df = allocation_df['coupon'].value_counts().reset_index()
        count_df.columns = ['coupon', 'count']
        fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor=self.palette['white'])
        colors = [self.palette['gold'], self.palette['orange'], self.palette['gray'], self.palette['black']]
        wedges, texts, autotexts = ax.pie(
            count_df['count'],
            labels=count_df['coupon'],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors[:len(count_df)],
            wedgeprops={'linewidth': 1.0, 'edgecolor': self.palette['white']},
            textprops={'fontsize': 9.5, 'color': self.palette['charcoal']}
        )
        for t in autotexts:
            t.set_color(self.palette['black'])
            t.set_fontsize(9)
        ax.set_title('最优发券构成', loc='left', fontsize=13, fontweight='bold', pad=18)
        ax.axis('equal')
        self._add_subtitle(ax, '发券覆盖集中在少数主力券型，结构清晰')
        self._add_conclusion(ax, f"总覆盖 = {int(count_df['count'].sum())} 用户")
        plt.tight_layout()
        if save:
            plt.savefig('pic/allocation_mix.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_sensitivity(self, sensitivity_df, save=True):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), facecolor=self.palette['white'])
        sns.barplot(data=sensitivity_df, x='scenario', y='roi', ax=axes[0], palette=[self.palette['gray'], self.palette['gold'], self.palette['orange'], self.palette['gray']], width=0.52)
        self._style_axis(axes[0], '敏感性分析: ROI', '', 'ROI')
        self._add_subtitle(axes[0], '参数扰动下 ROI 波动可控，策略具备稳健性')
        self._add_conclusion(axes[0], f"ROI波动范围 = {sensitivity_df['roi'].min():.2f}~{sensitivity_df['roi'].max():.2f}")
        axes[0].tick_params(axis='x', rotation=20)
        sns.barplot(data=sensitivity_df, x='scenario', y='delta_gtv_causal', ax=axes[1], palette=[self.palette['gray'], self.palette['gold'], self.palette['orange'], self.palette['gray']], width=0.52)
        self._style_axis(axes[1], '敏感性分析: 因果增量GTV', '', 'ΔGTV causal')
        self._add_subtitle(axes[1], '增量收益在多场景下方向一致，结论稳定')
        self._add_conclusion(axes[1], f"最大增量 = {sensitivity_df['delta_gtv_causal'].max():.1f}")
        axes[1].tick_params(axis='x', rotation=20)
        plt.tight_layout()
        if save:
            plt.savefig('pic/sensitivity.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_calibration(self, calibration, save=True):
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), facecolor=self.palette['white'])
        value_df = pd.DataFrame({
            '数据类型': ['原始数据', '半合成用户'],
            '值': [calibration['raw_avg_order_value'], calibration['synthetic_avg_value']]
        })
        rate_df = pd.DataFrame({
            '数据类型': ['原始数据', '半合成用户'],
            '值': [calibration['raw_coupon_rate'], calibration['synthetic_avg_sensitivity']]
        })
        sns.barplot(data=value_df, x='数据类型', y='值', ax=axes[0], palette=[self.palette['gold'], self.palette['orange']], width=0.5)
        self._style_axis(axes[0], '平均客单价校准', '', '金额')
        self._add_subtitle(axes[0], '半合成客单价与原始数据同量级，价值建模可信')
        self._add_conclusion(axes[0], f"偏差 = {abs(calibration['raw_avg_order_value'] - calibration['synthetic_avg_value']):.2f}")
        sns.barplot(data=rate_df, x='数据类型', y='值', ax=axes[1], palette=[self.palette['gray'], self.palette['gold']], width=0.5)
        self._style_axis(axes[1], '券使用率校准', '', '比例')
        self._add_subtitle(axes[1], '半合成用券率与真实水平接近，敏感度刻画可靠')
        self._add_conclusion(axes[1], f"偏差 = {abs(calibration['raw_coupon_rate'] - calibration['synthetic_avg_sensitivity']):.3f}")
        plt.tight_layout()
        if save:
            plt.savefig('pic/calibration.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_profile_distribution(self, profile_df, save=True):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=self.palette['white'])

        lifecycle = profile_df['用户生命周期阶段'].value_counts().reset_index()
        lifecycle.columns = ['类型', '用户数']
        sns.barplot(data=lifecycle, x='类型', y='用户数', ax=axes[0], color=self.palette['gold'], edgecolor=self.palette['orange'], width=0.55)
        self._style_axis(axes[0], '用户生命周期阶段分布', '类型', '用户数')
        self._add_subtitle(axes[0], '生命周期分布有明显主群体，可支撑分群策略')
        self._add_conclusion(axes[0], f"样本用户数 = {len(profile_df)}")

        member = profile_df['会员状态'].value_counts().reset_index()
        member.columns = ['类型', '用户数']
        sns.barplot(data=member, x='类型', y='用户数', ax=axes[1], color=self.palette['orange'], edgecolor=self.palette['gray'], width=0.55)
        self._style_axis(axes[1], '会员状态分布', '类型', '用户数')
        self._add_subtitle(axes[1], '会员结构差异明显，是策略差异化的重要依据')
        self._add_conclusion(axes[1], f"最高频层级 = {member.iloc[0]['类型']}")

        payment = profile_df['历史实付分层（365天）'].value_counts().reset_index()
        payment.columns = ['类型', '用户数']
        sns.barplot(data=payment, x='类型', y='用户数', ax=axes[2], color=self.palette['gray'], edgecolor=self.palette['orange'], width=0.55)
        self._style_axis(axes[2], '历史实付分层分布', '类型', '用户数')
        self._add_subtitle(axes[2], '历史消费分层清晰，价值先验设置合理')
        self._add_conclusion(axes[2], f"主流分层 = {payment.iloc[0]['类型']}")
        axes[2].tick_params(axis='x', rotation=18)

        plt.tight_layout()
        if save:
            plt.savefig('pic/profile_distribution.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_order_amount_analysis(self, order_df, scene_orders, save=True):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=self.palette['white'])
        sns.histplot(scene_orders['订单金额'], bins=18, kde=True, ax=axes[0], color=self.palette['gold'], edgecolor=self.palette['orange'])
        self._style_axis(axes[0], '下午茶场景订单金额分布', '订单金额', '订单数')
        self._add_subtitle(axes[0], '场景订单金额集中，用户价值区间可稳定识别')
        self._add_conclusion(axes[0], f"场景均值 = {scene_orders['订单金额'].mean():.1f}")

        plot_df = scene_orders.copy()
        plot_df['订单类型'] = np.where(plot_df['美补金额'].fillna(0) > 0, '有券订单', '无券订单')
        sns.boxplot(
            data=plot_df, x='订单类型', y='订单金额', ax=axes[1],
            palette={'无券订单': self.palette['gray'], '有券订单': self.palette['gold']},
            width=0.45, linewidth=1.1
        )
        sns.stripplot(data=plot_df, x='订单类型', y='订单金额', ax=axes[1], color=self.palette['orange'], alpha=0.28, jitter=0.18, size=3)
        self._style_axis(axes[1], '有券 vs 无券订单金额', '', '订单金额')
        self._add_subtitle(axes[1], '有券订单金额中位数更高，存在客单提升效应')
        self._add_conclusion(axes[1], f"有券均值 = {plot_df.loc[plot_df['订单类型']=='有券订单', '订单金额'].mean():.1f}")
        plt.tight_layout()
        if save:
            plt.savefig('pic/order_amount_analysis.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_afternoon_time_distribution(self, scene_orders, save=True):
        plot_df = scene_orders.copy()
        plot_df['minute_of_day'] = plot_df['下单时间'].apply(lambda x: x.hour * 60 + x.minute if hasattr(x, 'hour') else 0)
        plot_df['bucket'] = (plot_df['minute_of_day'] // 30) * 30
        counts = plot_df.groupby('bucket').size()
        all_buckets = np.arange(14 * 60, 17 * 60 + 1, 30)
        counts = counts.reindex(all_buckets, fill_value=0)
        labels = [f"{m // 60:02d}:{m % 60:02d}" for m in counts.index]
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(11, 6), facecolor=self.palette['white'])
        bars = ax.bar(x, counts.values, width=0.8, color=self.palette['gold'], edgecolor='#A67C00', linewidth=1.2, zorder=2)
        ax.plot(x, counts.values, color=self.palette['black'], linewidth=2.5, marker='o', markersize=7, zorder=3)
        self._style_axis(ax, '下午茶时段订单时间分布', '时间点', '订单量')
        self._add_subtitle(ax, '时段峰谷清晰，时间修正因子设定有数据支撑')
        peak_idx = int(np.argmax(counts.values)) if len(counts.values) else 0
        self._add_conclusion(ax, f"峰值时间 = {labels[peak_idx]}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        for rect, value in zip(bars, counts.values):
            ax.text(rect.get_x() + rect.get_width() / 2, value + max(counts.max() * 0.04, 0.2), f'{int(value)}', ha='center', fontsize=10)
        plt.tight_layout()
        if save:
            plt.savefig('pic/afternoon_time_distribution.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_beta_value_relation(self, beta_df, save=True):
        fig, ax = plt.subplots(figsize=(10.5, 6.5), facecolor=self.palette['white'])
        sns.regplot(data=beta_df, x='V', y='beta', scatter=False, color=self.palette['black'], ax=ax, line_kws={'linewidth': 2.2})
        sns.scatterplot(
            data=beta_df,
            x='V',
            y='beta',
            hue='会员状态',
            style='用户生命周期阶段',
            palette='crest',
            alpha=0.7,
            s=55,
            ax=ax
        )
        self._style_axis(ax, '自然转化率β与用户价值关系', '用户价值 V', 'β')
        self._add_subtitle(ax, '自然转化率随用户价值上升，符合业务直觉')
        self._add_conclusion(ax, f"β均值 = {beta_df['beta'].mean():.3f}")
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        if save:
            plt.savefig('pic/beta_value_relation.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_delta_curve(self, delta_df, save=True):
        fig, ax = plt.subplots(figsize=(8.8, 4.8), facecolor=self.palette['white'])
        sns.lineplot(
            data=delta_df,
            x='coupon_value',
            y='delta',
            hue='敏感度分组',
            marker='o',
            linewidth=2.4,
            palette=[self.palette['gray'], self.palette['orange'], self.palette['gold']],
            ax=ax
        )
        self._style_axis(ax, '券增量概率 δ 随券面额变化', '券面额（元）', 'δ')
        self._add_subtitle(ax, '敏感度越高，券面额提升带来的增量越明显')
        self._add_conclusion(ax, f"最高 δ = {delta_df['delta'].max():.3f}")
        ax.legend(title='', frameon=False)
        plt.tight_layout()
        if save:
            plt.savefig('pic/delta_curve.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_stage_difference(self, stage_df, save=True):
        # 将segment转换为中文
        stage_df['用户分层'] = stage_df['segment'].map({
            'high_value': '高价值用户',
            'mid_value': '中价值用户',
            'low_value': '低价值用户'
        })
        fig, ax = plt.subplots(figsize=(9.0, 4.9), facecolor=self.palette['white'])
        sns.barplot(
            data=stage_df,
            x='stage',
            y='probability',
            hue='用户分层',
            palette=[self.palette['gray'], self.palette['orange'], self.palette['gold']],
            ax=ax
        )
        self._style_axis(ax, '不同用户分层的Funnel阶段差异', '阶段', '平均概率')
        self._add_subtitle(ax, '高价值分层在关键阶段通过率更高')
        top_segment = stage_df[stage_df['stage']=='p3'].sort_values('probability', ascending=False).iloc[0]['用户分层']
        self._add_conclusion(ax, f"purchase最高分层 = {top_segment}")
        ax.legend(title='', frameon=False)
        plt.tight_layout()
        if save:
            plt.savefig('pic/stage_difference.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_segment_lift(self, lift_df, save=True):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), facecolor=self.palette['white'])
        plot_df = lift_df.melt(
            id_vars='敏感度分组',
            value_vars=['prob_no_coupon', 'prob_with_coupon'],
            var_name='类型',
            value_name='购买概率'
        )
        plot_df['类型'] = plot_df['类型'].map({'prob_no_coupon': '无券', 'prob_with_coupon': '有券'})
        sns.barplot(
            data=plot_df,
            x='敏感度分组',
            y='购买概率',
            hue='类型',
            ax=axes[0],
            palette={'无券': self.palette['gold'], '有券': self.palette['orange']},
            edgecolor=self.palette['gray'],
            width=0.5
        )
        self._style_axis(axes[0], '无券 vs 有券转化率对比', '敏感度分组', '购买概率')
        self._add_subtitle(axes[0], '高敏感组在发券后转化提升最明显')
        self._add_conclusion(axes[0], f"最大 uplift 组 = {lift_df.sort_values('lift', ascending=False).iloc[0]['敏感度分组']}")
        axes[0].legend(title='', frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')

        sns.barplot(data=lift_df, x='敏感度分组', y='lift', ax=axes[1], palette=[self.palette['gold'], self.palette['orange'], self.palette['gold']], width=0.5)
        self._style_axis(axes[1], '分组转化增量', '敏感度分组', 'prob uplift')
        self._add_subtitle(axes[1], '分群增量差异显著，差异化发券是必要策略')
        self._add_conclusion(axes[1], f"平均增量 = {lift_df['lift'].mean():.3f}")
        plt.tight_layout()
        if save:
            plt.savefig('pic/segment_lift.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_budget_frontier(self, frontier_df, save=True):
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), facecolor=self.palette['white'])
        sns.lineplot(data=frontier_df, x='budget', y='delta_gtv_causal', marker='o', linewidth=2.6, color=self.palette['gold'], ax=axes[0])
        self._style_axis(axes[0], '预算敏感性: 因果增量GTV', '预算', 'ΔGTV causal')
        self._add_subtitle(axes[0], '预算增加能持续拉动增量收益')
        self._add_conclusion(axes[0], f"最高增量 = {frontier_df['delta_gtv_causal'].max():.1f}")

        sns.lineplot(data=frontier_df, x='budget', y='roi', marker='o', linewidth=2.6, color=self.palette['black'], ax=axes[1])
        self._style_axis(axes[1], '预算敏感性: ROI', '预算', 'ROI')
        self._add_subtitle(axes[1], '预算到高位后 ROI 增速放缓，存在边际递减')
        self._add_conclusion(axes[1], f"最佳 ROI = {frontier_df['roi'].max():.2f}")
        plt.tight_layout()
        if save:
            plt.savefig('pic/budget_frontier.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_coverage_by_segment(self, coverage_df, save=True):
        # 将segment转换为中文
        coverage_df['用户分层'] = coverage_df['segment'].map({
            'high_value': '高价值用户',
            'mid_value': '中价值用户',
            'low_value': '低价值用户'
        })
        fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor=self.palette['white'])
        sns.barplot(data=coverage_df, x='用户分层', y='count', hue='是否发券', palette=[self.palette['gray'], self.palette['gold']], ax=ax, width=0.52)
        self._style_axis(ax, '最优策略的用户覆盖结构', '用户分层', '用户数')
        self._add_subtitle(ax, '发券覆盖更集中于高价值和高响应人群')
        sent_ratio = coverage_df[coverage_df['是否发券'] == '发券']['count'].sum() / max(coverage_df['count'].sum(), 1)
        self._add_conclusion(ax, f"总体发券占比 = {sent_ratio:.0%}")
        ax.legend(title='', frameon=False)
        plt.tight_layout()
        if save:
            plt.savefig('pic/coverage_by_segment.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_monte_carlo_distribution(self, mc_df, save=True):
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), facecolor=self.palette['white'])
        sns.violinplot(data=mc_df, y='delta_gtv_causal', ax=axes[0], color=self.palette['gold'], inner='box')
        self._style_axis(axes[0], '蒙特卡洛: 因果增量GTV分布', '', 'ΔGTV causal')
        self._add_subtitle(axes[0], '增量收益分布稳定，极端亏损风险较低')
        self._add_conclusion(axes[0], f"均值 = {mc_df['delta_gtv_causal'].mean():.1f}")

        sns.violinplot(data=mc_df, y='roi', ax=axes[1], color=self.palette['gray'], inner='box')
        self._style_axis(axes[1], '蒙特卡洛: ROI分布', '', 'ROI')
        self._add_subtitle(axes[1], 'ROI 存在可控波动，均值表现稳定')
        self._add_conclusion(axes[1], f"均值 = {mc_df['roi'].mean():.2f}")
        plt.tight_layout()
        if save:
            plt.savefig('pic/monte_carlo_distribution.png', dpi=300, bbox_inches='tight')
        return fig

    def plot_sensitivity_heatmap(self, heatmap_df, value_col='roi', save=True):
        pivot = heatmap_df.pivot(index='beta_scale', columns='delta_scale', values=value_col)
        fig, ax = plt.subplots(figsize=(8.5, 6.5), facecolor=self.palette['white'])
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrBr', linewidths=0.6, cbar=True, ax=ax)
        self._style_axis(ax, f'参数敏感性热力图: {value_col}', 'δ 缩放系数', 'β 缩放系数')
        self._add_subtitle(ax, '参数组合对结果有影响，但最优区域具备可解释结构')
        self._add_conclusion(ax, f"峰值 = {heatmap_df[value_col].max():.2f}")
        plt.tight_layout()
        if save:
            plt.savefig(f'pic/sensitivity_heatmap_{value_col}.png', dpi=300, bbox_inches='tight')
        return fig
