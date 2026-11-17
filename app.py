"""Interactive presentation of the Meituan T15 competition project."""

import pandas as pd
import streamlit as st

from contextual_bandit import CouponBanditEnvironment
from project_pipeline import ProjectConfig, format_action, run_project


st.set_page_config(page_title="Meituan T15 Coupon Strategy", layout="wide")


@st.cache_data(show_spinner=False)
def execute_project(config: ProjectConfig):
    """Run the same project pipeline used by the command-line tools."""
    return run_project(config)


def explain_decision(row: pd.Series, user: pd.Series, environment: CouponBanditEnvironment) -> str:
    """Create an auditable business explanation from the values shown on screen."""
    action_label = format_action(row["action"], environment)
    sensitivity = float(user["S_alpha_post"] / (user["S_alpha_post"] + user["S_beta_post"]))
    if row["action"] is None or pd.isna(row["action"]):
        decision = "The policy kept the no-coupon action for this update"
    else:
        decision = f"The policy selected a {action_label} for this update"
    return (
        f"{decision}. The synthetic user has an expected basket value of CNY {user['V']:.1f}, "
        f"activity {user['A']:.2f}, and estimated price sensitivity {sensitivity:.2f}. "
        f"The selected action's modelled incremental net reward is {row['expected_reward']:.3f}; "
        f"its gap to the simulator oracle is {row['expected_regret']:.3f}."
    )


st.title("Meituan T15 Afternoon-Tea Coupon Strategy")

with st.sidebar:
    st.header("Experiment settings")
    n_users = st.slider("Synthetic users", 100, 1_000, 300, 100)
    steps = st.slider("Learning updates", 1_000, 20_000, 1_000, 1_000)
    feedback_batch = st.select_slider("Exposures per update", [1, 4, 8, 16, 32], value=16)
    alpha = st.slider("Exploration coefficient", 0.0, 1.0, 0.25, 0.05)
    budget = st.number_input("Simulated budget (CNY)", 1_000.0, 200_000.0, 60_000.0, 1_000.0)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42)
    run_requested = st.button("Run experiment", type="primary", width="stretch")

if run_requested or "experiment" not in st.session_state:
    with st.spinner("Generating the population and training the policy..."):
        run = execute_project(
            ProjectConfig(
                users=int(n_users),
                steps=int(steps),
                budget=float(budget),
                alpha=float(alpha),
                feedback_batch_size=int(feedback_batch),
                seed=int(seed),
            )
        )
        st.session_state.experiment = {
            "users": run.users,
            "parameters": run.parameters,
            "history": run.history,
            "baselines": run.baselines,
            "settings": {
                "users": int(n_users),
                "steps": int(steps),
                "feedback_batch": int(feedback_batch),
                "alpha": float(alpha),
                "budget": float(budget),
                "seed": int(seed),
            },
        }

experiment = st.session_state.experiment
users = experiment["users"]
parameters = experiment["parameters"]
history = experiment["history"].copy()
baselines = experiment["baselines"]
settings = experiment["settings"]
window = min(1_000, max(100, settings["steps"] // 5))
final_expected = float(history["expected_reward"].tail(window).mean())
final_oracle = float(history["oracle_expected_reward"].tail(window).mean())
first_regret = float(history["expected_regret"].head(window).mean())
final_regret = float(history["expected_regret"].tail(window).mean())
oracle_capture = 100 * final_expected / final_oracle if final_oracle else 0.0
regret_reduction = 100 * (1 - final_regret / first_regret) if first_regret else 0.0

summary_columns = st.columns(5)
summary_columns[0].metric("Final expected reward", f"{final_expected:.3f}")
summary_columns[1].metric("Best fixed action", f"{baselines['best_fixed']:.3f}")
summary_columns[2].metric("Contextual oracle", f"{baselines['oracle']:.3f}")
summary_columns[3].metric("Oracle captured", f"{oracle_capture:.1f}%")
summary_columns[4].metric("Regret reduction", f"{regret_reduction:.1f}%")

overview_tab, data_tab, learning_tab, actions_tab, assumptions_tab = st.tabs(
    ["Project", "Data & model", "Learning", "Decisions", "Method"]
)

with overview_tab:
    st.subheader("Problem and solution")
    st.write(
        "Developed for the Meituan T15 business-data-science competition, this project "
        "studies how afternoon-tea coupons can be allocated across heterogeneous users "
        "under a limited subsidy budget when the supplied tables do not form a complete "
        "context–treatment–outcome history."
    )
    st.code(
        "Synthetic spreadsheets → feature adapter → funnel simulator\n"
        "→ context-action features → LinUCB selection → partial feedback → update",
        language="text",
    )
    st.write(
        "The competition entry established scene filtering, user-state modelling, "
        "a purchase funnel, and constrained coupon comparison. The sequential layer "
        "a sequential learning loop: the policy observes "
        "only the outcome of its selected action, shares evidence across coupon types, and "
        "checks budget feasibility before every decision."
    )
    settings_frame = pd.DataFrame([settings]).rename(columns={
        "users": "Users",
        "steps": "Updates",
        "feedback_batch": "Exposures/update",
        "alpha": "Exploration coefficient",
        "budget": "Budget",
        "seed": "Seed",
    })
    st.dataframe(settings_frame, hide_index=True, width="stretch")

with data_tab:
    st.subheader("Input structure and calibrated scenario")
    calibration = parameters["calibration"]
    data_columns = st.columns(4)
    data_columns[0].metric("Filtered orders", f"{calibration['raw_scene_orders']:,}")
    data_columns[1].metric("Source users", f"{calibration['raw_scene_users']:,}")
    data_columns[2].metric("Average order value", f"CNY {calibration['raw_avg_order_value']:.2f}")
    data_columns[3].metric("Coupon-order share", f"{100 * calibration['raw_coupon_rate']:.1f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("**Synthetic user state**")
        user_plot = users[["V", "A", "S"]].rename(
            columns={"V": "Basket value", "A": "Activity", "S": "Price sensitivity"}
        )
        st.scatter_chart(
            user_plot,
            x="Basket value",
            y="Activity",
            color="Price sensitivity",
            size=10,
        )
    with right:
        st.markdown("**Purchase-funnel structure**")
        funnel = parameters["funnel"]
        funnel_frame = pd.DataFrame({
            "Stage": ["View → click", "Click → cart", "Cart → purchase"],
            "Probability": [funnel["p1"], funnel["p2"], funnel["p3"]],
        })
        st.bar_chart(funnel_frame, x="Stage", y="Probability", horizontal=True)

    st.markdown("**Candidate coupon catalog**")
    environment_for_catalog = CouponBanditEnvironment(users, parameters, seed=settings["seed"])
    catalog_rows = []
    for action in environment_for_catalog.actions:
        metadata = environment_for_catalog.action_metadata(action)
        catalog_rows.append({
            "Action": format_action(action, environment_for_catalog),
            "Face value": metadata["coupon_value"],
            "Free coupon": bool(metadata["free"]),
            "Expanded value": bool(metadata["expand"]),
            "Modelled basket uplift": metadata["uplift"],
        })
    st.dataframe(pd.DataFrame(catalog_rows), hide_index=True, width="stretch")

with learning_tab:
    st.subheader("Expected reward and regret")
    chart = history[["step", "expected_reward", "oracle_expected_reward", "expected_regret"]].copy()
    chart["LinUCB expected reward"] = chart["expected_reward"].rolling(window, min_periods=1).mean()
    chart["Contextual oracle"] = chart["oracle_expected_reward"].rolling(window, min_periods=1).mean()
    chart["Expected regret"] = chart["expected_regret"].rolling(window, min_periods=1).mean()
    st.line_chart(
        chart.set_index("step")[["LinUCB expected reward", "Contextual oracle"]],
        y_label="Incremental net reward",
    )
    st.line_chart(chart.set_index("step")[["Expected regret"]], y_label="Expected regret")
    st.write(f"Rolling window: {window:,} updates. The oracle is used only for evaluation.")

with actions_tab:
    st.subheader("Selected actions and budget use")
    environment = CouponBanditEnvironment(users, parameters, seed=settings["seed"])
    history["Action"] = history["action"].apply(lambda value: format_action(value, environment))
    st.bar_chart(history["Action"].value_counts().rename("Selections").to_frame())
    left, right = st.columns(2)
    left.metric("Realized simulated coupon cost", f"CNY {history['cost'].sum():,.2f}")
    right.metric("Remaining simulated budget", f"CNY {history['remaining_budget'].iloc[-1]:,.2f}")
    st.dataframe(
        history[["step", "Action", "purchase_rate", "expected_reward", "expected_regret", "remaining_budget"]]
        .tail(20)
        .rename(columns={
            "step": "Update",
            "purchase_rate": "Purchase rate",
            "expected_reward": "Expected reward",
            "expected_regret": "Expected regret",
            "remaining_budget": "Remaining budget",
        }),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Decision mix by user segment")
    decision_users = history.merge(users[["user_id", "segment"]], on="user_id", how="left")
    segment_mix = pd.crosstab(decision_users["segment"], decision_users["Action"])
    st.bar_chart(segment_mix)

    st.subheader("Business-readable decision explanations")
    recent = history.tail(3)
    indexed_users = users.set_index("user_id")
    for _, row in recent.iterrows():
        with st.expander(f"Update {int(row['step'])}: {row['Action']}"):
            st.write(explain_decision(row, indexed_users.loc[row["user_id"]], environment))

with assumptions_tab:
    st.subheader("Development")
    st.markdown(
        """
1. **Competition entry:** filtered the afternoon-tea scene, modelled user state and the purchase funnel, and compared coupon strategies under a subsidy budget.
2. **Reproducible workflow:** added deterministic public fixtures, one input-to-decision pipeline, and automated tests.
3. **Sequential allocation:** implemented shared LinUCB with context-action interactions and selected-action feedback.
4. **Integrated delivery:** connected the application, benchmark, figures, documentation, tests, and CI through the same execution path.
        """
    )
    st.subheader("Evaluation")
    st.markdown(
        """
- Public fixtures reproduce the competition-table schema; coupon outcomes are generated by the documented funnel simulator.
- Personalization varies with user value, activity, sensitivity, and coupon-to-basket fit.
- Reward is incremental net value: coupon GTV minus a coupon-cost penalty and a no-coupon baseline.
- Production evaluation would add randomized exploration, propensity logging, delayed-outcome attribution, guardrails, and offline policy evaluation.
        """
    )
