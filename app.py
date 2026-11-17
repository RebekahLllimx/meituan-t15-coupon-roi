"""Interactive presentation of the maintained Meituan T15 competition project."""

import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from contextual_bandit import CouponBanditEnvironment, run_contextual_bandit
from feature_engineering import FeatureEngineering
from generate_synthetic_demo import generate


st.set_page_config(page_title="Meituan T15 Coupon Strategy", page_icon="🎟️", layout="wide")


@st.cache_data(show_spinner=False)
def build_demo_population(seed: int, n_users: int):
    """Build an invented population through the same public input adapter."""
    with tempfile.TemporaryDirectory() as directory:
        generate(directory, seed=seed)
        features = FeatureEngineering()
        if not features.load_data(directory):
            raise RuntimeError("The synthetic fixture could not be loaded.")
        return features.build_features_and_params(
            n_users=n_users, random_state=seed, use_auto_distribution_selector=False
        )


def policy_baselines(users: pd.DataFrame, parameters: dict) -> dict:
    """Compute simulator-only fixed-action and contextual-oracle references."""
    environment = CouponBanditEnvironment(users, parameters, seed=0)
    rewards = np.array([
        [environment.expected_reward(index, action) for action in environment.actions]
        for index in range(len(users))
    ])
    return {
        "random": float(rewards.mean()),
        "best_fixed": float(rewards.mean(axis=0).max()),
        "oracle": float(rewards.max(axis=1).mean()),
    }


def display_action(action, environment: CouponBanditEnvironment) -> str:
    """Translate raw source coupon labels into an English display label."""
    if action is None or pd.isna(action):
        return "No coupon"
    metadata = environment.action_metadata(action)
    properties = []
    if metadata["free"]:
        properties.append("free")
    if metadata["expand"]:
        properties.append("expanded-value")
    suffix = f" ({', '.join(properties)})" if properties else ""
    return f"CNY {metadata['coupon_value']:g} coupon{suffix}"


def explain_decision(row: pd.Series, user: pd.Series, environment: CouponBanditEnvironment) -> str:
    """Create an auditable business explanation from the values shown on screen."""
    action_label = display_action(row["action"], environment)
    sensitivity = float(user["S_alpha_post"] / (user["S_alpha_post"] + user["S_beta_post"]))
    if row["action"] is None or pd.isna(row["action"]):
        decision = "The policy kept the no-coupon action for this update"
    else:
        decision = f"The policy selected a {action_label} for this update"
    return (
        f"{decision}. The synthetic user has an expected basket value of CNY {user['V']:.1f}, "
        f"activity {user['A']:.2f}, and estimated price sensitivity {sensitivity:.2f}. "
        f"The selected action's modelled incremental net reward is {row['expected_reward']:.3f}; "
        f"its gap to the simulator oracle is {row['expected_regret']:.3f}. The choice may include "
        "exploration, so this explanation reports the policy inputs rather than inventing a causal reason."
    )


st.title("Meituan T15 Afternoon-Tea Coupon Strategy")
st.caption("A maintained student competition project for coupon simulation, personalization, and budget-aware allocation")
st.warning(
    "Research prototype only. All outcomes on this page are simulated under documented "
    "assumptions; they are not production Meituan results or causal-effect estimates."
)

with st.sidebar:
    st.header("Experiment settings")
    n_users = st.slider("Synthetic users", 100, 1_000, 300, 100)
    steps = st.slider("Learning updates", 1_000, 20_000, 10_000, 1_000)
    feedback_batch = st.select_slider("Exposures per update", [1, 4, 8, 16, 32], value=16)
    alpha = st.slider("Exploration coefficient", 0.0, 1.0, 0.25, 0.05)
    budget = st.number_input("Simulated budget (CNY)", 1_000.0, 200_000.0, 60_000.0, 1_000.0)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42)
    run_requested = st.button("Run experiment", type="primary", width="stretch")

if run_requested or "experiment" not in st.session_state:
    with st.spinner("Generating the population and training the policy..."):
        users, parameters = build_demo_population(int(seed), int(n_users))
        history = run_contextual_bandit(
            users,
            parameters,
            steps=int(steps),
            budget=float(budget),
            alpha=float(alpha),
            seed=int(seed),
            policy_type="shared",
            feedback_batch_size=int(feedback_batch),
        )
        st.session_state.experiment = {
            "users": users,
            "parameters": parameters,
            "history": history,
            "baselines": policy_baselines(users, parameters),
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
    ["Project", "Data & model", "Policy learning", "Decisions", "Evidence & evolution"]
)

with overview_tab:
    st.subheader("Competition problem")
    st.write(
        "This project was developed for the Meituan T15 business-data-science competition. "
        "It studies how afternoon-tea coupons can be allocated across heterogeneous users "
        "under a limited subsidy budget when the supplied tables do not form a complete "
        "context–treatment–outcome history."
    )
    st.subheader("Current solution")
    st.code(
        "Synthetic spreadsheets → feature adapter → funnel simulator\n"
        "→ context-action features → LinUCB selection → partial feedback → update",
        language="text",
    )
    st.write(
        "The original competition prototype established scene filtering, user-state modelling, "
        "a purchase funnel, and constrained coupon comparison. The maintained version keeps "
        "that business problem and adds a valid sequential learning loop: the policy observes "
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
            "Action": display_action(action, environment_for_catalog),
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
    st.caption(
        f"Lines are rolling means over {window:,} updates. Oracle values are used only for "
        "evaluation and are never passed to the policy update."
    )

with actions_tab:
    st.subheader("Selected actions and budget use")
    environment = CouponBanditEnvironment(users, parameters, seed=settings["seed"])
    history["Action"] = history["action"].apply(lambda value: display_action(value, environment))
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
    st.caption(
        "The historical prototype used an optional LLM to paraphrase model output. The maintained "
        "page generates explanations directly from displayed values so they are reproducible and "
        "cannot add unsupported business claims."
    )
    recent = history.tail(3)
    indexed_users = users.set_index("user_id")
    for _, row in recent.iterrows():
        with st.expander(f"Update {int(row['step'])}: {row['Action']}"):
            st.write(explain_decision(row, indexed_users.loc[row["user_id"]], environment))

with assumptions_tab:
    st.subheader("How the project evolved")
    st.markdown(
        """
1. **Competition prototype:** filtered the afternoon-tea scene, fitted user-state distributions, constructed a purchase funnel, and compared coupon strategies under a subsidy budget.
2. **Reproducibility pass:** removed private inputs from Git, added deterministic synthetic files, clarified the simulation boundary, and introduced automated tests.
3. **Learning-method correction:** separated the old full-information allocator from a real partial-feedback policy, then implemented shared LinUCB with context-action interactions.
4. **Public presentation pass:** consolidated duplicate reports into one English technical report and rebuilt the dashboard around the maintained method and inspectable evidence.
        """
    )
    st.subheader("Evidence boundary")
    st.markdown(
        """
- The public demo uses deterministic, invented spreadsheets with the same schema as the original competition inputs.
- Coupon outcomes come from a documented funnel simulator. The source tables do not contain randomized treatment assignments or logging propensities.
- Personalization is an explicit scenario assumption: response varies with user value, activity, sensitivity, and coupon-to-basket fit.
- The reward is modelled incremental net value: coupon GTV minus a coupon-cost penalty and an action-independent no-coupon baseline.
- Reported improvement is valid only inside this semi-synthetic environment. A live deployment would require randomized exploration, propensity logging, guardrails, and offline policy evaluation.
        """
    )
    st.info("See docs/ENGINEERING_DESIGN.md and docs/FINAL_REPORT.md for the complete method and benchmark protocol.")
