"""Interactive demonstration of the maintained coupon-allocation experiment."""

import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from contextual_bandit import CouponBanditEnvironment, run_contextual_bandit
from feature_engineering import FeatureEngineering
from generate_synthetic_demo import generate


st.set_page_config(page_title="Contextual Coupon Allocation", page_icon="🎟️", layout="wide")


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


st.title("Contextual Coupon Allocation")
st.caption("A reproducible, semi-synthetic LinUCB experiment for budget-aware coupon decisions")
st.warning(
    "Research prototype only. All outcomes on this page are simulated under documented "
    "assumptions; they are not production Meituan results or causal-effect estimates."
)

with st.sidebar:
    st.header("Experiment settings")
    n_users = st.slider("Synthetic users", 100, 1_000, 300, 100)
    steps = st.slider("Learning updates", 1_000, 20_000, 3_000, 1_000)
    feedback_batch = st.select_slider("Exposures per update", [1, 4, 8, 16, 32], value=16)
    alpha = st.slider("Exploration coefficient", 0.0, 1.0, 0.25, 0.05)
    budget = st.number_input("Simulated budget (CNY)", 1_000.0, 200_000.0, 18_000.0, 1_000.0)
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

overview_tab, learning_tab, actions_tab, assumptions_tab = st.tabs(
    ["Overview", "Learning curve", "Actions", "Assumptions"]
)

with overview_tab:
    st.subheader("What the experiment does")
    st.write(
        "The policy observes a user context, selects one feasible coupon action, and updates "
        "a shared LinUCB model using only the selected action's simulated feedback. User and "
        "coupon features share statistical strength, while a remaining-budget check prevents "
        "infeasible choices."
    )
    st.code(
        "Synthetic spreadsheets → feature adapter → funnel simulator\n"
        "→ context-action features → LinUCB selection → partial feedback → update",
        language="text",
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

with assumptions_tab:
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
