# Engineering design: Meituan T15 coupon strategy

## 1. Problem and requirements

This competition project asks how afternoon-tea coupons can be modelled, compared, and allocated across heterogeneous users while respecting a campaign budget. Contextual learning is the allocation method inside the broader Meituan T15 workflow, which also covers scene extraction, user-state construction, funnel calibration, evaluation, and presentation.

### Functional requirements

- adapt the three competition-style spreadsheet schemas without publishing private inputs;
- build a reproducible semi-synthetic user population;
- expose a finite action set containing no coupon and candidate coupons;
- select one feasible action per update and reveal only its stochastic outcome;
- update a contextual policy from selected-action feedback;
- record reward, expected regret, action mix, coupon cost, and remaining budget;
- reproduce results from fixed configurations and seeds.

### Non-functional requirements

- deterministic fixture generation and seeded experiments;
- explicit separation of simulator results from observed business outcomes;
- domain code independent of Streamlit;
- input validation, budget safety, and automated tests;
- keep credentials, private source tables, runtime exports, and local caches out of Git; selected report figures and documentation are committed.

## 2. System architecture

```text
Synthetic fixture / authorized local spreadsheets
                       │
                       ▼
              FeatureEngineering
          schema adapter + distribution fit
                       │
             users + environment parameters
                       │
                       ▼
               SimulationEngine
      funnel probabilities + coupon response
                       │
                       ▼
              CouponBanditEnvironment
   context/action features + partial stochastic feedback
                       │
                       ▼
                  SharedLinUCB
      action selection + online parameter update
                 /                 \
                ▼                   ▼
          Streamlit UI        benchmark CLI
```

`feature_engineering.py` is the input adapter. `simulation_engine.py` owns outcome assumptions. `contextual_bandit.py` owns the environment-policy boundary and learning loop. `project_pipeline.py` composes these modules and is the project-level path used by the dashboard, CLI, benchmark, and integration tests. `app.py` presents the competition problem, data calibration, model structure, policy results, decision explanations, and project evolution, but contains no policy logic.

The simulator exposes expected and stochastic action outcomes to the bandit. The project pipeline manages both modules and their shared configuration.

## 3. Interfaces and data design

The generated user table contains one unique `user_id` and the state required by the simulator: expected basket value (`V`), activity (`A`), posterior sensitivity parameters, and exposure/success counts. Duplicate identifiers and empty populations are rejected.

The action catalog contains a no-coupon action plus coupon metadata: face value, free/paid indicator, expanded-value indicator, basket uplift assumption, and attraction factor. Source-language coupon names remain an adapter detail; the UI translates them into English labels.

For shared LinUCB, the feature vector combines:

- an intercept and standardized user value, activity, and sensitivity;
- quadratic and pairwise user terms for nonlinear thresholds;
- coupon properties;
- user-property interactions that permit personalized action response.

Each `BanditStep` records the chosen action's realized reward, cost, and purchase rate, plus expected values used for evaluation. The optimal-action oracle does not enter policy updates. The simulated no-coupon expectation separately enters the centered reward, as specified below.

The programmatic entry point returns the input population, parameters, decision history, and references together:

```python
from project_pipeline import ProjectConfig, run_project

run = run_project(ProjectConfig(users=40, steps=80, budget=500, seed=42))
print(run.summary())
```

In memory, `action` is `None` or a source-catalog key. Use `format_action()` for an English display label. The history stores one row per decision, including `expected_reward`, `expected_regret`, `purchase_rate`, `cost`, and `remaining_budget`.

## 4. Implementation decisions

### Partial feedback

At each update, a user is sampled, the policy scores feasible actions, and the environment returns stochastic feedback only for the selected action. The shared model transfers information across coupon actions; the disjoint implementation is retained as a diagnostic baseline.

### Incremental reward

The raw selected-action reward is simulated coupon GTV minus `0.25 × coupon cost`. The simulator supplies expected no-coupon GTV, which is subtracted to center the learning target. At a fixed user context this baseline is action-independent, so it does not change the true reward-maximizing action. A live campaign would need an estimated baseline.

### Numerical update

LinUCB uses ridge regularization and a Sherman–Morrison rank-one inverse update. This avoids recomputing a matrix inverse after every observation. Monetary rewards are scaled before learning so that confidence bounds remain numerically meaningful; unscaled values are retained in the history.

### Personalization scenario

The source tables do not identify individual treatment effects. The simulator therefore declares a scenario assumption: response is strongest when coupon value as a fraction of basket value is near a sensitivity-dependent target. Rare-coupon uplift is shrunk toward an amount-based prior using reliability weight `n / (n + 30)`. Both choices are modeling assumptions, not recovered causal relationships.

### Budget handling

An action is feasible only when its face value does not exceed remaining budget. With a feedback batch of 16, the purchase rate is the mean of 16 Bernoulli draws, and the debit is `face value × purchase rate` for that decision. Reported expenditure sums these batch-mean costs, not the raw cost of all 16 exposures. This prevents overspending under the stated accounting convention.

## 5. Testing and validation

The test suite checks:

- deterministic fixture generation and successful schema adaptation;
- finite simulator outputs and multiple context-dependent optimal actions;
- selected-action-only policy updates;
- the incremental inverse against a direct matrix inverse;
- seeded stochastic reproducibility;
- non-negative expected regret and budget safety;
- invalid configuration and malformed population rejection;
- the complete public pipeline from generated spreadsheets to decisions and summary metrics.

The benchmark compares random action, best fixed action, contextual oracle, and learned shared LinUCB under one environment and protocol. It holds the campaign population fixed across policy seeds so that variation reflects the learning process rather than regenerated users.

## 6. Deployment and operations

Install with `requirements.txt`; GitHub Actions runs the tests, and `streamlit run app.py` starts the application. Each campaign executes in one process and returns an in-memory decision history. Decision explanations use the displayed inputs and metrics.

Raw data, exports, logs, virtual environments, and editor artifacts are excluded by `.gitignore`. The public demo generates its own temporary input files and deletes them after feature construction.

## 7. Evaluation boundary and next steps

- The public benchmark measures the declared semi-synthetic campaign environment; the optimal-action oracle is used only as an evaluation reference. The environment separately supplies the expected no-coupon baseline used to center training rewards.
- The source tables lack randomized assignments and logging propensities. Live evaluation therefore requires controlled exploration and propensity-aware analysis.
- Delayed outcomes, campaign eligibility, budget pacing, safety constraints, and model drift should be added with a production event log.
- Feature and schema versions should accompany stored models and cross-version result comparison.
