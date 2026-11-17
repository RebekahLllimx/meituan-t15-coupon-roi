# Engineering Design: Afternoon Tea Coupon Allocation Simulator

## 1. Problem and scope

The project explores a constrained allocation question: given a fixed coupon
budget and incomplete historical data, how can candidate policies be compared
without presenting simulated outcomes as production evidence?

The intended users are analysts and competition reviewers. The application is
an interactive research prototype, not a production promotion service. It does
not identify causal treatment effects, learn from live traffic, or issue real
coupons.

### Functional requirements

- load three authorized input tables or deterministic synthetic fixtures;
- construct a semi-synthetic population while preserving documented marginal
  distributions and segment constraints;
- simulate a view-to-purchase funnel with and without a coupon;
- allocate at most one coupon per user under budget and coverage constraints;
- compare the allocation with no-coupon and same-budget uniform baselines;
- export inspectable metrics and sensitivity analyses.

### Non-functional requirements

- deterministic runs when the same seed and configuration are used;
- explicit separation between simulated metrics and observed business results;
- no raw or derived private data in Git;
- framework-neutral optimization code that can run without Streamlit;
- validation of allocation constraints and malformed inputs.

## 2. Architecture

```text
authorized spreadsheets / synthetic fixture
                    |
                    v
          FeatureEngineering
                    |
       users + model configuration
                    |
          SimulationEngine
      user-coupon expected outcomes
                    |
              Optimization
   UCB-style score + constrained greedy allocation
                    |
      result object + fair policy baselines
          /                         \
     Streamlit UI             batch report pipeline
```

`feature_engineering.py` owns input parsing, distribution fitting, and
semi-synthetic population construction. `simulation_engine.py` owns funnel and
accounting assumptions. `optimization.py` owns scoring and allocation and emits
framework-neutral progress events. `app.py` renders the interactive workflow;
it is not part of the optimization domain layer.

## 3. Interfaces and data contracts

The optimizer requires one row per `user_id` and the state fields `V`, `A`,
`S_alpha_post`, `S_beta_post`, `exposure`, `successes`, and `failures`. It rejects
duplicate identifiers, missing columns, invalid coverage shares, negative UCB
weights, negative budgets, and empty coupon catalogs.

The result object contains the final allocation, selected user-coupon rows,
strategy metrics, coupon usage, allocation history, and explicit constraint
values. Each user can receive at most one coupon. Total modeled cost must not
exceed the budget, and each coupon is capped at a configured share of users.

The legacy internal key `delta_gtv_causal` denotes a model-adjusted incremental
GTV value. It is retained temporarily for compatibility with historical reports;
it is **not** a causal estimate. New prose and UI labels use “model-adjusted
incremental GTV.”

## 4. Core mechanism and technical decisions

The simulator transforms user value, activity, sensitivity, time-of-day, and
coupon metadata into funnel probabilities. It compares modeled GTV under a
coupon with the no-coupon counterfactual and applies a documented natural-
conversion discount. This is a heuristic inside the simulation, not causal
identification from treatment and control observations.

The optimizer precomputes user-coupon outcomes, adds an uncertainty bonus, and
ranks feasible pairs by adjusted value per unit cost. A greedy pass enforces the
budget, one-coupon-per-user rule, and per-coupon coverage cap. Repeated rounds
update synthetic beliefs using model-implied responses. For that reason the
method is described as **iterative UCB-style allocation**, not an online
contextual bandit.

The same-budget uniform baselines are intentional: comparing a constrained
policy with an unconstrained full-population policy would confound policy quality
with budget and exposure differences.

### Sequential contextual-bandit experiment

`contextual_bandit.py` is deliberately separate from the full-information
allocator. The maintained policy uses one shared LinUCB model over user context,
coupon properties, and their interactions; a disjoint implementation remains as
a diagnostic baseline. At each step the policy selects one feasible action,
observes only that action's batched stochastic purchase, GTV, and coupon cost,
and updates from that selected-action feedback.

The environment and learner have separate responsibilities. The environment may
compute expected rewards for evaluation and oracle regret, but those values are
not passed to the policy update. Budget feasibility is checked before selection,
and realized coupon cost is deducted only after a simulated purchase. This is a
real partial-feedback contextual-bandit loop evaluated in simulation, not a
claim that the competition tables contain logged online-bandit feedback.

## 5. Testing and validation

The public suite uses deterministic invented spreadsheets. It checks:

- fixture generation and reproducibility;
- end-to-end feature construction, simulation, and allocation;
- budget, uniqueness, and coupon-coverage constraints;
- zero-budget behavior;
- progress reporting without importing Streamlit;
- rejection of malformed optimizer inputs;
- cache identifier validation against path traversal;
- partial-feedback LinUCB updates, seeded stochastic outcomes, budget safety,
  and non-negative expected regret.

The suite does not prove that the behavioral assumptions hold in production.
Sensitivity analysis and Monte Carlo runs inspect model behavior; a randomized
online experiment would be required to estimate real incremental impact.

## 6. Security, privacy, and operations

Raw data, exports, local environments, logs, and result caches are ignored by
Git. The optional explanation service reads its credential from the environment
and is not required for the core workflow. Because it sends selected records to
an external provider, it must not be enabled with personal or commercially
sensitive fields without authorization and minimization.

The local result cache uses pickle to retain DataFrames. Run identifiers are
validated to prevent path traversal, but pickle remains unsafe for untrusted
artifacts. Only artifacts created by the local application may be loaded.

## 7. Limitations and next decisions

- The population and outcomes are semi-synthetic; reported ROI and GTV values
  are scenarios rather than observed gains.
- Repeated UCB-style rounds do not consume live feedback. A real online-learning
  production extension would need a deployed observation stream. The separate
  LinUCB experiment now supplies a valid simulated learning loop, but does not
  remove this production-data limitation.

### Why a contextual bandit is conditionally appropriate

The competition question combines repeated user contexts, a finite action set,
immediate coupon outcomes, and a budget. This matches contextual bandits only if
the system can observe action–outcome feedback after deployment. The supplied
tables do not contain a usable logging propensity and unified treatment/outcome
record, so they cannot support unbiased offline policy evaluation. The current
experiment is therefore a simulator study, not an offline estimate of Meituan
policy value.

The raw group-mean uplift for rare coupons was also unstable and confounded by
user mix. Coupon uplift is now shrunk toward an amount-based prior with weight
`n / (n + 30)`. The bandit environment adds an explicit, switchable scenario
assumption: response is highest when coupon value as a fraction of basket value
is near a sensitivity-dependent target. This creates a transparent personalized
decision problem; it is not claimed as a relationship identified from the source
tables.

The maintained policy uses shared LinUCB features for user variables, coupon
properties, and their interactions. Feedback is aggregated over 16 simulated
exposures per update to represent batched campaign telemetry. A control variate
subtracts action-independent no-coupon expected GTV, so the reward targets
incremental value rather than learning which users naturally spend more.

The reproducible benchmark uses 500 users, 20,000 updates, a 1,000-update
reporting window, three seeds, `alpha=0.25`, and a 120,000 simulated budget. In
the reviewed fixture, the best fixed action has expected incremental net reward
1.0688 and the contextual oracle 1.2560. The three final-window LinUCB rewards
are 1.2146, 1.2137, and 1.2111; final-window regrets are 0.0467, 0.0454, and
0.0548. Thus the learned policy captures roughly 96% of the simulator oracle and
exceeds the best fixed action by about 13%, within this declared scenario only.

Method references: Li et al., *A Contextual-Bandit Approach to Personalized News
Article Recommendation* (WWW 2010), and Chu et al., *Contextual Bandits with
Linear Payoff Functions* (AISTATS 2011). Logged production evaluation would
require propensities and methods such as IPS or doubly robust evaluation rather
than this simulator oracle.
- The historical metric key should eventually be migrated to a neutral name
  across reports and visualizations, with a versioned result schema.
- `app.py` still performs substantial orchestration. A future application service
  can produce a typed result object shared by the UI and batch pipeline.
- Cache artifacts need a schema/version field before long-term compatibility can
  be promised.

These limitations are preferred to adding microservices or cloud deployment
without a demonstrated operational need.
