# Meituan T15 Afternoon-Tea Coupon Strategy

This repository presents a student project developed for the Meituan T15 business-data-science competition. The project studies how afternoon-tea coupons can be evaluated and allocated across heterogeneous users under a limited subsidy budget.

The supplied competition tables describe user profiles, coupon orders, and behavior sequences, but they do not form a complete context–treatment–outcome log. The project therefore builds a **semi-synthetic decision environment**: source tables define schemas, scene constraints, broad distributions, and funnel structure; simulated users and outcomes make alternative coupon policies comparable.

> **Scope:** this is an independently maintained competition project, not an official Meituan system. Public inputs are invented, and all reported rewards, regret, and policy improvements are simulator results—not measured business impact or causal-effect estimates.

## Business question

The afternoon-tea scenario presents three linked problems:

1. users differ in basket value, activity, and price sensitivity;
2. one coupon policy may overspend on users who would purchase without a subsidy;
3. a useful strategy must balance personalization, learning, and a finite campaign budget.

The current project turns this into a sequential decision problem. For each update, the policy observes a user context, selects either no coupon or one candidate coupon, receives only the selected action's stochastic outcome, and updates its model without exceeding the remaining budget.

## System design

```text
competition-style spreadsheets / invented public fixture
                         │
                         ▼
                FeatureEngineering
      scene filtering + distributions + user generation
                         │
                         ▼
                 SimulationEngine
        purchase funnel + coupon-response assumptions
                         │
                         ▼
                CouponBanditEnvironment
       context-action features + partial feedback
                         │
                         ▼
                    SharedLinUCB
       exploration + online update + budget state
                    /              \
                   ▼                ▼
           Streamlit project   benchmark CLI
              presentation     and test suite
```

### Data and scenario layer

`feature_engineering.py` adapts the original Chinese spreadsheet schema, filters afternoon-tea orders, extracts a view → click → cart → purchase funnel, fits customer-state distributions, and creates a reproducible semi-synthetic population. Chinese identifiers remain only at this raw-data boundary.

`simulation_engine.py` maps user state and coupon properties to purchase probability, expected coupon cost, and modelled incremental value. Heterogeneous coupon fit is an explicit scenario assumption rather than an estimated causal relationship.

### Allocation and learning layer

`contextual_bandit.py` implements shared LinUCB over user features, coupon properties, and their interactions. It uses:

- selected-action-only stochastic feedback;
- ridge regularization and Sherman–Morrison inverse updates;
- an incremental-net reward with a no-coupon control variate;
- batched campaign feedback;
- budget-aware action feasibility.

A disjoint LinUCB implementation is retained as a diagnostic baseline. Oracle rewards are calculated only for evaluation and are never passed to policy updates.

### Presentation layer

`app.py` presents the project rather than only the latest algorithm. Its English interface includes:

- the competition problem and current system architecture;
- source calibration, a synthetic-user map, funnel probabilities, and coupon catalog;
- learning and expected-regret curves;
- action mix, budget use, and segment-level decisions;
- reproducible business-readable explanations;
- the project's technical evolution and evidence boundary.

The historical prototype used an optional external LLM to paraphrase strategy output. The maintained page instead builds explanations directly from displayed values so that they remain deterministic, auditable, and free of unsupported claims.

## How the project changed

### 1. Competition prototype

The first version implemented afternoon-tea scene filtering, user heterogeneity, a purchase funnel, coupon-response simulation, budget-constrained allocation, sensitivity analysis, and a multi-step Streamlit report. It established the business workflow but used simulator access to score every user–coupon pair.

### 2. Public reproducibility

The repository was separated from private competition artifacts. A deterministic synthetic-data generator was added, claims were relabelled as simulated rather than causal, and tests began covering the public input-to-output path.

### 3. Sequential-learning correction

The maintained method introduced a true partial-feedback loop: the learner now observes only the action it selected. Shared context-action features replaced isolated coupon models, batched feedback reduced variance, reward centering emphasized subsidy increment, and a reliability prior reduced small-sample coupon noise.

### 4. Public sample consolidation

Duplicate Chinese drafts, stale generated figures, the old full-information allocator, external-LLM dependency, cache layer, and historical report pipeline were removed from the maintained path. Their useful ideas—progressive explanation, calibration visuals, strategy interpretation, and robustness reporting—were rebuilt directly in the current English dashboard and documentation.

## Reference benchmark

The committed benchmark uses 500 synthetic users, 20,000 learning updates, 16 simulated exposures per update, three policy seeds, `alpha=0.25`, and a CNY 120,000 simulated budget.

| Measure | Result |
| --- | ---: |
| Random-action expected reward | 0.5939 |
| Best fixed-action expected reward | 1.0688 |
| Contextual-oracle expected reward | 1.2560 |
| Mean final-window LinUCB reward | 1.2131 |
| Mean oracle capture | 96.12% |
| Mean first-to-final regret reduction | 90.00% |

Within this declared simulator scenario, the learned policy is about 13.5% above the best fixed action. This value must not be presented as live Meituan campaign lift.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m unittest discover -s tests -v
python run_bandit_demo.py --steps 500 --budget 3000 --seed 42
python benchmark_contextual_bandit.py
streamlit run app.py
```

`generate_synthetic_demo.py` can materialize the invented spreadsheets in `data/raw/`. It refuses to overwrite existing files.

## Repository guide

```text
app.py                         English project dashboard
contextual_bandit.py           environment-policy boundary and LinUCB
feature_engineering.py         source-schema adapter and user generation
simulation_engine.py           funnel and coupon-response simulator
benchmark_contextual_bandit.py fixed reproducible benchmark
generate_synthetic_demo.py     invented public spreadsheet fixture
tests/                          unit and integration tests
docs/ENGINEERING_DESIGN.md      software design and technical decisions
docs/FINAL_REPORT.md            current experiment and technical results
```

## Data, privacy, and limitations

The original competition spreadsheets are not included. `data/`, exported artifacts, credentials, local environments, and logs are ignored by Git. Do not commit personal or commercially sensitive records.

The public tables cannot support unbiased offline policy evaluation because they lack randomized assignment and logging propensities. Production use would require controlled exploration, propensity logging, delayed-outcome attribution, safety guardrails, and IPS or doubly robust evaluation.
