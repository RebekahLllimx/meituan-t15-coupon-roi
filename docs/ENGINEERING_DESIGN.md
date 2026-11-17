# Engineering Design

## 1. Problem and requirements

The project asks whether a coupon policy can learn different actions for different users while respecting a campaign budget. It is designed for reviewers and analysts who want an inspectable contextual-bandit example, not for issuing real coupons.

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
- no credentials, private data, generated reports, or local caches in Git.

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

`feature_engineering.py` is the input adapter. `simulation_engine.py` owns outcome assumptions. `contextual_bandit.py` owns the environment-policy boundary and learning loop. `app.py` renders results but contains no policy logic. The command-line scripts provide small and full benchmark entry points.

## 3. Interfaces and data design

The generated user table contains one unique `user_id` and the state required by the simulator: expected basket value (`V`), activity (`A`), posterior sensitivity parameters, and exposure/success counts. Duplicate identifiers and empty populations are rejected.

The action catalog contains a no-coupon action plus coupon metadata: face value, free/paid indicator, expanded-value indicator, basket uplift assumption, and attraction factor. Source-language coupon names remain an adapter detail; the UI translates them into English labels.

For shared LinUCB, the feature vector combines:

- an intercept and standardized user value, activity, and sensitivity;
- quadratic and pairwise user terms for nonlinear thresholds;
- coupon properties;
- user-property interactions that permit personalized action response.

Each `BanditStep` records only the chosen action's realized reward, cost, and purchase rate, plus expected values used for evaluation. Oracle expected rewards are never provided to the policy update.

## 4. Implementation decisions

### Partial feedback

At each update, a user is sampled, the policy scores feasible actions, and the environment returns stochastic feedback only for the selected action. The maintained shared model transfers information across coupon actions; the disjoint implementation is retained as a diagnostic baseline.

### Incremental reward

The raw selected-action reward is simulated coupon GTV minus `0.25 × coupon cost`. An action-independent expected no-coupon GTV is subtracted as a control variate. This reduces variance and focuses the learner on incremental value without revealing another coupon's realized outcome.

### Numerical update

LinUCB uses ridge regularization and a Sherman–Morrison rank-one inverse update. This avoids recomputing a matrix inverse after every observation. Monetary rewards are scaled before learning so that confidence bounds remain numerically meaningful; unscaled values are retained in the history.

### Personalization scenario

The source tables do not identify individual treatment effects. The simulator therefore declares a scenario assumption: response is strongest when coupon value as a fraction of basket value is near a sensitivity-dependent target. Rare-coupon uplift is shrunk toward an amount-based prior using reliability weight `n / (n + 30)`. Both choices are modeling assumptions, not recovered causal relationships.

### Budget handling

An action is feasible only when its face value does not exceed remaining budget. Realized coupon cost is deducted only when a simulated purchase occurs. The loop therefore cannot spend more than the supplied budget.

## 5. Testing and validation

The test suite checks:

- deterministic fixture generation and successful schema adaptation;
- finite simulator outputs and multiple context-dependent optimal actions;
- selected-action-only policy updates;
- the incremental inverse against a direct matrix inverse;
- seeded stochastic reproducibility;
- non-negative expected regret and budget safety;
- invalid configuration and malformed population rejection.

The benchmark compares random action, best fixed action, contextual oracle, and learned shared LinUCB under one environment and protocol. This validates code behavior inside the simulator; it does not validate the behavioral assumptions against production traffic.

## 6. Deployment and operations

The project is intentionally a local research application. Installation is defined by `requirements.txt`; tests run in GitHub Actions; Streamlit provides the demonstration page. No database, microservice, external LLM, or cloud deployment is added without an operational requirement.

Raw data, exports, logs, virtual environments, and editor artifacts are excluded by `.gitignore`. The public demo generates its own temporary input files and deletes them after feature construction.

## 7. Limitations and next steps

- Public results are semi-synthetic and are not evidence of campaign lift.
- The simulator oracle is available only for evaluation; a production system cannot query it.
- Source tables lack randomized assignments and logging propensities, so they cannot support unbiased offline policy evaluation.
- Production use would require controlled exploration, propensity logging, delayed-outcome handling, safety constraints, drift monitoring, and IPS or doubly robust evaluation.
- A feature/schema version should be introduced before stored models or cross-version result comparison are supported.

The current design favors a narrow, testable learning system over decorative microservices or unsupported scalability claims.
