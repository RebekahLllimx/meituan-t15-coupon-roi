# Final Technical Report

## Executive summary

The maintained version replaces the original full-information, iterative allocation demonstration with a genuine sequential contextual-bandit experiment. A shared LinUCB policy now selects one coupon action, receives only that action's stochastic feedback, and updates online while respecting a remaining budget.

In the fixed reference scenario, the learned policy reached an expected incremental net reward of 1.2111–1.2146 in the final 1,000-update window across three seeds. The best fixed action scored 1.0688 and the contextual oracle scored 1.2560. The learned policy therefore captured about 96% of the simulator oracle and exceeded the best fixed action by about 13% inside this semi-synthetic environment.

These are technical experiment results, not observed Meituan revenue lift.

## 1. Decision problem

The policy receives a user context and chooses among no coupon and a small coupon catalog. The desired policy must learn heterogeneous response rather than issue one globally preferred coupon, and it must avoid selecting actions that the remaining budget cannot support.

The public source tables cannot answer this decision problem directly: they do not provide randomized treatment assignment, action propensities, or a unified record of context, action, and outcome. The project therefore uses them only to define schema and broad scenario parameters, then evaluates learning in a declared semi-synthetic environment.

## 2. Maintained method

### User and action representation

User context includes standardized basket value, activity, and posterior coupon sensitivity, together with quadratic and interaction terms. Coupon features include value, free/paid status, expanded-value status, uplift assumption, and attraction factor. User-coupon interactions allow one shared linear model to learn different responses across contexts while transferring information between actions.

### Feedback and reward

Each update aggregates 16 simulated exposures for the selected user-action pair. The learner sees only the selected action's realized purchase rate, coupon cost, and reward. Reward is simulated coupon GTV minus a coupon-cost penalty and an action-independent expected no-coupon baseline.

### Learning rule

Shared LinUCB estimates a linear reward model and adds an uncertainty bonus controlled by `alpha`. Ridge regularization stabilizes early estimates. The inverse covariance matrix is maintained with a Sherman–Morrison update, reducing per-update linear-algebra cost compared with repeated matrix inversion.

## 3. Benchmark protocol

The committed benchmark command is:

```bash
python benchmark_contextual_bandit.py \
  --users 500 \
  --steps 20000 \
  --window 1000 \
  --seeds 3 \
  --alpha 0.25 \
  --feedback-batch-size 16 \
  --budget 120000
```

The population generator uses seed 42. Policy seeds are 0, 1, and 2. All policies are evaluated under the same simulator and incremental-net reward definition.

## 4. Results

### Environment references

| Reference policy | Expected incremental net reward |
| --- | ---: |
| Random action | 0.5939 |
| Best fixed action | 1.0688 |
| Contextual oracle | 1.2560 |

### Learned policy by seed

| Seed | Final-window expected reward | Final-window regret | Oracle capture |
| ---: | ---: | ---: | ---: |
| 0 | 1.2146 | 0.0467 | approximately 96% |
| 1 | 1.2137 | 0.0454 | approximately 96% |
| 2 | 1.2111 | 0.0548 | approximately 96% |

The mean final-window reward is 1.2131, with a standard deviation of 0.0018 across seeds. Relative to the best fixed action, it is about 13.5% higher. Mean oracle capture is 96.12%, and mean regret falls by 90.00% from the first to the final reporting window. The remaining gap to the contextual oracle is roughly 0.05 reward units per update.

## 5. What changed technically

| Earlier version | Maintained version | Effect |
| --- | --- | --- |
| Scored all user-coupon pairs from the simulator | Reveals feedback only for the selected action | Establishes a real exploration-and-learning loop |
| Disjoint action models | Shared context-action representation | Transfers evidence across coupon actions |
| Repeated direct inverse operations | Sherman–Morrison rank-one update | Makes sequential updates computationally cheaper |
| Noisy single simulated outcome | Batched feedback over 16 exposures | Lowers reward variance while retaining partial feedback |
| Reward dominated by natural spender value | Incremental-net reward with a no-coupon control variate | Better aligns learning with subsidy value |
| Raw rare-coupon group means | Reliability shrinkage toward an amount prior | Reduces small-sample instability |
| Historical multi-page report UI | One English experiment dashboard and one final report | Makes the maintained method and evidence boundary unambiguous |

## 6. Validation

The automated suite covers deterministic inputs, end-to-end feature construction, simulator validity, partial-feedback updates, inverse-update accuracy, seeded stochastic behavior, contextual personalization, budget safety, and invalid configuration. GitHub Actions runs the suite on every push and pull request.

The Streamlit page uses the same domain modules as the benchmark. It exposes settings, rolling learning curves, action frequencies, budget use, and assumptions; it does not call an external API or load private inputs.

## 7. Interpretation and limits

The result demonstrates that a contextual policy can converge in the specified simulator and outperform a non-personalized fixed action. It does not establish that contextual bandits will produce the same gain in live traffic, nor that the response assumptions describe Meituan users.

A production evaluation would require randomized or known-propensity logging, treatment guardrails, delayed-conversion attribution, operational budget pacing, and offline policy evaluation such as inverse propensity scoring or doubly robust estimation before an online experiment.

## 8. Reproducibility

Run the test suite and benchmark from the repository root after installing `requirements.txt`. The synthetic generator and all experiment seeds are explicit. Original competition files, private data, caches, and generated outputs are intentionally excluded from version control.
