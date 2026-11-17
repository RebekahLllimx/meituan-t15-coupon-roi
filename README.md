# Contextual Coupon Allocation

A reproducible research prototype for studying budget-aware coupon decisions with a contextual bandit. The maintained policy uses shared LinUCB features to learn from partial, stochastic feedback in a semi-synthetic afternoon-tea ordering environment.

> **Evidence boundary:** this repository does not contain Meituan production data. All public inputs are invented, and all rewards, regret, and policy improvements are simulator results—not measured business impact or causal-effect estimates.

## What is implemented

- a raw-data adapter that constructs customer value, activity, and price-sensitivity features;
- a documented purchase-funnel simulator with heterogeneous coupon response;
- shared and disjoint LinUCB policies with incremental inverse updates;
- one-action-at-a-time feedback, budget feasibility, and seeded stochastic outcomes;
- deterministic synthetic spreadsheets that exercise the original input contract;
- a Streamlit interface for experiment settings, learning curves, action mix, and assumptions;
- reproducible tests and a three-seed benchmark protocol.

## Architecture

```text
invented spreadsheets
        │
        ▼
feature_engineering.py ──► semi-synthetic users + environment parameters
        │
        ▼
simulation_engine.py ────► stochastic selected-action feedback
        │
        ▼
contextual_bandit.py ────► shared LinUCB update + budget state
        │
        ├──► app.py                         interactive demonstration
        └──► benchmark_contextual_bandit.py reproducible evaluation
```

Chinese column names and categorical values are retained only at the raw spreadsheet boundary because they are part of the original competition schema. Internal documentation, comments, interfaces, and the public UI are in English.

## Reproduce locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m unittest discover -s tests -v
python run_bandit_demo.py --steps 500 --budget 3000 --seed 42
python benchmark_contextual_bandit.py
streamlit run app.py
```

`generate_synthetic_demo.py` can also materialize the invented spreadsheets in `data/raw/`. The generator refuses to overwrite existing files.

## Maintained documentation

- [Engineering design](docs/ENGINEERING_DESIGN.md): requirements, architecture, interfaces, decisions, tests, and limitations.
- [Final technical report](docs/FINAL_REPORT.md): experiment protocol, benchmark results, interpretation, and production gap.

The repository intentionally keeps one report path. Historical competition drafts, duplicate formula notes, and generated figures are excluded from the maintained public sample.

## Reference benchmark

Using 500 synthetic users, 20,000 policy updates, 16 simulated exposures per update, three seeds, `alpha=0.25`, and a CNY 120,000 simulated budget:

| Reference | Expected incremental net reward |
| --- | ---: |
| Best fixed action | 1.0688 |
| Contextual oracle | 1.2560 |
| Shared LinUCB, final 1,000-update window | 1.2111–1.2146 |

Within this declared simulator scenario, the policy captured about 96% of oracle reward and exceeded the best fixed action by about 13%. These values must not be presented as live campaign lift.

## Data and privacy

The original competition spreadsheets are not included. `data/`, exported artifacts, local environments, and logs are ignored by Git. Do not commit personal or commercially sensitive records.
