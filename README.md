# Norne Momentum Backtest

Research code for testing momentum and moving-average strategies in the Norwegian equity market. The project studies whether historical price trends and moving-average configurations can be used to rank OSEBX constituents and construct systematic long-only and long/short portfolios.

The repository contains two generations of the project:

- `backtest_v1/` preserves the original research workspace, including the first notebooks, tests, source layout, and public sample data.
- `backtest_v2/` contains the current strategy and scoring implementations in a cleaner, public-safe structure.

## Current implementation

The v2 research covers two related approaches:

1. **Winner and loser momentum strategies** rank eligible OSEBX stocks by past returns and evaluate overlapping holding-period portfolios.
2. **Moving-average scoring** compares 50-, 100-, and 200-day moving averages, evaluates alternative ordering rules, and tests equal-weighted and capitalization-weighted portfolio constructions.

The backtests account for historical index membership and use delayed signal execution to reduce look-ahead bias. Results from parameter grids should still be interpreted as in-sample research rather than out-of-sample evidence.

## Repository structure

```text
.
├── backtest_v1/               # Archived first version of the project
│   ├── configs/
│   ├── data/
│   ├── notebooks/
│   ├── reports/
│   ├── src/
│   └── tests/
└── backtest_v2/               # Current implementation
    ├── data/                  # Local private inputs (not committed)
    ├── notebooks/
    │   ├── data_preparation.ipynb # Builds derived local inputs
    │   ├── Ls.ipynb          # Long-only loser strategy
    │   ├── MVA_Scores.ipynb  # Moving-average scoring and portfolio tests
    │   ├── Ws.ipynb          # Long-only winner strategy
    │   └── WsLs_new.ipynb    # Winner-minus-loser strategy
    ├── results/               # Generated tables (not committed)
    ├── src/
    │   ├── WsLs_new.py        # Script version of the long/short backtest
    │   ├── equal_weight_ma_strategy_fixed_calendar.py
    │   ├── equal_weight_ma_strategy_score_exposure.py
    │   ├── momentum_validation.py
    │   └── mva_order_utils.py # Data loading and MA-scoring utilities
    └── tests/
        └── test_wsls_strategy.py
```

Notebook outputs and execution counts are removed before publication so the repository does not expose generated results or embedded data.

## Private data

The v2 market data is licensed/private and is intentionally excluded from the repository. To run the current code, place these source files in `backtest_v2/data/`:

```text
TOTRET_DAILY.csv
MCAP_DAILY.csv
ON_INDEX.csv
membership_m.csv
weights_wide.csv
```

The first three files are private source inputs. `membership_m.csv` and `weights_wide.csv` are derived inputs created locally by `data_preparation.ipynb`; they are listed here because the winner and loser notebooks consume them directly.

The expected inputs are wide time-series tables containing total-return levels, market capitalizations, and historical OSEBX membership. See [`backtest_v2/data/README.md`](backtest_v2/data/README.md) for the local data directory convention.

Do not commit private data. The repository's `.gitignore` excludes everything in `backtest_v2/data/` except its README and placeholder file.

## Getting started

Create a Python environment and install the main research dependencies:

```bash
python -m venv .venv
python -m pip install numpy pandas matplotlib statsmodels jupyter
```

After adding the required data, start Jupyter from `backtest_v2/notebooks/` so the notebooks' relative data paths resolve correctly:

```bash
cd backtest_v2/notebooks
jupyter notebook
```

The standalone long/short implementation can be run from the repository root:

```bash
python backtest_v2/src/WsLs_new.py
```

## Methodology notes

- Signals are formed using information available at the portfolio-formation date.
- Historical OSEBX membership is used to reduce survivorship bias.
- Momentum portfolios use overlapping cohorts and configurable formation and holding periods.
- Moving-average experiments compare alignment and full-ordering scores based on 50-, 100-, and 200-day averages.
- Transaction costs, missing observations, weighting conventions, and parameter selection can materially affect reported performance.

This repository is intended for research and educational use. It is not investment advice.
