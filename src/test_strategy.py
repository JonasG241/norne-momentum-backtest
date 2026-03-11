from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _read_processed_data(yield_path: Path) -> pd.Series:
    # Simple file reading - assume file and expected columns are present
    ny = pd.read_csv(yield_path, index_col=0, parse_dates=True)
    rf_ann = ny['Price']
    # convert annualized percent yield to daily rate (252 trading days) then to log-return
    rf_simple = rf_ann / 100.0 / 252.0
    rf = np.log1p(rf_simple)
    return rf


def test_strategy(
    strategy_returns: Sequence[float],
    market_returns: Sequence[float],
    dates: Optional[Sequence] = None,
    data_dir: str = 'data/processed',
    market_name: str = 'Market',
) -> Dict[str, Any]:
    """Run a CAPM regression and produce diagnostic plots for a strategy.

    The CAPM equation estimated is:

        r_strat - r_f = alpha + beta * (r_mkt - r_f) + epsilon

    All returns are converted internally to log-returns before the regression.

    Parameters
    ----------
    strategy_returns : sequence of float
        Daily simple returns of the strategy, one value per trading day.
        The first entry may be NaN (treated as 0 return for cumulative curves).
    market_returns : sequence of float
        Daily simple returns of the market portfolio used as ``r_mkt`` in the
        CAPM equation.  The caller is responsible for constructing this series;
        common choices are:
          * OBX (or another broad index) log-returns read from a CSV;
          * an equal-weight daily portfolio of the relevant stocks;
          * a market-cap-weighted portfolio built from ``weights * asset_returns``.
        Must be the same length as ``strategy_returns``.
    dates : sequence of datetime-like, optional
        Explicit date index matching the length of ``strategy_returns``.  When
        omitted the function assumes the inputs correspond to the *most recent*
        ``n`` trading days present in the 10-year yield file (used for r_f).
    data_dir : str, optional
        Path to the folder containing ``10-Year-Government-Bond-Yield-Norway.csv``
        (the risk-free rate source).  Defaults to ``'data/processed'``.
    market_name : str, optional
        Label used for the market series in plots and returned series dict.
        Defaults to ``'Market'``.

    Returns
    -------
    dict with keys:
      'model'   – statsmodels OLS RegressionResults (alpha, beta, p-values, …)
      'series'  – dict of pandas Series (log-returns, excess returns, cumulatives)
      'figures' – list of two matplotlib Figure objects:
                    [0] cumulative returns: strategy vs market
                    [1] cumulative excess returns: strategy vs risk-free
    """
    if len(market_returns) != len(strategy_returns):
        raise ValueError(
            f'market_returns (len={len(market_returns)}) must be the same length '
            f'as strategy_returns (len={len(strategy_returns)})'
        )

    root = Path(data_dir)
    yield_path = root / '10-Year-Government-Bond-Yield-Norway.csv'
    rf_full = _read_processed_data(yield_path)

    # ------------------------------------------------------------------ #
    # Build the date index                                                 #
    # ------------------------------------------------------------------ #
    n = len(strategy_returns)
    if dates is not None:
        idx = pd.DatetimeIndex(dates)
        if len(idx) != n:
            raise ValueError('Length of dates must match strategy_returns')
    else:
        available_dates = rf_full.index.sort_values()
        if n > len(available_dates):
            raise ValueError('Not enough dates in the yield file to align with provided returns')
        # Assume inputs cover the most recent n trading days in the yield file
        idx = available_dates[-n:]

    # ------------------------------------------------------------------ #
    # Build log-return series                                              #
    # strategy and market come in as simple returns; convert to log-returns
    # so that cumulative returns are simply exp(cumsum) - 1.              #
    # ------------------------------------------------------------------ #
    strat_simple = pd.Series(np.asarray(strategy_returns, dtype=float), index=idx, name='strategy_simple')
    strat = np.log1p(strat_simple).rename('strategy')

    mkt_simple = pd.Series(np.asarray(market_returns, dtype=float), index=idx, name=f'{market_name}_simple')
    mkt = np.log1p(mkt_simple).rename(market_name)

    # Risk-free rate: annualised 10y yield converted to daily log-return
    rf = rf_full.reindex(idx).astype(float).rename('rf')

    # ------------------------------------------------------------------ #
    # CAPM regression:  (r_strat - r_f) = alpha + beta*(r_mkt - r_f)     #
    # All quantities are daily log-returns.                               #
    # ------------------------------------------------------------------ #
    excess_strat = (strat - rf).rename('excess_strat')
    excess_mkt   = (mkt   - rf).rename('excess_mkt')

    df_reg = pd.concat([excess_strat, excess_mkt], axis=1).dropna()
    if df_reg.empty:
        raise ValueError('No overlapping non-NaN data available to run the CAPM regression')

    X = sm.add_constant(df_reg['excess_mkt'])   # adds the intercept (= alpha)
    y = df_reg['excess_strat']
    model = sm.OLS(y, X).fit()

    # ------------------------------------------------------------------ #
    # Cumulative return curves  (convert log-return cumsum to pct return) #
    # ------------------------------------------------------------------ #
    cumulative_strat = np.expm1(strat.fillna(0).cumsum())
    cumulative_mkt   = np.expm1(mkt.fillna(0).cumsum())

    # ------------------------------------------------------------------ #
    # Plot 1: cumulative total returns — strategy vs market                #
    # ------------------------------------------------------------------ #
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(cumulative_mkt.index,   cumulative_mkt   * 100, label=market_name,  color='black',   linewidth=2)
    ax1.plot(cumulative_strat.index, cumulative_strat * 100, label='Strategy', color='tab:blue', linewidth=2)
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.set_title(f'Cumulative Returns: Strategy vs {market_name}')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ------------------------------------------------------------------ #
    # Plot 2: cumulative excess returns over risk-free rate                #
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    cum_excess_strat = np.expm1(excess_strat.fillna(0).cumsum())
    cum_excess_mkt   = np.expm1(excess_mkt.fillna(0).cumsum())
    cum_rf           = np.expm1(rf.fillna(0).cumsum())
    ax2.plot(cum_excess_strat.index, cum_excess_strat * 100, label='Strategy (excess)',       color='tab:blue', linewidth=2)
    ax2.plot(cum_excess_mkt.index,   cum_excess_mkt   * 100, label=f'{market_name} (excess)', color='black',   linewidth=2, linestyle='--')
    ax2.plot(cum_rf.index,           cum_rf           * 100, label='Risk-free (cum)',         color='gray',    linewidth=1, linestyle=':')
    ax2.set_ylabel('Cumulative Excess Return (%)')
    ax2.set_title('Cumulative Excess Returns over Risk-Free Rate')
    ax2.legend()
    ax2.grid(alpha=0.3)

    print(model.summary())

    return {
        'model': model,
        'series': {
            'strategy_log':        strat,
            'strategy_simple':     strat_simple,
            'market_log':          mkt,
            'market_simple':       mkt_simple,
            'rf_log':              rf,
            'excess_strat_log':    excess_strat,
            'excess_mkt_log':      excess_mkt,
            'cumulative_strategy': cumulative_strat,
            'cumulative_market':   cumulative_mkt,
        },
        'figures': [fig1, fig2],
    }
