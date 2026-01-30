from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _read_processed_data(obx_path: Path, yield_path: Path) -> Tuple[pd.Series, pd.Series]:
    # Simple file reading — assume files and expected columns are present
    obx = pd.read_csv(obx_path, index_col=0, parse_dates=True)
    # 'Change %' expected like '1.23%'
    market_simple = obx['Change %'].astype(str).str.rstrip('%').astype(float) / 100
    market_ret = np.log1p(market_simple)

    ny = pd.read_csv(yield_path, index_col=0, parse_dates=True)
    rf_ann = ny['Price']
    # convert annualized percent yield to daily rate (252 trading days) then to log-return
    rf_simple = rf_ann / 100.0 / 252.0
    rf = np.log1p(rf_simple)
    return market_ret, rf


def test_strategy(
    strategy_returns: Sequence[float],
    underlying_prices: Sequence[Sequence[float]],
    underlying_names: Optional[Sequence[str]] = None,
    dates: Optional[Sequence] = None,
    data_dir: str = 'data/processed',
    initial_capital: float = 1_000_000.0,
) -> Dict[str, Any]:
    """Run CAPM test and produce diagnostic plots for a strategy.

    Parameters
    - strategy_returns: sequence of daily strategy returns (floats). May contain NaN for first value.
    - underlying_prices: list of sequences (one per underlying) of raw prices (not returns), same length as strategy_returns.
    - underlying_names: optional names for the underlying stocks (defaults to Stock1, Stock2, ...).
    - dates: optional sequence of datetimes matching the length of the provided arrays. If omitted,
      the function will align the input to the last N dates available in processed OBX and yield files.
    - data_dir: path to processed data (contains OBX and 10y yield files).

    Returns a dict with keys: 'model' (statsmodels RegressionResults), 'series' (dict of pandas Series),
    and 'figures' (list of matplotlib.Figure).
    """
    root = Path(data_dir)
    obx_path = root / 'OBX-Historical-2000-2025.csv'
    yield_path = root / '10-Year-Government-Bond-Yield-Norway.csv'

    market_ret_full, rf_full = _read_processed_data(obx_path, yield_path)

    # determine date index to use
    common_dates = market_ret_full.index.intersection(rf_full.index).sort_values()
    n = len(strategy_returns)

    if dates is not None:
        idx = pd.DatetimeIndex(dates)
        if len(idx) != n:
            raise ValueError('Length of dates must match strategy_returns')
    else:
        if n > len(common_dates):
            raise ValueError('Not enough processed data dates to align with provided returns')
        # assume inputs correspond to the most recent n common dates
        idx = common_dates[-n:]

    # build series (inputs expected as simple returns; convert to log-returns)
    strat_simple = pd.Series(np.asarray(strategy_returns, dtype=float), index=idx, name='strategy_simple')
    strat = np.log1p(strat_simple).rename('strategy')

    # prepare market and rf aligned (already returned as log-returns by helper)
    market_ret = market_ret_full.reindex(idx).astype(float).rename('market')
    rf = rf_full.reindex(idx).astype(float).rename('rf')

    # build underlying price series and compute buy-and-hold returns
    underlying_names = list(underlying_names) if underlying_names is not None else None
    underlying_series: List[pd.Series] = []
    bh_returns: Dict[str, pd.Series] = {}
    if underlying_names is None:
        underlying_names = [f'Stock{i+1}' for i in range(len(underlying_prices))]

    for name, prices in zip(underlying_names, underlying_prices):
        s = pd.Series(np.asarray(prices, dtype=float), index=idx, name=name)
        underlying_series.append(s)
        # log returns from prices
        ret = np.log(s).diff().fillna(0).rename(name)
        bh_returns[name] = ret

    # cumulative returns
    # cumulative returns from log-returns: exp(cumsum) - 1
    cumulative_strat = np.expm1(strat.fillna(0).cumsum())
    cumulative_mkt = np.expm1(market_ret.fillna(0).cumsum())

    # cumulative buy-and-hold for underlyings
    cumulative_bh = {name: np.expm1(r.cumsum()) for name, r in bh_returns.items()}

    # CAPM regression on excess returns
    # excess returns in log-space: difference of log-returns corresponds to log(1+r)/(1+rf)
    excess_strat = (strat - rf).rename('excess_strat')
    excess_mkt = (market_ret - rf).rename('excess_mkt')
    # align and drop NA
    df_reg = pd.concat([excess_strat, excess_mkt], axis=1).dropna()
    if df_reg.empty:
        raise ValueError('No overlapping data to run regression after alignment')
    X = sm.add_constant(df_reg['excess_mkt'])
    y = df_reg['excess_strat']
    model = sm.OLS(y, X).fit()

    # Plot 1: cumulative returns (Market vs Strategy vs buy-and-hold underlyings)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(cumulative_mkt.index, cumulative_mkt * 100, label='Market (OBX)', color='black', linewidth=2)
    ax1.plot(cumulative_strat.index, cumulative_strat * 100, label='Strategy', color='tab:blue', linewidth=2)
    for name, cum in cumulative_bh.items():
        ax1.plot(cum.index, cum * 100, label=f'{name} (Buy&Hold)', linestyle='--', alpha=0.8)
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.set_title('Cumulative Returns: Market vs Strategy vs Buy&Hold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: cumulative excess returns
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    cum_excess_strat = np.expm1(excess_strat.fillna(0).cumsum())
    ax2.plot(cum_excess_strat.index, cum_excess_strat * 100, label='Strategy (Excess)', color='tab:blue', linewidth=2)
    # for name, r in bh_returns.items():
    #     excess_under = (r - rf).rename(name)
    #     cum_excess_under = np.expm1(excess_under.fillna(0).cumsum())
    #     ax2.plot(cum_excess_under.index, cum_excess_under * 100, label=f'{name} (Excess)', linestyle='--')
    cum_rf = np.expm1(rf.fillna(0).cumsum())
    ax2.plot(cum_rf.index, cum_rf * 100, label='Risk-free (cum)', linestyle=':', color='gray')
    ax2.set_ylabel('Cumulative Excess Return (%)')
    ax2.set_title('Cumulative Excess Returns: Strategy')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # print regression summary to console and include in return
    print(model.summary())

    results = {
        'model': model,
        'series': {
            # log-return series
            'strategy_log': strat,
            'strategy_simple': strat_simple,
            'market_log': market_ret,
            'rf_log': rf,
            'excess_strat_log': excess_strat,
            'excess_mkt_log': excess_mkt,
            'cumulative_strategy': cumulative_strat,
            'cumulative_market': cumulative_mkt,
            'cumulative_bh': cumulative_bh,
        },
        'figures': [fig1, fig2],
    }

    return results
