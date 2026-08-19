from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TOTRET_CSV = DATA_DIR / "TOTRET_DAILY.csv"
MCAP_CSV = DATA_DIR / "MCAP_DAILY.csv"
ON_INDEX_CSV = DATA_DIR / "ON_INDEX.csv"


# ---------------------------------------------------------------------
# Strategy assumptions
# ---------------------------------------------------------------------

ANNUAL_RISK_FREE_RATE = 0.03
TRADING_DAYS = 252
TRANSACTION_COST = 0 #0.0015
HAC_LAGS = 5

# Binary MA-score rule:
# score >= 1 -> in the stock
# score <= 0 -> out of the stock
SCORE_IN_THRESHOLD = 1


def read_daily_csv(path: str) -> pd.DataFrame:
    return (
        pd.read_csv(
            path,
            sep=";",
            decimal=",",
            parse_dates=["Date"],
            dayfirst=True,
            na_values=["#N/A"],
        )
        .set_index("Date")
        .sort_index()
    )


def read_membership_csv(path: str) -> pd.DataFrame:
    membership = pd.read_csv(path, sep=";", engine="python")
    membership = membership.rename(columns={membership.columns[0]: "date"})

    membership["date"] = pd.to_datetime(
        membership["date"],
        format="%d.%m.%Y",
        dayfirst=True,
        errors="coerce",
    )

    membership = membership.dropna(subset=["date"]).set_index("date").sort_index()

    id_rows = membership.apply(
        lambda row: row.astype(str).str.contains(r"id\(\)", case=False, na=False)
    ).any(axis=1)
    membership = membership.loc[~id_rows]

    membership = membership.astype("string")
    return (membership.notna() & (membership != "")).astype(int)


def daily_risk_free(index: pd.Index) -> pd.Series:
    daily_rf = (1.0 + ANNUAL_RISK_FREE_RATE) ** (1.0 / TRADING_DAYS) - 1.0
    return pd.Series(daily_rf, index=index, name="rf")


def portfolio_returns_with_tc(
    target_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    rf: pd.Series,
    fee: float,
) -> pd.Series:
    """
    Computes daily portfolio returns with transaction costs.

    The function uses target weights, lets yesterday's holdings drift with returns,
    then charges transaction costs on the value traded to reach today's target weights.

    fee is charged on total absolute stock turnover:
        turnover = sum_i abs(target_weight_i - drifted_weight_i)

    Cash earns the daily risk-free rate.
    """
    target = target_weights.fillna(0.0)
    rets = asset_returns.reindex_like(target).fillna(0.0)
    rf = rf.reindex(target.index).ffill().fillna(0.0)

    out = pd.Series(0.0, index=target.index, name="strategy_return")

    prev_stock_w = pd.Series(0.0, index=target.columns, dtype=float)
    prev_cash_w = 1.0

    for date in target.index:
        r_t = rets.loc[date]
        rf_t = float(rf.loc[date])
        target_stock_w = target.loc[date].clip(lower=0.0)
        target_stock_w = target_stock_w.where(np.isfinite(target_stock_w), 0.0)

        stock_sum = float(target_stock_w.sum())
        if stock_sum > 1.0 + 1e-10:
            target_stock_w = target_stock_w / stock_sum
            stock_sum = 1.0

        target_cash_w = max(0.0, 1.0 - stock_sum)

        # Drift yesterday's portfolio before rebalancing.
        gross_stock_value = prev_stock_w * (1.0 + r_t)
        gross_cash_value = prev_cash_w * (1.0 + rf_t)
        gross_total = float(gross_stock_value.sum() + gross_cash_value)

        if gross_total > 0:
            drifted_stock_w = gross_stock_value / gross_total
        else:
            drifted_stock_w = pd.Series(0.0, index=target.columns, dtype=float)

        turnover = float((target_stock_w - drifted_stock_w).abs().sum())
        cost = fee * turnover

        net_total = gross_total - cost
        out.loc[date] = net_total - 1.0

        # After rebalancing, next day's starting portfolio equals the target weights.
        # Costs reduce wealth, but do not create artificial negative cash/leverage.
        prev_stock_w = target_stock_w.copy()
        prev_cash_w = target_cash_w

    return out


def prepare_data() -> dict[str, pd.DataFrame | pd.Series]:
    """
    Prepares equal-weight data.

    TOTRET_DAILY.csv is assumed to contain total-return price/index levels,
    i.e. price including reinvested dividends.

    Important:
    Moving averages and returns are calculated from the raw total-return series.
    Membership is applied only when deciding whether a stock is tradable/holdable.
    This avoids destroying MA history when a stock enters/leaves the index.
    """
    price_raw = read_daily_csv(TOTRET_CSV).apply(pd.to_numeric, errors="coerce")

    # Important: remove dates where the entire total-return file has no prices.
    # Otherwise long market holidays / blank rows can break rolling MA windows and
    # create artificial sparse strategy calendars.
    price_raw = price_raw.dropna(how="all")

    membership_m = read_membership_csv(ON_INDEX_CSV)

    tickers = price_raw.columns

    membership_d = (
        membership_m.reindex(price_raw.index, method="ffill")
        .fillna(0)
        .reindex(columns=tickers)
        .fillna(0)
        .astype(int)
    )

    # Do NOT mask price by membership before calculating returns/MAs.
    price = price_raw.copy()

    ma50 = price.rolling(window=50, min_periods=50).mean()
    ma100 = price.rolling(window=100, min_periods=100).mean()
    ma200 = price.rolling(window=200, min_periods=200).mean()

    asset_returns = price.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    # Tradable means both current index member and valid return.
    valid_returns = asset_returns.notna() & membership_d.eq(1)

    member_count = membership_d.sum(axis=1).replace(0, np.nan)
    weights = membership_d.div(member_count, axis=0).where(membership_d.eq(1), 0.0)
    weights_lag = weights.shift(1).fillna(0.0)

    benchmark_raw = weights_lag.where(valid_returns, 0.0)
    benchmark_target = benchmark_raw.div(
        benchmark_raw.sum(axis=1).replace(0.0, np.nan),
        axis=0,
    ).fillna(0.0)

    rf = daily_risk_free(price.index)

    market_returns = portfolio_returns_with_tc(
        target_weights=benchmark_target,
        asset_returns=asset_returns.where(valid_returns),
        rf=rf,
        fee=0.0,
    )

    warmup_mask = pd.Series(np.arange(len(price)) >= 199, index=price.index)

    return {
        "price": price,
        "membership": membership_d,
        "ma50": ma50,
        "ma100": ma100,
        "ma200": ma200,
        "asset_returns": asset_returns,
        "valid_returns": valid_returns,
        "weights": weights,
        "weights_lag": weights_lag,
        "benchmark_target": benchmark_target,
        "market_returns": market_returns,
        "rf": rf,
        "warmup_mask": warmup_mask,
    }

def make_target_from_signal(
    signal: pd.DataFrame,
    data: dict[str, pd.DataFrame | pd.Series],
    mode: str,
) -> pd.DataFrame:
    """
    Converts a 0/1 signal into daily target stock weights.

    overlay:
        Keep the stock's equal index weight if signal=1.
        Otherwise replace that weight with cash.

    selection:
        Select stocks with signal=1, then renormalize selected stocks to 100%.
        If no stocks are selected, the portfolio is 100% cash.
    """
    if mode not in {"overlay", "selection"}:
        raise ValueError("mode must be either 'overlay' or 'selection'")

    weights_lag = data["weights_lag"]
    valid_returns = data["valid_returns"]

    signal = signal.reindex_like(weights_lag).fillna(0.0)
    raw = (weights_lag * signal).where(valid_returns, 0.0).fillna(0.0)

    if mode == "overlay":
        return raw

    denom = raw.sum(axis=1).replace(0.0, np.nan)
    return raw.div(denom, axis=0).fillna(0.0)


def cumulative_simple_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0


def evaluate_strategy(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    rf: pd.Series,
    hac_lags: int = HAC_LAGS,
) -> dict:
    """
    CAPM regression on daily log excess returns.

    Returns both ordinary OLS p-values and HAC/Newey-West p-values.
    """
    idx = strategy_returns.index.intersection(market_returns.index).intersection(rf.index)

    strat_simple = strategy_returns.reindex(idx).astype(float)
    market_simple = market_returns.reindex(idx).astype(float)
    rf_simple = rf.reindex(idx).astype(float)

    strat_log = np.log1p(strat_simple).rename("strategy")
    market_log = np.log1p(market_simple).rename("market")
    rf_log = np.log1p(rf_simple).rename("rf")

    excess_strat = (strat_log - rf_log).rename("excess_strat")
    excess_mkt = (market_log - rf_log).rename("excess_mkt")

    regression_data = pd.concat([excess_strat, excess_mkt], axis=1).dropna()
    if regression_data.empty:
        raise ValueError("No overlapping data for CAPM regression")

    x = sm.add_constant(regression_data["excess_mkt"])

    ols_model = sm.OLS(regression_data["excess_strat"], x).fit()
    hac_model = sm.OLS(regression_data["excess_strat"], x).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags},
    )

    alpha_daily = float(ols_model.params["const"])
    beta = float(ols_model.params["excess_mkt"])

    return {
        "alpha_daily_log": alpha_daily,
        "alpha_annual_log": alpha_daily * TRADING_DAYS,
        "alpha_annual_simple_approx": np.expm1(alpha_daily * TRADING_DAYS),
        "beta": beta,
        "alpha_pvalue_ols": float(ols_model.pvalues["const"]),
        "alpha_pvalue_hac": float(hac_model.pvalues["const"]),
        "beta_pvalue_ols": float(ols_model.pvalues["excess_mkt"]),
        "beta_pvalue_hac": float(hac_model.pvalues["excess_mkt"]),
        "ols_model": ols_model,
        "hac_model": hac_model,
        "n_obs": int(len(regression_data)),
        "cumulative_strategy": cumulative_simple_returns(strategy_returns),
        "cumulative_market": cumulative_simple_returns(market_returns),
    }


def ma_crossover_signal(short_ma: pd.DataFrame, long_ma: pd.DataFrame) -> pd.DataFrame:
    return (short_ma > long_ma).astype(float).shift(1).fillna(0.0)


def ma_score(
    price: pd.DataFrame,
    ma50: pd.DataFrame,
    ma100: pd.DataFrame,
    ma200: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mutually exclusive MA-score.
    """
    score = pd.DataFrame(0, index=price.index, columns=price.columns, dtype=float)

    conditions = [
        (price > ma50) & (ma50 > ma100) & (ma100 > ma200),
        (ma50 > ma100) & (ma100 > ma200),
        (ma50 > ma200),
        (ma50 > ma100),
        (ma200 > ma100) & (ma100 > ma50) & (ma50 > price),
        (ma200 > ma100) & (ma100 > ma50),
        (ma200 > ma50),
        (ma100 > ma50),
    ]
    values = [4, 3, 2, 1, -4, -3, -2, -1]

    arr = np.select([c.fillna(False).to_numpy() for c in conditions], values, default=0)
    score.loc[:, :] = arr
    return score


def ma_score_binary_signal(data: dict[str, pd.DataFrame | pd.Series]) -> pd.DataFrame:
    score = ma_score(data["price"], data["ma50"], data["ma100"], data["ma200"])
    return (score >= SCORE_IN_THRESHOLD).astype(float).shift(1).fillna(0.0)


def run_signal_strategy(
    strategy_name: str,
    signal: pd.DataFrame,
    data: dict[str, pd.DataFrame | pd.Series],
    mode: str,
    fee: float = TRANSACTION_COST,
) -> dict:
    target = make_target_from_signal(signal, data, mode=mode)

    eval_mask = data["warmup_mask"] & target.notna().any(axis=1)

    strategy_returns = portfolio_returns_with_tc(
        target_weights=target,
        asset_returns=data["asset_returns"].where(data["valid_returns"]),
        rf=data["rf"],
        fee=fee,
    )

    strategy_eval = strategy_returns.loc[eval_mask].fillna(0.0)
    market_eval = data["market_returns"].loc[eval_mask].fillna(0.0)
    rf_eval = data["rf"].loc[eval_mask].fillna(0.0)

    result = evaluate_strategy(strategy_eval, market_eval, rf_eval)

    result["strategy_name"] = strategy_name
    result["mode"] = mode
    result["strategy_returns"] = strategy_eval
    result["market_returns"] = market_eval
    result["target_weights"] = target.loc[eval_mask]
    result["eval_mask"] = eval_mask

    return result


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []

    for name, result in results.items():
        rows.append(
            {
                "strategy": name,
                "mode": result["mode"],
                "alpha_annual_log": result["alpha_annual_log"],
                "alpha_annual_simple_approx": result["alpha_annual_simple_approx"],
                "beta": result["beta"],
                "p_alpha_OLS": result["alpha_pvalue_ols"],
                "p_alpha_HAC": result["alpha_pvalue_hac"],
                "p_beta_OLS": result["beta_pvalue_ols"],
                "p_beta_HAC": result["beta_pvalue_hac"],
                "n_obs": result["n_obs"],
            }
        )

    return pd.DataFrame(rows).sort_values(["mode", "strategy"]).reset_index(drop=True)



# ---------------------------------------------------------------------
# Graphs: strategy vs equal-weight and market-weight OSEBX
# ---------------------------------------------------------------------

def load_market_cap_weights(
    data: dict[str, pd.DataFrame | pd.Series],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Market-cap weights for current index members, aligned to the return data."""
    mcap = read_daily_csv(MCAP_CSV).apply(pd.to_numeric, errors="coerce")
    mcap = mcap.reindex(index=data["asset_returns"].index, columns=data["asset_returns"].columns)
    mcap = mcap.where(data["membership"].eq(1))

    index_mcap = mcap.sum(axis=1, min_count=1).rename("osebx_index_mcap")

    mcap_lag = mcap.shift(1)
    market_weight_raw = mcap_lag.where(data["valid_returns"], 0.0)
    market_weight_target = market_weight_raw.div(
        market_weight_raw.sum(axis=1).replace(0.0, np.nan),
        axis=0,
    ).fillna(0.0)

    return market_weight_target, index_mcap, mcap


def market_weighted_osebx_returns(
    data: dict[str, pd.DataFrame | pd.Series],
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Market-cap weighted OSEBX-like benchmark return series."""
    market_weight_target, index_mcap, mcap = load_market_cap_weights(data)
    returns = portfolio_returns_with_tc(
        target_weights=market_weight_target,
        asset_returns=data["asset_returns"],
        rf=data["rf"],
        fee=0.0,
    )
    return returns.rename("market_weight_osebx"), index_mcap, mcap


def plot_strategy_against_osebx(
    results: dict[str, dict],
    equal_weight_osebx: pd.Series,
    market_weight_osebx: pd.Series,
) -> None:
    """
    One chart per strategy: strategy, equal-weight OSEBX, market-weight OSEBX.

    Important fix:
    The benchmark lines are plotted on the full benchmark calendar, not on each
    strategy's possibly sparse evaluation calendar. The strategy is reindexed to
    the benchmark calendar instead. Missing strategy dates are treated as 0%
    return, because those dates are benchmark dates where the strategy has no
    recorded trading return.
    """
    full_benchmark_idx = equal_weight_osebx.index.intersection(market_weight_osebx.index)

    for key, result in results.items():
        strategy_returns = result["strategy_returns"].astype(float)

        if strategy_returns.empty:
            continue

        benchmark_idx = full_benchmark_idx[
            (full_benchmark_idx >= strategy_returns.index.min())
            & (full_benchmark_idx <= strategy_returns.index.max())
        ]

        if len(benchmark_idx) == 0:
            continue

        strategy_aligned = strategy_returns.reindex(benchmark_idx).fillna(0.0)
        equal_weight_aligned = equal_weight_osebx.reindex(benchmark_idx).fillna(0.0)
        market_weight_aligned = market_weight_osebx.reindex(benchmark_idx).fillna(0.0)

        chart = pd.concat(
            [
                cumulative_simple_returns(strategy_aligned).rename("Strategy"),
                cumulative_simple_returns(equal_weight_aligned).rename("Equal-weight OSEBX"),
                cumulative_simple_returns(market_weight_aligned).rename("Market-weight OSEBX"),
            ],
            axis=1,
        )

        ax = chart.plot(figsize=(11, 5), linewidth=1.8)
        ax.set_title(key)
        ax.set_ylabel("Cumulative return")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

# ---------------------------------------------------------------------
# Diagnostics: stock contribution and holding periods
# ---------------------------------------------------------------------

def load_market_cap_for_diagnostics(data: dict[str, pd.DataFrame | pd.Series]) -> pd.DataFrame:
    """
    Loads market cap only for diagnostics.

    The strategy itself remains equal-weight. Market cap is used only to check
    whether return contributions are coming from small or large companies.
    """
    mcap = read_daily_csv(MCAP_CSV).apply(pd.to_numeric, errors="coerce")
    return mcap.reindex(index=data["asset_returns"].index, columns=data["asset_returns"].columns)


def get_hold_periods(position_mask: pd.DataFrame) -> pd.DataFrame:
    """
    Finds holding periods for each stock.

    position_mask:
        True if the stock is held on that date, False otherwise.

    Returns:
        One row per holding spell.
    """
    rows = []

    for ticker in position_mask.columns:
        held = position_mask[ticker].fillna(False).astype(bool)
        if held.sum() == 0:
            continue

        starts = held & ~held.shift(1, fill_value=False)
        groups = starts.cumsum()

        for group_id in groups[held].unique():
            dates = held.index[(groups == group_id) & held]
            if len(dates) == 0:
                continue

            rows.append(
                {
                    "ticker": ticker,
                    "start_date": dates[0],
                    "end_date": dates[-1],
                    "holding_days": len(dates),
                }
            )

    return pd.DataFrame(rows)


def stock_contribution_diagnostics(
    strategy_name: str,
    target_weights: pd.DataFrame,
    data: dict[str, pd.DataFrame | pd.Series],
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Explains which stocks drive strategy returns.

    total_excess_contribution is contribution relative to the equal-weight benchmark.
    It is not the same as CAPM alpha.
    """
    asset_returns = data["asset_returns"].reindex_like(target_weights)
    benchmark_weights = data["benchmark_target"].reindex_like(target_weights)

    target = target_weights.fillna(0.0)
    bench = benchmark_weights.fillna(0.0)
    rets = asset_returns.fillna(0.0)

    mcap = load_market_cap_for_diagnostics(data).loc[target.index]

    strategy_contrib_daily = target * rets
    benchmark_contrib_daily = bench * rets
    excess_contrib_daily = strategy_contrib_daily - benchmark_contrib_daily

    held = target > 1e-12
    hold_periods = get_hold_periods(held)

    if hold_periods.empty:
        hold_summary = pd.DataFrame(
            index=target.columns,
            data={
                "n_holding_periods": 0,
                "avg_holding_days": np.nan,
                "median_holding_days": np.nan,
                "max_holding_days": np.nan,
            },
        )
    else:
        hold_summary = (
            hold_periods.groupby("ticker")["holding_days"]
            .agg(
                n_holding_periods="count",
                avg_holding_days="mean",
                median_holding_days="median",
                max_holding_days="max",
            )
        )

    avg_mcap_when_held = mcap.where(held).mean()
    median_mcap_when_held = mcap.where(held).median()

    # 1.00 = among largest stocks in the index that day.
    # 0.00 = among smallest stocks in the index that day.
    mcap_rank_pct = mcap.rank(axis=1, pct=True)
    avg_mcap_rank_when_held = mcap_rank_pct.where(held).mean()

    stock_table = pd.DataFrame(
        {
            "strategy": strategy_name,
            "total_strategy_contribution": strategy_contrib_daily.sum(),
            "total_benchmark_contribution": benchmark_contrib_daily.sum(),
            "total_excess_contribution": excess_contrib_daily.sum(),
            "avg_strategy_weight": target.mean(),
            "avg_benchmark_weight": bench.mean(),
            "days_held": held.sum(),
            "share_of_days_held": held.mean(),
            "avg_mcap_when_held": avg_mcap_when_held,
            "median_mcap_when_held": median_mcap_when_held,
            "avg_mcap_rank_when_held": avg_mcap_rank_when_held,
        }
    )

    stock_table = stock_table.join(hold_summary, how="left")

    fill_cols = [
        "n_holding_periods",
        "avg_holding_days",
        "median_holding_days",
        "max_holding_days",
    ]
    stock_table[fill_cols] = stock_table[fill_cols].fillna(0)

    total_excess = float(stock_table["total_excess_contribution"].sum())
    if abs(total_excess) > 1e-12:
        stock_table["share_of_total_excess"] = stock_table["total_excess_contribution"] / total_excess
    else:
        stock_table["share_of_total_excess"] = np.nan

    stock_table = stock_table.sort_values("total_excess_contribution", ascending=False)
    top_excess = stock_table.head(top_n)

    return stock_table, top_excess, hold_periods


def summarize_strategy_holdings(strategy_name: str, stock_table: pd.DataFrame) -> pd.Series:
    held_stocks = stock_table[stock_table["days_held"] > 0].copy()

    if held_stocks.empty:
        return pd.Series(
            {
                "strategy": strategy_name,
                "n_stocks_ever_held": 0,
                "avg_holding_days_equal_weighted": np.nan,
                "avg_holding_days_weighted_by_days_held": np.nan,
                "avg_mcap_rank_when_held": np.nan,
                "avg_mcap_when_held": np.nan,
            }
        )

    return pd.Series(
        {
            "strategy": strategy_name,
            "n_stocks_ever_held": int(len(held_stocks)),
            "avg_holding_days_equal_weighted": held_stocks["avg_holding_days"].mean(),
            "avg_holding_days_weighted_by_days_held": np.average(
                held_stocks["avg_holding_days"],
                weights=held_stocks["days_held"],
            ),
            "avg_mcap_rank_when_held": np.average(
                held_stocks["avg_mcap_rank_when_held"].fillna(0),
                weights=held_stocks["days_held"],
            ),
            "avg_mcap_when_held": np.average(
                held_stocks["avg_mcap_when_held"].fillna(0),
                weights=held_stocks["days_held"],
            ),
        }
    )


def run_all_diagnostics(
    results: dict[str, dict],
    data: dict[str, pd.DataFrame | pd.Series],
    top_n: int = 20,
    strategy_names: list[str] | None = None,
) -> dict:
    """Runs contribution and holding-period diagnostics for selected strategies."""
    stock_tables = {}
    top_excess_tables = {}
    hold_period_tables = {}
    holding_summaries = []

    if strategy_names is None:
        selected_items = list(results.items())
    else:
        selected_items = [(name, results[name]) for name in strategy_names if name in results]

    for strategy_name, result in selected_items:
        target_weights = result["target_weights"]

        stock_table, top_excess, hold_periods = stock_contribution_diagnostics(
            strategy_name=strategy_name,
            target_weights=target_weights,
            data=data,
            top_n=top_n,
        )

        stock_tables[strategy_name] = stock_table
        top_excess_tables[strategy_name] = top_excess
        hold_period_tables[strategy_name] = hold_periods
        holding_summaries.append(summarize_strategy_holdings(strategy_name, stock_table))

    holding_summary_table = pd.DataFrame(holding_summaries)

    return {
        "stock_tables": stock_tables,
        "top_excess_tables": top_excess_tables,
        "hold_period_tables": hold_period_tables,
        "holding_summary_table": holding_summary_table,
    }


def print_diagnostics(diagnostics: dict, top_n: int = 10) -> None:
    holding_summary = diagnostics["holding_summary_table"]

    print("\nHolding summary by strategy:")
    print(holding_summary.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    print(f"\nTop {top_n} stock excess contributors by strategy:")
    for strategy_name, table in diagnostics["top_excess_tables"].items():
        cols = [
            "total_excess_contribution",
            "share_of_total_excess",
            "avg_strategy_weight",
            "days_held",
            "avg_holding_days",
            "avg_mcap_when_held",
            "avg_mcap_rank_when_held",
        ]

        show = table[cols].head(top_n).copy()

        print(f"\n{strategy_name}")
        print(show.to_string(float_format=lambda x: f"{x:,.6f}"))




def print_signal_health_diagnostics(
    results: dict[str, dict],
    data: dict[str, pd.DataFrame | pd.Series],
    cutoff: str = "2021-01-01",
) -> None:
    """Checks whether the MA50/200 strategy is actually invested after cutoff."""
    start = pd.Timestamp(cutoff)

    print(f"\nSignal/position health after {cutoff}:")
    print("=" * 80)
    print(f"Average index members: {data['membership'].loc[start:].sum(axis=1).mean():.2f}")
    print(f"Average valid return stocks: {data['valid_returns'].loc[start:].sum(axis=1).mean():.2f}")
    print(f"Average stocks with MA200: {data['ma200'].loc[start:].notna().sum(axis=1).mean():.2f}")

    for name in ["MA50_200_overlay", "MA50_200_selection"]:
        if name not in results:
            continue

        target = results[name]["target_weights"].loc[start:]
        if target.empty:
            continue

        invested = target.sum(axis=1)
        n_held = (target > 1e-12).sum(axis=1)

        print(f"\n{name}")
        print(f"Average invested weight: {invested.mean():.4f}")
        print(f"Median invested weight:  {invested.median():.4f}")
        print(f"Average number held:     {n_held.mean():.2f}")
        print(f"Median number held:      {n_held.median():.2f}")

def main() -> tuple[dict[str, dict], pd.DataFrame, dict]:
    data = prepare_data()

    strategy_specs = {
        "MA50_100": ma_crossover_signal(data["ma50"], data["ma100"]),
        "MA50_200": ma_crossover_signal(data["ma50"], data["ma200"]),
        "MA100_200": ma_crossover_signal(data["ma100"], data["ma200"]),
        "MA_score_binary": ma_score_binary_signal(data),
    }

    results: dict[str, dict] = {}

    for signal_name, signal in strategy_specs.items():
        for mode in ["overlay", "selection"]:
            key = f"{signal_name}_{mode}"
            results[key] = run_signal_strategy(
                strategy_name=key,
                signal=signal,
                data=data,
                mode=mode,
                fee=TRANSACTION_COST,
            )

    summary = results_table(results)

    diagnostics = run_all_diagnostics(
        results=results,
        data=data,
        top_n=20,
        strategy_names=["MA50_200_selection"],
    )

    return results, summary, diagnostics


if __name__ == "__main__":
    data = prepare_data()

    strategy_specs = {
        "MA50_100": ma_crossover_signal(data["ma50"], data["ma100"]),
        "MA50_200": ma_crossover_signal(data["ma50"], data["ma200"]),
        "MA100_200": ma_crossover_signal(data["ma100"], data["ma200"]),
        "MA_score_binary": ma_score_binary_signal(data),
    }

    results: dict[str, dict] = {}
    for signal_name, signal in strategy_specs.items():
        for mode in ["overlay", "selection"]:
            key = f"{signal_name}_{mode}"
            results[key] = run_signal_strategy(
                strategy_name=key,
                signal=signal,
                data=data,
                mode=mode,
                fee=TRANSACTION_COST,
            )

    summary = results_table(results)

    diagnostics = run_all_diagnostics(
        results=results,
        data=data,
        top_n=20,
        strategy_names=["MA50_200_selection"],
    )

    print("\nStrategy regression table:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))

    print_signal_health_diagnostics(results, data, cutoff="2021-01-01")

    market_weight_osebx, index_mcap, mcap = market_weighted_osebx_returns(data)
    plot_strategy_against_osebx(
        results=results,
        equal_weight_osebx=data["market_returns"],
        market_weight_osebx=market_weight_osebx,
    )

    print_diagnostics(diagnostics, top_n=10)
