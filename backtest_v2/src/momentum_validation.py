from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


RISK_FREE_ANNUAL = 0.03
RF_MONTHLY = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / 12.0) - 1.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def summarize_returns(r: pd.Series, periods_per_year: int = 12) -> pd.Series:
    r = pd.Series(r).dropna()
    equity = (1.0 + r).cumprod()
    years = len(r) / periods_per_year
    vol = r.std() * np.sqrt(periods_per_year)
    sharpe = (r.mean() / r.std()) * np.sqrt(periods_per_year) if r.std() > 0 else np.nan
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and len(equity) else np.nan
    return pd.Series(
        {
            "CAGR": cagr,
            "Vol": vol,
            "Sharpe": sharpe,
            "MaxDD": max_drawdown(equity) if len(equity) else np.nan,
            "Periods": len(r),
            "FinalEquity": equity.iloc[-1] if len(equity) else np.nan,
        }
    )


def expand_membership_to_daily(
    membership: pd.DataFrame, daily_index: pd.DatetimeIndex
) -> pd.DataFrame:
    return membership.sort_index().reindex(daily_index, method="ffill").fillna(0).astype(int)


def month_end_last_trading_dates(daily_index: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(daily_index, index=daily_index)
    return s.resample("ME").max().dropna()


def rank_top_tail(signal_xs: pd.DataFrame, q: float) -> pd.DataFrame:
    winners = pd.DataFrame(False, index=signal_xs.index, columns=signal_xs.columns)
    for date, row in signal_xs.iterrows():
        valid = row.dropna()
        n = len(valid)
        if n == 0:
            continue
        n_tail = max(1, int(np.floor(q * n)))
        ordered = valid.sort_values(kind="mergesort")
        winners.loc[date, ordered.index[-n_tail:]] = True
    return winners


def normalize_long_only(weights: pd.DataFrame) -> pd.DataFrame:
    denom = weights.sum(axis=1).replace(0, np.nan)
    return weights.div(denom, axis=0).fillna(0.0)


def winners_only_backtest(
    tr_daily: pd.DataFrame,
    membership_m: pd.DataFrame,
    J: int = 12,
    K: int = 12,
    q: float = 0.10,
    skip: int = 1,
    min_names: int = 10,
    require_member_entire_formation: bool = False,
    fill_missing_prices_while_held: bool = True,
) -> Dict[str, Any]:
    """Long-only overlapping buy-and-hold winner cohorts from Ws.ipynb."""
    tr_daily = tr_daily.sort_index()
    membership_m = membership_m.sort_index().reindex(columns=tr_daily.columns).fillna(0).astype(int)
    membership_d = expand_membership_to_daily(membership_m, tr_daily.index)

    tr_for_returns = tr_daily.ffill() if fill_missing_prices_while_held else tr_daily
    ret_d_raw = tr_for_returns.pct_change(fill_method=None)
    ret_d = ret_d_raw.fillna(0.0) if fill_missing_prices_while_held else ret_d_raw.copy()

    tr_m = tr_daily.resample("ME").last()
    member_m = membership_d.resample("ME").last().reindex(tr_m.index).fillna(0).astype(bool)

    signal = tr_m.shift(skip).pct_change(periods=J, fill_method=None)
    eligible = member_m & signal.notna()

    if require_member_entire_formation:
        formation_member = member_m.astype(int).shift(skip).rolling(J, min_periods=J).min().eq(1)
        eligible &= formation_member

    eligible = eligible.where(eligible.sum(axis=1).ge(min_names), False)
    winners = rank_top_tail(signal.where(eligible), q=q)
    formed_weights = normalize_long_only(winners.astype(float))

    last_trade_by_month = month_end_last_trading_dates(tr_daily.index)
    trading_days = pd.Series(tr_daily.index, index=tr_daily.index)
    entry_member = pd.DataFrame(False, index=formed_weights.index, columns=formed_weights.columns)
    entry_price_ok = pd.DataFrame(False, index=formed_weights.index, columns=formed_weights.columns)
    entry_possible = pd.Series(False, index=formed_weights.index)
    entry_dates = pd.Series(pd.NaT, index=formed_weights.index, dtype="datetime64[ns]")

    for month_end in formed_weights.index:
        if month_end not in last_trade_by_month.index:
            continue
        formation_trade_date = last_trade_by_month.loc[month_end]
        next_days = trading_days.loc[trading_days.index > formation_trade_date]
        if next_days.empty:
            continue
        entry_date = next_days.iloc[0]
        entry_possible.loc[month_end] = True
        entry_dates.loc[month_end] = entry_date
        entry_member.loc[month_end] = membership_d.loc[entry_date].astype(bool)
        entry_price_ok.loc[month_end] = tr_daily.loc[entry_date].notna() & ret_d.loc[entry_date].notna()

    opened_weights = normalize_long_only(formed_weights.where(entry_member & entry_price_ok, 0.0))

    dates = tr_daily.index
    date_pos = pd.Series(np.arange(len(dates)), index=dates)
    ret_arr = ret_d.to_numpy(dtype=float)
    position_sum_arr = np.zeros_like(ret_arr, dtype=float)
    cohort_ret_sum_arr = np.zeros(len(dates), dtype=float)
    active_cohorts_arr = np.zeros(len(dates), dtype=int)
    missing_return_cohorts_arr = np.zeros(len(dates), dtype=int)

    formation_months = list(opened_weights.index)
    for i, formation_month in enumerate(formation_months):
        entry_date = entry_dates.loc[formation_month]
        w0 = opened_weights.loc[formation_month]
        if pd.isna(entry_date) or np.isclose(w0.abs().sum(), 0.0):
            continue

        exit_date = entry_dates.iloc[i + K] if i + K < len(formation_months) else pd.NaT
        entry_trade_i = int(date_pos.loc[entry_date])
        first_return_i = entry_trade_i + 1
        exit_i = len(dates) if pd.isna(exit_date) else int(date_pos.loc[exit_date]) + 1
        if exit_i <= first_return_i:
            continue

        cohort_rets = ret_arr[first_return_i:exit_i]
        cohort_rets_for_drift = np.nan_to_num(cohort_rets, nan=0.0)
        w0_arr = w0.to_numpy(dtype=float)
        drift = np.cumprod(1.0 + cohort_rets_for_drift, axis=0)
        drift = np.vstack([np.ones((1, drift.shape[1])), drift[:-1]])
        cohort_positions = drift * w0_arr
        missing_held = np.isnan(cohort_rets) & (np.abs(cohort_positions) > 1e-12)
        cohort_daily_ret = np.sum(cohort_positions * cohort_rets_for_drift, axis=1)
        if not fill_missing_prices_while_held:
            cohort_daily_ret[missing_held.any(axis=1)] = np.nan

        active_slice = slice(first_return_i, exit_i)
        position_sum_arr[active_slice] += cohort_positions
        cohort_ret_sum_arr[active_slice] += np.nan_to_num(cohort_daily_ret, nan=0.0)
        active_cohorts_arr[active_slice] += 1
        missing_return_cohorts_arr[active_slice] += np.isnan(cohort_daily_ret).astype(int)

    active_cohorts = pd.Series(active_cohorts_arr, index=dates, dtype=int)
    position_denominator = np.where(active_cohorts_arr[:, None] > 0, active_cohorts_arr[:, None], np.nan)
    positions_d = pd.DataFrame(
        position_sum_arr / position_denominator, index=dates, columns=tr_daily.columns
    ).fillna(0.0)
    daily_ret_arr = np.divide(
        cohort_ret_sum_arr,
        active_cohorts_arr,
        out=np.zeros_like(cohort_ret_sum_arr),
        where=active_cohorts_arr > 0,
    )
    if not fill_missing_prices_while_held:
        daily_ret_arr[missing_return_cohorts_arr > 0] = np.nan
    daily_ret = pd.Series(daily_ret_arr, index=dates, name="strategy")

    active = active_cohorts.gt(0)
    daily_ret = daily_ret.loc[active.idxmax() :] if active.any() else daily_ret.iloc[0:0]
    monthly_ret = ((1.0 + daily_ret).resample("ME").prod() - 1.0).rename("strategy")
    if not fill_missing_prices_while_held:
        monthly_ret = monthly_ret.mask(daily_ret.isna().resample("ME").any())

    return {
        "daily_ret": daily_ret,
        "monthly_ret": monthly_ret,
        "daily_equity": (1.0 + daily_ret).cumprod(),
        "monthly_equity": (1.0 + monthly_ret).cumprod(),
        "daily_positions": positions_d,
        "daily_stock_returns": ret_d,
        "active_cohorts": active_cohorts,
    }


def market_returns_from_weights(
    tr_daily: pd.DataFrame,
    weights_wide: pd.DataFrame,
    on_index_m: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    stock_ret = tr_daily.ffill().pct_change(fill_method=None)
    weights = weights_wide.reindex(index=tr_daily.index, columns=tr_daily.columns).fillna(0.0)
    on_index_d = expand_membership_to_daily(on_index_m, tr_daily.index).reindex(
        columns=tr_daily.columns
    ).fillna(0).astype(bool)
    weights = weights.where(on_index_d, 0.0)
    weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    daily_market_ret = (weights.shift(1) * stock_ret).sum(axis=1, min_count=1).fillna(0.0)
    monthly_market_ret = ((1.0 + daily_market_ret).resample("ME").prod() - 1.0).rename("market")
    return daily_market_ret.rename("market"), monthly_market_ret


def _rf_series(
    index: pd.DatetimeIndex,
    rf: float | pd.Series = RF_MONTHLY,
) -> pd.Series:
    if isinstance(rf, pd.Series):
        return rf.reindex(index).astype(float).fillna(0.0).rename("rf")
    return pd.Series(float(rf), index=index, name="rf")


def test_strategy(
    strategy_returns: Sequence[float] | pd.Series,
    market_returns: Sequence[float] | pd.Series,
    dates: Optional[Sequence] = None,
    rf: float | pd.Series = RF_MONTHLY,
    nw_lags: Optional[int] = None,
    periods_per_year: int = 12,
    market_name: str = "Market",
    make_plots: bool = True,
) -> Dict[str, Any]:
    """Run CAPM alpha with HAC/Newey-West standard errors.

    The regression is:

        R_strategy - R_f = alpha + beta * (R_market - R_f) + epsilon

    Returns are expected to be simple periodic returns. For Ws strategies this
    should be monthly returns, with ``nw_lags=K-1``.
    """
    if len(market_returns) != len(strategy_returns):
        raise ValueError(
            f"market_returns (len={len(market_returns)}) must match "
            f"strategy_returns (len={len(strategy_returns)})"
        )

    n = len(strategy_returns)
    if dates is not None:
        idx = pd.DatetimeIndex(dates)
        if len(idx) != n:
            raise ValueError("Length of dates must match strategy_returns")
    elif isinstance(strategy_returns, pd.Series):
        idx = pd.DatetimeIndex(strategy_returns.index)
    elif isinstance(market_returns, pd.Series):
        idx = pd.DatetimeIndex(market_returns.index)
    else:
        idx = pd.RangeIndex(n)

    strat = pd.Series(np.asarray(strategy_returns, dtype=float), index=idx, name="strategy")
    mkt = pd.Series(np.asarray(market_returns, dtype=float), index=idx, name="market")
    data = pd.concat([strat, mkt], axis=1).dropna()
    if data.empty:
        raise ValueError("No overlapping non-NaN data available for the CAPM regression")

    rf_s = _rf_series(data.index, rf)
    excess_strat = (data["strategy"] - rf_s).rename("excess_strategy")
    excess_mkt = (data["market"] - rf_s).rename("excess_market")
    reg = pd.concat([excess_strat, excess_mkt], axis=1).dropna()
    if len(reg) < 3:
        raise ValueError("At least three observations are required for the CAPM regression")

    if nw_lags is None:
        nw_lags = int(np.floor(4.0 * (len(reg) / 100.0) ** (2.0 / 9.0)))
    nw_lags = int(max(0, min(nw_lags, len(reg) - 1)))

    X = sm.add_constant(reg["excess_market"])
    y = reg["excess_strategy"]
    ols_model = sm.OLS(y, X).fit()
    hac_model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

    alpha = float(hac_model.params["const"])
    beta = float(hac_model.params["excess_market"])
    alpha_se = float(hac_model.bse["const"])
    alpha_t = float(hac_model.tvalues["const"])

    summary = pd.Series(
        {
            "N periods": len(reg),
            "NW lags": nw_lags,
            "Alpha periodic": alpha,
            "Alpha annualized": alpha * periods_per_year,
            "Alpha HAC SE": alpha_se,
            "Alpha t-stat": alpha_t,
            "p-value alpha != 0": float(hac_model.pvalues["const"]),
            "p-value alpha > 0": float(hac_model.pvalues["const"] / 2.0)
            if alpha_t > 0
            else float(1.0 - hac_model.pvalues["const"] / 2.0),
            "Market beta": beta,
            "R2": float(ols_model.rsquared),
        }
    )

    figures = []
    if make_plots:
        cumulative_strat = (1.0 + data["strategy"]).cumprod() - 1.0
        cumulative_mkt = (1.0 + data["market"]).cumprod() - 1.0
        cumulative_rf = (1.0 + rf_s).cumprod() - 1.0

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(cumulative_mkt.index, cumulative_mkt * 100.0, label=market_name, color="black", linewidth=2)
        ax1.plot(cumulative_strat.index, cumulative_strat * 100.0, label="Strategy", color="tab:blue", linewidth=2)
        ax1.set_ylabel("Cumulative Return (%)")
        ax1.set_title(f"Cumulative Returns: Strategy vs {market_name}")
        ax1.legend()
        ax1.grid(alpha=0.3)

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        cum_excess_strat = (1.0 + excess_strat).cumprod() - 1.0
        cum_excess_mkt = (1.0 + excess_mkt).cumprod() - 1.0
        ax2.plot(cum_excess_strat.index, cum_excess_strat * 100.0, label="Strategy (excess)", color="tab:blue", linewidth=2)
        ax2.plot(cum_excess_mkt.index, cum_excess_mkt * 100.0, label=f"{market_name} (excess)", color="black", linewidth=2, linestyle="--")
        ax2.plot(cumulative_rf.index, cumulative_rf * 100.0, label="Risk-free (cum)", color="gray", linewidth=1, linestyle=":")
        ax2.set_ylabel("Cumulative Excess Return (%)")
        ax2.set_title("Cumulative Excess Returns over Risk-Free Rate")
        ax2.legend()
        ax2.grid(alpha=0.3)
        figures = [fig1, fig2]

    return {
        "model": hac_model,
        "ols_model": ols_model,
        "summary": summary,
        "series": {
            "strategy": strat,
            "market": mkt,
            "rf": _rf_series(strat.index, rf),
            "excess_strategy": excess_strat,
            "excess_market": excess_mkt,
        },
        "figures": figures,
    }


def load_on_index_membership(
    data_dir: str | Path = DATA_DIR,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    root = Path(data_dir)
    on_index_path = root / "ON_INDEX.csv"
    mem_raw = pd.read_csv(on_index_path, sep=";", engine="python")
    mem_raw = mem_raw.rename(columns={mem_raw.columns[0]: "date"})
    mem_raw = mem_raw.loc[~mem_raw["date"].astype(str).str.fullmatch("DATES", case=False, na=False)]
    mem_raw["date"] = pd.to_datetime(
        mem_raw["date"], format="%d.%m.%Y", dayfirst=True, errors="coerce"
    )
    mem_raw = mem_raw.dropna(subset=["date"]).set_index("date").sort_index()

    mask_id_row = mem_raw.apply(
        lambda row: row.astype(str).str.contains(r"id\(\)", case=False, na=False)
    ).any(axis=1)
    mem_raw = mem_raw.loc[~mask_id_row]

    mem_str = mem_raw.astype("string")
    on_index_m = (mem_str.notna() & (mem_str != "")).astype(int)
    if columns is not None:
        on_index_m = on_index_m.reindex(columns=columns).fillna(0).astype(int)
    return on_index_m


def load_ws_inputs(
    data_dir: str | Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir)
    tr_daily = pd.read_csv(
        root / "TOTRET_DAILY.csv",
        sep=";",
        decimal=",",
        parse_dates=["Date"],
        dayfirst=True,
        na_values=["#N/A"],
    )
    tr_daily = tr_daily.set_index("Date").sort_index()
    tr_daily = tr_daily.loc[:, tr_daily.notna().any(axis=0)]

    membership_m = pd.read_csv(root / "membership_m.csv", parse_dates=["date"])
    membership_m = membership_m.set_index("date").sort_index()
    membership_m = membership_m.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    membership_m = membership_m.reindex(columns=tr_daily.columns).fillna(0).astype(int)

    weights_wide = pd.read_csv(root / "weights_wide.csv", index_col=0, parse_dates=True)
    weights_wide = weights_wide.reindex(columns=tr_daily.columns).fillna(0.0)
    on_index_m = load_on_index_membership(root, columns=tr_daily.columns)
    return tr_daily, membership_m, weights_wide, on_index_m


def test_ws_strategies(
    data_dir: str | Path = DATA_DIR,
    Js: Sequence[int] = (3, 6, 9, 12),
    Ks: Sequence[int] = (1, 3, 6, 9, 12),
    qs: Sequence[float] = (0.05, 0.10, 0.20),
    skip: int = 1,
    min_names: int = 10,
    rf_monthly: float | pd.Series = RF_MONTHLY,
) -> pd.DataFrame:
    """Test all Ws winners-only strategies with monthly HAC/Newey-West CAPM."""
    tr_daily, membership_m, weights_wide, on_index_m = load_ws_inputs(data_dir)
    _, market_monthly_ret = market_returns_from_weights(tr_daily, weights_wide, on_index_m)

    rows = []
    for J in Js:
        for K in Ks:
            for q in qs:
                bt = winners_only_backtest(
                    tr_daily=tr_daily,
                    membership_m=membership_m,
                    J=J,
                    K=K,
                    q=q,
                    skip=skip,
                    min_names=min_names,
                )
                perf = summarize_returns(bt["monthly_ret"], periods_per_year=12)
                capm = test_strategy(
                    strategy_returns=bt["monthly_ret"],
                    market_returns=market_monthly_ret.reindex(bt["monthly_ret"].index),
                    rf=rf_monthly,
                    nw_lags=max(0, K - 1),
                    periods_per_year=12,
                    make_plots=False,
                )["summary"]
                rows.append(
                    {
                        "J": J,
                        "K": K,
                        "q": q,
                        "skip": skip,
                        **perf.to_dict(),
                        **capm.to_dict(),
                    }
                )

    return pd.DataFrame(rows).sort_values("p-value alpha > 0").reset_index(drop=True)


if __name__ == "__main__":
    results = test_ws_strategies()
    pd.set_option("display.float_format", lambda x: f"{x:,.6f}")
    print(results)
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "ws_hac_newey_west_results.csv", index=False)
