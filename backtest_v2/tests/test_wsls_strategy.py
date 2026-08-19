from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from backtest_v2.src.momentum_validation import (
    DATA_DIR,
    RESULTS_DIR,
    RF_MONTHLY,
    expand_membership_to_daily,
    load_ws_inputs,
    market_returns_from_weights,
    month_end_last_trading_dates,
    summarize_returns,
    test_strategy as evaluate_strategy,
)


def normalize_long_short(weights: pd.DataFrame) -> pd.DataFrame:
    """Normalize each row to +1 gross long and -1 gross short when possible."""
    long_side = weights.clip(lower=0)
    short_side = weights.clip(upper=0)

    long_sum = long_side.sum(axis=1).replace(0, np.nan)
    short_sum = short_side.abs().sum(axis=1).replace(0, np.nan)

    long_norm = long_side.div(long_sum, axis=0).fillna(0.0)
    short_norm = short_side.div(short_sum, axis=0).fillna(0.0)
    return long_norm + short_norm


def rank_tail_portfolios(signal_xs: pd.DataFrame, q: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select bottom/top tails by rank so winner and loser sets cannot overlap on ties."""
    winners = pd.DataFrame(False, index=signal_xs.index, columns=signal_xs.columns)
    losers = pd.DataFrame(False, index=signal_xs.index, columns=signal_xs.columns)

    for date, row in signal_xs.iterrows():
        valid = row.dropna()
        n = len(valid)
        if n == 0:
            continue
        n_tail = max(1, int(np.floor(q * n)))
        n_tail = min(n_tail, n // 2)
        if n_tail == 0:
            continue

        ordered = valid.sort_values(kind="mergesort")
        losers.loc[date, ordered.index[:n_tail]] = True
        winners.loc[date, ordered.index[-n_tail:]] = True

    return winners, losers


def winners_losers_ls_backtest(
    tr_daily: pd.DataFrame,
    membership_m: pd.DataFrame,
    J: int = 12,
    K: int = 12,
    q: float = 0.10,
    skip: int = 1,
    min_names: int = 10,
    require_member_entire_formation: bool = False,
    renormalize_after_entry_filter: bool = True,
    fill_missing_prices_while_held: bool = True,
) -> Dict[str, Any]:
    """WsLs_new membership-constrained overlapping winners/losers strategy."""
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
    winners, losers = rank_tail_portfolios(signal.where(eligible), q=q)
    formed_weights = normalize_long_short(winners.astype(float) - losers.astype(float))

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

    opened_weights = formed_weights.where(entry_member & entry_price_ok, 0.0)
    if renormalize_after_entry_filter:
        opened_weights = normalize_long_short(opened_weights)

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


def test_wsls_new_strategies(
    data_dir: str | Path = DATA_DIR,
    Js: Sequence[int] = (3, 6, 9, 12),
    Ks: Sequence[int] = (1, 3, 6, 9, 12),
    qs: Sequence[float] = (0.05, 0.10, 0.20),
    skip: int = 1,
    min_names: int = 10,
    rf_monthly: float | pd.Series = RF_MONTHLY,
) -> pd.DataFrame:
    """Test all WsLs_new strategies with monthly HAC/Newey-West CAPM."""
    tr_daily, membership_m, weights_wide, on_index_m = load_ws_inputs(data_dir)
    _, market_monthly_ret = market_returns_from_weights(tr_daily, weights_wide, on_index_m)

    rows = []
    for J in Js:
        for K in Ks:
            for q in qs:
                bt = winners_losers_ls_backtest(
                    tr_daily=tr_daily,
                    membership_m=membership_m,
                    J=J,
                    K=K,
                    q=q,
                    skip=skip,
                    min_names=min_names,
                    require_member_entire_formation=False,
                    renormalize_after_entry_filter=True,
                )
                perf = summarize_returns(bt["monthly_ret"], periods_per_year=12)
                capm = evaluate_strategy(
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
    results = test_wsls_new_strategies()
    pd.set_option("display.float_format", lambda x: f"{x:,.6f}")
    print(results)
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "wsls_new_hac_newey_west_results.csv", index=False)
