import math
import warnings

import numpy as np
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive.*", category=UserWarning)
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)

tr_daily = pd.read_csv(
    Path(__file__).resolve().parents[1] / "data" / "TOTRET_DAILY.csv",
    sep=";",
    decimal=",",
    parse_dates=["Date"],
    dayfirst=True,
    na_values=["#N/A"],
)
tr_daily = tr_daily.set_index("Date").sort_index()

# TOTRET_DAILY is already a wide Date x ticker total-return panel.
# Only total-return levels are loaded for this strategy.

# ## Methodology notes
#
# This notebook uses an overlapping buy-and-hold cohort construction. Each month forms a winner/loser basket from OSEBX members, checks that selected names are still eligible on the next trading day, trades at that next trading day's close, and then starts earning returns from the following close-to-close return. Each cohort's stock weights drift until its K-month exit. This is different from averaging fixed target weights every month.
#
# The strategy return is reported on a long/short capital convention: each opened cohort starts near +1 gross long and -1 gross short exposure, so gross exposure is about 2 before buy-and-hold drift. The parameter grid is in-sample exploration, not out-of-sample evidence. Forward-filled total-return levels are used by default for continuous mark-to-market; set `fill_missing_prices_while_held=False` for a stricter missing-data sensitivity check that propagates missing held returns to NaN daily strategy returns.
#

# OSEBX membership universe, aligned to the total-return panel.
tr_daily = tr_daily.sort_index()
tr_daily = tr_daily.loc[:, tr_daily.notna().any(axis=0)]

membership_m = pd.read_csv(
    Path(__file__).resolve().parents[1] / "data" / "membership_m.csv",
    parse_dates=["date"],
)
membership_m = membership_m.set_index("date").sort_index()
membership_m = membership_m.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
membership_m = membership_m.reindex(columns=tr_daily.columns).fillna(0).astype(int)

print("TR daily:", tr_daily.shape, tr_daily.index.min().date(), "to", tr_daily.index.max().date())
print("Membership monthly:", membership_m.shape, membership_m.index.min().date(), "to", membership_m.index.max().date())
print("Average OSEBX members:", round(membership_m.sum(axis=1).mean(), 1))

def expand_membership_to_daily(membership: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill dated OSEBX membership observations to trading days."""
    return membership.sort_index().reindex(daily_index, method="ffill").fillna(0).astype(int)


def month_end_last_trading_dates(daily_index: pd.DatetimeIndex) -> pd.Series:
    """Map each month-end label to the last observed trading date in that month."""
    s = pd.Series(daily_index, index=daily_index)
    return s.resample("ME").max().dropna()


def normalize_long_short(weights: pd.DataFrame) -> pd.DataFrame:
    """Normalize each row to +1 gross long and -1 gross short when possible."""
    long_side = weights.clip(lower=0)
    short_side = weights.clip(upper=0)

    long_sum = long_side.sum(axis=1).replace(0, np.nan)
    short_sum = short_side.abs().sum(axis=1).replace(0, np.nan)

    long_norm = long_side.div(long_sum, axis=0).fillna(0.0)
    short_norm = short_side.div(short_sum, axis=0).fillna(0.0)
    return long_norm + short_norm


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def summarize_returns(r: pd.Series, periods_per_year: int = 252) -> pd.Series:
    r = r.dropna()
    equity = (1 + r).cumprod()
    years = len(r) / periods_per_year
    vol = r.std() * np.sqrt(periods_per_year)
    sharpe = (r.mean() / r.std()) * np.sqrt(periods_per_year) if r.std() > 0 else np.nan
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 and len(equity) else np.nan
    return pd.Series({
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "MaxDD": max_drawdown(equity) if len(equity) else np.nan,
        "Periods": len(r),
        "FinalEquity": equity.iloc[-1] if len(equity) else np.nan,
    })

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
) -> dict:
    """
    Classic overlapping winners/losers cohort backtest with OSEBX entry constraints.

    No-lookahead rules:
    - Signal at formation month t uses total-return levels only through t-skip.
    - Stocks are ranked only if they are OSEBX members at formation month t.
    - With close-only data, a position is opened at the close of the first trading
      day after formation. The first earned close-to-close return is the next row.
    - A position is opened only if the stock is still an OSEBX member and has a
      usable total-return observation on the entry trading day.
    - Once opened, a cohort is buy-and-held until the entry date of the cohort
      formed K months later. Stock weights drift with realized returns.
    - Later OSEBX exits do not force early liquidation.

    Return convention:
    - Each opened cohort starts with +1 gross long and -1 gross short exposure.
    - The reported daily strategy return is the average PnL contribution of active
      cohorts, i.e. a zero-net, roughly 200% gross long/short book before drift.

    Missing-price convention:
    - If fill_missing_prices_while_held=True, total-return levels are forward-filled
      for mark-to-market continuity. This treats missing quotes as stale prices.
      Set it False to inspect sensitivity, but expect gaps/NaNs when held names
      have missing daily returns.
    """
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
        formation_member = (
            member_m.astype(int)
            .shift(skip)
            .rolling(J, min_periods=J)
            .min()
            .eq(1)
        )
        eligible &= formation_member

    eligible = eligible.where(eligible.sum(axis=1).ge(min_names), False)
    signal_xs = signal.where(eligible)

    winners, losers = rank_tail_portfolios(signal_xs, q=q)
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

    entry_ok = entry_member & entry_price_ok
    opened_weights = formed_weights.where(entry_ok, 0.0)
    if renormalize_after_entry_filter:
        opened_weights = normalize_long_short(opened_weights)

    dates = tr_daily.index
    columns = tr_daily.columns
    date_pos = pd.Series(np.arange(len(dates)), index=dates)
    ret_arr = ret_d.to_numpy(dtype=float)
    position_sum_arr = np.zeros_like(ret_arr, dtype=float)
    cohort_ret_sum_arr = np.zeros(len(dates), dtype=float)
    active_cohorts_arr = np.zeros(len(dates), dtype=int)
    missing_return_cohorts_arr = np.zeros(len(dates), dtype=int)
    cohort_rows = []

    formation_months = list(opened_weights.index)
    for i, formation_month in enumerate(formation_months):
        entry_date = entry_dates.loc[formation_month]
        w0 = opened_weights.loc[formation_month]
        if pd.isna(entry_date) or np.isclose(w0.abs().sum(), 0.0):
            continue

        if i + K < len(formation_months):
            exit_date = entry_dates.iloc[i + K]
        else:
            exit_date = pd.NaT

        entry_trade_i = int(date_pos.loc[entry_date])
        first_return_i = entry_trade_i + 1
        exit_i = len(dates) if pd.isna(exit_date) else int(date_pos.loc[exit_date]) + 1
        if exit_i <= first_return_i:
            continue

        cohort_rets = ret_arr[first_return_i:exit_i]
        cohort_rets_for_drift = np.nan_to_num(cohort_rets, nan=0.0)
        w0_arr = w0.to_numpy(dtype=float)
        # Start-of-day buy-and-hold exposures: initial weights drift with prior returns.
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

        cohort_rows.append({
            "formation_date": formation_month,
            "entry_date": entry_date,
            "first_return_date": dates[first_return_i],
            "exit_date": exit_date,
            "holding_days": exit_i - first_return_i,
            "winners": int((w0 > 0).sum()),
            "losers": int((w0 < 0).sum()),
            "initial_gross": float(w0.abs().sum()),
        })

    active_cohorts = pd.Series(active_cohorts_arr, index=dates, dtype=int)
    position_denominator = np.where(active_cohorts_arr[:, None] > 0, active_cohorts_arr[:, None], np.nan)
    positions_d = pd.DataFrame(position_sum_arr / position_denominator, index=dates, columns=columns).fillna(0.0)
    daily_ret_arr = np.divide(
        cohort_ret_sum_arr,
        active_cohorts_arr,
        out=np.zeros_like(cohort_ret_sum_arr),
        where=active_cohorts_arr > 0,
    )
    if not fill_missing_prices_while_held:
        daily_ret_arr[missing_return_cohorts_arr > 0] = np.nan
    daily_ret = pd.Series(daily_ret_arr, index=dates)

    active = active_cohorts.gt(0)
    daily_ret = daily_ret.loc[active.idxmax():] if active.any() else daily_ret.iloc[0:0]
    monthly_ret = (1 + daily_ret).resample("ME").prod() - 1
    if not fill_missing_prices_while_held:
        monthly_has_missing = daily_ret.isna().resample("ME").any()
        monthly_ret = monthly_ret.mask(monthly_has_missing)

    missing_held_return = positions_d.ne(0) & ret_d_raw.isna()
    diagnostics = pd.DataFrame({
        "eligible": eligible.sum(axis=1),
        "winners": winners.sum(axis=1),
        "losers": losers.sum(axis=1),
        "members": member_m.sum(axis=1),
        "gross_formed": formed_weights.abs().sum(axis=1),
        "gross_opened": opened_weights.abs().sum(axis=1),
        "entry_date": entry_dates,
        "entry_rejected_no_next_day": formed_weights.where(~entry_possible, 0.0).abs().sum(axis=1),
        "entry_rejected_not_member": formed_weights.where(entry_possible, 0.0).where(~entry_member, 0.0).abs().sum(axis=1),
        "entry_rejected_no_price": formed_weights.where(entry_member & ~entry_price_ok, 0.0).abs().sum(axis=1),
        "days_with_missing_held_returns": missing_held_return.any(axis=1).resample("ME").sum().reindex(formed_weights.index),
    })

    return {
        "daily_ret": daily_ret,
        "monthly_ret": monthly_ret,
        "daily_equity": (1 + daily_ret).cumprod(),
        "monthly_equity": (1 + monthly_ret).cumprod(),
        "formed_weights": formed_weights,
        "opened_weights": opened_weights,
        "entry_member": entry_member,
        "entry_price_ok": entry_price_ok,
        "entry_possible": entry_possible,
        "cohorts": pd.DataFrame(cohort_rows),
        "active_cohorts": active_cohorts,
        "daily_positions": positions_d,
        "daily_stock_returns": ret_d,
        "signal": signal,
        "membership_daily": membership_d,
        "diagnostics": diagnostics,
    }

bt = winners_losers_ls_backtest(
    tr_daily=tr_daily,
    membership_m=membership_m,
    J=6,
    K=6,
    q=0.10,
    skip=1,
    min_names=10,
    require_member_entire_formation=False,
    renormalize_after_entry_filter=True,
)

print("Daily performance")
display(summarize_returns(bt["daily_ret"], periods_per_year=252).to_frame("J6_K6_q10_skip1"))

print("Monthly performance")
display(summarize_returns(bt["monthly_ret"], periods_per_year=12).to_frame("J6_K6_q10_skip1"))

display(bt["diagnostics"].tail())

fig, ax = plt.subplots(figsize=(11, 5))
bt["daily_equity"].plot(ax=ax, lw=1.8)
ax.set_title("OSEBX Membership-Constrained Winners/Losers Long/Short")
ax.set_ylabel("Equity, start = 1")
ax.grid(True, alpha=0.25)
plt.show()

def run_parameter_grid(
    tr_daily: pd.DataFrame,
    membership_m: pd.DataFrame,
    Js=(3, 6, 12),
    Ks=(1, 3, 6, 12),
    qs=(0.05, 0.10, 0.20),
    skip: int = 1,
    min_names: int = 10,
) -> pd.DataFrame:
    rows = []
    for J in Js:
        for K in Ks:
            for q in qs:
                bt_i = winners_losers_ls_backtest(
                    tr_daily=tr_daily,
                    membership_m=membership_m,
                    J=J,
                    K=K,
                    q=q,
                    skip=skip,
                    min_names=min_names,
                )
                stats = summarize_returns(bt_i["monthly_ret"], periods_per_year=12)
                rows.append({"J": J, "K": K, "q": q, "skip": skip, **stats.to_dict()})
    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)

# Uncomment to search parameters.
# grid = run_parameter_grid(tr_daily, membership_m)
# display(grid.head(15))

# Sanity checks: entries obey OSEBX membership and price availability; the equity curve has no large date gaps.
formed = bt["formed_weights"]
opened = bt["opened_weights"]
entry_member = bt["entry_member"].astype(bool)
entry_price_ok = bt["entry_price_ok"].astype(bool)
positions = bt["daily_positions"]
membership_d = bt["membership_daily"].astype(bool)

blocked_entries = opened.where(~entry_member).abs().sum().sum()
opened_without_price = opened.where(~entry_price_ok).abs().sum().sum()
positions_after_later_exit = positions.where(~membership_d).abs().sum().sum()
first_position_date = positions.abs().sum(axis=1).gt(0).idxmax()
first_signal_date = bt["signal"].notna().any(axis=1).idxmax()
max_equity_gap_days = bt["daily_equity"].index.to_series().diff().dt.days.max()

print("Absolute entry exposure opened outside OSEBX membership:", blocked_entries)
print("Absolute entry exposure opened without an entry price:", opened_without_price)
print("Absolute ongoing exposure after later OSEBX exits:", positions_after_later_exit)
print("First usable signal month-end:", first_signal_date.date())
print("First day with live positions:", first_position_date.date())
print("Largest gap in plotted daily equity, days:", max_equity_gap_days)
assert np.isclose(blocked_entries, 0.0)
assert np.isclose(opened_without_price, 0.0)
assert first_position_date > first_signal_date
assert max_equity_gap_days <= 7

# In-sample parameter exploration. Do not treat the best row as out-of-sample evidence.
grid = run_parameter_grid(
    tr_daily=tr_daily,
    membership_m=membership_m,
    Js=(3, 6, 9, 12),
    Ks=(1, 3, 6, 9, 12),
    qs=(0.05, 0.10, 0.20),
    skip=1,
    min_names=10,
)

best = grid.iloc[0]
best_J = int(best["J"])
best_K = int(best["K"])
best_q = float(best["q"])
best_skip = int(best["skip"])

print("Best parameters by monthly Sharpe, selected in sample")
display(best.to_frame("best_in_sample"))

bt_best = winners_losers_ls_backtest(
    tr_daily=tr_daily,
    membership_m=membership_m,
    J=best_J,
    K=best_K,
    q=best_q,
    skip=best_skip,
    min_names=10,
    require_member_entire_formation=False,
    renormalize_after_entry_filter=True,
)

perf_best = pd.concat(
    [
        summarize_returns(bt_best["daily_ret"], periods_per_year=252).rename("Daily"),
        summarize_returns(bt_best["monthly_ret"], periods_per_year=12).rename("Monthly"),
    ],
    axis=1,
)

print("Performance with in-sample best parameters")
display(perf_best)

fig, ax = plt.subplots(figsize=(11, 5))
bt_best["daily_equity"].plot(ax=ax, lw=1.8)
ax.set_title(f"In-Sample Best OSEBX W-L Cohort LS: J={best_J}, K={best_K}, q={best_q:.0%}, skip={best_skip}")
ax.set_ylabel("Equity on long/short capital convention, start = 1")
ax.grid(True, alpha=0.25)
plt.show()

# Significance test for the W-L momentum spread: H0 E[r_WL] = 0.
def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))


def newey_west_mean_test(r: pd.Series, lags: int | None = None, annualization: int = 12) -> pd.Series:
    """
    Newey-West/HAC test for whether the monthly W-L mean return is zero.

    This tests the first-pass momentum hypothesis H0: E[r_WL] = 0. P-values use
    the asymptotic normal reference. For overlapping K-month cohorts, using K-1
    lags is the standard simple choice.
    """
    x = pd.Series(r).dropna().astype(float)
    n = len(x)
    if n < 3:
        return pd.Series(dtype=float)

    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    lags = int(max(0, min(lags, n - 1)))

    demeaned = x - x.mean()
    gamma0 = float((demeaned @ demeaned) / n)
    long_run_var = gamma0
    for ell in range(1, lags + 1):
        weight = 1.0 - ell / (lags + 1)
        gamma = float((demeaned.iloc[ell:].to_numpy() @ demeaned.iloc[:-ell].to_numpy()) / n)
        long_run_var += 2.0 * weight * gamma

    se_mean = np.sqrt(max(long_run_var, 0.0) / n)
    t_stat = x.mean() / se_mean if se_mean > 0 else np.nan
    p_two_sided = 2.0 * (1.0 - normal_cdf(abs(t_stat))) if np.isfinite(t_stat) else np.nan
    p_one_sided_positive = 1.0 - normal_cdf(t_stat) if np.isfinite(t_stat) else np.nan

    return pd.Series({
        "N months": n,
        "NW lags": lags,
        "Mean monthly W-L": x.mean(),
        "Annualized mean": x.mean() * annualization,
        "NW SE monthly mean": se_mean,
        "t-stat H0 mean=0": t_stat,
        "p-value two-sided": p_two_sided,
        "p-value one-sided >0": p_one_sided_positive,
    })


def run_significance_grid(
    tr_daily: pd.DataFrame,
    membership_m: pd.DataFrame,
    Js=(3, 6, 9, 12),
    Ks=(1, 3, 6, 9, 12),
    qs=(0.05, 0.10, 0.20),
    skip: int = 1,
    min_names: int = 10,
) -> pd.DataFrame:
    rows = []
    for J in Js:
        for K in Ks:
            for q in qs:
                bt_i = winners_losers_ls_backtest(
                    tr_daily=tr_daily,
                    membership_m=membership_m,
                    J=J,
                    K=K,
                    q=q,
                    skip=skip,
                    min_names=min_names,
                )
                test = newey_west_mean_test(bt_i["monthly_ret"], lags=max(0, K - 1))
                stats = summarize_returns(bt_i["monthly_ret"], periods_per_year=12)
                rows.append({
                    "J": J,
                    "K": K,
                    "q": q,
                    "skip": skip,
                    "Monthly CAGR": stats["CAGR"],
                    "Monthly Sharpe": stats["Sharpe"],
                    "Final equity": stats["FinalEquity"],
                    **test.to_dict(),
                })
    return pd.DataFrame(rows).sort_values("p-value one-sided >0").reset_index(drop=True)


baseline_lags = max(0, 12 - 1)
best_lags = max(0, best_K - 1)

momentum_tests = pd.concat(
    {
        "J6_K6_q10_baseline": newey_west_mean_test(bt["monthly_ret"], lags=baseline_lags),
        "In_sample_best": newey_west_mean_test(bt_best["monthly_ret"], lags=best_lags),
    },
    axis=1,
)

print("Newey-West tests of H0: monthly W-L mean return = 0")
display(momentum_tests)

print("All grid W-L spreads and Newey-West p-values")
wl_significance_grid = run_significance_grid(
    tr_daily=tr_daily,
    membership_m=membership_m,
    Js=(3, 6, 9, 12),
    Ks=(1, 3, 6, 9, 12),
    qs=(0.05, 0.10, 0.20),
    skip=1,
    min_names=10,
)
display(wl_significance_grid)

print("Interpretation: the one-sided p-value tests E[r_WL] > 0. The grid is in-sample exploration, so small p-values there are descriptive, not out-of-sample evidence.")

# Display winner and loser cohort legs separately.
def leg_returns_from_backtest(bt_result: dict) -> pd.DataFrame:
    positions = bt_result["daily_positions"]
    stock_ret = bt_result["daily_stock_returns"].reindex_like(positions).fillna(0.0)

    winner_weights = positions.clip(lower=0)
    loser_short_weights = positions.clip(upper=0)
    loser_long_weights = -loser_short_weights

    winner_ret = (winner_weights * stock_ret).sum(axis=1, min_count=1).fillna(0.0)
    loser_short_ret = (loser_short_weights * stock_ret).sum(axis=1, min_count=1).fillna(0.0)
    loser_long_ret = (loser_long_weights * stock_ret).sum(axis=1, min_count=1).fillna(0.0)

    legs = pd.DataFrame(
        {
            "Winners long": winner_ret,
            "Losers long": loser_long_ret,
            "Losers short": loser_short_ret,
            "Winners plus losers short": winner_ret + loser_short_ret,
        }
    )
    active = positions.abs().sum(axis=1).gt(0)
    return legs.loc[active.idxmax():]


def summarize_leg_contributions(legs: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    rows = []
    for col in legs.columns:
        r = legs[col].dropna()
        vol = r.std() * np.sqrt(periods_per_year)
        rows.append(
            {
                "Annualized mean": r.mean() * periods_per_year,
                "Vol": vol,
                "Sharpe": (r.mean() / r.std()) * np.sqrt(periods_per_year) if r.std() > 0 else np.nan,
                "Cumulative PnL contribution": r.sum(),
                "Periods": len(r),
            }
        )
    return pd.DataFrame(rows, index=legs.columns).T


legs_best = leg_returns_from_backtest(bt_best)
winner_equity = (1 + legs_best["Winners long"]).cumprod()
strategy_equity_from_legs = (1 + legs_best["Winners plus losers short"]).cumprod()
loser_pnl_curves = 1 + legs_best[["Losers long", "Losers short"]].cumsum()

print("Leg attribution, in-sample best cohort strategy")
display(summarize_leg_contributions(legs_best))

fig, ax = plt.subplots(figsize=(11, 5))
winner_equity.plot(ax=ax, lw=1.8, color="tab:green", label="Winners long compounded NAV")
ax.set_title(f"Winners Long Portfolio: J={best_J}, K={best_K}, q={best_q:.0%}, skip={best_skip}")
ax.set_ylabel("Compounded equity, start = 1")
ax.grid(True, alpha=0.25)
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(11, 5))
loser_pnl_curves.plot(ax=ax, lw=1.8)
ax.axhline(1.0, color="black", lw=1, alpha=0.4)
ax.set_title(f"Losers Basket PnL Contribution: J={best_J}, K={best_K}, q={best_q:.0%}, skip={best_skip}")
ax.set_ylabel("1 + cumulative daily PnL contribution")
ax.grid(True, alpha=0.25)
plt.show()

fig, ax = plt.subplots(figsize=(11, 5))
strategy_equity_from_legs.plot(ax=ax, lw=1.8, color="tab:blue", label="Winners plus losers short")
ax.set_title("Combined Winner + Short-Loser Leg")
ax.set_ylabel("Compounded equity, start = 1")
ax.grid(True, alpha=0.25)
ax.legend()
plt.show()
# display(losers_2024)
