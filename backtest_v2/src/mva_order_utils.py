from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


RISK_FREE_ANNUAL = 0.03
RF_MONTHLY = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / 12.0) - 1.0


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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_on_index_membership(
    data_dir: str | Path = DATA_DIR,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    root = Path(data_dir)
    on_index_path = root / "ON_INDEX.csv"
    mem_raw = pd.read_csv(on_index_path, sep=";", engine="python")
    mem_raw = mem_raw.rename(columns={mem_raw.columns[0]: "date"})
    mem_raw = mem_raw.loc[
        ~mem_raw["date"].astype(str).str.fullmatch("DATES", case=False, na=False)
    ]
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


def _rf_series(index: pd.DatetimeIndex, rf: float | pd.Series = RF_MONTHLY) -> pd.Series:
    if isinstance(rf, pd.Series):
        return rf.reindex(index).astype(float).fillna(0.0).rename("rf")
    return pd.Series(float(rf), index=index, name="rf")


def capm_alpha_summary(
    strategy_returns: Sequence[float] | pd.Series,
    market_returns: Sequence[float] | pd.Series,
    rf: float | pd.Series = RF_MONTHLY,
    nw_lags: Optional[int] = None,
    periods_per_year: int = 12,
) -> pd.Series:
    strat = pd.Series(strategy_returns, dtype=float)
    mkt = pd.Series(market_returns, dtype=float)

    if isinstance(strategy_returns, pd.Series):
        idx = pd.DatetimeIndex(strategy_returns.index)
        strat.index = idx
    if isinstance(market_returns, pd.Series):
        mkt.index = pd.DatetimeIndex(market_returns.index)

    data = pd.concat([strat.rename("strategy"), mkt.rename("market")], axis=1).dropna()
    if data.empty:
        raise ValueError("No overlapping non-NaN data available for the CAPM regression")

    rf_s = _rf_series(pd.DatetimeIndex(data.index), rf)
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
    alpha_se = float(hac_model.bse["const"])
    alpha_t = float(hac_model.tvalues["const"])

    return pd.Series(
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
            "Market beta": float(hac_model.params["excess_market"]),
            "R2": float(ols_model.rsquared),
        }
    )


def membership_aware_rolling_mean(
    values: pd.DataFrame,
    membership_daily: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=values.index, columns=values.columns, dtype=float)
    for col in values.columns:
        member_mask = membership_daily[col].eq(1) & values[col].notna()
        if not member_mask.any():
            continue
        compressed = values.loc[member_mask, col]
        rolled = compressed.rolling(window=window, min_periods=window).mean()
        out.loc[rolled.index, col] = rolled
    return out


def load_mva_inputs(data_dir: str | Path = DATA_DIR) -> dict[str, Any]:
    root = Path(data_dir)

    tr_daily = pd.read_csv(
        root / "TOTRET_DAILY.csv",
        sep=";",
        decimal=",",
        parse_dates=["Date"],
        dayfirst=True,
        na_values=["#N/A"],
    ).set_index("Date").sort_index()
    tr_daily = tr_daily.apply(pd.to_numeric, errors="coerce")
    tr_daily = tr_daily.loc[:, tr_daily.notna().any(axis=0)]

    mcap_daily = pd.read_csv(
        root / "MCAP_DAILY.csv",
        sep=";",
        decimal=",",
        parse_dates=["Date"],
        dayfirst=True,
        na_values=["#N/A"],
    ).set_index("Date").sort_index()
    mcap_daily = mcap_daily.apply(pd.to_numeric, errors="coerce")
    mcap_daily = mcap_daily.reindex(columns=tr_daily.columns)

    weights_wide = mcap_daily.div(mcap_daily.sum(axis=1), axis=0)
    on_index_m = load_on_index_membership(root, columns=tr_daily.columns)
    on_index_d = expand_membership_to_daily(on_index_m, tr_daily.index).reindex(
        columns=tr_daily.columns
    ).fillna(0).astype(int)

    tr_in_index = tr_daily.where(on_index_d.eq(1))
    mva50 = membership_aware_rolling_mean(tr_daily, on_index_d, 50)
    mva100 = membership_aware_rolling_mean(tr_daily, on_index_d, 100)
    mva200 = membership_aware_rolling_mean(tr_daily, on_index_d, 200)

    features = pd.concat(
        {"tr": tr_in_index, "mva50": mva50, "mva100": mva100, "mva200": mva200},
        axis=1,
    ).sort_index(axis=1)

    market_daily_ret, market_monthly_ret = market_returns_from_weights(
        tr_daily=tr_daily,
        weights_wide=weights_wide,
        on_index_m=on_index_m,
    )

    return {
        "tr_daily": tr_daily,
        "tr_in_index": tr_in_index,
        "weights_wide": weights_wide,
        "on_index_m": on_index_m,
        "on_index_d": on_index_d,
        "features": features,
        "market_daily_ret": market_daily_ret,
        "market_monthly_ret": market_monthly_ret,
    }


def compute_order_score(features: pd.DataFrame) -> pd.DataFrame:
    m50 = features["mva50"]
    m100 = features["mva100"]
    m200 = features["mva200"]

    score = pd.DataFrame(np.nan, index=m50.index, columns=m50.columns)

    mask_50_100_200 = (m50 > m100) & (m100 > m200)
    mask_50_200_100 = (m50 > m200) & (m200 > m100)
    mask_100_50_200 = (m100 > m50) & (m50 > m200)
    mask_100_200_50 = (m100 > m200) & (m200 > m50)
    mask_200_50_100 = (m200 > m50) & (m50 > m100)
    mask_200_100_50 = (m200 > m100) & (m100 > m50)

    score = score.mask(mask_200_100_50, 1)
    score = score.mask(mask_100_200_50, 2)
    score = score.mask(mask_200_50_100, 3)
    score = score.mask(mask_50_200_100, 4)
    score = score.mask(mask_100_50_200, 5)
    score = score.mask(mask_50_100_200, 6)
    return score


def compute_clean_returns(features: pd.DataFrame, cap: float = 1.0) -> pd.DataFrame:
    ret = features["tr"].pct_change(fill_method=None)
    ret = ret.where(ret > -0.999999)
    return ret.clip(-cap, cap)


def daily_cap_weights(
    weights_wide: pd.DataFrame,
    index_like: pd.DatetimeIndex,
    tickers: pd.Index,
    on_index_d: pd.DataFrame | None = None,
) -> pd.DataFrame:
    weights = weights_wide.reindex(columns=tickers).copy()
    weights.index = pd.to_datetime(weights.index)
    weights = weights.sort_index().reindex(index_like).ffill()
    weights = weights.apply(pd.to_numeric, errors="coerce")
    if on_index_d is not None:
        weights = weights.where(on_index_d.reindex(index_like, columns=tickers).eq(1), 0.0)
    return weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0)


def portfolio_returns_equal(ret: pd.DataFrame, pos: pd.DataFrame) -> pd.Series:
    active = pos.astype(bool) & ret.notna()
    weights = active.astype(float)
    weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return (weights * ret.fillna(0.0)).sum(axis=1).rename("strategy")


def portfolio_returns_cap(
    ret: pd.DataFrame,
    pos: pd.DataFrame,
    w_daily: pd.DataFrame,
) -> pd.Series:
    active = pos.astype(bool) & ret.notna() & w_daily.notna()
    weights = w_daily.where(active, 0.0)
    weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return (weights * ret.fillna(0.0)).sum(axis=1).rename("strategy")


def build_order_strategy(
    features: pd.DataFrame,
    weights_wide: pd.DataFrame,
    on_index_m: pd.DataFrame,
    subset: tuple[int, ...],
    weighting: str,
    cap: float = 1.0,
) -> dict[str, Any]:
    ret = compute_clean_returns(features, cap=cap)
    on_index_d = expand_membership_to_daily(on_index_m, ret.index).reindex(
        columns=ret.columns
    ).fillna(0).astype(int)
    order = compute_order_score(features).where(on_index_d.eq(1))

    pos = order.isin(subset).shift(1, fill_value=False).astype(bool)
    pos = pos & on_index_d.eq(1) & ret.notna()

    if weighting == "equal":
        daily_ret = portfolio_returns_equal(ret, pos)
    elif weighting == "cap":
        w_daily = daily_cap_weights(weights_wide, ret.index, ret.columns, on_index_d=on_index_d)
        daily_ret = portfolio_returns_cap(ret, pos, w_daily)
    else:
        raise ValueError("weighting must be 'equal' or 'cap'")

    active = pos.any(axis=1)
    daily_ret = daily_ret.loc[active.idxmax() :] if active.any() else daily_ret.iloc[0:0]
    monthly_ret = ((1.0 + daily_ret).resample("ME").prod() - 1.0).rename("strategy")

    return {
        "subset": subset,
        "weighting": weighting,
        "positions": pos,
        "daily_ret": daily_ret,
        "monthly_ret": monthly_ret,
        "daily_equity": (1.0 + daily_ret).cumprod(),
        "monthly_equity": (1.0 + monthly_ret).cumprod(),
    }


def evaluate_all_subsets_order(
    features: pd.DataFrame,
    weights_wide: pd.DataFrame,
    on_index_m: pd.DataFrame,
    market_monthly_ret: pd.Series,
    cap: float = 1.0,
    min_months: int = 60,
    rf_monthly: float | pd.Series = RF_MONTHLY,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    regimes = [1, 2, 3, 4, 5, 6]

    for r in range(1, 7):
        for subset in itertools.combinations(regimes, r):
            for weighting in ("equal", "cap"):
                strategy = build_order_strategy(
                    features=features,
                    weights_wide=weights_wide,
                    on_index_m=on_index_m,
                    subset=subset,
                    weighting=weighting,
                    cap=cap,
                )
                monthly_ret = strategy["monthly_ret"].dropna()
                if len(monthly_ret) < min_months:
                    continue

                perf = summarize_returns(monthly_ret, periods_per_year=12)
                capm = capm_alpha_summary(
                    strategy_returns=monthly_ret,
                    market_returns=market_monthly_ret.reindex(monthly_ret.index),
                    rf=rf_monthly,
                    periods_per_year=12,
                    nw_lags=None,
                )

                rows.append(
                    {
                        "subset_in": subset,
                        "weighting": weighting,
                        **perf.to_dict(),
                        **capm.to_dict(),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["weighting", "Sharpe", "CAGR"],
        ascending=[True, False, False],
    ).reset_index(drop=True)



def summarize_best_order_methods(
    features: pd.DataFrame,
    weights_wide: pd.DataFrame,
    on_index_m: pd.DataFrame,
    market_monthly_ret: pd.Series,
    cap: float = 1.0,
    min_months: int = 60,
    rf_monthly: float | pd.Series = RF_MONTHLY,
    selection_metric: str = "Sharpe",
    **kwargs: Any,
) -> dict[str, Any]:
    if "min_obs" in kwargs:
        min_months = int(kwargs.pop("min_obs"))
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    search = evaluate_all_subsets_order(
        features=features,
        weights_wide=weights_wide,
        on_index_m=on_index_m,
        market_monthly_ret=market_monthly_ret,
        cap=cap,
        min_months=min_months,
        rf_monthly=rf_monthly,
    )

    equal_search = search.loc[search["weighting"].eq("equal")].sort_values(
        selection_metric, ascending=False
    )
    cap_search = search.loc[search["weighting"].eq("cap")].sort_values(
        selection_metric, ascending=False
    )

    best_equal_row = equal_search.iloc[0]
    best_cap_row = cap_search.iloc[0]
    best_equal = tuple(best_equal_row["subset_in"])
    best_cap = tuple(best_cap_row["subset_in"])

    equal_strategy = build_order_strategy(
        features=features,
        weights_wide=weights_wide,
        on_index_m=on_index_m,
        subset=best_equal,
        weighting="equal",
        cap=cap,
    )
    cap_strategy = build_order_strategy(
        features=features,
        weights_wide=weights_wide,
        on_index_m=on_index_m,
        subset=best_cap,
        weighting="cap",
        cap=cap,
    )

    equal_capm = capm_alpha_summary(
        strategy_returns=equal_strategy["monthly_ret"],
        market_returns=market_monthly_ret.reindex(equal_strategy["monthly_ret"].index),
        rf=rf_monthly,
        periods_per_year=12,
        nw_lags=None,
    )
    cap_capm = capm_alpha_summary(
        strategy_returns=cap_strategy["monthly_ret"],
        market_returns=market_monthly_ret.reindex(cap_strategy["monthly_ret"].index),
        rf=rf_monthly,
        periods_per_year=12,
        nw_lags=None,
    )

    summary = pd.DataFrame(
        {
            "equal_weight_best": {
                "selection_metric": selection_metric,
                "subset_in": best_equal,
                **summarize_returns(equal_strategy["monthly_ret"], periods_per_year=12).to_dict(),
                **equal_capm.to_dict(),
            },
            "cap_weight_best": {
                "selection_metric": selection_metric,
                "subset_in": best_cap,
                **summarize_returns(cap_strategy["monthly_ret"], periods_per_year=12).to_dict(),
                **cap_capm.to_dict(),
            },
        }
    )

    return {
        "search_results": search,
        "best_equal_row": best_equal_row,
        "best_cap_row": best_cap_row,
        "best_equal_subset": best_equal,
        "best_cap_subset": best_cap,
        "equal_strategy": equal_strategy,
        "cap_strategy": cap_strategy,
        "equal_capm": equal_capm,
        "cap_capm": cap_capm,
        "summary": summary,
    }
