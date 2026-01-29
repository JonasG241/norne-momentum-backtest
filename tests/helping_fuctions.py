import pandas as pd
import pandas_market_calendars as mcal


def align_missing_dates_as_nan(dt: pd.DataFrame, exchange: str = "OSE") -> pd.DataFrame:
    """
    Aligns a stock dataframe to *trading days* for the given exchange.
    Any missing trading days are inserted and will have NaNs for price columns.

    Input:
        dt: DataFrame with column "Date" (datetime-like) and price columns (e.g. Close)
        exchange: calendar name for pandas_market_calendars (e.g. "OSE")

    Output:
        DataFrame indexed by Date (tz-naive), sorted, reindexed to trading days
    """
    if "Date" not in dt.columns:
        raise ValueError("align_missing_dates_as_nan: dt must contain a 'Date' column")

    out = dt.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date")

    start_date = out["Date"].iloc[0]
    end_date = out["Date"].iloc[-1]

    cal = mcal.get_calendar(exchange)
    schedule = cal.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index.tz_localize(None).normalize()

    out["Date"] = out["Date"].dt.tz_localize(None).dt.normalize()
    out = out.drop_duplicates(subset=["Date"], keep="last")
    out = out.set_index("Date").sort_index()

    # Reindex to trading days -> missing days appear as NaN rows
    out = out.reindex(trading_days)

    return out
