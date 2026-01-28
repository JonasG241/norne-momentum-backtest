import pandas as pd
import pandas_market_calendars as mcal


def forward_fill_missing_dates(dt, exchange):
    dt = dt.copy()
    dt["Date"] = pd.to_datetime(dt["Date"])
    dt = dt.sort_values("Date")
    dt = dt.set_index("Date")

    start_date = dt.index.min()
    end_date = dt.index.max()

    schedule = mcal.get_calendar(exchange).schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index.tz_localize(None).normalize()

    dt = dt.reindex(trading_days)
    dt.ffill(inplace=True)

    return dt



def align_missing_dates_as_nan(dt, exchange, date_col="Date"):
    dt = dt.copy()
    dt[date_col] = pd.to_datetime(dt[date_col])
    dt = dt.sort_values(date_col).set_index(date_col)

    start_date = dt.index.min()
    end_date = dt.index.max()

    schedule = mcal.get_calendar(exchange).schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index.tz_localize(None).normalize()

    dt = dt.reindex(trading_days)
    return dt