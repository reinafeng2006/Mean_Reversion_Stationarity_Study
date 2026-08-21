"""Download and validate daily China A-share prices from AKShare.

The default ``qfq`` adjustment is a prototype convention. AKShare's adjusted
history is not treated here as point-in-time data and must not be used as such
without a separate bias and methodology review.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd

LOGGER = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "adjust",
]

_TICKER_PATTERN = re.compile(r"^(?P<code>\d{6})\.(?P<exchange>SZ|SH|BJ)$")
_COLUMN_ALIASES = {
    "日期": "date",
    "date": "date",
    "股票代码": "source_code",
    "开盘": "open",
    "open": "open",
    "最高": "high",
    "high": "high",
    "最低": "low",
    "low": "low",
    "收盘": "close",
    "close": "close",
    "成交量": "volume",
    "volume": "volume",
    "成交额": "amount",
    "amount": "amount",
}


class PriceDataValidationError(ValueError):
    """Raised when downloaded price observations fail required checks."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__("; ".join(issues))


def ticker_to_akshare_symbol(ticker: str) -> str:
    """Convert an exchange-qualified ticker to AKShare's six-digit symbol."""

    normalized = str(ticker).strip().upper()
    match = _TICKER_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(
            f"Invalid A-share ticker {ticker!r}; expected formats such as "
            "'000001.SZ' or '600519.SH'."
        )
    return match.group("code")


def _normalize_date(value: Any, field_name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}") from exc
    if pd.isna(result):
        raise ValueError(f"Invalid {field_name}: {value!r}")
    return result


def _akshare_date(value: pd.Timestamp) -> str:
    return value.strftime("%Y%m%d")


def normalize_price_frame(
    raw: pd.DataFrame,
    *,
    ticker: str,
    adjust: str,
) -> pd.DataFrame:
    """Standardize an AKShare ``stock_zh_a_hist`` response."""

    if raw is None or raw.empty:
        raise PriceDataValidationError(["AKShare returned no rows"])

    renamed = raw.rename(
        columns={column: _COLUMN_ALIASES.get(str(column), str(column)) for column in raw.columns}
    )
    required_source = ["date", "open", "high", "low", "close", "volume", "amount"]
    missing_columns = [column for column in required_source if column not in renamed.columns]
    if missing_columns:
        raise PriceDataValidationError(
            [f"AKShare response is missing required columns: {missing_columns}"]
        )

    result = renamed[required_source].copy()
    result.insert(1, "ticker", ticker)
    result["adjust"] = adjust
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result[OUTPUT_COLUMNS].sort_values("date").reset_index(drop=True)
    validate_price_data(result)
    return result


def validate_price_data(frame: pd.DataFrame) -> None:
    """Raise with all detected violations; never fill or drop observations."""

    issues: list[str] = []
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
    if missing_columns:
        issues.append(f"missing output columns: {missing_columns}")
        raise PriceDataValidationError(issues)

    invalid_dates = int(frame["date"].isna().sum())
    if invalid_dates:
        issues.append(f"{invalid_dates} missing or invalid dates")

    duplicate_dates = int(frame["date"].duplicated(keep=False).sum())
    if duplicate_dates:
        issues.append(f"{duplicate_dates} rows have duplicate dates")

    missing_close = int(frame["close"].isna().sum())
    if missing_close:
        issues.append(f"{missing_close} rows have missing close prices")

    price_columns = ["open", "high", "low", "close"]
    non_positive = int((frame[price_columns] <= 0).any(axis=1).sum())
    if non_positive:
        issues.append(f"{non_positive} rows have non-positive prices")

    for column in ["volume", "amount"]:
        missing = int(frame[column].isna().sum())
        if missing:
            issues.append(f"{missing} rows have missing {column}")

    if issues:
        raise PriceDataValidationError(issues)


def _status(
    ticker: str,
    *,
    success: bool,
    rows: int = 0,
    start_date: str | None = None,
    end_date: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "success": success,
        "rows": rows,
        "start_date": start_date,
        "end_date": end_date,
        "error": error,
    }


def _existing_file_status(
    path: Path,
    *,
    ticker: str,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    adjust: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        existing = pd.read_parquet(path)
        validate_price_data(existing)
        dates = pd.to_datetime(existing["date"], errors="coerce")
        same_adjustment = "adjust" in existing and bool(
            existing["adjust"].eq(adjust).all()
        )
        covers_range = bool(
            not dates.empty
            and dates.min() <= requested_start
            and dates.max() >= requested_end
        )
        if same_adjustment and covers_range:
            LOGGER.info("Skipping %s; existing file covers requested range", ticker)
            return _status(
                ticker,
                success=True,
                rows=len(existing),
                start_date=dates.min().date().isoformat(),
                end_date=dates.max().date().isoformat(),
                error=None,
            )
    except Exception as exc:  # A corrupt/incompatible cache should trigger refresh.
        LOGGER.warning("Could not reuse existing file %s: %s", path, exc)
    return None


def download_stock_daily(
    ticker: str,
    start_date: Any,
    end_date: Any,
    adjust: str = "qfq",
    output_dir: str | Path = "data/raw/prices",
    *,
    force: bool = False,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> dict[str, Any]:
    """Download, validate, and persist one stock's daily price history.

    ``qfq`` is only a prototype convention. The returned series must not be
    described as point-in-time adjusted without additional methodology work.
    """

    normalized_ticker = str(ticker).strip().upper()
    try:
        symbol = ticker_to_akshare_symbol(normalized_ticker)
        requested_start = _normalize_date(start_date, "start_date")
        requested_end = _normalize_date(end_date, "end_date")
        if requested_start > requested_end:
            raise ValueError("start_date must be on or before end_date")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds cannot be negative")

        destination = Path(output_dir) / f"{normalized_ticker}.parquet"
        if not force:
            cached = _existing_file_status(
                destination,
                ticker=normalized_ticker,
                requested_start=requested_start,
                requested_end=requested_end,
                adjust=adjust,
            )
            if cached is not None:
                return cached

        raw = None
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=_akshare_date(requested_start),
                    end_date=_akshare_date(requested_end),
                    adjust=adjust,
                )
                break
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "AKShare download failed for %s (attempt %s/%s): %s",
                    normalized_ticker,
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(backoff_seconds * (2 ** (attempt - 1)))
        if raw is None:
            raise RuntimeError(f"AKShare failed after {max_retries} attempts: {last_error}")

        result = normalize_price_frame(
            raw,
            ticker=normalized_ticker,
            adjust=adjust,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.parquet")
        result.to_parquet(temporary, index=False)
        temporary.replace(destination)

        dates = result["date"]
        return _status(
            normalized_ticker,
            success=True,
            rows=len(result),
            start_date=dates.min().date().isoformat(),
            end_date=dates.max().date().isoformat(),
            error=None,
        )
    except Exception as exc:
        LOGGER.exception("Daily-price download failed for %s", normalized_ticker)
        return _status(normalized_ticker, success=False, error=str(exc))


def download_stock_daily_batch(
    tickers: Iterable[str],
    start_date: Any,
    end_date: Any,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Download multiple tickers, logging failures and continuing the batch."""

    statuses = []
    for ticker in tickers:
        status = download_stock_daily(ticker, start_date, end_date, **kwargs)
        statuses.append(status)
        if not status["success"]:
            LOGGER.error("Batch item failed: %s: %s", ticker, status["error"])
    return statuses
