from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from data import download_prices


def _akshare_frame(dates=("2024-01-02", "2024-01-03")) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日期": list(dates),
            "股票代码": ["000001"] * len(dates),
            "开盘": [10.0, 10.2][: len(dates)],
            "收盘": [10.1, 10.3][: len(dates)],
            "最高": [10.4, 10.5][: len(dates)],
            "最低": [9.9, 10.0][: len(dates)],
            "成交量": [1000, 1200][: len(dates)],
            "成交额": [10100.0, 12360.0][: len(dates)],
        }
    )


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [("000001.SZ", "000001"), ("600519.SH", "600519"), ("430047.BJ", "430047")],
)
def test_ticker_conversion(ticker: str, expected: str) -> None:
    assert download_prices.ticker_to_akshare_symbol(ticker) == expected


def test_ticker_conversion_rejects_unqualified_code() -> None:
    with pytest.raises(ValueError, match="Invalid A-share ticker"):
        download_prices.ticker_to_akshare_symbol("000001")


def test_output_schema_and_akshare_call(monkeypatch, tmp_path) -> None:
    client = Mock(return_value=_akshare_frame())
    monkeypatch.setattr(download_prices.ak, "stock_zh_a_hist", client)

    status = download_prices.download_stock_daily(
        "000001.SZ",
        "2024-01-02",
        "2024-01-03",
        output_dir=tmp_path,
        backoff_seconds=0,
    )

    assert status["success"] is True
    saved = pd.read_parquet(tmp_path / "000001.SZ.parquet")
    assert list(saved.columns) == download_prices.OUTPUT_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(saved["date"])
    assert saved["date"].is_monotonic_increasing
    assert saved["ticker"].eq("000001.SZ").all()
    client.assert_called_once_with(
        symbol="000001",
        period="daily",
        start_date="20240102",
        end_date="20240103",
        adjust="qfq",
    )


def test_duplicate_dates_are_rejected(monkeypatch, tmp_path) -> None:
    duplicate_frame = _akshare_frame(("2024-01-02", "2024-01-02"))
    monkeypatch.setattr(
        download_prices.ak,
        "stock_zh_a_hist",
        Mock(return_value=duplicate_frame),
    )

    status = download_prices.download_stock_daily(
        "000001.SZ",
        "2024-01-02",
        "2024-01-03",
        output_dir=tmp_path,
        backoff_seconds=0,
    )

    assert status["success"] is False
    assert "duplicate dates" in status["error"]
    assert not (tmp_path / "000001.SZ.parquet").exists()


def test_existing_file_covering_range_skips_download(monkeypatch, tmp_path) -> None:
    existing = download_prices.normalize_price_frame(
        _akshare_frame(),
        ticker="000001.SZ",
        adjust="qfq",
    )
    destination = tmp_path / "000001.SZ.parquet"
    existing.to_parquet(destination, index=False)
    client = Mock(side_effect=AssertionError("AKShare should not be called"))
    monkeypatch.setattr(download_prices.ak, "stock_zh_a_hist", client)

    status = download_prices.download_stock_daily(
        "000001.SZ",
        "2024-01-02",
        "2024-01-03",
        output_dir=tmp_path,
    )

    assert status == {
        "ticker": "000001.SZ",
        "success": True,
        "rows": 2,
        "start_date": "2024-01-02",
        "end_date": "2024-01-03",
        "error": None,
    }
    client.assert_not_called()


def test_force_redownloads_existing_file(monkeypatch, tmp_path) -> None:
    existing = download_prices.normalize_price_frame(
        _akshare_frame(),
        ticker="000001.SZ",
        adjust="qfq",
    )
    existing.to_parquet(tmp_path / "000001.SZ.parquet", index=False)
    client = Mock(return_value=_akshare_frame())
    monkeypatch.setattr(download_prices.ak, "stock_zh_a_hist", client)

    status = download_prices.download_stock_daily(
        "000001.SZ",
        "2024-01-02",
        "2024-01-03",
        output_dir=tmp_path,
        force=True,
        backoff_seconds=0,
    )

    assert status["success"] is True
    client.assert_called_once()
