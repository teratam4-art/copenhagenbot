#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
티커 하나만으로 1/3/6개월 뒤 기대 수익률을 계산하고, 목표 수익률을 가장 빨리
충족할 가능성이 높은 매도 시점을 추천하는 요약형 리포트 도구입니다.

예시 사용법:
    python future_exit_planner.py AAPL
    python future_exit_planner.py TSLA --target 0.10  # 목표 수익률 10%
    python future_exit_planner.py 005930.KS
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - 패키지 미설치 안내
    raise SystemExit(
        "yfinance 패키지가 필요합니다.\n"
        "pip install yfinance 명령으로 설치한 뒤 다시 실행하세요."
    )

try:
    from data_fetcher import fetch_korean_stock_data
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    DATA_FETCHER_AVAILABLE = False

try:
    from stock_screener import fetch_stock_data as fetch_stock_data_basic
    STOCK_SCREENER_FETCH_AVAILABLE = True
except ImportError:
    STOCK_SCREENER_FETCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# 설정 값
# ---------------------------------------------------------------------------
HORIZONS: Dict[str, int] = {
    "1개월": 30,
    "3개월": 90,
    "6개월": 180,
}


@dataclass
class HorizonStat:
    label: str
    days: int
    expected_return: float
    median_return: float
    low_return: float
    high_return: float
    sample_size: int


# ---------------------------------------------------------------------------
# 데이터 수집 및 통계 계산
# ---------------------------------------------------------------------------
def _is_korean_symbol(ticker: str) -> bool:
    if ticker.endswith(('.KS', '.KQ', '.KN', '.KO')):
        return True
    core = ticker.split('.')[0]
    return core.isdigit() and len(core) == 6


def _period_to_trading_days(period: str) -> int:
    try:
        value = float(period[:-1])
        unit = period[-1].lower()
    except (ValueError, IndexError):
        return 252  # 기본 1년

    if unit == 'y':
        return int(value * 252)
    if unit == 'm':
        return int(value * 21)
    if unit == 'w':
        return int(value * 5)
    if unit == 'd':
        return int(value)
    return 252


def _period_to_pages(period: str) -> int:
    days = _period_to_trading_days(period)
    rows_per_page = 10
    pages = max(5, int(math.ceil(days / rows_per_page)) + 5)
    return pages


def _fetch_korean_history(code: str, period: str) -> pd.DataFrame:
    pages = _period_to_pages(period)
    df_raw = None

    if DATA_FETCHER_AVAILABLE:
        try:
            result = fetch_korean_stock_data(code, pages=pages)
            if result and isinstance(result, dict):
                df_raw = result.get('price_data')
        except Exception:
            df_raw = None

    if df_raw is None and STOCK_SCREENER_FETCH_AVAILABLE:
        try:
            df_raw = fetch_stock_data_basic(code, pages=pages)
        except Exception:
            df_raw = None

    if df_raw is None or df_raw.empty:
        raise ValueError(f"{code} 한국 주식 데이터를 가져오지 못했습니다.")

    column_map = {
        '날짜': 'Date',
        '시가': 'Open',
        '고가': 'High',
        '저가': 'Low',
        '종가': 'Close',
        '거래량': 'Volume',
    }

    df = df_raw.rename(columns=column_map)

    required = ['Date', 'Close']
    for col in required:
        if col not in df.columns:
            raise ValueError("필수 컬럼이 부족합니다.")

    df = df.dropna(subset=['Date', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'])

    optional_map = {
        'Open': '시가',
        'High': '고가',
        'Low': '저가',
        'Volume': '거래량',
    }

    for eng_col, original_col in optional_map.items():
        if eng_col not in df.columns and original_col in df_raw.columns:
            df[eng_col] = df_raw[original_col]

    keep_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[[col for col in keep_cols if col in df.columns]]
    df = df.sort_values('Date').reset_index(drop=True)

    df = df.set_index('Date')
    return df


def _fetch_yfinance_history(symbol: str, period: str) -> pd.DataFrame:
    ticker_obj = yf.Ticker(symbol)
    df = ticker_obj.history(period=period, interval="1d", auto_adjust=False)

    if df.empty:
        raise ValueError(f"{symbol} 데이터가 비어 있습니다. 티커를 확인하세요.")

    df = df.dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def fetch_price_history(ticker: str, period: str = "3y") -> pd.DataFrame:
    ticker_normalized = ticker.strip().upper()

    candidates: list[str]
    if _is_korean_symbol(ticker_normalized):
        core = ticker_normalized.split('.')[0]
        if '.' in ticker_normalized:
            candidates = [ticker_normalized]
        else:
            candidates = [f"{core}.KS", f"{core}.KQ"]
    else:
        candidates = [ticker_normalized]

    errors: list[str] = []
    for symbol in candidates:
        try:
            return _fetch_yfinance_history(symbol, period)
        except Exception as exc:  # pragma: no cover - yfinance 실패 대비
            errors.append(f"{symbol}: {exc}")

    hint = " | ".join(errors) if errors else "알 수 없는 오류"
    raise ValueError(f"{ticker_normalized} 데이터를 가져오지 못했습니다. ({hint})")


def compute_horizon_stats(df: pd.DataFrame) -> Dict[str, HorizonStat]:
    close = df["Close"].astype(float)
    stats: Dict[str, HorizonStat] = {}

    for label, days in HORIZONS.items():
        if len(close) <= days:
            continue
        forward = close.shift(-days) / close - 1
        series = forward.iloc[:-days].dropna()
        if series.empty:
            continue

        expected = float(series.mean())
        median = float(series.median())
        low = float(series.quantile(0.1))
        high = float(series.quantile(0.9))

        stats[label] = HorizonStat(
            label=label,
            days=days,
            expected_return=expected,
            median_return=median,
            low_return=low,
            high_return=high,
            sample_size=len(series),
        )
    return stats


def detect_trend(df: pd.DataFrame, window: int = 60) -> tuple[str, float]:
    if len(df) < window + 1:
        return "추세 데이터 부족", float("nan")

    recent = df["Close"].iloc[-window:]
    x = np.arange(len(recent))
    y = np.log(recent)
    coeffs = np.polyfit(x, y, 1)  # 기울기
    slope = coeffs[0]

    annualized = (math.exp(slope * 252) - 1) * 100

    if annualized > 7:
        label = "상승"
    elif annualized < -7:
        label = "하락"
    else:
        label = "횡보"
    return label, annualized


def classify_momentum(annualized_return: float) -> str:
    if math.isnan(annualized_return):
        return "약"

    abs_val = abs(annualized_return)
    if abs_val >= 18:
        strength = "강"
    elif abs_val >= 8:
        strength = "보통"
    else:
        strength = "약"

    if annualized_return > 0:
        direction = "상승"
    elif annualized_return < 0:
        direction = "하락"
    else:
        direction = "중립"
    return f"{strength} ({direction})"


def compute_risk_level(df: pd.DataFrame) -> tuple[str, float]:
    returns = df["Close"].pct_change().dropna()
    if returns.empty:
        return "정보 부족", float("nan")

    daily_std = float(returns.std())

    if daily_std < 0.015:
        level = "낮음"
    elif daily_std < 0.03:
        level = "중간"
    else:
        level = "높음"
    return level, daily_std * 100


def determine_recommendation(
    stats: Dict[str, HorizonStat], target_return: float
) -> tuple[str, str]:
    one = stats.get("1개월")
    three = stats.get("3개월")
    six = stats.get("6개월")

    if one and one.expected_return >= target_return:
        return "단기 익절", "1개월 내 목표 수익률 달성 가능성이 가장 높습니다."
    if three and three.expected_return >= target_return:
        return "중기 보유", "3개월 보유 시 목표 수익률 충족 가능성이 높습니다."
    if six and six.expected_return >= target_return:
        return "장기 홀딩", "6개월 이상 보유해야 목표 접근 가능성이 큽니다."
    return "관망", "목표 수익률 달성 확률이 낮아 방어적으로 접근하세요."


def format_pct(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:.2f}%"


# ---------------------------------------------------------------------------
# 리포트 생성
# ---------------------------------------------------------------------------
def build_report(
    ticker: str,
    df: pd.DataFrame,
    stats: Dict[str, HorizonStat],
    target_return: float,
    avg_price: Optional[float] = None,
    stop_buffer: float = 0.05,
) -> str:
    if not stats:
        return "충분한 과거 데이터가 없어 분석할 수 없습니다."

    latest_close = float(df["Close"].iloc[-1])
    trend_label, annualized = detect_trend(df)
    momentum_desc = classify_momentum(annualized)
    risk_level, risk_pct = compute_risk_level(df)
    action, reason = determine_recommendation(stats, target_return)

    lines = []
    lines.append("=" * 70)
    lines.append(f"📊 {ticker} 미래 수익률 리포트")
    lines.append("=" * 70)
    lines.append(f"현재가: ${latest_close:,.2f}")
    if math.isnan(annualized):
        lines.append("최근 추세: 데이터 부족")
    else:
        lines.append(f"최근 추세: {trend_label} (연환산 {annualized:+.1f}%)")
    lines.append(f"모멘텀: {momentum_desc} | 리스크: {risk_level}")
    if risk_level == "높음" and not math.isnan(risk_pct):
        lines.append(f"⚠️ 변동성 높음: 일간 표준편차 {risk_pct:.2f}%")
    lines.append(f"목표 수익률: {target_return * 100:.1f}%")
    lines.append("")
    lines.append("[기간별 예상 수익률]")

    for label in ("1개월", "3개월", "6개월"):
        stat = stats.get(label)
        if not stat:
            continue
        exp_pct = stat.expected_return * 100
        lines.append(f"  • {label}: {exp_pct:+.2f}%")

    lines.append("")
    lines.append(f"🎯 추천 액션: {action}")
    lines.append(f"💡 해석: {reason}")

    base_price = avg_price if avg_price else latest_close
    target_price = base_price * (1 + target_return)
    stop_price = base_price * (1 - stop_buffer)

    lines.append(
        f"🎯 목표가: ${target_price:,.2f} (목표 수익률 {target_return*100:.1f}%)"
    )
    lines.append(f"🛡️ 손절가: ${stop_price:,.2f} (-{stop_buffer*100:.1f}%)")
    if avg_price:
        remaining_pct = (target_price / latest_close - 1) * 100
        lines.append(f"📌 현재가 대비 목표까지 {remaining_pct:.2f}% 남았습니다.")
    else:
        lines.append("📌 평단가를 입력하면 맞춤형 목표/손절가를 제공합니다.")
    lines.append("")

    card_line = "=" * 54
    one_val = stats.get("1개월")
    three_val = stats.get("3개월")
    six_val = stats.get("6개월")
    summary_line = (
        f"⏱ 1개월: {format_pct(one_val.expected_return * 100 if one_val else None)} | "
        f"3개월: {format_pct(three_val.expected_return * 100 if three_val else None)} | "
        f"6개월: {format_pct(six_val.expected_return * 100 if six_val else None)}"
    )

    trend_text = "데이터 부족" if math.isnan(annualized) else trend_label
    momentum_text = momentum_desc
    risk_text = risk_level

    lines.append(card_line)
    lines.append(f"📊 {ticker} 미래 수익률 리포트")
    lines.append(card_line)
    lines.append(
        f"현재가: ${latest_close:,.2f} | 최근 추세: {trend_text} | 모멘텀: {momentum_text} | 리스크: {risk_text}"
    )
    lines.append("-" * 54)
    lines.append(summary_line)
    lines.append(f"🎯 추천 액션: {action}")
    lines.append(
        f"🎯 목표가: ${target_price:,.2f} | 🛡️ 손절가: ${stop_price:,.2f}"
    )
    lines.append(card_line)
    lines.append("")
    lines.append("※ 과거 통계 기반 수익률로, 미래 결과를 보장하지 않습니다.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="티커만 입력하면 1/3/6개월 기대 수익률을 계산해 주는 도구",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="분석할 티커 (예: AAPL, TSLA, 005930.KS, 035720.KQ)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.07,
        help="목표 수익률 (0.07 = 7%)",
    )
    parser.add_argument(
        "--avg-price",
        type=float,
        default=None,
        help="내 평단가 (입력 시 맞춤형 목표/손절가 계산)",
    )
    parser.add_argument(
        "--stop-buffer",
        type=float,
        default=0.05,
        help="손절 버퍼 비율 (기본 0.05 = -5%)",
    )
    parser.add_argument(
        "--market",
        type=str,
        choices=["KS", "KQ"],
        default=None,
        help="한국 종목용 시장 코드 (예: KS=코스피, KQ=코스닥)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="3y",
        help="과거 분석 기간 (예: 3y, 5y, 10y)",
    )
    return parser.parse_args()

def normalize_ticker(raw: str, market: Optional[str]) -> str:
    ticker = raw.strip().upper()
    if "." in ticker:
        return ticker

    if ticker.isdigit() and len(ticker) == 6:
        return ticker

    if market and ticker:
        return f"{ticker}.{market}"

    return ticker


def main() -> None:
    args = parse_args()
    ticker = normalize_ticker(args.ticker, args.market)

    df = fetch_price_history(ticker, period=args.period)
    stats = compute_horizon_stats(df)
    report = build_report(
        ticker,
        df,
        stats,
        target_return=args.target,
        avg_price=args.avg_price,
        stop_buffer=args.stop_buffer,
    )
    print(report)


if __name__ == "__main__":
    main()

