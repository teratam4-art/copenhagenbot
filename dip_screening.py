"""Dip (눌림목) screening utilities for stock_screener.py."""

from __future__ import annotations

import io
import math
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dip_screening requires yfinance. Install with `pip install yfinance`."
    ) from exc


# ---------------------------------------------------------------------------
# 사용자 조정 변수 (필요 시 숫자만 바꿔서 사용하세요)
# ---------------------------------------------------------------------------

LOOKBACK_DAYS = 160
RSI_WINDOW = 14

MAX_SHOW_KR = 15
MAX_SHOW_US = 15
MIN_VOLUME_KR = 1_000_000_000
RSI_MIN = 40
RSI_MAX = 55

RECENT_CANDLE_WINDOW = 3
MIN_NEGATIVE_CANDLES = 2

US_RECENT_RETURN_WINDOW = 5
US_RETURN_THRESHOLD = -0.03
US_VOLUME_RATIO_RANGE = (0.8, 1.5)

OUTPUT_ROOT = Path("outputs")
TXT_OUTPUT_DIR = OUTPUT_ROOT / "txt"
CSV_OUTPUT_DIR = OUTPUT_ROOT / "csv"

for directory in (TXT_OUTPUT_DIR, CSV_OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class DipScreeningResult:
    ticker: str
    name: str
    close: float
    ma5: float
    ma20: float
    ma60: float
    rsi: float
    today_change_pct: float
    dist_to_ma20_pct: float
    extra: dict
    comment: str


def _compute_rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _compute_ma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def _infer_korean_candidates(code: str) -> List[str]:
    code = code.strip().upper()
    if code.endswith((".KS", ".KQ")):
        return [code]
    if not code.isdigit() or len(code) != 6:
        return [code]
    return [f"{code}.KS", f"{code}.KQ"]


def _fetch_history(ticker: str, lookback_days: int, error_log: List[str]) -> Optional[pd.DataFrame]:
    calendar_days = int(lookback_days * 1.6)
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            df = yf.Ticker(ticker).history(
                period=f"{calendar_days}d", interval="1d", auto_adjust=False
            )
    except Exception as exc:
        error_log.append(f"{ticker}: {exc}")
        return None

    combined_log = (buf_out.getvalue() + buf_err.getvalue()).strip()
    if combined_log:
        for line in combined_log.splitlines():
            line = line.strip()
            if line:
                error_log.append(line)

    if df.empty:
        error_log.append(f"{ticker}: no price data")
        return None

    df = df.dropna(subset=["Close"]).copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    if len(df) < lookback_days:
        error_log.append(f"{ticker}: insufficient history ({len(df)} bars)")
        return None
    return df


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA5"] = _compute_ma(df["Close"], 5)
    df["MA20"] = _compute_ma(df["Close"], 20)
    df["MA60"] = _compute_ma(df["Close"], 60)
    df["RSI"] = _compute_rsi(df["Close"])
    df["AvgValue20"] = (df["Close"] * df.get("Volume", 0)).rolling(20).mean()
    df["VolMA20"] = df.get("Volume", 0).rolling(20).mean()
    return df.dropna().copy()


def _safe_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


def _screen_korea(codes: Sequence[str], error_log: List[str]) -> Tuple[List[DipScreeningResult], List[str]]:
    results: List[DipScreeningResult] = []
    missing: List[str] = []

    for code in codes:
        dataset: Optional[pd.DataFrame] = None
        resolved: Optional[str] = None

        for candidate in _infer_korean_candidates(code):
            dataset = _fetch_history(candidate, LOOKBACK_DAYS, error_log)
            if dataset is not None:
                resolved = candidate
                break

        if dataset is None or resolved is None:
            missing.append(code)
            continue

        enriched = _enrich(dataset)
        if enriched.empty:
            missing.append(code)
            continue

        latest = enriched.iloc[-1]

        if not (latest["MA20"] > latest["MA60"]):
            continue

        short_pullback = False
        if latest["Close"] < latest["MA5"]:
            short_pullback = True
        else:
            recent = enriched["Close"].diff().iloc[-RECENT_CANDLE_WINDOW:]
            if (recent < 0).sum() >= MIN_NEGATIVE_CANDLES:
                short_pullback = True

        if not short_pullback:
            continue

        rsi_value = float(latest["RSI"])
        if not (RSI_MIN <= rsi_value <= RSI_MAX):
            continue

        avg_trade_value = float(latest["AvgValue20"])
        if math.isnan(avg_trade_value) or avg_trade_value < MIN_VOLUME_KR:
            continue

        close = float(latest["Close"])
        ma20 = float(latest["MA20"])
        ma5 = float(latest["MA5"])
        ma60 = float(latest["MA60"])
        today_change = float(enriched["Close"].pct_change().iloc[-1] * 100)
        dist_ma20_pct = (close / ma20 - 1) * 100

        notes = []
        if close < ma5:
            notes.append("MA5 이탈")
        if close < ma20:
            notes.append("MA20 근접")
        if enriched["Close"].diff().iloc[-RECENT_CANDLE_WINDOW:].lt(0).sum() >= MIN_NEGATIVE_CANDLES:
            notes.append("최근 음봉 다수")
        notes.append(f"RSI {RSI_MIN}~{RSI_MAX} 구간")
        notes.append("중기 추세 유지")

        results.append(
            DipScreeningResult(
                ticker=resolved,
                name=_safe_name(resolved),
                close=close,
                ma5=ma5,
                ma20=ma20,
                ma60=ma60,
                rsi=rsi_value,
                today_change_pct=today_change,
                dist_to_ma20_pct=dist_ma20_pct,
                extra={"avg_value": avg_trade_value},
                comment=" / ".join(notes),
            )
        )

    results.sort(
        key=lambda r: (
            abs(r.dist_to_ma20_pct),
            abs(r.rsi - (RSI_MIN + RSI_MAX) / 2),
            -r.extra.get("avg_value", 0),
        )
    )

    return results, missing


def _screen_us(tickers: Sequence[str], error_log: List[str]) -> Tuple[List[DipScreeningResult], List[str]]:
    results: List[DipScreeningResult] = []
    missing: List[str] = []

    for symbol in tickers:
        dataset = _fetch_history(symbol, LOOKBACK_DAYS, error_log)
        if dataset is None:
            missing.append(symbol)
            continue

        enriched = _enrich(dataset)
        if enriched.empty:
            missing.append(symbol)
            continue

        latest = enriched.iloc[-1]

        if not (latest["MA20"] > latest["MA60"]):
            continue

        short_pullback = False
        if latest["Close"] < latest["MA5"]:
            short_pullback = True
        else:
            recent_return = (
                enriched["Close"].iloc[-US_RECENT_RETURN_WINDOW:] / enriched["Close"].iloc[-US_RECENT_RETURN_WINDOW] - 1
            )
            if recent_return.iloc[-1] <= US_RETURN_THRESHOLD:
                short_pullback = True

        if not short_pullback:
            continue

        rsi_value = float(latest["RSI"])
        if not (RSI_MIN <= rsi_value <= RSI_MAX):
            continue

        vol_ma20 = float(latest.get("VolMA20", np.nan))
        vol_today = float(latest.get("Volume", np.nan))
        if math.isnan(vol_ma20) or vol_ma20 <= 0:
            volume_ratio = float("nan")
        else:
            volume_ratio = vol_today / vol_ma20

        close = float(latest["Close"])
        ma20 = float(latest["MA20"])
        ma5 = float(latest["MA5"])
        ma60 = float(latest["MA60"])
        today_change = float(enriched["Close"].pct_change().iloc[-1] * 100)
        dist_ma20_pct = (close / ma20 - 1) * 100

        notes = []
        if close < ma5:
            notes.append("MA5 하향 이탈")
        recent_prod = enriched["Close"].iloc[-US_RECENT_RETURN_WINDOW:].pct_change().add(1).prod() - 1
        if recent_prod <= US_RETURN_THRESHOLD:
            notes.append(f"최근 {US_RECENT_RETURN_WINDOW}일 약세")
        notes.append(f"RSI {RSI_MIN}~{RSI_MAX}")
        if US_VOLUME_RATIO_RANGE[0] <= volume_ratio <= US_VOLUME_RATIO_RANGE[1]:
            notes.append("거래량 평소 대비 적정")
        elif not math.isnan(volume_ratio) and volume_ratio > US_VOLUME_RATIO_RANGE[1]:
            notes.append("거래량 급증 주의")
        notes.append("중기 추세 유지")

        results.append(
            DipScreeningResult(
                ticker=symbol,
                name=_safe_name(symbol),
                close=close,
                ma5=ma5,
                ma20=ma20,
                ma60=ma60,
                rsi=rsi_value,
                today_change_pct=today_change,
                dist_to_ma20_pct=dist_ma20_pct,
                extra={"volume_ratio": volume_ratio},
                comment=" / ".join(notes),
            )
        )

    results.sort(
        key=lambda r: (
            abs(r.dist_to_ma20_pct),
            abs(r.rsi - (RSI_MIN + RSI_MAX) / 2),
            -r.extra.get("volume_ratio", 0),
        )
    )

    return results, missing


def _format_currency(value: float, market: str) -> str:
    if market == "KR":
        return f"{value:,.0f}원"
    return f"${value:,.2f}"


def _format_price_pair(value: float, market: str) -> str:
    if market == "KR":
        return f"{value:,.0f}"
    return f"${value:,.2f}"


def _format_percentage(value: float) -> str:
    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:+.2f}%"


def _build_state_text(item: DipScreeningResult) -> str:
    return item.comment.replace(" / ", ", ")


def _build_comment(item: DipScreeningResult, market: str) -> str:
    base_parts: List[str] = []
    if item.close < item.ma5 and item.dist_to_ma20_pct < 0:
        base_parts.append("단기선이 꺾였지만 중기선은 위쪽이라 눌림 이후 반등을 노릴 자리입니다")
    elif item.dist_to_ma20_pct < 0:
        base_parts.append("MA20 아래에서 눌리고 있어 재상승 시점을 기다리는 구간입니다")
    else:
        base_parts.append("MA20 위에서 쉬고 있어 상승 추세가 유지되고 있습니다")

    if item.rsi <= RSI_MIN + 2:
        base_parts.append("RSI가 낮아 과열 신호는 아닙니다")
    elif item.rsi >= RSI_MAX - 2:
        base_parts.append("RSI가 회복 중인지 체크하세요")

    return ", ".join(base_parts)


def _describe_rsi(value: float) -> str:
    if value >= RSI_MAX:
        return "(과열 주의)"
    if value <= RSI_MIN:
        return "(저점 탐색)"
    return "(과열 아님)"


def _build_dataframe(results: List[DipScreeningResult], market: str) -> pd.DataFrame:
    rows = []
    for item in results:
        row = {
            "Ticker": item.ticker,
            "Name": item.name,
            "Close": item.close,
            "MA5": item.ma5,
            "MA20": item.ma20,
            "MA60": item.ma60,
            "RSI": item.rsi,
            "오늘등락(%)": item.today_change_pct,
            "MA20乂(%)": item.dist_to_ma20_pct,
            "Comment": item.comment,
        }
        if market == "KR":
            row["평균거래대금(20일)"] = item.extra.get("avg_value", 0)
        else:
            row["VolumeRatio"] = item.extra.get("volume_ratio")
        rows.append(row)
    return pd.DataFrame(rows)


def _save_full_results(label: str, df: pd.DataFrame) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = CSV_OUTPUT_DIR / f"dip_{label}_{timestamp}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def _shorten_list(values: Sequence[str], max_items: int = 8) -> str:
    unique = list(dict.fromkeys(values))
    if not unique:
        return ""
    preview = unique[:max_items]
    text = ", ".join(preview)
    if len(unique) > max_items:
        text += ", ..."
    return text


def _render_cards(results: List[DipScreeningResult], market: str) -> List[str]:
    cards: List[str] = []
    header = "KR" if market == "KR" else "US"
    for idx, item in enumerate(results, 1):
        name = item.name or "-"
        lines = [f"[{header} {idx}] {item.ticker} ({name})"]
        lines.append(f" - 현재가: {_format_currency(item.close, market)}")
        lines.append(
            " - 이평선: MA5 "
            f"{_format_price_pair(item.ma5, market)} | MA20 {_format_price_pair(item.ma20, market)}"
        )
        lines.append(f" - MA20와의 거리: {_format_percentage(item.dist_to_ma20_pct)}")
        if item.today_change_pct is not None and not math.isnan(item.today_change_pct):
            lines.append(f" - 오늘: {_format_percentage(item.today_change_pct)}")
        lines.append(f" - RSI: {item.rsi:.1f} {_describe_rsi(item.rsi)}")
        lines.append(f" - 눌림 상태: {_build_state_text(item)}")
        lines.append(f" - 코멘트: \"{_build_comment(item, market)}\"")
        cards.append("\n".join(lines))
    return cards


def run_dip_screening(
    get_top_korean_stocks: Optional[Callable[..., Sequence[str]]] = None,
    get_top_us_stocks: Optional[Callable[..., Sequence[str]]] = None,
    kr_limit: int = 200,
    us_limit: int = 300,
    max_show_kr: int = MAX_SHOW_KR,
    max_show_us: int = MAX_SHOW_US,
) -> None:
    error_log: List[str] = []

    if get_top_korean_stocks:
        try:
            korean_codes = list(get_top_korean_stocks(limit=kr_limit))
        except Exception:
            korean_codes = []
    else:
        korean_codes = []

    if not korean_codes:
        korean_codes = [
            "005930",
            "000660",
            "068270",
            "035420",
            "035720",
            "051910",
            "207940",
            "006400",
            "373220",
            "096770",
        ]

    kr_results, kr_missing = _screen_korea(korean_codes, error_log)

    if get_top_us_stocks:
        try:
            us_tickers = list(get_top_us_stocks(limit=us_limit))
        except Exception:
            us_tickers = []
    else:
        us_tickers = []

    if not us_tickers:
        us_tickers = [
            "AAPL",
            "MSFT",
            "NVDA",
            "META",
            "TSLA",
            "AMD",
            "NFLX",
            "GOOGL",
            "AMZN",
            "AVGO",
            "ADBE",
            "CRM",
        ]

    us_results, us_missing = _screen_us(us_tickers, error_log)

    failed_codes = sorted(set(kr_missing + us_missing))
    if failed_codes:
        summary = _shorten_list(failed_codes, max_items=6)
        print(f"[데이터 미수집 종목] {len(failed_codes)}개: {summary}")
        print("(야후에 없어서 건너뜀)\n")

    unique_errors = sorted(set(error_log))
    if unique_errors:
        first = unique_errors[0]
        more = f" 외 {len(unique_errors) - 1}건" if len(unique_errors) > 1 else ""
        print(f"[참고] 데이터 수집 중 경고: {first}{more}\n")

    print("=" * 80)
    print("🇰🇷 한국 KOSPI/KOSDAQ 눌림목 후보")
    print("=" * 80)
    print(
        "아래 종목들은 ‘상승 추세는 살려둔 상태에서 오늘/최근에만 눌린’ 종목들입니다.\n"
        "눌림목이라는 뜻일 뿐, 실제 매수는 추가 확인이 필요합니다.\n"
    )

    kr_total = len(kr_results)
    kr_show = kr_results[:max_show_kr]
    if kr_show:
        cards = _render_cards(kr_show, "KR")
        print("\n\n".join(cards))
        if kr_total > len(kr_show):
            df_full = _build_dataframe(kr_results, "KR")
            path = _save_full_results("KR", df_full)
            print(
                f"\n💾 나머지 {kr_total - len(kr_show)}개 종목은 '{path}' 파일에 저장했습니다."
            )
    else:
        print("조건을 만족하는 한국 종목이 없습니다.")

    print("\n" + "=" * 80)
    print("🇺🇸 미국 (NYSE/NASDAQ) dip 후보")
    print("=" * 80)
    print(
        "달러 종목도 같은 기준으로 눌림만 모아봤습니다.\n"
        f"RSI가 {RSI_MIN}~{RSI_MAX} 사이면 다시 위로 돌 가능성이 있습니다.\n"
    )

    us_total = len(us_results)
    us_show = us_results[:max_show_us]
    if us_show:
        cards = _render_cards(us_show, "US")
        print("\n\n".join(cards))
        if us_total > len(us_show):
            df_full = _build_dataframe(us_results, "US")
            path = _save_full_results("US", df_full)
            print(
                f"\n💾 나머지 {us_total - len(us_show)}개 종목은 '{path}' 파일에 저장했습니다."
            )
    else:
        print("조건을 만족하는 미국 종목이 없습니다.")

    print("\n[요약]")
    print(
        f"- 한국 눌림 후보: {len(kr_show)}개 표시 (전체 {kr_total}개 중)"
    )
    print(
        f"- 미국 눌림 후보: {len(us_show)}개 표시 (전체 {us_total}개 중)"
    )
    print(
        "- 이 스크리닝은 기술적 조건만 본 것이므로 공시/실적/뉴스는 별도로 확인하세요."
    )

