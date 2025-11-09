#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render 배포용 실시간 신호 감시 스크립트

기능
------
1. 지정한 종목들의 현재 시세와 수급 데이터를 조회
2. 매수/손절/익절 조건 충족 여부 판단
3. 조건 충족 시 텔레그램 알림 전송 (중복 방지 및 쿨다운 지원)
4. 매 사이클마다 환경변수를 다시 읽어, Render 재배포 없이 설정 변경을 반영

환경 변수
---------
STOCK_CODES               : 감시할 종목 코드 목록 (쉼표 구분, 예: "005930,035720,AAPL")
WATCH_SYMBOLS             : 맞춤 감시 종목 (예: "KTG:033780,CSCO")
POSITIONS                 : 보유 종목 평균단가 (예: "005930=72000,AAPL=182.5@50")
TELEGRAM_BOT_TOKEN        : 텔레그램 봇 토큰
TELEGRAM_CHAT_ID          : 텔레그램 채팅 ID
CHECK_INTERVAL_SECONDS    : 감시 주기(초). 기본 1800 (30분)
ALERT_COOLDOWN_MINUTES    : 동일 알림 재발송 최소 간격(분). 기본 60
ENTRY_TOLERANCE_PCT       : 매수 목표가 대비 허용 오차(%) 기본 1.0
STOP_LOSS_TOLERANCE_PCT   : 손절 라인 초과 허용 폭(%) 기본 0
TAKE_PROFIT_TOLERANCE_PCT : 익절 라인 허용 오차(%) 기본 0
ALERT_STATE_PATH          : 알림 상태 저장 파일 경로. 기본 outputs/txt/alert_state.json
RUN_ONCE                  : "1"/"true" 등으로 설정하면 1회 실행 후 종료

맞춤 알림 변수 (ALIAS는 WATCH_SYMBOLS에서 지정한 별칭을 의미)
------------------------------------------------------------
ALIAS_LOW_PRICE / ALIAS_HIGH_PRICE      : 가격 구간 감시
ALIAS_BREAKOUT_PRICE                    : 돌파 감시 기준가
ALIAS_PRICE_BUFFER_PCT                  : 가격 허용 오차 (%)
ALIAS_VOLUME_MULT / ALIAS_VOLUME_MIN    : 거래량 배수·최소 거래량
ALIAS_TIMEFRAME_MIN / ALIAS_BULL_CANDLES: 분봉 단위 및 연속 양봉 개수
ALIAS_ALLOW_WICK                        : 긴 윗꼬리 허용 여부 (기본 true)
ALIAS_STOP_LOSS / ALIAS_STOP_WARN_PCT   : 손절가 및 예고 퍼센트
ALIAS_TAKE_PROFIT_1 / _2                : 익절 목표가
ALIAS_ALERT_COOLDOWN_SEC                : 맞춤 알림 쿨다운(초)
ALIAS_ONCE_PER_DAY                      : 하루 1회 제한 (true/false)
ALIAS_ACTIVE_FROM / ALIAS_ACTIVE_TO     : 알림 유효 시간 (HH:MM)
ALIAS_ACTIVE_TZ                         : 해당 알림 전용 타임존 (예: Asia/Seoul)

Render에서의 사용
-----------------
Start Command 예시:
    python signal_watcher.py
또는 개발/테스트용:
    RUN_ONCE=1 python signal_watcher.py
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from data_fetcher import (
    fetch_investor_trading_data,
    fetch_korean_stock_data,
    fetch_intraday_data,
    fetch_technical_indicators,
    fetch_us_stock_data,
    is_us_stock,
)
from pattern_detector import analyze_investor_pattern, detect_recovery_signal
from risk_manager import calculate_stop_loss
from signal_generator import generate_buy_signals, generate_sell_signals

# --------------------------------------------------------------------------- #
# 설정 & 로거
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_PATH = BASE_DIR / "outputs" / "txt" / "alert_state.json"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("signal_watcher")


# --------------------------------------------------------------------------- #
# 유틸
# --------------------------------------------------------------------------- #

def refresh_environment() -> None:
    """로컬 실행 시 .env를 재로딩하고, Render에서는 무시."""
    if load_dotenv is None:
        return
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


def parse_stock_codes(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [code.strip() for code in raw.split(",") if code.strip()]


def parse_positions(raw: Optional[str]) -> Dict[str, Dict[str, float]]:
    """
    POSITIONS 환경 변수 파서

    지원되는 형식:
        CODE=price
        CODE=price@quantity
    여러 종목은 쉼표로 구분
    """
    positions: Dict[str, Dict[str, float]] = {}
    if not raw:
        return positions

    entries = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    for entry in entries:
        if "=" not in entry:
            continue
        code_part, value_part = entry.split("=", 1)
        code = code_part.strip()
        if not code:
            continue

        price_str, qty_str = value_part, None
        if "@" in value_part:
            price_str, qty_str = value_part.split("@", 1)

        try:
            price = float(price_str.strip())
        except ValueError:
            logger.warning("평단가 파싱 실패: %s", entry)
            continue

        quantity = None
        if qty_str:
            try:
                quantity = float(qty_str.strip())
            except ValueError:
                logger.warning("보유 수량 파싱 실패: %s", entry)

        positions[code] = {
            "buy_price": price,
            "quantity": quantity if quantity is not None else 0.0,
        }
    return positions


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_price(value: float, is_us: bool) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.2f}" if is_us else f"{value:,.0f}원"


def format_integer(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        try:
            return f"{float(value):,.0f}"
        except (TypeError, ValueError):
            return str(value)


def current_timestamp() -> str:
    return datetime.utcnow().isoformat()


def parse_timestamp(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def parse_watch_symbols(raw: Optional[str]) -> List[Tuple[str, str]]:
    if not raw:
        return []

    symbols: List[Tuple[str, str]] = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        if ":" in token:
            alias, code = token.split(":", 1)
        else:
            alias, code = token, token
        alias = alias.strip()
        code = code.strip()
        if not code:
            continue
        symbols.append((alias, code))
    return symbols


def sanitize_alias(alias: str) -> str:
    if not alias:
        return ""
    safe = "".join(ch if ch.isalnum() else "_" for ch in alias.upper())
    return safe


def get_env_float(key: str) -> Optional[float]:
    value = os.getenv(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        logger.warning("환경 변수 %s 값을 float으로 변환할 수 없습니다: %s", key, value)
        return None


def get_env_int(key: str) -> Optional[int]:
    value = os.getenv(key)
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        logger.warning("환경 변수 %s 값을 int로 변환할 수 없습니다: %s", key, value)
        return None


def parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_time_string(value: Optional[str]) -> Optional[dt_time]:
    if value is None or not value.strip():
        return None
    try:
        hour, minute = value.strip().split(":")
        return dt_time(int(hour), int(minute))
    except Exception:
        logger.warning("시간 문자열을 파싱할 수 없습니다 (%s)", value)
        return None


def parse_timezone(value: Optional[str]) -> Optional[ZoneInfo]:
    if value is None or not value.strip():
        return None
    try:
        return ZoneInfo(value.strip())
    except Exception:
        logger.warning("타임존을 파싱할 수 없습니다 (%s)", value)
        return None


# --------------------------------------------------------------------------- #
# 알림 상태 관리
# --------------------------------------------------------------------------- #

class AlertState:
    def __init__(self, path: Path, cooldown_minutes: int):
        self.path = path
        self.cooldown_minutes = cooldown_minutes
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.state: Dict[str, Dict[str, Dict[str, str]]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, dict):
                self.state = data
        except Exception as exc:
            logger.warning("알림 상태 파일 읽기 실패 (%s): %s", self.path, exc)

    def _save(self) -> None:
        ensure_directory(self.path)
        try:
            with self.path.open("w", encoding="utf-8") as fp:
                json.dump(self.state, fp, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error("알림 상태 파일 저장 실패 (%s): %s", self.path, exc)

    def should_notify(
        self,
        code: str,
        alert_type: str,
        cooldown_seconds: Optional[int] = None,
        once_per_day: bool = False,
    ) -> bool:
        code_state = self.state.get(code, {})
        alert_info = code_state.get(alert_type)
        if not alert_info:
            return True

        timestamp_str = alert_info.get("timestamp")
        sent_at = parse_timestamp(timestamp_str) if timestamp_str else None
        if not sent_at:
            return True

        now = datetime.utcnow()

        if once_per_day:
            if sent_at.date() == now.date():
                return False
            else:
                return True

        cooldown = self.cooldown
        if cooldown_seconds is not None and cooldown_seconds > 0:
            cooldown = timedelta(seconds=cooldown_seconds)

        if now - sent_at >= cooldown:
            return True

        return False

    def mark_sent(self, code: str, alert_type: str, context: Optional[Dict[str, str]] = None) -> None:
        if code not in self.state:
            self.state[code] = {}
        payload = {"timestamp": current_timestamp()}
        if context:
            payload.update(context)
        self.state[code][alert_type] = payload
        self._save()


# --------------------------------------------------------------------------- #
# 알림 메시지 및 감시 규칙 정의
# --------------------------------------------------------------------------- #


@dataclass
class PendingAlert:
    alert_type: str
    message: str
    cooldown_seconds: Optional[int] = None
    once_per_day: bool = False
    context: Optional[Dict[str, Any]] = None


@dataclass
class WatchRule:
    alias: str
    code: str
    label: str
    low_price: Optional[float] = None
    high_price: Optional[float] = None
    breakout_price: Optional[float] = None
    price_buffer_pct: float = 0.0
    volume_mult: Optional[float] = None
    volume_min: Optional[float] = None
    timeframe_min: Optional[int] = None
    bull_candles: Optional[int] = None
    allow_wick: bool = True
    stop_loss: Optional[float] = None
    stop_warn_pct: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    alert_cooldown_sec: Optional[int] = None
    once_per_day: bool = False
    active_from: Optional[dt_time] = None
    active_to: Optional[dt_time] = None
    active_timezone: Optional[ZoneInfo] = None


def is_rule_active(rule: WatchRule, fallback_timezone: Optional[ZoneInfo]) -> bool:
    if rule.active_from is None and rule.active_to is None:
        return True

    tz = rule.active_timezone or fallback_timezone
    try:
        now = datetime.now(tz) if tz else datetime.now()
    except Exception:
        now = datetime.now()

    start = rule.active_from or dt_time(0, 0)
    end = rule.active_to or dt_time(23, 59, 59)
    current_time = now.time()

    if start <= end:
        return start <= current_time <= end
    return current_time >= start or current_time <= end


def get_intraday_frame(
    code: str,
    timeframe_min: int,
    cache: Dict[Tuple[str, int], Optional[pd.DataFrame]],
) -> Optional[pd.DataFrame]:
    key = (code, timeframe_min)
    if key not in cache:
        df = fetch_intraday_data(code, timeframe_min)
        if df is not None and not df.empty:
            try:
                df = df.sort_index()
            except Exception:
                pass
        cache[key] = df
    return cache[key]


def evaluate_intraday_requirements(
    rule: WatchRule,
    cache: Dict[Tuple[str, int], Optional[pd.DataFrame]],
) -> Tuple[bool, Dict[str, Any]]:
    requires_intraday = any(
        [
            rule.timeframe_min,
            rule.bull_candles,
            rule.volume_mult,
            rule.volume_min,
        ]
    )

    if not requires_intraday:
        return True, {}

    timeframe_min = rule.timeframe_min or 1
    df = get_intraday_frame(rule.code, timeframe_min, cache)
    if df is None or df.empty:
        logger.debug("[%s] 분봉 데이터를 가져올 수 없어 조건을 건너뜁니다.", rule.code)
        return False, {}

    if len(df) < max(rule.bull_candles or 0, 2):
        logger.debug("[%s] 분봉 데이터가 충분하지 않습니다. (필요: %s)", rule.code, rule.bull_candles)
        return False, {}

    df = df.dropna(subset=["open", "close", "high", "low", "volume"])
    if df.empty:
        return False, {}

    recent = df.tail(max(rule.bull_candles or 1, 1))
    last_row = recent.iloc[-1]

    info: Dict[str, Any] = {
        "timeframe_min": timeframe_min,
        "last_volume": float(last_row["volume"]),
        "last_close": float(last_row["close"]),
    }

    if rule.bull_candles:
        bull_df = df.tail(rule.bull_candles)
        bull_condition = (bull_df["close"] >= bull_df["open"]).all()
        if rule.allow_wick is False:
            bodies = (bull_df["close"] - bull_df["open"]).abs()
            upper_wicks = bull_df["high"] - bull_df["close"]
            wick_condition = (upper_wicks <= bodies).all()
        else:
            wick_condition = True

        if not (bull_condition and wick_condition):
            return False, info

        info["bull_candles"] = rule.bull_candles

    if rule.volume_mult:
        history = df.iloc[:-1]
        history_count = min(len(history), max(10, (rule.bull_candles or 0) * 2 + 5))
        if history_count > 0:
            avg_volume = history.tail(history_count)["volume"].mean()
        else:
            avg_volume = history["volume"].mean() if not history.empty else 0

        info["avg_volume"] = float(avg_volume) if avg_volume is not None else 0.0
        if avg_volume and avg_volume > 0:
            if last_row["volume"] < avg_volume * rule.volume_mult:
                return False, info

    if rule.volume_min and last_row["volume"] < rule.volume_min:
        info["volume_min"] = rule.volume_min
        return False, info

    return True, info


def build_alias_display(rule: WatchRule, ctx: StockContext) -> str:
    label = rule.label or ctx.name or rule.code
    label = label.strip()
    if ctx.code not in label:
        return f"{label} ({ctx.code})"
    return label


def evaluate_watch_rules(
    ctx: StockContext,
    rules: List[WatchRule],
    cache: Dict[Tuple[str, int], Optional[pd.DataFrame]],
    fallback_timezone: Optional[ZoneInfo],
) -> List[PendingAlert]:
    if not rules:
        return []

    alerts: List[PendingAlert] = []
    current_price = ctx.current_price

    if current_price is None:
        return alerts

    for rule in rules:
        if not is_rule_active(rule, fallback_timezone):
            continue

        display_name = build_alias_display(rule, ctx)

        intraday_ready, intraday_info = evaluate_intraday_requirements(rule, cache)
        intraday_required = any(
            [
                rule.timeframe_min,
                rule.bull_candles,
                rule.volume_mult,
                rule.volume_min,
            ]
        )
        if intraday_required and not intraday_ready:
            continue

        shared_kwargs = {
            "cooldown_seconds": rule.alert_cooldown_sec,
            "once_per_day": rule.once_per_day,
            "context": {
                "rule": rule.alias,
                "label": rule.label,
            },
        }

        # 사용자 정의 가격 구간
        if rule.low_price is not None or rule.high_price is not None:
            low = rule.low_price
            high = rule.high_price
            low_threshold = low * (1 - rule.price_buffer_pct / 100.0) if low else None
            high_threshold = high * (1 + rule.price_buffer_pct / 100.0) if high else None

            in_range = True
            if low_threshold is not None and current_price < low_threshold:
                in_range = False
            if high_threshold is not None and current_price > high_threshold:
                in_range = False

            if in_range:
                parts = [
                    f"🎯 *{display_name}* 맞춤 구간 도달",
                    f"- 현재가: {format_price(current_price, ctx.is_us)}",
                ]
                if low and high:
                    parts.append(
                        f"- 목표 구간: {format_price(low, ctx.is_us)} ~ {format_price(high, ctx.is_us)}"
                    )
                elif low:
                    parts.append(f"- 하단 감시가: {format_price(low, ctx.is_us)}")
                elif high:
                    parts.append(f"- 상단 감시가: {format_price(high, ctx.is_us)}")

                if intraday_info:
                    timeframe = intraday_info.get("timeframe_min") or rule.timeframe_min or "?"
                    bull = rule.bull_candles or "-"
                    parts.append(
                        f"- 분봉 조건: {timeframe}분봉, 양봉 {bull}개"
                    )
                    if intraday_info.get("avg_volume"):
                        parts.append(
                            f"- 거래량: {format_integer(intraday_info.get('last_volume'))} (평균 {format_integer(intraday_info.get('avg_volume'))})"
                        )
                alerts.append(
                    PendingAlert(
                        alert_type=f"{rule.alias}_price_band",
                        message="\n".join(parts),
                        **shared_kwargs,
                    )
                )

        # 브레이크아웃
        if rule.breakout_price is not None:
            breakout_threshold = rule.breakout_price * (1 - rule.price_buffer_pct / 100.0)
            if current_price >= breakout_threshold:
                parts = [
                    f"🚀 *{display_name}* 돌파 감지",
                    f"- 현재가: {format_price(current_price, ctx.is_us)}",
                    f"- 돌파 기준가: {format_price(rule.breakout_price, ctx.is_us)}",
                ]
                if intraday_info:
                    timeframe = intraday_info.get("timeframe_min") or rule.timeframe_min or "?"
                    parts.append(
                        f"- 분봉 조건: {timeframe}분봉, 거래량 {format_integer(intraday_info.get('last_volume'))}"
                    )
                alerts.append(
                    PendingAlert(
                        alert_type=f"{rule.alias}_breakout",
                        message="\n".join(parts),
                        **shared_kwargs,
                    )
                )

        # 손절/경고
        if rule.stop_loss is not None:
            if current_price <= rule.stop_loss:
                parts = [
                    f"🛑 *{display_name}* 손절가 이탈",
                    f"- 현재가: {format_price(current_price, ctx.is_us)}",
                    f"- 손절가: {format_price(rule.stop_loss, ctx.is_us)}",
                ]
                alerts.append(
                    PendingAlert(
                        alert_type=f"{rule.alias}_stop_loss",
                        message="\n".join(parts),
                        **shared_kwargs,
                    )
                )
            elif rule.stop_warn_pct:
                warn_threshold = rule.stop_loss * (1 + rule.stop_warn_pct / 100.0)
                if current_price <= warn_threshold:
                    parts = [
                        f"⚠️ *{display_name}* 손절가 근접",
                        f"- 현재가: {format_price(current_price, ctx.is_us)}",
                        f"- 손절가: {format_price(rule.stop_loss, ctx.is_us)}",
                        f"- 경고 범위: {rule.stop_warn_pct:.2f}%",
                    ]
                    alerts.append(
                        PendingAlert(
                            alert_type=f"{rule.alias}_stop_warn",
                            message="\n".join(parts),
                            **shared_kwargs,
                        )
                    )

        # 익절
        if rule.take_profit_1 and current_price >= rule.take_profit_1:
            parts = [
                f"🏁 *{display_name}* 1차 목표 달성",
                f"- 현재가: {format_price(current_price, ctx.is_us)}",
                f"- 1차 목표가: {format_price(rule.take_profit_1, ctx.is_us)}",
            ]
            alerts.append(
                PendingAlert(
                    alert_type=f"{rule.alias}_take_profit_1",
                    message="\n".join(parts),
                    **shared_kwargs,
                )
            )

        if rule.take_profit_2 and current_price >= rule.take_profit_2:
            parts = [
                f"🏁 *{display_name}* 2차 목표 달성",
                f"- 현재가: {format_price(current_price, ctx.is_us)}",
                f"- 2차 목표가: {format_price(rule.take_profit_2, ctx.is_us)}",
            ]
            alerts.append(
                PendingAlert(
                    alert_type=f"{rule.alias}_take_profit_2",
                    message="\n".join(parts),
                    **shared_kwargs,
                )
            )

    return alerts

# --------------------------------------------------------------------------- #
# 텔레그램 연동
# --------------------------------------------------------------------------- #

def send_telegram_message(token: Optional[str], chat_id: Optional[str], text: str) -> bool:
    if not token or not chat_id:
        logger.info("텔레그램 토큰 또는 채팅 ID가 설정되지 않아 알림을 건너뜁니다.")
        return False

    endpoint = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(endpoint, data=payload, timeout=10)
        if response.status_code != 200:
            logger.error("텔레그램 전송 실패 (%s): %s", response.status_code, response.text)
            return False
        logger.info("텔레그램 알림 전송 완료")
        return True
    except requests.RequestException as exc:
        logger.error("텔레그램 전송 중 예외 발생: %s", exc)
        return False


# --------------------------------------------------------------------------- #
# 신호 계산
# --------------------------------------------------------------------------- #

class StockContext:
    def __init__(
        self,
        code: str,
        name: str,
        is_us: bool,
        current_price: float,
        signals: Optional[dict],
        stop_loss: Optional[dict],
        take_profit: Optional[dict],
        pattern_info: dict,
        recovery_signal: Optional[dict],
        buy_price: Optional[float],
    ):
        self.code = code
        self.name = name
        self.is_us = is_us
        self.current_price = current_price
        self.signals = signals or {}
        self.stop_loss = stop_loss or {}
        self.take_profit = take_profit or {}
        self.pattern_info = pattern_info or {}
        self.recovery_signal = recovery_signal or {}
        self.buy_price = buy_price


def fetch_stock_context(code: str, positions: Dict[str, Dict[str, float]]) -> Optional[StockContext]:
    try:
        logger.info("===== [%s] 데이터 수집 시작 =====", code)
        stock_info = fetch_us_stock_data(code) if is_us_stock(code) else fetch_korean_stock_data(code)
        if not stock_info:
            logger.warning("[%s] 데이터를 가져오지 못했습니다.", code)
            return None

        price_df = stock_info.get("price_data")
        current_price = stock_info.get("current_price")
        stock_name = stock_info.get("name", code)
        is_us = is_us_stock(code)

        if price_df is not None:
            price_df = fetch_technical_indicators(price_df)

        investor_data = None
        if not is_us:
            investor_data = fetch_investor_trading_data(code)

        pattern_info = analyze_investor_pattern(investor_data, price_df, days=5)
        signals = None
        if price_df is not None:
            signals = generate_buy_signals(price_df, pattern_info, current_price)

        # 보유 포지션 정보
        position = positions.get(code)
        buy_price = position["buy_price"] if position else None

        # 손절 및 익절 계산
        stop_loss = None
        take_profit = None
        if price_df is not None:
            reference_buy_price = buy_price
            if reference_buy_price is None and signals and signals.get("buy_1"):
                reference_buy_price = signals["buy_1"]["price"]

            if reference_buy_price is not None:
                stop_loss = calculate_stop_loss(price_df, reference_buy_price, pattern_info)
                take_profit = generate_sell_signals(price_df, pattern_info, current_price, reference_buy_price)

        recovery_signal = detect_recovery_signal(investor_data, price_df) if investor_data is not None else None

        return StockContext(
            code=code,
            name=stock_name,
            is_us=is_us,
            current_price=current_price,
            signals=signals,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pattern_info=pattern_info,
            recovery_signal=recovery_signal,
            buy_price=buy_price,
        )
    except Exception as exc:
        logger.exception("[%s] 데이터 처리 중 오류 발생: %s", code, exc)
        return None


def evaluate_alerts(
    ctx: StockContext,
    entry_tolerance_pct: float,
    stop_loss_tolerance_pct: float,
    take_profit_tolerance_pct: float,
) -> List[PendingAlert]:
    """기존 전략 기반 알림 생성"""
    alerts: List[PendingAlert] = []
    current_price = ctx.current_price

    if current_price is None:
        return alerts

    pattern_type = ctx.pattern_info.get("pattern_type", "불명확")
    confidence = ctx.pattern_info.get("confidence", 0)
    pattern_summary = f"{pattern_type} (신뢰도 {confidence}%)"

    buy_1 = ctx.signals.get("buy_1") if ctx.signals else None
    buy_2 = ctx.signals.get("buy_2") if ctx.signals else None

    # 1차 매수 진입
    if buy_1 and buy_1.get("price"):
        threshold = buy_1["price"] * (1 + entry_tolerance_pct / 100.0)
        if current_price <= threshold:
            message = (
                f"🟢 *{ctx.name} ({ctx.code})* 진입 신호\n"
                f"- 현재가: {format_price(current_price, ctx.is_us)}\n"
                f"- 1차 매수 목표가: {format_price(buy_1['price'], ctx.is_us)}\n"
                f"- 사유: {buy_1.get('reason', '목표가 근접')}\n"
                f"- 패턴: {pattern_summary}"
            )
            alerts.append(PendingAlert("entry_buy1", message))

    # 2차 매수 진입
    if buy_2 and buy_2.get("price"):
        threshold = buy_2["price"] * (1 + entry_tolerance_pct / 100.0)
        if current_price <= threshold:
            message = (
                f"🟢 *{ctx.name} ({ctx.code})* 2차 매수 구간 도달\n"
                f"- 현재가: {format_price(current_price, ctx.is_us)}\n"
                f"- 2차 매수 목표가: {format_price(buy_2['price'], ctx.is_us)}\n"
                f"- 사유: {buy_2.get('reason', '목표가 근접')}\n"
                f"- 패턴: {pattern_summary}"
            )
            alerts.append(PendingAlert("entry_buy2", message))

    # 손절 라인
    if ctx.stop_loss and ctx.stop_loss.get("stop_loss"):
        stop_loss_price = ctx.stop_loss["stop_loss"]
        threshold = stop_loss_price * (1 + stop_loss_tolerance_pct / 100.0)
        if current_price <= threshold:
            message = (
                f"🔴 *{ctx.name} ({ctx.code})* 손절 라인 이탈 경고\n"
                f"- 현재가: {format_price(current_price, ctx.is_us)}\n"
                f"- 손절가: {format_price(stop_loss_price, ctx.is_us)}\n"
                f"- 손실률: {ctx.stop_loss.get('loss_pct', 0):.1f}%\n"
                f"- 이유: {ctx.stop_loss.get('reason', '')}"
            )
            alerts.append(PendingAlert("stop_loss", message))

    # 익절 라인
    if ctx.take_profit:
        tp1 = ctx.take_profit.get("take_profit_1")
        tp2 = ctx.take_profit.get("take_profit_2")
        reason = ctx.take_profit.get("reason", "")

        if tp1:
            threshold = tp1 * (1 - take_profit_tolerance_pct / 100.0)
            if current_price >= threshold:
                message = (
                    f"🏁 *{ctx.name} ({ctx.code})* 1차 익절 도달\n"
                    f"- 현재가: {format_price(current_price, ctx.is_us)}\n"
                    f"- 1차 익절 목표가: {format_price(tp1, ctx.is_us)}\n"
                    f"- 이유: {reason}"
                )
                alerts.append(PendingAlert("take_profit_1", message))

        if tp2:
            threshold = tp2 * (1 - take_profit_tolerance_pct / 100.0)
            if current_price >= threshold:
                message = (
                    f"🏁 *{ctx.name} ({ctx.code})* 2차 익절 도달\n"
                    f"- 현재가: {format_price(current_price, ctx.is_us)}\n"
                    f"- 2차 익절 목표가: {format_price(tp2, ctx.is_us)}\n"
                    f"- 이유: {reason}"
                )
                alerts.append(PendingAlert("take_profit_2", message))

    # 회복 신호
    if ctx.recovery_signal and ctx.recovery_signal.get("has_recovery_signal"):
        message = (
            f"🟢 *{ctx.name} ({ctx.code})* 회복 신호 감지\n"
            f"- 내용: {ctx.recovery_signal.get('message', '')}\n"
            f"- 패턴: {pattern_summary}"
        )
        alerts.append(PendingAlert("recovery_signal", message))

    return alerts


# --------------------------------------------------------------------------- #
# 런타임 설정 & 메인 루프
# --------------------------------------------------------------------------- #

@dataclass
class RuntimeConfig:
    stock_codes: List[str]
    positions: Dict[str, Dict[str, float]]
    telegram_token: Optional[str]
    telegram_chat_id: Optional[str]
    check_interval: int
    cooldown_minutes: int
    entry_tolerance_pct: float
    stop_loss_tolerance_pct: float
    take_profit_tolerance_pct: float
    state_path: Path
    watch_rules: List[WatchRule]
    market_timezone: Optional[ZoneInfo]
    run_once: bool


def load_runtime_config() -> RuntimeConfig:
    refresh_environment()

    stock_codes = parse_stock_codes(os.getenv("STOCK_CODES"))
    positions = parse_positions(os.getenv("POSITIONS"))

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    check_interval = max(30, int(os.getenv("CHECK_INTERVAL_SECONDS", "1800")))
    cooldown_minutes = max(1, int(os.getenv("ALERT_COOLDOWN_MINUTES", "60")))
    entry_tolerance_pct = float(os.getenv("ENTRY_TOLERANCE_PCT", "1.0"))
    stop_loss_tolerance_pct = float(os.getenv("STOP_LOSS_TOLERANCE_PCT", "0.0"))
    take_profit_tolerance_pct = float(os.getenv("TAKE_PROFIT_TOLERANCE_PCT", "0.0"))

    state_path = resolve_path(os.getenv("ALERT_STATE_PATH", str(DEFAULT_STATE_PATH)))

    market_timezone = parse_timezone(os.getenv("MARKET_TIMEZONE"))

    watch_rules: List[WatchRule] = []
    default_price_buffer = get_env_float("PRICE_BUFFER_PCT") or 0.0
    default_volume_mult = get_env_float("VOLUME_MULT")
    default_volume_min = get_env_float("VOLUME_MIN")

    watch_symbol_entries = parse_watch_symbols(os.getenv("WATCH_SYMBOLS"))
    for alias_raw, code in watch_symbol_entries:
        alias = alias_raw or code
        prefix = sanitize_alias(alias)
        if not prefix:
            continue

        label = os.getenv(f"{prefix}_LABEL") or alias or code
        low_price = get_env_float(f"{prefix}_LOW_PRICE")
        high_price = get_env_float(f"{prefix}_HIGH_PRICE")
        breakout_price = get_env_float(f"{prefix}_BREAKOUT_PRICE")

        price_buffer_pct = get_env_float(f"{prefix}_PRICE_BUFFER_PCT")
        if price_buffer_pct is None:
            price_buffer_pct = default_price_buffer
        volume_mult = get_env_float(f"{prefix}_VOLUME_MULT")
        if volume_mult is None:
            volume_mult = default_volume_mult
        volume_min = get_env_float(f"{prefix}_VOLUME_MIN")
        if volume_min is None:
            volume_min = default_volume_min

        timeframe_min = get_env_int(f"{prefix}_TIMEFRAME_MIN")
        bull_candles = get_env_int(f"{prefix}_BULL_CANDLES")
        allow_wick = parse_bool(os.getenv(f"{prefix}_ALLOW_WICK"), default=True)

        stop_loss = get_env_float(f"{prefix}_STOP_LOSS")
        stop_warn_pct = get_env_float(f"{prefix}_STOP_WARN_PCT")
        take_profit_1 = get_env_float(f"{prefix}_TAKE_PROFIT_1")
        take_profit_2 = get_env_float(f"{prefix}_TAKE_PROFIT_2")

        alert_cooldown_sec = get_env_int(f"{prefix}_ALERT_COOLDOWN_SEC")
        once_per_day = parse_bool(os.getenv(f"{prefix}_ONCE_PER_DAY"))

        active_from = parse_time_string(os.getenv(f"{prefix}_ACTIVE_FROM"))
        active_to = parse_time_string(os.getenv(f"{prefix}_ACTIVE_TO"))
        active_timezone = parse_timezone(os.getenv(f"{prefix}_ACTIVE_TZ")) or market_timezone

        rule = WatchRule(
            alias=prefix,
            code=code,
            label=label,
            low_price=low_price,
            high_price=high_price,
            breakout_price=breakout_price,
            price_buffer_pct=price_buffer_pct or 0.0,
            volume_mult=volume_mult,
            volume_min=volume_min,
            timeframe_min=timeframe_min,
            bull_candles=bull_candles,
            allow_wick=allow_wick,
            stop_loss=stop_loss,
            stop_warn_pct=stop_warn_pct,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            alert_cooldown_sec=alert_cooldown_sec,
            once_per_day=once_per_day,
            active_from=active_from,
            active_to=active_to,
            active_timezone=active_timezone,
        )

        watch_rules.append(rule)

    if watch_rules:
        existing_codes = {code for code in stock_codes}
        for rule in watch_rules:
            if rule.code not in existing_codes:
                stock_codes.append(rule.code)
                existing_codes.add(rule.code)

    run_once_raw = os.getenv("RUN_ONCE", "")
    run_once = run_once_raw.lower() in {"1", "true", "yes"}

    return RuntimeConfig(
        stock_codes=stock_codes,
        positions=positions,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        check_interval=check_interval,
        cooldown_minutes=cooldown_minutes,
        entry_tolerance_pct=entry_tolerance_pct,
        stop_loss_tolerance_pct=stop_loss_tolerance_pct,
        take_profit_tolerance_pct=take_profit_tolerance_pct,
        state_path=state_path,
        watch_rules=watch_rules,
        market_timezone=market_timezone,
        run_once=run_once,
    )


def run_cycle(config: RuntimeConfig, state: AlertState) -> None:
    if not config.stock_codes:
        logger.warning("감시할 종목(STOCK_CODES)이 설정되지 않았습니다.")
        return

    rules_by_code: Dict[str, List[WatchRule]] = {}
    for rule in config.watch_rules:
        rules_by_code.setdefault(rule.code, []).append(rule)

    intraday_cache: Dict[Tuple[str, int], Optional[pd.DataFrame]] = {}

    for code in config.stock_codes:
        ctx = fetch_stock_context(code, config.positions)
        if ctx is None:
            continue

        alerts = evaluate_alerts(
            ctx,
            entry_tolerance_pct=config.entry_tolerance_pct,
            stop_loss_tolerance_pct=config.stop_loss_tolerance_pct,
            take_profit_tolerance_pct=config.take_profit_tolerance_pct,
        )

        custom_alerts = evaluate_watch_rules(
            ctx,
            rules_by_code.get(code, []),
            intraday_cache,
            config.market_timezone,
        )

        if custom_alerts:
            alerts.extend(custom_alerts)

        if not alerts:
            logger.info("[%s] 전송할 알림이 없습니다.", code)
            continue

        for alert in alerts:
            if not state.should_notify(
                code,
                alert.alert_type,
                cooldown_seconds=alert.cooldown_seconds,
                once_per_day=alert.once_per_day,
            ):
                logger.debug("[%s][%s] 조건은 충족했으나 쿨다운/일일 제한으로 건너뜀", code, alert.alert_type)
                continue

            logger.info("[%s][%s] 알림 전송 준비", code, alert.alert_type)
            sent = send_telegram_message(config.telegram_token, config.telegram_chat_id, alert.message)
            if sent:
                context = alert.context or {}
                context.setdefault("message_preview", alert.message[:80])
                state.mark_sent(code, alert.alert_type, context)
            else:
                logger.warning("[%s][%s] 텔레그램 전송 실패 – 상태는 갱신하지 않습니다.", code, alert.alert_type)


def main() -> None:
    state: Optional[AlertState] = None
    state_signature: Optional[Tuple[str, int]] = None

    while True:
        config = load_runtime_config()

        desired_signature = (str(config.state_path), config.cooldown_minutes)
        if state is None or state_signature != desired_signature:
            logger.info("알림 상태 관리자 초기화 (path=%s, cooldown=%s분)", config.state_path, config.cooldown_minutes)
            state = AlertState(config.state_path, cooldown_minutes=config.cooldown_minutes)
            state_signature = desired_signature

        run_cycle(config, state)

        if config.run_once:
            logger.info("RUN_ONCE 설정으로 1회 실행 후 종료합니다.")
            break

        sleep_seconds = max(config.check_interval, 30)
        logger.info("다음 감시까지 %s초 대기합니다.", sleep_seconds)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("사용자 요청으로 종료합니다.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("signal_watcher 실행 중 치명적 오류: %s", exc)
        sys.exit(1)

