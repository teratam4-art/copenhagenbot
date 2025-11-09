#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render 배포용 실시간 신호 감시 스크립트

기능
------
1. 지정한 종목들의 현재 시세와 수급 데이터를 조회
2. 매수/손절/익절 조건 충족 여부 판단
3. 조건 충족 시 텔레그램 알림 전송 (중복 방지 및 쿨다운 지원)

환경 변수
---------
STOCK_CODES               : 감시할 종목 코드 목록 (쉼표 구분, 예: "005930,035720,AAPL")
POSITIONS                 : 보유 종목 평균단가 (예: "005930=72000,AAPL=182.5@50")
TELEGRAM_BOT_TOKEN        : 텔레그램 봇 토큰
TELEGRAM_CHAT_ID          : 텔레그램 채팅 ID
CHECK_INTERVAL_SECONDS    : 감시 주기(초). 기본 1800 (30분)
ALERT_COOLDOWN_MINUTES    : 동일 알림 재발송 최소 간격(분). 기본 60
ENTRY_TOLERANCE_PCT       : 매수 목표가 대비 허용 오차(%) 기본 1.0
STOP_LOSS_TOLERANCE_PCT   : 손절 라인 초과 허용 폭(%) 기본 0
TAKE_PROFIT_TOLERANCE_PCT : 익절 라인 허용 오차(%) 기본 0
ALERT_STATE_PATH          : 알림 상태 저장 파일 경로. 기본 outputs/txt/alert_state.json
RUN_ONCE                  : "1"이면 1회 실행 후 종료

Render에서의 사용
-----------------
Start Command 예시:
    python signal_watcher.py
또는 개발/테스트용으로:
    RUN_ONCE=1 python signal_watcher.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from data_fetcher import (
    fetch_investor_trading_data,
    fetch_korean_stock_data,
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


def current_timestamp() -> str:
    return datetime.utcnow().isoformat()


def parse_timestamp(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 알림 상태 관리
# --------------------------------------------------------------------------- #

class AlertState:
    def __init__(self, path: Path, cooldown_minutes: int):
        self.path = path
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

    def should_notify(self, code: str, alert_type: str) -> bool:
        code_state = self.state.get(code, {})
        alert_info = code_state.get(alert_type)
        if not alert_info:
            return True

        timestamp_str = alert_info.get("timestamp")
        sent_at = parse_timestamp(timestamp_str) if timestamp_str else None
        if not sent_at:
            return True

        if datetime.utcnow() - sent_at >= self.cooldown:
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
# 텔레그램 연동
# --------------------------------------------------------------------------- #

def send_telegram_message(token: str, chat_id: str, text: str) -> bool:
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
) -> List[Tuple[str, str]]:
    """조건을 만족하는 알림 목록 반환 -> (alert_type, message)"""
    alerts: List[Tuple[str, str]] = []
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
            alerts.append(("entry_buy1", message))

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
            alerts.append(("entry_buy2", message))

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
            alerts.append(("stop_loss", message))

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
                alerts.append(("take_profit_1", message))

        if tp2:
            threshold = tp2 * (1 - take_profit_tolerance_pct / 100.0)
            if current_price >= threshold:
                message = (
                    f"🏁 *{ctx.name} ({ctx.code})* 2차 익절 도달\n"
                    f"- 현재가: {format_price(current_price, ctx.is_us)}\n"
                    f"- 2차 익절 목표가: {format_price(tp2, ctx.is_us)}\n"
                    f"- 이유: {reason}"
                )
                alerts.append(("take_profit_2", message))

    # 회복 신호
    if ctx.recovery_signal and ctx.recovery_signal.get("has_recovery_signal"):
        message = (
            f"🟢 *{ctx.name} ({ctx.code})* 회복 신호 감지\n"
            f"- 내용: {ctx.recovery_signal.get('message', '')}\n"
            f"- 패턴: {pattern_summary}"
        )
        alerts.append(("recovery_signal", message))

    return alerts


# --------------------------------------------------------------------------- #
# 메인 루프
# --------------------------------------------------------------------------- #

def run_cycle(
    codes: List[str],
    positions: Dict[str, Dict[str, float]],
    state: AlertState,
    telegram_token: Optional[str],
    telegram_chat_id: Optional[str],
    entry_tolerance_pct: float,
    stop_loss_tolerance_pct: float,
    take_profit_tolerance_pct: float,
) -> None:
    if not codes:
        logger.warning("감시할 종목(STOCK_CODES)이 설정되지 않았습니다.")
        return

    for code in codes:
        ctx = fetch_stock_context(code, positions)
        if ctx is None:
            continue

        alerts = evaluate_alerts(
            ctx,
            entry_tolerance_pct=entry_tolerance_pct,
            stop_loss_tolerance_pct=stop_loss_tolerance_pct,
            take_profit_tolerance_pct=take_profit_tolerance_pct,
        )

        if not alerts:
            logger.info("[%s] 전송할 알림이 없습니다.", code)
            continue

        for alert_type, message in alerts:
            if not state.should_notify(code, alert_type):
                logger.debug("[%s][%s] 쿨다운 미충족으로 알림 건너뜀", code, alert_type)
                continue

            logger.info("[%s][%s] 알림 전송 준비", code, alert_type)
            sent = send_telegram_message(telegram_token, telegram_chat_id, message)
            if sent:
                state.mark_sent(code, alert_type, {"message_preview": message[:80]})
            else:
                logger.warning("[%s][%s] 텔레그램 전송 실패 – 상태는 갱신하지 않습니다.", code, alert_type)


def main() -> None:
    stock_codes = parse_stock_codes(os.getenv("STOCK_CODES"))
    positions = parse_positions(os.getenv("POSITIONS"))

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    check_interval = int(os.getenv("CHECK_INTERVAL_SECONDS", "1800"))
    cooldown_minutes = int(os.getenv("ALERT_COOLDOWN_MINUTES", "60"))
    entry_tolerance_pct = float(os.getenv("ENTRY_TOLERANCE_PCT", "1.0"))
    stop_loss_tolerance_pct = float(os.getenv("STOP_LOSS_TOLERANCE_PCT", "0.0"))
    take_profit_tolerance_pct = float(os.getenv("TAKE_PROFIT_TOLERANCE_PCT", "0.0"))

    state_path = Path(os.getenv("ALERT_STATE_PATH", str(DEFAULT_STATE_PATH)))
    state = AlertState(state_path, cooldown_minutes=cooldown_minutes)

    run_cycle(
        codes=stock_codes,
        positions=positions,
        state=state,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        entry_tolerance_pct=entry_tolerance_pct,
        stop_loss_tolerance_pct=stop_loss_tolerance_pct,
        take_profit_tolerance_pct=take_profit_tolerance_pct,
    )

    if os.getenv("RUN_ONCE") == "1":
        logger.info("RUN_ONCE=1 설정으로 1회 실행 후 종료합니다.")
        return

    while True:
        logger.info("다음 감시까지 %s초 대기합니다.", check_interval)
        time.sleep(max(check_interval, 60))
        run_cycle(
            codes=stock_codes,
            positions=positions,
            state=state,
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id,
            entry_tolerance_pct=entry_tolerance_pct,
            stop_loss_tolerance_pct=stop_loss_tolerance_pct,
            take_profit_tolerance_pct=take_profit_tolerance_pct,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("사용자 요청으로 종료합니다.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("signal_watcher 실행 중 치명적 오류: %s", exc)
        sys.exit(1)

