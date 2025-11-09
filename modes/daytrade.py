"""
Daytrade (단타) 분석 로직.
입력 데이터는 stock_screener.py에서 전달하는 사전 형태를 사용한다.
"""

from __future__ import annotations

from typing import Dict, Any, List


STOP_LOSS_PCT = 2.0  # %


def _safe_float(value):
    if value is None:
        return None
    try:
        if hasattr(value, "__float__"):
            return float(value)
    except Exception:
        return None
    try:
        return float(value)
    except Exception:
        return None


def analyze(data: Dict[str, Any]) -> Dict[str, Any]:
    symbol = data.get("symbol")
    name = data.get("name", "")
    price = _safe_float(data.get("current_price"))
    ma5 = _safe_float(data.get("ma5"))
    rsi = _safe_float(data.get("rsi"))
    volume_ratio = _safe_float(data.get("volume_ratio"))

    reasons: List[str] = []
    entry_signal = False
    exit_signal = False

    near_ma5 = False
    if price and ma5:
        near_ma5 = abs(price - ma5) / ma5 <= 0.012  # 약 ±1.2%
        if near_ma5:
            reasons.append("MA5 근처 눌림")

    vol_ok = volume_ratio is not None and volume_ratio >= 2.0
    if vol_ok:
        reasons.append(f"거래량 {volume_ratio:.1f}배")

    rsi_ok = rsi is not None and 35 <= rsi <= 45
    if rsi_ok:
        reasons.append(f"RSI {rsi:.1f}")

    entry_signal = near_ma5 and vol_ok and rsi_ok

    exit_reasons: List[str] = []
    if rsi is not None and rsi >= 70:
        exit_signal = True
        exit_reasons.append("RSI 과열")
    if price and ma5 and price < ma5:
        exit_signal = True
        exit_reasons.append("MA5 하향 이탈")

    if exit_signal and exit_reasons:
        reason_text = ", ".join(exit_reasons)
        status = "단타 매도 타이밍 임박"
        recommendation = "익절 또는 일부 청산 권장"
    elif entry_signal and reasons:
        reason_text = ", ".join(reasons)
        status = "단타 매수 후보"
        recommendation = "눌림 직후 단기 반등 노리기"
    else:
        combined = exit_reasons or reasons
        reason_text = ", ".join(combined) if combined else "조건 미충족"
        status = "단타 관망 구간"
        recommendation = "명확한 신호 대기"

    stop_loss_price = price * (1 - STOP_LOSS_PCT / 100) if price else None
    summary = reason_text if reason_text else status

    return {
        "mode": "daytrade",
        "symbol": symbol,
        "name": name,
        "entry_signal": entry_signal,
        "exit_signal": exit_signal,
        "status": status,
        "reason": reason_text,
        "stop_loss_pct": STOP_LOSS_PCT,
        "stop_loss_price": stop_loss_price,
        "summary": summary,
        "recommendation": recommendation,
    }




