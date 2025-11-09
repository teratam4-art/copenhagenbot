"""
Swing (스윙) 분석 로직.
"""

from __future__ import annotations

from typing import Dict, Any, List


STOP_LOSS_PCT = 4.0  # %


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
    ma20 = _safe_float(data.get("ma20"))
    ma60 = _safe_float(data.get("ma60"))
    ma20_slope = _safe_float(data.get("ma20_slope"))
    rsi = _safe_float(data.get("rsi"))
    volume_ratio = _safe_float(data.get("volume_ratio"))
    macd = _safe_float(data.get("macd"))
    macd_signal = _safe_float(data.get("macd_signal"))

    reasons: List[str] = []
    entry_signal = False
    exit_signal = False

    near_ma20 = False
    if price and ma20:
        near_ma20 = abs(price - ma20) / ma20 <= 0.02  # ±2%
        if near_ma20:
            reasons.append("MA20 지지 확인")

    slope_positive = ma20_slope is not None and ma20_slope > 0
    if slope_positive:
        reasons.append("MA20 상승 기울기 유지")

    macd_bullish = macd is not None and macd_signal is not None and macd > macd_signal
    if macd_bullish:
        reasons.append("MACD 골든크로스")

    rsi_ok = rsi is not None and 40 <= rsi <= 50
    if rsi_ok:
        reasons.append(f"RSI {rsi:.1f}")

    volume_ok = volume_ratio is not None and volume_ratio >= 1.2
    if volume_ok:
        reasons.append(f"거래량 {volume_ratio:.1f}배")

    entry_signal = near_ma20 and slope_positive and macd_bullish and rsi_ok

    exit_reasons: List[str] = []
    if rsi is not None and rsi >= 70:
        exit_signal = True
        exit_reasons.append("RSI 과열")
    if price and ma20 and price < ma20:
        exit_signal = True
        exit_reasons.append("MA20 하향 이탈")

    if exit_signal and exit_reasons:
        reason_text = ", ".join(exit_reasons)
        status = "스윙 청산 신호 감지"
        recommendation = "익절 또는 비중 축소 검토"
    elif entry_signal and reasons:
        reason_text = ", ".join(reasons)
        status = "스윙 진입 유효"
        recommendation = "MA20 지지 확인 후 분할 매수 대응"
    else:
        combined = exit_reasons or reasons
        reason_text = ", ".join(combined) if combined else "추세 확인 필요"
        status = "추세 지속 중, 보유 권장"
        recommendation = "추세 유지 시 보유, 추가 눌림 시 분할 매수 고려"

    stop_loss_price = price * (1 - STOP_LOSS_PCT / 100) if price else None
    summary = reason_text if reason_text else status

    return {
        "mode": "swing",
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




