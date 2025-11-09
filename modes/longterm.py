"""
Longterm (장기 투자) 분석 로직.
"""

from __future__ import annotations

from typing import Dict, Any, List


STOP_LOSS_PCT = 10.0  # %


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
    ma60 = _safe_float(data.get("ma60"))
    ma120 = _safe_float(data.get("ma120"))
    ma60_slope = _safe_float(data.get("ma60_slope"))
    fundamentals = data.get("fundamentals", {}) or {}

    pe = _safe_float(fundamentals.get("pe"))
    roe = _safe_float(fundamentals.get("roe"))  # already percentage
    eps = _safe_float(fundamentals.get("eps"))

    reasons: List[str] = []
    entry_signal = False
    exit_signal = False

    trend_ok = False
    if price and ma60:
        trend_ok = price >= ma60 and (ma60_slope is None or ma60_slope >= 0)
        if trend_ok:
            reasons.append("MA60 위에서 추세 유지")

    fundamentals_ok = False
    fundamentals_reasons = []
    if pe is not None and pe > 0 and pe < 15:
        fundamentals_reasons.append(f"PER {pe:.1f}")
    if roe is not None and roe > 8:
        fundamentals_reasons.append(f"ROE {roe:.1f}%")
    if eps is not None and eps > 0:
        fundamentals_reasons.append(f"EPS {eps:.2f}")

    fundamentals_ok = len(fundamentals_reasons) >= 2  # 최소 2개 만족
    if fundamentals_reasons:
        reasons.append(", ".join(fundamentals_reasons))

    entry_signal = trend_ok and fundamentals_ok

    exit_reasons: List[str] = []
    if price and ma60 and price < ma60 * 0.97:
        exit_signal = True
        exit_reasons.append("MA60 하향 이탈")
    if eps is not None and eps <= 0:
        exit_signal = True
        exit_reasons.append("EPS 적자 전환")
    if roe is not None and roe < 5:
        exit_signal = True
        exit_reasons.append("ROE 저하")

    if exit_signal and exit_reasons:
        reason_text = ", ".join(exit_reasons)
        status = "가치 경고 신호"
        recommendation = "재무 및 추세 재점검 필요"
    elif entry_signal and reasons:
        reason_text = ", ".join(reasons)
        status = "저평가 구간"
        recommendation = "장기 분할매수 적합"
    else:
        combined = exit_reasons or reasons
        reason_text = ", ".join(combined) if combined else "재무 데이터 부족"
        status = "가치 중립"
        recommendation = "기존 보유 유지, 추가 지표 모니터링"

    stop_loss_price = price * (1 - STOP_LOSS_PCT / 100) if price else None
    summary = reason_text if reason_text else status

    return {
        "mode": "longterm",
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




