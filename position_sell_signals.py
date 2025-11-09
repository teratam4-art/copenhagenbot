#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
보유 중인 종목의 평단/현재가/보유기간/기술지표를 기반으로
익절 또는 보유 판단을 도와주는 간단한 도구.

실제 가격 데이터는 사용자가 별도의 API나 데이터베이스에서 주입하면 된다.

사용 방법
---------
1. Position 객체에 현재 보유 정보(평단, 현재가, MA20, RSI 등)를 채운다.

   ```python
   my_pos = Position(
       ticker="FTNT",
       avg_price=80.0,
       current_price=86.0,
       horizon="3m",
       ma20=82.5,
       rsi=58,
       rsi_peak=72,
   )
   ```

2. `decide_sell(my_pos)`를 호출해 매도/보유 사유를 확인한다.

   ```python
   result = decide_sell(my_pos)
   print(result)

   # 여러 종목이면 리스트로 만들어 반복
   positions = [pos1, pos2, ...]
   for p in positions:
       print(p.ticker, decide_sell(p))
   ```

3. 현재가, MA20, RSI 등은 미리 구축해 둔 스크리너나 DB에서 값을 불러와 넣으면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. 보유 기간별 목표 수익률 설정
#    필요에 따라 수익률을 조정해서 사용할 수 있다.
# ---------------------------------------------------------------------------
TARGET_PROFITS: Dict[str, float] = {
    "1m": 0.05,    # 1개월 보유 시 목표 +5%
    "3m": 0.12,   # 3개월 보유 시 목표 +12%
    "6m": 0.20,   # 6개월 보유 시 목표 +20%
    "12m+": 0.30,  # 12개월 이상 보유 시 목표 +30%
}


# ---------------------------------------------------------------------------
# 2. 포지션 정보 데이터클래스
# ---------------------------------------------------------------------------
@dataclass
class Position:
    ticker: str
    avg_price: float            # 보유 평단
    current_price: float        # 현재가 (API/DB에서 주입)
    horizon: str                # "1m", "3m", "6m", "12m+"
    ma20: Optional[float] = None
    rsi: Optional[float] = None
    rsi_peak: Optional[float] = None  # 최근 고점 RSI (예: 70 이상 찍은 값)
    stop_loss_pct: float = 0.05       # 평단 대비 손절 허용폭 (기본 5%)


# ---------------------------------------------------------------------------
# 3. 판단 로직
# ---------------------------------------------------------------------------
def decide_sell(pos: Position) -> Dict[str, object]:
    """
    내 평단, 현재가, 보유기간, MA20, RSI 등을 조합해
    "팔자 / 부분익절 / 계속 보유" 여부와 사유를 반환한다.
    """

    target_profit = TARGET_PROFITS.get(pos.horizon, 0.10)
    target_price = pos.avg_price * (1 + target_profit)

    result: Dict[str, object] = {
        "ticker": pos.ticker,
        "should_sell": False,
        "partial_sell": False,
        "reason": [],
        "target_price": round(target_price, 2),
    }

    # 1) 목표 수익률 도달 여부
    if pos.current_price >= target_price:
        result["should_sell"] = True
        result["reason"].append(
            f"보유기간 {pos.horizon} 기준 목표가 {target_price:.2f} 달성"
        )

    # 2) 평단 대비 손절 라인 체크
    stop_loss_price = pos.avg_price * (1 - pos.stop_loss_pct)
    if pos.current_price <= stop_loss_price:
        result["should_sell"] = True
        result["reason"].append(
            f"평단 대비 -{pos.stop_loss_pct * 100:.1f}% 손절 라인 {stop_loss_price:.2f} 하회"
        )

    # 3) MA20 이탈 (추세 약화 시그널)
    if pos.ma20 is not None and pos.current_price < pos.ma20 * 0.98:
        result["partial_sell"] = True
        result["reason"].append("MA20 2% 이상 이탈 → 추세 약화, 부분익절/축소 고려")

    # 4) RSI 피크 이후 식는 구간 (보조 신호)
    if pos.rsi is not None:
        if pos.rsi < 60 and pos.rsi_peak and pos.rsi_peak >= 70:
            result["partial_sell"] = True
            result["reason"].append(
                "RSI가 과열권(≥70) 이후 60 아래 → 단기 피크 가능성, 일부 익절 유리"
            )
        elif pos.rsi >= 70:
            result["reason"].append("RSI 70 이상 → 과열 구간, 분할 익절 고려")

    # 5) 최종 판단 보조
    if not result["reason"]:
        result["reason"].append("목표 미달 & 추세 유지 → 보유")

    return result


# ---------------------------------------------------------------------------
# 4. 예시 실행 (직접 실행 시)
# ---------------------------------------------------------------------------
def _example() -> None:
    positions: List[Position] = [
        Position(
            ticker="FTNT",
            avg_price=80.0,
            current_price=86.0,
            horizon="3m",
            ma20=82.5,
            rsi=58,
            rsi_peak=72,
        ),
        Position(
            ticker="DDOG",
            avg_price=145.0,
            current_price=154.0,
            horizon="1m",
            ma20=150.0,
            rsi=62,
            rsi_peak=68,
        ),
        Position(
            ticker="MRVL",
            avg_price=88.0,
            current_price=82.0,
            horizon="6m",
            ma20=87.7,
            rsi=55,
            rsi_peak=75,
        ),
    ]

    for pos in positions:
        decision = decide_sell(pos)
        print("-" * 60)
        print(f"[{pos.ticker}] 결과")
        print(f"  목표가: {decision['target_price']} / 현재가: {pos.current_price:.2f}")
        print(f"  전체 매도: {decision['should_sell']} | 부분 매도: {decision['partial_sell']}")
        for reason in decision['reason']:
            print(f"   · {reason}")


if __name__ == "__main__":
    _example()

