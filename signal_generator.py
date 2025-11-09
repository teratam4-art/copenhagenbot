#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
신호 생성 모듈
매수/매도 타이밍을 계산합니다.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def generate_buy_signals(price_df, pattern_info, current_price):
    """
    매수 신호 생성 (1차, 2차)
    
    Args:
        price_df: 가격 데이터프레임
        pattern_info: 패턴 분석 결과
        current_price: 현재가
    
    Returns:
        dict: {
            'buy_1': {'price': float, 'days': int, 'reason': str},
            'buy_2': {'price': float, 'days': int, 'reason': str},
            'strategy': str
        }
    """
    if price_df is None or len(price_df) == 0:
        return None
    
    # 이동평균선 계산
    if 'MA5' not in price_df.columns:
        price_df['MA5'] = price_df['종가'].rolling(window=5).mean()
    if 'MA20' not in price_df.columns:
        price_df['MA20'] = price_df['종가'].rolling(window=20).mean()
    
    ma5 = price_df['MA5'].iloc[-1] if pd.notna(price_df['MA5'].iloc[-1]) else current_price
    ma20 = price_df['MA20'].iloc[-1] if pd.notna(price_df['MA20'].iloc[-1]) else current_price
    
    pattern_type = pattern_info.get('pattern_type', '불명확')
    
    buy_1 = None
    buy_2 = None
    strategy = ""
    
    # 물량 털기 패턴: MA5 근처 1차, MA20 근처 2차
    if pattern_type == '물량_털기':
        # 1차 매수: MA5 * 0.99 ~ MA5 (약간 아래)
        buy_1_price = ma5 * 0.99
        buy_1_days = estimate_days_to_price(price_df, buy_1_price, current_price)
        
        # 2차 매수: MA20 * 0.985 ~ MA20 (더 아래)
        buy_2_price = ma20 * 0.985
        buy_2_days = estimate_days_to_price(price_df, buy_2_price, current_price)
        
        buy_1 = {
            'price': round(buy_1_price),
            'days': buy_1_days,
            'reason': 'MA5 근처 눌림 구간 (1차 매수)'
        }
        buy_2 = {
            'price': round(buy_2_price),
            'days': buy_2_days,
            'reason': 'MA20 근처 눌림 구간 (2차 매수)'
        }
        strategy = "물량 털기 패턴: 단기 조정 후 상승 가능성이 높으므로 눌림 구간에서 매수 추천"
    
    # 상승 신호: 현재가 근처 1차, 약간 아래 2차
    elif pattern_type == '상승_신호':
        # 1차 매수: 현재가 * 0.98 (약간 아래)
        buy_1_price = current_price * 0.98
        buy_1_days = 1  # 즉시 또는 1일 내
        
        # 2차 매수: MA5 * 0.97 (더 아래)
        buy_2_price = ma5 * 0.97
        buy_2_days = estimate_days_to_price(price_df, buy_2_price, current_price)
        
        buy_1 = {
            'price': round(buy_1_price),
            'days': buy_1_days,
            'reason': '현재가 근처 (1차 매수)'
        }
        buy_2 = {
            'price': round(buy_2_price),
            'days': buy_2_days,
            'reason': 'MA5 근처 (2차 매수)'
        }
        strategy = "상승 신호: 외국인과 기관이 동시 매수 중이므로 적극 매수 추천"
    
    # 진짜 이탈: 매수 비추천
    elif pattern_type == '진짜_이탈':
        strategy = "진짜 이탈 패턴: 매수 비추천, 관망 권장"
    
    # 불명확: 보수적 접근
    else:
        # 보수적 매수 구간
        buy_1_price = ma20 * 0.97
        buy_1_days = estimate_days_to_price(price_df, buy_1_price, current_price)
        
        buy_2_price = ma20 * 0.95
        buy_2_days = estimate_days_to_price(price_df, buy_2_price, current_price)
        
        buy_1 = {
            'price': round(buy_1_price),
            'days': buy_1_days,
            'reason': 'MA20 근처 보수적 매수 (1차)'
        }
        buy_2 = {
            'price': round(buy_2_price),
            'days': buy_2_days,
            'reason': 'MA20 아래 보수적 매수 (2차)'
        }
        strategy = "불명확한 패턴: 보수적 접근 권장"
    
    return {
        'buy_1': buy_1,
        'buy_2': buy_2,
        'strategy': strategy
    }


def estimate_days_to_price(price_df, target_price, current_price):
    """
    목표 가격까지 도달 예상 일수 계산
    
    Args:
        price_df: 가격 데이터프레임
        target_price: 목표 가격
        current_price: 현재가
    
    Returns:
        int: 예상 일수
    """
    if price_df is None or len(price_df) < 5:
        return 3  # 기본값
    
    # 최근 10일 가격 변화율 평균
    recent = price_df.tail(10)
    price_changes = recent['종가'].pct_change().dropna()
    
    if len(price_changes) == 0:
        return 3
    
    avg_daily_change = abs(price_changes.mean())
    if avg_daily_change < 0.003:  # 0.3% 미만이면 최소값
        avg_daily_change = 0.003
    
    # 목표까지 필요한 변화율
    price_diff_pct = abs((target_price - current_price) / current_price)
    
    # 예상 일수 계산
    if price_diff_pct == 0:
        return 0
    
    estimated_days = int(price_diff_pct / avg_daily_change)
    
    # 최대 30일로 제한
    return min(30, max(1, estimated_days))


def generate_sell_signals(price_df, pattern_info, current_price, buy_price=None):
    """
    매도 신호 생성 (익절가)
    
    Args:
        price_df: 가격 데이터프레임
        pattern_info: 패턴 분석 결과
        current_price: 현재가
        buy_price: 평균 매수가 (없으면 현재가 기준)
    
    Returns:
        dict: {
            'take_profit_1': float,
            'take_profit_2': float,
            'reason': str
        }
    """
    if price_df is None or len(price_df) == 0:
        return None
    
    if buy_price is None:
        buy_price = current_price
    
    # RSI 확인
    if 'RSI' in price_df.columns:
        rsi = price_df['RSI'].iloc[-1]
    else:
        rsi = 50
    
    pattern_type = pattern_info.get('pattern_type', '불명확')
    
    # 상승 신호나 물량 털기: 공격적 익절
    if pattern_type in ['상승_신호', '물량_털기']:
        # 1차 익절: 매수가 대비 5-8%
        take_profit_1 = buy_price * 1.06
        # 2차 익절: 매수가 대비 12-15%
        take_profit_2 = buy_price * 1.13
        
        reason = "상승 패턴이므로 공격적 익절 추천"
    
    # 진짜 이탈: 빠른 익절
    elif pattern_type == '진짜_이탈':
        take_profit_1 = buy_price * 1.03
        take_profit_2 = buy_price * 1.05
        reason = "하락 패턴이므로 빠른 익절 추천"
    
    # 불명확: 보수적 익절
    else:
        take_profit_1 = buy_price * 1.05
        take_profit_2 = buy_price * 1.10
        reason = "불명확한 패턴이므로 보수적 익절 추천"
    
    # RSI 과열 시 조정
    if rsi > 70:
        take_profit_1 = buy_price * 1.04  # 더 보수적으로
        take_profit_2 = buy_price * 1.08
        reason += " (RSI 과열 구간)"
    
    return {
        'take_profit_1': round(take_profit_1),
        'take_profit_2': round(take_profit_2),
        'reason': reason
    }




