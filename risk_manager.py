#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
리스크 관리 모듈
손절가/익절가를 계산합니다.
"""

import pandas as pd


def calculate_stop_loss(price_df, buy_price, pattern_info):
    """
    손절가 계산
    
    Args:
        price_df: 가격 데이터프레임
        buy_price: 매수가
        pattern_info: 패턴 분석 결과
    
    Returns:
        dict: {
            'stop_loss': float,
            'loss_pct': float,
            'reason': str
        }
    """
    if price_df is None or len(price_df) == 0:
        return None
    
    # MA20 계산
    if 'MA20' not in price_df.columns:
        price_df['MA20'] = price_df['종가'].rolling(window=20).mean()
    
    ma20 = price_df['MA20'].iloc[-1] if pd.notna(price_df['MA20'].iloc[-1]) else buy_price
    
    pattern_type = pattern_info.get('pattern_type', '불명확')
    
    # 물량 털기나 상승 신호: MA20 * 0.97 (3% 하락)
    if pattern_type in ['물량_털기', '상승_신호']:
        stop_loss = ma20 * 0.97
        reason = "MA20 이탈 시 손절 (3% 하락)"
    
    # 진짜 이탈: 더 빠른 손절 (5% 하락)
    elif pattern_type == '진짜_이탈':
        stop_loss = buy_price * 0.95
        reason = "하락 패턴이므로 빠른 손절 (5% 하락)"
    
    # 불명확: 보수적 손절
    else:
        stop_loss = min(ma20 * 0.97, buy_price * 0.97)
        reason = "보수적 손절 (3% 하락 또는 MA20 이탈)"
    
    loss_pct = ((stop_loss - buy_price) / buy_price) * 100
    
    return {
        'stop_loss': round(stop_loss),
        'loss_pct': round(loss_pct, 2),
        'reason': reason
    }


def calculate_risk_reward_ratio(buy_price, take_profit, stop_loss):
    """
    리스크/리워드 비율 계산
    
    Args:
        buy_price: 매수가
        take_profit: 익절가
        stop_loss: 손절가
    
    Returns:
        float: 리스크/리워드 비율
    """
    reward = take_profit - buy_price
    risk = buy_price - stop_loss
    
    if risk <= 0:
        return 0
    
    return round(reward / risk, 2)


def assess_overheating(price_df):
    """
    과열 구간 판단
    
    Returns:
        dict: {
            'is_overheated': bool,
            'rsi': float,
            'price_vs_ma': float,  # 현재가 / MA20 비율
            'recommendation': str
        }
    """
    if price_df is None or len(price_df) == 0:
        return None
    
    current_price = price_df['종가'].iloc[-1]
    
    # RSI
    if 'RSI' in price_df.columns:
        rsi = price_df['RSI'].iloc[-1]
    else:
        rsi = 50
    
    # MA20 대비 가격
    if 'MA20' in price_df.columns:
        ma20 = price_df['MA20'].iloc[-1]
        if pd.notna(ma20) and ma20 > 0:
            price_vs_ma = (current_price / ma20) * 100
        else:
            price_vs_ma = 100
    else:
        price_vs_ma = 100
    
    # 과열 판단
    is_overheated = False
    recommendation = ""
    
    if rsi > 70 or price_vs_ma > 115:
        is_overheated = True
        if rsi > 70:
            recommendation = f"RSI {rsi:.1f}로 과열 구간입니다. 익절 고려하세요."
        else:
            recommendation = f"현재가가 MA20 대비 {price_vs_ma-100:.1f}% 높습니다. 익절 고려하세요."
    elif rsi > 60:
        recommendation = f"RSI {rsi:.1f}로 상승 중입니다. 익절 준비하세요."
    else:
        recommendation = "과열 구간이 아닙니다."
    
    return {
        'is_overheated': is_overheated,
        'rsi': float(rsi) if pd.notna(rsi) else 50,
        'price_vs_ma': round(price_vs_ma, 2),
        'recommendation': recommendation
    }




