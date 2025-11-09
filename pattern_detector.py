#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
패턴 감지 모듈
물량 털기 vs 진짜 이탈을 판단합니다.
"""

import pandas as pd
import numpy as np


def analyze_investor_pattern(investor_df, price_df, days=5):
    """
    외국인/기관 매매 패턴 분석
    
    Args:
        investor_df: 외국인/기관 매매 데이터
        price_df: 가격 데이터
        days: 분석 기간 (일)
    
    Returns:
        dict: {
            'foreign_trend': str,  # '매수', '매도', '중립'
            'institution_trend': str,
            'foreign_avg': float,  # 최근 N일 평균 순매수
            'institution_avg': float,
            'volume_trend': str,  # '증가', '감소', '유지'
            'pattern_type': str,  # '물량_털기', '진짜_이탈', '상승_신호', '불명확'
            'confidence': float,  # 0-100
            'reason': str
        }
    """
    if investor_df is None or len(investor_df) == 0:
        return {
            'pattern_type': '불명확',
            'confidence': 0,
            'reason': '외국인/기관 매매 데이터를 가져올 수 없습니다. 네이버 증권 페이지에서 수급 데이터를 확인하세요.',
            'data_available': False
        }
    
    # 최근 N일 데이터
    recent = investor_df.tail(days).copy()
    
    # 평균 순매수 계산
    foreign_avg = recent['외국인_순매수'].mean()
    institution_avg = recent['기관_순매수'].mean()
    
    # 추세 판단
    foreign_trend = '매수' if foreign_avg > 0 else ('매도' if foreign_avg < 0 else '중립')
    institution_trend = '매수' if institution_avg > 0 else ('매도' if institution_avg < 0 else '중립')
    
    # 거래량 추세
    if len(recent) >= 2:
        volume_change = (recent['거래량'].iloc[-1] - recent['거래량'].iloc[-2]) / recent['거래량'].iloc[-2] * 100
        if volume_change > 10:
            volume_trend = '증가'
        elif volume_change < -10:
            volume_trend = '감소'
        else:
            volume_trend = '유지'
    else:
        volume_trend = '유지'
    
    # 가격 추세 (MA20 기준)
    price_trend = None
    if price_df is not None and len(price_df) >= 20:
        current_price = price_df['종가'].iloc[-1]
        ma20 = price_df['MA20'].iloc[-1]
        if pd.notna(ma20):
            price_trend = '상승' if current_price > ma20 else '하락'
    
    # 패턴 판단
    pattern_type = '불명확'
    confidence = 50
    reason = ""
    data_available = True  # 데이터가 있는지 여부
    
    # 1. 물량 털기 패턴
    # - 외국인 매도 + 기관 매수 + 거래량 유지/증가 + 가격이 MA20 위
    if (foreign_trend == '매도' and institution_trend == '매수' and 
        volume_trend in ['유지', '증가'] and price_trend == '상승'):
        pattern_type = '물량_털기'
        confidence = 75
        reason = f"외국인 매도 중이지만 기관이 매수하고 있고, 거래량이 {volume_trend}하며 가격이 20일선 위에 있어 물량 털기 가능성이 높습니다."
    
    # 2. 진짜 이탈 패턴
    # - 외국인 매도 + 기관도 매도 + 거래량 감소
    elif (foreign_trend == '매도' and institution_trend == '매도' and 
          volume_trend == '감소'):
        pattern_type = '진짜_이탈'
        confidence = 80
        reason = f"외국인과 기관이 모두 매도하고 있고, 거래량이 감소하여 진짜 이탈 가능성이 높습니다."
    
    # 3. 상승 신호
    # - 외국인 매수 + 기관 매수
    elif foreign_trend == '매수' and institution_trend == '매수':
        pattern_type = '상승_신호'
        confidence = 85
        reason = "외국인과 기관이 동시에 매수하고 있어 상승 신호입니다."
    
    # 4. 외국인만 매도 (기관은 중립/매수)
    elif foreign_trend == '매도' and institution_trend in ['매수', '중립']:
        if volume_trend == '유지' and price_trend == '상승':
            pattern_type = '물량_털기'
            confidence = 65
            reason = "외국인만 매도 중이지만 기관은 매수/중립이고, 거래량과 가격이 유지되어 물량 털기 가능성이 있습니다."
        else:
            pattern_type = '불명확'
            confidence = 50
            reason = "외국인 매도 중이지만 기관은 매수/중립입니다. 추가 관찰이 필요합니다."
    
    # 5. 기관만 매도 (외국인은 중립/매수)
    elif institution_trend == '매도' and foreign_trend in ['매수', '중립']:
        pattern_type = '불명확'
        confidence = 55
        reason = "기관만 매도 중입니다. 외국인 동향을 주시하세요."
    
    else:
        pattern_type = '불명확'
        confidence = 50
        reason = "명확한 패턴이 감지되지 않았습니다."
    
    return {
        'foreign_trend': foreign_trend,
        'institution_trend': institution_trend,
        'foreign_avg': float(foreign_avg),
        'institution_avg': float(institution_avg),
        'volume_trend': volume_trend,
        'pattern_type': pattern_type,
        'confidence': confidence,
        'reason': reason,
        'price_trend': price_trend,
        'data_available': data_available
    }


def detect_recovery_signal(investor_df, price_df):
    """
    회복 신호 감지
    - 외국인 1일이라도 플러스로 전환
    - 거래량 전일 대비 +30%
    
    Returns:
        dict: {
            'has_recovery_signal': bool,
            'recovery_type': str,  # 'foreign_buy', 'volume_surge', 'both'
            'message': str
        }
    """
    if investor_df is None or len(investor_df) == 0:
        return {
            'has_recovery_signal': False,
            'recovery_type': None,
            'message': ''
        }
    
    recovery_signals = []
    
    # 1. 외국인 1일이라도 플러스로 전환
    if len(investor_df) >= 1:
        latest_foreign = investor_df['외국인_순매수'].iloc[-1]
        if latest_foreign > 0:
            recovery_signals.append('foreign_buy')
    
    # 2. 거래량 전일 대비 +30%
    if len(investor_df) >= 2:
        latest_volume = investor_df['거래량'].iloc[-1]
        prev_volume = investor_df['거래량'].iloc[-2]
        if prev_volume > 0:
            volume_change_pct = (latest_volume - prev_volume) / prev_volume * 100
            if volume_change_pct >= 30:
                recovery_signals.append('volume_surge')
    
    has_signal = len(recovery_signals) > 0
    
    if has_signal:
        if 'foreign_buy' in recovery_signals and 'volume_surge' in recovery_signals:
            recovery_type = 'both'
            message = "외국인 매수 전환 + 거래량 급증: 재매수 후보로 전환"
        elif 'foreign_buy' in recovery_signals:
            recovery_type = 'foreign_buy'
            message = "외국인 매수 전환: 재매수 후보로 전환"
        else:
            recovery_type = 'volume_surge'
            message = "거래량 급증 (+30% 이상): 재매수 후보로 전환"
    else:
        recovery_type = None
        message = ''
    
    return {
        'has_recovery_signal': has_signal,
        'recovery_type': recovery_type,
        'message': message
    }


def calculate_pattern_strength(investor_df, price_df):
    """
    패턴 강도 계산 (0-100)
    
    Returns:
        float: 패턴 강도 점수
    """
    if investor_df is None or len(investor_df) == 0:
        return 0
    
    score = 50  # 기본 점수
    
    recent = investor_df.tail(5)
    
    # 외국인/기관 동향 일치도
    foreign_consistent = (recent['외국인_순매수'] > 0).sum() >= 3  # 최근 5일 중 3일 이상 매수
    institution_consistent = (recent['기관_순매수'] > 0).sum() >= 3
    
    if foreign_consistent and institution_consistent:
        score += 20  # 둘 다 매수 중
    elif not foreign_consistent and not institution_consistent:
        score -= 20  # 둘 다 매도 중
    
    # 거래량 증가
    if len(recent) >= 2:
        volume_ratio = recent['거래량'].iloc[-1] / recent['거래량'].iloc[-2]
        if volume_ratio > 1.2:
            score += 10
        elif volume_ratio < 0.8:
            score -= 10
    
    # 가격 추세
    if price_df is not None and len(price_df) >= 20:
        current_price = price_df['종가'].iloc[-1]
        ma20 = price_df['MA20'].iloc[-1]
        if pd.notna(ma20):
            if current_price > ma20 * 1.05:
                score += 10
            elif current_price < ma20 * 0.95:
                score -= 10
    
    return max(0, min(100, score))

