#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
나만의 주식 도우미 (AI 수급 분석 트레이더)
메인 실행 파일

사용법:
    python main_trader.py --code 035720  # 한국 주식
    python main_trader.py --code AAPL   # 미국 주식
"""

import argparse
import sys
import os

# 현재 디렉토리를 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from data_fetcher import (
    is_us_stock, fetch_korean_stock_data, fetch_us_stock_data,
    fetch_investor_trading_data, fetch_technical_indicators
)
from pattern_detector import analyze_investor_pattern, calculate_pattern_strength, detect_recovery_signal
from signal_generator import generate_buy_signals, generate_sell_signals
from risk_manager import calculate_stop_loss, assess_overheating


def format_price(price, is_us=False):
    """가격 포맷팅"""
    if is_us:
        return f"${price:,.2f}"
    else:
        return f"{price:,.0f}원"


def print_analysis_result(code, stock_name, is_us, stock_data, investor_data, pattern_info, signals, stop_loss, overheating, recovery_signal=None):
    """분석 결과 출력"""
    
    print("\n" + "=" * 70)
    print(f"📊 {stock_name} ({code}) 분석 결과")
    print("=" * 70)
    
    # 현재 상황
    print("\n[현재 상황]")
    current_price = stock_data['current_price']
    price_df = stock_data['price_data']
    
    print(f"현재가: {format_price(current_price, is_us)}")
    
    if investor_data is not None and len(investor_data) > 0:
        foreign_trend = pattern_info.get('foreign_trend', '불명확')
        institution_trend = pattern_info.get('institution_trend', '불명확')
        foreign_avg = pattern_info.get('foreign_avg', 0)
        institution_avg = pattern_info.get('institution_avg', 0)
        
        foreign_emoji = "📈" if foreign_trend == '매수' else ("📉" if foreign_trend == '매도' else "➡️")
        institution_emoji = "📈" if institution_trend == '매수' else ("📉" if institution_trend == '매도' else "➡️")
        
        print(f"외국인: {foreign_emoji} {foreign_trend} (최근 5일 평균: {foreign_avg:+,.0f}주)")
        print(f"기관: {institution_emoji} {institution_trend} (최근 5일 평균: {institution_avg:+,.0f}주)")
        
        volume_trend = pattern_info.get('volume_trend', '불명확')
        print(f"거래량 추세: {volume_trend}")
    
    # 패턴 판단
    pattern_type = pattern_info.get('pattern_type', '불명확')
    confidence = pattern_info.get('confidence', 0)
    reason = pattern_info.get('reason', '')
    data_available = pattern_info.get('data_available', True)
    
    pattern_emoji = {
        '물량_털기': '🟡',
        '진짜_이탈': '🔴',
        '상승_신호': '🟢',
        '불명확': '⚪'
    }
    
    print(f"\n패턴 판단: {pattern_emoji.get(pattern_type, '⚪')} {pattern_type} (신뢰도: {confidence}%)")
    if not data_available:
        print(f"→ ⚠️ {reason}")
        print(f"   크롤링 경로: https://finance.naver.com/item/frgn.naver?code={code}")
        print(f"   수급 데이터 없이 기술적 지표만으로 분석했습니다.")
    else:
        print(f"→ {reason}")
    
    # 기술적 지표
    if price_df is not None and len(price_df) > 0:
        if 'RSI' in price_df.columns:
            rsi = price_df['RSI'].iloc[-1]
            if pd.notna(rsi):
                rsi_status = "과열" if rsi > 70 else ("강세" if rsi > 50 else "약세")
                print(f"RSI: {rsi:.1f} ({rsi_status})")
        
        if 'MA20' in price_df.columns:
            ma20 = price_df['MA20'].iloc[-1]
            if pd.notna(ma20):
                price_vs_ma = (current_price / ma20 - 1) * 100
                print(f"20일선 대비: {price_vs_ma:+.1f}%")
    
    # 회복 신호 감지
    recovery_signal = None
    if investor_data is not None and len(investor_data) > 0:
        recovery_signal = detect_recovery_signal(investor_data, price_df)
        if recovery_signal and recovery_signal.get('has_recovery_signal'):
            print(f"\n🟢 회복 신호 감지: {recovery_signal['message']}")
    
    # 추천 전략
    print("\n" + "=" * 70)
    print("[추천 전략]")
    print("=" * 70)
    
    # 진짜_이탈 패턴일 때 보유자/미보유자 분기
    if pattern_type == '진짜_이탈':
        # 보유자용: 익절가/손절가 제시
        print("\n📌 [보유자용]")
        if price_df is not None:
            # 평균 매수가를 현재가로 가정 (실제로는 보유자의 평균 매수가를 입력받아야 함)
            assumed_buy_price = current_price * 0.95  # 예시: 현재가보다 5% 낮게 매수했다고 가정
            sell_signals = generate_sell_signals(price_df, pattern_info, current_price, assumed_buy_price)
            if sell_signals:
                print(f"📈 1차 익절: {format_price(sell_signals['take_profit_1'], is_us)}")
                print(f"📈 2차 익절: {format_price(sell_signals['take_profit_2'], is_us)}")
                print(f"   이유: {sell_signals['reason']}")
        
        if stop_loss:
            print(f"\n🛑 손절가: {format_price(stop_loss['stop_loss'], is_us)}")
            print(f"   손실률: {stop_loss['loss_pct']:.1f}%")
            print(f"   이유: {stop_loss['reason']}")
        
        # 미보유자용: 신규 매수 비추천
        print("\n📌 [신규 진입자용]")
        if recovery_signal and recovery_signal.get('has_recovery_signal'):
            print(f"🟢 {recovery_signal['message']}")
        else:
            print("🚫 신규 매수 비추천")
            print("   수급·거래량 회복 시점까지 대기 권장")
            print("   회복 신호: 외국인 매수 전환 또는 거래량 +30% 이상 증가 시 재검토")
    
    # 다른 패턴: 기존 로직 유지
    elif signals:
        buy_1 = signals.get('buy_1')
        buy_2 = signals.get('buy_2')
        strategy = signals.get('strategy', '')
        
        if buy_1:
            print(f"\n💰 1차 매수: {format_price(buy_1['price'], is_us)}")
            if buy_1['days'] > 0:
                print(f"   예상 도달: 약 {buy_1['days']}일 후")
            print(f"   이유: {buy_1['reason']}")
        
        if buy_2:
            print(f"\n💰 2차 매수: {format_price(buy_2['price'], is_us)}")
            if buy_2['days'] > 0:
                print(f"   예상 도달: 약 {buy_2['days']}일 후")
            print(f"   이유: {buy_2['reason']}")
        
        if strategy:
            print(f"\n💡 전략: {strategy}")
        
        # 익절가
        if price_df is not None:
            sell_signals = generate_sell_signals(price_df, pattern_info, current_price, buy_1['price'] if buy_1 else current_price)
            if sell_signals:
                print(f"\n📈 1차 익절: {format_price(sell_signals['take_profit_1'], is_us)}")
                print(f"📈 2차 익절: {format_price(sell_signals['take_profit_2'], is_us)}")
                print(f"   이유: {sell_signals['reason']}")
        
        # 손절가
        if stop_loss and buy_1:
            print(f"\n🛑 손절가: {format_price(stop_loss['stop_loss'], is_us)}")
            print(f"   손실률: {stop_loss['loss_pct']:.1f}%")
            print(f"   이유: {stop_loss['reason']}")
    
    # 과열 구간 알림
    if overheating and overheating.get('is_overheated'):
        print(f"\n⚠️  과열 구간 알림:")
        print(f"   {overheating['recommendation']}")
    
    # 요약
    print("\n" + "=" * 70)
    print("[요약]")
    print("=" * 70)
    
    summary_emoji = pattern_emoji.get(pattern_type, '⚪')
    if pattern_type == '물량_털기':
        summary = f"{summary_emoji} 단기 조정 후 상승 가능성 {confidence}%"
    elif pattern_type == '상승_신호':
        summary = f"{summary_emoji} 상승 신호 강함 (신뢰도 {confidence}%)"
    elif pattern_type == '진짜_이탈':
        summary = f"{summary_emoji} 하락 전환 가능성 높음 (관망 권장)"
    else:
        summary = f"{summary_emoji} 추가 관찰 필요"
    
    print(f"\n{summary}\n")


def main():
    parser = argparse.ArgumentParser(
        description='나만의 주식 도우미 - 외국인/기관 수급 분석 트레이더',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main_trader.py --code 035720    # 한국 주식 (카카오)
  python main_trader.py --code AAPL     # 미국 주식 (애플)
        """
    )
    parser.add_argument('--code', type=str, required=True,
                       help='종목 코드 (한국: 종목번호, 미국: 티커)')
    
    args = parser.parse_args()
    
    code = args.code
    is_us = is_us_stock(code)
    
    print("=" * 70)
    print("🤖 나만의 주식 도우미 (AI 수급 분석 트레이더)")
    print("=" * 70)
    print(f"\n📥 데이터 수집 중... ({'미국' if is_us else '한국'} 주식)")
    
    # 1. 기본 주식 데이터 수집
    if is_us:
        stock_data = fetch_us_stock_data(code)
    else:
        stock_data = fetch_korean_stock_data(code)
    
    if stock_data is None:
        print(f"❌ 종목 코드 {code}의 데이터를 가져올 수 없습니다.")
        return
    
    stock_name = stock_data['name']
    price_df = stock_data['price_data']
    current_price = stock_data['current_price']
    
    # 2. 기술적 지표 계산
    if price_df is not None and len(price_df) > 0:
        price_df = fetch_technical_indicators(price_df)
    else:
        print("⚠️ 일봉 데이터가 없어 기술적 지표를 계산할 수 없습니다.")
        print("현재가 정보만 사용합니다.")
    
    # 3. 외국인/기관 매매 데이터 수집 (한국 주식만)
    investor_data = None
    if not is_us:
        print(f"📊 외국인/기관 매매 데이터 수집 중...")
        investor_data = fetch_investor_trading_data(code)
        if investor_data is not None and len(investor_data) > 0:
            print(f"✅ 외국인/기관 데이터 수집 완료 ({len(investor_data)}일)")
        else:
            print(f"⚠️ 외국인/기관 데이터를 가져올 수 없습니다. (기술적 지표만 사용)")
    
    # 4. 패턴 분석
    pattern_info = analyze_investor_pattern(investor_data, price_df, days=5)
    
    # 4-1. 회복 신호 감지
    recovery_signal = None
    if investor_data is not None and len(investor_data) > 0:
        recovery_signal = detect_recovery_signal(investor_data, price_df)
    
    # 5. 매수 신호 생성
    signals = None
    if price_df is not None and len(price_df) > 0:
        signals = generate_buy_signals(price_df, pattern_info, current_price)
    else:
        print("⚠️ 일봉 데이터가 없어 매수 신호를 생성할 수 없습니다.")
    
    # 6. 손절가 계산
    stop_loss = None
    if signals and signals.get('buy_1') and price_df is not None:
        stop_loss = calculate_stop_loss(price_df, signals['buy_1']['price'], pattern_info)
    
    # 7. 과열 구간 판단
    overheating = None
    if price_df is not None and len(price_df) > 0:
        overheating = assess_overheating(price_df)
    
    # 8. 결과 출력
    print_analysis_result(code, stock_name, is_us, stock_data, investor_data, 
                         pattern_info, signals, stop_loss, overheating, recovery_signal)


if __name__ == "__main__":
    import pandas as pd
    main()

