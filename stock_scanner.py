#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단기/스윙용 종목 탐색기
종목번호나 티커를 입력하면 이해하기 쉬운 신호로 정보를 제공합니다.
"""

import sys
import os
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import pytz

# stock_screener.py의 함수들을 import
# 같은 디렉토리에 있다고 가정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from stock_screener import (
        is_market_closed, fetch_stock_data, fetch_stock_data_yahoo,
        calculate_ma, calculate_rsi, calculate_macd,
        analyze_entry_opportunity, analyze_granville_rules,
        analyze_ma_energy_state, calculate_ma_energy_momentum_score,
        is_us_stock, postprocess_signal
    )
    YFINANCE_AVAILABLE = True
except ImportError as e:
    print(f"❌ 필요한 모듈을 import할 수 없습니다: {e}")
    print("stock_screener.py 파일이 같은 디렉토리에 있는지 확인해주세요.")
    sys.exit(1)

# yfinance import
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance 패키지가 설치되지 않았습니다. 미국 주식 조회를 위해 설치해주세요:")
    print("   pip install yfinance")


def get_current_price(ticker, is_us, include_prepost=False):
    """
    실시간 현재가 가져오기
    
    Args:
        ticker: 종목 티커
        is_us: 미국 주식 여부
        include_prepost: True면 프리마켓/애프터마켓 포함 (기본값: False)
    
    Returns:
        float: 현재가
    """
    current_price = None
    
    if is_us:
        if not YFINANCE_AVAILABLE:
            return None
        try:
            ticker_obj = yf.Ticker(ticker)
            
            # 방법 1: 최근 1분 데이터에서 가격 가져오기 (가장 정확하고 실시간)
            try:
                # prepost=True로 프리마켓/애프터마켓 포함
                hist = ticker_obj.history(period='1d', interval='1m', prepost=include_prepost)
                if not hist.empty and len(hist) > 0:
                    # 가장 최근 데이터 사용
                    latest_price = hist['Close'].iloc[-1]
                    if latest_price is not None and not pd.isna(latest_price) and latest_price > 0:
                        current_price = float(latest_price)
                        # 유효성 검증: 0보다 크고 합리적인 범위 내
                        if current_price > 0:
                            return current_price
            except Exception as e:
                # 디버깅: 에러가 발생해도 다음 방법 시도
                pass
            
            # 방법 2: fast_info의 lastPrice 사용 (빠름)
            try:
                fast_info = ticker_obj.fast_info
                current_price = fast_info.get('lastPrice')
                if current_price is not None and not pd.isna(current_price) and current_price > 0:
                    return current_price
            except Exception as e:
                pass
            
            # 방법 3: info에서 가격 가져오기
            try:
                info = ticker_obj.info
                # currentPrice가 가장 정확, 없으면 regularMarketPrice
                current_price = info.get('currentPrice')
                if current_price is None or pd.isna(current_price) or current_price <= 0:
                    current_price = info.get('regularMarketPrice')
                if current_price is None or pd.isna(current_price) or current_price <= 0:
                    current_price = info.get('previousClose')
                
                if current_price is not None and not pd.isna(current_price) and current_price > 0:
                    return current_price
            except Exception as e:
                pass
        except Exception:
            pass
    else:
        try:
            url = f"https://finance.naver.com/item/main.naver?code={ticker}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=5)
            response.encoding = 'euc-kr'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 현재가 찾기 (여러 방법 시도)
            price_element = soup.find('p', {'class': 'no_today'})
            if not price_element:
                price_element = soup.find('div', {'class': 'no_today'})
            
            if price_element:
                # 방법 1: price_element 내부의 blind span 찾기 (가장 정확)
                blind_span = price_element.find('span', {'class': 'blind'})
                if blind_span:
                    blind_text = blind_span.text.strip()
                    # 숫자와 쉼표만 있는 경우
                    if re.match(r'^[\d,]+$', blind_text):
                        clean_price = blind_text.replace(',', '')
                        if len(clean_price) >= 4 and len(clean_price) <= 8:
                            try:
                                price_candidate = float(clean_price)
                                # 합리적인 범위 (1000원 ~ 1억원)
                                if 1000 <= price_candidate <= 100000000:
                                    current_price = price_candidate
                            except:
                                pass
                
                # 방법 2: blind 클래스 전체에서 숫자만 있는 것 찾기
                if current_price is None:
                    blind_spans = soup.find_all('span', {'class': 'blind'})
                    for span in blind_spans:
                        span_text = span.text.strip()
                        # 숫자와 쉼표만 있는 경우 (가격일 가능성)
                        if re.match(r'^[\d,]+$', span_text):
                            clean_price = span_text.replace(',', '')
                            if len(clean_price) >= 4 and len(clean_price) <= 8:
                                try:
                                    price_candidate = float(clean_price)
                                    # 합리적인 범위 (1000원 ~ 1억원)
                                    if 1000 <= price_candidate <= 100000000:
                                        current_price = price_candidate
                                        break
                                except:
                                    pass
                
                price_text = price_element.get_text(strip=True)
                
                # 방법 2: 기존 방법 (blind가 없을 때 대비)
                if current_price is None:
                    # 모든 숫자 추출 (쉼표 제거 후)
                    price_text_clean = price_text.replace(',', '').replace('원', '').replace(' ', '')
                    # 숫자만 추출
                    numbers = re.findall(r'\d+', price_text_clean)
                    if numbers:
                        # 가장 긴 숫자를 찾되, 중복 제거
                        unique_numbers = []
                        for num in numbers:
                            if len(num) >= 4 and len(num) <= 8:
                                unique_numbers.append(num)
                        
                        if unique_numbers:
                            # 가장 긴 숫자 사용
                            price_str = max(unique_numbers, key=len)
                            # 중복된 경우: 6자리 이상이면 절반으로 나누기
                            if len(price_str) >= 12:  # 12자리 이상이면 중복 가능성
                                mid = len(price_str) // 2
                                price_str = price_str[:mid]
                            
                            if price_str.isdigit() and 4 <= len(price_str) <= 8:
                                try:
                                    current_price = float(price_str)
                                    # 추가 검증: 합리적인 가격 범위 (1000원 ~ 1억원)
                                    if current_price < 1000 or current_price > 100000000:
                                        current_price = None
                                except:
                                    current_price = None
        except Exception:
            pass
    
    return current_price


def fix_encoding(name: str) -> str:
    """크롤링하다가 '쇱...' 이런 식으로 깨진 한글 종목명을 복원하려는 시도용."""
    if not name:
        return name
    # 가장 흔한 깨짐 패턴: latin1로 들어왔는데 실제는 utf-8일 때
    try:
        fixed = name.encode("latin1").decode("utf-8")
        return fixed
    except Exception:
        # 그래도 안 되면 그냥 원본 반환
        return name


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ATR(평균 진폭 범위) 계산."""
    required_cols = {'고가', '저가', '종가'}
    if not required_cols.issubset(df.columns):
        df['TR'] = pd.NA
        df['ATR14'] = pd.NA
        return df

    high = df['고가'].astype(float)
    low = df['저가'].astype(float)
    close = df['종가'].astype(float)

    prev_close = close.shift(1)
    tr_components = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1)

    df['TR'] = tr_components.max(axis=1)
    df['ATR14'] = df['TR'].rolling(window=period, min_periods=period).mean()

    return df


def detect_market_phase(row: pd.Series):
    ma5 = row.get('MA5')
    ma20 = row.get('MA20')
    ma60 = row.get('MA60')
    macd = row.get('MACD')
    macd_signal = row.get('MACD_Signal')
    rsi = row.get('RSI')

    if all(pd.notna(val) for val in (ma5, ma20, ma60, macd, macd_signal, rsi)):
        if ma60 < ma20 < ma5 and macd > macd_signal and rsi >= 50:
            return "📈 상승장", "정배열 + 모멘텀 양호 → 눌림목 분할매수 유효"
        if ma5 < ma20 < ma60 and macd < macd_signal and rsi <= 45:
            return "📉 하락장", "역배열 + 모멘텀 약세 → 현금 비중 확대 추천"

    return "⚖️ 전환/횡보 레짐", "이평선이 엉켜 있어 방향이 모호 → 소액 탐색 또는 관망"


def explain_ma(ma5, ma20, ma60):
    print("📈 [1] 이동평균 해석")
    if not all(pd.notna(val) for val in (ma5, ma20, ma60)):
        print("   → 이동평균 데이터를 충분히 확보하지 못했습니다.\n")
        return
    print(f"   단기(MA5)={ma5:.2f}, 중기(MA20)={ma20:.2f}, 장기(MA60)={ma60:.2f}")
    if ma60 < ma20 < ma5:
        print("   → 세 선이 아래에서 위로 순서대로: '정배열' = 상승 추세.")
        print("   👉 이럴 땐 내려올 때(눌림) 조금씩 사두는 게 기본입니다.\n")
    elif ma5 < ma20 < ma60:
        print("   → 위에서 아래로: '역배열' = 하락 추세.")
        print("   👉 지금 사기보단 올라탈 타이밍을 기다리는 게 좋아요.\n")
    else:
        print("   → 선들이 섞여 있어서 추세가 애매해요.")
        print("   👉 확실한 방향이 나올 때까지 소액만 시도하거나 관망.\n")


def explain_macd(macd, macd_signal):
    print("📊 [2] MACD 해석")
    if pd.isna(macd) or pd.isna(macd_signal):
        print("   → MACD 데이터를 충분히 확보하지 못했습니다.\n")
        return
    print(f"   MACD={macd:.2f}, Signal={macd_signal:.2f}")
    if macd > macd_signal:
        print("   → MACD가 시그널 위에 있음 = 최근 올라가려는 힘이 살아있음.")
        print("   👉 이 힘이 이평선 신호랑 같이 나오면 매수 신호로 봐도 돼요.\n")
    else:
        print("   → MACD가 시그널 아래 = 힘이 약해지거나 하락 쪽 힘이 커짐.")
        print("   👉 단독으로는 '사라'가 아니고, 오히려 조심하라는 신호에 가까워요.\n")


def explain_rsi(rsi):
    print("💡 [3] RSI 해석")
    if pd.isna(rsi):
        print("   → RSI 데이터를 충분히 확보하지 못했습니다.\n")
        return
    print(f"   RSI={rsi:.1f}")
    if rsi >= 70:
        print("   → 너무 많이 오른 상태(과매수)라서 바로 들어가면 물릴 수 있어요.")
    elif rsi <= 30:
        print("   → 너무 많이 내린 상태(과매도)라서 반등 나올 수도 있어요.")
    else:
        print("   → 매수/매도 힘이 비슷한 중립 구간이에요.")
    print("   👉 RSI는 30 근처에서 사고, 70 근처에서 파는 연습을 하면 이해가 빨라요.\n")


def explain_atr(close_price, atr14, formatter, currency_label, multiplier=2.0):
    if pd.isna(atr14):
        print("🛡️ [4] 변동성(ATR) 기반 손절선")
        print("   → ATR 데이터를 계산하지 못했습니다.\n")
        return None
    stop = close_price - atr14 * multiplier
    print("🛡️ [4] 변동성(ATR) 기반 손절선")
    print(f"   ATR(14)={formatter(atr14)}{currency_label}, 배수={multiplier}")
    print(f"   → 최근 평균 변동 폭을 감안한 손절 라인: {formatter(stop)}{currency_label}")
    print("   → 이 아래로 내려가면 '평소 흔들림'을 넘었다고 판단할 수 있어요.\n")
    return stop


def explain_conclusion(phase_label, ma5, ma20, ma60, macd, macd_signal, rsi):
    print("📉 [5] 오늘의 행동 가이드")
    phase_label = phase_label or "⚖️ 전환/횡보 레짐"
    if phase_label.startswith("📉"):
        print("   🔴 하락장으로 판정 → 지키는 모드 우선")
        print("   1) 이미 들고 있으면 반등 때 분할로 줄이기")
        print("   2) 신규 매수는 골든크로스·거래량 급증 같은 확실한 신호 기다리기\n")
    elif phase_label.startswith("📈"):
        print("   🟢 상승장 → 눌림목 분할 매수 전략 유효")
        print("   1) 바로 몰빵 대신 2~3회 나눠 담기")
        print("   2) 손절선은 이동평균 또는 ATR 기준으로 설정\n")
    else:
        print("   ⚪ 방향이 확실치 않은 전환 구간")
        print("   → 소액으로 탐색하거나, 추세가 확정될 때까지 공부/기다림이 안전해요.\n")

def get_stock_name(ticker, is_us):
    """종목명 가져오기"""
    if is_us:
        if not YFINANCE_AVAILABLE:
            return ticker
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            return info.get('longName') or info.get('shortName') or ticker
        except:
            return ticker
    else:
        try:
            # 네이버 증권 시세 페이지 사용 (더 간단한 구조)
            url = f"https://finance.naver.com/item/sise.naver?code={ticker}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=5)
            
            # 인코딩 문제 해결: response.content를 직접 디코딩
            try:
                # euc-kr로 디코딩 시도
                content = response.content.decode('euc-kr', errors='ignore')
                soup = BeautifulSoup(content, 'html.parser')
            except:
                # 실패 시 기본 인코딩 사용
                response.encoding = 'euc-kr'
                soup = BeautifulSoup(response.text, 'html.parser')
            
            # 방법 0: 시세 페이지의 strong.tlt 태그 (가장 정확)
            name_elem = soup.find('strong', {'class': 'tlt'})
            if name_elem:
                name = name_elem.get_text(strip=True)
                name = fix_encoding(name)
                if name and any('\uAC00' <= char <= '\uD7A3' for char in name):
                    return name
            
            # 방법 0-1: title에서 추출 (시세 페이지)
            title = soup.find('title')
            if title:
                title_text = title.text
                if ':' in title_text:
                    name = title_text.split(':')[0].strip()
                    name = fix_encoding(name)
                    if name and any('\uAC00' <= char <= '\uD7A3' for char in name):
                        return name
            
            # 메인 페이지도 시도
            url_main = f"https://finance.naver.com/item/main.naver?code={ticker}"
            response_main = requests.get(url_main, headers=headers, timeout=5)
            try:
                content_main = response_main.content.decode('euc-kr', errors='ignore')
                soup_main = BeautifulSoup(content_main, 'html.parser')
            except:
                response_main.encoding = 'euc-kr'
                soup_main = BeautifulSoup(response_main.text, 'html.parser')
            
            # 종목명 찾기 (여러 방법 시도)
            name = None
            
            # 방법 1: h2.wrap_company > a 태그 (가장 정확)
            h2 = soup.find('h2', {'class': 'wrap_company'})
            if h2:
                a_tag = h2.find('a')
                if a_tag:
                    name = a_tag.get_text(strip=True)
                    # 인코딩 복원 시도
                    name = fix_encoding(name)
                    # 한글 확인 (가-힣 범위에 한글이 있는지)
                    if name and any('\uAC00' <= char <= '\uD7A3' for char in name):
                        return name
            
            # 방법 2: wrap_company 내부에서 찾기
            if not name or len(name) < 2:
                wrap_company = soup.find('div', {'class': 'wrap_company'})
                if wrap_company:
                    h2_in_wrap = wrap_company.find('h2')
                    if h2_in_wrap:
                        a_in_h2 = h2_in_wrap.find('a')
                        if a_in_h2:
                            name = a_in_h2.get_text(strip=True)
                            # 인코딩 복원 시도
                            name = fix_encoding(name)
                            if name and any('\uAC00' <= char <= '\uD7A3' for char in name):
                                return name
            
            # 방법 3: title 태그에서 추출
            if not name or len(name) < 2:
                title = soup.find('title')
                if title:
                    title_text = title.text
                    # "종목명 :" 패턴 찾기
                    if ':' in title_text:
                        name = title_text.split(':')[0].strip()
                        # 인코딩 복원 시도
                        name = fix_encoding(name)
                        if name and any('\uAC00' <= char <= '\uD7A3' for char in name):
                            return name
            
            # 방법 4: description meta 태그
            if not name or len(name) < 2:
                meta_desc = soup.find('meta', {'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    content = meta_desc.get('content')
                    if '종목' in content:
                        parts = content.split('종목')
                        if len(parts) > 0:
                            name = parts[0].strip()
                            # 인코딩 복원 시도
                            name = fix_encoding(name)
                            if name and any('\uAC00' <= char <= '\uD7A3' for char in name):
                                return name
            
            # 한글이 없는 경우 티커 반환
            return ticker
        except Exception as e:
            # 오류 발생 시 티커 반환
            return ticker
    return ticker


def _safe_float(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def evaluate_daytrade(ticker, latest, df, current_price):
    ma5 = _safe_float(latest.get('MA5'))
    ma20 = _safe_float(latest.get('MA20'))
    volume_ratio = _safe_float(latest.get('volume_ratio'))
    rsi = _safe_float(latest.get('RSI'))
    atr = _safe_float(latest.get('ATR14'))

    conditions = []
    entry_signal = False
    exit_signal = False
    status = "단타 진입 부적합"

    if ma5:
        diff_ma5 = abs(current_price - ma5) / ma5
        near_ma5 = diff_ma5 <= 0.012
    else:
        near_ma5 = False
    if near_ma5:
        conditions.append("MA5 근처 눌림")

    vol_ok = volume_ratio is not None and volume_ratio >= 2.0
    if vol_ok:
        conditions.append(f"거래량 {volume_ratio:.1f}배")

    rsi_ok = rsi is not None and 35 <= rsi <= 45
    if rsi_ok:
        conditions.append(f"RSI {rsi:.1f}")

    entry_signal = near_ma5 and vol_ok and rsi_ok

    # Exit conditions
    if rsi is not None and rsi >= 70:
        exit_signal = True
        conditions.append("RSI 과열")
    elif ma5 and current_price < ma5 * 0.985:
        exit_signal = True
        conditions.append("MA5 하향 이탈")

    atr_ratio = None
    if atr and current_price:
        atr_ratio = atr / current_price

    if exit_signal:
        status = "단타 매도 타이밍 임박"
    elif entry_signal:
        status = "단타 매수 후보"
    elif atr_ratio and atr_ratio >= 0.05:
        status = "단타 진입 부적합(변동성 과다)"

    stop_loss = round(current_price * 0.98, 4) if current_price else None
    stop_loss_pct = round(100 * (1 - (stop_loss / current_price)), 2) if stop_loss else None

    if not conditions:
        conditions.append("조건 미충족")

    summary = f"{', '.join(conditions)}, {status}"
    reason = ", ".join(conditions)

    return {
        "mode": "daytrade",
        "symbol": ticker,
        "entry_signal": entry_signal,
        "exit_signal": exit_signal,
        "status": status,
        "summary": summary,
        "reason": reason,
        "stop_loss": stop_loss,
        "stop_loss_pct": stop_loss_pct,
        "atr_ratio": atr_ratio,
    }


def evaluate_swing(ticker, latest, df, current_price):
    ma20 = _safe_float(latest.get('MA20'))
    ma60 = _safe_float(latest.get('MA60'))
    volume_ratio = _safe_float(latest.get('volume_ratio'))
    rsi = _safe_float(latest.get('RSI'))

    ma20_slope = None
    if ma20 and 'MA20' in df.columns and len(df) >= 5:
        prev_ma20 = _safe_float(df['MA20'].iloc[-3])
        if prev_ma20:
            ma20_slope = ma20 - prev_ma20

    conditions = []
    entry_signal = False
    exit_signal = False
    status = "추세 지속 중, 보유 권장"

    if ma20:
        diff_ma20 = abs(current_price - ma20) / ma20
        near_ma20 = diff_ma20 <= 0.02
    else:
        near_ma20 = False
    if near_ma20:
        conditions.append("MA20 근처 눌림")

    slope_positive = ma20_slope is not None and ma20_slope > 0
    if slope_positive:
        conditions.append("MA20 상승 기울기")

    rsi_ok = rsi is not None and 40 <= rsi <= 50
    if rsi_ok:
        conditions.append(f"RSI {rsi:.1f}")

    vol_ok = volume_ratio is not None and volume_ratio >= 1.2
    if vol_ok:
        conditions.append(f"거래량 {volume_ratio:.1f}배")

    entry_signal = near_ma20 and slope_positive and rsi_ok and vol_ok

    recent_high = None
    if '고가' in df.columns and len(df) >= 10:
        recent_high = _safe_float(df['고가'].tail(15).max())

    if rsi is not None and rsi >= 70:
        exit_signal = True
        conditions.append("RSI 과열")
    elif ma20 and current_price < ma20 * 0.98:
        exit_signal = True
        conditions.append("MA20 하락 이탈")
    elif recent_high and current_price >= recent_high * 0.995:
        exit_signal = True
        conditions.append("전고점 근접")

    if exit_signal:
        status = "스윙 청산 신호 감지"
    elif entry_signal:
        status = "스윙 진입 유효"
    elif not slope_positive:
        status = "스윙 청산 신호 감지"

    stop_loss = None
    if ma20:
        stop_loss = round(ma20 * 0.98, 4)
    elif current_price:
        stop_loss = round(current_price * 0.97, 4)
    stop_loss_pct = round(100 * (1 - (stop_loss / current_price)), 2) if stop_loss else None

    if not conditions:
        conditions.append("조건 미충족")

    summary = f"{', '.join(conditions)}, {status}"
    reason = ", ".join(conditions)

    return {
        "mode": "swing",
        "symbol": ticker,
        "entry_signal": entry_signal,
        "exit_signal": exit_signal,
        "status": status,
        "summary": summary,
        "reason": reason,
        "stop_loss": stop_loss,
        "stop_loss_pct": stop_loss_pct,
    }


def evaluate_strategy(mode, ticker, latest, df, current_price):
    if mode == "daytrade":
        result = evaluate_daytrade(ticker, latest, df, current_price)
    else:
        result = evaluate_swing(ticker, latest, df, current_price)
    return result


def format_strategy_output(result, currency_symbol, price_format):
    mode_label = "단타 (Daytrade)" if result["mode"] == "daytrade" else "스윙 (Swing)"
    status = result["status"]
    status_emoji = "⚠️" if ("매도" in status or result["exit_signal"]) else ("✅" if result["entry_signal"] else "ℹ️")

    entry_text = "✅ 발생" if result["entry_signal"] else "❌ 없음"
    exit_text = "✅ 발생" if result["exit_signal"] else "❌ 없음"

    reason_text = result.get("reason") or "조건 미충족"
    summary_text = result.get("summary") or status

    stop_loss = result.get("stop_loss")
    stop_loss_pct = result.get("stop_loss_pct")
    if stop_loss:
        stop_loss_text = f"{price_format.format(stop_loss)}{currency_symbol}"
        if stop_loss_pct is not None:
            stop_loss_text += f" (-{stop_loss_pct:.1f}%)"
    else:
        stop_loss_text = "N/A"

    lines = [
        "🧭 [전략 결과 요약]\n",
        f"📈 모드: {mode_label}",
        f"📊 종목: {result['symbol']}",
        f"💬 상태: {status_emoji} {status}",
        "────────────────────────────",
        f"🔹 진입 신호: {entry_text}",
        f"🔹 청산 신호: {exit_text} ({reason_text})",
        f"🔹 손절 기준: {stop_loss_text}",
        "────────────────────────────",
        f"💡 요약: {summary_text}",
    ]
    return "\n".join(lines)


def analyze_stock(ticker, mode="swing", use_premarket=False):
    """
    종목을 분석하고 이해하기 쉬운 신호로 출력
    
    Args:
        ticker: 종목번호(한국) 또는 티커(미국)
        mode: 'swing' 또는 'daytrade' 전략 모드
        use_premarket: True면 프리마켓/애프터마켓 가격 사용 (미국 주식만)
    """
    print("\n" + "="*80)
    print(f"🔍 종목 분석 중: {ticker}")
    print("="*80)
    
    is_us = is_us_stock(ticker)
    market_name = "미국 주식" if is_us else "한국 주식"
    
    # 종목명 가져오기
    stock_name = get_stock_name(ticker, is_us)
    print(f"📊 종목명: {stock_name} ({market_name})")
    
    # 데이터 수집
    print("\n📥 데이터 수집 중...")
    if is_us:
        df = fetch_stock_data_yahoo(ticker, period="6mo")
    else:
        df = fetch_stock_data(ticker, pages=10)
    
    if df is None or df.empty:
        print("❌ 데이터를 가져올 수 없습니다.")
        return
    
    # 장 상태 확인 및 데이터 처리
    market = "US" if is_us else "KR"
    market_closed = is_market_closed(market)
    if not market_closed and len(df) > 1:
        df = df.iloc[:-1].reset_index(drop=True)
    
    # 지표 계산
    df = calculate_ma(df, periods=[5, 20, 60])
    df = calculate_atr(df, period=14)
    df['avg_vol_20'] = df['거래량'].rolling(20, min_periods=5).mean()
    df['volume_ratio'] = df['거래량'] / df['avg_vol_20']
    df = calculate_rsi(df, period=14)
    df = calculate_macd(df)
    
    if len(df) < 20:
        print("❌ 데이터가 부족합니다 (최소 20일 필요)")
        return
    
    latest = df.iloc[-1]
    
    # 현재가 가져오기
    # 프리마켓 옵션이 있고 미국 주식이면 프리마켓 가격 사용
    current_price_raw = get_current_price(ticker, is_us, include_prepost=use_premarket and is_us)
    close_price = float(latest['종가'])
    
    # 프리마켓 가격 사용 시 표시
    price_source = ""
    if use_premarket and is_us:
        from datetime import datetime
        import pytz
        try:
            est = pytz.timezone('US/Eastern')
            now_est = datetime.now(est)
            hour = now_est.hour
            if 4 <= hour < 9 or (hour == 9 and now_est.minute < 30):
                price_source = " (프리마켓)"
            elif 16 <= hour < 20:
                price_source = " (애프터마켓)"
        except:
            price_source = " (프리마켓/애프터마켓)"
    
    if current_price_raw is None or pd.isna(current_price_raw):
        current_price = close_price
        # 실시간 가격을 가져오지 못했을 때 경고
        if is_us:
            print("   ⚠️  실시간 가격을 가져올 수 없어 종가를 사용합니다.")
    else:
        try:
            current_price = float(current_price_raw)
            # 검증: 합리적인 가격 범위 (달러/원 구분)
            if is_us:
                # 미국 주식: 0.01달러 ~ 100만 달러
                if current_price < 0.01 or current_price > 1000000:
                    current_price = close_price
                    print("   ⚠️  가져온 가격이 비정상적이어서 종가를 사용합니다.")
            else:
                # 한국 주식: 100원 ~ 10억원
                if current_price < 100 or current_price > 1000000000:
                    current_price = close_price
                    print("   ⚠️  가져온 가격이 비정상적이어서 종가를 사용합니다.")
            
            # 종가와 현재가가 같으면 추가 확인 (장 마감 후일 수 있음)
            if abs(current_price - close_price) < 0.01:
                # 장 마감 후라면 정상, 장 중이면 문제
                if is_us:
                    from datetime import datetime
                    import pytz
                    try:
                        est = pytz.timezone('US/Eastern')
                        now_est = datetime.now(est)
                        hour = now_est.hour
                        # 장 마감 시간 체크 (16:00 = 4 PM)
                        if hour < 16:  # 장 마감 전
                            print("   ⚠️  실시간 가격이 종가와 동일합니다. (장 마감 후이거나 데이터 오류 가능)")
                    except:
                        pass
        except (ValueError, TypeError):
            current_price = close_price
            if is_us:
                print("   ⚠️  가격 변환 오류로 종가를 사용합니다.")
    
    # 전일 대비 계산 시 close_price가 0이 아닌지 확인
    if close_price <= 0:
        close_price = current_price  # 안전장치
    
    # 기본 지표
    ma5 = latest['MA5'] if pd.notna(latest['MA5']) else None
    ma20 = latest['MA20'] if pd.notna(latest['MA20']) else None
    ma60 = latest['MA60'] if 'MA60' in latest.index and pd.notna(latest['MA60']) else None
    rsi = latest['RSI'] if pd.notna(latest['RSI']) else None
    volume_ratio = latest.get('volume_ratio') if pd.notna(latest.get('volume_ratio')) else None
    macd = latest.get('MACD') if pd.notna(latest.get('MACD')) else None
    macd_signal = latest.get('MACD_Signal') if pd.notna(latest.get('MACD_Signal')) else None
    currency = "원" if not is_us else "달러"
    price_format = "{:,.2f}" if is_us else "{:,.0f}"

    strategy_mode = (mode or "swing").lower()
    if strategy_mode not in {"swing", "daytrade"}:
        strategy_mode = "swing"
    strategy_result = evaluate_strategy(strategy_mode, ticker, latest, df, current_price)

    print("\n🧭 전략 모드 결과")
    print(format_strategy_output(strategy_result, currency, price_format))

    json_payload = {
        "mode": strategy_result["mode"],
        "symbol": ticker,
        "entry_signal": bool(strategy_result["entry_signal"]),
        "exit_signal": bool(strategy_result["exit_signal"]),
        "status": strategy_result["status"],
        "summary": strategy_result["summary"],
        "reason": strategy_result["reason"],
        "stop_loss": float(strategy_result["stop_loss"]) if strategy_result.get("stop_loss") else None,
        "stop_loss_pct": strategy_result.get("stop_loss_pct"),
    }
    print("\nJSON 결과:")
    print(json.dumps(json_payload, ensure_ascii=False, indent=2))
    
    # 정배열 확인
    is_perfect_alignment = False
    if ma60 is not None and ma20 is not None and ma5 is not None:
        is_perfect_alignment = (ma60 < ma20 < ma5)
    
    # 분석 수행
    entry_analysis = analyze_entry_opportunity(
        close_price, ma5, ma20, rsi, volume_ratio, is_us=is_us, current_price=current_price
    )

    if entry_analysis:
        trend_context = {
            "entry_analysis": entry_analysis,
            "price": current_price,
            "is_us": is_us,
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "ma5": ma5,
        }
        trend_context = postprocess_signal(trend_context)
        entry_analysis = trend_context.get("entry_analysis", entry_analysis)
        entry_mode = trend_context.get("entry_mode")
        max_entry_price = trend_context.get("max_entry_price")
        trend_comment = trend_context.get("comment")
    else:
        entry_mode = None
        max_entry_price = None
        trend_comment = None
    
    ma_energy_state = analyze_ma_energy_state(df, ma5, ma20)
    ma_energy_score = calculate_ma_energy_momentum_score(ma_energy_state, rsi) if ma_energy_state else 0
    
    granville_ma20 = analyze_granville_rules(df, current_price, ma_period=20)
    granville_ma5 = analyze_granville_rules(df, current_price, ma_period=5)
    
    # 결과 출력
    print("\n" + "="*80)
    print("📊 종목 분석 결과")
    print("="*80)
    
    # 1. 가격 정보
    print(f"\n💰 가격 정보")
    print(f"   종가 (확정): {price_format.format(close_price)}{currency}")
    print(f"   현재가 (실시간{price_source}): {price_format.format(current_price)}{currency}")
    # 전일 대비 계산 시 close_price가 0이 아닌지 확인
    if close_price > 0 and abs(current_price - close_price) > 0.01:  # 0.01원 이상 차이
        price_change = current_price - close_price
        price_change_pct = (price_change / close_price) * 100
        # 합리적인 범위 체크 (일일 변동폭이 -30% ~ +30% 범위 내)
        if -30 <= price_change_pct <= 30:
            change_emoji = "📈" if price_change > 0 else "📉"
            print(f"   {change_emoji} 전일 대비: {price_format.format(price_change)}{currency} ({price_change_pct:+.2f}%)")
    
    # 2. 이동평균선
    print(f"\n📈 이동평균선")
    if ma5:
        print(f"   MA5:  {price_format.format(ma5)}{currency}")
    else:
        print(f"   MA5:  N/A")
    if ma20:
        print(f"   MA20: {price_format.format(ma20)}{currency}")
    else:
        print(f"   MA20: N/A")
    if ma60:
        print(f"   MA60: {price_format.format(ma60)}{currency}")
    else:
        print(f"   MA60: N/A")
    
    if is_perfect_alignment:
        print(f"   🔥 정배열: MA60 < MA20 < MA5 (상승 추세 완벽)")
    
    # 3. 진입 판단 (이해하기 쉬운 신호)
    print(f"\n🎯 진입 판단")
    if entry_analysis:
        judgment = entry_analysis.get('judgment', '👀')
        entry_status = entry_analysis.get('entry_status', '관망')
        print(f"   {judgment} {entry_status}")

        if entry_mode:
            mode_label = {
                "pullback": "🟢 눌림 매수 모드",
                "watch": "👀 관망 모드",
                "trend": "🔥 추세 진입 모드",
            }.get(entry_mode, entry_mode)
            print(f"   {mode_label}")
            if max_entry_price is not None:
                print(f"   ➡️ 추세 상단가: {price_format.format(max_entry_price)}{currency}")
        
        if entry_analysis.get('entry_reason'):
            print(f"   ✅ 이유: {', '.join(entry_analysis['entry_reason'])}")
        
        combined_comment = trend_comment or entry_analysis.get('comment')
        if combined_comment:
            print(f"   💬 {combined_comment}")
    else:
        print("   👀 데이터 부족")
    
    # 목표가 도달 확률 계산 함수
    def calculate_reach_probability(current_price, target_price, df, rsi, macd, macd_signal, ma5, ma20, ma60, volume_ratio, max_days=60):
        """
        목표 가격 도달 확률 계산
        
        Args:
            current_price: 현재가
            target_price: 목표 가격
            df: 주가 데이터프레임
            rsi: 현재 RSI 값
            macd: 현재 MACD 값
            macd_signal: 현재 MACD Signal 값
            ma5, ma20, ma60: 이동평균선 값
            volume_ratio: 거래량 배수
            max_days: 분석할 최대 일수
        
        Returns:
            float: 도달 확률 (0-100%)
        """
        if target_price is None or current_price is None or target_price == current_price:
            return None
        
        target_change_pct = ((target_price - current_price) / current_price) * 100
        is_uptrend = target_change_pct > 0  # 상승 목표인지 하락 목표인지
        
        base_probability = 50.0  # 기본 확률
        
        # 1. 과거 패턴 분석 (최근 1년 데이터)
        historical_success = 0
        historical_count = 0
        
        if len(df) >= max_days:
            for i in range(max_days, len(df)):
                past_price = df.iloc[i]['종가']
                past_rsi = df.iloc[i].get('RSI') if 'RSI' in df.columns else None
                past_ma5 = df.iloc[i].get('MA5') if 'MA5' in df.columns else None
                past_ma20 = df.iloc[i].get('MA20') if 'MA20' in df.columns else None
                past_volume_ratio = df.iloc[i].get('volume_ratio') if 'volume_ratio' in df.columns else None
                
                # 현재 조건과 유사한지 체크
                similar_conditions = True
                similarity_score = 0
                
                # RSI 유사도 (15 이내 차이면 유사)
                if rsi is not None and past_rsi is not None:
                    rsi_diff = abs(rsi - past_rsi)
                    if rsi_diff <= 15:
                        similarity_score += 1
                    elif rsi_diff > 25:
                        similar_conditions = False
                
                # 이동평균 위치 유사도
                if ma5 and ma20 and past_ma5 and past_ma20:
                    past_above_ma = past_price > past_ma5
                    current_above_ma = current_price > ma5
                    if past_above_ma == current_above_ma:
                        similarity_score += 1
                
                # 거래량 유사도
                if volume_ratio and past_volume_ratio:
                    vol_diff = abs(volume_ratio - past_volume_ratio)
                    if vol_diff < 0.5:
                        similarity_score += 1
                
                # 유사도가 2 이상이면 유사한 조건으로 판단
                if similar_conditions and similarity_score >= 1:
                    # 목표가까지의 거리 계산
                    past_target_change_pct = abs(((target_price - past_price) / past_price) * 100)
                    
                    # 비슷한 거리의 목표가인지 확인 (30% 오차 허용)
                    if abs(past_target_change_pct - abs(target_change_pct)) <= abs(target_change_pct) * 0.3:
                        # max_days 내 목표가 도달했는지 확인
                        future_prices = df.iloc[i+1:min(i+max_days+1, len(df))]['종가']
                        if len(future_prices) > 0:
                            if is_uptrend:
                                reached = (future_prices >= target_price).any()
                            else:
                                reached = (future_prices <= target_price).any()
                            
                            historical_count += 1
                            if reached:
                                historical_success += 1
        
        if historical_count >= 5:  # 최소 5개 샘플 필요
            historical_prob = (historical_success / historical_count) * 100
            base_probability = historical_prob
        elif historical_count > 0:
            # 샘플이 적으면 가중치 낮춤
            historical_prob = (historical_success / historical_count) * 100
            base_probability = (base_probability * 0.6) + (historical_prob * 0.4)
        
        # 2. 기술적 지표 기반 조정
        indicator_bonus = 0
        
        # RSI 기반
        if rsi is not None:
            if is_uptrend:
                if 40 <= rsi <= 60:
                    indicator_bonus += 10  # 적정 구간
                elif 30 <= rsi < 40:
                    indicator_bonus += 5  # 약간 낮지만 상승 가능
                elif rsi > 70:
                    indicator_bonus -= 15  # 과열
                elif rsi < 30:
                    indicator_bonus += 3  # 과매도 후 반등 기대 (하지만 약함)
            else:  # 하락 목표
                if rsi > 70:
                    indicator_bonus += 10  # 과열 후 하락 기대
                elif rsi < 30:
                    indicator_bonus -= 15  # 이미 과매도
        
        # MACD 기반
        if macd is not None and macd_signal is not None:
            if is_uptrend and macd > macd_signal:
                indicator_bonus += 10  # 골든크로스
            elif not is_uptrend and macd < macd_signal:
                indicator_bonus += 10  # 데드크로스
            elif is_uptrend and macd < macd_signal:
                indicator_bonus -= 5  # 데드크로스 상태에서 상승 목표는 불리
            elif not is_uptrend and macd > macd_signal:
                indicator_bonus -= 5
        
        # 이동평균 정배열
        if ma60 and ma20 and ma5:
            if is_uptrend:
                if ma60 < ma20 < ma5:
                    indicator_bonus += 10  # 정배열 상승
                elif ma5 < ma20 < ma60:
                    indicator_bonus -= 10  # 역배열에서 상승 목표는 불리
            else:  # 하락 목표
                if ma5 < ma20 < ma60:
                    indicator_bonus += 10  # 역배열 하락
                elif ma60 < ma20 < ma5:
                    indicator_bonus -= 10
        
        # 거래량
        if volume_ratio:
            if volume_ratio > 1.3:
                indicator_bonus += 5  # 거래량 증가
            elif volume_ratio < 0.7:
                indicator_bonus -= 5  # 거래량 감소
        
        # 3. 거리 기반 조정
        distance_factor = 1.0
        abs_target_pct = abs(target_change_pct)
        
        if abs_target_pct < 1:
            distance_factor = 1.3  # 매우 가까운 거리는 확률 증가
        elif abs_target_pct < 2:
            distance_factor = 1.2
        elif abs_target_pct < 5:
            distance_factor = 1.0
        elif abs_target_pct < 10:
            distance_factor = 0.85
        elif abs_target_pct < 20:
            distance_factor = 0.7
        else:
            distance_factor = 0.5  # 먼 거리는 확률 감소
        
        # 4. 최종 확률 계산
        final_probability = (base_probability + indicator_bonus) * distance_factor
        
        # 0-100% 범위로 제한
        final_probability = max(0, min(100, final_probability))
        
        return round(final_probability, 1)
    
    # 예상 기간 계산 함수
    def estimate_days_to_target(current_price, target_price, df, lookback_days=20):
        """
        목표 가격까지 도달하는데 걸릴 예상 일수 계산
        
        Args:
            current_price: 현재가
            target_price: 목표 가격
            df: 주가 데이터프레임
            lookback_days: 최근 N일간의 변동률을 분석할 기간
        
        Returns:
            float: 예상 일수 (None if 계산 불가)
        """
        if target_price is None or current_price is None or target_price == current_price:
            return None
        
        # 목표 가격까지 필요한 변동률
        target_change_pct = abs(((target_price - current_price) / current_price) * 100)
        
        # 목표까지 거리가 매우 가까우면 (1% 이내) 빠른 도달 가능
        if target_change_pct < 1.0:
            # 최근 변동성을 기반으로 하되, 최소한 보수적으로 1-2일 이내로 예측
            if len(df) >= 5:
                recent_df = df.tail(min(10, len(df))).copy()
                recent_df['pct_change'] = recent_df['종가'].pct_change() * 100
                recent_volatility = recent_df['pct_change'].abs().mean()
                
                # 최근 변동성이 매우 낮으면 (0.1% 미만) 보수적으로 0.5% 일일 변동 가정
                if recent_volatility < 0.1:
                    daily_change = 0.5  # 최소 변동성 가정
                else:
                    daily_change = max(recent_volatility, 0.3)  # 최소 0.3% 보장
                
                estimated_days = target_change_pct / daily_change
                # 매우 가까운 거리는 최대 3일로 제한
                return min(estimated_days, 3.0)
            else:
                # 데이터 부족 시 보수적으로 1-2일
                return min(target_change_pct / 0.5, 2.0)
        
        # 목표까지 거리가 멀면 (1% 이상) 기존 로직 사용
        if len(df) < lookback_days:
            lookback_days = len(df)
        
        if lookback_days < 5:
            return None
        
        # 최근 N일간의 일일 변동률(%) 계산
        recent_df = df.tail(lookback_days).copy()
        if len(recent_df) < 2:
            return None
        
        # 일일 변동률 계산
        recent_df['pct_change'] = recent_df['종가'].pct_change() * 100
        avg_daily_change = recent_df['pct_change'].abs().mean()  # 절대값 평균 사용
        
        # 평균 일일 변동률이 너무 작으면 최소값 보장
        if avg_daily_change < 0.1:
            avg_daily_change = 0.3  # 최소 0.3% 일일 변동 가정
        
        # 예상 일수 = 목표 변동률 / 평균 일일 변동률
        estimated_days = target_change_pct / avg_daily_change
        
        # 합리적인 범위로 제한 (0.1일 ~ 60일)
        # 60일 이상이면 예측 의미 없음
        if estimated_days > 60:
            return None  # 예측 불가로 표시
        
        estimated_days = max(0.1, min(60, estimated_days))
        
        return estimated_days
    
    def format_estimated_time(estimated_days):
        """
        예상 기간을 포맷팅 (1일 이내는 시간으로 표시)
        
        Args:
            estimated_days: 예상 일수 (float)
        
        Returns:
            str: 포맷팅된 문자열
        """
        if estimated_days is None:
            return ""
        
        if estimated_days < 1.0:
            # 1일 미만: 시간으로 변환
            hours = int(estimated_days * 24)
            if hours == 0:
                minutes = int(estimated_days * 24 * 60)
                if minutes < 1:
                    return "즉시"
                return f"약 {minutes}분"
            else:
                return f"약 {hours}시간"
        elif estimated_days < 2.0:
            # 1~2일: 일과 시간으로 표시
            days = int(estimated_days)
            hours = int((estimated_days - days) * 24)
            if hours > 0:
                return f"약 {days}일 {hours}시간"
            else:
                return f"약 {days}일"
        else:
            # 2일 이상: 일로만 표시
            return f"약 {int(estimated_days)}일"
    
    # 4. 매수/손절/익절 추천 가격
    print(f"\n💡 투자 전략 가격 (실시간 가격 기준)")
    if entry_analysis and ma5 and ma20:
        buy_range_1_low = entry_analysis.get('buy_range_1_low', 0)
        buy_range_1_high = entry_analysis.get('buy_range_1_high', 0)
        buy_range_2_low = entry_analysis.get('buy_range_2_low', 0)
        buy_range_2_high = entry_analysis.get('buy_range_2_high', 0)
        stop_loss = entry_analysis.get('stop_loss_price', 0)
        
        # 매수 추천 가격
        print(f"\n📈 매수 추천 가격")
        if buy_range_1_low > 0 and buy_range_1_high > 0:
            buy_1_mid = (buy_range_1_low + buy_range_1_high) / 2
            in_range = entry_analysis.get('in_buy_range_1', False)
            status = "✅ 현재 매수 구간 내" if in_range else "⏳ 대기 중"
            # 현재가와 비교 설명 (매수 추천가는 항상 현재가보다 낮거나 같아야 함)
            diff_pct = ((buy_1_mid - current_price) / current_price) * 100
            if buy_1_mid > current_price:
                note = f" ⚠️ 경고: 현재가보다 높음! (로직 오류)"
            elif abs(diff_pct) < 0.5:
                note = f" (현재가와 거의 동일 - 즉시 매수 가능)"
            else:
                note = f" (현재가보다 {abs(diff_pct):.1f}% 낮음 - 지지선 근처 매수)"
            
            # 도달 확률 계산
            prob = calculate_reach_probability(
                current_price, buy_1_mid, df, rsi, macd, macd_signal, 
                ma5, ma20, ma60, volume_ratio
            )
            prob_text = f" (도달 확률: {prob}%)" if prob is not None else ""
            
            # 예상 일수 계산 (매수가까지 도달하는데 걸리는 시간)
            days_to_buy = estimate_days_to_target(current_price, buy_1_mid, df)
            days_text = ""
            if days_to_buy is not None:
                if days_to_buy < 1.0:
                    hours = int(days_to_buy * 24)
                    if hours > 0:
                        days_text = f" (예상 도달: {hours}시간 내)"
                    else:
                        days_text = " (예상 도달: 당일 가능)"
                elif days_to_buy < 7:
                    days_text = f" (예상 도달: 약 {int(days_to_buy)}일)"
                else:
                    days_text = f" (예상 도달: 약 {int(days_to_buy)}일)"
            
            print(f"   1차 매수: {price_format.format(buy_1_mid)}{currency} (구간: {price_format.format(buy_range_1_low)} ~ {price_format.format(buy_range_1_high)}{currency}) {status}{note}{days_text}{prob_text}")
        
        if buy_range_2_low > 0 and buy_range_2_high > 0:
            buy_2_mid = (buy_range_2_low + buy_range_2_high) / 2
            in_range = entry_analysis.get('in_buy_range_2', False)
            status = "✅ 현재 매수 구간 내" if in_range else "⏳ 대기 중"
            # 현재가와 비교 설명 (매수 추천가는 항상 현재가보다 낮거나 같아야 함)
            diff_pct = ((buy_2_mid - current_price) / current_price) * 100
            if buy_2_mid > current_price:
                note = f" ⚠️ 경고: 현재가보다 높음! (로직 오류)"
            elif abs(diff_pct) < 0.5:
                note = f" (현재가와 거의 동일 - 즉시 매수 가능)"
            else:
                note = f" (현재가보다 {abs(diff_pct):.1f}% 낮음 - 지지선 근처 매수)"
            
            # 도달 확률 계산
            prob = calculate_reach_probability(
                current_price, buy_2_mid, df, rsi, macd, macd_signal, 
                ma5, ma20, ma60, volume_ratio
            )
            prob_text = f" (도달 확률: {prob}%)" if prob is not None else ""
            
            # 예상 일수 계산
            days_to_buy = estimate_days_to_target(current_price, buy_2_mid, df)
            days_text = ""
            if days_to_buy is not None:
                if days_to_buy < 1.0:
                    hours = int(days_to_buy * 24)
                    if hours > 0:
                        days_text = f" (예상 도달: {hours}시간 내)"
                    else:
                        days_text = " (예상 도달: 당일 가능)"
                elif days_to_buy < 7:
                    days_text = f" (예상 도달: 약 {int(days_to_buy)}일)"
                else:
                    days_text = f" (예상 도달: 약 {int(days_to_buy)}일)"
            
            print(f"   2차 매수: {price_format.format(buy_2_mid)}{currency} (구간: {price_format.format(buy_range_2_low)} ~ {price_format.format(buy_range_2_high)}{currency}) {status}{note}{days_text}{prob_text}")
        
        # 손절 추천 가격
        print(f"\n🛑 손절 추천 가격")
        if buy_range_1_low > 0 and buy_range_1_high > 0 and buy_range_2_low > 0 and buy_range_2_high > 0:
            buy_1_mid = (buy_range_1_low + buy_range_1_high) / 2
            buy_2_mid = (buy_range_2_low + buy_range_2_high) / 2
            
            # 1차 매수 기준 손절가 (매수 가격의 3% 하락 또는 MA20 × 0.97 중 더 보수적인 값)
            stop_loss_1 = min(buy_1_mid * 0.97, stop_loss) if stop_loss > 0 else buy_1_mid * 0.97
            below_1 = current_price < stop_loss_1
            status_1 = "⚠️ 손절 구간 도달" if below_1 else "✅ 안전"
            
            print(f"   1차 손절: {price_format.format(stop_loss_1)}{currency} (1차 매수 대비 -{(1-stop_loss_1/buy_1_mid)*100:.1f}%) {status_1}")
            
            # 2차 매수 기준 손절가
            stop_loss_2 = min(buy_2_mid * 0.97, stop_loss) if stop_loss > 0 else buy_2_mid * 0.97
            below_2 = current_price < stop_loss_2
            status_2 = "⚠️ 손절 구간 도달" if below_2 else "✅ 안전"
            
            print(f"   2차 손절: {price_format.format(stop_loss_2)}{currency} (2차 매수 대비 -{(1-stop_loss_2/buy_2_mid)*100:.1f}%) {status_2}")
        elif stop_loss > 0:
            below = entry_analysis.get('below_stop_loss', False)
            status = "⚠️ 손절 구간 도달" if below else "✅ 안전"
            
            print(f"   손절 기준: {price_format.format(stop_loss)}{currency} {status}")
        
        # 익절 추천 가격
        print(f"\n💰 익절 추천 가격")
        buy_1_mid = (buy_range_1_low + buy_range_1_high) / 2 if (buy_range_1_low > 0 and buy_range_1_high > 0) else 0
        buy_2_mid = (buy_range_2_low + buy_range_2_high) / 2 if (buy_range_2_low > 0 and buy_range_2_high > 0) else 0
        
        # 1차 익절 (단기: 5-8%)
        if buy_1_mid > 0:
            take_profit_1 = buy_1_mid * 1.065  # 6.5% 수익
            if rsi and rsi > 70:
                take_profit_1 = buy_1_mid * 1.05  # RSI 과열 시 5% 수익
            
            # 목표 수익률만 표시 (매수 대비)
            target_pct_1 = (take_profit_1 / buy_1_mid - 1) * 100
            
            if current_price >= take_profit_1:
                status_1 = "✅ 도달"
            else:
                # 현재 매수 대비 수익률
                current_pct_from_buy = ((current_price - buy_1_mid) / buy_1_mid) * 100 if buy_1_mid > 0 else 0
                remaining_pct = target_pct_1 - current_pct_from_buy
                if remaining_pct > 0:
                    status_1 = f"⏳ 현재 {current_pct_from_buy:+.1f}% (목표까지 {remaining_pct:.1f}% 남음)"
                else:
                    status_1 = f"⏳ 현재 {current_pct_from_buy:+.1f}%"
            
            # 도달 확률 계산
            prob = calculate_reach_probability(
                current_price, take_profit_1, df, rsi, macd, macd_signal, 
                ma5, ma20, ma60, volume_ratio
            )
            prob_text = f" (도달 확률: {prob}%)" if prob is not None else ""
            
            # 예상 일수 계산
            days_to_tp = estimate_days_to_target(current_price, take_profit_1, df)
            days_text = ""
            if days_to_tp is not None:
                if days_to_tp < 1.0:
                    hours = int(days_to_tp * 24)
                    if hours > 0:
                        days_text = f" (예상 도달: {hours}시간 내)"
                    else:
                        days_text = " (예상 도달: 당일 가능)"
                elif days_to_tp < 7:
                    days_text = f" (예상 도달: 약 {int(days_to_tp)}일)"
                else:
                    days_text = f" (예상 도달: 약 {int(days_to_tp)}일)"
            
            print(f"   1차 익절: {price_format.format(take_profit_1)}{currency} (목표: +{target_pct_1:.1f}%) {status_1}{days_text}{prob_text}")
        
        # 2차 익절 (중기: 10-15%)
        if buy_2_mid > 0:
            base_take_profit_2 = buy_2_mid * 1.125  # 기본: 12.5% 수익
            if buy_1_mid > 0 and 'take_profit_1' in locals():
                # 1차 익절가보다 최소 3% 이상 높게 유지
                min_take_profit_2 = take_profit_1 * 1.03
                take_profit_2 = max(base_take_profit_2, min_take_profit_2)
            else:
                take_profit_2 = base_take_profit_2

            target_pct_2 = (take_profit_2 / buy_2_mid - 1) * 100
            
            if current_price >= take_profit_2:
                status_2 = "✅ 도달"
            else:
                # 현재 매수 대비 수익률
                current_pct_from_buy = ((current_price - buy_2_mid) / buy_2_mid) * 100 if buy_2_mid > 0 else 0
                remaining_pct = target_pct_2 - current_pct_from_buy
                if remaining_pct > 0:
                    status_2 = f"⏳ 현재 {current_pct_from_buy:+.1f}% (목표까지 {remaining_pct:.1f}% 남음)"
                else:
                    status_2 = f"⏳ 현재 {current_pct_from_buy:+.1f}%"
            
            # 도달 확률 계산
            prob = calculate_reach_probability(
                current_price, take_profit_2, df, rsi, macd, macd_signal, 
                ma5, ma20, ma60, volume_ratio
            )
            prob_text = f" (도달 확률: {prob}%)" if prob is not None else ""
            
            # 예상 일수 계산
            days_to_tp = estimate_days_to_target(current_price, take_profit_2, df)
            days_text = ""
            if days_to_tp is not None:
                if days_to_tp < 1.0:
                    hours = int(days_to_tp * 24)
                    if hours > 0:
                        days_text = f" (예상 도달: {hours}시간 내)"
                    else:
                        days_text = " (예상 도달: 당일 가능)"
                elif days_to_tp < 7:
                    days_text = f" (예상 도달: 약 {int(days_to_tp)}일)"
                else:
                    days_text = f" (예상 도달: 약 {int(days_to_tp)}일)"
            
            print(f"   2차 익절: {price_format.format(take_profit_2)}{currency} (목표: +{target_pct_2:.1f}%) {status_2}{days_text}{prob_text}")
        
        # 종합 익절 (평균 매수가 기준 장기: 20-25%)
        if buy_1_mid > 0 and buy_2_mid > 0:
            avg_buy_price = buy_1_mid * 0.6 + buy_2_mid * 0.4
            take_profit_long = avg_buy_price * 1.225  # 22.5% 수익
            target_pct_long = (take_profit_long / avg_buy_price - 1) * 100
            
            if current_price >= take_profit_long:
                status_long = "✅ 도달"
            else:
                # 현재 평균 매수 대비 수익률
                current_pct_from_avg = ((current_price - avg_buy_price) / avg_buy_price) * 100 if avg_buy_price > 0 else 0
                remaining_pct = target_pct_long - current_pct_from_avg
                if remaining_pct > 0:
                    status_long = f"⏳ 현재 {current_pct_from_avg:+.1f}% (목표까지 {remaining_pct:.1f}% 남음)"
                else:
                    status_long = f"⏳ 현재 {current_pct_from_avg:+.1f}%"
            
            # 도달 확률 계산
            prob = calculate_reach_probability(
                current_price, take_profit_long, df, rsi, macd, macd_signal, 
                ma5, ma20, ma60, volume_ratio
            )
            prob_text = f" (도달 확률: {prob}%)" if prob is not None else ""
            
            # 예상 일수 계산
            days_to_tp = estimate_days_to_target(current_price, take_profit_long, df)
            days_text = ""
            if days_to_tp is not None:
                if days_to_tp < 7:
                    days_text = f" (예상 도달: 약 {int(days_to_tp)}일)"
                elif days_to_tp < 30:
                    days_text = f" (예상 도달: 약 {int(days_to_tp)}일)"
                else:
                    days_text = f" (예상 도달: 약 {int(days_to_tp)}일)"
            
            print(f"   장기 익절: {price_format.format(take_profit_long)}{currency} (목표: +{target_pct_long:.1f}%) {status_long}{days_text}{prob_text}")
    else:
        print("   데이터 부족")
    
    # 상한가/하한가 예측 (한국 주식만)
    if not is_us and close_price > 0:
        print(f"\n🚨 상한가/하한가 예측")
        # 한국 주식: 상한가 +30%, 하한가 -30%
        limit_up_price = close_price * 1.30
        limit_down_price = close_price * 0.70
        
        limit_up_pct = ((limit_up_price - current_price) / current_price) * 100
        limit_down_pct = ((limit_down_price - current_price) / current_price) * 100
        
        # 상한가 정보
        if current_price >= limit_up_price:
            limit_up_status = "✅ 상한가 도달"
        else:
            limit_up_status = f"⏳ 상한가까지 {limit_up_pct:.1f}% 남음"
        
        print(f"   상한가: {price_format.format(limit_up_price)}{currency} (+30%) {limit_up_status}")
        
        # 하한가 정보
        if current_price <= limit_down_price:
            limit_down_status = "⚠️ 하한가 도달"
        else:
            limit_down_status = f"⏳ 하한가까지 {abs(limit_down_pct):.1f}% 남음"
        
        print(f"   하한가: {price_format.format(limit_down_price)}{currency} (-30%) {limit_down_status}")
    
    # 5. RSI 상태
    print(f"\n📊 RSI 상태")
    if rsi is not None:
        if rsi < 30:
            rsi_status = "🚨 과매도"
        elif 30 <= rsi < 45:
            rsi_status = "📉 낮음"
        elif 45 <= rsi <= 55:
            rsi_status = "✅ 적정"
        elif 55 < rsi <= 70:
            rsi_status = "📈 높음"
        else:
            rsi_status = "🚨 과열"
        print(f"   RSI: {rsi:.2f} {rsi_status}")
    else:
        print("   RSI: N/A")
    
    # 6. 거래량 상태
    print(f"\n📊 거래량 상태")
    if volume_ratio is not None:
        if volume_ratio < 0.8:
            vol_status = "🔹 저조"
        elif 0.8 <= volume_ratio < 1.2:
            vol_status = "⚖️ 정상"
        elif 1.2 <= volume_ratio <= 2.0:
            vol_status = "✅ 증가"
        elif 2.0 < volume_ratio <= 3.0:
            vol_status = "📊 활발"
        else:
            vol_status = "🚨 폭증"
        print(f"   거래량 배수: {volume_ratio:.2f}배 {vol_status}")
    else:
        print("   거래량: N/A")
    
    # 7. MA Energy State
    print(f"\n🧭 이평선 에너지 상태")
    if ma_energy_state:
        state_emoji = ma_energy_state.get('emoji', '⚫')
        state_name = ma_energy_state.get('state_name', 'N/A')
        gap_pct = ma_energy_state.get('gap_pct', 0)
        interpretation = ma_energy_state.get('interpretation', '')
        strategy = ma_energy_state.get('strategy', '')
        
        print(f"   {state_emoji} {state_name} (격차: {gap_pct:+.2f}%)")
        print(f"   💡 {interpretation}")
        print(f"   📋 전략: {strategy}")
        print(f"   ⚡ Energy Momentum Score: {ma_energy_score}/100")
    else:
        print("   데이터 부족")
    
    # 8. 그랜빌 법칙 (신호 우선순위 처리)
    print(f"\n📐 그랜빌 법칙")
    
    # 신호 우선순위: 매도 신호 > 상위 기간 (MA20) > 하위 기간 (MA5)
    def get_signal_priority(granville_result):
        """신호 우선순위 계산 (낮을수록 우선순위 높음)"""
        if not granville_result:
            return 999  # 최저 우선순위
        
        signal = granville_result.get('signal', '')
        rule = granville_result.get('rule', 0)
        
        # 매도 신호가 매수 신호보다 우선
        if '매도' in signal:
            return rule  # 매도 1~4: 5~8
        elif '매수' in signal:
            return rule + 10  # 매수 1~4: 11~14
        
        return 999
    
    # 신호 우선순위 계산
    priority_ma20 = get_signal_priority(granville_ma20)
    priority_ma5 = get_signal_priority(granville_ma5)
    
    # 우선순위가 높은 신호 선택
    if priority_ma20 < priority_ma5:
        # MA20이 우선
        primary_signal = granville_ma20
        secondary_signal = granville_ma5
        primary_label = "MA20 기준"
        secondary_label = "MA5 기준"
    else:
        # MA5가 우선 (또는 둘 다 같으면 MA20 우선)
        primary_signal = granville_ma5 if priority_ma5 < 999 else granville_ma20
        secondary_signal = granville_ma20 if priority_ma5 < 999 else granville_ma5
        primary_label = "MA5 기준" if priority_ma5 < 999 else "MA20 기준"
        secondary_label = "MA20 기준" if priority_ma5 < 999 else "MA5 기준"
    
    # 주 신호 출력
    if primary_signal:
        rule = primary_signal.get('rule', 0)
        signal = primary_signal.get('signal', '')
        description = primary_signal.get('description', '')
        strength = primary_signal.get('strength', '')
        emoji = primary_signal.get('emoji', '')
        print(f"   {primary_label}: {emoji} {signal} - {description} ({strength}) ⭐")
    else:
        print(f"   {primary_label}: 해당 사항 없음")
    
    # 보조 신호 출력 (주 신호와 다른 경우만)
    if secondary_signal and secondary_signal != primary_signal:
        rule = secondary_signal.get('rule', 0)
        signal = secondary_signal.get('signal', '')
        description = secondary_signal.get('description', '')
        strength = secondary_signal.get('strength', '')
        emoji = secondary_signal.get('emoji', '')
        
        # 주 신호와 반대 신호인 경우 명시
        primary_signal_name = primary_signal.get('signal', '') if primary_signal else ''
        if ('매도' in primary_signal_name and '매수' in signal) or ('매수' in primary_signal_name and '매도' in signal):
            print(f"   {secondary_label}: {emoji} {signal} - {description} ({strength}) 💡 단기 조정 중")
        else:
            print(f"   {secondary_label}: {emoji} {signal} - {description} ({strength})")
    elif secondary_signal and secondary_signal == primary_signal:
        # 같은 신호면 중복 출력 안 함
        pass
    else:
        if secondary_signal is None:
            print(f"   {secondary_label}: 해당 사항 없음")
    
    # 9. MACD 상태
    print(f"\n📊 MACD 상태")
    if macd is not None and macd_signal is not None:
        macd_gap = macd - macd_signal
        if macd > macd_signal:
            macd_status = "✅ 골든크로스 (상승 신호)"
        else:
            macd_status = "🚫 데드크로스 (하락 신호)"
        print(f"   MACD: {macd:.2f}, Signal: {macd_signal:.2f}")
        print(f"   {macd_status} (격차: {macd_gap:+.2f})")
    else:
        print("   데이터 부족")
    
    # 10. 스코어 요약
    print(f"\n🧮 스코어 요약")
    total_score = 0
    score_details = []

    if ma5 and ma20:
        ma_gap_pct = ((ma5 - ma20) / ma20) * 100 if ma20 > 0 else 0
        if ma5 >= ma20:
            gc_score = 40
            gc_desc = "✅ 골든크로스 완료"
        elif ma_gap_pct >= -2:
            gc_score = 30
            gc_desc = "⏳ 골든크로스 직전"
        elif ma_gap_pct >= -5:
            gc_score = 20
            gc_desc = "👀 골든크로스 대기"
        else:
            gc_score = 0
            gc_desc = "🚫 골든크로스 멀음"
        total_score += gc_score
        score_details.append(f"{gc_desc} ({gc_score}점)")

    if rsi is not None:
        if 45 <= rsi <= 55:
            rsi_score = 30
            rsi_desc = "✅ 적정"
        elif 40 <= rsi < 45 or 55 < rsi <= 60:
            rsi_score = 20
            rsi_desc = "📊 보통"
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            rsi_score = 10
            rsi_desc = "⚠️ 주의"
        else:
            rsi_score = 0
            rsi_desc = "🚫 비추천"
        total_score += rsi_score
        score_details.append(f"RSI {rsi_desc} ({rsi_score}점)")

    if volume_ratio is not None:
        if 1.2 <= volume_ratio <= 2.0:
            vol_score = 30
            vol_desc = "✅ 증가"
        elif 1.0 <= volume_ratio < 1.2 or 2.0 < volume_ratio <= 2.5:
            vol_score = 20
            vol_desc = "📊 보통"
        elif 0.8 <= volume_ratio < 1.0 or 2.5 < volume_ratio <= 3.0:
            vol_score = 10
            vol_desc = "⚠️ 주의"
        else:
            vol_score = 0
            vol_desc = "🚫 비정상"
        total_score += vol_score
        score_details.append(f"거래량 {vol_desc} ({vol_score}점)")

    print(f"   종합 점수: {total_score}/100")
    if score_details:
        print("   세부 항목:")
        for detail in score_details:
            print(f"      - {detail}")

    if total_score >= 80:
        algo_judgment = "🟢 매수 추천"
    elif total_score >= 60:
        algo_judgment = "🟡 관망 후 매수"
    elif total_score >= 40:
        algo_judgment = "🟠 신중 검토"
    else:
        algo_judgment = "🔴 비추천"
    print(f"   알고리즘 판단: {algo_judgment}")

    phase_label, phase_note = detect_market_phase(latest)
    print("\n📘 공부용 해설")
    print(f"   현재 레짐: {phase_label}")
    print(f"   해석: {phase_note}\n")

    explain_ma(ma5, ma20, ma60)
    explain_macd(macd, macd_signal)
    explain_rsi(rsi)

    value_formatter = (lambda val: price_format.format(val)) if price_format else (lambda val: f"{val:.2f}")
    explain_atr(close_price, latest.get('ATR14'), value_formatter, currency, multiplier=2.0)

    explain_conclusion(phase_label, ma5, ma20, ma60, macd, macd_signal, rsi)
    
    print("="*80)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='단기/스윙용 종목 탐색기 - 프리마켓 가격 기준 분석 지원',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python stock_scanner.py 005930              # 삼성전자
  python stock_scanner.py AAPL                # Apple (정규장 가격)
  python stock_scanner.py AAPL --premarket     # Apple (프리마켓/애프터마켓 가격)
  python stock_scanner.py AAPL -p              # Apple (프리마켓 가격, 축약형)
        """
    )
    parser.add_argument('ticker', nargs='?', help='종목번호 또는 티커')
    parser.add_argument('-p', '--premarket', action='store_true', 
                       help='프리마켓/애프터마켓 가격 사용 (미국 주식만)')
    parser.add_argument('--mode', choices=['swing', 'daytrade'], default='swing',
                       help='전략 모드 선택 (기본값: swing)')
    
    args = parser.parse_args()
    
    if not args.ticker:
        print("="*80)
        print("📊 단기/스윙용 종목 탐색기")
        print("="*80)
        print("\n사용법:")
        print("  python stock_scanner.py <종목번호 또는 티커>")
        print("  python stock_scanner.py <티커> --premarket  # 프리마켓 가격 사용 (미국 주식)")
        print("\n예시:")
        print("  python stock_scanner.py 005930              # 삼성전자")
        print("  python stock_scanner.py AAPL                # Apple")
        print("  python stock_scanner.py AAPL --premarket     # Apple (프리마켓 가격)")
        print("  python stock_scanner.py AAPL -p              # Apple (프리마켓 가격, 축약형)")
        print("="*80)
        
        # 대화형 모드
        while True:
            try:
                user_input = input("\n종목번호 또는 티커를 입력하세요 (종료: q, 프리마켓: -p 추가, 모드: mode=swing/daytrade): ").strip()
                if user_input.lower() == 'q':
                    break
                if user_input:
                    parts = user_input.split()
                    ticker = parts[0]
                    use_premarket = False
                    mode_override = args.mode
                    i = 1
                    while i < len(parts):
                        token = parts[i]
                        if token in {'--premarket', '-p'}:
                            use_premarket = True
                        elif token.startswith('mode='):
                            mode_override = token.split('=', 1)[1].lower()
                        elif token in {'--mode', 'mode'}:
                            if i + 1 < len(parts):
                                mode_override = parts[i + 1].lower()
                                i += 1
                        i += 1
                    if mode_override not in {'swing', 'daytrade'}:
                        mode_override = args.mode
                    analyze_stock(ticker, mode=mode_override, use_premarket=use_premarket)
            except KeyboardInterrupt:
                print("\n\n종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
    else:
        analyze_stock(args.ticker, mode=args.mode, use_premarket=args.premarket)


if __name__ == "__main__":
    main()

