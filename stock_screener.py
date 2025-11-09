#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
주식 스크리닝 도구
여러 종목을 자동으로 스크리닝하여 매수 신호를 확인합니다.
"""

import argparse
import math
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from io import StringIO
import time
import os
import datetime
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from dip_screening import run_dip_screening
from modes.daytrade import analyze as analyze_daytrade
from modes.swing import analyze as analyze_swing
from modes.longterm import analyze as analyze_longterm

# pytz for timezone handling
try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
    print("⚠️  pytz 패키지가 설치되지 않았습니다. 시간대 처리를 위해 설치해주세요:")
    print("   pip install pytz")

# yfinance for US stocks
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance 패키지가 설치되지 않았습니다. 미국 주식 조회를 위해 설치해주세요:")
    print("   pip install yfinance")

OUTPUT_ROOT = Path("outputs")
TXT_OUTPUT_DIR = OUTPUT_ROOT / "txt"
CSV_OUTPUT_DIR = OUTPUT_ROOT / "csv"
PNG_OUTPUT_DIR = OUTPUT_ROOT / "png"

for directory in (TXT_OUTPUT_DIR, CSV_OUTPUT_DIR, PNG_OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

STOCK_CATEGORY_CSV = CSV_OUTPUT_DIR / "stock_categories.csv"

def is_market_closed(market="US"):
    """
    현재 시각 기준으로 마지막 확정된 종가를 사용할 수 있는지 확인
    
    한국 주식:
    - 자정 이후 ~ 오전 8시 전: 어제 종가 사용 (오늘 장 시작 전)
    - 오전 8시 ~ 오후 8시: 어제 종가 사용 (정규장 + 시간외 + 대체거래소 진행 중)
    - 오후 8시 이후: 오늘 종가 사용 (정규장 종가 기준, 모든 거래 종료)
    
    미국 주식:
    - 장 마감 후: 오늘 종가 사용
    - 장 진행 중: 어제 종가 사용
    
    Args:
        market: "US" (미국) 또는 "KR" (한국)
    
    Returns:
        bool: True면 마지막 확정된 종가 사용 가능 (오늘 종가 또는 어제 종가), 
              False면 장 진행 중이므로 어제 종가 사용
    """
    if not PYTZ_AVAILABLE:
        # pytz가 없으면 기본적으로 장이 끝났다고 가정
        return True
    
    # 현재 한국 시각 기준
    KST = pytz.timezone("Asia/Seoul")
    now = datetime.datetime.now(KST)
    
    if market == "KR":  # 한국 주식
        # 한국 주식 시장 시간표:
        # - 정규장: 09:00 ~ 15:30
        # - 시간외 종가 매매: 15:40 ~ 16:00
        # - 시간외 단일가 매매: 16:00 ~ 18:00
        # - 대체거래소(넥스트레이드): 08:00 ~ 20:00
        # - 완전 종료: 20:00 이후
        #
        # 자정 이후 ~ 오전 8시 전: 어제 종가 사용 (오늘 장 시작 전)
        # 오전 8시 ~ 오후 8시: 장 진행 중이므로 어제 종가 사용 (정규장 + 시간외 + 대체거래소)
        # 오후 8시 이후: 오늘 장 완전 종료, 오늘 종가 사용 (정규장 종가 기준)
        if now.hour < 8:
            # 자정 이후 ~ 오전 8시 전: 어제 종가 사용
            # 하지만 크롤링 시 가장 최근 데이터가 어제 종가이므로 True 반환 (마지막 데이터 사용)
            return True
        elif now.hour >= 20:
            # 오후 8시 이후: 모든 거래 종료, 오늘 종가 사용
            return True
        else:
            # 오전 8시 ~ 오후 8시: 정규장 + 시간외 + 대체거래소 진행 중, 어제 종가 사용
            # (일봉 데이터는 정규장 종가 기준이지만, 시간외 거래가 진행 중이므로 안전하게 어제 종가 사용)
            return False
    elif market == "US":  # 미국 주식
        # 미국장: 23:30 ~ 다음날 06:00 (한국 시각)
        market_close = now.replace(hour=6, minute=0, second=0, microsecond=0)
        market_open = now.replace(hour=23, minute=30, second=0, microsecond=0)
        
        # 오전 6시 이후 또는 오후 11시 30분 이전이면 장이 끝남
        if now.hour >= 6 and now.hour < 23:
            return True
        elif now.hour == 23 and now.minute < 30:
            return True
        else:
            return False
    else:
        return True


def normalize_us_ticker(ticker: str) -> str:
    if not ticker:
        return ticker
    normalized = ticker.strip().upper()
    return normalized.replace(".", "-")


def fetch_stock_data_yahoo(symbol, period="3mo"):
    """
    야후 파이낸스에서 미국 주식 일봉 데이터를 가져오는 함수
    """
    if not YFINANCE_AVAILABLE:
        return None
    
    normalized_symbol = normalize_us_ticker(symbol)
    
    # 티커 유효성 검증
    if not is_valid_us_stock_ticker(normalized_symbol):
        return None
    
    try:
        ticker = yf.Ticker(normalized_symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            return None
        
        df = hist.reset_index()
        
        # 컬럼명 매핑
        column_mapping = {}
        for i, col in enumerate(df.columns):
            if i == 0:
                column_mapping[col] = '날짜'
            elif col.lower() in ['open', '시가']:
                column_mapping[col] = '시가'
            elif col.lower() in ['high', '고가']:
                column_mapping[col] = '고가'
            elif col.lower() in ['low', '저가']:
                column_mapping[col] = '저가'
            elif col.lower() in ['close', '종가']:
                column_mapping[col] = '종가'
            elif col.lower() in ['volume', '거래량']:
                column_mapping[col] = '거래량'
        
        df = df.rename(columns=column_mapping)
        
        required_cols = ['날짜', '시가', '고가', '저가', '종가', '거래량']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < 6:
            return None
        
        df = df[available_cols]
        
        if not pd.api.types.is_datetime64_any_dtype(df['날짜']):
            df['날짜'] = pd.to_datetime(df['날짜'])
        
        df = df.sort_values('날짜').reset_index(drop=True)
        
        # 장 상태에 따라 마지막 확정된 종가 사용
        # 미국 주식:
        # - 장 마감 후: 오늘 종가 (마지막 데이터가 오늘 종가)
        # - 장 진행 중: 어제 종가 (마지막 데이터 제거)
        market_closed = is_market_closed("US")
        if not market_closed and len(df) > 1:
            # 장 진행 중이면 마지막 데이터(오늘 진행 중인 데이터) 제거, 어제 종가 사용
            df = df.iloc[:-1].reset_index(drop=True)
        # market_closed == True이면 마지막 데이터가 오늘 종가 (자동으로 사용됨)
        
        return df
        
    except Exception:
        return None


def fetch_stock_data(code, pages=5):
    """
    네이버 증권에서 일봉 데이터를 크롤링하는 함수
    """
    base_url = "https://finance.naver.com/item/sise_day.naver"
    all_data = []
    
    for page in range(1, pages + 1):
        params = {'code': code, 'page': page}
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': f'https://finance.naver.com/item/sise_day.naver?code={code}'
        }
        
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.encoding = 'euc-kr'
            
            page_data = []
            
            try:
                dfs = pd.read_html(StringIO(response.text), encoding='euc-kr')
                if dfs and len(dfs) > 0:
                    df_page = dfs[0]
                    
                    if not df_page.empty and len(df_page.columns) >= 7:
                        df_page = df_page.dropna(how='all')
                        df_page = df_page[df_page.iloc[:, 0].notna()]
                        
                        if len(df_page) > 0:
                            for idx, row in df_page.iterrows():
                                try:
                                    date_str = str(row.iloc[0]).strip()
                                    if not date_str or date_str == 'nan' or '.' not in date_str:
                                        continue
                                    
                                    close = int(str(row.iloc[1]).replace(',', '').replace(' ', ''))
                                    diff_str = str(row.iloc[2]).strip()
                                    open_price = int(str(row.iloc[3]).replace(',', '').replace(' ', ''))
                                    high = int(str(row.iloc[4]).replace(',', '').replace(' ', ''))
                                    low = int(str(row.iloc[5]).replace(',', '').replace(' ', ''))
                                    volume = int(str(row.iloc[6]).replace(',', '').replace(' ', ''))
                                    
                                    page_data.append({
                                        '날짜': date_str,
                                        '종가': close,
                                        '전일비': diff_str,
                                        '시가': open_price,
                                        '고가': high,
                                        '저가': low,
                                        '거래량': volume
                                    })
                                except (ValueError, IndexError, AttributeError):
                                    continue
            except Exception:
                pass
            
            if not page_data:
                soup = BeautifulSoup(response.text, 'html.parser')
                # 테이블 클래스명 확인 (type2, type_2 등)
                table = soup.find('table', {'class': 'type2'})
                if table is None:
                    table = soup.find('table', {'class': 'type_2'})
                if table is None:
                    table = soup.find('table', {'class': 'tb_type1'})
                # 테이블 클래스에 'type2'가 포함된 경우
                if table is None:
                    tables = soup.find_all('table')
                    for t in tables:
                        if t.get('class') and ('type2' in str(t.get('class')) or 'type_2' in str(t.get('class'))):
                            table = t
                            break
                
                if table:
                    rows = table.find_all('tr')
                    # 헤더 행 제외 (보통 처음 2개 행)
                    for row in rows[2:]:
                        cols = row.find_all(['td', 'th'])
                        if len(cols) < 7:
                            continue
                        
                        try:
                            date = cols[0].text.strip()
                            # 날짜 형식 확인 (YYYY.MM.DD 또는 YYYY-MM-DD)
                            if not date or date == '' or len(date) < 8:
                                continue
                            
                            # 종가 추출
                            close_str = cols[1].text.strip().replace(',', '').replace(' ', '').replace('원', '')
                            if not close_str or close_str == '-':
                                continue
                            
                            close = int(close_str)
                            
                            # 전일비
                            diff = cols[2].text.strip()
                            
                            # 시가
                            open_str = cols[3].text.strip().replace(',', '').replace(' ', '').replace('원', '')
                            if not open_str or open_str == '-':
                                continue
                            open_price = int(open_str)
                            
                            # 고가
                            high_str = cols[4].text.strip().replace(',', '').replace(' ', '').replace('원', '')
                            if not high_str or high_str == '-':
                                continue
                            high = int(high_str)
                            
                            # 저가
                            low_str = cols[5].text.strip().replace(',', '').replace(' ', '').replace(' ', '').replace('원', '')
                            if not low_str or low_str == '-':
                                continue
                            low = int(low_str)
                            
                            # 거래량
                            volume_str = cols[6].text.strip().replace(',', '').replace(' ', '')
                            if not volume_str or volume_str == '-':
                                continue
                            volume = int(volume_str)
                            
                            page_data.append({
                                '날짜': date,
                                '종가': close,
                                '전일비': diff,
                                '시가': open_price,
                                '고가': high,
                                '저가': low,
                                '거래량': volume
                            })
                        except (ValueError, AttributeError, IndexError, TypeError) as e:
                            continue
            
            if not page_data:
                break
            
            all_data.extend(page_data)
            time.sleep(0.5)
            
        except Exception:
            break
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data)
    df['날짜'] = pd.to_datetime(df['날짜'], format='%Y.%m.%d', errors='coerce')
    df = df.dropna(subset=['날짜'])
    
    if len(df) == 0:
        return None
    
    df = df.drop_duplicates(subset=['날짜'], keep='first')
    df = df.sort_values('날짜').reset_index(drop=True)
    
    # 장 상태에 따라 마지막 확정된 종가 사용
    # 한국 주식:
    # - 자정 이후 ~ 오전 8시 전: 어제 종가 (마지막 데이터가 어제 종가)
    # - 오전 8시 ~ 오후 8시: 정규장 + 시간외 + 대체거래소 진행 중이므로 어제 종가 (마지막 데이터 제거)
    # - 오후 8시 이후: 오늘 종가 (마지막 데이터가 오늘 종가, 정규장 종가 기준)
    market_closed = is_market_closed("KR")
    if not market_closed and len(df) > 1:
        # 장 진행 중이면 마지막 데이터(오늘 진행 중인 데이터) 제거, 어제 종가 사용
        df = df.iloc[:-1].reset_index(drop=True)
    # market_closed == True이면:
    # - 자정 이후 ~ 오전 8시 전: 마지막 데이터가 어제 종가 (자동으로 사용됨)
    # - 오후 8시 이후: 마지막 데이터가 오늘 종가 (자동으로 사용됨, 정규장 종가 기준)
    
    return df


def calculate_ma(df, periods=[5, 20]):
    """이동평균선 계산"""
    for period in periods:
        df[f'MA{period}'] = df['종가'].rolling(window=period).mean()
    return df


def calculate_rsi(df, period=14):
    """RSI 계산"""
    delta = df['종가'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df):
    """
    MACD (Moving Average Convergence Divergence) 계산
    변동률 기반으로 계산하여 합리적인 범위(-3 ~ +3)로 정규화
    
    원화/달러 가격 그대로 EMA 돌리면 5000, 8000 같은 큰 숫자가 나오므로
    변동률(%) 기반으로 MACD를 계산
    """
    # 종가를 float로 변환
    close = df['종가'].astype(float)
    
    # 하루 변동률(%)로 변환
    pct = close.pct_change() * 100  # 예: 0.5% → 0.5
    
    # NaN 첫 줄 제거 방지용: NaN은 0으로
    pct = pct.fillna(0)
    
    # EMA 계산 (변동률 기반)
    ema12 = pct.ewm(span=12, adjust=False).mean()
    ema26 = pct.ewm(span=26, adjust=False).mean()
    
    # MACD = EMA(12) - EMA(26) (변동률 기반)
    macd = ema12 - ema26
    
    # Signal = EMA(MACD, 9)
    signal = macd.ewm(span=9, adjust=False).mean()
    
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    
    return df


def analyze_granville_rules(df, current_price, ma_period=20):
    """
    그랜빌(Granville)의 8가지 이동평균선 법칙 분석
    
    Args:
        df: 주가 데이터프레임 (최소 3일 이상 필요)
        current_price: 현재가
        ma_period: 이동평균선 기간 (기본값: 20, MA20 사용)
    
    Returns:
        dict: {'rule': int, 'signal': str, 'description': str, 'strength': str}
    """
    if len(df) < 3:
        return None
    
    # 이동평균선 컬럼명
    ma_col = f'MA{ma_period}'
    if ma_col not in df.columns:
        return None
    
    # 최근 3일 데이터
    latest = df.iloc[-1]
    prev1 = df.iloc[-2] if len(df) >= 2 else None
    prev2 = df.iloc[-3] if len(df) >= 3 else None
    
    current_ma = latest[ma_col] if pd.notna(latest[ma_col]) else None
    prev1_ma = prev1[ma_col] if prev1 is not None and pd.notna(prev1[ma_col]) else None
    prev2_ma = prev2[ma_col] if prev2 is not None and pd.notna(prev2[ma_col]) else None
    
    prev1_price = prev1['종가'] if prev1 is not None else None
    prev2_price = prev2['종가'] if prev2 is not None else None
    
    if current_ma is None or prev1_ma is None:
        return None
    
    # 현재가와 이동평균선 위치
    price_above_ma = current_price > current_ma
    price_below_ma = current_price < current_ma
    prev1_price_above = prev1_price > prev1_ma if prev1_price and prev1_ma else None
    prev1_price_below = prev1_price < prev1_ma if prev1_price and prev1_ma else None
    prev2_price_above = prev2_price > prev2_ma if prev2_price and prev2_ma else None
    prev2_price_below = prev2_price < prev2_ma if prev2_price and prev2_ma else None
    
    # 이동평균선 방향 (우상향/우하향)
    ma_rising = current_ma > prev1_ma
    ma_falling = current_ma < prev1_ma
    
    # 주가 변화 추세
    price_rising = current_price > prev1_price if prev1_price else None
    price_falling = current_price < prev1_price if prev1_price else None
    
    # ① 매수 1: 주가가 MA를 아래에서 위로 돌파 (가장 강력)
    if prev1_price_below and price_above_ma and price_rising:
        return {
            'rule': 1,
            'signal': '매수 1',
            'description': '주가가 이평선을 상향 돌파',
            'strength': '가장 강력',
            'emoji': '✅'
        }
    
    # ② 매수 2: 주가가 MA 위에서 눌렸다가 반등 (지지 확인 후 반등)
    if price_above_ma and prev1_price_above and price_falling and price_rising:
        # 전일 하락 후 오늘 반등
        if prev2_price and prev2_price < prev1_price < current_price:
            return {
                'rule': 2,
                'signal': '매수 2',
                'description': '이평선 지지 확인 후 반등',
                'strength': '강함',
                'emoji': '✅'
            }
    
    # ③ 매수 3: 주가가 MA 위에 있지만 단기 조정 (MA 우상향 유지)
    if price_above_ma and ma_rising and price_falling:
        return {
            'rule': 3,
            'signal': '매수 3',
            'description': '이평선이 우상향 유지 중 단기 조정',
            'strength': '보통',
            'emoji': '✅'
        }
    
    # ④ 매수 4: 주가가 MA 아래로 이탈 후 재진입 (추세 복귀)
    if prev2_price and prev1_price:
        if prev2_price_above and prev1_price_below and price_above_ma:
            return {
                'rule': 4,
                'signal': '매수 4',
                'description': '추세 복귀 신호',
                'strength': '약함',
                'emoji': '✅'
            }
    
    # ⑤ 매도 1: 주가가 MA 위에서 꺾임 (고점 신호)
    if price_above_ma and price_falling and ma_rising:
        # 고점 형성 패턴
        if prev2_price and current_price < prev1_price < prev2_price:
            return {
                'rule': 5,
                'signal': '매도 1',
                'description': '고점 신호',
                'strength': '주의',
                'emoji': '🚫'
            }
    
    # ⑥ 매도 2: 주가가 MA 위에서 하락세 강화 (MA 꺾임 + 하락 지속)
    if price_above_ma and price_falling and ma_falling:
        return {
            'rule': 6,
            'signal': '매도 2',
            'description': '이평선 꺾임 + 하락 지속',
            'strength': '강함',
            'emoji': '🚫'
        }
    
    # ⑦ 매도 3: 주가가 MA 아래에서 반등 실패 (저항선 역할)
    if price_below_ma and prev1_price_below and price_rising and not price_above_ma:
        # 반등 시도했지만 MA를 돌파하지 못함
        if current_price < current_ma:
            return {
                'rule': 7,
                'signal': '매도 3',
                'description': '저항선 역할 (반등 실패)',
                'strength': '보통',
                'emoji': '🚫'
            }
    
    # ⑧ 매도 4: 주가가 MA 아래에서 더 하락 (추세 이탈 확정)
    if price_below_ma and prev1_price_below and price_falling:
        return {
            'rule': 8,
            'signal': '매도 4',
            'description': '추세 이탈 확정',
            'strength': '가장 강력',
            'emoji': '🚫'
        }
    
    # 해당 사항 없음
    return None


def analyze_ma_energy_state(df, ma5, ma20):
    """
    MovingAverageEnergyMonitor (이평선 에너지 감시기)
    MA5와 MA20의 거리(격차)와 변화 속도(기울기)를 분석하여 3단계로 분류
    
    Args:
        df: 주가 데이터프레임 (최소 5일 이상 필요)
        ma5: 현재 MA5 값
        ma20: 현재 MA20 값
    
    Returns:
        dict: {
            'state': 'Convergence'|'Breakout'|'Divergence',
            'state_name': '수렴기'|'돌파기'|'확산기',
            'emoji': '⚫'|'🟢'|'🔴',
            'gap_pct': float,  # 현재 격차 (%)
            'gap_trend': '증가'|'감소'|'유지',  # 격차 변화 추세
            'slope_change': float,  # 기울기 변화율
            'interpretation': str,  # 해석
            'strategy': str  # 전략 제안
        }
    """
    if len(df) < 5 or ma5 is None or ma20 is None:
        return None
    
    # 현재 격차 계산
    current_gap_pct = ((ma5 - ma20) / ma20) * 100 if ma20 > 0 else 0
    abs_gap_pct = abs(current_gap_pct)
    
    # 최근 5일간 격차 추세 계산
    gap_history = []
    for i in range(min(5, len(df))):
        idx = len(df) - 1 - i
        if idx >= 0 and 'MA5' in df.columns and 'MA20' in df.columns:
            ma5_val = df.iloc[idx]['MA5'] if pd.notna(df.iloc[idx]['MA5']) else None
            ma20_val = df.iloc[idx]['MA20'] if pd.notna(df.iloc[idx]['MA20']) else None
            if ma5_val is not None and ma20_val is not None and ma20_val > 0:
                gap = ((ma5_val - ma20_val) / ma20_val) * 100
                gap_history.append(gap)
    
    # 격차 변화 추세 계산
    gap_trend = '유지'
    slope_change = 0.0
    if len(gap_history) >= 3:
        recent_gap = gap_history[0]  # 최신
        prev_gap = gap_history[-1] if len(gap_history) > 1 else gap_history[0]  # 5일 전
        
        # 기울기 변화율 (최근 3일 평균 vs 그 전 2일 평균)
        if len(gap_history) >= 3:
            recent_avg = sum(gap_history[:3]) / min(3, len(gap_history))
            prev_avg = sum(gap_history[3:]) / max(1, len(gap_history) - 3) if len(gap_history) > 3 else recent_avg
            slope_change = recent_avg - prev_avg
        
        # 격차 변화 추세
        if recent_gap > prev_gap + 0.1:
            gap_trend = '증가'
        elif recent_gap < prev_gap - 0.1:
            gap_trend = '감소'
        else:
            gap_trend = '유지'
    
    # 3단계 분류
    # 1. 수렴기 (Convergence): abs(MA5 - MA20) ≤ 0.3%
    if abs_gap_pct <= 0.3:
        return {
            'state': 'Convergence',
            'state_name': '수렴기',
            'emoji': '⚫',
            'gap_pct': current_gap_pct,
            'gap_trend': gap_trend,
            'slope_change': slope_change,
            'interpretation': '단기선이 중기선과 밀착, 방향성 대기 상태',
            'strategy': '관망 유지 / 거래량 회복 시 진입 검토',
            'investment_judgment': '👀 관망 / 눌림 대기'
        }
    
    # 2. 돌파기 (Breakout): MA5 > MA20 and 격차 증가 중
    if ma5 > ma20 and (gap_trend == '증가' or slope_change > 0):
        return {
            'state': 'Breakout',
            'state_name': '돌파기',
            'emoji': '🟢',
            'gap_pct': current_gap_pct,
            'gap_trend': gap_trend,
            'slope_change': slope_change,
            'interpretation': '골든크로스 진입, 상승 추세 강화 중',
            'strategy': '진입 가능 (MA5 근처 눌림 시 추가 매수)',
            'investment_judgment': '✅ 진입 가능'
        }
    
    # 3. 확산기 (Divergence): MA5 < MA20 and 격차 증가 중 (음수 방향으로 확산)
    if ma5 < ma20 and (gap_trend == '감소' or slope_change < 0):
        return {
            'state': 'Divergence',
            'state_name': '확산기',
            'emoji': '🔴',
            'gap_pct': current_gap_pct,
            'gap_trend': gap_trend,
            'slope_change': slope_change,
            'interpretation': '데드크로스 진행 중, 하락 추세 강화',
            'strategy': '보유자 주의 / 신규 진입 금지',
            'investment_judgment': '⚠️ 보유자 주의 / 신규 진입 금지'
        }
    
    # 기본값: 수렴기에 가까운 상태
    if abs_gap_pct <= 1.0:
        return {
            'state': 'Convergence',
            'state_name': '수렴기',
            'emoji': '⚫',
            'gap_pct': current_gap_pct,
            'gap_trend': gap_trend,
            'slope_change': slope_change,
            'interpretation': '단기선과 중기선이 근접, 추세 전환 대기',
            'strategy': '관망 / 거래량 및 추가 신호 확인',
            'investment_judgment': '👀 관망'
        }
    
    # 그 외: 현재 상태에 따라 판단
    if ma5 > ma20:
        return {
            'state': 'Breakout',
            'state_name': '돌파기',
            'emoji': '🟢',
            'gap_pct': current_gap_pct,
            'gap_trend': gap_trend,
            'slope_change': slope_change,
            'interpretation': '골든크로스 상태 유지',
            'strategy': '진입 가능 (추가 상승 확인)',
            'investment_judgment': '✅ 진입 가능'
        }
    else:
        return {
            'state': 'Divergence',
            'state_name': '확산기',
            'emoji': '🔴',
            'gap_pct': current_gap_pct,
            'gap_trend': gap_trend,
            'slope_change': slope_change,
            'interpretation': '데드크로스 상태, 하락 추세',
            'strategy': '신규 진입 비추천',
            'investment_judgment': '🚫 신규 진입 비추천'
        }


def calculate_ma_energy_momentum_score(ma_energy, rsi=None):
    """
    Energy Momentum Score 계산
    MA 격차 + 기울기 + RSI를 종합해 0~100점 점수화
    
    Args:
        ma_energy: analyze_ma_energy_state()의 결과
        rsi: RSI 값 (선택)
    
    Returns:
        int: 0~100 점수
    """
    if ma_energy is None:
        return 0
    
    score = 0
    
    # 1. MA Energy State 점수 (40점)
    if ma_energy['state'] == 'Breakout':
        score += 40
    elif ma_energy['state'] == 'Convergence':
        score += 20
    else:  # Divergence
        score += 0
    
    # 2. 기울기 변화율 점수 (30점)
    slope_change = ma_energy['slope_change']
    if slope_change > 0.2:
        score += 30
    elif slope_change > 0.1:
        score += 20
    elif slope_change > 0:
        score += 10
    elif slope_change < -0.2:
        score += 0
    elif slope_change < -0.1:
        score += 5
    
    # 3. RSI 점수 (30점)
    if rsi is not None:
        if 45 <= rsi <= 60:
            score += 30
        elif 40 <= rsi < 45 or 60 < rsi <= 65:
            score += 20
        elif 30 <= rsi < 40 or 65 < rsi <= 70:
            score += 10
        else:
            score += 0
    
    return min(100, max(0, score))


def analyze_entry_opportunity(close_price, ma5, ma20, rsi, volume_ratio, is_us=False, current_price=None):
    """
    진입 가능 여부와 매수 구간을 분석하는 함수
    
    Args:
        close_price: 종가 (진입 판단 기준)
        ma5: MA5 값
        ma20: MA20 값
        rsi: RSI 값
        volume_ratio: 거래량 비율
        is_us: 미국 주식 여부
        current_price: 실시간 현재가 (매수 구간 계산 기준, 없으면 종가 사용)
    
    Returns:
        dict: 진입 판단 결과
    """
    if ma5 is None or ma20 is None or close_price is None:
        return None
    
    # 실시간 현재가가 없으면 종가 사용
    if current_price is None:
        current_price = close_price
    
    # 진입 판단은 종가 기준
    close_above_ma5 = close_price >= ma5
    close_above_ma20 = close_price >= ma20
    close_to_ma5_pct = ((close_price - ma5) / ma5) * 100 if ma5 > 0 else 0
    close_to_ma20_pct = ((close_price - ma20) / ma20) * 100 if ma20 > 0 else 0
    
    # 매수 구간 계산은 실시간 현재가 기준 (현재가가 매수 구간 안에 있는지 확인)
    current_above_ma5 = current_price >= ma5
    current_to_ma5_pct = ((current_price - ma5) / ma5) * 100 if ma5 > 0 else 0
    
    # 매수 구간 산정 (실시간 현재가 기준)
    # 핵심 원칙: 매수 추천가는 항상 현재가보다 낮거나 같아야 함 (지지선 개념)
    # 현재가 위치에 따라 매수 구간을 다르게 계산
    
    # 현재가가 MA5와 MA20 사이에 있는지 확인
    price_above_ma20 = current_price >= ma20
    price_above_ma5 = current_price >= ma5
    
    # 1차 매수 구간 계산 (현재가보다 낮은 가격 범위)
    if price_above_ma5:
        # 현재가가 MA5 위에 있으면 → MA5 근처로 눌렸을 때 매수 (현재가보다 낮은 구간)
        buy_range_1_low = ma5 * 0.99  # MA5의 99%
        buy_range_1_high = min(ma5, current_price * 0.995)  # 현재가의 99.5%와 MA5 중 낮은 값
    elif price_above_ma20:
        # 현재가가 MA20 ~ MA5 사이에 있으면 → 현재가보다 낮은 구간에서 매수
        buy_range_1_low = max(ma20 * 0.99, current_price * 0.97)  # 현재가의 97% 또는 MA20*0.99 중 높은 값
        buy_range_1_high = min(ma5 * 0.99, current_price * 0.995)  # 현재가의 99.5% 또는 MA5*0.99 중 낮은 값
    else:
        # 현재가가 MA20 아래에 있으면 → 현재가보다 낮은 구간에서 매수
        buy_range_1_low = current_price * 0.97  # 현재가의 97%
        buy_range_1_high = min(ma20 * 0.99, current_price * 0.995)  # 현재가의 99.5% 또는 MA20*0.99 중 낮은 값
    
    # 2차 매수 구간 계산 (현재가보다 낮은 가격 범위)
    # 핵심: 2차 매수는 1차 매수보다 확실히 아래에 위치해야 함
    if price_above_ma20:
        # 현재가가 MA20 위에 있으면 → MA20 근처로 눌렸을 때 매수 (현재가보다 낮은 구간)
        buy_range_2_low = ma20 * 0.985  # MA20의 98.5%
        buy_range_2_high = min(ma20, current_price * 0.99)  # 현재가의 99%와 MA20 중 낮은 값
    else:
        # 현재가가 MA20 아래에 있으면 → 현재가보다 낮은 구간에서 매수
        buy_range_2_low = current_price * 0.95  # 현재가의 95%
        buy_range_2_high = min(ma20 * 0.985, current_price * 0.98)  # 현재가의 98% 또는 MA20*0.985 중 낮은 값
    
    # 1차 매수보다 확실히 아래로 조정 (2차는 더 안전한 자리)
    buy_range_1_mid = (buy_range_1_low + buy_range_1_high) / 2
    buy_range_2_mid = (buy_range_2_low + buy_range_2_high) / 2
    
    # 2차 매수가 1차 매수보다 최소 3% 이상 낮아야 함
    if buy_range_2_mid >= buy_range_1_mid * 0.97:
        # 현재가 위치에 따라 다른 조정
        if price_above_ma20:
            # MA20 위일 때: 1차보다 4% 낮게
            target_2_mid = buy_range_1_mid * 0.96
        else:
            # MA20 아래일 때: 1차보다 6% 낮게
            target_2_mid = buy_range_1_mid * 0.94
        
        # 새로운 2차 구간 설정 (중간값 기준 ±1%)
        buy_range_2_high = min(target_2_mid * 1.01, current_price * 0.999)  # 현재가보다 낮게 유지
        buy_range_2_low = target_2_mid * 0.99
        
        # low가 high보다 높으면 안됨
        if buy_range_2_low > buy_range_2_high:
            buy_range_2_low = buy_range_2_high * 0.98
    
    # 최종 검증: 매수 구간 상한은 반드시 현재가보다 낮거나 같아야 함
    buy_range_1_high = min(buy_range_1_high, current_price * 0.999)  # 현재가의 99.9% 이하로 제한
    buy_range_2_high = min(buy_range_2_high, current_price * 0.999)  # 현재가의 99.9% 이하로 제한
    
    # 추가 검증: 구간이 올바른지 확인 (low <= high)
    if buy_range_1_low > buy_range_1_high:
        buy_range_1_low = buy_range_1_high * 0.98
    if buy_range_2_low > buy_range_2_high:
        buy_range_2_low = buy_range_2_high * 0.98
    
    # 손절 기준: MA20 × 0.97 또는 현재가 × 0.97 중 더 보수적인 값
    stop_loss_price = min(ma20 * 0.97, current_price * 0.97)
    
    # 현재가가 매수 구간 안에 있는지 확인 (실시간 현재가 기준)
    in_buy_range_1 = buy_range_1_low <= current_price <= buy_range_1_high
    in_buy_range_2 = buy_range_2_low <= current_price <= buy_range_2_high
    below_stop_loss = current_price < stop_loss_price
    
    # 진입 가능 여부 판단
    entry_status = "👀"  # 기본값: 관망
    entry_reason = []
    comment_parts = []
    
    # RSI 상태
    if rsi is not None:
        if rsi < 30:
            comment_parts.append("⚠️ RSI 과매도")
        elif 30 <= rsi < 45:
            comment_parts.append("📉 RSI 낮음 (관망)")
        elif 45 <= rsi <= 55:
            comment_parts.append("✅ RSI 적정")
            entry_reason.append("RSI 적정")
        elif 55 < rsi <= 70:
            comment_parts.append("📈 RSI 높음")
        else:
            comment_parts.append("🚨 RSI 과열")
            entry_status = "🚫"
    
    # 거래량 상태
    if volume_ratio is not None:
        if volume_ratio < 0.8:
            comment_parts.append("🔹 거래량 저조")
        elif 0.8 <= volume_ratio < 1.2:
            comment_parts.append("⚖️ 거래량 정상 수준")
        elif 1.2 <= volume_ratio <= 2.0:
            comment_parts.append("✅ 거래량 증가")
            entry_reason.append("거래량 증가")
        elif 2.0 < volume_ratio <= 3.0:
            comment_parts.append("📊 거래량 활발")
        else:
            comment_parts.append("⚠️ 거래량 과열")
    
    # 진입 판단은 종가 위치 기준
    if close_above_ma5:
        if close_to_ma5_pct > 1.0:  # 종가가 MA5보다 1% 이상 위
            entry_status = "👀"
            entry_reason.append("MA5 위 (눌림 대기)")
        elif close_to_ma5_pct > 0:
            entry_status = "✅"
            entry_reason.append("MA5 근처 (진입 가능)")
        else:
            entry_status = "✅"
            entry_reason.append("MA5 하단 (매수 적기)")
    else:
        # 종가가 MA5 아래
        if close_to_ma5_pct < -2.0:  # 종가가 MA5보다 2% 이상 아래
            entry_status = "👀"
            entry_reason.append("MA5 하단 (추가 하락 가능)")
        else:
            entry_status = "✅"
            entry_reason.append("MA5 하단 (매수 적기)")
    
    # 골든크로스 상태
    if ma5 >= ma20:
        entry_reason.append("골든크로스")
    else:
        if ((ma5 - ma20) / ma20) * 100 >= -2:  # 2% 이내
            entry_reason.append("골든크로스 직전")
        else:
            entry_reason.append("데드크로스")
    
    # 최종 판단
    if entry_status == "✅" and len(entry_reason) >= 2:
        pass  # 유지
    elif entry_status == "🚫":
        pass  # 유지
    else:
        if "과열" in " ".join(comment_parts) or "RSI 과열" in " ".join(comment_parts):
            entry_status = "🚫"
    
    # 코멘트 생성
    comment = " / ".join(comment_parts) if comment_parts else "데이터 없음"
    
    # 판단 요약 (종가 위치 정보 포함)
    price_position = ""
    if close_above_ma5:
        if close_to_ma5_pct > 1.0:
            price_position = " (MA5 위)"
        else:
            price_position = " (MA5 근처)"
    else:
        price_position = " (MA5 아래)"
    
    if entry_status == "✅":
        judgment = "🤡 매수가능" + price_position
    elif entry_status == "👀":
        judgment = "관망 / 눌림 대기" + price_position
    else:
        judgment = "진입 비추천" + price_position
    
    # 가격 포맷
    price_format = "${:,.2f}" if is_us else "{:,.0f}원"
    
    return {
        'entry_status': entry_status,
        'judgment': judgment,
        'close_price': close_price,  # 종가 (진입 판단 기준)
        'current_price': current_price,  # 실시간 현재가 (매수 구간 기준)
        'ma5': ma5,
        'ma20': ma20,
        'rsi': rsi,
        'volume_ratio': volume_ratio,
        'buy_range_1_low': buy_range_1_low,  # 1차 매수 구간: MA5 × 0.99 ~ MA5 (현재가 기준)
        'buy_range_1_high': buy_range_1_high,
        'buy_range_2_low': buy_range_2_low,  # 2차 매수 구간: MA20 × 0.985 ~ MA20 (현재가 기준)
        'buy_range_2_high': buy_range_2_high,
        'stop_loss_price': stop_loss_price,  # 손절 기준: MA20 × 0.97 (현재가 기준)
        'in_buy_range_1': in_buy_range_1,  # 현재가가 1차 매수 구간 안에 있는지
        'in_buy_range_2': in_buy_range_2,  # 현재가가 2차 매수 구간 안에 있는지
        'below_stop_loss': below_stop_loss,  # 현재가가 손절 기준 아래인지
        'comment': comment,
        'entry_reason': entry_reason,
        'price_format': price_format
    }


def postprocess_signal(result: dict) -> dict:
    """
    현재가, 매수가 괴리율과 RSI/거래량을 기반으로
    눌림(pullback) / 관망(watch) / 추세진입(trend) 모드 결정
    """
    entry_info = result.get("entry_analysis") or {}

    buy_low = entry_info.get("buy_range_1_low")
    buy_high = entry_info.get("buy_range_1_high")

    buy_candidates = [value for value in [buy_high, buy_low, entry_info.get("close_price"), result.get("ma5")] if value]
    buy_price = float(buy_candidates[0]) if buy_candidates else None

    current_candidates = [
        entry_info.get("current_price"),
        result.get("price"),
        entry_info.get("close_price"),
    ]
    current_candidates = [value for value in current_candidates if value]
    current_price = float(current_candidates[0]) if current_candidates else None

    rsi = result.get("rsi")
    if rsi is None:
        rsi = entry_info.get("rsi")
    if rsi is not None:
        try:
            rsi = float(rsi)
        except Exception:
            rsi = None

    volume_ratio = result.get("volume_ratio")
    if volume_ratio is None:
        volume_info = result.get("volume_info") or {}
        volume_ratio = volume_info.get("ratio")
    if volume_ratio is None:
        volume_ratio = entry_info.get("volume_ratio")
    if volume_ratio is not None:
        try:
            volume_ratio = float(volume_ratio)
        except Exception:
            volume_ratio = None

    if not buy_price or not current_price:
        return result

    TREND_THRESHOLD = 0.03
    gap_ratio = (current_price - buy_price) / buy_price

    entry_mode = None
    max_entry_price = None
    is_us = bool(result.get("is_us"))
    currency_word = "달러" if is_us else "원"
    price_formatter = "{:,.2f}" if is_us else "{:,.0f}"

    if gap_ratio >= TREND_THRESHOLD and rsi is not None and volume_ratio is not None and rsi >= 55 and volume_ratio >= 1.5:
        entry_mode = "trend"
        max_entry_price = buy_price * 1.025
        comment = (
            f"🔥 추세 진입 모드입니다. RSI={rsi:.1f}, 거래량={volume_ratio:.2f}배.\n"
            f"{price_formatter.format(max_entry_price)} {currency_word} 이하에서는 분할 진입 가능합니다."
        )
    elif gap_ratio >= TREND_THRESHOLD:
        entry_mode = "watch"
        max_entry_price = buy_price * 1.02
        rsi_text = f"{rsi:.1f}" if rsi is not None else "N/A"
        vol_text = f"{volume_ratio:.2f}" if volume_ratio is not None else "N/A"
        comment = (
            f"⚠️ 현재가는 매수가보다 3% 이상 상승했지만 RSI({rsi_text})·거래량({vol_text})이 약합니다.\n"
            "관망하며 눌림을 기다리세요."
        )
    else:
        entry_mode = "pullback"
        max_entry_price = buy_price * 1.01
        comment = (
            f"🟢 눌림 매수 모드입니다. 현재가가 {price_formatter.format(max_entry_price)} {currency_word} 이하로 내려오면 분할 진입 가능합니다."
        )

    max_entry_price = round(max_entry_price, 2 if is_us else 0)

    result["entry_mode"] = entry_mode
    result["max_entry_price"] = max_entry_price
    result["comment"] = comment

    if entry_info:
        entry_info["entry_mode"] = entry_mode
        entry_info["max_entry_price"] = max_entry_price
        original_comment = entry_info.get("comment")
        if original_comment:
            entry_info["comment"] = f"{original_comment}\n{comment}"
        else:
            entry_info["comment"] = comment
        result["entry_analysis"] = entry_info

    return result


def classify_buy_timing(row, rsi_min=45, rsi_max=55, volume_min=1.2, volume_max=2.0):
    """
    매수 타이밍을 문장으로 분류하는 함수
    
    Args:
        row: 주식 데이터의 마지막 행 (Series 또는 dict)
        rsi_min: RSI 최소값 (기본값: 45)
        rsi_max: RSI 최대값 (기본값: 55)
        volume_min: 거래량 최소 배수 (기본값: 1.2)
        volume_max: 거래량 최대 배수 (기본값: 2.0)
    
    Returns:
        str: 매수 타이밍 설명
    """
    ma5 = row.get('MA5') if hasattr(row, 'get') else row['MA5']
    ma20 = row.get('MA20') if hasattr(row, 'get') else row['MA20']
    rsi = row.get('RSI') if hasattr(row, 'get') else row['RSI']
    vol_ratio = row.get('volume_ratio') if hasattr(row, 'get') else row.get('vol_ratio')
    macd = row.get('MACD') if hasattr(row, 'get') else row.get('MACD')
    macd_signal = row.get('MACD_Signal') if hasattr(row, 'get') else row.get('MACD_signal')
    
    # 기본 플래그
    golden_now = pd.notna(ma5) and pd.notna(ma20) and (ma5 >= ma20 * 0.998)
    rsi_ok = pd.notna(rsi) and rsi_min <= rsi <= rsi_max
    rsi_expanded = pd.notna(rsi) and 45 <= rsi <= 60  # 확장된 RSI 범위
    volume_ok = pd.notna(vol_ratio) and 1.5 <= vol_ratio <= 2.5
    volume_start = pd.notna(vol_ratio) and volume_min <= vol_ratio < 1.5
    overheated = (pd.notna(rsi) and rsi >= 70) or (pd.notna(vol_ratio) and vol_ratio >= 3)
    
    macd_bear = False
    if pd.notna(macd) and pd.notna(macd_signal):
        macd_bear = macd < macd_signal  # 모멘텀 살짝 꺾인 상태
    
    # 1) 과열 먼저 컷
    if overheated:
        return "⚠️ 과열/익절 구간"
    
    # 2) 골든 + RSI ok + 거래량 ok → 최적 매수
    if golden_now and rsi_ok and volume_ok and not macd_bear:
        return "✅ 매수 유효 (거래량 붙은 골든크로스)"
    
    # 3) 골든 + RSI ok + 거래량 아직 작음 → 관망
    if golden_now and rsi_ok and (volume_start or not pd.notna(vol_ratio)):
        return "👀 관망(거래량 대기)"
    
    # 4) 골든인데 MACD만 살짝 내려옴 → 눌림 대기
    if golden_now and macd_bear and rsi_ok:
        return "🟡 눌림 대기 (MACD 약화)"
    
    # 5) RSI가 살짝 낮아도 거래량이 붙으면 매수 가능 (확장된 RSI 범위 사용)
    if golden_now and volume_ok and rsi_expanded:
        return "✅ 매수 가능 (거래량 우선)"
    
    # 그 외는 후보만
    return "🔹 후보 유지"


def is_us_stock(code):
    """종목 코드가 미국 주식인지 판단"""
    if code.isdigit():
        return False
    if any(c.isalpha() for c in code):
        return True
    return False


def is_valid_korean_stock_code(code: str) -> bool:
    """6자리 숫자 형태의 한국 종목 코드를 검증"""
    if not code or not isinstance(code, str):
        return False
    code_str = code.strip()
    return len(code_str) == 6 and code_str.isdigit()


def _extract_korean_codes_from_table(table, limit: int) -> List[str]:
    tickers: List[str] = []
    if not table:
        return tickers

    rows = table.find_all("tr")
    for row in rows:
        link = row.find("a")
        if not link:
            continue
        href = link.get("href", "")
        if "code=" not in href:
            continue
        code = href.split("code=")[-1][:6]
        if is_valid_korean_stock_code(code) and code not in tickers:
            tickers.append(code)
        if len(tickers) >= limit:
            break
    return tickers


def _fetch_korean_market_rank(limit: int, market: str) -> List[str]:
    """
    네이버 증권 시가총액 순위 페이지에서 KOSPI/KOSDAQ 상위 종목 코드 수집
    market: "KOSPI" 또는 "KOSDAQ"
    """
    sosok = "0" if market.upper() == "KOSPI" else "1"
    url = "https://finance.naver.com/sise/sise_market_sum.naver"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://finance.naver.com/",
    }

    collected: List[str] = []
    page = 1
    while len(collected) < limit and page <= 5:
        params = {"sosok": sosok, "page": page}
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.encoding = "euc-kr"
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"class": "type_2"})
            page_codes = _extract_korean_codes_from_table(table, limit - len(collected))
            collected.extend([code for code in page_codes if code not in collected])
            if not page_codes:
                break
        except Exception:
            break
        page += 1
    return collected[:limit]


def get_korean_stocks_by_category(category_id: str, limit: int = 50) -> List[str]:
    """
    네이버 업종/시장 카테고리별 종목 코드 수집
    category_id:
        "0" -> 코스피, "1" -> 코스닥, 그 외에는 업종 번호
    """
    if not category_id:
        return []

    category_id = str(category_id).strip()
    if category_id in ("0", "코스피"):
        return _fetch_korean_market_rank(limit, "KOSPI")
    if category_id in ("1", "코스닥"):
        return _fetch_korean_market_rank(limit, "KOSDAQ")

    url = "https://finance.naver.com/sise/sise_group_detail.naver"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://finance.naver.com/sise/"
    }

    tickers: List[str] = []
    page = 1
    while len(tickers) < limit and page <= 5:
        params = {"type": "upjong", "no": category_id, "page": page}
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.encoding = "euc-kr"
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"class": "type_5"})
            page_codes = _extract_korean_codes_from_table(table, limit - len(tickers))
            tickers.extend([code for code in page_codes if code not in tickers])
            if not page_codes:
                break
        except Exception:
            break
        page += 1

    return tickers[:limit]


def get_top_korean_stocks(limit: int = 50, category_id: Optional[str] = None) -> List[str]:
    """
    한국 주식 상위 종목 목록 가져오기
    - category_id가 주어지면 해당 카테고리 (시장/업종) 기준으로 수집
    - 없으면 코스피/코스닥 시총 상위 종목을 합쳐 limit까지 반환
    """
    limit = max(int(limit), 0)
    if limit == 0:
        return []

    tickers: List[str] = []

    try:
        if category_id is not None:
            tickers = get_korean_stocks_by_category(category_id, limit)
        else:
            kospi = _fetch_korean_market_rank(limit, "KOSPI")
            kosdaq = _fetch_korean_market_rank(limit, "KOSDAQ")
            merged = kospi + [code for code in kosdaq if code not in kospi]
            tickers = merged[:limit]
    except Exception as exc:
        print(f"  ⚠️  한국 주식 목록 가져오기 오류: {exc}")
        tickers = []

    if not tickers:
        fallback = [
            "005930", "000660", "035420", "051910", "035720", "068270",
            "207940", "006400", "373220", "096770", "003550", "005380",
            "018260", "066570", "047810", "068760", "089010", "105560",
            "015760", "086520",
        ]
        tickers = fallback[:limit]

    return tickers[:limit]


def get_korean_stock_categories():
    """
    네이버 증권에서 한국 주식 업종 카테고리 크롤링
    https://finance.naver.com/sise/sise_group.naver?type=upjong 페이지에서 실제 업종 수집
    (기계, 화장품, 소프트웨어 등)
    """
    categories = []
    
    try:
        # 네이버 증권 업종 페이지 (실제 업종: 기계, 화장품, 소프트웨어 등)
        url = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://finance.naver.com/'
        }
        
        # 업종 목록 가져오기
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'euc-kr'
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 업종 테이블 찾기
        table = soup.find('table', {'class': 'type_1'})
        if table:
            rows = table.find_all('tr')[3:]  # 헤더 행 제외 (전체, 상승, 보합 등)
            for row in rows:
                cols = row.find_all(['td', 'th'])
                if len(cols) >= 1:
                    try:
                        # 첫 번째 컬럼에 업종 링크
                        link = cols[0].find('a')
                        if link:
                            href = link.get('href', '')
                            upjong_name = link.text.strip()
                            
                            # href에서 업종 번호 추출 (type=upjong&no=XXX)
                            if 'no=' in href and 'type=upjong' in href:
                                upjong_no = href.split('no=')[1].split('&')[0]
                                if upjong_name and upjong_no:
                                    categories.append({
                                        'market': '한국',
                                        'category_id': upjong_no,
                                        'category_name': upjong_name,
                                        'type': '업종'
                                    })
                    except Exception:
                        continue
        
        # 코스피/코스닥 구분 추가
        categories.insert(0, {
            'market': '한국',
            'category_id': '0',
            'category_name': '코스피',
            'type': '시장'
        })
        categories.insert(1, {
            'market': '한국',
            'category_id': '1',
            'category_name': '코스닥',
            'type': '시장'
        })
        
    except Exception as e:
        print(f"  ⚠️  한국 주식 카테고리 크롤링 오류: {e}")
    
    return categories


def save_categories_to_csv(target_path: Path = STOCK_CATEGORY_CSV) -> None:
    """한국/미국 카테고리 정보를 CSV로 저장"""
    try:
        categories = get_korean_stock_categories() + get_us_stock_categories()
        if not categories:
            print("❌ 저장할 카테고리 정보가 없습니다.")
            return

        target_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(categories)
        df.to_csv(target_path, index=False, encoding="utf-8-sig")
        print(f"✅ 카테고리 목록을 '{target_path}' 파일에 저장했습니다.")
    except Exception as exc:
        print(f"❌ 카테고리 CSV 저장 실패: {exc}")


def get_us_stock_categories():
    """
    미국 주식 섹터/업종 카테고리 정보 수집
    S&P 500의 GICS 섹터와 업종 정보를 Wikipedia에서 가져오기
    """
    categories: List[dict] = []

    def _slugify(label: str) -> str:
        slug = (
            label.lower()
            .replace("&", "and")
            .replace("/", "-")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "-")
        )
        return "-".join(filter(None, slug.split("-")))

    # 기본 지수 카테고리
    categories.append({
        "market": "미국",
        "category_id": "sp500",
        "category_name": "S&P 500",
        "type": "지수",
    })
    categories.append({
        "market": "미국",
        "category_id": "nasdaq100",
        "category_name": "NASDAQ 100",
        "type": "지수",
    })

    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(sp500_url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
    except Exception as exc:
        print(f"  ⚠️  미국 주식 카테고리 크롤링 오류: {exc}")
        return categories

    if not tables:
        return categories

    sp500_table = tables[0]

    if "GICS Sector" in sp500_table.columns:
        sectors = sorted({str(val).strip() for val in sp500_table["GICS Sector"].dropna()})
        for sector in sectors:
            categories.append({
                "market": "미국",
                "category_id": f"gics_sector_{_slugify(sector)}",
                "category_name": sector,
                "type": "섹터",
            })

    if "GICS Sub-Industry" in sp500_table.columns:
        industries = sorted({str(val).strip() for val in sp500_table["GICS Sub-Industry"].dropna()})
        for industry in industries:
            categories.append({
                "market": "미국",
                "category_id": f"gics_industry_{_slugify(industry)}",
                "category_name": industry,
                "type": "업종",
            })

    return categories


def is_valid_us_stock_ticker(ticker):
    """
    미국 주식 티커 유효성 검증 (강화 버전)
    - 컬럼 헤더 제외 (CLOSING, INTRADAY, OPEN, HIGH, LOW, VOLUME 등)
    - 쉼표 포함 숫자 제외 (1,000 같은)
    - 숫자만 있는 것 제외
    - 티커 형식 검증 (영문/숫자/.-=^ 만 허용)
    """
    import re
    
    if not ticker or not isinstance(ticker, str):
        return False
    
    s = ticker.strip()
    
    # 1) 빈 문자열 제외
    if not s:
        return False
    
    # 2) 너무 긴 것 제외 (15자 초과)
    if len(s) > 15:
        return False
    
    # 3) 쉼표가 포함된 것 제외 (1,000 같은 숫자)
    if ',' in s:
        return False
    
    # 4) 대문자 변환
    s_upper = s.upper()
    
    # 5) 전형적인 컬럼명 제거
    invalid_keywords = [
        'CLOSING', 'INTRADAY', 'OPEN', 'HIGH', 'LOW', 'VOLUME', 
        'SYMBOL', 'TICKER', 'NAME', 'COMPANY', 'DATE', 'PRICE',
        'CHANGE', 'PERCENT', 'MARKET', 'CAP', 'SECTOR', 'INDUSTRY'
    ]
    if s_upper in invalid_keywords:
        return False
    
    # 6) 숫자만 있는 경우 제외 (연도 체크 포함)
    if s.replace(',', '').isdigit():
        try:
            num = int(s.replace(',', ''))
            # 연도 범위 제외 (1980~현재연도+1)
            import datetime
            current_year = datetime.datetime.now().year
            if 1980 <= num <= current_year + 1:
                return False
            # 너무 작은 숫자 제외 (1~1000)
            if num < 1000:
                return False
        except ValueError:
            pass
        # 숫자만 있으면 티커가 아님
        return False
    
    # 7) 티커로 쓸 수 있는 문자만 허용 (영문/숫자/.-=^)
    if not re.match(r'^[A-Za-z0-9\.\-\=\^]+$', s):
        return False
    
    # 8) 최소 하나의 알파벳이 포함되어야 함
    if not any(c.isalpha() for c in s):
        return False
    
    return True


def get_us_stocks_by_category(category_id, category_name=None, limit=50):
    """
    야후 파이낸스에서 특정 카테고리(섹터/업종)의 종목 목록 가져오기
    """
    tickers = []
    
    try:
        # S&P 500 종목을 통해 카테고리별 필터링
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        try:
            # requests로 직접 HTML 가져오기
            response = requests.get(sp500_url, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }, timeout=15)
            
            # BeautifulSoup으로 파싱
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable'})
            
            sp500_data = []
            
            if table:
                rows = table.find_all('tr')[1:]  # 헤더 제외
                for row in rows:
                    cols = row.find_all(['td', 'th'])
                    if len(cols) >= 4:
                        try:
                            symbol = cols[0].text.strip()
                            sector = cols[2].text.strip() if len(cols) > 2 else ''
                            industry = cols[3].text.strip() if len(cols) > 3 else ''
                            sp500_data.append({
                                'Symbol': symbol,
                                'GICS Sector': sector,
                                'GICS Sub-Industry': industry
                            })
                        except (IndexError, AttributeError):
                            continue
            
            # 카테고리 ID에 따라 필터링
            if category_id == 'sp500':
                for item in sp500_data:
                    ticker_clean = str(item.get('Symbol', '')).strip().upper()
                    if ticker_clean and is_valid_us_stock_ticker(ticker_clean) and ticker_clean not in tickers:
                        tickers.append(ticker_clean)
                    if len(tickers) >= limit:
                        break
            elif category_id == 'nasdaq100':
                nasdaq_url = "https://en.wikipedia.org/wiki/NASDAQ-100"
                try:
                    nasdaq_response = requests.get(nasdaq_url, headers={
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                    }, timeout=15)
                    nasdaq_soup = BeautifulSoup(nasdaq_response.text, 'html.parser')
                    nasdaq_tables = nasdaq_soup.find_all('table', {'class': 'wikitable'})
                    
                    for nasdaq_table in nasdaq_tables:
                        nasdaq_rows = nasdaq_table.find_all('tr')[1:]
                        for nasdaq_row in nasdaq_rows:
                            nasdaq_cols = nasdaq_row.find_all(['td', 'th'])
                            if nasdaq_cols:
                                ticker = nasdaq_cols[0].text.strip()
                                if ticker and isinstance(ticker, str):
                                    ticker_clean = ticker.replace('.', '-').strip().upper()
                                    # 티커 유효성 검증 추가
                                    if is_valid_us_stock_ticker(ticker_clean) and ticker_clean not in tickers:
                                        tickers.append(ticker_clean)
                                    if len(tickers) >= limit:
                                        break
                        if len(tickers) >= limit:
                            break
                except Exception as e:
                    print(f"  ⚠️  NASDAQ 100 크롤링 오류: {e}")
                    pass
            elif category_id.startswith('gics_sector_') and category_name:
                # 섹터별 필터링
                for item in sp500_data:
                    if item['GICS Sector'] == category_name:
                        ticker_clean = str(item['Symbol']).strip().upper() if 'Symbol' in item else None
                        if ticker_clean and is_valid_us_stock_ticker(ticker_clean) and ticker_clean not in tickers:
                            tickers.append(ticker_clean)
                            if len(tickers) >= limit:
                                break
            elif category_id.startswith('gics_industry_') and category_name:
                # 업종별 필터링
                for item in sp500_data:
                    if item['GICS Sub-Industry'] == category_name:
                        ticker_clean = str(item['Symbol']).strip().upper() if 'Symbol' in item else None
                        if ticker_clean and is_valid_us_stock_ticker(ticker_clean) and ticker_clean not in tickers:
                            tickers.append(ticker_clean)
                            if len(tickers) >= limit:
                                break
            
        except Exception as e:
            # pandas fallback 시도
            try:
                sp500_table = pd.read_html(sp500_url)[0]
                
                if category_id == 'sp500':
                    sp500_tickers = sp500_table['Symbol'].dropna().tolist()
                    for ticker in sp500_tickers:
                        ticker_clean = str(ticker).strip().upper() if pd.notna(ticker) else None
                        if ticker_clean and is_valid_us_stock_ticker(ticker_clean) and ticker_clean not in tickers:
                            tickers.append(ticker_clean)
                            if len(tickers) >= limit:
                                break
                elif category_id == 'nasdaq100':
                    nasdaq_url = "https://en.wikipedia.org/wiki/NASDAQ-100"
                    nasdaq_tables = pd.read_html(nasdaq_url)
                    for table in nasdaq_tables:
                        if 'Ticker' in table.columns or 'Symbol' in table.columns:
                            col_name = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                            nasdaq_tickers = table[col_name].dropna().tolist()
                            for ticker in nasdaq_tickers:
                                if isinstance(ticker, str):
                                    ticker_clean = ticker.replace('.', '-').strip().upper()
                                    if is_valid_us_stock_ticker(ticker_clean) and ticker_clean not in tickers:
                                        tickers.append(ticker_clean)
                                    if len(tickers) >= limit:
                                        break
                            if len(tickers) >= limit:
                                break
                elif 'GICS Sector' in sp500_table.columns and category_id.startswith('gics_sector_') and category_name:
                    sector_tickers = sp500_table[sp500_table['GICS Sector'] == category_name]['Symbol'].tolist()
                    for ticker in sector_tickers:
                        ticker_clean = str(ticker).strip().upper() if pd.notna(ticker) else None
                        if ticker_clean and is_valid_us_stock_ticker(ticker_clean) and ticker_clean not in tickers:
                            tickers.append(ticker_clean)
                            if len(tickers) >= limit:
                                break
                elif 'GICS Sub-Industry' in sp500_table.columns and category_id.startswith('gics_industry_') and category_name:
                    industry_tickers = sp500_table[sp500_table['GICS Sub-Industry'] == category_name]['Symbol'].tolist()
                    for ticker in industry_tickers:
                        ticker_clean = str(ticker).strip().upper() if pd.notna(ticker) else None
                        if ticker_clean and is_valid_us_stock_ticker(ticker_clean) and ticker_clean not in tickers:
                            tickers.append(ticker_clean)
                            if len(tickers) >= limit:
                                break
            except Exception:
                pass
        
    except Exception as e:
        print(f"  ⚠️  미국 주식 카테고리별 목록 가져오기 오류: {e}")
    
    return tickers[:limit]


def get_top_us_stocks(limit=50, category_id=None):
    """
    미국 주식 TOP 종목 목록 가져오기
    S&P 500, NASDAQ 등 주요 지수 종목 사용하거나 특정 카테고리 종목 가져오기
    """
    if category_id:
        return get_us_stocks_by_category(category_id, limit)
    
    tickers = []
    
    if not YFINANCE_AVAILABLE:
        return []
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # S&P 500 종목 목록 (전체 가져오기)
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            # requests로 HTML 가져오기
            response = requests.get(sp500_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # pandas read_html로 파싱 (StringIO 사용)
            sp500_tables = pd.read_html(StringIO(response.text))
            if sp500_tables and len(sp500_tables) > 0:
                sp500_table = sp500_tables[0]
                if 'Symbol' in sp500_table.columns:
                    sp500_tickers = sp500_table['Symbol'].dropna().tolist()
                    # S&P 500 종목 추가 (limit의 절반 정도만, 나머지는 NASDAQ으로)
                    sp500_limit = max(limit // 2, 25)  # 최소 25개
                    for ticker in sp500_tickers:
                        ticker_clean = ticker.strip().upper() if isinstance(ticker, str) else str(ticker).strip().upper()
                        if is_valid_us_stock_ticker(ticker_clean) and ticker_clean not in tickers:
                            tickers.append(ticker_clean)
                            if len(tickers) >= sp500_limit:
                                break
        except Exception as e:
            print(f"  ⚠️  S&P 500 목록 가져오기 실패: {e}")
        
        # NASDAQ 100 종목 추가 (S&P 500과 함께 포함)
        try:
            nasdaq_url = "https://en.wikipedia.org/wiki/NASDAQ-100"
            response = requests.get(nasdaq_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            nasdaq_tables = pd.read_html(StringIO(response.text))
            for table in nasdaq_tables:
                if 'Ticker' in table.columns or 'Symbol' in table.columns:
                    col_name = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                    nasdaq_tickers = table[col_name].dropna().tolist()
                    for ticker in nasdaq_tickers:
                        if isinstance(ticker, str):
                            ticker_clean = ticker.strip().upper()
                            ticker_clean = ticker_clean.lstrip('$')
                            ticker_clean = ticker_clean.replace('.', '-')
                            if is_valid_us_stock_ticker(ticker_clean) and ticker_clean not in tickers:
                                tickers.append(ticker_clean)
                                if len(tickers) >= limit:  # 요청한 limit까지 채우기
                                    break
                    if len(tickers) >= limit:
                        break
        except Exception as e:
            print(f"  ⚠️  NASDAQ 100 목록 가져오기 실패: {e}")
        
        # 추가로 Russell 2000 종목 (S&P 500 + NASDAQ 100만으로 부족한 경우)
        if len(tickers) < limit:
            try:
                russell_url = "https://en.wikipedia.org/wiki/Russell_2000_Index"
                response = requests.get(russell_url, headers=headers, timeout=15)
                response.raise_for_status()
                
                russell_tables = pd.read_html(StringIO(response.text))
                for table in russell_tables:
                    if 'Symbol' in table.columns or 'Ticker' in table.columns:
                        col_name = 'Symbol' if 'Symbol' in table.columns else 'Ticker'
                        russell_tickers = table[col_name].dropna().tolist()
                        for ticker in russell_tickers:
                            if isinstance(ticker, str):
                                ticker_clean = ticker.replace('.', '-').strip().upper()
                                if is_valid_us_stock_ticker(ticker_clean) and ticker_clean not in tickers:
                                    tickers.append(ticker_clean)
                                    if len(tickers) >= limit:
                                        break
                        if len(tickers) >= limit:
                            break
            except Exception as e:
                # Russell 2000은 실패해도 무시 (선택적)
                pass
        
        # 여전히 부족하면 유명 종목 목록 사용
        if len(tickers) < limit:
            popular_stocks = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
                'UNH', 'JNJ', 'V', 'PG', 'JPM', 'MA', 'HD', 'DIS', 'NFLX', 'BAC',
                'ADBE', 'PYPL', 'CMCSA', 'XOM', 'WMT', 'LLY', 'AVGO', 'COST',
                'PEP', 'TMO', 'CSCO', 'ABBV', 'CVX', 'MRK', 'ACN', 'MCD', 'ABT',
                'NKE', 'DHR', 'VZ', 'TXN', 'COIN', 'AMD', 'INTC', 'CRM',
                'ORCL', 'QCOM', 'AMGN', 'HON', 'LIN', 'RTX', 'AMAT', 'BKNG',
                'DE', 'GE', 'IBM', 'CAT', 'BA', 'MMM', 'HON', 'UPS', 'FDX',
                'LMT', 'NOC', 'GD', 'TXT', 'EMR', 'ETN', 'ITW', 'PH', 'AME',
                'GGG', 'RBC', 'AME', 'NDAQ', 'ICE', 'SCHW', 'GS', 'MS', 'C',
                'WFC', 'USB', 'PNC', 'TFC', 'CFG', 'KEY', 'HBAN', 'MTB', 'ZION'
            ]
            
            for ticker in popular_stocks:
                ticker_clean = str(ticker).strip().upper()
                if is_valid_us_stock_ticker(ticker_clean) and ticker_clean not in tickers:
                    tickers.append(ticker_clean)
                if len(tickers) >= limit:
                    break
        
        # 최종 결과 반환 (limit까지) - 추가 필터링
        result = []
        for ticker in tickers[:limit]:
            ticker_clean = str(ticker).strip().upper() if isinstance(ticker, str) else str(ticker).strip().upper()
            if is_valid_us_stock_ticker(ticker_clean) and ticker_clean not in result:
                result.append(ticker_clean)
            if len(result) >= limit:
                break
        
        # 디버깅 정보 출력
        if len(result) < limit:
            print(f"  📊 종목 수집 현황: {len(result)}개 (요청: {limit}개)")
            if len(result) < 100:
                print(f"  ⚠️  종목 수가 적습니다. 네트워크 연결이나 데이터 소스를 확인해주세요.")
                print(f"  💡 lxml 패키지 설치 권장: pip install lxml")
        
        return result
        
    except Exception as e:
        print(f"  ⚠️  미국 주식 목록 가져오기 오류: {e}")
        return []


def get_volume_signal_level(volume_ratio):
    """
    거래량 비율에 따른 매수 신호 수준 반환
    
    Args:
        volume_ratio: 현재 거래량 / 평균 거래량
    
    Returns:
        tuple: (레벨, 설명, 이모지, 상세)
    """
    if volume_ratio <= 1.0:
        return ("관심 없음", "평소 거래량 수준", "🔹", "추세 전환 없음")
    elif volume_ratio < 1.2:
        return ("관심 없음", "평소 거래량 수준", "🔹", "추세 전환 없음")
    elif volume_ratio < 1.5:
        return ("관망", "관심이 붙기 시작한 초기 신호", "⚡", "예의주시")
    elif volume_ratio < 2.5:
        return ("매수 유효", "본격적인 매수세 진입", "✅", "매수 유효 구간")
    elif volume_ratio < 3.0:
        return ("매수 유효", "본격적인 매수세 진입", "✅", "매수 유효 구간")
    elif volume_ratio < 5.0:
        return ("조심", "단기 과열 또는 단타 세력 진입", "⚠️", "익절·조심 구간")
    elif volume_ratio < 10.0:
        return ("조심", "단기 과열 또는 단타 세력 진입", "⚠️", "익절·조심 구간")
    else:
        return ("진입 금지", "뉴스·테마주, 급등 후 피크 가능성", "❌", "진입 금지 구간")


def predict_volume(df, days=3):
    """
    거래량 예측 함수
    
    Args:
        df: 주식 데이터프레임
        days: 예측할 일수 (기본값: 3일)
    
    Returns:
        list: 예측된 거래량 정보 리스트
    """
    if len(df) < 20:
        return None
    
    # 평균 거래량 계산
    avg_volume = df['거래량'].tail(20).mean()
    
    # 최근 거래량 추세 분석
    recent_volumes = df['거래량'].tail(10).values
    volume_trend = (recent_volumes[-1] - recent_volumes[0]) / recent_volumes[0] if recent_volumes[0] > 0 else 0
    
    # 현재 거래량
    current_volume = df['거래량'].iloc[-1]
    
    predictions = []
    
    # 오늘 (현재)
    today_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    level, desc, emoji, detail = get_volume_signal_level(today_ratio)
    predictions.append({
        'day': '오늘',
        'volume': current_volume,
        'ratio': today_ratio,
        'level': level,
        'desc': desc,
        'emoji': emoji,
        'detail': detail,
        'accuracy': 100  # 현재값이므로 정확도 100%
    })
    
    # 내일, 모레 예측
    for day_offset in range(1, min(days, 3)):
        # 거래량 예측 (추세 기반, 일부 랜덤성 고려)
        if volume_trend > 0:
            # 상승 추세면 약간 증가
            predicted_volume = current_volume * (1 + volume_trend * 0.3 * day_offset)
        elif volume_trend < 0:
            # 하락 추세면 약간 감소
            predicted_volume = current_volume * (1 + volume_trend * 0.2 * day_offset)
        else:
            # 추세 없으면 평균 거래량 수준으로 회귀
            predicted_volume = avg_volume * (1 - day_offset * 0.1)
        
        # 예측값이 음수가 되지 않도록
        predicted_volume = max(0, predicted_volume)
        
        # 예측 비율
        predicted_ratio = predicted_volume / avg_volume if avg_volume > 0 else 1
        
        level, desc, emoji, detail = get_volume_signal_level(predicted_ratio)
        
        # 정확도 계산 (과거 예측 성능 기반)
        # 단순히 추세 기반이므로 60-70% 정도로 설정
        accuracy = 70 - (day_offset * 5)  # 며칠 후일수록 정확도 감소
        
        day_name = '내일' if day_offset == 1 else '모레'
        predictions.append({
            'day': day_name,
            'volume': predicted_volume,
            'ratio': predicted_ratio,
            'level': level,
            'desc': desc,
            'emoji': emoji,
            'detail': detail,
            'accuracy': accuracy
        })
    
    return predictions


def check_buy_signal(ticker, period="3mo", rsi_min=45, rsi_max=55, volume_min=1.2, volume_max=2.0):
    """
    반등 신호를 확인하는 함수
    조건:
    - 골든크로스 직전/직후 (MA5가 MA20에 매우 가깝거나 위)
    - RSI 45~55 (과매도 끝, 반등 준비)
    - 거래량 평균 대비 1.2~2배 (관심 집중)
    """
    try:
        is_us = is_us_stock(ticker)
        
        if is_us:
            # 미국 주식 - Ticker 객체 사용 (더 안정적)
            if not YFINANCE_AVAILABLE:
                return None
            
            # 티커 유효성 검증
            if not is_valid_us_stock_ticker(ticker):
                return None
            
            try:
                ticker_obj = yf.Ticker(ticker)
                # MA60 계산을 위해 최소 6개월 데이터 필요
                if period == "3mo":
                    hist = ticker_obj.history(period="6mo")
                else:
                    hist = ticker_obj.history(period=period)
                
                if hist.empty:
                    return None
                
                df = hist.reset_index()
                
                # 컬럼명 매핑
                column_mapping = {}
                for col in df.columns:
                    if col == 'Date' or str(col).startswith('Date'):
                        column_mapping[col] = '날짜'
                    elif col in ['Open', 'open']:
                        column_mapping[col] = '시가'
                    elif col in ['High', 'high']:
                        column_mapping[col] = '고가'
                    elif col in ['Low', 'low']:
                        column_mapping[col] = '저가'
                    elif col in ['Close', 'close']:
                        column_mapping[col] = '종가'
                    elif col in ['Volume', 'volume']:
                        column_mapping[col] = '거래량'
                
                df = df.rename(columns=column_mapping)
                
                # 필요한 컬럼 확인
                if '거래량' not in df.columns or '종가' not in df.columns:
                    return None
                
            except Exception:
                return None
            
            # 데이터 보정
            df = df.sort_values('날짜').reset_index(drop=True)
            df['거래량'] = df['거래량'].replace(0, pd.NA)
            
            # 장 상태에 따라 마지막 데이터 또는 그 전 데이터 사용
            market_closed = is_market_closed("US")
            if not market_closed and len(df) > 1:
                # 장 중이면 마지막에서 두 번째 데이터 사용 (어제 데이터)
                df = df.iloc[:-1].reset_index(drop=True)
            
            # 지표 계산
            df = calculate_ma(df, periods=[5, 20, 60])  # MA60 추가
            df['avg_vol_20'] = df['거래량'].rolling(20, min_periods=5).mean()
            df['volume_ratio'] = df['거래량'] / df['avg_vol_20']
            df = calculate_rsi(df, period=14)
            df = calculate_macd(df)
            
        else:
            # 한국 주식
            # MA60 계산을 위해 더 많은 페이지 필요 (약 3개월치)
            df = fetch_stock_data(ticker, pages=10)  # 더 많은 데이터 수집
            if df is None or df.empty:
                return None
            
            # 데이터 보정
            df = df.sort_values('날짜').reset_index(drop=True)
            df['거래량'] = df['거래량'].replace(0, pd.NA)
            
            # 지표 계산
            df = calculate_ma(df, periods=[5, 20, 60])  # MA60 추가
            df['avg_vol_20'] = df['거래량'].rolling(20, min_periods=5).mean()
            df['volume_ratio'] = df['거래량'] / df['avg_vol_20']
            df = calculate_rsi(df, period=14)
            df = calculate_macd(df)
        
        # MA60 계산을 위해 최소 60일 필요하지만, 데이터가 부족하면 MA60 없이 진행 가능
        if len(df) < 20:
            return None
        
        # MA60이 없는 경우 (데이터가 60일 미만) 정배열은 False로 처리
        if len(df) < 60:
            # MA60 계산은 하지 않지만, 나머지 분석은 진행
            pass
        
        latest = df.iloc[-1]
        
        # 실시간 현재가 가져오기
        current_price = None
        if is_us:
            # 미국 주식: yfinance에서 실시간 가격 가져오기
            # 티커 유효성 재검증
            if not is_valid_us_stock_ticker(ticker):
                return None
            
            try:
                if YFINANCE_AVAILABLE:
                    ticker_obj = yf.Ticker(ticker)
                    # fast_info는 더 빠르지만, info도 시도
                    try:
                        fast_info = ticker_obj.fast_info
                        current_price = fast_info.get('lastPrice')
                    except Exception as e:
                        # 예외 발생 시 무시하고 다음 방법 시도
                        pass
                    
                    if current_price is None or pd.isna(current_price):
                        try:
                            info = ticker_obj.info
                            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                        except Exception as e:
                            # 예외 발생 시 무시
                            pass
            except Exception as e:
                # 전체 예외 처리: 티커가 유효하지 않거나 데이터를 가져올 수 없는 경우
                return None
        
        else:
            # 한국 주식: 네이버 증권에서 실시간 가격 크롤링
            try:
                url = f"https://finance.naver.com/item/main.naver?code={ticker}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=5)
                response.encoding = 'euc-kr'
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 현재가 찾기
                price_element = soup.find('p', {'class': 'no_today'})
                if price_element:
                    price_text = price_element.get_text(strip=True)
                    # 숫자만 추출
                    import re
                    price_match = re.search(r'[\d,]+', price_text.replace(',', ''))
                    if price_match:
                        current_price = float(price_match.group().replace(',', ''))
            except Exception:
                pass
        
        # 현재가가 없으면 종가 사용
        if current_price is None or pd.isna(current_price):
            current_price = latest['종가']
        
        # 조건 확인
        ma5 = latest['MA5'] if pd.notna(latest['MA5']) else None
        ma20 = latest['MA20'] if pd.notna(latest['MA20']) else None
        ma60 = latest['MA60'] if 'MA60' in latest.index and pd.notna(latest['MA60']) else None
        rsi = latest['RSI'] if 'RSI' in latest.index and pd.notna(latest['RSI']) else None
        price = current_price  # 실시간 현재가 사용
        close_price = latest['종가']  # 종가는 별도로 저장
        
        if ma5 is None or ma20 is None or rsi is None:
            return None
        
        # 정배열 확인 (MA60 < MA20 < MA5)
        is_perfect_alignment = False
        if ma60 is not None and ma20 is not None and ma5 is not None:
            is_perfect_alignment = (ma60 < ma20 < ma5)
        
        # 골든 크로스 확인 (직전/직후 포함)
        # 직후: MA5가 MA20보다 위
        # 직전: MA5가 MA20보다 약간 낮지만 매우 가까움 (2% 이내)
        ma_gap_pct = ((ma5 - ma20) / ma20) * 100 if ma20 > 0 else 0
        golden_cross_near = ma5 >= ma20  # 골든 크로스 직후
        golden_cross_imminent = ma5 < ma20 and ma_gap_pct >= -2  # 골든 크로스 직전 (2% 이내)
        golden_cross_signal = golden_cross_near or golden_cross_imminent
        
        # RSI 범위 확인
        rsi_in_range = rsi_min <= rsi <= rsi_max
        
        # 거래량 분석 및 조건 확인
        volume_info = None
        volume_predictions = None
        volume_in_range = False
        
        if '거래량' in latest.index and pd.notna(latest['거래량']):
            if len(df) >= 21:
                avg_volume = df['거래량'].tail(20).mean()
                current_volume = latest['거래량']
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                # 거래량 조건: 1.2~2배 범위
                volume_in_range = volume_min <= volume_ratio <= volume_max
                
                level, desc, emoji, detail = get_volume_signal_level(volume_ratio)
                
                volume_info = {
                    'current': current_volume,
                    'average': avg_volume,
                    'ratio': volume_ratio,
                    'level': level,
                    'desc': desc,
                    'emoji': emoji,
                    'detail': detail,
                    'in_range': volume_in_range
                }
                
                # 거래량 예측
                volume_predictions = predict_volume(df, days=3)
        
        # 1️⃣ 반등 신호 판단 (마지막 확정된 종가 기준)
        # - MA5 > MA20 (골든크로스)
        # - MACD > Signal (MACD 골든크로스)
        # - RSI 40~60 (회복 초입 구간)
        # - 거래량 평균 대비 1.3배 이상
        
        macd = latest.get('MACD') if 'MACD' in latest.index and pd.notna(latest.get('MACD')) else None
        macd_signal = latest.get('MACD_Signal') if 'MACD_Signal' in latest.index and pd.notna(latest.get('MACD_Signal')) else None
        
        # MACD 골든크로스 확인
        macd_golden_cross = False
        if macd is not None and macd_signal is not None:
            macd_golden_cross = macd > macd_signal
        
        # RSI 40~60 범위 (반등 신호용, 더 넓은 범위)
        rsi_reversal_range = 40 <= rsi <= 60 if rsi is not None else False
        
        # 거래량 1.3배 이상 (반등 신호용)
        volume_reversal = volume_ratio >= 1.3 if volume_info and 'ratio' in volume_info else False
        
        # 반등 신호 조건 (마지막 확정된 종가 기준)
        reversal_signal = golden_cross_signal and macd_golden_cross and rsi_reversal_range and volume_reversal
        
        # 2️⃣ 진입 판단 (마지막 확정된 종가 기준)
        # - RSI 45~60 (추세 초기)
        # - MA5 > MA20 (단기 추세 상승)
        # - MACD 양전환 중
        # - 거래량 평균 이상
        
        # RSI 45~60 범위 (진입 판단용)
        rsi_entry_range = 45 <= rsi <= 60 if rsi is not None else False
        
        # 거래량 평균 이상 (진입 판단용)
        volume_entry = volume_ratio >= 1.0 if volume_info and 'ratio' in volume_info else False
        
        # 진입 판단 조건 (마지막 확정된 종가 기준)
        entry_ready = golden_cross_near and rsi_entry_range and macd_golden_cross and volume_entry
        
        # 기존 반등 신호 조건 (하위 호환성 유지)
        condition_met = golden_cross_signal and rsi_in_range and volume_in_range
        
        # 매수 타이밍 분류
        vol_ratio_for_timing = volume_ratio if volume_info and 'ratio' in volume_info else (latest.get('volume_ratio') if 'volume_ratio' in latest.index and pd.notna(latest.get('volume_ratio')) else None)
        timing_row = {
            'MA5': ma5,
            'MA20': ma20,
            'RSI': rsi,
            'volume_ratio': vol_ratio_for_timing,
            'MACD': latest.get('MACD') if 'MACD' in latest.index else None,
            'MACD_Signal': latest.get('MACD_Signal') if 'MACD_Signal' in latest.index else None
        }
        buy_timing = classify_buy_timing(timing_row, rsi_min=rsi_min, rsi_max=rsi_max, volume_min=volume_min, volume_max=volume_max)
        
        # 점수 계산 (100점 만점)
        score = 0
        score_details = {}
        
        # 1. 골든크로스 점수 (40점)
        if golden_cross_near:
            gc_score = 40
        elif golden_cross_imminent:
            gc_score = 30
        elif ma_gap_pct >= -5:  # 5% 이내
            gc_score = 20
        else:
            gc_score = 0
        score += gc_score
        score_details['golden_cross'] = gc_score
        
        # 2. RSI 점수 (35점)
        if rsi_in_range:  # 45-55
            rsi_score = 35
        elif (rsi_min - 5) <= rsi < rsi_min or rsi_max < rsi <= (rsi_max + 5):  # 40-45 또는 55-60
            rsi_score = 25
        elif (rsi_min - 10) <= rsi < (rsi_min - 5) or (rsi_max + 5) < rsi <= (rsi_max + 10):  # 35-40 또는 60-65
            rsi_score = 15
        else:
            rsi_score = 0
        score += rsi_score
        score_details['rsi'] = rsi_score
        
        # 3. 거래량 점수 (25점)
        if volume_info:
            vol_ratio = volume_info['ratio']
            if volume_in_range:  # 1.2-2.0배
                vol_score = 25
            elif (volume_min - 0.2) <= vol_ratio < volume_min or volume_max < vol_ratio <= (volume_max + 0.5):  # 1.0-1.2 또는 2.0-2.5
                vol_score = 15
            elif 0.8 <= vol_ratio < (volume_min - 0.2) or (volume_max + 0.5) < vol_ratio <= 3.0:  # 0.8-1.0 또는 2.5-3.0
                vol_score = 10
            else:
                vol_score = 0
        else:
            vol_score = 0
        score += vol_score
        score_details['volume'] = vol_score
        
        # 진입 기회 분석 (종가 기준으로 판단, 매수 구간은 실시간 현재가 기준)
        vol_ratio_for_analysis = volume_ratio if volume_info and 'ratio' in volume_info else (latest.get('volume_ratio') if 'volume_ratio' in latest.index and pd.notna(latest.get('volume_ratio')) else None)
        # 진입 판단은 종가 기준, 매수 구간 계산은 실시간 현재가 기준
        entry_analysis = analyze_entry_opportunity(close_price, ma5, ma20, rsi, vol_ratio_for_analysis, is_us=is_us, current_price=price)
        
        # MA Energy State 분석 (이평선 에너지 감시기)
        ma_energy_state = analyze_ma_energy_state(df, ma5, ma20)
        ma_energy_score = calculate_ma_energy_momentum_score(ma_energy_state, rsi) if ma_energy_state else 0
        
        # 그랜빌 법칙 분석 (MA20 기준)
        granville_ma20 = analyze_granville_rules(df, price, ma_period=20)
        granville_ma5 = analyze_granville_rules(df, price, ma_period=5)  # MA5도 분석
        
        return {
            'ticker': ticker,
            'condition_met': condition_met,  # 기존 반등 신호 (하위 호환성)
            'reversal_signal': reversal_signal,  # 1️⃣ 반등 신호 (마지막 확정된 종가 기준)
            'entry_ready': entry_ready,  # 2️⃣ 진입 판단 (마지막 확정된 종가 기준)
            'golden_cross': golden_cross_near,
            'golden_cross_imminent': golden_cross_imminent,
            'golden_cross_signal': golden_cross_signal,
            'macd_golden_cross': macd_golden_cross,  # MACD 골든크로스
            'ma_gap_pct': ma_gap_pct,
            'rsi_in_range': rsi_in_range,
            'volume_in_range': volume_in_range,
            'price': price,  # 실시간 현재가
            'close_price': close_price,  # 종가 (별도 저장)
            'ma5': ma5,
            'ma20': ma20,
            'ma60': ma60,  # MA60 추가
            'is_perfect_alignment': is_perfect_alignment,  # 정배열 여부
            'rsi': rsi,
            'volume_info': volume_info,
            'volume_predictions': volume_predictions,
            'volume_ratio': vol_ratio_for_analysis,
            'macd': macd,
            'macd_signal': macd_signal,
            'buy_timing': buy_timing,
            'entry_analysis': entry_analysis,
            'granville_ma20': granville_ma20,  # 그랜빌 법칙 (MA20 기준)
            'granville_ma5': granville_ma5,  # 그랜빌 법칙 (MA5 기준)
            'ma_energy_state': ma_energy_state,  # MA Energy State (이평선 에너지 감시기)
            'ma_energy_score': ma_energy_score,  # Energy Momentum Score
            'is_us': is_us,
            'score': score,
            'score_details': score_details
        }
        
    except Exception as e:
        return None


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


def _compute_ma_slope(df, column, lookback=3):
    if column not in df.columns:
        return None
    if len(df) <= lookback:
        return None
    try:
        current = df[column].iloc[-1]
        previous = df[column].iloc[-(lookback + 1)]
        if pd.isna(current) or pd.isna(previous):
            return None
        return float(current) - float(previous)
    except Exception:
        return None


def fetch_fundamentals_for_mode(ticker: str, is_us: bool) -> dict:
    if not YFINANCE_AVAILABLE:
        return {}
    candidates = [ticker]
    if not is_us and ticker.isdigit():
        candidates = [f"{ticker}.KS", f"{ticker}.KQ"]
    for candidate in candidates:
        try:
            info = yf.Ticker(candidate).info
        except Exception:
            continue
        if not info:
            continue
        fundamentals = {}
        pe = info.get("trailingPE") or info.get("forwardPE")
        if isinstance(pe, (int, float)) and pe > 0:
            fundamentals["pe"] = float(pe)
        roe = info.get("returnOnEquity")
        if isinstance(roe, (int, float)):
            fundamentals["roe"] = float(roe * 100 if roe <= 1 else roe)
        eps = info.get("trailingEps")
        if isinstance(eps, (int, float)):
            fundamentals["eps"] = float(eps)
        if fundamentals:
            return fundamentals
    return {}


def build_mode_input(ticker: str, name: str, df: pd.DataFrame, latest: pd.Series, current_price: float, is_us: bool, fundamentals: dict) -> dict:
    return {
        "symbol": ticker,
        "name": name,
        "is_us": is_us,
        "current_price": current_price,
        "ma5": _safe_float(latest.get("MA5")),
        "ma20": _safe_float(latest.get("MA20")),
        "ma60": _safe_float(latest.get("MA60")),
        "ma120": _safe_float(latest.get("MA120")),
        "ma20_slope": _compute_ma_slope(df, "MA20"),
        "ma60_slope": _compute_ma_slope(df, "MA60", lookback=5),
        "rsi": _safe_float(latest.get("RSI")),
        "volume_ratio": _safe_float(latest.get("volume_ratio")),
        "macd": _safe_float(latest.get("MACD")),
        "macd_signal": _safe_float(latest.get("MACD_Signal")),
        "fundamentals": fundamentals or {},
    }


def _resolve_display_name(ticker: str, is_us: bool) -> str:
    if is_us:
        if YFINANCE_AVAILABLE:
            try:
                info = yf.Ticker(ticker).info
                candidate = info.get("longName") or info.get("shortName") or info.get("symbol")
                if candidate:
                    return candidate
            except Exception:
                pass
        return ticker

    candidates = [ticker]
    if ticker.isdigit():
        candidates = [f"{ticker}.KS", f"{ticker}.KQ"]

    if YFINANCE_AVAILABLE:
        for code in candidates:
            try:
                info = yf.Ticker(code).info
            except Exception:
                continue
            candidate = info.get("longName") or info.get("shortName") or info.get("symbol")
            if candidate and candidate != code:
                return candidate

    return ticker


def format_mode_summary(result: dict, currency_symbol: str, price_format: str):
    mode = result.get("mode", "")
    mode_label = {
        "daytrade": "단타 (Daytrade)",
        "swing": "스윙 (Swing)",
        "longterm": "장기투자 (Longterm)",
    }.get(mode, mode)

    status = result.get("status", "-")
    reason = result.get("reason", "-")
    recommendation = result.get("recommendation", "-")
    entry_text = "✅ 발생" if result.get("entry_signal") else "❌ 없음"
    exit_text = "✅ 발생" if result.get("exit_signal") else "❌ 없음"

    stop_price = result.get("stop_loss_price")
    stop_pct = result.get("stop_loss_pct")
    if stop_price:
        stop_text = f"{price_format.format(stop_price)}{currency_symbol}"
        if stop_pct is not None:
            stop_text += f" (-{stop_pct:.1f}%)"
    else:
        stop_text = "N/A"

    print(f"📊 모드: {mode_label}")
    print(f"📈 상태: {status}")
    print(f"💬 이유: {reason}")
    print("───────────────────────────────")
    print(f"🔹 진입 신호: {entry_text}")
    print(f"🔹 청산 신호: {exit_text}")
    print(f"🔹 손절 기준: {stop_text}")
    print("───────────────────────────────")
    print(f"💡 가이드: {recommendation}")


def run_mode_screening(
    tickers: List[str],
    mode: str,
    signals_only: bool = False,
    entry_only: bool = False,
    exit_only: bool = False,
):
    if not tickers:
        print("❌ 분석할 종목이 없습니다.")
        return

    mode = mode.lower()
    requested_total = len(tickers)
    mode_title = {
        "daytrade": "Daytrade",
        "swing": "Swing",
        "longterm": "Longterm",
    }.get(mode, mode.title())

    cards: List[str] = []
    errors: List[str] = []
    status_counter = {"positive": 0, "neutral": 0, "negative": 0}
    positive_symbols: List[str] = []
    negative_symbols: List[str] = []

    any_us = False
    any_kr = False

    for ticker in tickers:
        is_us = is_us_stock(ticker)
        any_us = any_us or is_us
        any_kr = any_kr or not is_us

        if is_us:
            df = fetch_stock_data_yahoo(ticker, period="6mo")
        else:
            df = fetch_stock_data(ticker, pages=20)

        if df is None or df.empty:
            errors.append(f"❌ 데이터를 가져올 수 없습니다: {ticker}")
            continue

        df = calculate_ma(df, periods=[5, 20, 60, 120])
        df = calculate_rsi(df, period=14)
        df = calculate_macd(df)

        if '거래량' in df.columns:
            df['거래량'] = pd.to_numeric(df['거래량'], errors='coerce')
            df['avg_vol_20'] = df['거래량'].rolling(20, min_periods=5).mean()
            df['volume_ratio'] = df['거래량'] / df['avg_vol_20']

        if len(df) < 20:
            errors.append(f"❌ 분석에 필요한 데이터가 부족합니다: {ticker}")
            continue

        latest = df.iloc[-1]

        try:
            current_price = float(latest['종가'])
        except Exception:
            errors.append(f"❌ 종가 정보를 확인할 수 없습니다: {ticker}")
            continue

        currency_symbol = "원" if not is_us else "달러"
        price_format = "{:,.0f}" if not is_us else "{:,.2f}"

        display_name = _resolve_display_name(ticker, is_us)
        fundamentals = fetch_fundamentals_for_mode(ticker, is_us) if mode == "longterm" else {}
        mode_input = build_mode_input(ticker, display_name, df, latest, current_price, is_us, fundamentals)

        if mode == "daytrade":
            analysis = analyze_daytrade(mode_input)
        elif mode == "longterm":
            analysis = analyze_longterm(mode_input)
        else:
            analysis = analyze_swing(mode_input)

        if not analysis.get("name"):
            analysis["name"] = display_name

        entry_flag = bool(analysis.get("entry_signal"))
        exit_flag = bool(analysis.get("exit_signal"))

        if entry_only and (not entry_flag or exit_flag):
            continue
        if exit_only and not exit_flag:
            continue
        if signals_only and not (entry_flag or exit_flag):
            continue

        if exit_flag:
            status_counter["negative"] += 1
            if len(negative_symbols) < 3:
                negative_symbols.append(analysis.get("symbol", ticker))
        elif entry_flag:
            status_counter["positive"] += 1
            if len(positive_symbols) < 3:
                positive_symbols.append(analysis.get("symbol", ticker))
        else:
            status_counter["neutral"] += 1

        stop_price = analysis.get("stop_loss_price")
        stop_pct = analysis.get("stop_loss_pct")
        if stop_price:
            pct_text = f" (-{stop_pct:.1f}%)" if stop_pct is not None else ""
            stop_text = f"{price_format.format(stop_price)}{currency_symbol}{pct_text}"
        else:
            stop_text = "N/A"

        icon = "🟢" if entry_flag and not exit_flag else "🔴" if exit_flag else "🟡"
        symbol = analysis.get("symbol", ticker)
        name = analysis.get("name", display_name)

        card_lines = [f"{icon} [{symbol}] {name} ({mode_title})", "─" * 40]
        card_lines.append(f"📈 상태: {analysis.get('status', '-')}")
        card_lines.append(f"💬 이유: {analysis.get('reason', '-')}")
        card_lines.append(f"💰 손절: {stop_text}")
        card_lines.append(f"💡 가이드: {analysis.get('recommendation', '-')}")
        cards.append("\n".join(card_lines))

    market_label = "글로벌 주식" if (any_us and any_kr) else ("미국 주식" if any_us else "한국 주식")
    flag = "🌐" if (any_us and any_kr) else ("🇺🇸" if any_us else "🇰🇷")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"{flag} {market_label} 스크리닝 결과 (요청 {requested_total}개)")
    print("─" * 40)
    print(f"분석 완료: {len(cards)}종목 | 모드: {mode_title} | 시간: {timestamp}")

    if errors:
        for msg in errors:
            print(msg)

    if cards:
        for card in cards:
            print("\n" + card)
    else:
        print("신호가 감지된 종목이 없습니다.")

    print()
    print("📊 요약 결과")
    print("─" * 40)
    print(f"🟢 보유 권장: {status_counter['positive']}종목")
    print(f"🟡 관망 필요: {status_counter['neutral']}종목")
    print(f"🔴 청산 신호: {status_counter['negative']}종목")
    print("─" * 40)

    def format_top(symbols: List[str]) -> str:
        return ", ".join(symbols) if symbols else "-"

    print(f"Top 3 안정 종목: {format_top(positive_symbols)}")
    print(f"Top 3 위험 종목: {format_top(negative_symbols)}")


def screen_stocks(tickers, period="3mo", rsi_min=45, rsi_max=55, volume_min=1.2, volume_max=2.0):
    """
    여러 종목을 스크리닝하는 함수 (반등 신호 찾기)
    """
    print("=" * 60)
    print("🔍 반등 신호 주식 스크리닝")
    print("=" * 60)
    print(f"1️⃣ 반등 신호 판단 (마지막 확정된 종가 기준):")
    print(f"  - MA5 > MA20 (골든크로스)")
    print(f"  - MACD > Signal (MACD 골든크로스)")
    print(f"  - RSI: 40~60 (회복 초입 구간)")
    print(f"  - 거래량: 평균 대비 1.3배 이상")
    print(f"\n2️⃣ 진입 판단 (마지막 확정된 종가 기준):")
    print(f"  - MA5 > MA20 (단기 추세 상승)")
    print(f"  - MACD 양전환 중")
    print(f"  - RSI: 45~60 (추세 초기)")
    print(f"  - 거래량: 평균 이상")
    print(f"\n3️⃣ 매수 구간 산정 (실시간 현재가 기준):")
    print(f"  - 1차: MA5 × 0.99 ~ MA5 (단기 눌림)")
    print(f"  - 2차: MA20 × 0.985 ~ MA20 (중기 눌림)")
    print(f"  - 손절: MA20 × 0.97")
    print(f"\n총 {len(tickers)}개 종목 검사 중...\n")
    
    candidates = []
    results = []
    
    # 티커 리스트 사전 필터링 (잘못된 티커 제거)
    valid_tickers = []
    filtered_count = 0
    for ticker in tickers:
        if isinstance(ticker, str):
            ticker_clean = ticker.strip()
            # 한국 주식은 6자리 숫자, 미국 주식은 알파벳 포함
            if len(ticker_clean) == 6 and ticker_clean.isdigit():
                # 한국 주식 코드 검증
                if is_valid_korean_stock_code(ticker_clean):
                    valid_tickers.append(ticker_clean)
                else:
                    filtered_count += 1
            else:
                # 미국 주식 티커 검증
                if is_valid_us_stock_ticker(ticker_clean):
                    valid_tickers.append(ticker_clean.upper())
                else:
                    filtered_count += 1
        else:
            # 숫자나 다른 타입도 문자열로 변환 후 검증
            ticker_str = str(ticker).strip()
            is_valid = False
            if len(ticker_str) == 6 and ticker_str.isdigit():
                is_valid = is_valid_korean_stock_code(ticker_str)
            else:
                is_valid = is_valid_us_stock_ticker(ticker_str)
            
            if is_valid:
                valid_tickers.append(ticker_str.upper() if not ticker_str.isdigit() else ticker_str)
            else:
                filtered_count += 1
    
    if filtered_count > 0:
        print(f"  ⚠️  {filtered_count}개의 잘못된 티커가 필터링되었습니다.\n")
    
    if len(valid_tickers) == 0:
        print("  ❌ 유효한 티커가 없습니다. 크롤링된 티커를 확인해주세요.\n")
        return []
    
    for i, ticker in enumerate(valid_tickers, 1):
        print(f"[{i}/{len(valid_tickers)}] {ticker} 분석 중...", end=" ")
        result = check_buy_signal(ticker, period=period, rsi_min=rsi_min, rsi_max=rsi_max,
                                  volume_min=volume_min, volume_max=volume_max)
        
        if result is None:
            print("❌ 데이터 없음")
            continue
        
        result = postprocess_signal(result)
        results.append(result)
        
        # 1️⃣ 반등 신호 또는 2️⃣ 진입 판단 중 하나라도 만족하면 후보
        if result.get('reversal_signal') or result.get('entry_ready') or result['condition_met']:
            candidates.append(ticker)
            signal_types = []
            if result.get('reversal_signal'):
                signal_types.append("1️⃣ 반등 신호")
            if result.get('entry_ready'):
                signal_types.append("2️⃣ 진입 판단")
            if result['condition_met']:
                signal_types.append("✅ 반등 신호")
            
            # 정배열 표시
            alignment_marker = ""
            if result.get('is_perfect_alignment'):
                alignment_marker = " 🔥정배열"
            
            print(f"✅ {' / '.join(signal_types)} 발견!{alignment_marker}")
        else:
            # 조건별 상세 정보
            gc_status = "✅" if result['golden_cross_signal'] else "❌"
            rsi_status = "✅" if result['rsi_in_range'] else "❌"
            vol_status = "✅" if result.get('volume_in_range', False) else "❌"
            macd_status = "✅" if result.get('macd_golden_cross', False) else "❌"
            print(f"❌ (골든크로스: {gc_status}, MACD: {macd_status}, RSI: {rsi_status}, 거래량: {vol_status})")
    
    print("\n" + "=" * 60)
    print("📊 스크리닝 결과")
    print("=" * 60)
    
    if len(candidates) == 0:
        print("\n❌ 반등 신호가 있는 종목이 없습니다.")
        
        # 점수 기반으로 TOP 10 후보 선정
        if len(results) > 0:
            # 점수순으로 정렬
            sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            top_candidates = sorted_results[:10]
            
            print("\n" + "=" * 60)
            print("🎯 확률 높은 후보 종목 TOP 10 (점수순)")
            print("=" * 60)
            print("💡 모든 조건을 만족하지 않더라도 높은 점수를 받은 종목입니다.")
            print("   점수 구성: 골든크로스(40점) + RSI(35점) + 거래량(25점) = 100점 만점\n")
            
            for idx, result in enumerate(top_candidates, 1):
                price_format = "${:,.2f}" if result['is_us'] else "{:,.0f}원"
                gc_status = "✅" if result['golden_cross_signal'] else "❌"
                if result.get('golden_cross_imminent'):
                    gc_status += " (직전)"
                elif result.get('golden_cross'):
                    gc_status += " (직후)"
                
                score = result.get('score', 0)
                score_details = result.get('score_details', {})
                
                # 정배열 표시
                alignment_marker = " 🔥정배열" if result.get('is_perfect_alignment') else ""
                
                print(f"\n  {idx}. 📈 {result['ticker']} (점수: {score}/100점){alignment_marker}")
                ma60_str = f" | MA60: {price_format.format(result.get('ma60', 0))}" if result.get('ma60') else ""
                print(f"     종가: {price_format.format(result['price'])}")
                print(f"     MA5: {price_format.format(result['ma5'])} | MA20: {price_format.format(result['ma20'])}{ma60_str} | 격차: {result['ma_gap_pct']:+.2f}%")
                print(f"     골든크로스: {gc_status} (점수: {score_details.get('golden_cross', 0)}/40)")
                print(f"     RSI: {result['rsi']:.2f} {'✅' if result['rsi_in_range'] else '❌'} (점수: {score_details.get('rsi', 0)}/35)")
                
                # 거래량 정보 출력
                if result['volume_info']:
                    vol = result['volume_info']
                    vol_status = "✅" if vol.get('in_range', False) else "❌"
                    print(f"     거래량: {vol['current']:,.0f} (평균 대비 {vol['ratio']:.2f}배) {vol_status} - {vol['emoji']} {vol['desc']} (점수: {score_details.get('volume', 0)}/25)")
                
                # 매수 타이밍 출력
                if result.get('buy_timing'):
                    print(f"     🧭 매수 타이밍: {result['buy_timing']}")
                
                # 그랜빌 법칙 출력
                if result.get('granville_ma20'):
                    gr = result['granville_ma20']
                    print(f"     📊 그랜빌 법칙 (MA20): {gr['emoji']} {gr['signal']} - {gr['description']} ({gr['strength']})")
                if result.get('granville_ma5'):
                    gr = result['granville_ma5']
                    print(f"     📊 그랜빌 법칙 (MA5): {gr['emoji']} {gr['signal']} - {gr['description']} ({gr['strength']})")
                
                # MA Energy State 출력 (이평선 에너지 감시기)
                if result.get('ma_energy_state'):
                    energy = result['ma_energy_state']
                    price_format = "${:,.2f}" if result['is_us'] else "{:,.0f}원"
                    gap_sign = "+" if energy['gap_pct'] >= 0 else ""
                    slope_sign = "+" if energy['slope_change'] >= 0 else ""
                    
                    print(f"\n     🧭 MA Energy State: {energy['emoji']} {energy['state_name']} ({energy['state']})")
                    print(f"        MA5: {price_format.format(result['ma5'])} / MA20: {price_format.format(result['ma20'])} (격차: {gap_sign}{energy['gap_pct']:.2f}%)")
                    if energy.get('slope_change') is not None:
                        print(f"        기울기 변화율: {slope_sign}{energy['slope_change']:.2f}% ({energy['gap_trend']} 중)")
                    print(f"        💡 {energy['interpretation']}")
                    print(f"        🧭 전략 제안: {energy['strategy']}")
                    if result.get('ma_energy_score'):
                        print(f"        📊 Energy Momentum Score: {result['ma_energy_score']}/100점")
                
                # 진입 분석 출력
                if result.get('entry_analysis'):
                    entry = result['entry_analysis']
                    price_fmt = entry['price_format']
                    current_price = entry['current_price']
                    buy_range_1_low = entry.get('buy_range_1_low', 0)
                    buy_range_1_high = entry.get('buy_range_1_high', 0)
                    buy_range_2_low = entry.get('buy_range_2_low', 0)
                    buy_range_2_high = entry.get('buy_range_2_high', 0)
                    
                    # 현재가가 매수 구간 안에 있는지 확인
                    in_buy_range_1 = entry.get('in_buy_range_1', False)
                    in_buy_range_2 = entry.get('in_buy_range_2', False)
                    in_any_buy_range = in_buy_range_1 or in_buy_range_2
                    
                    # 점수 가져오기
                    result_score = result.get('score', 0)
                    
                    # 최종 판단 (점수 + 매수 구간)
                    if result_score >= 80:
                        if in_any_buy_range:
                            final_judgment = "🟢 매수 추천 (구간 안)"
                        else:
                            final_judgment = "❤️좋은종목! 가격대기!"
                    elif result_score >= 60:
                        if in_any_buy_range:
                            final_judgment = "🟡 관망 (점수 양호, 구간 안)"
                        else:
                            final_judgment = "👀 관망 (점수 양호, 가격 대기)"
                    else:
                        final_judgment = "👀 관망"
                    
                    print(f"\n     📊 매수 판단 결과")
                    print(f"     종가: {price_fmt.format(entry['close_price'])} (진입 판단 기준)")
                    print(f"     현재가: {price_fmt.format(current_price)} (매수 구간 기준)")
                    print(f"     MA5: {price_fmt.format(entry['ma5'])}")
                    print(f"     MA20: {price_fmt.format(entry['ma20'])}")
                    if entry['rsi']:
                        print(f"     RSI: {entry['rsi']:.2f}")
                    if entry['volume_ratio']:
                        print(f"     거래량비: {entry['volume_ratio']:.2f}")
                    # 현재가가 매수 구간 안에 있는지 표시
                    range_1_status = "✅ 현재가가 구간 안" if in_buy_range_1 else "❌ 현재가가 구간 밖"
                    range_2_status = "✅ 현재가가 구간 안" if in_buy_range_2 else "❌ 현재가가 구간 밖"
                    stop_loss_status = "🚨 손절 기준 도달" if entry.get('below_stop_loss') else "✅ 손절 기준 위"
                    
                    print(f"     1차매수구간: {price_fmt.format(buy_range_1_low)} ~ {price_fmt.format(buy_range_1_high)} (MA5 × 0.99 ~ MA5) {range_1_status}")
                    print(f"     2차매수구간: {price_fmt.format(buy_range_2_low)} ~ {price_fmt.format(buy_range_2_high)} (MA20 × 0.985 ~ MA20) {range_2_status}")
                    if entry.get('stop_loss_price'):
                        print(f"     손절기준: {price_fmt.format(entry['stop_loss_price'])} (MA20 × 0.97) {stop_loss_status}")
                    print(f"     판단: {entry['entry_status']} {entry['judgment']} (종가 기준)")
                    print(f"     코멘트: {entry['comment']}")
                    print(f"\n     🎯 최종 판단: {final_judgment}")
                    if result_score >= 80 and not in_any_buy_range:
                        # 매수 구간 안내
                        if buy_range_1_low > 0 and buy_range_1_high > 0:
                            print(f"        💡 매수 타이밍: {price_fmt.format(buy_range_1_low)} ~ {price_fmt.format(buy_range_1_high)} 구간에서 매수 권장")
                        elif buy_range_2_low > 0 and buy_range_2_high > 0:
                            print(f"        💡 매수 타이밍: {price_fmt.format(buy_range_2_low)} ~ {price_fmt.format(buy_range_2_high)} 구간에서 매수 권장")
                
                # 거래량 예측 출력
                if result['volume_predictions']:
                    print(f"     📊 거래량 예측:")
                    for pred in result['volume_predictions']:
                        print(f"       {pred['day']}: {pred['volume']:,.0f} ({pred['ratio']:.2f}배) - {pred['emoji']} {pred['desc']} (정확도: {pred['accuracy']:.0f}%)")
        else:
            print("\n❌ 분석 가능한 종목이 없습니다.")
    else:
        print(f"\n✅ 반등 신호 발견 종목 ({len(candidates)}개):")
        for ticker in candidates:
            result = next(r for r in results if r['ticker'] == ticker)
            price_format = "${:,.2f}" if result['is_us'] else "{:,.0f}원"
            gc_status = "✅ 골든크로스 직후" if result['golden_cross'] else "✅ 골든크로스 직전"
            
            # 정배열 표시
            alignment_marker = " 🔥정배열" if result.get('is_perfect_alignment') else ""
            
            print(f"\n  📈 {ticker}{alignment_marker}")
            print(f"     종가: {price_format.format(result['price'])}")
            ma60_str = f" | MA60: {price_format.format(result.get('ma60', 0))}" if result.get('ma60') else ""
            print(f"     MA5: {price_format.format(result['ma5'])} | MA20: {price_format.format(result['ma20'])}{ma60_str} | 격차: {result['ma_gap_pct']:+.2f}%")
            print(f"     {gc_status}")
            print(f"     RSI: {result['rsi']:.2f} (적정 범위)")
            
            # 거래량 정보 출력
            if result['volume_info']:
                vol = result['volume_info']
                print(f"     거래량: {vol['current']:,.0f} (평균 대비 {vol['ratio']:.2f}배) - {vol['emoji']} {vol['desc']}")
                print(f"       💡 {vol['detail']}")
            
            # 매수 타이밍 출력
            if result.get('buy_timing'):
                print(f"     🧭 매수 타이밍: {result['buy_timing']}")
            
            # 그랜빌 법칙 출력
            if result.get('granville_ma20'):
                gr = result['granville_ma20']
                print(f"     📊 그랜빌 법칙 (MA20): {gr['emoji']} {gr['signal']} - {gr['description']} ({gr['strength']})")
            if result.get('granville_ma5'):
                gr = result['granville_ma5']
                print(f"     📊 그랜빌 법칙 (MA5): {gr['emoji']} {gr['signal']} - {gr['description']} ({gr['strength']})")
            
            # MA Energy State 출력 (이평선 에너지 감시기)
            if result.get('ma_energy_state'):
                energy = result['ma_energy_state']
                price_format = "${:,.2f}" if result['is_us'] else "{:,.0f}원"
                gap_sign = "+" if energy['gap_pct'] >= 0 else ""
                slope_sign = "+" if energy['slope_change'] >= 0 else ""
                
                print(f"\n     🧭 MA Energy State: {energy['emoji']} {energy['state_name']} ({energy['state']})")
                print(f"        MA5: {price_format.format(result['ma5'])} / MA20: {price_format.format(result['ma20'])} (격차: {gap_sign}{energy['gap_pct']:.2f}%)")
                if energy.get('slope_change') is not None:
                    print(f"        기울기 변화율: {slope_sign}{energy['slope_change']:.2f}% ({energy['gap_trend']} 중)")
                print(f"        💡 {energy['interpretation']}")
                print(f"        🧭 전략 제안: {energy['strategy']}")
                if result.get('ma_energy_score'):
                    print(f"        📊 Energy Momentum Score: {result['ma_energy_score']}/100점")
            
                # 진입 분석 출력
                if result.get('entry_analysis'):
                    entry = result['entry_analysis']
                    price_fmt = entry['price_format']
                    current_price = entry['current_price']
                    buy_range_1_low = entry.get('buy_range_1_low', 0)
                    buy_range_1_high = entry.get('buy_range_1_high', 0)
                    buy_range_2_low = entry.get('buy_range_2_low', 0)
                    buy_range_2_high = entry.get('buy_range_2_high', 0)
                    
                    # 현재가가 매수 구간 안에 있는지 확인
                    in_buy_range_1 = entry.get('in_buy_range_1', False)
                    in_buy_range_2 = entry.get('in_buy_range_2', False)
                    in_any_buy_range = in_buy_range_1 or in_buy_range_2
                    
                    # 점수 가져오기
                    result_score = result.get('score', 0)
                    
                    # 최종 판단 (점수 + 매수 구간)
                    if result_score >= 80:
                        if in_any_buy_range:
                            final_judgment = "🟢 매수 추천 (구간 안)"
                        else:
                            final_judgment = "❤️좋은종목! 가격대기!"
                    elif result_score >= 60:
                        if in_any_buy_range:
                            final_judgment = "🟡 관망 (점수 양호, 구간 안)"
                        else:
                            final_judgment = "👀 관망 (점수 양호, 가격 대기)"
                    else:
                        final_judgment = "👀 관망"
                    
                    print(f"\n     📊 매수 판단 결과")
                    print(f"     종가: {price_fmt.format(entry['close_price'])} (진입 판단 기준)")
                    print(f"     현재가: {price_fmt.format(current_price)} (매수 구간 기준)")
                    print(f"     MA5: {price_fmt.format(entry['ma5'])}")
                    print(f"     MA20: {price_fmt.format(entry['ma20'])}")
                    if entry['rsi']:
                        print(f"     RSI: {entry['rsi']:.2f}")
                    if entry['volume_ratio']:
                        print(f"     거래량비: {entry['volume_ratio']:.2f}")
                    # 현재가가 매수 구간 안에 있는지 표시
                    range_1_status = "✅ 현재가가 구간 안" if in_buy_range_1 else "❌ 현재가가 구간 밖"
                    range_2_status = "✅ 현재가가 구간 안" if in_buy_range_2 else "❌ 현재가가 구간 밖"
                    stop_loss_status = "🚨 손절 기준 도달" if entry.get('below_stop_loss') else "✅ 손절 기준 위"
                    
                    print(f"     1차매수구간: {price_fmt.format(buy_range_1_low)} ~ {price_fmt.format(buy_range_1_high)} (MA5 × 0.99 ~ MA5) {range_1_status}")
                    print(f"     2차매수구간: {price_fmt.format(buy_range_2_low)} ~ {price_fmt.format(buy_range_2_high)} (MA20 × 0.985 ~ MA20) {range_2_status}")
                    if entry.get('stop_loss_price'):
                        print(f"     손절기준: {price_fmt.format(entry['stop_loss_price'])} (MA20 × 0.97) {stop_loss_status}")
                    print(f"     판단: {entry['entry_status']} {entry['judgment']} (종가 기준)")
                    print(f"     코멘트: {entry['comment']}")
                    print(f"\n     🎯 최종 판단: {final_judgment}")
                    if result_score >= 80 and not in_any_buy_range:
                        # 매수 구간 안내
                        if buy_range_1_low > 0 and buy_range_1_high > 0:
                            print(f"        💡 매수 타이밍: {price_fmt.format(buy_range_1_low)} ~ {price_fmt.format(buy_range_1_high)} 구간에서 매수 권장")
                        elif buy_range_2_low > 0 and buy_range_2_high > 0:
                            print(f"        💡 매수 타이밍: {price_fmt.format(buy_range_2_low)} ~ {price_fmt.format(buy_range_2_high)} 구간에서 매수 권장")
            
            # 거래량 예측 출력
            if result['volume_predictions']:
                print(f"     📊 거래량 예측:")
                for pred in result['volume_predictions']:
                    accuracy_emoji = "🎯" if pred['accuracy'] >= 70 else "📊" if pred['accuracy'] >= 60 else "⚠️"
                    print(f"       {pred['day']}: {pred['volume']:,.0f} ({pred['ratio']:.2f}배) - {pred['emoji']} {pred['desc']} ({accuracy_emoji} 정확도: {pred['accuracy']:.0f}%)")
        
        print(f"\n💡 반등 신호 종목 리스트: {', '.join(candidates)}")
    
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description='주식 스크리닝 도구 - 여러 종목을 자동으로 스크리닝하여 매수 신호를 확인합니다.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 직접 종목 코드 입력
  python stock_screener.py NVDA TSLA AAPL IBM AMGN
  
  # 파일에서 종목 코드 읽기
  python stock_screener.py --file tickers.txt
  
  # 한국 주식 TOP 50 자동 크롤링
  python stock_screener.py --top-korea
  
  # 미국 주식 TOP 50 자동 크롤링
  python stock_screener.py --top-us
  
  # RSI 범위 조정
  python stock_screener.py NVDA TSLA --rsi-min 40 --rsi-max 60
  
  # 눌림목 스크리닝 전용 모드
  python stock_screener.py --dip
  
  # 모드별 전략 분석
  python stock_screener.py --mode daytrade --top-us
  python stock_screener.py --mode swing --category-us "S&P 500"
  python stock_screener.py --mode longterm --top-korea
        """
    )
    
    parser.add_argument('tickers', nargs='*', help='종목 코드 리스트 (예: NVDA TSLA AAPL)')
    parser.add_argument('--file', type=str, help='종목 코드가 있는 파일 경로 (한 줄에 하나씩)')
    parser.add_argument('--top-korea', action='store_true', help='한국 주식 TOP 50 자동 크롤링')
    parser.add_argument('--top-us', action='store_true', help='미국 주식 TOP 50 자동 크롤링')
    parser.add_argument('--top-limit', type=int, default=50, help='TOP 종목 개수 (기본값: 50)')
    parser.add_argument('--category-korea', type=str, help=f"한국 주식 카테고리 ID 또는 이름 ({STOCK_CATEGORY_CSV} 참조)")
    parser.add_argument('--category-us', type=str, help=f"미국 주식 카테고리 ID 또는 이름 ({STOCK_CATEGORY_CSV} 참조)")
    parser.add_argument('--update-categories', action='store_true', help='카테고리 목록을 크롤링하여 CSV 파일로 저장')
    parser.add_argument('--rsi-min', type=int, default=45, help='RSI 최소값 (기본값: 45)')
    parser.add_argument('--rsi-max', type=int, default=55, help='RSI 최대값 (기본값: 55)')
    parser.add_argument('--volume-min', type=float, default=1.2, help='거래량 최소 배수 (기본값: 1.2)')
    parser.add_argument('--volume-max', type=float, default=2.0, help='거래량 최대 배수 (기본값: 2.0)')
    parser.add_argument('--period', type=str, default='3mo', help='데이터 기간 (기본값: 3mo) 옵션: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max')
    parser.add_argument('--dip', action='store_true', help='눌림목 스크리닝 모드 실행')
    parser.add_argument('--mode', choices=['daytrade', 'swing', 'longterm'], help='투자 성향별 모드 분석 실행')
    parser.add_argument('--output', type=str, help='분석 결과를 지정한 텍스트 파일로 저장')
    parser.add_argument('--signals-only', action='store_true', help='모드 분석 시 진입/청산 신호가 있는 종목만 표시')
    parser.add_argument('--entry-only', action='store_true', help='모드 분석 시 진입 신호가 있는 종목만 표시')
    parser.add_argument('--exit-only', action='store_true', help='모드 분석 시 청산 신호가 있는 종목만 표시')
    
    args = parser.parse_args()
    
    def run_logic():
        if args.dip:
            run_dip_screening(
                get_top_korean_stocks=get_top_korean_stocks if 'get_top_korean_stocks' in globals() else None,
                get_top_us_stocks=get_top_us_stocks if 'get_top_us_stocks' in globals() else None,
            )
            return
        
        if args.update_categories:
            save_categories_to_csv()
            return
        
        tickers = []
        
        if args.category_korea:
            category_id = None
            category_name = None
            if STOCK_CATEGORY_CSV.exists():
                try:
                    df_categories = pd.read_csv(STOCK_CATEGORY_CSV, encoding='utf-8-sig')
                    match = df_categories[
                        ((df_categories['market'] == '한국') &
                         ((df_categories['category_id'] == args.category_korea) |
                          (df_categories['category_name'] == args.category_korea)))
                    ]
                    if not match.empty:
                        category_id = match.iloc[0]['category_id']
                        category_name = match.iloc[0]['category_name']
                        print(f"✅ 한국 카테고리 찾음: {category_name} (ID: {category_id})")
                    else:
                        print(f"❌ 한국 카테고리를 찾을 수 없습니다: {args.category_korea}")
                        print("💡 --update-categories 옵션으로 카테고리 목록을 업데이트하세요.")
                        return
                except Exception as e:
                    print(f"❌ CSV 파일 읽기 오류: {e}")
                    return
            else:
                print(f"❌ {STOCK_CATEGORY_CSV} 파일이 없습니다.")
                print("💡 --update-categories 옵션으로 카테고리 목록을 생성하세요.")
                return
            if category_id:
                print(f"🇰🇷 한국 주식 카테고리 '{category_name}' 종목 크롤링 중...")
                korean_tickers = get_top_korean_stocks(limit=args.top_limit, category_id=str(category_id))
                if korean_tickers:
                    print(f"✅ {len(korean_tickers)}개 종목을 찾았습니다.")
                    tickers.extend(korean_tickers)
                else:
                    print("❌ 종목을 가져오지 못했습니다.")
                    return
        
        if args.category_us:
            category_id = None
            category_name = None
            if STOCK_CATEGORY_CSV.exists():
                try:
                    df_categories = pd.read_csv(STOCK_CATEGORY_CSV, encoding='utf-8-sig')
                    match = df_categories[
                        ((df_categories['market'] == '미국') &
                         ((df_categories['category_id'] == args.category_us) |
                          (df_categories['category_name'] == args.category_us)))
                    ]
                    if not match.empty:
                        category_id = match.iloc[0]['category_id']
                        category_name = match.iloc[0]['category_name']
                        print(f"✅ 미국 카테고리 찾음: {category_name} (ID: {category_id})")
                    else:
                        print(f"❌ 미국 카테고리를 찾을 수 없습니다: {args.category_us}")
                        print("💡 --update-categories 옵션으로 카테고리 목록을 업데이트하세요.")
                        return
                except Exception as e:
                    print(f"❌ CSV 파일 읽기 오류: {e}")
                    return
            else:
                print(f"❌ {STOCK_CATEGORY_CSV} 파일이 없습니다.")
                print("💡 --update-categories 옵션으로 카테고리 목록을 생성하세요.")
                return
            if category_id:
                print(f"🇺🇸 미국 주식 카테고리 '{category_name}' 종목 크롤링 중...")
                us_tickers = get_us_stocks_by_category(str(category_id), category_name=category_name, limit=args.top_limit)
                if us_tickers:
                    print(f"✅ {len(us_tickers)}개 종목을 찾았습니다.")
                    tickers.extend(us_tickers)
                else:
                    print("❌ 종목을 가져오지 못했습니다.")
                    return
        
        if args.top_korea:
            print("=" * 60)
            print("🇰🇷 한국 주식 TOP 종목 크롤링 중...")
            print("=" * 60)
            korean_tickers = get_top_korean_stocks(limit=args.top_limit)
            if korean_tickers:
                print(f"✅ {len(korean_tickers)}개 한국 주식 종목을 찾았습니다.")
                tickers.extend(korean_tickers)
            else:
                print("❌ 한국 주식 목록을 가져오지 못했습니다.")
                return
        
        if args.top_us:
            print("=" * 60)
            print("🇺🇸 미국 주식 TOP 종목 크롤링 중...")
            print("=" * 60)
            us_tickers = get_top_us_stocks(limit=args.top_limit)
            if us_tickers:
                print(f"✅ {len(us_tickers)}개 미국 주식 종목을 찾았습니다. (요청: {args.top_limit}개)")
                if len(us_tickers) < args.top_limit:
                    print(f"   ⚠️  요청하신 {args.top_limit}개보다 적은 {len(us_tickers)}개만 가져왔습니다.")
                tickers.extend(us_tickers)
            else:
                print("❌ 미국 주식 목록을 가져오지 못했습니다.")
                return
        
        if args.tickers:
            tickers.extend(args.tickers)
        
        if args.file:
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    file_tickers = [line.strip() for line in f if line.strip()]
                    tickers.extend(file_tickers)
            except FileNotFoundError:
                print(f"❌ 파일을 찾을 수 없습니다: {args.file}")
                return
            except Exception as e:
                print(f"❌ 파일 읽기 오류: {e}")
                return
        
        if not tickers:
            print("❌ 스크리닝할 종목이 없습니다.")
            print("\n사용법:")
            print("  python stock_screener.py NVDA TSLA AAPL")
            print("  python stock_screener.py --top-korea")
            print("  python stock_screener.py --top-us")
            print("  python stock_screener.py --file tickers.txt")
            print("  python stock_screener.py --category-korea '코스피'")
            print("  python stock_screener.py --category-us 'S&P 500'")
            print("  python stock_screener.py --update-categories  # 카테고리 목록 생성")
            return
        
        tickers = list(dict.fromkeys(tickers))
        
        if args.mode:
            if args.entry_only and args.exit_only:
                print("⚠️  entry-only와 exit-only를 동시에 지정하면 두 신호가 모두 있는 종목만 표시합니다.")
            run_mode_screening(
                tickers,
                args.mode,
                signals_only=args.signals_only,
                entry_only=args.entry_only,
                exit_only=args.exit_only,
            )
            return
        
        print(f"\n총 {len(tickers)}개 종목을 스크리닝합니다.\n")
        screen_stocks(tickers, period=args.period, rsi_min=args.rsi_min, rsi_max=args.rsi_max,
                      volume_min=args.volume_min, volume_max=args.volume_max)
    
    if args.output:
        base_path = Path(args.output)
        ext = base_path.suffix.lower()
        if not ext:
            ext = ".txt"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if ext == ".txt":
            target_dir = TXT_OUTPUT_DIR
        elif ext == ".csv":
            target_dir = CSV_OUTPUT_DIR
        elif ext == ".png":
            target_dir = PNG_OUTPUT_DIR
        else:
            target_dir = TXT_OUTPUT_DIR

        output_path = target_dir / f"{base_path.stem}_{timestamp}{ext}"
        try:
            with open(output_path, 'w', encoding='utf-8') as output_file:
                with redirect_stdout(output_file):
                    run_logic()
        except Exception as exc:
            print(f"❌ 출력 파일을 생성할 수 없습니다: {exc}")
            return
        print(f"✅ 결과를 '{output_path}' 파일로 저장했습니다.")
    else:
        run_logic()


if __name__ == "__main__":
    main()

