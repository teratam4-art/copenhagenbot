import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from io import StringIO
import time
import re
import numpy as np
import os

# yfinance for US stocks
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance 패키지가 설치되지 않았습니다. 미국 주식 조회를 위해 설치해주세요:")
    print("   pip install yfinance")


def get_currency_code(is_us: bool) -> str:
    return "USD" if is_us else "KRW"


def format_price(value, currency="KRW") -> str:
    if value is None or (isinstance(value, (float, int)) and pd.isna(value)):
        return "N/A"
    if currency == "USD":
        return f"${value:,.2f}"
    return f"{value:,.0f}원"


def format_percentage(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:+.2f}%"


def fetch_stock_data_yahoo(symbol, period="1y"):
    """
    야후 파이낸스에서 미국 주식 일봉 데이터를 가져오는 함수
    period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    """
    if not YFINANCE_AVAILABLE:
        print("❌ yfinance 패키지가 설치되지 않았습니다.")
        print("설치: pip install yfinance")
        return None
    
    print(f"야후 파이낸스에서 종목 코드 {symbol}의 데이터를 수집 중...")
    
    try:
        ticker = yf.Ticker(symbol)
        
        # 일봉 데이터 가져오기 (period로 기간 설정)
        hist = ticker.history(period=period)
        
        if hist.empty:
            print(f"❌ 종목 코드 {symbol}에 대한 데이터를 찾을 수 없습니다.")
            return None
        
        # 데이터프레임 정리
        df = hist.reset_index()
        
        # 실제 컬럼 수와 이름 확인
        original_columns = list(df.columns)
        num_cols = len(original_columns)
        
        # 컬럼명 매핑 (yfinance 기본 컬럼명 기반)
        column_mapping = {}
        for i, col in enumerate(original_columns):
            if i == 0:
                # 첫 번째 컬럼은 날짜
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
            # Stock Splits, Dividends, Capital Gains 등은 무시
        
        # 컬럼명 변경
        df = df.rename(columns=column_mapping)
        
        # 필요한 컬럼만 선택 (존재하는 컬럼만)
        required_cols = ['날짜', '시가', '고가', '저가', '종가', '거래량']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < 6:
            print(f"❌ 필수 컬럼이 부족합니다. 실제 컬럼: {original_columns}")
            return None
        
        df = df[available_cols]
        
        # 날짜를 datetime으로 변환 (이미 datetime일 수도 있음)
        if not pd.api.types.is_datetime64_any_dtype(df['날짜']):
            df['날짜'] = pd.to_datetime(df['날짜'])
        
        # 정렬 (날짜 오름차순)
        df = df.sort_values('날짜').reset_index(drop=True)
        try:
            if df['날짜'].dt.tz is not None:
                df['날짜'] = df['날짜'].dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
            else:
                df['날짜'] = df['날짜'].dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
        except (TypeError, AttributeError):
            # 이미 타임존이 설정되어 있지 않거나 변환 불가한 경우는 그대로 유지
            pass
        
        # 전일비 계산 (한국 주식 형식과 유사하게)
        df['전일비'] = df['종가'].diff().fillna(0)
        
        print(f"총 {len(df)}개의 일봉 데이터를 수집했습니다.")
        print(f"기간: {df['날짜'].min().strftime('%Y-%m-%d')} ~ {df['날짜'].max().strftime('%Y-%m-%d')}")
        
        return df
        
    except Exception as e:
        print(f"❌ 데이터 수집 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def fetch_stock_data(code, pages=20):
    """
    네이버 증권에서 일봉 데이터를 크롤링하는 함수
    https://finance.naver.com/item/sise_day.naver?code={code}&page={page}
    """
    base_url = "https://finance.naver.com/item/sise_day.naver"
    all_data = []
    
    print(f"네이버 증권에서 종목 코드 {code}의 데이터를 수집 중...")
    
    for page in range(1, pages + 1):
        params = {
            'code': code,
            'page': page
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': f'https://finance.naver.com/item/sise_day.naver?code={code}'
        }
        
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.encoding = 'euc-kr'  # 네이버는 euc-kr 인코딩 사용
            
            page_data = []
            
            # 방법 1: pandas read_html 시도 (가장 안정적)
            try:
                dfs = pd.read_html(StringIO(response.text), encoding='euc-kr')
                if dfs and len(dfs) > 0:
                    df_page = dfs[0]
                    
                    # 데이터프레임이 비어있지 않고 컬럼이 충분한지 확인
                    if not df_page.empty and len(df_page.columns) >= 7:
                        # 빈 행 제거
                        df_page = df_page.dropna(how='all')
                        df_page = df_page[df_page.iloc[:, 0].notna()]
                        
                        if len(df_page) > 0:
                            for idx, row in df_page.iterrows():
                                try:
                                    date_str = str(row.iloc[0]).strip()
                                    # 날짜 형식 체크 (YYYY.MM.DD)
                                    if not date_str or date_str == 'nan' or '.' not in date_str:
                                        continue
                                    
                                    # 숫자 데이터 추출
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
                pass  # BeautifulSoup으로 전환
            
            # 방법 2: BeautifulSoup 사용 (pandas 실패 시)
            if not page_data:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 여러 방법으로 테이블 찾기
                table = None
                table = soup.find('table', {'class': 'type_2'})
                if table is None:
                    table = soup.find('table', {'class': 'tb_type1'})
                if table is None:
                    table = soup.find('table', {'class': 'type_2 tb_type1'})
                if table is None:
                    # 모든 테이블 찾아서 데이터가 많은 것 선택
                    tables = soup.find_all('table')
                    for t in tables:
                        rows = t.find_all('tr')
                        if len(rows) > 3 and len(t.find_all('td')) > 20:
                            table = t
                            break
                
                if table is None:
                    if page == 1:
                        print(f"페이지 {page}: 데이터 테이블을 찾을 수 없습니다.")
                        print(f"응답 상태 코드: {response.status_code}")
                        print(f"HTML 일부 확인을 위해 페이지를 확인해주세요.")
                    break
                
                rows = table.find_all('tr')
                if len(rows) < 3:
                    print(f"페이지 {page}: 충분한 데이터 행이 없습니다.")
                    break
                
                for row in rows[2:]:  # 헤더 2줄 제외
                    cols = row.find_all(['td', 'th'])
                    if len(cols) < 7:
                        continue
                    
                    try:
                        date = cols[0].text.strip()
                        if not date or date == '' or len(date) < 8:
                            continue
                        
                        # 숫자 추출 시 공백과 콤마 제거
                        close_str = cols[1].text.strip().replace(',', '').replace(' ', '')
                        if not close_str:
                            continue
                        
                        close = int(close_str)
                        diff = cols[2].text.strip()
                        open_price = int(cols[3].text.strip().replace(',', '').replace(' ', ''))
                        high = int(cols[4].text.strip().replace(',', '').replace(' ', ''))
                        low = int(cols[5].text.strip().replace(',', '').replace(' ', ''))
                        volume = int(cols[6].text.strip().replace(',', '').replace(' ', ''))
                        
                        page_data.append({
                            '날짜': date,
                            '종가': close,
                            '전일비': diff,
                            '시가': open_price,
                            '고가': high,
                            '저가': low,
                            '거래량': volume
                        })
                    except (ValueError, AttributeError, IndexError):
                        continue
            
            if not page_data:
                print(f"페이지 {page}: 데이터가 없습니다. 크롤링 종료.")
                break
            
            all_data.extend(page_data)
            print(f"페이지 {page}/{pages} 완료 ({len(page_data)}개 행)")
            time.sleep(0.5)  # 서버 부하 방지
            
        except Exception as e:
            print(f"페이지 {page} 처리 중 오류 발생: {e}")
            if page == 1:
                import traceback
                traceback.print_exc()
            break
    
    if not all_data:
        print("수집된 데이터가 없습니다.")
        return None
    
    df = pd.DataFrame(all_data)
    
    # 날짜 형식 변환
    df['날짜'] = pd.to_datetime(df['날짜'], format='%Y.%m.%d', errors='coerce')
    df = df.dropna(subset=['날짜'])
    
    if len(df) == 0:
        print("유효한 날짜 데이터가 없습니다.")
        return None
    
    # 중복 제거 (같은 날짜가 여러 번 나온 경우)
    df = df.drop_duplicates(subset=['날짜'], keep='first')
    df = df.sort_values('날짜').reset_index(drop=True)
    
    print(f"\n총 {len(df)}개의 일봉 데이터를 수집했습니다.")
    print(f"기간: {df['날짜'].min().strftime('%Y-%m-%d')} ~ {df['날짜'].max().strftime('%Y-%m-%d')}")
    
    return df


def calculate_ma(df, periods=[5, 20, 60]):
    """
    이동평균선 계산
    """
    for period in periods:
        df[f'MA{period}'] = df['종가'].rolling(window=period).mean()
    return df


def calculate_rsi(df, period=14):
    """
    RSI (Relative Strength Index) 계산
    RSI = 100 - (100 / (1 + RS))
    RS = 평균 상승폭 / 평균 하락폭
    """
    # 전일 대비 변화량 계산
    delta = df['종가'].diff()
    
    # 상승폭과 하락폭 분리
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 평균 상승폭과 평균 하락폭 계산 (EMA 방식)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    # RS와 RSI 계산
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
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


def calculate_macd(df):
    """
    MACD (Moving Average Convergence Divergence) 계산
    MACD = EMA(12) - EMA(26)
    Signal = EMA(MACD, 9)
    """
    # EMA 계산
    ema12 = df['종가'].ewm(span=12, adjust=False).mean()
    ema26 = df['종가'].ewm(span=26, adjust=False).mean()
    
    # MACD = EMA(12) - EMA(26)
    df['MACD'] = ema12 - ema26
    
    # Signal = EMA(MACD, 9)
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # MACD Histogram = MACD - Signal
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    return df


def calculate_volume_signal(df, period=20, multiplier=1.5):
    """
    거래량 분석
    오늘 거래량이 최근 N일 평균 거래량의 배수 이상이면 강한 매수세
    """
    # 평균 거래량 계산
    df['평균거래량'] = df['거래량'].rolling(window=period).mean()
    
    # 거래량 신호 (오늘 거래량 / 평균 거래량)
    df['거래량비율'] = df['거래량'] / df['평균거래량']
    df['거래량신호'] = df['거래량비율'] >= multiplier
    
    return df


def calculate_atr(df, period=14):
    """Average True Range(ATR) 계산"""
    if not {'고가', '저가', '종가'} <= set(df.columns):
        df['TR'] = np.nan
        df['ATR'] = np.nan
        return df

    high = df['고가']
    low = df['저가']
    close = df['종가']

    prev_close = close.shift(1)
    tr_components = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1)
    df['TR'] = tr_components.max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()

    return df


def calculate_prediction_accuracy(df, days_ahead=1):
    """
    예측 정확도를 계산하는 함수
    과거 데이터를 기반으로 예측 모델의 정확도를 평가
    
    Returns:
        accuracy_score: 0-100 사이의 정확도 점수
        confidence_level: "높음", "보통", "낮음"
        factors: 정확도에 영향을 미친 요소들
    """
    if len(df) < 30:  # 최소 30일 데이터 필요
        return 50, "보통", ["데이터 부족"]
    
    accuracy_factors = []
    score = 100  # 기본 점수 100점에서 감점
    
    # 1. 과거 예측 정확도 (백테스팅)
    # 최근 10일 동안의 예측 모델과 실제 결과 비교
    if len(df) >= 15:
        backtest_days = min(10, len(df) - 5)
        prediction_errors = []
        
        for i in range(len(df) - backtest_days, len(df) - 1):
            if i >= 5:
                # 과거 시점의 추세 계산
                past_data = df.iloc[:i+1]
                if len(past_data) >= 10:
                    price_trend = past_data['종가'].tail(10).pct_change().mean()
                    actual_price = past_data['종가'].iloc[-1]
                    
                    # 1일 후 예측
                    predicted_price = actual_price * (1 + price_trend * 1 * 0.1) if price_trend > 0 else actual_price * (1 + price_trend * 1 * 0.05)
                    
                    # 실제 다음 날 가격
                    if i + 1 < len(df):
                        actual_next_price = df.iloc[i + 1]['종가']
                        error_pct = abs((predicted_price - actual_next_price) / actual_next_price) * 100
                        prediction_errors.append(error_pct)
        
        if len(prediction_errors) > 0:
            avg_error = sum(prediction_errors) / len(prediction_errors)
            # 평균 오차가 5% 이하면 높은 정확도, 10% 이하면 보통, 그 이상이면 낮음
            if avg_error <= 3:
                accuracy_factors.append("과거 예측 정확도 우수")
            elif avg_error <= 5:
                score -= 5
                accuracy_factors.append("과거 예측 정확도 양호")
            elif avg_error <= 10:
                score -= 15
                accuracy_factors.append("과거 예측 정확도 보통")
            else:
                score -= 30
                accuracy_factors.append(f"과거 예측 오차 큼 ({avg_error:.1f}%)")
    
    # 2. 추세의 강도와 일관성
    recent_df = df.tail(30)
    if len(recent_df) >= 10:
        price_changes = recent_df['종가'].tail(10).pct_change().dropna()
        trend_strength = abs(price_changes.mean())
        trend_consistency = 1 - (price_changes.std() / abs(price_changes.mean())) if price_changes.mean() != 0 else 0
        
        if trend_strength > 0.02 and trend_consistency > 0.5:
            accuracy_factors.append("강한 추세 지속")
        elif trend_strength > 0.01:
            score -= 5
            accuracy_factors.append("중간 추세")
        else:
            score -= 10
            accuracy_factors.append("약한 추세")
        
        if trend_consistency < 0.3:
            score -= 10
            accuracy_factors.append("추세 불안정")
    
    # 3. 지표 간 합의도 (Consensus)
    last_row = df.iloc[-1]
    indicators_agreement = 0
    total_indicators = 0
    
    # MA5 vs MA20
    if pd.notna(last_row.get('MA5')) and pd.notna(last_row.get('MA20')):
        ma_signal = 1 if last_row['MA5'] >= last_row['MA20'] else -1
        indicators_agreement += ma_signal
        total_indicators += 1
    
    # MACD vs Signal
    if pd.notna(last_row.get('MACD')) and pd.notna(last_row.get('MACD_Signal')):
        macd_signal = 1 if last_row['MACD'] >= last_row['MACD_Signal'] else -1
        indicators_agreement += macd_signal
        total_indicators += 1
    
    # RSI
    if pd.notna(last_row.get('RSI')):
        rsi_signal = 1 if last_row['RSI'] >= 50 else -1
        indicators_agreement += rsi_signal
        total_indicators += 1
    
    if total_indicators > 0:
        agreement_ratio = abs(indicators_agreement) / total_indicators
        if agreement_ratio >= 0.8:
            accuracy_factors.append("지표 합의도 높음")
        elif agreement_ratio >= 0.5:
            score -= 5
            accuracy_factors.append("지표 합의도 보통")
        else:
            score -= 15
            accuracy_factors.append("지표 신호 혼재")
    
    # 4. 데이터 품질
    missing_data = df.isnull().sum().sum()
    if missing_data == 0:
        accuracy_factors.append("데이터 완전")
    elif missing_data < len(df) * 0.05:
        score -= 5
        accuracy_factors.append("데이터 거의 완전")
    else:
        score -= 15
        accuracy_factors.append(f"데이터 누락 ({missing_data}개)")
    
    # 5. 변동성 (변동성이 높을수록 예측 어려움)
    if len(recent_df) >= 10:
        volatility = recent_df['종가'].tail(10).pct_change().std()
        if volatility < 0.02:
            accuracy_factors.append("변동성 낮음")
        elif volatility < 0.04:
            score -= 5
            accuracy_factors.append("변동성 보통")
        else:
            score -= 10
            accuracy_factors.append("변동성 높음")
    
    # 6. 거래량 신뢰도
    if pd.notna(last_row.get('거래량')) and pd.notna(last_row.get('평균거래량')):
        volume_ratio = last_row['거래량'] / last_row['평균거래량'] if last_row['평균거래량'] > 0 else 1
        if 0.8 <= volume_ratio <= 1.5:
            accuracy_factors.append("거래량 정상")
        elif volume_ratio > 2.0 or volume_ratio < 0.5:
            score -= 5
            accuracy_factors.append("거래량 이상")
    
    # 며칠 후 예측인지에 따라 정확도 조정 (먼 미래일수록 정확도 감소)
    if days_ahead > 1:
        # 내일모레(2일 후)는 정확도 10점 감소, 3일 후는 20점 감소
        score -= (days_ahead - 1) * 10
        if days_ahead == 2:
            accuracy_factors.append("2일 후 예측 (정확도 감소)")
        elif days_ahead >= 3:
            accuracy_factors.append(f"{days_ahead}일 후 예측 (정확도 크게 감소)")
    
    # 점수를 0-100 범위로 제한
    score = max(0, min(100, score))
    
    # 신뢰도 레벨 결정
    if score >= 80:
        confidence = "높음"
    elif score >= 60:
        confidence = "보통"
    else:
        confidence = "낮음"
    
    return score, confidence, accuracy_factors


def predict_golden_cross(df, days=7):
    """
    골든 크로스와 MACD 골든 크로스를 7일 내로 예측하는 함수
    """
    if len(df) < 26:  # MACD 계산을 위해 최소 26일 필요
        return None, None
    
    # 최근 데이터만 사용 (최근 30일)
    recent_df = df.tail(30).copy().reset_index(drop=True)
    
    # 가격 추세 분석 (최근 10일의 변화율)
    price_trend = recent_df['종가'].tail(10).pct_change().mean()
    ma5_trend = recent_df['MA5'].tail(10).diff().mean() if 'MA5' in recent_df.columns and recent_df['MA5'].notna().sum() >= 10 else 0
    ma20_trend = recent_df['MA20'].tail(10).diff().mean() if 'MA20' in recent_df.columns and recent_df['MA20'].notna().sum() >= 10 else 0
    
    # 현재 값
    last_date = recent_df['날짜'].iloc[-1]
    last_price = recent_df['종가'].iloc[-1]
    last_ma5 = recent_df['MA5'].iloc[-1] if pd.notna(recent_df['MA5'].iloc[-1]) else last_price
    last_ma20 = recent_df['MA20'].iloc[-1] if pd.notna(recent_df['MA20'].iloc[-1]) else last_price
    last_macd = recent_df['MACD'].iloc[-1] if 'MACD' in recent_df.columns and pd.notna(recent_df['MACD'].iloc[-1]) else 0
    last_signal = recent_df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in recent_df.columns and pd.notna(recent_df['MACD_Signal'].iloc[-1]) else 0
    
    # 미래 날짜 생성
    future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
    
    # 예측 데이터 저장
    predictions = []
    
    for i, future_date in enumerate(future_dates):
        day_num = i + 1
        
        # 가격 예측 (EMA 추세 사용)
        if price_trend > 0:
            predicted_price = last_price * (1 + price_trend * day_num * 0.1)  # 보수적인 예측
        else:
            predicted_price = last_price * (1 + price_trend * day_num * 0.05)
        
        # 이동평균선 예측 (최근 추세 기반)
        # MA5는 최근 5일 평균, MA20은 최근 20일 평균이므로 점진적으로 업데이트
        if day_num <= 5:
            # MA5는 새로운 데이터가 들어오면서 변화
            predicted_ma5 = (last_ma5 * 5 + predicted_price) / 6 if day_num == 1 else \
                           (last_ma5 * (5 + day_num - 1) + predicted_price) / (5 + day_num)
        else:
            # 5일 이후에는 예측된 가격들만 사용
            predicted_ma5 = predicted_price  # 단순화: 최근 추세 유지
        
        if day_num <= 20:
            # MA20도 유사하게 계산
            predicted_ma20 = (last_ma20 * 20 + predicted_price) / 21 if day_num == 1 else \
                           (last_ma20 * (20 + day_num - 1) + predicted_price) / (20 + day_num)
        else:
            predicted_ma20 = last_ma20 * (1 + ma20_trend * day_num * 0.01)
        
        # MACD 예측 (EMA 추세 사용)
        if 'MACD' in recent_df.columns:
            macd_trend = recent_df['MACD'].tail(5).diff().mean() if recent_df['MACD'].notna().sum() >= 5 else 0
            signal_trend = recent_df['MACD_Signal'].tail(5).diff().mean() if 'MACD_Signal' in recent_df.columns and recent_df['MACD_Signal'].notna().sum() >= 5 else 0
            
            predicted_macd = last_macd + macd_trend * day_num * 1.2
            predicted_signal = last_signal + signal_trend * day_num * 1.1
        
        predictions.append({
            '날짜': future_date,
            '예측종가': predicted_price,
            '예측MA5': predicted_ma5,
            '예측MA20': predicted_ma20,
            '예측MACD': predicted_macd if 'MACD' in recent_df.columns else None,
            '예측Signal': predicted_signal if 'MACD_Signal' in recent_df.columns else None,
        })
    
    # 골든 크로스 예측 결과
    gc_predictions = []
    macd_predictions = []
    
    # 이전 날짜 값 추적
    prev_ma5 = last_ma5
    prev_ma20 = last_ma20
    prev_macd = last_macd if 'MACD' in recent_df.columns else None
    prev_signal = last_signal if 'MACD_Signal' in recent_df.columns else None
    
    for i, pred in enumerate(predictions):
        day_num = i + 1
        
        # 이동평균선 골든 크로스 체크
        # 이전 날 MA5 < MA20이고 현재 예측 MA5 > MA20이면 골든 크로스 발생
        if prev_ma5 < prev_ma20 and pred['예측MA5'] > pred['예측MA20']:
            gc_predictions.append({
                'day': day_num,
                'date': pred['날짜'],
                'ma_gap_pct': ((pred['예측MA5'] - pred['예측MA20']) / pred['예측MA20']) * 100,
                'predicted_price': pred['예측종가'],
                'predicted_ma5': pred['예측MA5'],
                'predicted_ma20': pred['예측MA20']
            })
        
        # MACD 골든 크로스 체크
        if pred['예측MACD'] is not None and pred['예측Signal'] is not None:
            if prev_macd is not None and prev_signal is not None:
                if prev_macd < prev_signal and pred['예측MACD'] > pred['예측Signal']:
                    macd_predictions.append({
                        'day': day_num,
                        'date': pred['날짜'],
                        'macd_gap': pred['예측MACD'] - pred['예측Signal'],
                        'predicted_price': pred['예측종가'],
                        'predicted_macd': pred['예측MACD'],
                        'predicted_signal': pred['예측Signal']
                    })
        
        # 골든 크로스 발생 가능성 계산 (아직 발생하지 않았지만 가까워지고 있는 경우)
        ma_gap = pred['예측MA5'] - pred['예측MA20']
        ma_gap_pct = (ma_gap / pred['예측MA20']) * 100 if pred['예측MA20'] > 0 else 0
        
        # MA5가 MA20에 가까워지고 있고 상승 추세인 경우 (2% 이내)
        if pred['예측MA5'] < pred['예측MA20'] and ma_gap_pct > -2:
            # 이미 예측 목록에 없는 경우에만 추가
            if not any(p['day'] == day_num for p in gc_predictions):
                gc_predictions.append({
                    'day': day_num,
                    'date': pred['날짜'],
                    'ma_gap_pct': ma_gap_pct,
                    'predicted_price': pred['예측종가'],
                    'predicted_ma5': pred['예측MA5'],
                    'predicted_ma20': pred['예측MA20'],
                    'possibility': '높음' if ma_gap_pct > -0.5 else '보통'
                })
        
        # MACD 골든 크로스 가능성 계산
        if pred['예측MACD'] is not None and pred['예측Signal'] is not None:
            macd_gap = pred['예측MACD'] - pred['예측Signal']
            macd_gap_pct = (macd_gap / abs(pred['예측Signal'])) * 100 if pred['예측Signal'] != 0 else None
            
            if pred['예측MACD'] < pred['예측Signal'] and macd_gap_pct is not None and macd_gap_pct > -5:
                # 이미 예측 목록에 없는 경우에만 추가
                if not any(p['day'] == day_num for p in macd_predictions):
                    macd_predictions.append({
                        'day': day_num,
                        'date': pred['날짜'],
                        'macd_gap': macd_gap,
                        'predicted_price': pred['예측종가'],
                        'predicted_macd': pred['예측MACD'],
                        'predicted_signal': pred['예측Signal'],
                        'possibility': '높음' if macd_gap_pct > -1 else '보통'
                    })
        
        # 다음 반복을 위해 현재 값을 이전 값으로 업데이트
        prev_ma5 = pred['예측MA5']
        prev_ma20 = pred['예측MA20']
        if pred['예측MACD'] is not None:
            prev_macd = pred['예측MACD']
        if pred['예측Signal'] is not None:
            prev_signal = pred['예측Signal']
    
    return gc_predictions, macd_predictions


def calculate_next_golden_cross_day(df, max_days=60):
    """
    다음 골든 크로스가 발생할 정확한 일수를 계산하는 함수
    선형 보간을 사용하여 MA5와 MA20의 교차점 계산
    """
    if len(df) < 26:
        return None, None
    
    recent_df = df.tail(30).copy().reset_index(drop=True)
    last_date = recent_df['날짜'].iloc[-1]
    last_price = recent_df['종가'].iloc[-1]
    last_ma5 = recent_df['MA5'].iloc[-1] if pd.notna(recent_df['MA5'].iloc[-1]) else last_price
    last_ma20 = recent_df['MA20'].iloc[-1] if pd.notna(recent_df['MA20'].iloc[-1]) else last_price
    
    if last_ma5 >= last_ma20:
        return None, None

    prev_ma5 = recent_df['MA5'].iloc[-2] if len(recent_df) >= 2 else np.nan
    prev_ma20 = recent_df['MA20'].iloc[-2] if len(recent_df) >= 2 else np.nan
    ma5_slope_latest = last_ma5 - prev_ma5 if pd.notna(prev_ma5) else 0
    ma20_slope_latest = last_ma20 - prev_ma20 if pd.notna(prev_ma20) else 0

    if ma5_slope_latest <= 0 or ma20_slope_latest < 0:
        return None, None

    if len(recent_df) >= 10:
        ma5_slope = (recent_df['MA5'].iloc[-1] - recent_df['MA5'].iloc[-10]) / 10 if pd.notna(recent_df['MA5'].iloc[-10]) else 0
        ma20_slope = (recent_df['MA20'].iloc[-1] - recent_df['MA20'].iloc[-10]) / 10 if pd.notna(recent_df['MA20'].iloc[-10]) else 0
    else:
        ma5_slope = recent_df['MA5'].diff().mean() if 'MA5' in recent_df.columns else 0
        ma20_slope = recent_df['MA20'].diff().mean() if 'MA20' in recent_df.columns else 0
    
    # 가격 추세 기반으로 더 정확한 추세 계산
    price_trend = recent_df['종가'].tail(10).pct_change().mean()
    
    # 선형 방정식으로 교차점 계산
    # MA5(t) = last_ma5 + ma5_slope * t
    # MA20(t) = last_ma20 + ma20_slope * t
    # MA5(t) = MA20(t) 인 t를 구함
    
    gap = last_ma5 - last_ma20  # 현재 격차 (음수)
    slope_diff = ma5_slope - ma20_slope  # MA5가 MA20보다 빠르게 상승하면 양수
    
    # MACD 예측
    last_macd = recent_df['MACD'].iloc[-1] if 'MACD' in recent_df.columns and pd.notna(recent_df['MACD'].iloc[-1]) else None
    last_signal = recent_df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in recent_df.columns and pd.notna(recent_df['MACD_Signal'].iloc[-1]) else None
    
    gc_days = None
    macd_days = None
    
    if slope_diff > 0:
        gc_days = abs(gap / slope_diff) if slope_diff > 0.1 else None
        
        if gc_days and price_trend > 0:
            gc_days *= 0.9
        elif gc_days and price_trend < 0:
            gc_days *= 1.2
        
        if gc_days and gc_days > max_days:
            gc_days = None
    
    # MACD 골든 크로스 일수 계산
    if last_macd is not None and last_signal is not None and last_macd < last_signal:
        macd_gap = last_macd - last_signal
        if len(recent_df) >= 5:
            macd_slope = recent_df['MACD'].tail(5).diff().mean() if recent_df['MACD'].notna().sum() >= 5 else 0
            signal_slope = recent_df['MACD_Signal'].tail(5).diff().mean() if 'MACD_Signal' in recent_df.columns and recent_df['MACD_Signal'].notna().sum() >= 5 else 0
            macd_slope_diff = macd_slope - signal_slope
            
            if macd_slope_diff > 0:
                macd_days = abs(macd_gap / macd_slope_diff) if macd_slope_diff > 0.01 else None
                if macd_days and macd_days > max_days:
                    macd_days = None
    
    # 예측 일자 계산
    gc_date = None
    macd_date = None
    
    gc_days_rounded = None
    gc_range = None
    gc_date_range = None
    if gc_days:
        gc_days_rounded = max(1, int(round(gc_days)))
        low = max(1, int(gc_days * 0.75))
        high = max(low, int(gc_days * 1.25))
        gc_range = (low, high)
        gc_date = last_date + timedelta(days=gc_days_rounded)
        gc_date_range = (
            last_date + timedelta(days=low),
            last_date + timedelta(days=high)
        )
    else:
        gc_date = None
    
    if macd_days:
        macd_days = max(1, macd_days)
        macd_low = max(1, int(macd_days * 0.75))
        macd_high = max(macd_low, int(macd_days * 1.25))
        macd_days_int = int(round(macd_days))
        macd_date = last_date + timedelta(days=macd_days_int)
        macd_range = (macd_low, macd_high)
        macd_date_range = (
            last_date + timedelta(days=macd_low),
            last_date + timedelta(days=macd_high)
        )
    else:
        macd_date = None
        macd_range = None
        macd_date_range = None
        macd_days_int = None

    gc_result = None
    if gc_days_rounded:
        gc_result = {
            'days': gc_days_rounded,
            'days_range': gc_range,
        'date': gc_date,
            'date_range': gc_date_range,
        'current_gap': gap,
        'current_gap_pct': (gap / last_ma20) * 100 if last_ma20 > 0 else 0
        }

    macd_result = None
    if macd_days_int:
        macd_result = {
            'days': macd_days_int,
            'days_range': macd_range,
        'date': macd_date,
            'date_range': macd_date_range,
        'current_gap': macd_gap if last_macd is not None else None
        }

    return gc_result, macd_result


def calculate_next_dead_cross_day(df, max_days=60):
    """
    골든 크로스 상태일 때 다음 데드 크로스가 발생할 일수를 계산하는 함수
    데드 크로스 = MA5가 MA20을 하향 돌파
    """
    if len(df) < 26:
        return None, None
    
    recent_df = df.tail(30).copy().reset_index(drop=True)
    
    # 현재 값
    last_date = recent_df['날짜'].iloc[-1]
    last_price = recent_df['종가'].iloc[-1]
    last_ma5 = recent_df['MA5'].iloc[-1] if pd.notna(recent_df['MA5'].iloc[-1]) else last_price
    last_ma20 = recent_df['MA20'].iloc[-1] if pd.notna(recent_df['MA20'].iloc[-1]) else last_price
    
    # 골든 크로스 상태가 아니면 None 반환
    if last_ma5 < last_ma20:
        return None, None  # 이미 데드 크로스 상태이거나 골든 크로스 상태가 아님
    
    # 추세 계산 (최근 10일)
    if len(recent_df) >= 10:
        ma5_slope = (recent_df['MA5'].iloc[-1] - recent_df['MA5'].iloc[-10]) / 10 if pd.notna(recent_df['MA5'].iloc[-10]) else 0
        ma20_slope = (recent_df['MA20'].iloc[-1] - recent_df['MA20'].iloc[-10]) / 10 if pd.notna(recent_df['MA20'].iloc[-10]) else 0
    else:
        ma5_slope = recent_df['MA5'].diff().mean() if 'MA5' in recent_df.columns else 0
        ma20_slope = recent_df['MA20'].diff().mean() if 'MA20' in recent_df.columns else 0
    
    # 가격 추세
    price_trend = recent_df['종가'].tail(10).pct_change().mean()
    
    # 선형 방정식으로 교차점 계산 (데드 크로스 = MA5가 MA20을 하향 돌파)
    gap = last_ma5 - last_ma20  # 현재 격차 (양수)
    slope_diff = ma5_slope - ma20_slope  # MA5가 더 빠르게 하락하면 음수
    
    # MACD 데드 크로스 예측
    last_macd = recent_df['MACD'].iloc[-1] if 'MACD' in recent_df.columns and pd.notna(recent_df['MACD'].iloc[-1]) else None
    last_signal = recent_df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in recent_df.columns and pd.notna(recent_df['MACD_Signal'].iloc[-1]) else None
    
    dc_days = None
    macd_dc_days = None
    
    if slope_diff < 0:  # MA5가 MA20보다 빠르게 하락 중
        # 선형 보간으로 교차점 계산
        dc_days = abs(gap / abs(slope_diff)) if abs(slope_diff) > 0.1 else None
        
        # 가격 추세를 고려한 보정
        if dc_days and price_trend < 0:
            # 하락 추세면 더 빠르게 발생할 수 있음 (10% 가속)
            dc_days = dc_days * 0.9
        elif dc_days and price_trend > 0:
            # 상승 추세면 더 늦게 발생할 수 있음 (20% 지연)
            dc_days = dc_days * 1.2
        
        # 최대 기간 제한
        if dc_days and dc_days > max_days:
            dc_days = None
    
    # MACD 데드 크로스 일수 계산 (MACD가 Signal을 하향 돌파)
    if last_macd is not None and last_signal is not None and last_macd > last_signal:
        macd_gap = last_macd - last_signal
        if len(recent_df) >= 5:
            macd_slope = recent_df['MACD'].tail(5).diff().mean() if recent_df['MACD'].notna().sum() >= 5 else 0
            signal_slope = recent_df['MACD_Signal'].tail(5).diff().mean() if 'MACD_Signal' in recent_df.columns and recent_df['MACD_Signal'].notna().sum() >= 5 else 0
            macd_slope_diff = macd_slope - signal_slope
            
            if macd_slope_diff < 0:  # MACD가 Signal보다 빠르게 하락
                macd_dc_days = abs(macd_gap / abs(macd_slope_diff)) if abs(macd_slope_diff) > 0.01 else None
                if macd_dc_days and macd_dc_days > max_days:
                    macd_dc_days = None
    
    # 예측 일자 계산
    dc_date = None
    macd_dc_date = None
    
    if dc_days:
        dc_date = last_date + timedelta(days=int(round(dc_days)))
        dc_days = int(round(dc_days))
    
    if macd_dc_days:
        macd_dc_date = last_date + timedelta(days=int(round(macd_dc_days)))
        macd_dc_days = int(round(macd_dc_days))
    
    return {
        'days': dc_days,
        'date': dc_date,
        'current_gap': gap,
        'current_gap_pct': (gap / last_ma20) * 100 if last_ma20 > 0 else 0
    } if dc_days else None, {
        'days': macd_dc_days,
        'date': macd_dc_date,
        'current_gap': macd_gap if last_macd is not None else None
    } if macd_dc_days else None


def predict_adjustment_period(df, max_days=60):
    """
    데드 크로스 상태일 때 조정 기간 예측 (골든 크로스 재발생까지의 기간)
    """
    if len(df) < 26:
        return None
    
    recent_df = df.tail(30).copy().reset_index(drop=True)
    
    # 현재 상태 확인
    last_date = recent_df['날짜'].iloc[-1]
    last_ma5 = recent_df['MA5'].iloc[-1] if pd.notna(recent_df['MA5'].iloc[-1]) else None
    last_ma20 = recent_df['MA20'].iloc[-1] if pd.notna(recent_df['MA20'].iloc[-1]) else None
    last_rsi = recent_df['RSI'].iloc[-1] if 'RSI' in recent_df.columns and pd.notna(recent_df['RSI'].iloc[-1]) else None
    
    # 데드 크로스 상태가 아니면 None 반환
    if not last_ma5 or not last_ma20 or last_ma5 >= last_ma20:
        return None
    
    # 데드 크로스 이후 얼마나 지났는지 계산
    dead_cross_days_ago = 0
    for i in range(len(recent_df) - 1, -1, -1):
        if i > 0:
            prev_ma5 = recent_df.loc[i-1, 'MA5']
            prev_ma20 = recent_df.loc[i-1, 'MA20']
            curr_ma5 = recent_df.loc[i, 'MA5']
            curr_ma20 = recent_df.loc[i, 'MA20']
            
            if pd.notna(prev_ma5) and pd.notna(prev_ma20) and pd.notna(curr_ma5) and pd.notna(curr_ma20):
                if prev_ma5 >= prev_ma20 and curr_ma5 < curr_ma20:
                    dead_cross_days_ago = len(recent_df) - 1 - i
                    break
    
    # 현재 MA 격차
    current_gap_pct = ((last_ma5 - last_ma20) / last_ma20) * 100 if last_ma20 > 0 else 0
    
    # 과거 데드 크로스 패턴 분석
    if len(df) >= 60:
        past_dead_crosses = []
        for i in range(1, len(df)):
            prev_ma5 = df.loc[i-1, 'MA5']
            prev_ma20 = df.loc[i-1, 'MA20']
            curr_ma5 = df.loc[i, 'MA5']
            curr_ma20 = df.loc[i, 'MA20']
            
            if pd.notna(prev_ma5) and pd.notna(prev_ma20) and pd.notna(curr_ma5) and pd.notna(curr_ma20):
                if prev_ma5 >= prev_ma20 and curr_ma5 < curr_ma20:
                    # 데드 크로스 발생 시점부터 다음 골든 크로스까지 기간 찾기
                    for j in range(i, min(i + 60, len(df))):
                        if j < len(df) - 1:
                            next_ma5 = df.loc[j+1, 'MA5']
                            next_ma20 = df.loc[j+1, 'MA20']
                            if pd.notna(next_ma5) and pd.notna(next_ma20):
                                if next_ma5 > next_ma20:
                                    # 골든 크로스 재발생
                                    duration = j - i + 1
                                    past_dead_crosses.append({
                                        'duration': duration
                                    })
                                    break
        
        if len(past_dead_crosses) > 0:
            durations = [dc['duration'] for dc in past_dead_crosses]
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
        else:
            # 과거 데이터가 없으면 일반적인 패턴 사용
            avg_duration = 15
            min_duration = 7
            max_duration = 25
    else:
        # 데이터가 부족하면 일반 패턴 사용
        avg_duration = 15
        min_duration = 7
        max_duration = 25
    
    # RSI 기반 조정 종료 예측
    if last_rsi is not None:
        if last_rsi > 40:
            # RSI가 여전히 높으면 조정이 시작 단계
            adjustment_factor = 1.2
        elif last_rsi > 30:
            # RSI가 중간이면 조정 중
            adjustment_factor = 1.0
        else:
            # RSI가 낮으면 조정이 거의 끝나감
            adjustment_factor = 0.8
    else:
        adjustment_factor = 1.0
    
    # 현재 격차 기반 예측
    if abs(current_gap_pct) > 5:
        # 격차가 크면 조정이 더 길 수 있음
        gap_factor = 1.3
    else:
        # 격차가 작으면 조정이 곧 끝날 수 있음
        gap_factor = 0.9
    
    # 최종 예측 계산
    final_min = max(1, int(min_duration * adjustment_factor * gap_factor) - dead_cross_days_ago)
    final_max = max(final_min + 1, int(max_duration * adjustment_factor * gap_factor) - dead_cross_days_ago)
    
    # 최대 기간 제한
    final_max = min(final_max, max_days - dead_cross_days_ago)
    
    if final_min >= final_max:
        final_max = final_min + 5
    
    # 진행률 계산
    if avg_duration > 0:
        progress_pct = min(100, (dead_cross_days_ago / avg_duration) * 100)
    else:
        progress_pct = 0
    
    end_date_min = last_date + timedelta(days=int(final_min))
    end_date_max = last_date + timedelta(days=int(final_max))
    
    return {
        'min_days': int(final_min),
        'max_days': int(final_max),
        'end_date_min': end_date_min.strftime('%Y-%m-%d'),
        'end_date_max': end_date_max.strftime('%Y-%m-%d'),
        'days_since_dead_cross': dead_cross_days_ago,
        'current_gap_pct': current_gap_pct,
        'progress_pct': progress_pct,
        'current_rsi': float(last_rsi) if last_rsi is not None else None
    }


def predict_peak_after_golden_cross(df, max_days=60):
    """
    골든 크로스 상태일 때 피크 시점 예측 (최소 N일, 최대 M일)
    """
    if len(df) < 26:
        return None
    
    recent_df = df.tail(30).copy().reset_index(drop=True)
    
    # 현재 상태 확인
    last_date = recent_df['날짜'].iloc[-1]
    last_ma5 = recent_df['MA5'].iloc[-1] if pd.notna(recent_df['MA5'].iloc[-1]) else None
    last_ma20 = recent_df['MA20'].iloc[-1] if pd.notna(recent_df['MA20'].iloc[-1]) else None
    last_rsi = recent_df['RSI'].iloc[-1] if 'RSI' in recent_df.columns and pd.notna(recent_df['RSI'].iloc[-1]) else None
    last_price = recent_df['종가'].iloc[-1]
    
    # 골든 크로스 상태가 아니면 None 반환
    if not last_ma5 or not last_ma20 or last_ma5 < last_ma20:
        return None
    
    # 골든 크로스 이후 얼마나 지났는지 계산
    golden_cross_days_ago = 0
    for i in range(len(recent_df) - 1, -1, -1):
        if i > 0:
            prev_ma5 = recent_df.loc[i-1, 'MA5']
            prev_ma20 = recent_df.loc[i-1, 'MA20']
            curr_ma5 = recent_df.loc[i, 'MA5']
            curr_ma20 = recent_df.loc[i, 'MA20']
            
            if pd.notna(prev_ma5) and pd.notna(prev_ma20) and pd.notna(curr_ma5) and pd.notna(curr_ma20):
                if prev_ma5 < prev_ma20 and curr_ma5 >= curr_ma20:
                    golden_cross_days_ago = len(recent_df) - 1 - i
                    break
    
    # 현재 MA 격차
    current_gap_pct = ((last_ma5 - last_ma20) / last_ma20) * 100 if last_ma20 > 0 else 0
    
    # 과거 골든 크로스 패턴 분석 (전체 데이터에서)
    if len(df) >= 60:
        past_golden_crosses = []
        for i in range(1, len(df)):
            prev_ma5 = df.loc[i-1, 'MA5']
            prev_ma20 = df.loc[i-1, 'MA20']
            curr_ma5 = df.loc[i, 'MA5']
            curr_ma20 = df.loc[i, 'MA20']
            
            if pd.notna(prev_ma5) and pd.notna(prev_ma20) and pd.notna(curr_ma5) and pd.notna(curr_ma20):
                if prev_ma5 < prev_ma20 and curr_ma5 >= curr_ma20:
                    # 골든 크로스 발생 시점부터 데드 크로스까지 기간 찾기
                    for j in range(i, min(i + 60, len(df))):
                        if j < len(df) - 1:
                            next_ma5 = df.loc[j+1, 'MA5']
                            next_ma20 = df.loc[j+1, 'MA20']
                            if pd.notna(next_ma5) and pd.notna(next_ma20):
                                if next_ma5 < next_ma20:
                                    # 데드 크로스 발생
                                    duration = j - i + 1
                                    # 최대 격차 찾기
                                    max_gap = 0
                                    max_gap_day = 0
                                    for k in range(i, j+1):
                                        gap = ((df.loc[k, 'MA5'] - df.loc[k, 'MA20']) / df.loc[k, 'MA20']) * 100 if pd.notna(df.loc[k, 'MA20']) and df.loc[k, 'MA20'] > 0 else 0
                                        if gap > max_gap:
                                            max_gap = gap
                                            max_gap_day = k - i
                                    past_golden_crosses.append({
                                        'duration': duration,
                                        'peak_day': max_gap_day
                                    })
                                    break
        
        if len(past_golden_crosses) > 0:
            avg_duration = sum([gc['duration'] for gc in past_golden_crosses]) / len(past_golden_crosses)
            avg_peak_day = sum([gc['peak_day'] for gc in past_golden_crosses]) / len(past_golden_crosses)
            min_peak_day = min([gc['peak_day'] for gc in past_golden_crosses])
            max_peak_day = max([gc['peak_day'] for gc in past_golden_crosses])
        else:
            # 과거 데이터가 없으면 일반적인 패턴 사용
            avg_duration = 20
            avg_peak_day = 10
            min_peak_day = 5
            max_peak_day = 15
    else:
        # 데이터가 부족하면 일반 패턴 사용
        avg_duration = 20
        avg_peak_day = 10
        min_peak_day = 5
        max_peak_day = 15
    
    # RSI 기반 피크 예측
    rsi_peak_days = None
    if last_rsi is not None:
        if last_rsi < 60:
            # RSI가 60 미만이면 평균적으로 5-15일 후에 과열 구간 도달 예상
            rsi_peak_days = {
                'min': max(5, int(avg_peak_day * 0.8)),
                'max': max(15, int(avg_peak_day * 1.5))
            }
        elif last_rsi < 70:
            # RSI가 60-70 사이면 곧 과열 구간 도달
            rsi_peak_days = {
                'min': 2,
                'max': 8
            }
        else:
            # 이미 과열 구간이면 곧 조정 가능
            rsi_peak_days = {
                'min': 0,
                'max': 5
            }
    
    # 현재 격차 기반 예측
    if current_gap_pct < 5:
        # 격차가 작으면 아직 상승 여지 있음
        gap_based_days = {
            'min': int(avg_peak_day * 0.7),
            'max': int(avg_peak_day * 1.8)
        }
    elif current_gap_pct < 10:
        # 격차가 중간이면 상승 중
        gap_based_days = {
            'min': int(avg_peak_day * 0.5),
            'max': int(avg_peak_day * 1.3)
        }
    else:
        # 격차가 크면 곧 피크일 가능성
        gap_based_days = {
            'min': 0,
            'max': int(avg_peak_day * 0.8)
        }
    
    # 여러 방법의 예측을 종합
    predictions = []
    if rsi_peak_days:
        predictions.append(rsi_peak_days)
    predictions.append(gap_based_days)
    predictions.append({
        'min': min_peak_day,
        'max': max_peak_day
    })
    
    # 이미 경과한 일수 고려
    if golden_cross_days_ago > 0:
        # 이미 N일 지났으므로 예측 일수에서 빼기
        final_min = max(0, min([p['min'] for p in predictions]) - golden_cross_days_ago)
        final_max = max(0, max([p['max'] for p in predictions]) - golden_cross_days_ago)
    else:
        final_min = min([p['min'] for p in predictions])
        final_max = max([p['max'] for p in predictions])
    
    # 최대 기간 제한
    final_max = min(final_max, max_days - golden_cross_days_ago)
    
    if final_min >= final_max:
        final_max = final_min + 5
    
    peak_date_min = last_date + timedelta(days=int(final_min))
    peak_date_max = last_date + timedelta(days=int(final_max))
    
    return {
        'min_days': int(final_min),
        'max_days': int(final_max),
        'peak_date_min': peak_date_min.strftime('%Y-%m-%d'),
        'peak_date_max': peak_date_max.strftime('%Y-%m-%d'),
        'days_since_golden_cross': golden_cross_days_ago,
        'current_gap_pct': current_gap_pct,
        'current_rsi': float(last_rsi) if last_rsi is not None else None
    }


def analyze_momentum(df):
    """
    상승 모멘텀 분석 함수
    RSI, MACD, 거래량, 이동평균선을 종합 분석
    """
    if len(df) < 26:
        return {
            'momentum': '분석 불가',
            'score': 0,
            'signals': [],
            'warnings': [],
            'message': '데이터가 부족하여 분석할 수 없습니다.'
        }
    
    recent_df = df.tail(10).copy().reset_index(drop=True)
    last_row = recent_df.iloc[-1]
    
    signals = []  # 모멘텀 있는 신호
    warnings = []  # 모멘텀 약한 신호
    
    momentum_score = 0
    
    # 1. RSI 분석
    if 'RSI' in recent_df.columns and pd.notna(last_row['RSI']):
        last_rsi = last_row['RSI']
        prev_rsi = recent_df.iloc[-2]['RSI'] if len(recent_df) >= 2 and pd.notna(recent_df.iloc[-2]['RSI']) else None
        
        if last_rsi > 50:
            if prev_rsi and last_rsi > prev_rsi:
                signals.append(f"RSI가 50 위에서 상승 중 ({prev_rsi:.1f} → {last_rsi:.1f})")
                momentum_score += 2
            elif last_rsi >= 50:
                signals.append(f"RSI가 50 위에 위치 ({last_rsi:.1f})")
                momentum_score += 1
        else:
            warnings.append(f"RSI가 50 미만 ({last_rsi:.1f})")
            momentum_score -= 1
        
        if 60 <= last_rsi <= 70:
            if prev_rsi and last_rsi <= prev_rsi:
                warnings.append(f"RSI가 과열 구간(60-70)에서 하락 또는 정체 ({last_rsi:.1f})")
                momentum_score -= 1
        elif last_rsi > 70:
            warnings.append(f"RSI 과열 상태 ({last_rsi:.1f})")
            momentum_score -= 1
    
    # 2. MACD 분석
    if 'MACD' in recent_df.columns and 'MACD_Signal' in recent_df.columns:
        if pd.notna(last_row['MACD']) and pd.notna(last_row['MACD_Signal']):
            last_macd = last_row['MACD']
            last_signal = last_row['MACD_Signal']
            macd_gap = last_macd - last_signal
            
            if len(recent_df) >= 2:
                prev_macd = recent_df.iloc[-2]['MACD']
                prev_signal = recent_df.iloc[-2]['MACD_Signal']
                prev_gap = prev_macd - prev_signal if pd.notna(prev_macd) and pd.notna(prev_signal) else 0
                
                if last_macd > last_signal:
                    if macd_gap > prev_gap:
                        signals.append(f"MACD가 신호선 위에서 격차 증가 ({prev_gap:.2f} → {macd_gap:.2f})")
                        momentum_score += 2
                    else:
                        signals.append(f"MACD가 신호선 위에 있음 (격차: {macd_gap:.2f})")
                        momentum_score += 1
                else:
                    warnings.append(f"MACD가 신호선 아래 ({macd_gap:.2f})")
                    momentum_score -= 1
                
                if macd_gap < prev_gap and macd_gap > 0:
                    warnings.append(f"MACD 격차가 줄어드는 중 ({prev_gap:.2f} → {macd_gap:.2f})")
                    momentum_score -= 1
                elif macd_gap < 0 and prev_gap > 0:
                    warnings.append("MACD가 신호선 아래로 교차할 조짐")
                    momentum_score -= 2
    
    # 3. 거래량 분석
    if '거래량' in recent_df.columns and '평균거래량' in recent_df.columns:
        if pd.notna(last_row['거래량']) and pd.notna(last_row['평균거래량']):
            last_volume = last_row['거래량']
            avg_volume = last_row['평균거래량']
            volume_ratio = last_volume / avg_volume if avg_volume > 0 else 1
            
            if len(recent_df) >= 2:
                prev_volume = recent_df.iloc[-2]['거래량']
                prev_price = recent_df.iloc[-2]['종가']
                last_price = last_row['종가']
                
                price_change = ((last_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0
                
                if volume_ratio >= 1.2 and price_change > 0:
                    signals.append(f"거래량 증가와 함께 주가 상승 (거래량 비율: {volume_ratio:.2f}배, 가격 상승: {price_change:.2f}%)")
                    momentum_score += 2
                elif volume_ratio >= 1.0:
                    signals.append(f"거래량이 평균 이상 (비율: {volume_ratio:.2f}배)")
                    momentum_score += 1
                
                if volume_ratio < 0.8 and price_change > 0:
                    warnings.append(f"거래량 감소하는데 주가만 오름 (거래량 비율: {volume_ratio:.2f}배)")
                    momentum_score -= 2
    
    # 4. 이동평균선 분석
    if 'MA5' in recent_df.columns and 'MA20' in recent_df.columns:
        if pd.notna(last_row['MA5']) and pd.notna(last_row['MA20']):
            last_ma5 = last_row['MA5']
            last_ma20 = last_row['MA20']
            ma_gap = last_ma5 - last_ma20
            ma_gap_pct = (ma_gap / last_ma20) * 100 if last_ma20 > 0 else 0
            
            if len(recent_df) >= 2:
                prev_ma5 = recent_df.iloc[-2]['MA5']
                prev_ma20 = recent_df.iloc[-2]['MA20']
                prev_gap = prev_ma5 - prev_ma20
                prev_gap_pct = (prev_gap / prev_ma20) * 100 if prev_ma20 > 0 else 0
                
                # MA5 기울기 확인
                ma5_slope = last_ma5 - prev_ma5 if pd.notna(prev_ma5) else 0
                
                if last_ma5 > last_ma20:
                    if ma_gap_pct > prev_gap_pct:
                        signals.append(f"MA5가 MA20 위에서 격차 벌어짐 (격차: {prev_gap_pct:.2f}% → {ma_gap_pct:.2f}%)")
                        momentum_score += 2
                    elif abs(ma_gap_pct - prev_gap_pct) < 0.5:
                        signals.append(f"MA5가 MA20 위에서 격차 유지 (격차: {ma_gap_pct:.2f}%)")
                        momentum_score += 1
                    
                    if ma5_slope < 0:
                        warnings.append(f"MA5가 하향으로 기울기 전환 (격차: {ma_gap_pct:.2f}%)")
                        momentum_score -= 1
                    
                    if ma_gap_pct < prev_gap_pct:
                        warnings.append(f"MA5-MA20 격차가 줄어드는 중 ({prev_gap_pct:.2f}% → {ma_gap_pct:.2f}%)")
                        momentum_score -= 1
                else:
                    warnings.append(f"MA5가 MA20 아래 ({ma_gap_pct:.2f}%)")
                    momentum_score -= 2
    
    # 종합 판단
    if momentum_score >= 5:
        momentum_status = "강한 상승 모멘텀"
    elif momentum_score >= 3:
        momentum_status = "상승 모멘텀 있음"
    elif momentum_score >= 1:
        momentum_status = "약한 상승 모멘텀"
    elif momentum_score >= -1:
        momentum_status = "모멘텀 중립"
    elif momentum_score >= -3:
        momentum_status = "모멘텀 약화"
    else:
        momentum_status = "하락 전환 가능성"
    
    message = f"종합 점수: {momentum_score}점"
    if len(signals) > len(warnings):
        message += " - 상승 모멘텀 신호가 더 많습니다."
    elif len(warnings) > len(signals):
        message += " - 주의 신호가 더 많습니다."
    else:
        message += " - 신호가 혼재되어 있습니다."
    
    return {
        'momentum': momentum_status,
        'score': momentum_score,
        'signals': signals,
        'warnings': warnings,
        'message': message
    }


def determine_market_regime(df):
    if len(df) == 0:
        return "데이터 부족", ["데이터 없음"]

    last = df.iloc[-1]
    regime = "횡보 레짐"
    descriptors = []

    ma5 = last.get('MA5')
    ma20 = last.get('MA20')
    ma60 = last.get('MA60')

    if pd.notna(ma5) and pd.notna(ma20) and pd.notna(ma60):
        if ma5 > ma20 > ma60:
            regime = "상승 레짐"
            descriptors.append("MA60 < MA20 < MA5")
        elif ma5 < ma20 < ma60:
            regime = "하락 레짐"
            descriptors.append("MA5 < MA20 < MA60")
        else:
            descriptors.append("이동평균 혼재")
    else:
        descriptors.append("이동평균 데이터 부족")

    macd = last.get('MACD')
    macd_signal = last.get('MACD_Signal')
    if pd.notna(macd) and pd.notna(macd_signal):
        descriptors.append("MACD>Signal" if macd >= macd_signal else "MACD<Signal")

    return regime, descriptors


def build_signal_summary(momentum_info):
    positives = len(momentum_info.get('signals', []))
    negatives = len(momentum_info.get('warnings', []))
    score = momentum_info.get('score', 0)
    status = momentum_info.get('momentum', '상태 미확인')
    return f"[신호 요약] 매수 {positives} / 주의 {negatives} → 총점 {score} ({status})"


def compute_atr_stop(df, multiplier=2.0):
    if 'ATR' not in df.columns or len(df) == 0:
        return None
    atr_value = df['ATR'].iloc[-1]
    if pd.isna(atr_value):
        return None
    close_price = df['종가'].iloc[-1]
    stop_price = max(0, close_price - atr_value * multiplier)
    drop_pct = (close_price - stop_price) / close_price * 100 if close_price else None
    return {
        'atr': atr_value,
        'stop': stop_price,
        'multiplier': multiplier,
        'drop_pct': drop_pct
    }


def explain_ma_relationship(ma5, ma20, ma60, price_formatter):
    print("📈 [1] 이동평균 해석")
    if not all(pd.notna(val) for val in (ma5, ma20, ma60)):
        print("   → 이동평균 데이터를 충분히 확보하지 못했습니다.\n")
        return
    print(f"   단기(MA5)={price_formatter(ma5)}, 중기(MA20)={price_formatter(ma20)}, 장기(MA60)={price_formatter(ma60)}")
    if ma60 < ma20 < ma5:
        print("   → 정배열 (상승 추세)."
              " 단기선이 위에 있고 장기선이 아래에 있어 상승 흐름이 정돈돼 있습니다.")
        print("   👉 눌림이 나오면 분할 매수 전략이 유효합니다.\n")
    elif ma5 < ma20 < ma60:
        print("   → 역배열 (하락 추세)."
              " 단기/중기선이 모두 장기선 아래로 꺾여 있어 하락 압력이 큽니다.")
        print("   👉 대기 모드 유지, 추세 전환 신호가 나올 때까지 현금 비중을 높게 두세요.\n")
    else:
        print("   → 선들이 섞여 있어 추세가 명확하지 않습니다.")
        print("   👉 소액 탐색만 하거나, 방향이 확실해질 때까지 관망하는 편이 안전합니다.\n")


def explain_macd_signal(macd, signal):
    print("📊 [2] MACD 해석")
    if pd.isna(macd) or pd.isna(signal):
        print("   → MACD 데이터를 충분히 확보하지 못했습니다.\n")
        return
    print(f"   MACD={macd:.2f}, Signal={signal:.2f}")
    if macd > signal:
        print("   → MACD가 신호선 위: 최근 12일 상승력(EMA12)이 26일 평균보다 강합니다.")
        print("   👉 모멘텀이 살아나고 있어 추가 상승을 기대할 수 있습니다.\n")
    else:
        print("   → MACD가 신호선 아래: 단기 모멘텀이 약해졌거나 하락 쪽 힘이 커졌습니다.")
        print("   👉 단독 매수 신호로 보기보단 조정에 대비하는 게 좋습니다.\n")


def explain_rsi_signal(rsi):
    print("💡 [3] RSI 해석")
    if pd.isna(rsi):
        print("   → RSI 데이터를 충분히 확보하지 못했습니다.\n")
        return
    print(f"   RSI={rsi:.1f}")
    if rsi >= 70:
        print("   → 과매수 구간. 너무 많이 올라서 단기 조정 가능성이 있습니다.")
    elif rsi <= 30:
        print("   → 과매도 구간. 급락 후 기술적 반등이 나올 수 있는 자리입니다.")
            else:
        print("   → 매수·매도 힘이 비슷한 중립 구간입니다.")
    print("   👉 RSI는 30 근처에서 분할 매수, 70 근처에서 분할 매도를 연습하면 이해가 빨라요.\n")


def explain_atr_strategy(atr_info, price_formatter):
    print("🛡️ [4] ATR 손절 가이드")
    if not atr_info:
        print("   → ATR 데이터를 계산하지 못했습니다.\n")
        return
    atr_val = atr_info['atr']
    stop_price = atr_info['stop']
    drop_pct = atr_info.get('drop_pct')
    print(f"   ATR(14)={atr_val:.2f}, 권장 손절={price_formatter(stop_price)}")
    if drop_pct is not None:
        print(f"   → 현재가 대비 약 {drop_pct:.2f}% 아래에서 리스크를 관리할 수 있습니다.\n")
            else:
        print("   → 하루 평균 변동폭을 감안해 손실을 제한하는 위치입니다.\n")


def explain_action_plan(regime_label):
    print("📉 [5] 오늘의 행동 가이드")
    label = regime_label or "⚖️ 전환/횡보 레짐"
    if "하락" in label:
        print("   🔴 하락 레짐으로 판단됩니다.")
        print("   - 기존 보유분은 반등 시 분할 축소를 고려하세요.")
        print("   - 신규 진입은 골든크로스·거래량 폭증 등 확실한 전환 신호 이후가 안전합니다.\n")
    elif "상승" in label:
        print("   🟢 상승 레짐입니다.")
        print("   - 눌림목에서 2~3회 분할 매수를 계획하세요.")
        print("   - 손절선은 이평선 혹은 ATR 손절 가이드를 기준으로 잡아두세요.\n")
                    else:
        print("   ⚪ 방향성이 뚜렷하지 않은 전환/횡보 구간입니다.")
        print("   - 소액 탐색, 공부, 데이터 수집에 집중하는 것이 좋습니다.")
        print("   - 명확한 추세가 형성될 때까지 큰 금액 투자는 미루세요.\n")


def calculate_periodic_returns(df, period_map):
    results = {}
    if len(df) == 0:
        return results

    close_series = df['종가']
    latest_price = close_series.iloc[-1]

    for label, days in period_map.items():
        shifted = close_series.shift(days)
        past_price = shifted.iloc[-1]
        if pd.notna(past_price) and past_price > 0:
            pct = (latest_price / past_price - 1) * 100
            results[label] = pct
                else:
            results[label] = None

    return results


def print_periodic_returns(returns_map):
    print("📆 기간별 누적 수익률")
    for label, value in returns_map.items():
        if value is None:
            print(f"   {label}: 데이터 부족")
        else:
            print(f"   {label}: {format_percentage(value)}")


def print_golden_cross_events(events, currency_code):
    if not events:
        print("📉 최근 1.5년 내 조건을 충족한 골든크로스가 없습니다.")
        return

    print(f"📈 의미 있는 골든크로스 {len(events)}건")
    for event in events:
        price_text = format_price(event['종가'], currency_code)
        ma5_text = format_price(event['MA5'], currency_code)
        ma20_text = format_price(event['MA20'], currency_code)
        ma60_text = format_price(event['MA60'], currency_code)
        rsi_val = event.get('RSI')
        rsi_text = f"RSI {rsi_val:.1f}" if rsi_val is not None and not pd.isna(rsi_val) else "RSI N/A"
        volume_ratio = event.get('거래량비율')
        volume_text = f"거래량 {volume_ratio:.2f}배" if volume_ratio is not None and not pd.isna(volume_ratio) else "거래량 확인 필요"
        macd_flag = "MACD 동시 골크" if event.get('MACD_골든') else "MACD 대기"
        print(f"   - {event['날짜'].strftime('%Y-%m-%d')}: {price_text}, MA5 {ma5_text}, MA20 {ma20_text}, MA60 {ma60_text}")
        print(f"     ▸ {rsi_text} | {volume_text} | {macd_flag}")


def print_prediction_section(next_gc, next_macd):
    print("🔮 향후 교차 예측")

    if next_gc:
        low, high = next_gc['days_range'] if next_gc.get('days_range') else (next_gc['days'], next_gc['days'])
        date_low, date_high = next_gc['date_range'] if next_gc.get('date_range') else (next_gc['date'], next_gc['date'])
        print(f"   이동평균 골든크로스: {low}~{high}일 내 (예상: {date_low.strftime('%Y-%m-%d')} ~ {date_high.strftime('%Y-%m-%d')})")
        print(f"     현재 MA5-MA20 격차: {format_percentage(next_gc['current_gap_pct'])}")
                else:
        print("   이동평균 골든크로스: 예측 불가 (추세 정체 또는 하락)")
    
    if next_macd:
        low, high = next_macd['days_range'] if next_macd.get('days_range') else (next_macd['days'], next_macd['days'])
        date_low, date_high = next_macd['date_range'] if next_macd.get('date_range') else (next_macd['date'], next_macd['date'])
        print(f"   MACD 골든크로스: {low}~{high}일 내 (예상: {date_low.strftime('%Y-%m-%d')} ~ {date_high.strftime('%Y-%m-%d')})")
    else:
        print("   MACD 골든크로스: 예측 불가 (모멘텀 부족)")


def generate_analysis_report(df, code, is_us, golden_events):
    currency_code = get_currency_code(is_us)
    last_row = df.iloc[-1]
    current_price = last_row['종가']
    prev_price = df['종가'].iloc[-2] if len(df) >= 2 else None
    price_change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price else None

    regime, descriptors = determine_market_regime(df)
    momentum_info = analyze_momentum(df)
    signal_summary = build_signal_summary(momentum_info)

    print(f"\n[시장 상태] {regime} ({', '.join(descriptors)})")
    print(signal_summary)

    print("\n📌 현재 가격 & 기준선")
    print(f"   종가: {format_price(current_price, currency_code)} ({format_percentage(price_change_pct)})")
    ma5 = last_row.get('MA5')
    ma20 = last_row.get('MA20')
    ma60 = last_row.get('MA60') if 'MA60' in last_row else None
    if pd.notna(ma5) and pd.notna(ma20):
        print(f"   MA5: {format_price(ma5, currency_code)} | MA20: {format_price(ma20, currency_code)}")
    if ma60 is not None and pd.notna(ma60):
        print(f"   MA60: {format_price(ma60, currency_code)}")
    rsi_val = last_row.get('RSI')
    if pd.notna(rsi_val):
        print(f"   RSI: {rsi_val:.1f}")
    
    atr_info = compute_atr_stop(df)

    print("\n")
    print_golden_cross_events(golden_events, currency_code)
    print("\n")
    
    next_gc, next_macd = calculate_next_golden_cross_day(df, max_days=60)
    print_prediction_section(next_gc, next_macd)
    
    print("\n")
    period_map = {
        "1개월": 21,
        "3개월": 63,
        "6개월": 126,
        "1년": 252,
        "2년": 504,
        "5년": 1260
    }
    period_returns = calculate_periodic_returns(df, period_map)
    print_periodic_returns(period_returns)
    
    if momentum_info.get('signals'):
        top_signals = momentum_info['signals'][:2]
        print("\n✅ 주요 매수 신호")
        for sig in top_signals:
            print(f"   - {sig}")
    
    if momentum_info.get('warnings'):
        top_warnings = momentum_info['warnings'][:2]
        print("\n⚠️  주요 경고")
        for warn in top_warnings:
            print(f"   - {warn}")

    price_formatter = lambda value: format_price(value, currency_code)
    print("\n📘 공부용 해설")
    explain_ma_relationship(last_row.get('MA5'), last_row.get('MA20'), last_row.get('MA60'), price_formatter)
    explain_macd_signal(last_row.get('MACD'), last_row.get('MACD_Signal'))
    explain_rsi_signal(last_row.get('RSI'))
    explain_atr_strategy(atr_info, price_formatter)
    explain_action_plan(regime)


def find_golden_cross(df, code=None, volume_multiplier=1.3):
    """필터링된 골든크로스 이벤트를 탐지하고 DataFrame에 표시합니다."""

    required_columns = {'MA5', 'MA20', '종가'}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"find_golden_cross 실행 전 {missing_cols} 컬럼이 필요합니다.")

    df = df.copy()
    if 'MA60' not in df.columns:
        df['MA60'] = df['종가'].rolling(window=60).mean()

    if '거래량비율' not in df.columns:
        if '거래량' in df.columns:
            df['거래량비율'] = df['거래량'] / df['거래량'].rolling(window=20).mean()
            else:
            df['거래량비율'] = np.nan

    df['골든크로스'] = False
    df['데드크로스'] = False
    df['MACD_골든크로스'] = False

    events = []
    last_confirmed_date = None

    for i in range(1, len(df)):
        prev_ma5 = df.iloc[i - 1]['MA5']
        prev_ma20 = df.iloc[i - 1]['MA20']
        curr_ma5 = df.iloc[i]['MA5']
        curr_ma20 = df.iloc[i]['MA20']
        curr_ma60 = df.iloc[i]['MA60'] if 'MA60' in df.columns else np.nan
        curr_date = df.iloc[i]['날짜'] if '날짜' in df.columns else None

        if pd.notna(prev_ma5) and pd.notna(prev_ma20) and pd.notna(curr_ma5) and pd.notna(curr_ma20):
            crossed_up = prev_ma5 < prev_ma20 and curr_ma5 >= curr_ma20
            crossed_down = prev_ma5 > prev_ma20 and curr_ma5 <= curr_ma20

            if crossed_down:
                df.iloc[i, df.columns.get_loc('데드크로스')] = True

            if crossed_up:
                trend_ok = pd.notna(curr_ma20) and pd.notna(curr_ma60) and curr_ma20 > curr_ma60
                volume_ratio = df.iloc[i]['거래량비율'] if '거래량비율' in df.columns else np.nan
                volume_ok = pd.notna(volume_ratio) and volume_ratio >= volume_multiplier

                separated = True
                if last_confirmed_date is not None and curr_date is not None:
                    separated = (curr_date - last_confirmed_date).days > 5

                if trend_ok and volume_ok and separated:
                    df.iloc[i, df.columns.get_loc('골든크로스')] = True
                    last_confirmed_date = curr_date

                    macd_golden = False
                    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                        prev_macd = df.iloc[i - 1]['MACD']
                        prev_signal = df.iloc[i - 1]['MACD_Signal']
                        curr_macd = df.iloc[i]['MACD']
                        curr_signal = df.iloc[i]['MACD_Signal']
                        if pd.notna(prev_macd) and pd.notna(prev_signal) and pd.notna(curr_macd) and pd.notna(curr_signal):
                            macd_golden = prev_macd < prev_signal and curr_macd >= curr_signal
                    df.iloc[i, df.columns.get_loc('MACD_골든크로스')] = macd_golden

                    events.append({
                        '날짜': curr_date,
                        '종가': df.iloc[i]['종가'],
                        'MA5': curr_ma5,
                        'MA20': curr_ma20,
                        'MA60': curr_ma60,
                        '거래량비율': volume_ratio,
                        'RSI': df.iloc[i]['RSI'] if 'RSI' in df.columns else np.nan,
                        'MACD_골든': macd_golden
                    })
        # MACD 골든 크로스 체크 (필터 여부와 관계없이 표시)
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            prev_macd = df.iloc[i - 1]['MACD']
            prev_signal = df.iloc[i - 1]['MACD_Signal']
            curr_macd = df.iloc[i]['MACD']
            curr_signal = df.iloc[i]['MACD_Signal']
            if pd.notna(prev_macd) and pd.notna(prev_signal) and pd.notna(curr_macd) and pd.notna(curr_signal):
                if prev_macd < prev_signal and curr_macd >= curr_signal:
                    df.iloc[i, df.columns.get_loc('MACD_골든크로스')] = True

    return df, events


def plot_data(df, code):
    """
    그래프를 그리는 함수 (RSI, MACD 포함)
    """
    # 모든 지표 계산
    df = calculate_ma(df, periods=[5, 20, 60])
    df = calculate_rsi(df, period=14)
    df = calculate_macd(df)
    df = calculate_volume_signal(df, period=20, multiplier=1.5)
    
    plt.rcParams['font.family'] = 'AppleGothic'  # macOS
    plt.rcParams['axes.unicode_minus'] = False
    
    # 4개 서브플롯 생성
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    ax1, ax2, ax3, ax4 = axes
    
    # 차트 1: 가격과 이동평균선
    ax1.plot(df['날짜'], df['종가'], label='종가', linewidth=2, color='black')
    ax1.plot(df['날짜'], df['MA5'], label='MA5', linewidth=1.5, alpha=0.7)
    ax1.plot(df['날짜'], df['MA20'], label='MA20', linewidth=1.5, alpha=0.7)
    ax1.plot(df['날짜'], df['MA60'], label='MA60', linewidth=1.5, alpha=0.7)
    
    # 골든 크로스 표시
    golden_crosses = df[df['골든크로스'] == True]
    if len(golden_crosses) > 0:
        ax1.scatter(golden_crosses['날짜'], golden_crosses['종가'], 
                   color='red', marker='^', s=200, label='골든 크로스', zorder=5)
    
    ax1.set_ylabel('가격 (원)', fontsize=11)
    ax1.set_title(f'종목 코드 {code} - 종합 기술 분석', fontsize=16, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 차트 2: RSI
    ax2.plot(df['날짜'], df['RSI'], label='RSI', linewidth=2, color='purple')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='과열 (70)')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='중립 (50)')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='과매도 (30)')
    ax2.fill_between(df['날짜'], 70, 100, alpha=0.2, color='red', label='과열 구간')
    ax2.fill_between(df['날짜'], 0, 30, alpha=0.2, color='green', label='과매도 구간')
    ax2.set_ylabel('RSI', fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 차트 3: MACD
    ax3.plot(df['날짜'], df['MACD'], label='MACD', linewidth=2, color='blue')
    ax3.plot(df['날짜'], df['MACD_Signal'], label='Signal', linewidth=2, color='red', linestyle='--')
    ax3.bar(df['날짜'], df['MACD_Histogram'], label='Histogram', alpha=0.3, color='gray')
    
    # MACD 골든 크로스 표시
    macd_crosses = df[df['MACD_골든크로스'] == True]
    if len(macd_crosses) > 0:
        ax3.scatter(macd_crosses['날짜'], macd_crosses['MACD'], 
                   color='green', marker='^', s=150, label='MACD 골든 크로스', zorder=5)
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_ylabel('MACD', fontsize=11)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 차트 4: 거래량
    colors = ['red' if x else 'blue' for x in df['거래량신호']]
    ax4.bar(df['날짜'], df['거래량'], alpha=0.6, color=colors, label='거래량')
    ax4.plot(df['날짜'], df['평균거래량'], label='평균 거래량 (20일)', linewidth=2, color='orange')
    ax4.set_ylabel('거래량', fontsize=11)
    ax4.set_xlabel('날짜', fontsize=12)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 파일 저장 대신 화면 표시만 수행
    print("\n💡 차트 이미지는 더 이상 파일로 저장되지 않습니다. 창에서만 확인하세요.")
    try:
        plt.show()
    finally:
    plt.close()


def save_to_csv(df, code):
    """
    데이터를 CSV로 저장하는 함수
    """
    print("CSV 저장 기능이 비활성화되어 실행되지 않습니다.")


def save_to_xlsx(df, code):
    """
    데이터를 XLSX로 저장하는 함수
    """
    print("XLSX 저장 기능이 비활성화되어 실행되지 않습니다.")


def ai_interpret_signals(ma5, ma20, macd, signal):
    """
    AI를 사용하여 주가 지표를 해석하는 함수
    ai_interpret_signals.py의 interpret_signals 함수를 사용
    """
    try:
        from ai_interpret_signals import interpret_signals
        return interpret_signals(ma5, ma20, macd, signal)
    except ImportError:
        return "❌ ai_interpret_signals 모듈을 찾을 수 없습니다."
    except Exception as e:
        return f"❌ AI 해석 중 오류 발생: {str(e)}"


def is_us_stock(code):
    """
    종목 코드가 미국 주식인지 판단
    숫자만 있으면 한국 주식, 알파벳이 있으면 미국 주식
    """
    # 숫자만 있는지 확인
    if code.isdigit():
        return False
    # 알파벳이 있으면 미국 주식
    if any(c.isalpha() for c in code):
        return True
    return False


def main_analyze(args):
    """단일 종목 분석 함수"""
    # 종목 코드로 한국/미국 자동 판단
    is_us = is_us_stock(args.code)
    
    if is_us:
        market_name = "미국 주식"
        print(f"=" * 60)
        print(f"{market_name} 골든 크로스 분석")
        print(f"종목 코드: {args.code}")
        print(f"=" * 60)
        
        # 미국 주식 데이터 수집
        df = fetch_stock_data_yahoo(args.code, period=args.period)
    else:
        market_name = "KOSDAQ"
        print(f"=" * 60)
        print(f"{market_name} 골든 크로스 분석")
        print(f"종목 코드: {args.code}")
        print(f"=" * 60)
        
        # 한국 주식 데이터 수집
        df = fetch_stock_data(args.code, args.pages)
    
    if df is None or len(df) == 0:
        print("데이터 수집에 실패했습니다.")
        return
    
    # 모든 지표 계산 (find_golden_cross 함수 내부에서 계산하지만, 전체 데이터에도 적용)
    # find_golden_cross는 골든 크로스 발생일만 반환하므로, 전체 데이터에도 지표를 계산해야 함
    df_full = df.copy()
    df_full = calculate_ma(df_full, periods=[5, 20, 60])
    df_full = calculate_rsi(df_full, period=14)
    df_full = calculate_macd(df_full)
    df_full = calculate_atr(df_full, period=14)
    df_full = calculate_volume_signal(df_full, period=20, multiplier=1.5)
    df_full = calculate_atr(df_full, period=14)

    df_full, golden_events = find_golden_cross(df_full, code=args.code, volume_multiplier=1.3)

    generate_analysis_report(df_full, args.code, is_us, golden_events)

    save_df = df_full.copy()
    
    # 그래프 그리기
    if args.plot:
        plot_data(df_full, args.code)
    
    # 파일 저장 비활성화 안내
    print("\n💡 CSV/XLSX 등 파일 산출 기능은 제거되었습니다. 필요한 경우 직접 DataFrame을 활용하세요.")
    
    # AI 해석 (선택적)
    if args.ai:
        print("\n" + "=" * 60)
        print("🤖 AI 기반 지표 해석")
        print("=" * 60)
        
        # 최신 데이터의 지표값 가져오기
        last_row = df_full.iloc[-1]
        
        if pd.notna(last_row['MA5']) and pd.notna(last_row['MA20']) and \
           pd.notna(last_row['MACD']) and pd.notna(last_row['MACD_Signal']):
            
            ma5 = last_row['MA5']
            ma20 = last_row['MA20']
            macd = last_row['MACD']
            signal = last_row['MACD_Signal']
            
            print(f"\n📊 현재 지표값:")
            currency_code = get_currency_code(is_us)
            print(f"   MA5: {format_price(ma5, currency_code)}")
            print(f"   MA20: {format_price(ma20, currency_code)}")
            print(f"   MACD: {macd:.2f}")
            print(f"   Signal: {signal:.2f}")
            print(f"   날짜: {last_row['날짜'].strftime('%Y-%m-%d')}")
            
            print(f"\n🤖 AI 해석 결과:")
            print("=" * 60)
            
            interpretation = ai_interpret_signals(ma5, ma20, macd, signal)
            print(interpretation)
            
            # API 키가 없으면 안내 메시지
            if "OPENAI_API_KEY" in interpretation or "환경변수" in interpretation:
                print("\n💡 안내: AI 해석을 사용하려면 OpenAI API 키가 필요합니다.")
                print("   설정 방법:")
                print("   export OPENAI_API_KEY=\"sk-your-key-here\"")
                print("   또는 터미널에서 위 명령어를 실행한 후 다시 실행하세요.")
        else:
            print("\n❌ 최신 데이터에 지표값이 없어 AI 해석을 수행할 수 없습니다.")
    
    print("\n분석이 완료되었습니다!")


def main():
    parser = argparse.ArgumentParser(description='주식 골든 크로스 구간 분석 (한국/미국)')
    parser.add_argument('--code', type=str, required=True, help='종목 코드 (한국: 108860, 미국: AAPL, TSLA 등)')
    parser.add_argument('--pages', type=int, default=20, help='크롤링할 페이지 수 (한국 주식만, 기본값: 20)')
    parser.add_argument('--period', type=str, default='1y', help='데이터 기간 (미국 주식만, 기본값: 1y) 옵션: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max')
    parser.add_argument('--plot', action='store_true', help='그래프 그리기')
    parser.add_argument('--ai', action='store_true', help='AI 해석 포함 (OpenAI API 키 필요)')
    
    args = parser.parse_args()
    main_analyze(args)


if __name__ == "__main__":
    main()
