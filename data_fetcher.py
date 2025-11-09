#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
주식 데이터 수집 모듈
한국/미국 주식의 기본 데이터와 외국인/기관 매매 데이터를 가져옵니다.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import re
import os
import sys
from io import StringIO
from typing import Optional, Tuple

# 기존 stock_screener.py의 함수 재사용
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from stock_screener import fetch_stock_data, fetch_stock_data_yahoo
    STOCK_SCREENER_AVAILABLE = True
except ImportError:
    STOCK_SCREENER_AVAILABLE = False

try:
    from stock_scanner import get_stock_name
    STOCK_SCANNER_AVAILABLE = True
except ImportError:
    STOCK_SCANNER_AVAILABLE = False

# yfinance for US stocks
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


def fix_encoding(name: str) -> str:
    """크롤링하다가 깨진 한글 종목명을 복원"""
    if not name:
        return name
    try:
        # latin1로 인코딩된 utf-8 텍스트 복원
        fixed = name.encode("latin1").decode("utf-8")
        return fixed
    except Exception:
        try:
            # euc-kr로 인코딩 시도
            fixed = name.encode("latin1").decode("euc-kr")
            return fixed
        except Exception:
            return name


def is_us_stock(code):
    """종목 코드가 미국 주식인지 판단"""
    if code.isdigit():
        return False
    if any(c.isalpha() for c in code):
        return True
    return False


def fetch_korean_stock_data(code, pages=5):
    """
    한국 주식 기본 데이터 수집 (기존 stock_screener.py 함수 재사용)
    """
    if STOCK_SCREENER_AVAILABLE:
        try:
            df = fetch_stock_data(code, pages=pages)
            if df is not None and len(df) > 0:
                # 현재가 가져오기
                current_price = df['종가'].iloc[-1]
                
                # 종목명 가져오기
                url = f"https://finance.naver.com/item/main.naver?code={code}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=5)
                response.encoding = 'euc-kr'
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 종목명 추출 (stock_scanner.py의 함수 재사용)
                if STOCK_SCANNER_AVAILABLE:
                    try:
                        stock_name = get_stock_name(code, is_us=False)
                    except:
                        stock_name = code
                else:
                    # Fallback: 직접 추출
                    stock_name = code
                    try:
                        content_main = response.content.decode('euc-kr', errors='ignore')
                        soup_main = BeautifulSoup(content_main, 'html.parser')
                        h2 = soup_main.find('h2', {'class': 'wrap_company'})
                        if h2:
                            a_tag = h2.find('a')
                            if a_tag:
                                name = a_tag.get_text(strip=True)
                                if name:
                                    stock_name = name
                    except:
                        pass
                
                return {
                    'price_data': df,
                    'current_price': current_price,
                    'name': stock_name
                }
        except Exception as e:
            print(f"⚠️ 기존 함수 사용 실패: {e}")
    
    # Fallback: 직접 크롤링
    return _fetch_korean_stock_data_direct(code, pages)


def _fetch_korean_stock_data_direct(code, pages=5):
    """
    한국 주식 기본 데이터 수집 (네이버 증권)
    
    Returns:
        dict: {
            'price_data': DataFrame (일봉 데이터),
            'current_price': float,
            'name': str
        }
    """
    try:
        # 종목명 가져오기
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'euc-kr'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 종목명 추출 (stock_scanner.py의 함수 재사용)
        if STOCK_SCANNER_AVAILABLE:
            try:
                stock_name = get_stock_name(code, is_us=False)
            except:
                stock_name = code
        else:
            # Fallback: 직접 추출
            stock_name = code
            try:
                content_name = response.content.decode('euc-kr', errors='ignore')
                soup_name = BeautifulSoup(content_name, 'html.parser')
                h2 = soup_name.find('h2', {'class': 'wrap_company'})
                if h2:
                    a_tag = h2.find('a')
                    if a_tag:
                        name = a_tag.get_text(strip=True)
                        if name:
                            stock_name = name
            except:
                pass
        
        # 현재가 가져오기
        current_price = None
        price_element = soup.find('p', {'class': 'no_today'})
        if not price_element:
            price_element = soup.find('div', {'class': 'no_today'})
        
        if price_element:
            blind_span = price_element.find('span', {'class': 'blind'})
            if blind_span:
                price_text = blind_span.text.strip()
                if re.match(r'^[\d,]+$', price_text):
                    current_price = float(price_text.replace(',', ''))
        
        # 일봉 데이터 수집
        base_url = "https://finance.naver.com/item/sise_day.naver"
        all_data = []
        
        for page in range(1, pages + 1):
            params = {'code': code, 'page': page}
            try:
                response = requests.get(base_url, params=params, headers=headers, timeout=10)
                response.encoding = 'euc-kr'
                
                # pandas read_html 시도
                try:
                    dfs = pd.read_html(StringIO(response.text), encoding='euc-kr')
                    if dfs and len(dfs) > 0:
                        df_page = dfs[0]
                        if not df_page.empty and len(df_page.columns) >= 7:
                            df_page = df_page.dropna(how='all')
                            df_page = df_page[df_page.iloc[:, 0].notna()]
                            
                            for idx, row in df_page.iterrows():
                                try:
                                    date_str = str(row.iloc[0]).strip()
                                    if not date_str or date_str == 'nan' or '.' not in date_str:
                                        continue
                                    
                                    close = int(str(row.iloc[1]).replace(',', '').replace(' ', ''))
                                    open_price = int(str(row.iloc[3]).replace(',', '').replace(' ', ''))
                                    high = int(str(row.iloc[4]).replace(',', '').replace(' ', ''))
                                    low = int(str(row.iloc[5]).replace(',', '').replace(' ', ''))
                                    volume = int(str(row.iloc[6]).replace(',', '').replace(' ', ''))
                                    
                                    all_data.append({
                                        '날짜': date_str,
                                        '종가': close,
                                        '시가': open_price,
                                        '고가': high,
                                        '저가': low,
                                        '거래량': volume
                                    })
                                except (ValueError, IndexError, AttributeError):
                                    continue
                except:
                    # BeautifulSoup으로 파싱 (fallback)
                    soup_page = BeautifulSoup(response.text, 'html.parser')
                    table = soup_page.find('table', {'class': 'type_2'})
                    if not table:
                        table = soup_page.find('table', {'class': 'type2'})
                    if not table:
                        table = soup_page.find('table', {'class': 'tb_type1'})
                    
                    if table:
                        rows = table.find_all('tr')[2:]  # 헤더 제외
                        for row in rows:
                            cols = row.find_all(['td', 'th'])
                            if len(cols) < 7:
                                continue
                            try:
                                date_str = cols[0].text.strip()
                                if not date_str or '.' not in date_str:
                                    continue
                                
                                close_str = cols[1].text.strip().replace(',', '').replace(' ', '')
                                if not close_str or not close_str.isdigit():
                                    continue
                                
                                close = int(close_str)
                                open_price = int(cols[3].text.strip().replace(',', '').replace(' ', ''))
                                high = int(cols[4].text.strip().replace(',', '').replace(' ', ''))
                                low = int(cols[5].text.strip().replace(',', '').replace(' ', ''))
                                volume = int(cols[6].text.strip().replace(',', '').replace(' ', ''))
                                
                                all_data.append({
                                    '날짜': date_str,
                                    '종가': close,
                                    '시가': open_price,
                                    '고가': high,
                                    '저가': low,
                                    '거래량': volume
                                })
                            except (ValueError, IndexError, AttributeError):
                                continue
                
                time.sleep(0.3)
            except Exception as e:
                if page == 1:
                    print(f"⚠️ 페이지 {page} 수집 실패: {e}")
                break
        
        if not all_data:
            print(f"⚠️ 일봉 데이터를 가져올 수 없습니다.")
            # 최소한 현재가라도 반환
            if current_price:
                return {
                    'price_data': None,
                    'current_price': current_price,
                    'name': stock_name
                }
            return None
        
        df = pd.DataFrame(all_data)
        df['날짜'] = pd.to_datetime(df['날짜'], format='%Y.%m.%d', errors='coerce')
        df = df.dropna(subset=['날짜'])
        df = df.drop_duplicates(subset=['날짜'], keep='first')
        df = df.sort_values('날짜').reset_index(drop=True)
        
        if len(df) == 0:
            if current_price:
                return {
                    'price_data': None,
                    'current_price': current_price,
                    'name': stock_name
                }
            return None
        
        # 현재가가 없으면 최신 종가 사용
        if current_price is None:
            current_price = df['종가'].iloc[-1]
        
        return {
            'price_data': df,
            'current_price': current_price,
            'name': stock_name
        }
    except Exception as e:
        print(f"❌ 한국 주식 데이터 수집 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def fetch_us_stock_data(ticker):
    """
    미국 주식 기본 데이터 수집 (기존 stock_screener.py 함수 재사용)
    """
    if STOCK_SCREENER_AVAILABLE:
        try:
            df = fetch_stock_data_yahoo(ticker, period="3mo")
            if df is not None and len(df) > 0:
                current_price = df['종가'].iloc[-1]
                
                # 종목명 가져오기
                try:
                    ticker_obj = yf.Ticker(ticker)
                    info = ticker_obj.info
                    stock_name = info.get('longName') or info.get('shortName') or ticker
                except:
                    stock_name = ticker
                
                return {
                    'price_data': df,
                    'current_price': current_price,
                    'name': stock_name
                }
        except Exception as e:
            print(f"⚠️ 기존 함수 사용 실패: {e}")
    
    # Fallback: 직접 yfinance 사용
    if not YFINANCE_AVAILABLE:
        print("❌ yfinance 패키지가 필요합니다: pip install yfinance")
        return None
    
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period="3mo")
        
        if hist.empty:
            return None
        
        df = hist.reset_index()
        df = df.rename(columns={
            'Date': '날짜',
            'Open': '시가',
            'High': '고가',
            'Low': '저가',
            'Close': '종가',
            'Volume': '거래량'
        })
        
        df = df[['날짜', '시가', '고가', '저가', '종가', '거래량']].copy()
        df['종가'] = df['종가'].astype(float)
        
        # 현재가 가져오기
        info = ticker_obj.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or df['종가'].iloc[-1]
        stock_name = info.get('longName') or info.get('shortName') or ticker
        
        return {
            'price_data': df,
            'current_price': float(current_price),
            'name': stock_name
        }
    except Exception as e:
        print(f"❌ 미국 주식 데이터 수집 실패: {e}")
        return None


def _select_yf_interval(minutes: int) -> Tuple[str, bool]:
    supported = {
        1: "1m",
        2: "2m",
        5: "5m",
        15: "15m",
        30: "30m",
        60: "60m",
        90: "90m",
        120: "2h",
    }
    if minutes in supported:
        return supported[minutes], False
    return "1m", True


def _download_intraday_yf(ticker: str, interval: str, period: str):
    try:
        df = yf.download(
            tickers=ticker,
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=False,
        )
        if df is None or df.empty:
            return None
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = df[["open", "high", "low", "close", "volume"]]
        df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return None


def fetch_intraday_data(code: str, timeframe_min: int = 5, lookback_minutes: int = 360) -> Optional[pd.DataFrame]:
    """
    분봉 데이터를 가져옵니다. (가능한 경우 yfinance 사용)

    Returns:
        DataFrame: datetime index, columns [open, high, low, close, volume]
    """
    if not YFINANCE_AVAILABLE:
        return None
    timeframe_min = max(1, int(timeframe_min))

    interval, needs_resample = _select_yf_interval(timeframe_min)
    period = "1d" if timeframe_min <= 30 else "5d"

    tickers = []
    if is_us_stock(code):
        tickers.append(code)
    else:
        numeric_code = str(code).zfill(6)
        tickers.extend([f"{numeric_code}.KS", f"{numeric_code}.KQ"])

    df: Optional[pd.DataFrame] = None
    for ticker in tickers:
        df = _download_intraday_yf(ticker, interval, period)
        if df is not None:
            break

    if df is None or df.empty:
        return None

    if needs_resample:
        resampled = (
            df.resample(f"{timeframe_min}T")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna(subset=["open", "high", "low", "close"])
        )
        df = resampled

    if lookback_minutes and lookback_minutes > 0 and not df.empty:
        cutoff = df.index.max() - pd.Timedelta(minutes=lookback_minutes)
        df = df[df.index >= cutoff]

    return df


def _fetch_investor_data_naver(code):
    """
    네이버 증권에서 외국인/기관 매매 데이터 수집
    """
    try:
        # 종목코드 6자리 확인
        code = str(code).zfill(6)
        
        url = f"https://finance.naver.com/item/frgn.naver?code={code}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': f'https://finance.naver.com/item/main.naver?code={code}'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        # 🔴 핵심: 네이버는 euc-kr 인코딩
        response.encoding = 'euc-kr'
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 방법 1: BeautifulSoup으로 직접 파싱 (가장 안정적)
        # 외국인/기관 관련 테이블 찾기 (여러 table.type2 중에서)
        table = None
        for t in soup.find_all('table', {'class': 'type2'}):
            text = t.get_text()
            if '외국인' in text and '기관' in text and '순매매' in text:
                table = t
                break
        
        # 못 찾으면 일반 type2 테이블 시도
        if not table:
            table = soup.select_one("table.type2")
        if not table:
            table = soup.find('table', {'class': 'type_2'})
        
        if table:
            rows = table.select("tr")
            if len(rows) > 3:
                data = []
                for row in rows[3:]:  # 헤더 3줄 건너뛰기
                    cols = row.select("td")
                    if len(cols) < 7:  # 최소 7개 컬럼 필요
                        continue
                    
                    try:
                        # 컬럼 구조 확인: 날짜(0), 종가(1), 전일비(2), 등락률(3), 거래량(4), 기관순매매량(5), 외국인순매매량(6), ...
                        date_text = cols[0].get_text(strip=True)
                        if not date_text or '.' not in date_text or len(date_text) < 8:
                            continue
                        
                        # 기관 순매매량 (인덱스 5)
                        institution_text = cols[5].get_text(strip=True).replace(',', '').replace(' ', '').replace('원', '').replace('(', '-').replace(')', '')
                        # 외국인 순매매량 (인덱스 6)
                        foreign_text = cols[6].get_text(strip=True).replace(',', '').replace(' ', '').replace('원', '').replace('(', '-').replace(')', '')
                        # 거래량 (인덱스 4)
                        volume_text = cols[4].get_text(strip=True).replace(',', '').replace(' ', '')
                        
                        # 날짜 파싱
                        date_obj = pd.to_datetime(date_text, format='%Y.%m.%d', errors='coerce')
                        if pd.isna(date_obj):
                            continue
                        
                        # 숫자 변환
                        try:
                            foreign_buy = int(float(foreign_text)) if foreign_text and foreign_text != 'nan' else 0
                            institution_buy = int(float(institution_text)) if institution_text and institution_text != 'nan' else 0
                            volume = int(float(volume_text)) if volume_text and volume_text != 'nan' else 0
                        except (ValueError, TypeError):
                            continue
                        
                        # 개인 순매수 = -(외국인 + 기관)
                        individual_buy = -(foreign_buy + institution_buy)
                        
                        data.append({
                            '날짜': date_obj,
                            '외국인_순매수': foreign_buy,
                            '기관_순매수': institution_buy,
                            '개인_순매수': individual_buy,
                            '거래량': volume
                        })
                    except (ValueError, IndexError, AttributeError) as e:
                        continue
                
                if data:
                    df = pd.DataFrame(data)
                    df = df.dropna(subset=['날짜']).sort_values('날짜').reset_index(drop=True)
                    if len(df) > 0:
                        # 최소 1개 이상의 0이 아닌 값이 있는지 확인
                        has_valid_data = ((df['외국인_순매수'] != 0) | (df['기관_순매수'] != 0)).any()
                        if has_valid_data:
                            return df
        
        # 방법 2: pandas read_html로 테이블 읽기 시도 (Fallback)
        try:
            # StringIO로 감싸서 사용 (FutureWarning 방지)
            try:
                dfs = pd.read_html(StringIO(response.text), encoding='euc-kr')
            except:
                # BytesIO로 시도
                from io import BytesIO
                dfs = pd.read_html(BytesIO(response.content), encoding='euc-kr')
            
            if dfs and len(dfs) > 0:
                # 외국인/기관 매매 테이블 찾기 (더 확실하게)
                target_table = None
                for df_table in dfs:
                    # MultiIndex 컬럼 처리
                    if isinstance(df_table.columns, pd.MultiIndex):
                        col_names = [str(col) for col in df_table.columns]
                    else:
                        col_names = [str(col) for col in df_table.columns]
                    
                    # 외국인/기관 데이터가 있는 테이블 찾기
                    col_str = ' '.join(col_names).lower()
                    if ('외국인' in col_str and '기관' in col_str) and ('순매매' in col_str or '순매수' in col_str):
                        target_table = df_table
                        break
                
                if target_table is not None:
                    # 데이터 정리
                    target_table = target_table.dropna(how='all')
                    
                    # 첫 행이 헤더인지 확인하고 제거
                    first_row_first_col = str(target_table.iloc[0, 0]).strip() if len(target_table) > 0 else ''
                    if first_row_first_col in ['날짜', '일자', 'NaN', 'nan', ''] or '.' not in first_row_first_col:
                        target_table = target_table.iloc[1:].reset_index(drop=True)
                    
                    # 빈 행 제거
                    target_table = target_table[target_table.iloc[:, 0].notna()].reset_index(drop=True)
                    
                    data = []
                    for idx, row in target_table.iterrows():
                        try:
                            # 날짜 찾기 (첫 번째 컬럼)
                            date_val = row.iloc[0]
                            date_str = str(date_val).strip()
                            
                            # NaN이거나 날짜 형식이 아니면 스킵
                            if not date_str or date_str == 'nan' or date_str == 'NaN' or '.' not in date_str or len(date_str) < 8:
                                continue
                            
                            # MultiIndex 컬럼에서 직접 접근
                            volume = 0
                            foreign_buy = 0
                            institution_buy = 0
                            
                            if isinstance(target_table.columns, pd.MultiIndex):
                                # MultiIndex 컬럼 구조 확인
                                # 일반적으로: 날짜(0), 종가(1), 전일비(2), 등락률(3), 거래량(4), 기관 순매매량(5), 외국인 순매매량(6)
                                try:
                                    # 거래량 (인덱스 4)
                                    if len(row) > 4:
                                        vol_val = row.iloc[4]
                                        if pd.notna(vol_val):
                                            vol_str = str(vol_val).strip().replace(',', '').replace(' ', '').replace('원', '')
                                            if vol_str and vol_str != 'nan' and vol_str != 'NaN':
                                                try:
                                                    volume = int(float(vol_str))
                                                except:
                                                    pass
                                    
                                    # 기관 순매매량 (인덱스 5)
                                    if len(row) > 5:
                                        inst_val = row.iloc[5]
                                        if pd.notna(inst_val):
                                            inst_str = str(inst_val).strip().replace(',', '').replace(' ', '').replace('원', '').replace('(', '-').replace(')', '')
                                            if inst_str and inst_str != 'nan' and inst_str != 'NaN':
                                                try:
                                                    institution_buy = int(float(inst_str))
                                                except:
                                                    pass
                                    
                                    # 외국인 순매매량 (인덱스 6)
                                    if len(row) > 6:
                                        for_val = row.iloc[6]
                                        if pd.notna(for_val):
                                            for_str = str(for_val).strip().replace(',', '').replace(' ', '').replace('원', '').replace('(', '-').replace(')', '')
                                            if for_str and for_str != 'nan' and for_str != 'NaN':
                                                try:
                                                    foreign_buy = int(float(for_str))
                                                except:
                                                    pass
                                except (ValueError, IndexError, AttributeError, TypeError) as e:
                                    continue
                            else:
                                # 일반 컬럼인 경우 컬럼명으로 찾기
                                for i, col_name in enumerate(col_names):
                                    if i >= len(row):
                                        break
                                    if pd.isna(row.iloc[i]):
                                        continue
                                    
                                    col_lower = str(col_name).lower()
                                    val_str = str(row.iloc[i]).strip().replace(',', '').replace(' ', '').replace('원', '').replace('(', '-').replace(')', '')
                                    
                                    if '거래량' in col_lower and val_str and val_str != 'nan' and val_str != 'NaN':
                                        try:
                                            volume = int(float(val_str))
                                        except:
                                            pass
                                    elif '외국인' in col_lower and ('순매매' in col_lower or '순매수' in col_lower) and val_str and val_str != 'nan' and val_str != 'NaN':
                                        try:
                                            foreign_buy = int(float(val_str))
                                        except:
                                            pass
                                    elif '기관' in col_lower and ('순매매' in col_lower or '순매수' in col_lower) and val_str and val_str != 'nan' and val_str != 'NaN':
                                        try:
                                            institution_buy = int(float(val_str))
                                        except:
                                            pass
                            
                            # 날짜 파싱
                            try:
                                date_obj = pd.to_datetime(date_str, format='%Y.%m.%d', errors='coerce')
                                if pd.isna(date_obj):
                                    continue
                            except:
                                continue
                            
                            # 개인 순매수 = -(외국인 + 기관)
                            individual_buy = -(foreign_buy + institution_buy)
                            
                            data.append({
                                '날짜': date_obj,
                                '외국인_순매수': foreign_buy,
                                '기관_순매수': institution_buy,
                                '개인_순매수': individual_buy,
                                '거래량': volume
                            })
                        except (ValueError, IndexError, AttributeError, TypeError) as e:
                            continue
                    
                    if data:
                        df = pd.DataFrame(data)
                        df = df.dropna(subset=['날짜']).sort_values('날짜').reset_index(drop=True)
                        if len(df) > 0:
                            # 최소 1개 이상의 0이 아닌 값이 있는지 확인
                            has_valid_data = ((df['외국인_순매수'] != 0) | (df['기관_순매수'] != 0)).any()
                            if has_valid_data:
                                return df
        except Exception as e:
            # 디버깅용
            print(f"⚠️ pandas read_html 오류: {e}")
            pass
        
        # 모든 방법 실패
        return None
        
    except Exception as e:
        # 디버깅용 (필요시 주석 해제)
        # print(f"⚠️ 네이버 외국인/기관 데이터 수집 오류: {e}")
        return None


def _fetch_investor_data_daum(code):
    """
    다음 증권에서 외국인/기관 매매 데이터 수집 (Fallback)
    """
    try:
        # 다음 증권 URL 시도 (여러 패턴)
        urls = [
            f"https://finance.daum.net/item/investor.daum?code={code}",
            f"https://finance.daum.net/quotes/A{code}/investor",
            f"https://finance.daum.net/item/main.daum?code={code}&tab=investor"
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        }
        
        soup = None
        for url in urls:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    response.encoding = 'utf-8'
                    soup = BeautifulSoup(response.text, 'html.parser')
                    break
            except:
                continue
        
        if soup is None:
            return None
        
        # 테이블 찾기 (여러 패턴 시도)
        table = None
        table_patterns = [
            {'class': 'gTable clr'},
            {'class': 'gTable'},
            {'class': 'table'},
            {'id': 'investorTable'}
        ]
        
        for pattern in table_patterns:
            table = soup.find('table', pattern)
            if table:
                break
        
        # 모든 테이블 검색
        if not table:
            tables = soup.find_all('table')
            for t in tables:
                # 외국인/기관 관련 텍스트가 있는 테이블 찾기
                table_text = t.get_text().lower()
                if '외국인' in table_text and '기관' in table_text:
                    table = t
                    break
        
        if not table:
            return None
        
        rows = table.find_all('tr')[1:]  # 헤더 제외
        data = []
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 5:
                continue
            
            try:
                date_str = cols[0].text.strip()
                if not date_str or '.' not in date_str:
                    continue
                
                # 다음 증권 구조: 날짜, 외국인, 기관, 개인, 거래량
                foreign_buy_str = cols[1].text.strip().replace(',', '').replace(' ', '').replace('원', '').replace('(', '-').replace(')', '')
                institution_buy_str = cols[2].text.strip().replace(',', '').replace(' ', '').replace('원', '').replace('(', '-').replace(')', '')
                individual_buy_str = cols[3].text.strip().replace(',', '').replace(' ', '').replace('원', '').replace('(', '-').replace(')', '') if len(cols) > 3 else '0'
                volume_str = cols[4].text.strip().replace(',', '').replace(' ', '') if len(cols) > 4 else '0'
                
                foreign_buy = int(float(foreign_buy_str)) if foreign_buy_str and foreign_buy_str != 'nan' else 0
                institution_buy = int(float(institution_buy_str)) if institution_buy_str and institution_buy_str != 'nan' else 0
                individual_buy = int(float(individual_buy_str)) if individual_buy_str and individual_buy_str != 'nan' else 0
                volume = int(float(volume_str)) if volume_str and volume_str != 'nan' else 0
                
                date_obj = pd.to_datetime(date_str, format='%Y.%m.%d', errors='coerce')
                if pd.isna(date_obj):
                    continue
                
                data.append({
                    '날짜': date_obj,
                    '외국인_순매수': foreign_buy,
                    '기관_순매수': institution_buy,
                    '개인_순매수': individual_buy,
                    '거래량': volume
                })
            except (ValueError, IndexError, AttributeError) as e:
                continue
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        df = df.dropna(subset=['날짜']).sort_values('날짜').reset_index(drop=True)
        
        if len(df) == 0:
            return None
        
        # 최소 1개 이상의 0이 아닌 값이 있는지 확인
        has_valid_data = ((df['외국인_순매수'] != 0) | (df['기관_순매수'] != 0)).any()
        if has_valid_data:
            return df
        
        return None
        
    except Exception as e:
        return None


def fetch_investor_trading_data(code):
    """
    한국 주식의 외국인/기관 매매 데이터 수집
    네이버 증권 → 다음 증권 순서로 시도
    
    Returns:
        DataFrame: {
            '날짜': datetime,
            '외국인_순매수': int,
            '기관_순매수': int,
            '개인_순매수': int,
            '거래량': int
        }
    """
    # 1차: 네이버 증권 시도
    result = _fetch_investor_data_naver(code)
    if result is not None and len(result) > 0:
        has_valid_data = ((result['외국인_순매수'] != 0) | (result['기관_순매수'] != 0)).any()
        if has_valid_data:
            return result
    
    # 2차: 다음 증권 시도
    print(f"   ⚠️ 네이버 증권 실패, 다음 증권으로 재시도 중...")
    result = _fetch_investor_data_daum(code)
    if result is not None and len(result) > 0:
        has_valid_data = ((result['외국인_순매수'] != 0) | (result['기관_순매수'] != 0)).any()
        if has_valid_data:
            print(f"   ✅ 다음 증권에서 데이터 수집 성공")
            return result
    
    # 모든 소스 실패
    print(f"\n⚠️ 종목 {code}: 외국인/기관 매매 데이터를 찾을 수 없습니다.")
    print(f"   시도한 경로:")
    print(f"   1. 네이버 증권: https://finance.naver.com/item/frgn.naver?code={code}")
    print(f"   2. 다음 증권: (구조 변경으로 현재 미지원)")
    print(f"   💡 해결 방법:")
    print(f"   - 네이버 증권 페이지에서 직접 확인: https://finance.naver.com/item/frgn.naver?code={code}")
    print(f"   - 해당 종목이 외국인/기관 매매 데이터를 제공하는지 확인")
    print(f"   - 네트워크 연결 상태 확인")
    return None


def fetch_technical_indicators(df):
    """
    기술적 지표 계산 (MA, RSI 등)
    
    Args:
        df: 가격 데이터프레임
    
    Returns:
        DataFrame: 기술적 지표가 추가된 데이터프레임
    """
    if df is None or len(df) == 0:
        return df
    
    df = df.copy()
    
    # 이동평균선
    df['MA5'] = df['종가'].rolling(window=5).mean()
    df['MA20'] = df['종가'].rolling(window=20).mean()
    df['MA60'] = df['종가'].rolling(window=60).mean()
    
    # RSI 계산
    delta = df['종가'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

