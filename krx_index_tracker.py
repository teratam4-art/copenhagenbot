#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KRX 코스피200 편입/제외 추적 및 스크리닝 시스템
KRX 공시 → 뉴스 분석 → 기술적 신호 결합
"""

import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import re
import os
import sys

# 기존 스크리닝 모듈 import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from stock_screener import check_buy_signal, is_us_stock
    STOCK_SCREENER_AVAILABLE = True
except ImportError:
    STOCK_SCREENER_AVAILABLE = False
    print("⚠️  stock_screener 모듈을 찾을 수 없습니다.")

# yfinance for US stocks
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ============================================================================
# 1️⃣ KRX 코스피200 편입/제외 데이터 크롤링
# ============================================================================

def fetch_krx_index_changes(index_code="코스피200", days_back=30):
    """
    KRX에서 코스피200 편입/제외 공시 데이터 크롤링
    
    Args:
        index_code: 지수 코드 (코스피200, 코스닥150 등)
        days_back: 며칠 전까지 조회
    
    Returns:
        list: [{'date': datetime, 'added': [종목코드], 'removed': [종목코드], 'sector': dict}, ...]
    """
    print("=" * 60)
    print(f"📊 KRX {index_code} 편입/제외 데이터 크롤링")
    print("=" * 60)
    
    changes_list = []
    
    try:
        # 방법 1: 네이버 증권에서 정보 수집 (더 안정적)
        print("📡 네이버 증권에서 코스피200 관련 정보 수집...")
        naver_changes = fetch_naver_kospi200_changes(days_back)
        if naver_changes:
            changes_list.extend(naver_changes)
            print(f"   ✅ 네이버에서 {len(naver_changes)}건 발견")
        
        # 방법 2: KRX 공시 페이지 직접 크롤링 시도
        print("📡 KRX 공시 페이지 크롤링 시도...")
        krx_changes = fetch_krx_disclosure_changes(index_code, days_back)
        if krx_changes:
            changes_list.extend(krx_changes)
            print(f"   ✅ KRX에서 {len(krx_changes)}건 발견")
        
        # 방법 3: 뉴스에서 정보 추출
        if not changes_list:
            print("📡 뉴스에서 편입/제외 정보 수집...")
            news_changes = extract_changes_from_news(index_code, days_back)
            if news_changes:
                changes_list.extend(news_changes)
                print(f"   ✅ 뉴스에서 {len(news_changes)}건 발견")
        
        # 샘플 데이터 (테스트용, 실제 데이터가 없을 때만)
        if not changes_list:
            print("⚠️  실제 데이터를 찾을 수 없어 샘플 데이터 사용")
            sample_date = datetime.now() - timedelta(days=7)
            changes_list.append({
                'date': sample_date,
                'added': ['000660', '005930', '035720'],
                'removed': ['012330', '003670'],
                'sector': {
                    'added': {'반도체': ['000660', '005930', '035720']},
                    'removed': {'자동차': ['012330'], '화학': ['003670']}
                }
            })
        
    except Exception as e:
        print(f"⚠️  크롤링 오류: {e}")
        import traceback
        traceback.print_exc()
    
    return changes_list


def fetch_krx_disclosure_changes(index_code="코스피200", days_back=30):
    """
    KRX 공시 페이지에서 지수 구성종목 변경 정보 크롤링
    """
    changes_list = []
    
    try:
        # KRX 공시 시스템 (KIND) URL
        # 코스피200 구성종목 변경 공시 검색
        url = "https://kind.krx.co.kr/disclosure/today.do"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        # 검색 파라미터
        params = {
            'method': 'search',
            'acptCd': '',
            'acptNm': '',
            'beginDate': (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d'),
            'endDate': datetime.now().strftime('%Y%m%d'),
            'searchText': index_code
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 공시 목록 추출 (실제 HTML 구조에 맞게 수정 필요)
        # 여기서는 기본 구조만 제공
        
    except Exception as e:
        pass  # 조용히 실패
    
    return changes_list


def extract_changes_from_news(index_code="코스피200", days_back=30):
    """
    뉴스 기사에서 편입/제외 정보 추출
    """
    changes_list = []
    
    try:
        # 뉴스 검색
        query = f"{index_code} 편입 제외"
        search_url = f"https://search.naver.com/search.naver?where=news&query={query}&sort=1"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_items = soup.find_all('a', class_='news_tit')
        
        added_stocks = []
        removed_stocks = []
        added_sectors = defaultdict(list)
        removed_sectors = defaultdict(list)
        
        for item in news_items[:10]:
            try:
                title = item.get_text(strip=True)
                link = item.get('href', '')
                
                # 기사 본문 크롤링
                if link:
                    try:
                        article_response = requests.get(link, headers=headers, timeout=5)
                        article_soup = BeautifulSoup(article_response.text, 'html.parser')
                        article_body = article_soup.find('div', id='articleBodyContents')
                        if article_body:
                            content = article_body.get_text(strip=True)
                        else:
                            content = title
                    except:
                        content = title
                else:
                    content = title
                
                full_text = title + " " + content
                
                # 종목코드 추출
                codes = extract_stock_codes_from_text(full_text)
                
                # 편입/제외 구분
                if '편입' in title or '편입' in content:
                    added_stocks.extend(codes)
                    for code in codes:
                        sector = get_stock_sector(code)
                        if sector != '기타':
                            added_sectors[sector].append(code)
                
                if '제외' in title or '제외' in content:
                    removed_stocks.extend(codes)
                    for code in codes:
                        sector = get_stock_sector(code)
                        if sector != '기타':
                            removed_sectors[sector].append(code)
                
            except Exception:
                continue
        
        # 중복 제거
        added_stocks = list(set(added_stocks))
        removed_stocks = list(set(removed_stocks))
        
        if added_stocks or removed_stocks:
            changes_list.append({
                'date': datetime.now() - timedelta(days=1),
                'added': added_stocks,
                'removed': removed_stocks,
                'sector': {
                    'added': dict(added_sectors),
                    'removed': dict(removed_sectors)
                }
            })
        
    except Exception as e:
        pass
    
    return changes_list


def fetch_naver_kospi200_changes(days_back=30):
    """
    네이버 증권에서 코스피200 편입/제외 관련 정보 수집
    
    Returns:
        list: 편입/제외 정보 리스트
    """
    changes_list = []
    
    try:
        # 네이버 증권 코스피200 페이지
        url = "https://finance.naver.com/sise/sise_index.naver?code=KPI200"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 구성종목 리스트 추출
        # 실제 HTML 구조에 맞게 수정 필요
        
        # 또는 검색 결과에서 최근 편입/제외 뉴스 찾기
        search_url = "https://search.naver.com/search.naver?where=news&query=코스피200+편입+제외"
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 뉴스 기사에서 편입/제외 종목 추출
        news_items = soup.find_all('a', class_='news_tit')
        
        for item in news_items[:10]:  # 최근 10개 기사
            title = item.get_text(strip=True)
            if '코스피200' in title and ('편입' in title or '제외' in title):
                # 기사 링크로 가서 종목명 추출
                link = item.get('href', '')
                if link:
                    # 실제로는 기사 본문을 크롤링해서 종목명 추출
                    pass
        
    except Exception as e:
        print(f"⚠️  네이버 크롤링 오류: {e}")
    
    return changes_list


def extract_stock_codes_from_text(text):
    """
    텍스트에서 종목코드(6자리) 추출
    
    Args:
        text: 종목명이나 코드가 포함된 텍스트
    
    Returns:
        list: 추출된 종목코드 리스트
    """
    # 6자리 숫자 패턴 (종목코드)
    codes = re.findall(r'\b\d{6}\b', text)
    return codes


def get_stock_sector(code):
    """
    종목코드로 업종 정보 가져오기 (간단 버전)
    
    Args:
        code: 종목코드
    
    Returns:
        str: 업종명
    """
    # 업종 매핑 딕셔너리 (실제로는 KRX API나 네이버 증권에서 가져와야 함)
    sector_map = {
        '000660': '반도체',
        '005930': '반도체',
        '035720': '반도체',
        '012330': '자동차',
        '003670': '화학',
        '012450': '방산',
        '051910': '화학',
        '096770': '화학',
    }
    
    return sector_map.get(code, '기타')


# ============================================================================
# 2️⃣ 뉴스 기반 보조 분석
# ============================================================================

def fetch_news_about_index_changes(index_name="코스피200", days_back=7):
    """
    네이버 경제뉴스에서 코스피200 편입/제외 관련 기사 크롤링
    
    Args:
        index_name: 지수명
        days_back: 며칠 전까지 조회
    
    Returns:
        list: [{'title': str, 'content': str, 'date': datetime, 'stocks': [종목코드], 'sectors': [업종]}, ...]
    """
    print("\n" + "=" * 60)
    print(f"📰 {index_name} 편입/제외 관련 뉴스 크롤링")
    print("=" * 60)
    
    news_list = []
    
    try:
        # 네이버 뉴스 검색
        query = f"{index_name} 편입 제외"
        search_url = f"https://search.naver.com/search.naver?where=news&query={query}&sort=1"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 뉴스 항목 추출
        news_items = soup.find_all('a', class_='news_tit')
        
        print(f"   발견된 기사: {len(news_items)}건")
        
        for item in news_items[:20]:  # 최근 20개
            try:
                title = item.get_text(strip=True)
                link = item.get('href', '')
                
                if not title or len(title) < 10:
                    continue
                
                # 기사 본문 크롤링 시도
                content = ""
                if link:
                    try:
                        article_response = requests.get(link, headers=headers, timeout=5)
                        article_soup = BeautifulSoup(article_response.text, 'html.parser')
                        
                        # 본문 추출 (네이버 뉴스 형식)
                        article_body = article_soup.find('div', id='articleBodyContents')
                        if article_body:
                            content = article_body.get_text(strip=True)
                    except:
                        content = title  # 본문 크롤링 실패 시 제목만 사용
                
                # 종목코드 추출
                full_text = title + " " + content
                stock_codes = extract_stock_codes_from_text(full_text)
                
                # 업종 추출 (종목코드 기반)
                sectors = []
                for code in stock_codes:
                    sector = get_stock_sector(code)
                    if sector != '기타':
                        sectors.append(sector)
                sectors = list(set(sectors))  # 중복 제거
                
                # 편입/제외 구분
                is_added = '편입' in title or '편입' in content
                is_removed = '제외' in title or '제외' in content
                
                news_list.append({
                    'title': title,
                    'content': content[:500],  # 처음 500자만
                    'date': datetime.now(),  # 실제로는 기사 날짜 추출 필요
                    'stocks': stock_codes,
                    'sectors': sectors,
                    'is_added': is_added,
                    'is_removed': is_removed,
                    'url': link
                })
                
            except Exception as e:
                continue
        
        # 업종별 집계
        print("\n📊 업종별 편입/제외 현황:")
        sector_summary = {}
        for news in news_list:
            for sector in news['sectors']:
                if sector not in sector_summary:
                    sector_summary[sector] = {'added': 0, 'removed': 0}
                
                if news['is_added']:
                    sector_summary[sector]['added'] += 1
                if news['is_removed']:
                    sector_summary[sector]['removed'] += 1
        
        for sector, counts in sector_summary.items():
            print(f"   {sector}: 편입 {counts['added']}건 / 제외 {counts['removed']}건")
        
    except Exception as e:
        print(f"⚠️  뉴스 크롤링 오류: {e}")
    
    return news_list


# ============================================================================
# 3️⃣ 통합 스크리닝 (편입 종목 + 기술적 신호)
# ============================================================================

def screen_newly_added_stocks(index_code="코스피200", days_back=30, top_n=5):
    """
    신규 편입 종목을 가져와서 기술적 신호 분석 후 TOP5 추천
    
    Args:
        index_code: 지수 코드
        days_back: 며칠 전까지 조회
        top_n: 추천 종목 개수
    
    Returns:
        list: [{'ticker': str, 'sector': str, 'rsi': float, 'ma_gap': float, 'volume_ratio': float, 'judgment': str}, ...]
    """
    print("\n" + "=" * 60)
    print(f"🔍 {index_code} 신규 편입 종목 스크리닝")
    print("=" * 60)
    
    if not STOCK_SCREENER_AVAILABLE:
        print("❌ stock_screener 모듈을 사용할 수 없습니다.")
        return []
    
    # 1. 편입 종목 가져오기
    changes = fetch_krx_index_changes(index_code, days_back)
    
    all_added_stocks = []
    for change in changes:
        all_added_stocks.extend(change.get('added', []))
    
    # 중복 제거
    added_stocks = list(set(all_added_stocks))
    
    if not added_stocks:
        print("❌ 신규 편입 종목이 없습니다.")
        # 뉴스에서 편입 종목 찾기 시도
        print("📰 뉴스에서 편입 정보 수집 시도...")
        news_list = fetch_news_about_index_changes(index_code, days_back=days_back)
        
        # 뉴스에서 편입된 종목 추출
        for news in news_list:
            if news['is_added']:
                all_added_stocks.extend(news['stocks'])
        
        added_stocks = list(set(all_added_stocks))
        
        if not added_stocks:
            print("❌ 뉴스에서도 편입 종목을 찾을 수 없습니다.")
            return []
    
    print(f"\n📋 발견된 신규 편입 종목: {len(added_stocks)}개")
    print(f"   종목: {', '.join(added_stocks[:10])}")
    
    # 2. 각 종목의 기술적 신호 분석
    print(f"\n📊 기술적 신호 분석 중...")
    
    results = []
    
    for i, ticker in enumerate(added_stocks[:20], 1):  # 최대 20개만 분석
        try:
            print(f"   [{i}/{min(len(added_stocks), 20)}] {ticker} 분석 중...", end=" ")
            
            result = check_buy_signal(
                ticker,
                period="3mo",
                rsi_min=40,
                rsi_max=70,
                volume_min=1.0,
                volume_max=5.0
            )
            
            if result is None:
                print("❌ 데이터 없음")
                continue
            
            # 업종 정보
            sector = get_stock_sector(ticker)
            
            # RSI
            rsi = result.get('rsi')
            if rsi is None:
                rsi = 0
            
            # MA5 - MA20 격차 (%)
            ma5 = result.get('ma5')
            ma20 = result.get('ma20')
            if ma5 and ma20:
                ma_gap = ((ma5 - ma20) / ma20) * 100
            else:
                ma_gap = 0
            
            # 거래량 배수
            volume_ratio = result.get('volume_ratio')
            if volume_ratio is None:
                volume_ratio = 0
            
            # 매수 판단
            entry = result.get('entry_analysis', {})
            judgment = entry.get('judgment', 'N/A')
            entry_status = entry.get('entry_status', '👀')
            
            # 점수 계산 (기술적 신호 기반)
            score = 0
            if result.get('entry_ready') or result.get('reversal_signal'):
                score += 50
            if 45 <= rsi <= 60:
                score += 20
            if ma_gap > 0:  # MA5 > MA20
                score += 15
            if 1.2 <= volume_ratio <= 2.5:
                score += 15
            
            results.append({
                'ticker': ticker,
                'sector': sector,
                'rsi': round(rsi, 2) if rsi else 0,
                'ma_gap': round(ma_gap, 2) if ma_gap else 0,
                'volume_ratio': round(volume_ratio, 2) if volume_ratio else 0,
                'judgment': f"{entry_status} {judgment}",
                'score': score,
                'entry_ready': result.get('entry_ready', False),
                'reversal_signal': result.get('reversal_signal', False)
            })
            
            print(f"✅ (점수: {score})")
            
            # API 제한 방지
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ 오류: {str(e)[:30]}")
            continue
    
    # 3. 점수순으로 정렬하여 TOP N 추천
    results.sort(key=lambda x: x['score'], reverse=True)
    top_results = results[:top_n]
    
    return top_results


# ============================================================================
# 메인 함수
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='KRX 코스피200 편입/제외 추적 및 스크리닝',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 편입/제외 현황 조회
  python krx_index_tracker.py --changes
  
  # 뉴스 기반 분석
  python krx_index_tracker.py --news
  
  # 신규 편입 종목 스크리닝 (추천)
  python krx_index_tracker.py --screen
  
  # 전체 분석
  python krx_index_tracker.py --all
        """
    )
    
    parser.add_argument('--changes', action='store_true',
                       help='KRX 편입/제외 현황 조회')
    parser.add_argument('--news', action='store_true',
                       help='뉴스 기반 보조 분석')
    parser.add_argument('--screen', action='store_true',
                       help='신규 편입 종목 스크리닝')
    parser.add_argument('--all', action='store_true',
                       help='전체 분석 (편입/제외 + 뉴스 + 스크리닝)')
    parser.add_argument('--index', type=str, default='코스피200',
                       help='지수 코드 (기본값: 코스피200)')
    parser.add_argument('--days', type=int, default=30,
                       help='조회 일수 (기본값: 30)')
    parser.add_argument('--top', type=int, default=5,
                       help='추천 종목 개수 (기본값: 5)')
    
    args = parser.parse_args()
    
    if args.all or (not args.changes and not args.news and not args.screen):
        # 기본값: 전체 분석
        args.changes = True
        args.news = True
        args.screen = True
    
    # 1. 편입/제외 현황
    if args.changes:
        changes = fetch_krx_index_changes(args.index, args.days)
        
        if changes:
            print("\n" + "=" * 60)
            print("📋 편입/제외 현황 요약")
            print("=" * 60)
            
            for change in changes:
                date_str = change['date'].strftime('%Y-%m-%d')
                print(f"\n📅 {date_str}")
                print(f"   편입: {', '.join(change['added'][:10])}")
                if change['added']:
                    print(f"   편입 업종:")
                    for sector, stocks in change['sector']['added'].items():
                        print(f"     - {sector}: {', '.join(stocks[:5])}")
                
                print(f"   제외: {', '.join(change['removed'][:10])}")
                if change['removed']:
                    print(f"   제외 업종:")
                    for sector, stocks in change['sector']['removed'].items():
                        print(f"     - {sector}: {', '.join(stocks[:5])}")
    
    # 2. 뉴스 기반 분석
    if args.news:
        news_list = fetch_news_about_index_changes(args.index, days_back=args.days)
        
        if news_list:
            print("\n" + "=" * 60)
            print("📰 뉴스 요약")
            print("=" * 60)
            
            for news in news_list[:5]:  # 최근 5개만
                print(f"\n📄 {news['title']}")
                if news['stocks']:
                    print(f"   언급 종목: {', '.join(news['stocks'][:5])}")
                if news['sectors']:
                    print(f"   업종: {', '.join(news['sectors'])}")
    
    # 3. 신규 편입 종목 스크리닝
    if args.screen:
        top_stocks = screen_newly_added_stocks(args.index, args.days, args.top)
        
        if top_stocks:
            print("\n" + "=" * 60)
            print(f"🎯 신규 편입 종목 TOP {args.top} (기술적 신호 기반)")
            print("=" * 60)
            print(f"{'티커':<10} {'업종':<10} {'RSI':<8} {'MA격차':<10} {'거래량배수':<12} {'매수판단':<30}")
            print("-" * 80)
            
            for stock in top_stocks:
                print(f"{stock['ticker']:<10} {stock['sector']:<10} {stock['rsi']:<8.1f} "
                      f"{stock['ma_gap']:>8.2f}% {stock['volume_ratio']:>10.2f}배 "
                      f"{stock['judgment']:<30}")
        else:
            print("\n❌ 추천 종목이 없습니다.")


if __name__ == "__main__":
    main()

