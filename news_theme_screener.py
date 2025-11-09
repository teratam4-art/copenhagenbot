#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
뉴스 기반 테마 주식 스크리닝 시스템
뉴스 감지 → 테마 스코어링 → 종목 연결 → 기술적 신호 결합
"""

import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import sqlite3
import json
import re
from collections import defaultdict, Counter
import os

# yfinance for US stocks
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance 패키지가 설치되지 않았습니다.")
    print("   pip install yfinance")

# pytz for timezone
try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
    print("⚠️  pytz 패키지가 설치되지 않았습니다.")
    print("   pip install pytz")

# 한국 주식 스크리닝을 위한 import
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from stock_screener import check_buy_signal, is_us_stock
    STOCK_SCREENER_AVAILABLE = True
except ImportError:
    STOCK_SCREENER_AVAILABLE = False
    print("⚠️  stock_screener 모듈을 찾을 수 없습니다.")


# ============================================================================
# 1️⃣ 뉴스 감지 및 테마 스코어링
# ============================================================================

class NewsThemeScorer:
    """뉴스 기반 테마 스코어링 클래스"""
    
    def __init__(self):
        # 테마 키워드 딕셔너리 (산업별 키워드)
        self.theme_keywords = {
            '신재생에너지': {
                'keywords': ['신재생', '태양광', '풍력', '재생에너지', '탄소중립', '그린뉴딜', 'ESG', '친환경'],
                'co_occurrence': ['정부', '지원', '확대', '투자', '산업'],
                'positive': ['확대', '지원', '증가', '성장', '투자', '확산'],
                'negative': ['감축', '지연', '축소', '감소', '위축']
            },
            '반도체': {
                'keywords': ['반도체', '메모리', 'D램', '낸드', '웨이퍼', '파운드리', 'HBM', 'AI반도체'],
                'co_occurrence': ['수요', '공급', '가격', '투자', '설비'],
                'positive': ['수요 증가', '가격 상승', '투자 확대', '성장', '회복'],
                'negative': ['감산', '가격 하락', '수요 감소', '과잉', '부진']
            },
            'AI': {
                'keywords': ['AI', '인공지능', '머신러닝', '딥러닝', 'GPT', 'LLM', '생성AI', '챗GPT'],
                'co_occurrence': ['투자', '기술', '혁신', '플랫폼'],
                'positive': ['혁신', '투자', '성장', '확대', '도입'],
                'negative': ['규제', '우려', '위험', '한계']
            },
            'EV': {
                'keywords': ['전기차', 'EV', '배터리', '충전', '리튬', 'LFP', '전고체'],
                'co_occurrence': ['수요', '판매', '출시', '시장'],
                'positive': ['수요 증가', '출시', '확대', '성장'],
                'negative': ['감소', '부진', '지연']
            },
            '방산': {
                'keywords': ['방산', '국방', '무기', '미사일', '레이더', '방산수출'],
                'co_occurrence': ['수주', '계약', '정부', '국방'],
                'positive': ['수주', '계약', '증가', '확대'],
                'negative': ['지연', '취소', '감소']
            },
            '바이오': {
                'keywords': ['바이오', '신약', '임상', 'FDA', '허가', '바이오텍'],
                'co_occurrence': ['허가', '임상', '개발', '투자'],
                'positive': ['허가', '성공', '개발', '투자'],
                'negative': ['실패', '중단', '지연']
            },
            '2차전지': {
                'keywords': ['2차전지', '배터리', '양극재', '음극재', '전해액', '리튬'],
                'co_occurrence': ['수요', '투자', '공급', '가격'],
                'positive': ['수요 증가', '투자', '확대'],
                'negative': ['과잉', '가격 하락', '감소']
            }
        }
        
        # 공동출현 가중치
        self.co_occurrence_weight = 1.5
        # 긍정/부정 가중치
        self.sentiment_weight = 1.3
        
    def calculate_theme_score(self, title, content="", theme_name=""):
        """
        뉴스 제목과 본문에서 테마 점수 계산
        
        Args:
            title: 뉴스 제목
            content: 뉴스 본문 (선택)
            theme_name: 테마 이름
        
        Returns:
            dict: {theme: score, sentiment: 'positive'/'negative'/'neutral'}
        """
        if theme_name not in self.theme_keywords:
            return None
        
        theme_info = self.theme_keywords[theme_name]
        text = (title + " " + content).lower()
        
        score = 0.0
        sentiment_score = 0
        
        # 1. 기본 키워드 매칭
        keyword_count = 0
        for keyword in theme_info['keywords']:
            if keyword.lower() in text:
                keyword_count += 1
                score += 1.0
        
        if keyword_count == 0:
            return None
        
        # 2. 공동출현 키워드 가중치
        co_occurrence_count = 0
        for co_word in theme_info['co_occurrence']:
            if co_word.lower() in text:
                co_occurrence_count += 1
                score += self.co_occurrence_weight
        
        # 3. 감정 분석
        positive_count = 0
        negative_count = 0
        
        for pos_word in theme_info['positive']:
            if pos_word.lower() in text:
                positive_count += 1
                sentiment_score += 1
        
        for neg_word in theme_info['negative']:
            if neg_word.lower() in text:
                negative_count += 1
                sentiment_score -= 1
        
        # 감정 점수 반영
        if sentiment_score > 0:
            score *= self.sentiment_weight
            sentiment = 'positive'
        elif sentiment_score < 0:
            score *= (1 / self.sentiment_weight)
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # 공동출현 보너스
        if co_occurrence_count >= 2:
            score *= 1.2
        
        return {
            'theme': theme_name,
            'score': round(score, 2),
            'sentiment': sentiment,
            'keyword_count': keyword_count,
            'co_occurrence_count': co_occurrence_count,
            'sentiment_score': sentiment_score
        }
    
    def analyze_news_batch(self, news_list):
        """
        뉴스 리스트를 분석하여 테마별 점수 집계
        
        Args:
            news_list: [{'title': str, 'content': str, 'date': datetime}, ...]
        
        Returns:
            dict: {theme_name: {'total_score': float, 'count': int, 'avg_score': float, 'sentiment': str}}
        """
        theme_scores = defaultdict(lambda: {'total_score': 0, 'count': 0, 'scores': [], 'sentiments': []})
        
        for news in news_list:
            title = news.get('title', '')
            content = news.get('content', '')
            
            # 모든 테마에 대해 점수 계산
            for theme_name in self.theme_keywords.keys():
                result = self.calculate_theme_score(title, content, theme_name)
                if result:
                    theme_scores[theme_name]['total_score'] += result['score']
                    theme_scores[theme_name]['count'] += 1
                    theme_scores[theme_name]['scores'].append(result['score'])
                    theme_scores[theme_name]['sentiments'].append(result['sentiment'])
        
        # 집계 결과 정리
        theme_summary = {}
        for theme, data in theme_scores.items():
            if data['count'] > 0:
                # 총점 계산 (뉴스 건수 × 평균 점수에 가중치)
                total_score = data['total_score'] * (1 + data['count'] * 0.1)  # 건수가 많을수록 가중치
                avg_score = data['total_score'] / data['count']
                # 감정 집계
                sentiment_counts = Counter(data['sentiments'])
                dominant_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else 'neutral'
                
                theme_summary[theme] = {
                    'total_score': round(total_score, 2),
                    'count': data['count'],
                    'avg_score': round(avg_score, 2),
                    'sentiment': dominant_sentiment,
                    'score_change': 0  # 전일 대비 변화량 (추후 계산)
                }
        
        return theme_summary


# ============================================================================
# 2️⃣ 뉴스 크롤링
# ============================================================================

def fetch_korean_news(date=None, limit=50):
    """
    네이버 뉴스에서 경제/증권 뉴스 크롤링
    
    Args:
        date: 날짜 (기본값: 오늘)
        limit: 가져올 뉴스 개수
    
    Returns:
        list: [{'title': str, 'content': str, 'date': datetime, 'url': str}, ...]
    """
    news_list = []
    
    if date is None:
        date = datetime.now()
    
    try:
        # 네이버 뉴스 경제 섹션 (증권 뉴스)
        urls = [
            "https://news.naver.com/main/list.naver?mode=LS2D&mid=shm&sid1=101&sid2=259",  # 증권
            "https://news.naver.com/main/list.naver?mode=LS2D&mid=shm&sid1=101&sid2=258",  # 경제
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        count = 0
        for url in urls:
            if count >= limit:
                break
                
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 다양한 형식의 뉴스 리스트 시도
                news_items = []
                
                # 방법 1: type06 형식
                news_items = soup.find_all('li', class_=lambda x: x and ('type06' in x or '_sa_item' in x))
                
                # 방법 2: dt 태그
                if not news_items:
                    news_items = soup.find_all('dt')
                
                # 방법 3: a 태그로 링크 찾기
                if not news_items:
                    links = soup.find_all('a', href=lambda x: x and '/article/' in x)
                    news_items = links[:limit]
                
                for item in news_items:
                    if count >= limit:
                        break
                    
                    try:
                        # 제목 추출
                        if item.name == 'a':
                            title = item.get_text(strip=True)
                            link = item.get('href', '')
                        else:
                            title_elem = item.find('a')
                            if title_elem:
                                title = title_elem.get_text(strip=True)
                                link = title_elem.get('href', '')
                            else:
                                title = item.get_text(strip=True)
                                link = ''
                        
                        if not title or len(title) < 10:
                            continue
                        
                        # URL 완성
                        if link and not link.startswith('http'):
                            link = 'https://news.naver.com' + link
                        
                        # 본문은 제목으로 대체 (간단 버전)
                        content = title
                        
                        news_list.append({
                            'title': title,
                            'content': content,
                            'date': date,
                            'url': link
                        })
                        
                        count += 1
                        
                    except Exception:
                        continue
                        
            except Exception as e:
                continue
        
        # 뉴스가 없으면 샘플 데이터 추가 (테스트용)
        if not news_list:
            print("   ⚠️  뉴스 크롤링 실패, 샘플 데이터 사용")
            news_list = [
                {
                    'title': '정부, 탄소중립 신재생에너지 투자 확대 발표',
                    'content': '정부가 신재생에너지 산업 지원을 확대하고 태양광 및 풍력 발전 투자를 늘린다고 발표했다.',
                    'date': date,
                    'url': ''
                },
                {
                    'title': '반도체 수요 감소로 메모리 가격 하락 전망',
                    'content': '반도체 업계에서 D램과 낸드플래시 가격 하락이 예상되며, 공급 과잉 우려가 나온다.',
                    'date': date,
                    'url': ''
                },
                {
                    'title': 'AI 반도체 수요 증가로 엔비디아 주가 상승',
                    'content': '생성AI 기술 확산으로 AI 반도체 수요가 급증하며 엔비디아와 관련 기업 주가가 상승했다.',
                    'date': date,
                    'url': ''
                }
            ]
        
    except Exception as e:
        print(f"⚠️  뉴스 크롤링 오류: {e}")
        # 샘플 데이터 반환
        news_list = [
            {
                'title': '신재생에너지 정부 지원 확대',
                'content': '정부가 신재생에너지 산업 지원을 확대한다고 발표했다.',
                'date': date,
                'url': ''
            }
        ]
    
    return news_list


def fetch_us_news(limit=50):
    """
    미국 주식 관련 뉴스 크롤링 (Yahoo Finance News)
    
    Args:
        limit: 가져올 뉴스 개수
    
    Returns:
        list: [{'title': str, 'content': str, 'date': datetime, 'url': str}, ...]
    """
    news_list = []
    
    if not YFINANCE_AVAILABLE:
        return news_list
    
    try:
        # 주요 종목들의 뉴스 수집
        major_tickers = ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AAPL', 'MSFT']
        
        for ticker in major_tickers[:3]:  # 처음 3개만
            try:
                ticker_obj = yf.Ticker(ticker)
                news = ticker_obj.news
                
                for item in news[:limit//len(major_tickers)]:
                    news_list.append({
                        'title': item.get('title', ''),
                        'content': item.get('summary', ''),
                        'date': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                        'url': item.get('link', '')
                    })
            except Exception:
                continue
                
    except Exception as e:
        print(f"⚠️  미국 뉴스 크롤링 오류: {e}")
    
    return news_list


# ============================================================================
# 3️⃣ 산업 매핑 및 종목 연결
# ============================================================================

class IndustryMapper:
    """산업별 종목 자동 매핑 클래스"""
    
    def __init__(self):
        # 테마별 연관 종목 (자동 업데이트 가능하도록 구조화)
        self.theme_stocks = {
            '신재생에너지': {
                'korea': ['009830', '015890', '066570', '112610', '012320'],  # 씨에스윈드, 태경산업 등
                'us': ['ENPH', 'RUN', 'SEDG', 'FSLR', 'NEE']
            },
            '반도체': {
                'korea': ['000660', '006400', '005930', '000990', '035720'],  # SK하이닉스, 삼성전자 등
                'us': ['NVDA', 'AMD', 'TSM', 'INTC', 'ASML']
            },
            'AI': {
                'korea': ['035720', '000660', '005930', '000990'],  # 한미반도체, SK하이닉스 등
                'us': ['NVDA', 'AMD', 'MSFT', 'GOOGL', 'META', 'AAPL']
            },
            'EV': {
                'korea': ['005380', '012330', '003670', '051910', '096770'],  # 현대차, 기아, LG화학 등
                'us': ['TSLA', 'RIVN', 'LCID', 'F', 'GM']
            },
            '방산': {
                'korea': ['012450', '047810', '042660', '013520', '039130'],  # 한화, LIG넥스원 등
                'us': ['LMT', 'RTX', 'BA', 'NOC', 'GD']
            },
            '바이오': {
                'korea': ['068270', '207940', '095700', '086790', '003550'],  # 셀트리온, 삼성바이오로직스 등
                'us': ['AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB']
            },
            '2차전지': {
                'korea': ['051910', '096770', '373220', '357780', '247540'],  # LG화학, LG에너지솔루션 등
                'us': ['TSLA', 'ENPH', 'RUN']
            }
        }
    
    def get_stocks_by_theme(self, theme_name, market='both'):
        """
        테마별 종목 리스트 가져오기
        
        Args:
            theme_name: 테마 이름
            market: 'korea', 'us', 'both'
        
        Returns:
            list: 종목 코드 리스트
        """
        if theme_name not in self.theme_stocks:
            return []
        
        stocks = []
        theme_data = self.theme_stocks[theme_name]
        
        if market == 'both' or market == 'korea':
            stocks.extend(theme_data.get('korea', []))
        
        if market == 'both' or market == 'us':
            stocks.extend(theme_data.get('us', []))
        
        return stocks
    
    def get_all_theme_stocks(self, themes, market='both'):
        """
        여러 테마의 종목들을 합쳐서 반환
        
        Args:
            themes: 테마 이름 리스트
            market: 'korea', 'us', 'both'
        
        Returns:
            list: 종목 코드 리스트 (중복 제거)
        """
        all_stocks = []
        for theme in themes:
            stocks = self.get_stocks_by_theme(theme, market)
            all_stocks.extend(stocks)
        
        # 중복 제거
        return list(set(all_stocks))


# ============================================================================
# 4️⃣ 뉴스 반응 지연 패턴 학습
# ============================================================================

class NewsReactionTracker:
    """뉴스 반응 지연 패턴 학습 및 추적"""
    
    def __init__(self, db_path='news_reaction.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 뉴스 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                theme TEXT,
                date TEXT,
                score REAL,
                sentiment TEXT,
                title TEXT
            )
        ''')
        
        # 주가 반응 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_reactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                news_id INTEGER,
                ticker TEXT,
                date TEXT,
                price_change REAL,
                days_after INTEGER,
                FOREIGN KEY (news_id) REFERENCES news_events(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_news_event(self, theme, date, score, sentiment, title):
        """뉴스 이벤트 기록"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO news_events (theme, date, score, sentiment, title)
            VALUES (?, ?, ?, ?, ?)
        ''', (theme, date.strftime('%Y-%m-%d'), score, sentiment, title))
        
        news_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return news_id
    
    def record_price_reaction(self, news_id, ticker, date, price_change, days_after):
        """주가 반응 기록"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO price_reactions (news_id, ticker, date, price_change, days_after)
            VALUES (?, ?, ?, ?, ?)
        ''', (news_id, ticker, date.strftime('%Y-%m-%d'), price_change, days_after))
        
        conn.commit()
        conn.close()
    
    def calculate_reaction_pattern(self, theme, days_back=30):
        """
        테마별 평균 반응 지연 패턴 계산
        
        Returns:
            dict: {'avg_delay': float, 'avg_return': float, 'sample_count': int}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT AVG(pr.days_after), AVG(pr.price_change), COUNT(*)
            FROM price_reactions pr
            JOIN news_events ne ON pr.news_id = ne.id
            WHERE ne.theme = ? AND ne.date >= ?
        ''', (theme, cutoff_date))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[2] > 0:
            return {
                'avg_delay': round(result[0], 1),
                'avg_return': round(result[1], 2),
                'sample_count': result[2]
            }
        else:
            return {
                'avg_delay': 2.0,  # 기본값
                'avg_return': 0.0,
                'sample_count': 0
            }


# ============================================================================
# 5️⃣ 메인 스크리닝 함수
# ============================================================================

def screen_news_themes(days=1, market='both', min_score=5.0):
    """
    뉴스 기반 테마 스크리닝 메인 함수
    
    Args:
        days: 분석할 일수
        market: 'korea', 'us', 'both'
        min_score: 최소 테마 점수
    
    Returns:
        dict: 테마별 점수 및 종목 정보
    """
    print("=" * 60)
    print("📰 뉴스 기반 테마 주식 스크리닝")
    print("=" * 60)
    
    # 1. 뉴스 크롤링
    print(f"\n1️⃣ 뉴스 크롤링 중...")
    news_list = []
    
    if market == 'both' or market == 'korea':
        korean_news = fetch_korean_news(limit=50)
        news_list.extend(korean_news)
        print(f"   한국 뉴스: {len(korean_news)}건")
    
    if market == 'both' or market == 'us':
        us_news = fetch_us_news(limit=30)
        news_list.extend(us_news)
        print(f"   미국 뉴스: {len(us_news)}건")
    
    if not news_list:
        print("❌ 뉴스를 가져올 수 없습니다.")
        return {}
    
    # 2. 테마 스코어링
    print(f"\n2️⃣ 테마 스코어링 중...")
    scorer = NewsThemeScorer()
    theme_scores = scorer.analyze_news_batch(news_list)
    
    if not theme_scores:
        print("❌ 테마 점수를 계산할 수 없습니다.")
        # 디버깅: 샘플 뉴스로 테스트
        if news_list:
            print(f"   디버깅: 첫 번째 뉴스 제목: {news_list[0].get('title', '')[:50]}")
        return {}
    
    # 모든 테마 점수 출력 (디버깅)
    print(f"   계산된 테마 점수:")
    for theme, data in sorted(theme_scores.items(), key=lambda x: x[1]['total_score'], reverse=True):
        print(f"     {theme}: {data['total_score']:.2f}점 ({data['count']}건)")
    
    # 3. 상위 테마 선정
    sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
    top_themes = [(name, data) for name, data in sorted_themes if data['total_score'] >= min_score]
    
    print(f"\n3️⃣ 상위 테마 선정 (점수 {min_score} 이상):")
    for theme_name, data in top_themes:
        sentiment_emoji = "🔼" if data['sentiment'] == 'positive' else "🔽" if data['sentiment'] == 'negative' else "➡️"
        print(f"   {theme_name} {sentiment_emoji} +{data['total_score']:.1f} ({data['count']}건, 감정: {data['sentiment']})")
    
    # 4. 종목 매핑
    print(f"\n4️⃣ 종목 매핑 중...")
    mapper = IndustryMapper()
    
    results = {}
    for theme_name, theme_data in top_themes:
        stocks = mapper.get_all_theme_stocks([theme_name], market=market)
        results[theme_name] = {
            'score': theme_data['total_score'],
            'sentiment': theme_data['sentiment'],
            'news_count': theme_data['count'],
            'stocks': stocks
        }
        print(f"   {theme_name}: {len(stocks)}개 종목")
    
    return results


# ============================================================================
# 6️⃣ 기술적 신호와 결합
# ============================================================================

def combine_news_and_technical(news_results, rsi_min=45, rsi_max=60, volume_min=1.2):
    """
    뉴스 테마와 기술적 신호를 결합
    
    Args:
        news_results: screen_news_themes()의 결과
        rsi_min, rsi_max, volume_min: 기술적 조건
    
    Returns:
        list: 조건을 만족하는 종목 리스트
    """
    print("\n" + "=" * 60)
    print("🔗 뉴스 테마 + 기술적 신호 결합")
    print("=" * 60)
    
    if not STOCK_SCREENER_AVAILABLE:
        print("❌ stock_screener 모듈을 사용할 수 없습니다.")
        return []
    
    all_candidates = []
    
    for theme_name, theme_data in news_results.items():
        print(f"\n📊 {theme_name} 테마 분석 중...")
        stocks = theme_data['stocks']
        
        if not stocks:
            continue
        
        theme_candidates = []
        analyzed_count = 0
        
        for ticker in stocks[:20]:  # 각 테마당 최대 20개만
            try:
                analyzed_count += 1
                print(f"   [{analyzed_count}/{min(len(stocks), 20)}] {ticker} 분석 중...", end=" ")
                
                result = check_buy_signal(
                    ticker, 
                    period="3mo",
                    rsi_min=rsi_min,
                    rsi_max=rsi_max,
                    volume_min=volume_min,
                    volume_max=3.0
                )
                
                if result is None:
                    print("❌ 데이터 없음")
                    continue
                
                # 진입 가능 신호 또는 반등 신호가 있는 경우
                if result.get('entry_ready') or result.get('reversal_signal') or result.get('condition_met'):
                    theme_candidates.append({
                        'ticker': ticker,
                        'theme': theme_name,
                        'theme_score': theme_data['score'],
                        'technical': result
                    })
                    print(f"✅ 뉴스 테마 + 기술적 신호 일치")
                else:
                    print(f"❌ (기술적 신호 부족)")
                    
                # API 제한 방지
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ 오류: {str(e)[:30]}")
                continue
        
        all_candidates.extend(theme_candidates)
    
    return all_candidates


# ============================================================================
# 7️⃣ 주간 리포트 생성
# ============================================================================

def generate_weekly_report(output_file='weekly_theme_report.md'):
    """
    주간 테마 리포트 생성
    
    Args:
        output_file: 출력 파일 경로
    """
    print("\n" + "=" * 60)
    print("📊 주간 테마 리포트 생성")
    print("=" * 60)
    
    # 지난 7일간 뉴스 분석
    scorer = NewsThemeScorer()
    all_news = []
    
    for i in range(7):
        date = datetime.now() - timedelta(days=i)
        news = fetch_korean_news(date=date, limit=20)
        all_news.extend(news)
    
    theme_scores = scorer.analyze_news_batch(all_news)
    
    # 상위 5개 테마
    sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)[:5]
    
    # 리포트 작성
    report = f"""# 주간 테마 리포트
생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📈 Top 5 테마 (지난 7일)

"""
    
    for idx, (theme_name, data) in enumerate(sorted_themes, 1):
        sentiment_emoji = "🔼" if data['sentiment'] == 'positive' else "🔽" if data['sentiment'] == 'negative' else "➡️"
        report += f"""### {idx}️⃣ {theme_name} {sentiment_emoji}
- **점수**: {data['total_score']:.1f}
- **뉴스 건수**: {data['count']}건
- **평균 점수**: {data['avg_score']:.2f}
- **감정**: {data['sentiment']}

"""
    
    # 종목 추천
    mapper = IndustryMapper()
    report += "## 🎯 추천 종목\n\n"
    
    for idx, (theme_name, data) in enumerate(sorted_themes[:3], 1):
        stocks = mapper.get_all_theme_stocks([theme_name], market='both')
        report += f"### {theme_name}\n"
        report += f"관련 종목: {', '.join(stocks[:10])}\n\n"
    
    # 리포트 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 리포트 저장 완료: {output_file}")
    return report


# ============================================================================
# 6️⃣ 알림 시스템 (선택적)
# ============================================================================

def send_slack_notification(message, webhook_url=None):
    """
    Slack Webhook으로 알림 전송
    
    Args:
        message: 전송할 메시지
        webhook_url: Slack Webhook URL (환경변수 SLACK_WEBHOOK_URL 사용 가능)
    
    Returns:
        bool: 전송 성공 여부
    """
    if webhook_url is None:
        webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    
    if not webhook_url:
        return False
    
    try:
        payload = {'text': message}
        response = requests.post(webhook_url, json=payload, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def send_telegram_notification(message, bot_token=None, chat_id=None):
    """
    텔레그램 봇으로 알림 전송
    
    Args:
        message: 전송할 메시지
        bot_token: 텔레그램 봇 토큰 (환경변수 TELEGRAM_BOT_TOKEN 사용 가능)
        chat_id: 채팅 ID (환경변수 TELEGRAM_CHAT_ID 사용 가능)
    
    Returns:
        bool: 전송 성공 여부
    """
    if bot_token is None:
        bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if chat_id is None:
        chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        return False
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def check_trigger_conditions(candidates, notification_enabled=False):
    """
    트리거 조건 확인 및 알림 전송
    
    Args:
        candidates: combine_news_and_technical()의 결과
        notification_enabled: 알림 활성화 여부
    
    Returns:
        list: 알림 전송된 종목 리스트
    """
    if not notification_enabled:
        return []
    
    notified = []
    
    for candidate in candidates:
        result = candidate['technical']
        entry = result.get('entry_analysis', {})
        
        # 조건: 진입 가능 + RSI 적정 + 거래량 증가
        if entry.get('entry_status') == '✅' and entry.get('rsi', 0) >= 45:
            message = f"""🚀 **매수 신호 감지**
테마: {candidate['theme']}
종목: {candidate['ticker']}
테마 점수: {candidate['theme_score']:.1f}
현재가: {entry.get('current_price', 'N/A')}
판단: {entry.get('judgment', 'N/A')}
"""
            
            # Slack 알림 시도
            if send_slack_notification(message):
                notified.append(candidate['ticker'])
                continue
            
            # 텔레그램 알림 시도
            if send_telegram_notification(message):
                notified.append(candidate['ticker'])
    
    return notified


# ============================================================================
# 메인 함수
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='뉴스 기반 테마 주식 스크리닝 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 한국 뉴스 기반 테마 스크리닝
  python news_theme_screener.py --market korea
  
  # 미국 뉴스 기반 테마 스크리닝
  python news_theme_screener.py --market us
  
  # 뉴스 + 기술적 신호 결합
  python news_theme_screener.py --combine-technical
  
  # 주간 리포트 생성
  python news_theme_screener.py --weekly-report
        """
    )
    
    parser.add_argument('--market', type=str, default='both', choices=['korea', 'us', 'both'],
                       help='뉴스 시장 (기본값: both)')
    parser.add_argument('--min-score', type=float, default=1.0,
                       help='최소 테마 점수 (기본값: 1.0)')
    parser.add_argument('--combine-technical', action='store_true',
                       help='기술적 신호와 결합')
    parser.add_argument('--weekly-report', action='store_true',
                       help='주간 리포트 생성')
    parser.add_argument('--rsi-min', type=int, default=45,
                       help='RSI 최소값 (기본값: 45)')
    parser.add_argument('--rsi-max', type=int, default=60,
                       help='RSI 최대값 (기본값: 60)')
    parser.add_argument('--volume-min', type=float, default=1.2,
                       help='거래량 최소 배수 (기본값: 1.2)')
    parser.add_argument('--notify', action='store_true',
                       help='알림 전송 활성화 (SLACK_WEBHOOK_URL 또는 TELEGRAM 환경변수 필요)')
    
    args = parser.parse_args()
    
    if args.weekly_report:
        # 주간 리포트 생성
        generate_weekly_report()
        return
    
    # 뉴스 기반 테마 스크리닝
    news_results = screen_news_themes(
        days=1,
        market=args.market,
        min_score=args.min_score
    )
    
    if not news_results:
        print("\n❌ 분석할 테마가 없습니다.")
        return
    
    # 기술적 신호와 결합
    if args.combine_technical:
        candidates = combine_news_and_technical(
            news_results,
            rsi_min=args.rsi_min,
            rsi_max=args.rsi_max,
            volume_min=args.volume_min
        )
        
        if candidates:
            print("\n" + "=" * 60)
            print("🎯 최종 추천 종목 (뉴스 테마 + 기술적 신호)")
            print("=" * 60)
            
            for candidate in candidates:
                result = candidate['technical']
                entry = result.get('entry_analysis', {})
                price_fmt = entry.get('price_format', '${:,.2f}')
                
                print(f"\n📈 {candidate['ticker']} ({candidate['theme']})")
                print(f"   테마 점수: {candidate['theme_score']:.1f}")
                if entry:
                    print(f"   현재가: {price_fmt.format(entry.get('current_price', 0))}")
                    print(f"   판단: {entry.get('entry_status', 'N/A')} {entry.get('judgment', 'N/A')}")
            
            # 알림 전송
            if args.notify:
                print("\n" + "=" * 60)
                print("🔔 알림 전송 중...")
                notified = check_trigger_conditions(candidates, notification_enabled=True)
                if notified:
                    print(f"   ✅ {len(notified)}개 종목 알림 전송 완료")
                else:
                    print("   ⚠️  알림 전송 실패 (환경변수 확인 필요)")
        else:
            print("\n❌ 뉴스 테마와 기술적 신호가 일치하는 종목이 없습니다.")
    else:
        # 테마별 종목만 표시
        print("\n" + "=" * 60)
        print("📋 테마별 종목 리스트")
        print("=" * 60)
        
        for theme_name, theme_data in news_results.items():
            print(f"\n{theme_name} (점수: {theme_data['score']:.1f})")
            print(f"  종목: {', '.join(theme_data['stocks'][:10])}")


if __name__ == "__main__":
    main()

