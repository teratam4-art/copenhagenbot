#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crude oil sentiment pulse checker tuned for Devon Energy (DVN).

Pipeline
--------
1. Crawl selected energy news sources and retain links containing oil-related
   keywords.
2. Fetch each article, count bearish (supply surplus, weak demand, strong USD)
   and bullish (supply disruption, geopolitical risk) signals.
3. Pull quick market snapshots (DXY, WTI front-month) from Yahoo Finance.
4. Blend keyword scores with market data to classify today's crude tone:
      - score <= -2 : downside pressure
      - -1 <= score <= 1 : neutral / noisy
      - score >= 2 : tightening / upside risk
5. Translate the crude outlook into a DVN trading stance hint.

Run
---
    python crude_oil_sentiment.py
    python crude_oil_sentiment.py --max-articles 20 --json

Remarks
-------
- Network access is required for live crawling and market data.
- Designed for fast situational awareness, not investment advice.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Macro context (static primer printed in output header)
# ---------------------------------------------------------------------------

MACRO_CONTEXT = (
    "2025 유가는 공급 우위 기조가 기본값입니다. IEA/EIA 모두 비OPEC 증산(미국·브라질·가이아나)이 "
    "수요 증가를 상회할 것으로 보며, OPEC+ 또한 완전한 감산보다는 부분 완화에 가깝습니다. "
    "따라서 공급 차질 뉴스가 없다면 기본 시나리오는 '가격 눌림 유지' 쪽입니다."
)

# ---------------------------------------------------------------------------
# News sources and keyword lexicons
# ---------------------------------------------------------------------------

# (name, url, parser)
NEWS_SOURCES: List[Tuple[str, str]] = [
    ("EIA Today in Energy", "https://www.eia.gov/rss/todayinenergy.xml"),
    ("Oilprice", "https://oilprice.com/rss/main"),
    ("Reuters Commodities", "https://feeds.reuters.com/news/commodities"),
    ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
]

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

RSS_TIMEOUT = 20

# Keywords that imply bearish pressure (more supply, weaker demand, stronger USD)
SUPPLY_BEAR_KEYS = {
    "output", "production", "supply", "export", "opec", "opec+", "barrels per day",
    "inventory build", "stock build", "storage build", "sanction relief",
    "us shale", "rig count", "capacity increase"
}

DEMAND_BEAR_KEYS = {
    "weak demand", "sluggish demand", "slowdown", "pmi", "factory activity",
    "manufacturing contraction", "economic growth", "recession",
    "china demand", "jet fuel demand", "industrial slowdown", "consumption drop"
}

USD_BEAR_KEYS = {
    "dollar index", "dxy", "strong dollar", "usd strength", "fed", "interest rate",
    "rate hike", "higher for longer", "treasury yields"
}

# Keywords that imply bullish (tightening) forces
SUPPLY_BULL_KEYS = {
    "disruption", "outage", "pipeline shut", "supply cut", "unplanned outage",
    "geopolitical risk", "strike", "houthi", "strait", "hurricane", "force majeure",
    "attack", "sanction tightening", "production halt"
}

DEMAND_BULL_KEYS = {
    "demand recovery", "rebound in demand", "travel demand", "jet fuel surge",
    "china stimulus", "manufacturing rebound"
}

# Extra weight if a bullish keyword appears in these "special" outlets
SPECIAL_WEIGHT_SOURCES = ("iea.org", "opec.org", "eia.gov")
SPECIAL_WEIGHT = 2

# ---------------------------------------------------------------------------
# Market data endpoints (Yahoo Finance chart API is simple JSON)
# ---------------------------------------------------------------------------

YAHOO_DXY_URL = "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB"
YAHOO_WTI_URL = "https://query1.finance.yahoo.com/v8/finance/chart/CL=F"

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ArticleSignal:
    source: str
    title: str
    url: str
    supply_bear: int
    demand_bear: int
    usd_bear: int
    supply_bull: int
    demand_bull: int

    @property
    def bearish_sum(self) -> int:
        return self.supply_bear + self.demand_bear + self.usd_bear

    @property
    def bullish_sum(self) -> int:
        return self.supply_bull + self.demand_bull

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["bearish_total"] = self.bearish_sum
        payload["bullish_total"] = self.bullish_sum
        return payload


@dataclass
class MarketSnapshot:
    dxy: Optional[float]
    wti: Optional[float]


@dataclass
class SentimentSummary:
    timestamp_utc: str
    supply_pressure: int
    demand_pressure: int
    usd_pressure: int
    bullish_support: int
    total_score: int
    classification: str
    dvn_playbook: str
    market: MarketSnapshot
    articles: List[ArticleSignal]

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "macro_context": MACRO_CONTEXT,
            "aggregate": {
                "supply_pressure": self.supply_pressure,
                "demand_pressure": self.demand_pressure,
                "usd_pressure": self.usd_pressure,
                "bullish_support": self.bullish_support,
                "total_score": self.total_score,
                "classification": self.classification,
                "dvn_playbook": self.dvn_playbook,
            },
            "market": {
                "dxy": self.market.dxy,
                "wti": self.market.wti,
            },
            "articles": [article.to_dict() for article in self.articles],
        }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def fetch_html(url: str, timeout: int = RSS_TIMEOUT) -> str:
    resp = requests.get(url, timeout=timeout, headers=HTTP_HEADERS)
    resp.raise_for_status()
    return resp.text


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def contains_keyword(text_lower: str, keyword: str) -> bool:
    return keyword in text_lower


def score_text(text: str, origin_url: str) -> Tuple[int, int, int, int, int]:
    text_lower = text.lower()

    supply_bear = sum(1 for kw in SUPPLY_BEAR_KEYS if contains_keyword(text_lower, kw))
    demand_bear = sum(1 for kw in DEMAND_BEAR_KEYS if contains_keyword(text_lower, kw))
    usd_bear = sum(1 for kw in USD_BEAR_KEYS if contains_keyword(text_lower, kw))

    supply_bull = sum(1 for kw in SUPPLY_BULL_KEYS if contains_keyword(text_lower, kw))
    demand_bull = sum(1 for kw in DEMAND_BULL_KEYS if contains_keyword(text_lower, kw))

    if supply_bull or demand_bull:
        if origin_url and any(tag in origin_url for tag in SPECIAL_WEIGHT_SOURCES):
            supply_bull += SPECIAL_WEIGHT

    return supply_bear, demand_bear, usd_bear, supply_bull, demand_bull


def parse_rss_items(xml_text: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []

    def _append_item(title: str, description: str, link: str) -> None:
        content = normalize_whitespace(f"{title} {description}")
        items.append(
            {
                "title": title,
                "description": description,
                "link": link,
                "content": content,
            }
        )

    try:
        root = ET.fromstring(xml_text)
        for element in root.iter():
            if element.tag.lower().endswith("item"):
                title_node = element.find("./title")
                desc_node = element.find("./description")
                link_node = element.find("./link")
                title = title_node.text.strip() if title_node is not None and title_node.text else ""
                description = (
                    desc_node.text.strip() if desc_node is not None and desc_node.text else ""
                )
                link = link_node.text.strip() if link_node is not None and link_node.text else ""
                _append_item(title, description, link)
    except ET.ParseError:
        soup = BeautifulSoup(xml_text, "html.parser")
        for item in soup.find_all("item"):
            title = item.title.get_text(strip=True) if item.title else ""
            description = item.description.get_text(" ", strip=True) if item.description else ""
            link = item.link.get_text(strip=True) if item.link else ""
            _append_item(title, description, link)

    return items


def crawl_news(max_articles: int) -> List[ArticleSignal]:
    collected: List[ArticleSignal] = []

    per_source_limit = max(5, max_articles // max(1, len(NEWS_SOURCES)))

    for source_name, source_url in NEWS_SOURCES:
        try:
            feed_text = fetch_html(source_url)
            candidates = parse_rss_items(feed_text)
        except Exception as exc:
            print(f"⚠️  소스 수집 실패: {source_name} ({source_url}) -> {exc}")
            continue

        for item in candidates:
            if len(collected) >= max_articles:
                break
            title = item.get("title", "")
            url = item.get("link", "")
            content = item.get("content", "")

            if not url and item.get("link"):
                url = item["link"]

            if not content:
                content = normalize_whitespace(title or "")

            try:
                supply_bear, demand_bear, usd_bear, supply_bull, demand_bull = score_text(content, url)

                if not any([supply_bear, demand_bear, usd_bear, supply_bull, demand_bull]):
                    continue

                collected.append(
                    ArticleSignal(
                        source=source_name,
                        title=title,
                        url=url,
                        supply_bear=supply_bear,
                        demand_bear=demand_bear,
                        usd_bear=usd_bear,
                        supply_bull=supply_bull,
                        demand_bull=demand_bull,
                    )
                )
            except Exception as exc:
                print(f"⚠️  기사 분석 실패: {title or url} ({exc})")
                continue

    return collected


def fetch_market_value(url: str) -> Optional[float]:
    try:
        raw = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        raw.raise_for_status()
        payload = raw.json()
        result = payload.get("chart", {}).get("result", [])
        if not result:
            return None
        meta = result[0].get("meta", {})
        price = meta.get("regularMarketPrice") or meta.get("previousClose")
        if price is None:
            return None
        return round(float(price), 2)
    except Exception:
        return None


def build_market_snapshot() -> MarketSnapshot:
    dxy_val = fetch_market_value(YAHOO_DXY_URL)
    wti_val = fetch_market_value(YAHOO_WTI_URL)
    return MarketSnapshot(dxy=dxy_val, wti=wti_val)


def classify_sentiment(supply_pressure: int, demand_pressure: int, usd_pressure: int,
                       bullish_support: int, market: MarketSnapshot) -> Tuple[int, str]:
    bearish_total = supply_pressure + demand_pressure + usd_pressure
    total_score = bullish_support - bearish_total

    if market.dxy and market.dxy >= 102:
        total_score -= 1
    if market.wti and market.wti <= 70:
        total_score -= 1

    if total_score <= -2:
        classification = "오늘은 유가 하방 압력"
    elif total_score >= 2:
        classification = "타이트 가능성 (상방 경계)"
    else:
        classification = "중립 (뉴스 혼재)"

    return total_score, classification


def dvn_playbook(total_score: int) -> str:
    if total_score <= -2:
        return "DVN은 보수적 접근. 눌림 매수 대기 및 포지션 축소 검토."
    if total_score >= 2:
        return "DVN은 공격적 스윙 가능. 공급 차질 확인 시 분할 매수."
    return "DVN은 관망 또는 소규모 비중. 명확한 모멘텀 확인 필요."


def summarize(articles: List[ArticleSignal], market: MarketSnapshot) -> SentimentSummary:
    supply_pressure = sum(article.supply_bear for article in articles)
    demand_pressure = sum(article.demand_bear for article in articles)
    usd_pressure = sum(article.usd_bear for article in articles)
    bullish_support = sum(article.bullish_sum for article in articles)

    total_score, classification = classify_sentiment(
        supply_pressure,
        demand_pressure,
        usd_pressure,
        bullish_support,
        market,
    )

    playbook = dvn_playbook(total_score)

    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return SentimentSummary(
        timestamp_utc=timestamp,
        supply_pressure=supply_pressure,
        demand_pressure=demand_pressure,
        usd_pressure=usd_pressure,
        bullish_support=bullish_support,
        total_score=total_score,
        classification=classification,
        dvn_playbook=playbook,
        market=market,
        articles=articles,
    )


def render_text(summary: SentimentSummary) -> None:
    print("==============================================")
    print("🌍 2025 원유 시장 기본 배경")
    print("==============================================")
    print(MACRO_CONTEXT)
    print()

    print("==============================================")
    print("📥 수집 요약")
    print("==============================================")
    print(f"✅ 수집 성공한 기사 건수: {len(summary.articles)}")
    for article in summary.articles:
        print(f" - {article.source}: {article.title}")
    if summary.articles:
        print()

    print("==============================================")
    print("📊 뉴스 기반 신호 요약")
    print("==============================================")
    print(f"- 공급 과잉/해제 시그널: {summary.supply_pressure} 건 (유가 하방)")
    print(f"- 수요 둔화 시그널: {summary.demand_pressure} 건 (유가 하방)")
    print(f"- 달러 강세/금리 압력: {summary.usd_pressure} 건 (유가 하방)")
    print(f"- 공급 차질·수요 회복 시그널: {summary.bullish_support} 건 (유가 상방)")
    print()
    print(f"➡️  종합 점수: {summary.total_score} → {summary.classification}")
    print()

    print("==============================================")
    print("🛢  시장 스냅샷")
    print("==============================================")
    dxy = summary.market.dxy
    wti = summary.market.wti
    print(f"- DXY (달러 인덱스): {dxy if dxy is not None else 'N/A'}")
    print(f"- WTI 근월물: {wti if wti is not None else 'N/A'}")
    if dxy and dxy >= 102:
        print("  ⚠️  강달러 구간 → 유가에는 부담 요인")
    if wti and wti <= 70:
        print("  ⚠️  70달러 이하 → 추가 하락 시 셰일 버짓 체크")
    print()

    print("==============================================")
    print("📈 DVN 스윙 전략 가이드")
    print("==============================================")
    print(summary.dvn_playbook)
    print()

    if summary.articles:
        print("==============================================")
        print("📰 참고 기사 (상세 득표)")
        print("==============================================")
        for idx, article in enumerate(summary.articles, 1):
            print(f"[{idx}] {article.source} | {article.title}")
            print(f"    URL          : {article.url}")
            print(f"    공급↑(하방)  : {article.supply_bear}")
            print(f"    수요↓(하방)  : {article.demand_bear}")
            print(f"    달러/금리↑   : {article.usd_bear}")
            print(f"    공급차질(상방): {article.supply_bull}")
            print(f"    수요회복(상방): {article.demand_bull}")
            print()
    else:
        print("기사에서 유의미한 키워드가 포착되지 않았습니다. 추가 수집원을 확인하세요.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crude oil sentiment analyzer tuned for DVN actions."
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=30,
        help="분석할 기사 최대 건수 (기본 30)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="결과를 JSON으로 출력 (로깅/자동화 용)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_articles = max(5, args.max_articles)

    articles = crawl_news(max_articles)
    market = build_market_snapshot()
    summary = summarize(articles, market)

    if args.json:
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    else:
        render_text(summary)


if __name__ == "__main__":
    main()

