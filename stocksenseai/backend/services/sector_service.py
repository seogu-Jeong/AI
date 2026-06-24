"""업종(섹터) 히트맵 서비스.

Naver Finance 업종별 시세 페이지에서 HTML 스크래핑으로 데이터를 수집한다.
KRX 로그인·API 키 불필요 (market_index_service 와 동일 방식).

URL: https://finance.naver.com/sise/sise_group.nhn?type=upjong
컬럼 순서: 업종명, 등락률, 종목수(전체), 상승, 보합, 하락
"""
from __future__ import annotations

import asyncio
import json
import re

import httpx

from core.redis_client import get_redis
from services.market_service import _is_market_open, _last_trading_day

_CACHE_KEY = "sector_heatmap:v3"
_TTL_OPEN   = 300
_TTL_CLOSED = 86400

_NAVER_URL = "https://finance.naver.com/sise/sise_group.nhn?type=upjong"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9",
}

# <tr> 블록에서 업종 데이터를 추출하는 패턴
# 실제 HTML 구조:
#   <a href="...no=NO">NAME</a>
#   <span class="tah p11 red01/dn1/black03">\n+5.12%\n</span>
#   <td class="number">172</td>  ← total
#   <td class="number">38</td>   ← up
#   <td class="number">4</td>    ← flat
#   <td class="number">130</td>  ← down
_ROW_RE = re.compile(
    r'no=(\d+)[^>]*>([^<]+)</a>'                       # group 1=no, 2=name
    r'.*?'
    r'class="tah p11[^"]*">\s*([-+]?\d+\.?\d*)\s*%'   # group 3=change_pct
    r'.*?'
    r'class="number">(\d+)</td>'                        # group 4=total
    r'.*?'
    r'class="number">(\d+)</td>'                        # group 5=up
    r'.*?'
    r'class="number">(\d+)</td>'                        # group 6=flat
    r'.*?'
    r'class="number">(\d+)</td>',                       # group 7=down
    re.DOTALL,
)

# 일부 업종은 너무 세분화돼 있어 주요 업종만 추출.
# 목록이 없으면 전체 반환.
_MAJOR_SECTORS = {
    "반도체와반도체장비", "전자장비와기기", "컴퓨터와주변기기",
    "자동차와부품", "화학", "의약품", "제약",
    "은행", "증권", "보험", "생명보험", "손해보험",
    "소프트웨어", "IT서비스", "정보기술서비스",
    "철강", "금속과광업",
    "건설", "건설자재",
    "항공사", "해운사", "운수", "육상운송",
    "유통", "식품과음료", "음식료품",
    "통신서비스", "미디어와엔터테인먼트",
    "부동산", "전기가스및수도",
}


def _fetch_sectors_html() -> list[dict]:
    """동기 HTTP 호출 → asyncio.to_thread 에서 실행."""
    with httpx.Client(timeout=15.0, headers=_HEADERS, follow_redirects=True) as client:
        resp = client.get(_NAVER_URL)
        resp.raise_for_status()

    # Naver 업종 페이지는 EUC-KR 인코딩 — resp.text 는 잘못 디코딩될 수 있음
    try:
        content = resp.content.decode("euc-kr", errors="replace")
    except Exception:
        content = resp.content.decode("utf-8", errors="ignore")

    sectors = []
    for m in _ROW_RE.finditer(content):
        name       = m.group(2).strip()
        change_pct = float(m.group(3))
        total      = int(m.group(4))
        up         = int(m.group(5))
        flat       = int(m.group(6))
        down       = int(m.group(7))

        # "기타" 카테고리 및 주요 업종 외 세분화 항목 제외
        if name == "기타":
            continue
        if _MAJOR_SECTORS and name not in _MAJOR_SECTORS:
            continue

        sectors.append({
            "sector":       name,
            "change_pct":   round(change_pct, 2),
            "total_mktcap": total,      # 종목수를 크기 proxy로 사용
            "up_count":     up,
            "down_count":   down,
            "flat_count":   flat,
            "total_stocks": total,
            "top_stocks":   [],
        })

    # 변화율 내림차순 정렬
    sectors.sort(key=lambda x: x["change_pct"], reverse=True)
    return sectors


async def get_sector_heatmap() -> dict:
    """업종별 등락률 반환 (캐시 포함)."""
    redis = await get_redis()
    cached = await redis.get(_CACHE_KEY)
    if cached:
        return json.loads(cached)

    date_str = _last_trading_day()

    try:
        sectors = await asyncio.to_thread(_fetch_sectors_html)
    except Exception:
        sectors = []

    result = {
        "date":      date_str,
        "available": len(sectors) > 0,
        "sectors":   sectors,
    }

    if len(sectors) == 0:
        # 스크래핑 실패 시 5분 TTL로 짧게 캐싱해 재시도 가능하게
        await redis.setex(_CACHE_KEY, 300, json.dumps(result))
    else:
        ttl = _TTL_OPEN if _is_market_open() else _TTL_CLOSED
        await redis.setex(_CACHE_KEY, ttl, json.dumps(result))
    return result
