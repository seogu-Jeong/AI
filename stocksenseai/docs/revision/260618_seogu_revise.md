# 리비전 노트 — 2026-06-18 (정석우)

## 변경 개요
재무 종합 판단 기능 추가, 차트 기간 확장, PR 템플릿 도입, 환경변수 누수 수정

---

## 1. `backend/core/config.py` — `extra: ignore` 추가
- **왜**: `frontend/.env`의 `VITE_API_BASE` 환경변수가 pydantic-settings 2.x 에서
  예상치 못한 필드로 인식돼 `ValidationError` 발생. `"extra": "ignore"` 추가로 수정.
- **영향**: 로컬 개발 환경(`dev_startup.py`)에서 `frontend/.env` 로드 시 오류 방지.

---

## 2. `backend/services/fundamental_service.py` — ROE 지표 추가
- 네이버 금융 통합 API에서 `roe` 필드를 추출.
- ROE가 있을 때 점수 가중치 변경:
  - 기존: PER 45%, PBR 30%, EPS 15%, 배당 10%
  - 변경: PER 35%, PBR 25%, ROE 20%, EPS 12%, 배당 8%
- `metrics` 응답에 `roe` 필드 추가.

---

## 3. `backend/services/comprehensive_service.py` — 신규 종합 판단 서비스
- AI 시그널(40%) + 재무 평가(35%) + 시장 지수(25%) → 0-10점 종합 점수.
- 등급: 강력매수(≥8) / 매수(≥6.5) / 중립(≥5) / 매도주의(≥3.5) / 매도.
- Redis 캐시 5분.
- 각 컴포넌트 점수, 근거 텍스트, 지수 데이터 포함 반환.

---

## 4. `backend/api/routes/analysis.py` — `/analysis/comprehensive/{code}` 엔드포인트 추가
- Rate limit: 30/minute.
- 기존 `/fundamental`, `/recommendations`, `/indices` 엔드포인트 유지.

---

## 5. 차트 기간 확장
- `backend/services/market_service.py`: `5y: 1825` 기간 추가.
- `backend/api/routes/stocks.py`: period 파라미터 패턴에 `3y`, `5y` 허용.
- `frontend/src/components/MainPanel/ChartTab/ChartTab.tsx`:
  - `period` state 추가 (기본값: `1y`).
  - 기간 선택 UI 바 추가 (1개월 / 3개월 / 1년 / 2년 / 3년 / 5년).
  - 분봉 모드에서는 기간 바 숨김.

---

## 6. `frontend/src/components/Analysis/ComprehensivePanel.tsx` — 신규 종합 판단 패널
- 종합 점수 대형 표시, 등급 뱃지.
- AI / 재무 / 시장 컴포넌트 점수 바 (색상: 초록/노랑/빨강).
- 근거 목록.
- `AITab.tsx`에 추가 (FundamentalPanel, RecommendationPanel 상단).

---

## 7. `.github/PULL_REQUEST_TEMPLATE.md` — PR 템플릿 신규
- 변경 유형 체크리스트, 리비전 문서 링크, 개발 확인 항목.
- 앞으로 PR 작성 시 자동 적용.

---

## 참고
- 시세/AI 계산 로직 변경 없음 (ai_service.py 그대로).
- 네이버 모바일 API 엔드포인트 변경 없음.
- 기존 FundamentalPanel / RecommendationPanel 레이아웃 유지, ComprehensivePanel 상단 추가.

---
---

# 추가 리비전 노트 — 2026-06-18 (2차) (정석우)

## 변경 개요
스크리너 성능 개선 + SELL 경고, 급등 감지, 52주 신고가, 포트폴리오 백테스트,
외국인·기관 매매동향, 업종 히트맵(시장 탭) 추가 및 섹터 데이터 오류 수정

---

## 8. 스크리너 공유 캐시 성능 개선 + SELL 경고 표시

### `backend/services/stock_data_service.py` (캐시 v4로 업그레이드)
- 100개 종목 공유 캐시 키를 `stock_data_cache:v4`로 변경.
- 동시 수집 세마포어 20, TTL 5분 유지.

### `backend/services/recommend_service.py` (캐시 v4)
- `cache_key = f"recommendations_v4:{limit}"`.
- BUY 추천 목록 외 **SELL 경고 목록** (`sell_warnings`) 추가 — 시그널 SELL + AI 점수 하위 종목.

### `frontend/src/components/Analysis/RecommendationPanel.tsx`
- BUY / SELL 탭 전환 UI 추가.
- SELL 경고 종목 카드 (빨간 테두리) 표시.

### `frontend/src/components/MainPanel/ScreenerTab/ScreenerTab.tsx`
- 투자자 동향 필터 섹션 추가 (외국인 순매수 / 기관 순매수 체크박스).

---

## 9. 급등 감지 기능

### `backend/services/stock_data_service.py`
- `_calc_surge()`: 가격 변화율 +5% 이상 AND 거래량 비율 2배 이상 AND MA5 돌파 복합 조건.
- `_fetch_one()`: 3개월 OHLCV 추가 수집, 급등 감지 필드 반환.
- 신규 필드: `price_change_pct`, `volume_ratio`, `surge_detected`, `surge_reason`.

### `backend/services/recommend_service.py`
- `surge_alerts` 리스트 추가 (최대 10개, `price_change_pct` 내림차순).
- 응답에 `surge_count` 포함.

### `frontend/src/components/Analysis/RecommendationPanel.tsx`
- 급등 알림 섹션 추가 (추천 탭 상단, 주황색 카드).

---

## 10. 52주 신고가 브레이크아웃 감지

### `backend/services/stock_data_service.py`
- `_calc_52w()`: 1년 OHLCV → 52주 최고가 계산, 정확 돌파(high_breakout) / 근접(≥97%, near_high) 구분.
- 신규 필드: `w52_high`, `w52_low`, `high_breakout`, `near_high`, `w52_from_high_pct`.

### `backend/services/recommend_service.py`
- `high_breakouts` 리스트 추가 (최대 10개, 정확 돌파 우선 → `w52_from_high_pct` 정렬).
- 응답에 `high_breakout_count` 포함.

### `frontend/src/components/Analysis/RecommendationPanel.tsx`
- 52주 신고가 섹션 추가 (노란색 카드, 추천 탭 내).

---

## 11. 포트폴리오 백테스트

### `backend/services/backtest_service.py`
- `PortfolioStock` 데이터클래스 추가 (code, name, weight_pct).
- `run_portfolio_backtest()` 추가: 종목별 독립 백테스트 실행 → 비중 적용 → 일별 자산곡선 합산.
- 반환: `portfolio_metrics`(승률·샤프·MDD), `per_stock`(종목별 수익률), `equity_curve`.

### `backend/api/routes/backtest.py`
- `PortfolioStockItem`, `PortfolioBacktestRequest` Pydantic 모델 추가.
- `POST /backtest/portfolio-run` 엔드포인트 추가 (rate limit 3/min, 비중 합 100% 검증).

### `frontend/src/components/MainPanel/BacktestTab/BacktestTab.tsx` (전면 재작성)
- 단일 종목 / 포트폴리오 모드 토글.
- 포트폴리오 모드: 종목 검색(디바운스) + 관심종목 불러오기 + 종목별 비중 입력.
- 결과: 4개 지표 카드 + Recharts LineChart 자산 추이 + 종목별 수익률 테이블.

---

## 12. 외국인·기관 매매동향

### `backend/services/investor_service.py` (신규)
- pykrx `get_market_trading_value_by_date(fromdate, todate, code, on="순매수")` 호출.
- 컬럼: 외국인합계, 기관합계, 개인.
- 5일 누적 순매수 요약 + 일별 데이터 반환.
- 캐시 키: `investor_trend:{code}:{days}`.
- **주의**: pykrx KRX 로그인(`KRX_ID`/`KRX_PW`) 없으면 빈 데이터 반환 (로컬 개발 한계).

### `backend/api/routes/analysis.py`
- `GET /analysis/investor/{code}` 엔드포인트 추가 (rate limit 60/min).
- `GET /analysis/screener` 파라미터에 `foreign_net_buy`, `institution_net_buy` bool 추가.

### `backend/services/screener_service.py`
- `_apply_investor_filter()` 추가 (세마포어 10, 종목별 `get_investor_trend` 호출).
- `run_screener()` 파라미터에 `foreign_net_buy`, `institution_net_buy` 추가.

### `frontend/src/components/Analysis/InvestorPanel.tsx` (신규)
- 외국인/기관 탭 전환.
- 5일 누적 순매수 배지, BarChart (기준선 0) 일별 순매수 시각화.

### `frontend/src/components/MainPanel/AITab/AITab.tsx`
- `<InvestorPanel />` 추가 (FundamentalPanel과 RecommendationPanel 사이).

---

## 13. 업종 히트맵 — 시장 탭 신설

### `backend/services/sector_service.py` (신규)
- **초기 구현**: pykrx `get_market_sector_classifications` 사용 → KRX 로그인 없으면 빈 데이터.
- **오류 수정 (동일 세션)**: Naver Finance `finance.naver.com/sise/sise_group.nhn?type=upjong`
  HTML 스크래핑으로 교체. KRX 로그인·API 키 불필요.
  - 78개 업종 파싱 (업종명, 등락률, 종목수, 상승/보합/하락 수).
  - "기타" 카테고리 제외.
  - 캐시 키 `sector_heatmap:v3`, 장중 5분 / 장외 24시간 TTL.

### `backend/api/routes/analysis.py`
- `GET /analysis/sector` 엔드포인트 추가 (rate limit 10/min).

### `frontend/src/components/MainPanel/MarketTab/MarketTab.tsx` (신규)
- 지수 요약 카드 3개 (KOSPI / KOSDAQ / KOSPI200), Naver Finance 연동.
- Recharts `Treemap` 업종 히트맵: 크기=종목수, 색상=등락률.
- 업종 클릭 시 상세 패널: 등락률 + 종목수·상승·보합·하락 카운트.

### `frontend/src/components/MainPanel/MainPanel.tsx`
- `{ id: 'market', label: '시장' }` 탭 추가.

### `frontend/src/types/index.ts`
- `TabId`에 `'market'` 추가.

---

## 참고 (2차)
- pykrx KRX 로그인 필요 기능(투자자 매매동향 상세, 업종별 종목 목록)은 로컬 개발 환경에서
  데이터 없이 graceful fallback 처리됨. 운영 환경에서 `KRX_ID`/`KRX_PW` 설정 시 활성화.
- 섹터 히트맵의 업종별 상위 종목은 현재 빈 배열 반환 (KRX 로그인 필요).
- 스크리너 투자자 필터도 로컬에서는 필터 효과 없음 (동일 이유).
