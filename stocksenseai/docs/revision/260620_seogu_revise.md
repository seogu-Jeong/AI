# seogu-Jeong 수정 사항 기록 (2026-06-20)

## 작업 범위

이 세션에서 진행한 전체 작업 내역입니다.

---

## 1. 자동매매 오류 메시지 수정 (`fix`)

**커밋:** `44a3723` fix: 자동매매 실행 중 수동 실행 시 잘못된 오류 메시지 수정

**문제:** 자동매매가 이미 실행 중일 때 "지금 실행"을 누르면 "자동매매를 먼저 활성화해 주세요"라는 잘못된 메시지 표시.

**원인:** 프론트엔드에서 `skipped: true` 응답의 `reason` 필드를 구분하지 않고 항상 동일한 오류 메시지 표시.

**수정:** `AutoTradePanel.tsx` — `reason` 필드 분기 처리
- `already_running` → 노란색 안내 ("이미 실행 중입니다")
- `not_enabled` → 빨간색 오류 ("자동매매를 먼저 활성화해 주세요")

---

## 2. 모의매매 초기화 버튼 + 실거래 KIS 예수금 연동 (`feat`)

**커밋:** `cea88cf` feat: 모의매매 초기화 버튼 + 실거래 KIS 예수금 연동

### 2-1. 모의매매 초기화

**파일:** `backend/services/auto_trade_service.py`, `backend/api/routes/auto_trade.py`, `frontend/src/components/AutoTrade/AutoTradePanel.tsx`

- `reset_paper_data(user_id, db)` 함수 추가 — Paper 포트폴리오·자동매매 로그 전체 삭제 후 config `enabled=False`
- `POST /auto-trade/reset` 엔드포인트 추가 — 모의매매 모드에서만 허용 (실거래 모드 시 400)
- 프론트엔드 — 모의 모드 + 비활성화 상태에서만 🗑️ 초기화 버튼 노출, 확인 후 실행

### 2-2. 실거래 KIS 예수금 자동 연동

**파일:** `backend/services/auto_trade_service.py`

- `_execute_real_order()` 함수 추가 — `kis_service.place_order()` 호출 후 `AutoTradeLog` 기록
- `run_cycle()` 실거래 모드 처리 추가:
  - 보유종목: `get_balance_full(user)` → KIS 계좌 실제 보유 종목
  - 예수금: `get_balance(user)["cash"]` → KIS 실제 예수금 자동 적용
  - 매수/매도: `_execute_real_order()` → 실제 KIS 주문 실행
- 프론트엔드 — 실거래 모드 시 예산 입력란 대신 "KIS 예수금 자동 적용" 안내 표시

---

## 3. 앱 전체 리뷰 및 버그 수정 (`fix`)

**커밋:** `06c1d65` fix: 자동매매 스캔 RSI 표시 오류 + MAJOR_50 종목명 누락 수정

### 3-1. 자동매매 AI 분석 패널 RSI 전부 `-` 표시

**파일:** `backend/services/ai_service.py`

**원인:** `calculate_signal()` 반환값에 `indicators` 키가 없어 `scan_stocks()`에서 `rsi_14` 조회 시 항상 `0.0` 반환.

**수정:** `calculate_signal()` 결과 dict에 `"indicators": indicators` 추가.

```python
# 수정 전
result = { "code": code, "signal": signal, ... }

# 수정 후
result = { "code": code, "signal": signal, ..., "indicators": indicators }
```

### 3-2. 자동매매 MAJOR_50 종목명 코드로 표시

**파일:** `backend/ml/stock_names.json`

**원인:** `_MAJOR_50` 리스트에 포함된 6개 종목이 `stock_names.json`에 누락되어 코드가 그대로 표시됨.

**수정:** 6개 종목명 추가

| 코드 | 종목명 |
|---|---|
| 402340 | SK스퀘어 |
| 352820 | 하이브 |
| 316140 | 우리금융지주 |
| 323410 | 카카오뱅크 |
| 293490 | 카카오페이 |
| 259960 | 크래프톤 |

---

## 4. 앱 전체 탭 검사 결과

| 탭 | 상태 | 비고 |
|---|---|---|
| 차트 | ✅ 정상 | 캔들·RSI·MACD·종목 전환 |
| AI | ✅ 정상 | 시그널·점수 분해·멀티타임프레임·재무 평가 |
| 추천 | ✅ 정상 | AI 랭킹 리스트 |
| 시장 | ✅ 정상 | KOSPI/KOSDAQ 지수·업종 히트맵 |
| 시뮬 | ✅ 정상 | 시뮬레이션 결과 출력 |
| 포트폴리오 | ✅ 정상 | 빈 상태 메시지 정상 |
| 스크리너 | ✅ 정상 | 전체 종목 스캔 후 필터 적용 |
| 백테스트 | ✅ 정상 | 결과(수익률·MDD·샤프) 출력 |
| 자동매매 | ✅ 정상 | 종목명·RSI 수정 후 정상 동작 |

- TypeScript 컴파일 오류: 없음
- ESLint 오류: 없음
- HTTP 오류: `/auth/refresh` 401 (페이지 최초 로드 시 세션 확인 — 정상 동작)

---

## 브랜치 / 머지 이력

```
seogu-Jeong → dev  (fast-forward, 2026-06-20)
  06c1d65  fix: 자동매매 스캔 RSI 표시 오류 + MAJOR_50 종목명 누락 수정
  cea88cf  feat: 모의매매 초기화 버튼 + 실거래 KIS 예수금 연동
  44a3723  fix: 자동매매 실행 중 수동 실행 시 잘못된 오류 메시지 수정
```
