# 2026-06-19 seogu-Jeong 수정 내역

브랜치: `seogu-Jeong` → `dev` 머지

---

## 1. 차트 종목 전환 미갱신 버그 수정

### 증상
종목 A를 보다가 종목 B를 클릭해도 차트가 갱신되지 않는 문제.

### 원인
- `ChartTab.tsx`: 종목 전환 시 `candles`/`patterns`/`prediction` 상태가 초기화되지 않아 새 API 응답이 오기 전까지 이전 종목 데이터가 그대로 표시됨.
- `CandleChart.tsx`: `candles.length === 0` 일 때 early return 처리 → `series.setData([])` 를 호출하지 않아 이전 캔들이 화면에 잔존.
- `ChartTab.tsx`: patterns/predict `.then()` 콜백에 `cancelled` 플래그 체크 누락 → 빠른 종목 전환 시 이전 종목 응답이 현재 화면을 덮어쓰는 race condition.

### 수정
**`frontend/src/components/MainPanel/ChartTab/ChartTab.tsx`**
- 종목 전환 microtask에 `setCandles([])`, `setPatterns([])`, `setPrediction(null)`, `setLstmAvailable(false)` 추가.
- `patterns` fetch `.then()` 최상단에 `if (cancelled) return` 추가.
- `predict` fetch `.then()` 최상단에 `if (cancelled) return` 추가.

**`frontend/src/components/MainPanel/ChartTab/CandleChart.tsx`**
- `candles.length === 0` 시 `seriesRef.current.setData([])` 호출 후 return (이전 차트 클리어).

---

## 2. 보안 취약점 수정

### 2-1. Pre-Authentication 계정 탈취 (auth.py)
**파일:** `backend/api/routes/auth.py`  
**심각도:** Critical

공격자가 피해자 이메일로 로컬 비밀번호 계정을 미리 생성 후, 피해자가 Google OAuth로 로그인하면 기존 `password_hash` 가 무효화되지 않아 공격자가 설정한 비밀번호로 계정을 영구 탈취 가능.

**수정:** Google OAuth 기존 계정 연동 시 `user.password_hash = None` 추가.

### 2-2. 이메일 발송 실패 교착상태 (auth.py)
**파일:** `backend/api/routes/auth.py`

SendGrid 장애 시 `send_verification_email()` 예외로 502 발생. DB에는 미인증 유저가 저장된 상태라 동일 이메일 재가입 불가(409).

**수정:** `send_verification_email` try/except 처리 → 실패 시 `email_failed: True` 응답으로 프론트에서 재발송 안내 가능.

---

## 3. 런타임 버그 수정

### 3-1. asyncio.create_task GC 소멸 (analysis.py)
**파일:** `backend/api/routes/analysis.py`

`asyncio.create_task()` 반환값을 저장하지 않으면 Python GC가 태스크를 수거해 warmup 작업이 중단될 수 있음.

**수정:** 모듈 레벨 `_bg_tasks: set` 에 태스크 강한 참조 보관, `add_done_callback`으로 완료 시 제거.

### 3-2. Anomaly 슬라이딩 윈도우 off-by-one (anomaly_service.py)
**파일:** `backend/services/anomaly_service.py`

- 마지막 윈도우 누락: `range(len(returns) - _WINDOW)` → `range(len(returns) - _WINDOW + 1)`
- 날짜 매핑 오류: `date_idx = int(idx) + _WINDOW + 1` → `int(idx) + _WINDOW` (하루 미래 날짜에 이상점수 매핑되던 버그)
- 점수 리스트도 동일 수정: `i + _WINDOW + 1` → `i + _WINDOW`

### 3-3. Autoencoder CPU 대경합 (anomaly_service.py)
**파일:** `backend/services/anomaly_service.py`

동시 요청 시 PyTorch가 모든 CPU 코어를 점유해 전체 API 레이턴시 폭발.

**수정:** `torch.set_num_threads(1)` 추가.

---

## 4. 캐싱 정책 개선

### 4-1. 실패 결과 장기 캐싱 방지
**파일:** `backend/services/anomaly_service.py`, `trendline_service.py`, `support_resistance_service.py`, `sector_service.py`

스크래핑/학습 실패 시 빈 결과가 1시간(또는 24시간) 동안 캐싱되어 서비스 재개 후에도 오류 상태 유지.

**수정:** `available: False` / 빈 결과 시 TTL을 300초(5분)로 단축.

---

## 5. sector_service 경고 수정

**파일:** `backend/services/sector_service.py`

- `_MAJOR_SECTORS` 필터가 정의만 되고 미사용 → `_fetch_sectors_html()` 내 실제 필터링 로직 추가.
- `resp.text` → `resp.content.decode("euc-kr", errors="replace")` (EUC-KR 한글 인코딩 안전 처리).

---

## 6. MarketTab async 리팩터 (hwang 변경사항 통합)

**파일:** `frontend/src/components/MainPanel/MarketTab/MarketTab.tsx`

hwang의 `async/await + cancelled` 패턴으로 교체하며 dev warning(`.catch` 로깅) 유지.

---

## 영향 파일 목록

| 파일 | 변경 유형 |
|---|---|
| `frontend/src/components/MainPanel/ChartTab/ChartTab.tsx` | 버그 수정 |
| `frontend/src/components/MainPanel/ChartTab/CandleChart.tsx` | 버그 수정 |
| `frontend/src/components/MainPanel/MarketTab/MarketTab.tsx` | 리팩터 (hwang 통합) |
| `backend/api/routes/auth.py` | 보안 패치 |
| `backend/api/routes/analysis.py` | 버그 수정 |
| `backend/services/anomaly_service.py` | 버그 수정 + 성능 |
| `backend/services/sector_service.py` | 경고 수정 + 캐싱 |
| `backend/services/trendline_service.py` | 캐싱 정책 |
| `backend/services/support_resistance_service.py` | 캐싱 정책 |
| `backend/tasks/ai_tasks.py` | 로깅 개선 |
| `frontend/src/components/MainPanel/ChartTab/TrendlineOverlay.tsx` | 에러 처리 |
| `frontend/src/components/Risk/RiskSettingsModal.tsx` | 경고 로깅 |
