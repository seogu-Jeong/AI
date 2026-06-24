# Phase 3 Review

검토일: 2026-06-04

## 결론

Phase 3의 파일 구조, ML 모듈, AI API 8개 엔드포인트, Phase 3+4 DB 모델/마이그레이션, Celery 태스크, APScheduler 등록은 전반적으로 구현되어 있습니다. `backend/tests`의 ML/패턴/AI 서비스 유닛 테스트는 `23 passed`로 확인했고, 승인된 환경에서 Alembic `upgrade head`도 정상 적용됐습니다.

다만 `docs/superpowers/plans/2026-06-03-phase3-ai.md`의 목표인 “LSTM 기반 5일 주가 예측, 기술적 지표 시그널, 캔들 패턴 인식 API 8개 구현”을 운영 가능한 수준으로 봤을 때는 아직 미완성에 가깝습니다. 특히 실제 학습 가중치가 없고, 통합 테스트는 현재 기본 Python 환경에서 재현되지 않으며, 자동 갱신 태스크는 DB 기록 없이 계산만 수행합니다.

## 수정 필요

1. 통합 테스트가 현재 기본 환경에서 재현되지 않습니다.
   - 위치: `pytest.ini:5`, `backend/requirements.txt:12`, `backend/requirements.txt:29`
   - `pytest -q`와 `python3 -m pytest -q` 모두 `httpx` 미설치로 실패했습니다. Python 3.12로도 `apscheduler` 미설치 때문에 실패했습니다.
   - `backend/tests`는 별도 실행 시 `23 passed`지만, 루트 통합 테스트의 `65 passed`는 현재 환경에서 확인하지 못했습니다.
   - 권고: 가상환경을 고정하고 `python -m pip install -r backend/requirements.txt` 후 테스트를 재실행하세요. README에 Python 버전과 테스트 명령을 분리해서 적으세요.

2. `backend/tests`가 기본 `pytest` 실행 대상이 아닙니다.
   - 위치: `pytest.ini:5`, `backend/tests/test_ml.py:1`
   - `testpaths = tests`라서 루트에서 `pytest`만 실행하면 ML/패턴/AI 서비스 유닛 테스트 23개가 빠집니다.
   - 권고: `testpaths = tests backend/tests`로 바꾸거나 Phase 3 테스트를 `tests/` 아래로 옮기세요.

3. 실제 LSTM 가중치가 없습니다.
   - 위치: `backend/ml/weights/.gitkeep`
   - `backend/ml/weights`에는 `.gitkeep`만 있습니다. 따라서 `/ai/{code}/predict`는 기본적으로 빈 예측을 반환하고, `/ai/{code}/signal`은 기술적 지표 100% fallback입니다.
   - 권고: 최소 대표 종목 1~3개라도 학습 산출물을 만들거나, progress에 “모델 구조/추론 인터페이스 구현, 실가중치 미포함”으로 명확히 낮춰 표기하세요.

4. Celery AI 갱신 태스크가 히스토리를 저장하지 않습니다.
   - 위치: `backend/tasks/ai_tasks.py:12`, `backend/services/ai_service.py:119`
   - `refresh_ai_signals()`는 `calculate_signal(code)`를 DB 세션 없이 호출합니다. `calculate_signal`은 `db`가 있을 때만 `ai_signals_history`에 insert합니다.
   - 결과적으로 “장 종료 후 시그널 갱신”은 캐시/DB 갱신 없이 계산만 하고 버립니다.
   - 권고: 태스크에서 `AsyncSessionLocal`을 열어 `calculate_signal(code, db)`를 호출하고, 캐시도 명시적으로 갱신하세요.

5. `ai_signals_history`에 예측값과 confidence가 저장되지 않습니다.
   - 위치: `backend/services/ai_service.py:119`, `backend/models/ai_signal.py:20`
   - 모델에는 `predicted_prices`, `confidence`가 있지만 insert 시 넣지 않습니다.
   - 권고: `get_prediction` 결과 또는 `predict_scenarios` 결과를 signal 저장 시 함께 기록하세요. confidence 계산이 없다면 필드를 nullable로 두더라도 응답/문서에서 미구현으로 표시하세요.

6. `ai_signals_history` 복합 인덱스가 계획과 다릅니다.
   - 위치: `db/migrations/versions/a1b2c3d4e5f6_add_ai_signals_history.py:41`
   - 계획은 `idx_signals_code_date(stock_code, recorded_at DESC)`인데 현재는 `stock_code`, `recorded_at` 단일 인덱스 2개입니다.
   - `/ai/signals/history/{code}`는 code 필터 + recorded_at 정렬을 하므로 복합 인덱스가 더 적합합니다.
   - 권고: 새 migration으로 `idx_signals_code_date`를 추가하세요.

7. 패턴 탐지 테스트가 실제 탐지를 보장하지 않습니다.
   - 위치: `backend/services/pattern_service.py:35`, `backend/tests/test_patterns.py:20`
   - 테스트는 빈 리스트여도 통과합니다. 실제 캔들 패턴을 의도적으로 만든 fixture가 없습니다.
   - 권고: hammer, engulfing 등 하나 이상 확정 패턴 fixture를 만들고 결과가 non-empty인지 검증하세요.

8. `ai_service.py`의 broad ImportError 처리로 설정/의존성 오류가 숨겨질 수 있습니다.
   - 위치: `backend/services/ai_service.py:10`
   - import 실패를 전부 `pass` 처리하면 모듈 import는 되지만 런타임에서 `get_redis`, `get_ohlcv_cached`, `AISignalHistory` 같은 이름이 없어 NameError가 납니다.
   - 권고: 순수 함수 테스트를 위해 필요한 경우 함수 분리나 명시적 lazy import를 쓰고, broad `except ImportError: pass`는 제거하세요.

9. 학습 데이터 scaler가 train/validation/test split 전에 전체 데이터로 fit됩니다.
   - 위치: `backend/ml/train.py:45`
   - `_make_sequences`에서 전체 `feat_df`로 scaler를 fit한 뒤 split합니다. validation/test 정보가 scaler에 누출됩니다.
   - 권고: split 기준을 먼저 정하고 train 구간으로만 scaler를 fit한 뒤 val/test에 transform하세요.

10. `get_prediction` 응답이 PRD/TRD 스키마보다 빈약합니다.
    - 위치: `backend/services/ai_service.py:150`
    - 계획/문서의 예측 응답은 name, as_of, confidence, signal, signal_breakdown, indicators까지 포함하지만 현재는 `code`, `current_price`, `prediction`, `lstm_available`만 반환합니다.
    - 권고: 프론트 연동 전에 응답 스키마를 문서와 맞추거나 문서를 현재 API에 맞게 조정하세요.

## 권고 사항

1. 기술적 지표 점수 공식은 재검토가 필요합니다.
   - 위치: `backend/services/ai_service.py:34`
   - 현재 RSI 80도 높은 bullish score, RSI 20도 높은 bullish score가 됩니다. 과매수/과매도 해석이 섞여 있어 BUY/HOLD/SELL 의미가 불명확합니다.
   - 권고: 추세 추종인지 반전 매매인지 정책을 정하고 RSI 점수 함수를 분리 테스트하세요.

2. `calculate_signal`이 OHLCV를 중복 조회합니다.
   - 위치: `backend/services/ai_service.py:86`, `backend/services/ai_service.py:91`
   - `calculate_signal`에서 raw를 가져온 뒤 `get_indicators`가 다시 `get_ohlcv_cached`를 호출합니다.
   - 권고: 이미 만든 DataFrame/feature를 재사용하세요.

3. `/ai/{code}/patterns`는 데이터 부족도 200 빈 배열로 반환합니다.
   - 위치: `backend/api/routes/ai.py:82`
   - `/ai/{code}/indicators`는 데이터 부족 시 404인데 patterns는 항상 200입니다.
   - 권고: API별 빈 결과 정책을 일관되게 정하세요.

4. `get_top_picks`는 weights가 없으면 항상 빈 결과입니다.
   - 위치: `backend/services/ai_service.py:236`
   - 데모/프론트 시연에서는 `top-picks`가 핵심 화면인데 현재는 학습 가중치 없으면 빈 배열입니다.
   - 권고: fallback으로 top100 코드 대상 기술적 지표 기반 picks를 계산하거나 mock/demo 모드를 명시하세요.

5. `top100_codes.txt`는 100개가 아닙니다.
   - 위치: `backend/ml/top100_codes.txt`
   - 실제 줄 수는 98개입니다.
   - 권고: 파일명/문서와 맞추거나 누락 종목을 채우세요.

6. pandas-ta 실행 재현성을 확인하세요.
   - 위치: `backend/requirements.txt:27`
   - 현재 Python 3.13 환경에서 직접 `python3 -c "import pandas_ta"` 실행 시 numba cache 관련 오류가 발생했습니다. pytest에서는 backend unit이 통과했지만 실행 방식별 차이가 있습니다.
   - 권고: Python 3.11 또는 3.12로 고정하고 pandas-ta/numba 조합을 lock 하세요.

7. Phase 4 모델까지 Phase 3에서 만들었지만 제약조건이 부족합니다.
   - 위치: `backend/models/portfolio.py:16`, `backend/models/trade.py:18`
   - quantity 양수, mode/order_type/status check constraint 등이 없습니다.
   - 권고: Phase 4 시작 전에 DB 제약을 추가하세요.

8. README가 여전히 실행 문서 역할을 하지 못합니다.
   - 위치: `README.md:1`
   - Phase 3은 의존성/학습/마이그레이션/테스트 명령이 많아 문서 없이는 재현이 어렵습니다.
   - 권고: venv 생성, requirements 설치, alembic upgrade, unit/integration test, 학습 명령을 추가하세요.

## 확인한 점

- `pytest -q backend/tests`: 23 passed, 1 warning
- `alembic upgrade head`: 승인된 로컬 DB 환경에서 성공
- AI API 라우터 8개 엔드포인트 등록 확인
- AI rate limit 20/min 적용 확인
- APScheduler 15:35 Asia/Seoul 등록 확인
- `backend/ml/weights`: `.gitkeep`만 존재, 학습 가중치 없음
- `pytest -q`: 현재 기본 환경에서 `httpx` 미설치로 실패
- Python 3.12 직접 실행: `apscheduler` 미설치로 통합 테스트 수집 실패

