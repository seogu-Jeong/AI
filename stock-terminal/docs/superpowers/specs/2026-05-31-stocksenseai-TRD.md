# TRD — StockSenseAI
**Technical Requirements Document**
버전: 3.0 | 작성일: 2026-05-31 | 작업자: 2명
레퍼런스: TradingView · 토스증권 · KIS Open API WebSocket · TrendSpider · ifibought

---

## 1. 기술 스택

### 1.1 전체 스택 결정표

| 레이어 | 기술 | 버전 | 선택 근거 |
|---|---|---|---|
| Frontend | React + **TypeScript** | 18 / TS 5 | ifibought와 동일 스택, 타입 안전성 |
| 빌드 도구 | Vite | 5 | 기존 환경 유지 |
| 언어 | TypeScript | 5 | ifibought 코드베이스 기반 |
| UI 컴포넌트 | **shadcn/ui** | 최신 | ifibought 동일, Button/Dialog/Badge 등 |
| 차트 (거래) | Lightweight Charts (TradingView) | 4 | 캔들스틱, AI 예측 오버레이 |
| 차트 (시뮬레이터) | **Recharts** | 2.x | ifibought 동일, 비교/수익률 차트 |
| 상태 관리 | Zustand | 4 | 경량, Context보다 성능 우수 |
| HTTP 클라이언트 | Axios | 1.6 | interceptor로 JWT 자동 첨부 |
| 스타일 | **Tailwind CSS v4** | 4 | ifibought 동일, @import "tailwindcss" |
| 아이콘 | **lucide-react** | 최신 | ifibought 동일 (TrendingUp, BarChart2 등) |
| 소셜 로그인 | Google OAuth 2.0 | - | 랜딩 페이지 [Google로 시작하기] |
| Backend | FastAPI | 0.111 | 기존 코드, Python ML 통합 |
| ORM | SQLAlchemy | 2 | async 지원, Alembic 연동 |
| DB | PostgreSQL | 15 | JSONB 지원, 안정성 |
| Cache / Pub-Sub | Redis | 7 | 시세 캐시, WebSocket 메시지 브로커 |
| Task Queue | Celery | 5.3 | AI 추론, 이메일 비동기 처리 |
| 스케줄러 | APScheduler | 3.10 | 시그널 폴링, 주간 리포트 |
| ML | PyTorch | 2.2 | LSTM 모델 |
| 피처 엔지니어링 | pandas-ta | 0.3 | RSI/MACD/BB 계산 |
| 시세 수집 | pykrx | 1.0 | 한국 주식 OHLCV 무료 |
| 거래 API | python-kis | 최신 | KIS Open API 래퍼 |
| 이메일 | SendGrid Python SDK | 6 | 무료 100통/일 |
| 인증 | python-jose + passlib | - | JWT + bcrypt |
| 암호화 | cryptography (AES-256-GCM) | 42 | API 키 암호화 |
| DB 마이그레이션 | Alembic | 1.13 | 스키마 버전 관리 |
| 컨테이너 | Docker + Compose | - | 로컬/서버 환경 일치 |
| 역방향 프록시 | Nginx | 1.25 | 정적 파일, HTTPS, 프록시 |

---

## 2. 전체 시스템 아키텍처

```
                    [Browser / Mobile]
                          |
                    [Nginx Reverse Proxy]
                    /                   \
            [React SPA]           [FastAPI :8000]
                                        |
                    ┌───────────────────┼───────────────────┐
                    |                   |                   |
              [PostgreSQL]          [Redis]           [Celery Worker]
                    |                   |                   |
              영속 데이터          시세 캐시           비동기 작업
           유저/포트폴리오/       WebSocket           AI 추론
             거래 이력            메시지 큐          이메일 발송
                                   |               시그널 폴링
                    ┌──────────────┘
                    |
             [KIS WebSocket]
           실시간 시세/체결
```

### 2.1 요청 흐름

```
사용자 요청
  → Nginx (HTTPS 종료, 정적 파일 서빙)
  → FastAPI 라우터
  → JWT 검증 미들웨어
  → 비즈니스 로직 (서비스 레이어)
  → Redis 캐시 확인
    → 캐시 히트: 즉시 반환
    → 캐시 미스: pykrx / KIS API 호출 → Redis 저장 → 반환
  → Celery 큐 (무거운 작업 비동기 위임)
  → PostgreSQL (영속 데이터 저장)
```

---

## 3. 디렉토리 구조

```
stock-terminal/
│
├── frontend/                                # ifibought 구조 기반, TypeScript
│   ├── public/
│   │   ├── favicon.svg
│   │   └── icons.svg
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/                          # shadcn/ui 컴포넌트 (ifibought 동일)
│   │   │   │   ├── badge.tsx
│   │   │   │   ├── button.tsx
│   │   │   │   ├── dialog.tsx
│   │   │   │   ├── input.tsx
│   │   │   │   ├── label.tsx
│   │   │   │   ├── toast.tsx
│   │   │   │   ├── toaster.tsx
│   │   │   │   ├── tabs.tsx
│   │   │   │   ├── select.tsx
│   │   │   │   └── use-toast.ts
│   │   │   ├── Layout/
│   │   │   │   ├── MainLayout.tsx           # Sidebar + MainPanel + WatchlistPanel
│   │   │   │   ├── Header.tsx               # 모드 토글, 알림, 유저 메뉴, 업데이트 버튼
│   │   │   │   └── MobileTabBar.tsx         # 모바일 하단 탭 (360px~)
│   │   │   ├── Sidebar/                     # ifibought Sidebar 구조 확장
│   │   │   │   ├── Sidebar.tsx              # 종목 그룹 + 검색 + 관심종목 그룹
│   │   │   │   ├── StockGroup.tsx           # 국내주식 / 코스닥 그룹 (더블클릭 → 차트탭)
│   │   │   │   └── StockList.tsx            # 검색 + 종목 목록
│   │   │   ├── MainPanel/                   # ifibought MainPanel 확장
│   │   │   │   ├── MainPanel.tsx            # 탭 라우팅
│   │   │   │   ├── TabBar.tsx               # [차트][AI분석][시뮬레이터][포트폴리오] + 결과N탭
│   │   │   │   ├── ChartTab/
│   │   │   │   │   ├── ChartTab.tsx
│   │   │   │   │   ├── CandlestickChart.tsx     # Lightweight Charts 캔들스틱
│   │   │   │   │   ├── AIPredictionOverlay.tsx  # LSTM 3시나리오 점선
│   │   │   │   │   ├── IndicatorPanel.tsx       # RSI/MACD 서브차트
│   │   │   │   │   ├── PatternBadge.tsx         # 캔들 패턴 아이콘
│   │   │   │   │   └── DrawingToolbar.tsx       # 추세선, 피보나치
│   │   │   │   ├── AITab/
│   │   │   │   │   ├── AITab.tsx
│   │   │   │   │   ├── SignalCard.tsx            # BUY/HOLD/SELL + 점수 분해
│   │   │   │   │   ├── MultiFramePanel.tsx      # 일봉/주봉/월봉 시그널
│   │   │   │   │   └── SimilarPatterns.tsx      # 유사 패턴 히스토리
│   │   │   │   ├── SimulatorTab/               # ifibought ResultTab 기반
│   │   │   │   │   ├── SimulatorTab.tsx
│   │   │   │   │   ├── DateControls.tsx         # 매수일 / 매도일 입력
│   │   │   │   │   ├── LumpSumResultTable.tsx   # 일시불 결과 테이블
│   │   │   │   │   ├── RecurringResultTable.tsx # 적립식 결과 테이블
│   │   │   │   │   ├── LumpSumComparisonChart.tsx   # Recharts 비교 차트
│   │   │   │   │   └── RecurringComparisonChart.tsx # 누적투자 vs 평가액 차트
│   │   │   │   ├── PortfolioTab/
│   │   │   │   │   ├── Holdings.tsx             # 보유 종목 테이블
│   │   │   │   │   ├── PerformanceChart.tsx     # 수익률 히스토리 (Recharts)
│   │   │   │   │   ├── AllocationPie.tsx        # 비중 파이차트 (Recharts)
│   │   │   │   │   └── Metrics.tsx              # MDD, 샤프, 승률 카드
│   │   │   │   ├── ScreenerTab/
│   │   │   │   │   ├── ScreenerFilters.tsx
│   │   │   │   │   └── ScreenerResults.tsx
│   │   │   │   └── BacktestTab/
│   │   │   │       ├── BacktestConfig.tsx
│   │   │   │       └── BacktestResult.tsx
│   │   │   ├── WatchlistPanel/             # ifibought WatchlistPanel 그대로 확장
│   │   │   │   ├── WatchlistPanel.tsx       # 하단 고정 (56px), 종목 태그 + 버튼
│   │   │   │   ├── LumpSumModal.tsx         # 일시불 투자 모달
│   │   │   │   ├── RecurringModal.tsx       # 적립식 투자 모달
│   │   │   │   └── OrderModal.tsx           # 실거래 주문 모달 (추가)
│   │   │   ├── Trade/
│   │   │   │   ├── OrderBook.tsx            # 10단 호가창
│   │   │   │   └── OrderHistory.tsx         # 주문 내역
│   │   │   ├── Risk/
│   │   │   │   └── RiskSettings.tsx
│   │   │   ├── auth/
│   │   │   │   ├── OnboardingModal.tsx      # ifibought 동일 (신규 가입 온보딩)
│   │   │   │   └── ApiKeySetup.tsx          # KIS 키 등록
│   │   │   ├── payment/
│   │   │   │   └── UpgradeScreen.tsx        # ifibought 동일 (결제 유도 화면)
│   │   │   ├── InitialLoadingScreen.tsx     # ifibought 동일 (첫 실행 로딩)
│   │   │   └── Dashboard/
│   │   │       ├── IndexCard.tsx
│   │   │       ├── AITopPicks.tsx
│   │   │       └── MarketSentiment.tsx
│   │   ├── context/                         # ifibought 패턴 유지
│   │   │   ├── AppContext.tsx               # watchlist, dataReady
│   │   │   ├── AuthContext.tsx              # user, isLoading, isNewUser
│   │   │   └── TabContext.tsx               # activeTab, resultTabs
│   │   ├── hooks/
│   │   │   ├── useStockChart.ts             # ifibought 동일
│   │   │   ├── useSimulation.ts             # ifibought 동일 (시뮬레이터)
│   │   │   ├── useDataDownload.ts           # ifibought 동일 (SSE 진행률)
│   │   │   ├── useAISignal.ts               # AI 시그널 polling
│   │   │   ├── usePortfolio.ts              # 포트폴리오 상태
│   │   │   └── useWebSocket.ts              # KIS WebSocket
│   │   ├── store/
│   │   │   ├── authStore.ts                 # Zustand: 로그인, 모드
│   │   │   ├── tradeStore.ts                # Zustand: 주문 상태
│   │   │   └── alertStore.ts                # Zustand: 알림
│   │   ├── lib/
│   │   │   ├── api.ts                       # ifibought api.ts 확장
│   │   │   └── utils.ts                     # cn() 유틸 (shadcn 표준)
│   │   ├── pages/
│   │   │   ├── LandingPage.tsx              # ifibought 구조 기반
│   │   │   └── SettingsPage.tsx             # KIS 키 관리, 알림 설정
│   │   ├── types/
│   │   │   └── index.ts                     # 공통 타입 정의
│   │   ├── App.tsx                          # ifibought App.tsx 구조 기반
│   │   ├── main.tsx
│   │   └── index.css                        # @import "tailwindcss" + CSS 변수
│   ├── package.json
│   ├── tsconfig.json
│   ├── tsconfig.app.json
│   └── vite.config.ts
│
├── backend/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── auth.py                      # 인증 라우터
│   │   │   ├── stocks.py                    # 시세/차트 라우터
│   │   │   ├── ai.py                        # AI 예측/시그널 라우터
│   │   │   ├── trades.py                    # 주문 실행 라우터
│   │   │   ├── portfolio.py                 # 포트폴리오 라우터
│   │   │   ├── screener.py                  # 스크리너 라우터
│   │   │   ├── backtest.py                  # 백테스팅 라우터
│   │   │   ├── risk.py                      # 리스크 설정 라우터
│   │   │   └── alerts.py                    # 알림 설정 라우터
│   │   ├── middleware/
│   │   │   ├── auth_middleware.py           # JWT 검증
│   │   │   └── rate_limit.py               # Rate limiting
│   │   └── deps.py                          # 공통 의존성
│   │
│   ├── services/
│   │   ├── kis_service.py                   # KIS API 래퍼 (실거래/모의 URL 전환)
│   │   ├── market_service.py                # pykrx 시세 수집 + Redis 캐싱
│   │   ├── websocket_service.py             # KIS WebSocket 관리 (41종목 제한 처리)
│   │   ├── ai_service.py                    # AI 시그널 조율 (지표 + LSTM)
│   │   ├── backtest_service.py              # 백테스팅 엔진
│   │   ├── pattern_service.py               # 캔들 패턴 인식
│   │   ├── risk_service.py                  # 한도 체크 로직
│   │   └── email_service.py                 # SendGrid 이메일 발송
│   │
│   ├── ml/
│   │   ├── model.py                         # LSTM 모델 클래스 정의
│   │   ├── features.py                      # 피처 엔지니어링 (pandas-ta)
│   │   ├── train.py                         # 오프라인 학습 스크립트
│   │   ├── predict.py                       # 추론 인터페이스
│   │   ├── pattern_matcher.py               # 유사 패턴 히스토리 매칭
│   │   └── weights/                         # 학습된 모델 가중치 (.pth)
│   │       └── {stock_code}.pth
│   │
│   ├── models/                              # SQLAlchemy ORM 모델
│   │   ├── user.py
│   │   ├── portfolio.py
│   │   ├── trade.py
│   │   ├── watchlist.py
│   │   ├── risk_settings.py
│   │   ├── alert_settings.py
│   │   └── backtest_result.py
│   │
│   ├── tasks/                               # Celery 비동기 태스크
│   │   ├── ai_tasks.py                      # AI 시그널 갱신
│   │   ├── email_tasks.py                   # 이메일 발송
│   │   └── report_tasks.py                  # 주간 리포트 생성
│   │
│   ├── core/
│   │   ├── config.py                        # 환경변수 (pydantic Settings)
│   │   ├── security.py                      # JWT 발급/검증, AES 암호화
│   │   ├── database.py                      # DB 세션 팩토리 (async)
│   │   └── redis_client.py                  # Redis 연결 풀
│   │
│   └── main.py                              # FastAPI 앱 + 라우터 등록
│
├── db/
│   └── migrations/                          # Alembic 마이그레이션
│
├── docs/
│   └── superpowers/specs/
│       ├── 2026-05-31-stocksenseai-PRD.md
│       └── 2026-05-31-stocksenseai-TRD.md
│
├── docker-compose.yml
├── nginx.conf
├── .env.example
└── run.sh
```

---

## 4. 데이터베이스 스키마

### 4.1 users

```sql
CREATE TABLE users (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email            VARCHAR(255) UNIQUE NOT NULL,
    password_hash    VARCHAR(255) NOT NULL,
    is_verified      BOOLEAN DEFAULT FALSE,           -- 이메일 인증 완료
    mode             VARCHAR(20) DEFAULT 'demo',      -- 'demo' | 'paper' | 'real'
    kis_paper_key_enc  TEXT,                          -- AES-256 암호화된 모의투자 APP KEY
    kis_paper_secret_enc TEXT,
    kis_paper_account_no VARCHAR(20),                 -- 모의투자 계좌번호
    kis_real_key_enc   TEXT,                          -- AES-256 암호화된 실거래 APP KEY
    kis_real_secret_enc  TEXT,
    kis_real_account_no  VARCHAR(20),
    dark_mode        BOOLEAN DEFAULT TRUE,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);
```

### 4.2 portfolios

```sql
CREATE TABLE portfolios (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID REFERENCES users(id) ON DELETE CASCADE,
    stock_code   VARCHAR(10) NOT NULL,
    stock_name   VARCHAR(100),
    quantity     INTEGER NOT NULL CHECK (quantity > 0),
    avg_price    NUMERIC(12,2) NOT NULL,
    mode         VARCHAR(20) NOT NULL,               -- 'paper' | 'real'
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, stock_code, mode)
);
CREATE INDEX idx_portfolios_user ON portfolios(user_id, mode);
```

### 4.3 trades

```sql
CREATE TABLE trades (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        UUID REFERENCES users(id) ON DELETE CASCADE,
    stock_code     VARCHAR(10) NOT NULL,
    stock_name     VARCHAR(100),
    order_type     VARCHAR(10) NOT NULL,             -- 'BUY' | 'SELL'
    price_type     VARCHAR(10) NOT NULL,             -- 'MARKET' | 'LIMIT'
    quantity       INTEGER NOT NULL CHECK (quantity > 0),
    order_price    NUMERIC(12,2),                    -- 지정가 주문가
    executed_price NUMERIC(12,2),                    -- 체결가
    commission     NUMERIC(10,2) DEFAULT 0,          -- 수수료
    status         VARCHAR(20) DEFAULT 'PENDING',    -- 'PENDING'|'FILLED'|'CANCELLED'
    mode           VARCHAR(20) NOT NULL,             -- 'paper' | 'real'
    kis_order_no   VARCHAR(50),                      -- KIS 주문번호
    ai_signal_at_order VARCHAR(10),                  -- 주문 시점 AI 시그널 기록
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    filled_at      TIMESTAMPTZ
);
CREATE INDEX idx_trades_user_date ON trades(user_id, created_at DESC);
CREATE INDEX idx_trades_status ON trades(user_id, status, mode);
```

### 4.4 risk_settings

```sql
CREATE TABLE risk_settings (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                UUID REFERENCES users(id) UNIQUE ON DELETE CASCADE,
    max_per_stock_pct      NUMERIC(5,2) DEFAULT 20.0,   -- 종목당 최대 비중 (%)
    daily_loss_limit_pct   NUMERIC(5,2) DEFAULT 5.0,    -- 일일 손실 한도 (%)
    stop_loss_enabled      BOOLEAN DEFAULT FALSE,        -- 자동 손절 활성화
    trading_blocked        BOOLEAN DEFAULT FALSE,        -- 한도 도달 시 차단
    blocked_at             TIMESTAMPTZ,
    updated_at             TIMESTAMPTZ DEFAULT NOW()
);
```

### 4.5 watchlists

```sql
CREATE TABLE watchlist_groups (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID REFERENCES users(id) ON DELETE CASCADE,
    name       VARCHAR(50) NOT NULL,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE watchlist_items (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id   UUID REFERENCES watchlist_groups(id) ON DELETE CASCADE,
    user_id    UUID REFERENCES users(id) ON DELETE CASCADE,
    stock_code VARCHAR(10) NOT NULL,
    stock_name VARCHAR(100),
    target_price_high NUMERIC(12,2),                   -- 목표가 (이메일 알림)
    target_price_low  NUMERIC(12,2),                   -- 손절가 알림
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, stock_code)
);
```

### 4.6 alert_settings

```sql
CREATE TABLE alert_settings (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID REFERENCES users(id) UNIQUE ON DELETE CASCADE,
    signal_change       BOOLEAN DEFAULT TRUE,
    watchlist_price     BOOLEAN DEFAULT TRUE,
    daily_loss_limit    BOOLEAN DEFAULT TRUE,
    trade_filled        BOOLEAN DEFAULT TRUE,
    weekly_report       BOOLEAN DEFAULT FALSE,
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);
```

### 4.7 ai_signals_history

```sql
CREATE TABLE ai_signals_history (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stock_code   VARCHAR(10) NOT NULL,
    signal       VARCHAR(10) NOT NULL,              -- 'BUY' | 'HOLD' | 'SELL'
    signal_score NUMERIC(5,2),
    tech_score   NUMERIC(5,2),
    lstm_score   NUMERIC(5,2),
    rsi          NUMERIC(8,4),
    macd         NUMERIC(12,4),
    predicted_prices JSONB,                         -- [p1, p2, p3, p4, p5]
    confidence   NUMERIC(5,2),
    recorded_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_signals_code_date ON ai_signals_history(stock_code, recorded_at DESC);
```

### 4.8 backtest_results

```sql
CREATE TABLE backtest_results (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id) ON DELETE CASCADE,
    stock_code      VARCHAR(10),
    strategy_config JSONB NOT NULL,                 -- 전략 설정 스냅샷
    period_start    DATE NOT NULL,
    period_end      DATE NOT NULL,
    total_return_pct NUMERIC(10,4),
    mdd_pct         NUMERIC(10,4),
    sharpe_ratio    NUMERIC(8,4),
    win_rate_pct    NUMERIC(5,2),
    total_trades    INTEGER,
    result_detail   JSONB,                          -- 매매 시점 목록
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 5. API 엔드포인트 명세

### 5.1 인증 (/auth)

| Method | Path | Request Body | Response | 설명 |
|---|---|---|---|---|
| POST | /auth/register | `{email, password}` | `{user_id, message}` | 회원가입 |
| POST | /auth/login | `{email, password}` | `{access_token, user}` | 로그인 |
| POST | /auth/refresh | Cookie: refresh_token | `{access_token}` | 토큰 갱신 |
| POST | /auth/logout | - | `{message}` | 로그아웃 (Refresh 무효화) |
| POST | /auth/verify-email | `{token}` | `{message}` | 이메일 인증 |
| PUT | /auth/api-key | `{mode, app_key, app_secret, account_no}` | `{message, test_result}` | KIS 키 등록 |
| GET | /auth/me | - | `{user}` | 내 정보 조회 |

### 5.2 시세 (/stocks)

| Method | Path | Query Params | 설명 |
|---|---|---|---|
| GET | /stocks | `market, limit, page` | 종목 목록 (코스피/코스닥) |
| GET | /stocks/search | `q` | 종목 검색 |
| GET | /stocks/{code} | - | 종목 상세 (현재가, 기업정보) |
| GET | /stocks/{code}/chart | `period, interval` | OHLCV 차트 데이터 |
| GET | /stocks/{code}/orderbook | - | 10단 호가 |
| GET | /stocks/{code}/trades | - | 실시간 체결 최근 20건 |
| GET | /stocks/indices | - | 코스피/코스닥 지수 |

**`/stocks/{code}/chart` 파라미터:**
- `period`: `1d` / `1w` / `1m` / `3m` / `1y`
- `interval`: `1min` / `5min` / `15min` / `1h` / `1d` / `1w` / `1mo`

### 5.3 AI 예측 (/ai) — 핵심

| Method | Path | 설명 |
|---|---|---|
| GET | /ai/{code}/signal | 종합 AI 시그널 (BUY/HOLD/SELL + 점수) |
| GET | /ai/{code}/predict | LSTM 5일 예측 (Bullish/Base/Bearish) |
| GET | /ai/{code}/indicators | RSI, MACD, 볼린저밴드 원시 값 |
| GET | /ai/{code}/patterns | 감지된 캔들 패턴 목록 |
| GET | /ai/{code}/similar | 유사 패턴 히스토리 Top 5 |
| GET | /ai/{code}/multiframe | 일봉/주봉/월봉 멀티 타임프레임 시그널 |
| GET | /ai/top-picks | AI BUY 시그널 상위 종목 (스크리너) |
| GET | /ai/signals/history/{code} | 시그널 변경 이력 (최근 30일) |

**`/ai/{code}/predict` 응답 스키마:**
```json
{
  "code": "005930",
  "name": "삼성전자",
  "current_price": 73400,
  "as_of": "2026-05-31T15:30:00+09:00",
  "prediction": {
    "bullish": [74200, 75100, 75800, 76200, 77000],
    "base":    [73800, 74200, 74100, 74500, 74900],
    "bearish": [73100, 72800, 72500, 72000, 71800]
  },
  "confidence": 67.3,
  "signal": "BUY",
  "signal_score": 72,
  "signal_breakdown": {
    "technical_score": 68,
    "lstm_score": 74,
    "technical_weight": 0.4,
    "lstm_weight": 0.6
  },
  "indicators": {
    "rsi_14": 58.4,
    "macd": 142.3,
    "macd_signal": 98.7,
    "macd_hist": 43.6,
    "bb_upper": 75200,
    "bb_middle": 73200,
    "bb_lower": 71600,
    "ma5": 73100,
    "ma20": 72300,
    "ma60": 70100,
    "ma120": 68500
  }
}
```

### 5.4 거래 (/trades)

| Method | Path | 설명 |
|---|---|---|
| POST | /trades/order | 주문 실행 (리스크 체크 → KIS API) |
| GET | /trades | 주문 목록 (status, date 필터) |
| DELETE | /trades/{id} | 미체결 주문 취소 |

**`/trades/order` Request:**
```json
{
  "stock_code": "005930",
  "order_type": "BUY",
  "price_type": "LIMIT",
  "quantity": 10,
  "price": 73000,
  "mode": "paper"
}
```

### 5.5 포트폴리오 (/portfolio)

| Method | Path | 설명 |
|---|---|---|
| GET | /portfolio | 보유 종목 현황 + 수익률 |
| GET | /portfolio/performance | 일별 평가액 히스토리 |
| GET | /portfolio/metrics | MDD, 샤프 비율, 승률 |
| GET | /portfolio/export | CSV 다운로드 |

### 5.6 스크리너 / 리스크 / 알림

| Method | Path | 설명 |
|---|---|---|
| GET | /screener | 필터 적용 종목 목록 |
| POST | /backtest/run | 백테스트 실행 (Celery 비동기) |
| GET | /backtest/{id} | 백테스트 결과 조회 |
| GET/PUT | /risk/settings | 리스크 설정 조회/수정 |
| GET/PUT | /alerts/settings | 알림 설정 조회/수정 |

### 5.7 투자 시뮬레이터 (/simulate) — ifibought 통합

| Method | Path | 설명 |
|---|---|---|
| POST | /simulate/lumpsum | 일시불 투자 수익률 계산 |
| POST | /simulate/recurring | 적립식 투자 수익률 계산 |
| GET | /simulate/data-status | 캐시 데이터 존재 여부 |
| GET | /simulate/download | 전체 데이터 다운로드 (SSE 스트리밍) |

**`/simulate/lumpsum` 요청/응답:**
```json
// Request
{
  "tickers": ["005930", "000660"],
  "buy_date": "2022-01-03",
  "sell_date": "2026-05-31",
  "amount_krw": 1000000
}

// Response
{
  "buy_date_actual": "2022-01-03",
  "sell_date_actual": "2026-05-30",
  "results": [
    {
      "ticker": "005930",
      "name": "삼성전자",
      "shares": 13,
      "buy_price": 76000,
      "sell_price": 73400,
      "buy_value_krw": 988000,
      "sell_value_krw": 954200,
      "profit_krw": -33800,
      "return_pct": -3.42,
      "chart_data": [{"date": "2022-01-03", "return_pct": 0.0}, ...]
    }
  ]
}
```

**`/simulate/recurring` 요청/응답:**
```json
// Request
{
  "tickers": ["005930"],
  "start_date": "2020-01-02",
  "end_date": "2026-05-31",
  "monthly_amount_krw": 300000
}

// Response
{
  "results": [{
    "ticker": "005930",
    "name": "삼성전자",
    "total_invested_krw": 19200000,
    "total_shares": 252,
    "avg_buy_price": 76190,
    "current_value_krw": 18496800,
    "return_pct": -3.56,
    "total_purchases": 64,
    "chart_data": [
      {"date": "2020-01-02", "invested": 300000, "value": 300000},
      ...
    ]
  }]
}
```

**`/simulate/download` SSE 스트리밍:**
```
event: progress
data: {"current": 23, "total": 81, "ticker": "005930", "name": "삼성전자"}

event: complete
data: {"message": "다운로드 완료"}
```

### 5.8 인증 — Google OAuth (/auth/google)

| Method | Path | 설명 |
|---|---|---|
| GET | /auth/google | Google OAuth 리다이렉트 |
| GET | /auth/google/callback | 콜백 처리 → JWT 발급 |

**흐름:**
```
프론트 [Google로 시작하기] 클릭
  → GET /auth/google
  → Google OAuth 동의 화면
  → Google callback → /auth/google/callback
  → users 테이블 upsert (email 기준)
  → JWT (Access + Refresh) 발급
  → 프론트 리다이렉트 + 토큰 저장
  → isNewUser=true → OnboardingModal 표시
```

---

## 6. AI 차트 예측 파이프라인 (상세)

### 6.1 데이터 수집 (market_service.py)

```python
# 일봉 2년치 수집
pykrx.stock.get_market_ohlcv(
    fromdate="20240101",
    todate="20260531",
    ticker="005930"
)
# 반환: DataFrame [날짜, 시가, 고가, 저가, 종가, 거래량]

# 캐싱 전략
Redis Key: "ohlcv:{code}:{date}"
TTL: 장중(09:00-15:30) → 30초, 장외 → 24시간
```

### 6.2 피처 엔지니어링 (ml/features.py)

pandas-ta 라이브러리 사용:

```python
import pandas_ta as ta

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df['rsi_14']    = ta.rsi(df['close'], length=14)
    df['macd']      = ta.macd(df['close'])['MACD_12_26_9']
    df['macd_sig']  = ta.macd(df['close'])['MACDs_12_26_9']
    df['macd_hist'] = ta.macd(df['close'])['MACDh_12_26_9']
    bb = ta.bbands(df['close'], length=20, std=2)
    df['bb_upper']  = bb['BBU_20_2.0']
    df['bb_mid']    = bb['BBM_20_2.0']
    df['bb_lower']  = bb['BBL_20_2.0']
    df['ma5']       = ta.sma(df['close'], length=5)
    df['ma20']      = ta.sma(df['close'], length=20)
    df['ma60']      = ta.sma(df['close'], length=60)
    df['vol_ma5']   = ta.sma(df['volume'], length=5)
    df['stoch_k']   = ta.stoch(df['high'],df['low'],df['close'])['STOCHk_14_3_3']
    return df.dropna()
# 총 13개 피처 (OHLCV 5 + 지표 8)
```

### 6.3 LSTM 모델 구조 (ml/model.py)

```python
class StockLSTM(nn.Module):
    """
    입력: (batch, seq_len=60, features=13)
    출력: (batch, 5) — 다음 5 거래일 종가 변화율 예측
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=13,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Attention on last 10 time steps
        attn_out, _ = self.attention(
            lstm_out[:, -10:, :].permute(1, 0, 2),
            lstm_out[:, -10:, :].permute(1, 0, 2),
            lstm_out[:, -10:, :].permute(1, 0, 2)
        )
        out = attn_out[-1]  # (batch, 128)
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out)
```

**학습 설정:**
- 데이터: 종목별 2년치 일봉 (~500 거래일)
- Train/Val/Test: 70% / 15% / 15%
- 정규화: MinMaxScaler (종가 기준, 피처별 독립 스케일)
- 손실함수: Huber Loss (이상치 내성)
- 옵티마이저: AdamW (lr=0.001, weight_decay=1e-5)
- 스케줄러: CosineAnnealingLR
- Early Stopping: Val Loss 10 epoch 개선 없으면 중단
- 가중치 저장: `ml/weights/{code}.pth`

### 6.4 예측 시나리오 생성 (ml/predict.py)

```python
def predict_scenarios(code: str, model, scaler, df: pd.DataFrame) -> dict:
    features = build_features(df).iloc[-60:]
    x = scaler.transform(features.values)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    # Monte Carlo Dropout (불확실성 추정)
    model.train()  # dropout 활성화
    predictions = []
    for _ in range(50):
        with torch.no_grad():
            predictions.append(model(x_tensor).numpy())

    preds = np.array(predictions).squeeze()  # (50, 5)
    base    = np.median(preds, axis=0)
    bullish = np.percentile(preds, 75, axis=0)
    bearish = np.percentile(preds, 25, axis=0)

    # 변화율 → 실제 가격 역변환
    current = df['close'].iloc[-1]
    return {
        "base":    (current * (1 + base)).tolist(),
        "bullish": (current * (1 + bullish)).tolist(),
        "bearish": (current * (1 + bearish)).tolist(),
        "confidence": calc_recent_accuracy(code)  # 최근 20회 방향 정확도
    }
```

### 6.5 캔들 패턴 인식 (services/pattern_service.py)

pandas-ta의 `cdl_pattern()` 활용:
```python
patterns_to_detect = [
    'hammer', 'invertedhammer', 'doji', 'engulfing',
    'morningstar', 'eveningstar', 'shootingstar', 'hangingman',
    'threewhitesoldiers', 'threeblackcrows', 'piercingpattern',
    'darkcloudcover', 'harami', 'haramicross'
]
# 감지 결과: {pattern_name: signal_value} (100=강세, -100=약세, 0=없음)
```

### 6.6 종합 시그널 계산

```python
def calculate_signal(indicators: dict, lstm_direction: float) -> dict:
    # 기술적 지표 점수 (0~100)
    rsi_score   = 100 - indicators['rsi_14']  if indicators['rsi_14'] < 50 else indicators['rsi_14']
    macd_score  = 70 if indicators['macd_hist'] > 0 else 30
    bb_score    = calc_bb_score(indicators)
    tech_score  = (rsi_score * 0.4 + macd_score * 0.35 + bb_score * 0.25)

    # LSTM 방향 점수 (0~100)
    lstm_score  = 50 + lstm_direction * 50  # -1~1 → 0~100

    # 가중 합산
    final_score = tech_score * 0.4 + lstm_score * 0.6

    if final_score >= 65:   signal = "BUY"
    elif final_score <= 35: signal = "SELL"
    else:                   signal = "HOLD"

    return {"signal": signal, "score": round(final_score, 1),
            "tech_score": round(tech_score, 1),
            "lstm_score": round(lstm_score, 1)}
```

---

## 7. KIS WebSocket 실시간 데이터

### 7.1 WebSocket 제약 및 대응

**KIS 제약:** 1 WebSocket 세션 당 최대 41종목 구독

**대응 전략 (websocket_service.py):**
```python
class KISWebSocketPool:
    """
    접속 사용자 수에 따라 WebSocket 세션 풀 관리.
    관심 종목 / 보유 종목을 모아 41개씩 배치 구독.
    데이터 수신 → Redis Pub/Sub으로 해당 유저에게 라우팅.
    """
    MAX_SYMBOLS_PER_SESSION = 41

    async def subscribe(self, symbols: list[str]):
        batches = [symbols[i:i+41] for i in range(0, len(symbols), 41)]
        for batch in batches:
            session = await self._get_or_create_session()
            for symbol in batch:
                await session.subscribe(symbol)

    async def on_message(self, data: dict):
        # Redis Pub/Sub 채널에 발행 → FastAPI SSE로 클라이언트 전달
        await redis.publish(f"stock:{data['code']}", json.dumps(data))
```

**OAuth2 인증 흐름:**
```
POST https://openapi.koreainvestment.com:9443/oauth2/Approval
  → approval_key 획득
  → WebSocket 연결 시 헤더에 포함
  → 세션 만료(24h) 전 자동 갱신
```

### 7.2 KIS REST API TR ID 매핑

| 기능 | 실거래 TR ID | 모의투자 TR ID |
|---|---|---|
| 현재가 조회 | FHKST01010100 | 동일 |
| 매수 주문 | TTTC0802U | VTTC0802U |
| 매도 주문 | TTTC0801U | VTTC0801U |
| 잔고 조회 | TTTC8434R | VTTC8434R |
| 체결 내역 | TTTC8001R | VTTC8001R |
| 미체결 조회 | TTTC8036R | VTTC8036R |
| 주문 취소 | TTTC0803U | VTTC0803U |
| WS 체결가 | H0STCNT0 (국내) | 동일 |
| WS 호가 | H0STASP0 | 동일 |

### 7.3 주문 실행 흐름

```
POST /trades/order
  → JWT 검증 → 유저 조회
  → 데모 모드 체크 → 실거래 요청이면 차단
  → trading_blocked 체크
  → 종목별 한도 체크 (risk_service)
  → 일일 손실 한도 체크 (risk_service)
    → 한도 초과: 400 에러 + 이메일 발송 (Celery 비동기)
  → KIS API 주문 요청 (kis_service)
  → trades 테이블 INSERT (status=PENDING)
  → 체결 확인 폴링 (10초, 최대 5회)
    → 체결: status=FILLED, portfolio 업데이트
    → 미체결: status 유지, 사용자에게 응답
  → 체결 이메일 발송 (Celery 비동기)
```

---

## 8. 인증 & 보안

### 8.1 JWT 구조

```python
# Access Token Payload
{
    "sub": "user_id",
    "email": "user@example.com",
    "mode": "paper",           # 현재 모드
    "exp": 1748700000,         # 30분
    "iat": 1748698200
}

# Refresh Token: 7일, HttpOnly Secure Cookie
# Rotation: Refresh 사용 시 새 Refresh Token 발급, 이전 무효화
```

### 8.2 KIS API 키 암호화

```python
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def encrypt_api_key(plaintext: str, key: bytes) -> str:
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode(), None)
    return base64.b64encode(nonce + ciphertext).decode()

def decrypt_api_key(encrypted: str, key: bytes) -> str:
    data = base64.b64decode(encrypted)
    nonce, ciphertext = data[:12], data[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None).decode()
```

### 8.3 Rate Limiting (slowapi)

```python
# 로그인 엔드포인트
@limiter.limit("5/minute")
async def login(request: Request): ...

# 일반 API
@limiter.limit("100/minute")
async def get_stock(code: str): ...

# AI 예측 (무거운 추론)
@limiter.limit("20/minute")
async def predict(code: str): ...
```

### 8.4 보안 헤더 (Nginx)

```nginx
add_header X-Content-Type-Options nosniff;
add_header X-Frame-Options DENY;
add_header X-XSS-Protection "1; mode=block";
add_header Content-Security-Policy "default-src 'self'; ...";
add_header Strict-Transport-Security "max-age=31536000";
```

---

## 9. 비동기 태스크 (Celery)

### 9.1 태스크 정의

```python
# tasks/ai_tasks.py
@celery.task
def refresh_ai_signals():
    """장 종료 후(15:35) 전 종목 AI 시그널 갱신. APScheduler가 트리거."""
    codes = get_watched_codes()  # watchlist + portfolio 종목
    for code in codes:
        signal = ai_service.calculate_signal(code)
        prev = get_prev_signal(code)
        if signal['signal'] != prev:
            notify_signal_change.delay(code, prev, signal)  # 이메일 알림
        save_signal(code, signal)

# tasks/email_tasks.py
@celery.task(max_retries=3, default_retry_delay=60)
def send_email(to: str, template_id: str, data: dict):
    try:
        sendgrid.send(to, template_id, data)
    except Exception as e:
        raise self.retry(exc=e)

# tasks/report_tasks.py
@celery.task
def send_weekly_report():
    """매주 월요일 08:00 실행 (APScheduler cron)"""
    users = get_users_with_weekly_report_enabled()
    for user in users:
        report = build_weekly_report(user)
        send_email.delay(user.email, WEEKLY_REPORT_TEMPLATE, report)
```

### 9.2 스케줄 (APScheduler)

```python
scheduler.add_job(refresh_ai_signals,   'cron', hour=15, minute=35, day_of_week='mon-fri')
scheduler.add_job(send_weekly_report,   'cron', day_of_week='mon', hour=8)
scheduler.add_job(check_price_alerts,   'interval', minutes=5)  # 관심 종목 가격 알림
scheduler.add_job(check_daily_loss,     'interval', minutes=10) # 손실 한도 체크
```

---

## 10. 백테스팅 엔진 (services/backtest_service.py)

```python
def run_backtest(config: BacktestConfig) -> BacktestResult:
    """
    config: {code, start_date, end_date, initial_cash,
             entry_signal_score, exit_signal_score,
             stop_loss_pct, take_profit_pct, commission_rate}
    """
    df = load_ohlcv(config.code, config.start_date, config.end_date)
    signals = load_historical_signals(config.code, config.start_date, config.end_date)

    cash = config.initial_cash
    position = 0
    trades_log = []
    equity_curve = []

    for i, row in df.iterrows():
        signal = signals.get(i, {})
        score = signal.get('signal_score', 50)
        price = row['close']

        # 진입
        if score >= config.entry_signal_score and position == 0:
            shares = int(cash * 0.95 / price)
            cost = shares * price * (1 + config.commission_rate)
            cash -= cost
            position = shares
            entry_price = price

        # 청산
        elif position > 0:
            change = (price - entry_price) / entry_price
            if (score <= config.exit_signal_score
                or change <= -config.stop_loss_pct
                or change >= config.take_profit_pct):
                revenue = position * price * (1 - config.commission_rate)
                cash += revenue
                trades_log.append({'date': i, 'entry': entry_price,
                                   'exit': price, 'pnl': revenue - position * entry_price})
                position = 0

        equity_curve.append(cash + position * price)

    return compute_metrics(equity_curve, trades_log, config.initial_cash)
```

**성과 지표 계산:**
- **MDD**: max(cummax - current) / cummax
- **샤프 비율**: (mean_return - risk_free) / std_return * sqrt(252)
- **승률**: profitable_trades / total_trades

---

## 11. 프론트엔드 핵심 설계

### 11.1 Lightweight Charts AI 오버레이

```javascript
// AIPredictionOverlay.jsx
import { createChart, LineSeries } from 'lightweight-charts';

function addPredictionOverlay(chart, currentData, prediction) {
    const lastCandle = currentData[currentData.length - 1];
    const dates = generateNextTradingDates(lastCandle.time, 5);

    // Base (흰색 점선)
    const baseSeries = chart.addLineSeries({
        color: '#FFFFFF', lineWidth: 1,
        lineStyle: LineStyle.Dashed, title: 'AI Base'
    });
    baseSeries.setData([
        { time: lastCandle.time, value: lastCandle.close },
        ...dates.map((d, i) => ({ time: d, value: prediction.base[i] }))
    ]);

    // Bullish (초록 점선)
    const bullishSeries = chart.addLineSeries({
        color: '#00D68F', lineWidth: 1, lineStyle: LineStyle.Dashed
    });
    bullishSeries.setData([
        { time: lastCandle.time, value: lastCandle.close },
        ...dates.map((d, i) => ({ time: d, value: prediction.bullish[i] }))
    ]);

    // Bearish (빨강 점선)
    const bearishSeries = chart.addLineSeries({
        color: '#FF4D6A', lineWidth: 1, lineStyle: LineStyle.Dashed
    });
    bearishSeries.setData([...]);
}
```

### 11.2 Zustand 상태 구조

```javascript
// authStore.js
const useAuthStore = create((set) => ({
    user: null,
    accessToken: null,
    tradeMode: 'paper',   // 'paper' | 'real'
    login: (user, token) => set({ user, accessToken: token }),
    logout: () => set({ user: null, accessToken: null }),
    setTradeMode: (mode) => set({ tradeMode: mode }),
}));

// Axios interceptor
client.interceptors.request.use((config) => {
    const token = useAuthStore.getState().accessToken;
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
});
```

### 11.3 반응형 브레이크포인트 (Tailwind)

```javascript
// tailwind.config.js
theme: {
    screens: {
        'sm': '360px',   // 모바일: 탭 바 하단, 차트 풀스크린
        'md': '768px',   // 태블릿: 사이드바 아이콘만
        'lg': '1280px',  // 데스크탑: 3단 레이아웃
    }
}
```

---

## 12. 환경변수 (.env.example)

```env
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/stocksense
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-256-bit-secret-key-here
ENCRYPTION_KEY=your-32-byte-aes-key-here
REFRESH_TOKEN_SECRET=another-secret-key

# SendGrid
SENDGRID_API_KEY=SG.xxxxxxxxxxxx
SENDGRID_FROM_EMAIL=noreply@stocksenseai.com
SENDGRID_SIGNAL_TEMPLATE_ID=d-xxxx
SENDGRID_TRADE_TEMPLATE_ID=d-xxxx
SENDGRID_WEEKLY_TEMPLATE_ID=d-xxxx

# Demo Mode (공용 KIS 모의투자 키)
DEMO_KIS_PAPER_APP_KEY=PSxxxxxxxxxxxxxxxxxxxxxxxx
DEMO_KIS_PAPER_APP_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEMO_KIS_PAPER_ACCOUNT=xxxxxxxxxx

# Celery
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# App
APP_ENV=development   # development | production
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

---

## 13. Docker Compose

```yaml
version: '3.9'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: stocksense
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    env_file: .env
    ports:
      - "8000:8000"
    volumes:
      - ./backend/ml/weights:/app/ml/weights  # 모델 가중치 마운트

  celery:
    build:
      context: ./backend
    command: celery -A tasks worker --loglevel=info --concurrency=4
    depends_on: [redis, postgres]
    env_file: .env
    volumes:
      - ./backend/ml/weights:/app/ml/weights

  celery-beat:
    build:
      context: ./backend
    command: celery -A tasks beat --loglevel=info
    depends_on: [redis]
    env_file: .env

  frontend:
    build:
      context: ./frontend
    ports:
      - "3000:80"
    environment:
      - VITE_API_BASE=http://localhost:8000/api

  nginx:
    image: nginx:1.25-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on: [backend, frontend]

volumes:
  pgdata:
```

---

## 14. 테스트 전략

| 테스트 유형 | 대상 | 도구 |
|---|---|---|
| 단위 테스트 | AI 시그널 계산, 리스크 체크 로직 | pytest |
| 통합 테스트 | API 엔드포인트 → DB | pytest + httpx |
| ML 정확도 테스트 | LSTM 방향 정확도 55% 이상 | pytest + 검증 데이터셋 |
| 백테스팅 검증 | 과거 시그널 vs 실제 수익률 비교 | 별도 검증 스크립트 |
| E2E 테스트 | 주문 플로우 (데모 모드) | Playwright |
| 성능 테스트 | API 응답 시간 < 목표값 | Locust |

---

## 15. 작업 분담 가이드 (2인 개발)

### 담당 A — 백엔드 & ML & 인프라

**Week 1-2: 인프라 기반**
- Docker Compose 환경 구성 (PostgreSQL + Redis + Celery)
- Alembic 마이그레이션 초기 셋업
- FastAPI 앱 구조, JWT 인증, Rate Limiting

**Week 3-4: 핵심 서비스**
- KIS API 연동 (kis_service.py) — 실거래/모의 전환
- pykrx 시세 수집 + Redis 캐싱 (market_service.py)
- KIS WebSocket 풀 관리 (websocket_service.py)

**Week 5-6: AI 파이프라인**
- 피처 엔지니어링 (ml/features.py)
- LSTM 학습 스크립트 + 가중치 생성 (ml/train.py)
- AI 시그널 계산 (services/ai_service.py)
- 캔들 패턴 인식 (services/pattern_service.py)

**Week 7-8: 거래 & 알림**
- 주문 실행 + 리스크 체크 (risk_service.py)
- 이메일 알림 (Celery + SendGrid)
- 백테스팅 엔진 (backtest_service.py)
- APScheduler 시그널 갱신 스케줄

### 담당 B — 프론트엔드

**Week 1-2: 인증 & 레이아웃**
- 로그인/회원가입 화면
- 반응형 레이아웃 (사이드바, 헤더, 모바일 탭)
- Axios 클라이언트 + JWT interceptor
- Zustand 스토어 초기화

**Week 3-4: 차트 & 시세**
- Lightweight Charts 캔들스틱 구현
- 기술적 지표 서브차트 (RSI, MACD)
- 실시간 시세 WebSocket 연동
- 종목 검색 + 상세 페이지

**Week 5-6: AI 기능**
- AI 예측 오버레이 (3시나리오 점선)
- 캔들 패턴 배지 + 툴팁
- AI 시그널 카드 + 점수 분해 표시
- 멀티 타임프레임 패널

**Week 7-8: 거래 & 포트폴리오**
- 주문 모달 + 확인 플로우
- 호가창 (10단)
- 포트폴리오 현황 + 수익률 차트
- 리스크 설정 폼
- 백테스팅 UI

---

---

## 15-A. 투자 시뮬레이터 서비스 (services/simulator_service.py)

ifibought의 `services/calculator.py` + `services/data_loader.py` 로직을 통합.

### 데이터 캐시 전략

```python
# ifibought 방식: 로컬 JSON 파일 캐시
# StockSenseAI: PostgreSQL + yfinance (더 신뢰성 있는 데이터)

class SimulatorDataLoader:
    async def get_prices(self, ticker: str, start: date, end: date) -> dict[str, float]:
        """
        1. PostgreSQL price_cache 테이블 조회
        2. 없으면 yfinance로 다운로드 후 저장
        3. 결측치(휴장일): ffill() 처리
        """

    def get_next_trading_day(self, d: date) -> date:
        """휴장일/주말 → 다음 거래일 반환"""
```

### 일시불 계산 로직

```python
def calc_lumpsum(ticker, buy_date, sell_date, amount_krw):
    prices = get_prices(ticker, buy_date, sell_date)
    buy_price = prices[buy_date]
    sell_price = prices[sell_date]

    shares = int(amount_krw / buy_price)         # 정수 주 단위
    buy_value = shares * buy_price
    sell_value = shares * sell_price
    profit = sell_value - buy_value
    return_pct = (profit / buy_value) * 100

    # 날짜별 수익률 시리즈 (Recharts용)
    chart_data = [
        {"date": d, "return_pct": ((p / buy_price) - 1) * 100}
        for d, p in prices.items()
    ]
    return {...}
```

### 적립식 계산 로직 (ifibought calculator.py 기반)

```python
def calc_recurring(ticker, start_date, end_date, monthly_krw):
    trading_days = get_first_trading_days_of_month(start_date, end_date)
    prices = get_prices(ticker, start_date, end_date)

    total_shares = 0
    total_invested = 0
    purchases = []

    for trade_date in trading_days:
        price = prices[trade_date]
        shares = int(monthly_krw / price)
        total_shares += shares
        total_invested += shares * price
        purchases.append({"date": trade_date, "shares": shares, "price": price})

    final_price = prices[end_date]
    current_value = total_shares * final_price
    avg_buy_price = total_invested / total_shares if total_shares > 0 else 0

    # 누적 투자액 vs 평가액 차트 데이터 (원금 회복 시점 시각화)
    chart_data = build_cumulative_chart(purchases, prices, start_date, end_date)
    return {...}
```

### 앱 초기화 흐름 (ifibought InitialLoadingScreen 기반)

```
사용자 로그인 + access_allowed
  → GET /simulate/data-status → {ready: false}
  → InitialLoadingScreen 전체 화면 표시
  → GET /simulate/download (SSE)
    → "삼성전자 다운로드 중... (23/81)" 진행률 토스트
  → 완료 → MainLayout 렌더링
```

---

## 15-B. 프론트엔드 앱 초기화 흐름 (App.tsx)

ifibought App.tsx 구조를 직접 기반으로 확장:

```typescript
// App.tsx 핵심 흐름
export default function App() {
  const { dataReady, setDataReady } = useApp();
  const { user, isLoading, isNewUser } = useAuth();
  const [showSettings, setShowSettings] = useState(false);
  const [paymentSuccess, setPaymentSuccess] = useState(false);

  // 결제 완료 처리 (?payment=success)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("payment") === "success") {
      setPaymentSuccess(true);
      // URL 클린업
    }
  }, []);

  // 신규 가입 온보딩
  useEffect(() => {
    if (isNewUser) setShowOnboarding(true);
  }, [isNewUser]);

  if (isLoading) return <LoadingSpinner />;
  if (!user)     return <LandingPage />;         // 비로그인
  if (!user.access_allowed) return <UpgradeScreen />;  // 결제 필요

  return (
    <div className="min-h-screen bg-white" style={{ minWidth: "1280px" }}>
      <Header onSettingsClick={...} paymentSuccess={paymentSuccess} />
      {dataReady ? <MainLayout /> : <InitialLoadingScreen onReady={...} />}
      {showOnboarding && <OnboardingModal onClose={...} />}
      <Toaster />
    </div>
  );
}
```

---

## 16. 향후 확장 포인트

| 확장 내용 | 추가 위치 | 예상 공수 |
|---|---|---|
| 새 AI 모델 (Transformer) | `ml/model.py` + `ai_service.py` | 2주 |
| 자동 매매 (신중한 검토 필요) | `services/auto_trade_service.py` | 3주 + 법적 검토 |
| 해외 주식 (미국) | `market_service.py` + KIS 해외 TR ID | 2주 |
| 소셜 기능 (투자 공유) | 별도 `social/` 모듈 | 4주 |
| 모바일 앱 (React Native) | 별도 프로젝트, API 재사용 | 8주 |
| 실시간 뉴스 감성 분석 | `services/news_service.py` + KR-FinBERT | 3주 |
