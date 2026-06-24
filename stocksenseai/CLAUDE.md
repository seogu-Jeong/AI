# StockSenseAI — AI Agent 협업 가이드

이 문서는 두 명의 개발자가 각각 Claude Code(AI agent)를 사용해 협업할 때,
각 agent가 프로젝트 컨텍스트를 즉시 파악할 수 있도록 작성된 문서입니다.

---

## 1. 프로젝트 개요

**제품명:** StockSenseAI
**목적:** AI 기반 한국 주식 차트 예측 + 실제 거래 실행을 통합한 웹 서비스
**핵심 차별점:** TradingView 수준의 차트 분석 + 토스증권 수준의 UX + LSTM AI 예측 시나리오

**레포지토리:** https://github.com/hygrenn/FinalProject
**PRD:** `docs/superpowers/specs/2026-05-31-stocksenseai-PRD.md`
**TRD:** `docs/superpowers/specs/2026-05-31-stocksenseai-TRD.md`

> 작업 전 반드시 PRD와 TRD를 읽고 시작할 것.

---

## 2. 개발자 & 담당 분리

| 개발자 | GitHub ID | 브랜치 | 담당 영역 |
|---|---|---|---|
| 황윤광 | hygrenn | `hwang` | 백엔드 + ML + 인프라 |
| 정석우 | seogu-Jeong | `seogu-Jeong` | 프론트엔드 |

**절대 규칙: 상대방 담당 파일은 직접 수정하지 않는다.**
수정이 필요하면 카톡으로 먼저 협의 후 진행.

---

## 3. 기술 스택

### 프론트엔드 (seogu-Jeong 담당)
- React 18 + TypeScript 5
- Vite 5 (빌드 도구)
- shadcn/ui (컴포넌트)
- Tailwind CSS v4
- Zustand 4 (상태 관리)
- Axios 1.6 (HTTP, JWT interceptor)
- Lightweight Charts 4 (캔들스틱 차트)
- Recharts 2.x (시뮬레이터 비교 차트)
- lucide-react (아이콘)

### 백엔드 (hygrenn 담당)
- FastAPI 0.111 (Python)
- SQLAlchemy 2 + Alembic (ORM + 마이그레이션)
- PostgreSQL 15
- Redis 7 (시세 캐시, WebSocket 메시지 브로커)
- Celery 5.3 + APScheduler 3.10 (비동기 태스크, 스케줄러)
- PyTorch 2.2 (LSTM 모델)
- pandas-ta 0.3 (기술적 지표 피처 엔지니어링)
- pykrx 1.0 (한국 주식 시세 수집)
- python-kis (KIS Open API 래퍼)
- SendGrid Python SDK 6 (이메일 알림)
- python-jose + passlib (JWT + bcrypt)
- cryptography 42 (AES-256-GCM, KIS API 키 암호화)
- Docker + Compose + Nginx

---

## 4. 디렉토리 구조 및 파일 소유권

```
FinalProject/
├── frontend/                        # ← seogu-Jeong 전담
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/                  # shadcn/ui 컴포넌트
│   │   │   ├── Layout/              # MainLayout, Header, MobileTabBar
│   │   │   ├── Sidebar/             # Sidebar, StockGroup, StockList
│   │   │   ├── MainPanel/           # 탭 라우팅 + 각 탭 컴포넌트
│   │   │   │   ├── ChartTab/        # 캔들스틱 차트, AI 오버레이, 지표
│   │   │   │   ├── AITab/           # AI 시그널 카드, 멀티프레임
│   │   │   │   ├── SimulatorTab/    # 투자 시뮬레이터 (ifibought 통합)
│   │   │   │   ├── PortfolioTab/    # 포트폴리오 현황, 수익률 차트
│   │   │   │   ├── ScreenerTab/     # 스크리너
│   │   │   │   └── BacktestTab/     # 백테스팅 UI
│   │   │   ├── WatchlistPanel/      # 하단 고정 패널, 주문 모달
│   │   │   ├── Trade/               # 호가창, 주문 내역
│   │   │   ├── Risk/                # 리스크 설정
│   │   │   ├── auth/                # 온보딩, KIS 키 등록
│   │   │   └── payment/             # 결제 유도 화면
│   │   ├── context/                 # AppContext, AuthContext, TabContext
│   │   ├── hooks/                   # useStockChart, useSimulation 등
│   │   ├── store/                   # Zustand 스토어
│   │   ├── lib/                     # api.ts, utils.ts
│   │   ├── pages/                   # LandingPage, SettingsPage
│   │   └── types/                   # ← 양쪽 공동 관리 (협의 후 수정)
│   └── ...
│
├── backend/                         # ← hygrenn 전담
│   ├── api/
│   │   ├── routes/                  # auth, stocks, ai, trades, portfolio 등
│   │   └── middleware/              # JWT 검증, Rate Limiting
│   ├── services/                    # KIS, 시세, WebSocket, AI, 이메일 등
│   ├── ml/                          # LSTM 모델, 피처 엔지니어링, 추론
│   ├── models/                      # SQLAlchemy ORM 모델
│   ├── tasks/                       # Celery 비동기 태스크
│   ├── core/                        # config, security, database, redis
│   └── main.py
│
├── db/migrations/                   # ← hygrenn 전담 (Alembic)
├── docker-compose.yml               # ← hygrenn 전담
├── nginx.conf                       # ← hygrenn 전담
├── .env.example                     # ← 양쪽 공동 관리
├── CLAUDE.md                        # ← 양쪽 공동 관리
└── docs/
    └── superpowers/specs/           # PRD, TRD 원본
```

---

## 5. 공동 관리 파일 (수정 전 반드시 협의)

| 파일 | 이유 |
|---|---|
| `frontend/src/types/index.ts` | API 요청/응답 타입 — 양쪽이 동시에 사용 |
| `.env.example` | 환경변수 목록 공유 |
| `CLAUDE.md` | 이 문서 |
| `docker-compose.yml` | 인프라 구성 변경 시 양쪽 영향 |

---

## 6. API 계약

프론트엔드는 아래 API 명세를 기준으로 mock 데이터를 만들어 개발 진행.
백엔드 완성 후 mock을 실제 API 호출로 교체.

### Base URL
- 개발: `http://localhost:8000`
- 환경변수: `VITE_API_BASE`

### 주요 엔드포인트 요약

#### 인증
```
POST /auth/register        회원가입
POST /auth/login           로그인 → {access_token, user}
POST /auth/refresh         토큰 갱신 (Cookie: refresh_token)
POST /auth/logout          로그아웃
POST /auth/verify-email    이메일 인증
PUT  /auth/api-key         KIS API 키 등록
GET  /auth/me              내 정보
GET  /auth/google          Google OAuth
GET  /auth/google/callback Google OAuth 콜백
```

#### 시세
```
GET /stocks                종목 목록 (?market, limit, page)
GET /stocks/search         종목 검색 (?q)
GET /stocks/{code}         종목 상세
GET /stocks/{code}/chart   OHLCV 차트 (?period, interval)
GET /stocks/{code}/orderbook  10단 호가
GET /stocks/{code}/trades  실시간 체결
GET /stocks/indices        코스피/코스닥 지수
```

#### AI
```
GET /ai/{code}/signal      AI 시그널 (BUY/HOLD/SELL + 점수)
GET /ai/{code}/predict     LSTM 5일 예측 (Bullish/Base/Bearish)
GET /ai/{code}/indicators  기술적 지표 원시값
GET /ai/{code}/patterns    감지된 캔들 패턴
GET /ai/{code}/similar     유사 패턴 히스토리 Top 5
GET /ai/{code}/multiframe  멀티 타임프레임 시그널
GET /ai/top-picks          AI BUY 상위 종목
```

#### 거래
```
POST   /trades/order       주문 실행
GET    /trades             주문 목록
DELETE /trades/{id}        미체결 취소
```

#### 포트폴리오
```
GET /portfolio             보유 종목 + 수익률
GET /portfolio/performance 일별 평가액 히스토리
GET /portfolio/metrics     MDD, 샤프, 승률
GET /portfolio/export      CSV 다운로드
```

#### 기타
```
GET  /screener             스크리너 필터
POST /backtest/run         백테스트 실행 (Celery 비동기)
GET  /backtest/{id}        백테스트 결과
GET/PUT /risk/settings     리스크 설정
GET/PUT /alerts/settings   알림 설정
POST /simulate/lumpsum     일시불 시뮬레이션
POST /simulate/recurring   적립식 시뮬레이션
GET  /simulate/data-status 캐시 데이터 상태
GET  /simulate/download    전체 데이터 SSE 스트리밍
```

### 핵심 응답 타입 (`frontend/src/types/index.ts` 기준)

```typescript
// 사용자
interface User {
  id: string;
  email: string;
  mode: 'demo' | 'paper' | 'real';
  access_allowed: boolean;
  is_verified: boolean;
  dark_mode: boolean;
}

// AI 시그널
interface AISignal {
  signal: 'BUY' | 'HOLD' | 'SELL';
  signal_score: number;        // 0~100
  tech_score: number;
  lstm_score: number;
  confidence: number;
  indicators: {
    rsi_14: number;
    macd: number;
    macd_signal: number;
    macd_hist: number;
    bb_upper: number;
    bb_middle: number;
    bb_lower: number;
    ma5: number; ma20: number; ma60: number; ma120: number;
  };
}

// LSTM 예측
interface Prediction {
  bullish: number[];   // 다음 5 거래일 (상위 25%)
  base: number[];      // 중앙값
  bearish: number[];   // 하위 25%
  confidence: number;
}

// 차트 데이터 (Lightweight Charts)
interface Candle {
  time: string;   // 'YYYY-MM-DD'
  open: number; high: number; low: number; close: number;
  volume: number;
}

// 포트폴리오 보유 종목
interface Holding {
  stock_code: string;
  stock_name: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  profit_loss: number;
  return_pct: number;
  ai_signal: 'BUY' | 'HOLD' | 'SELL';
}

// 주문
interface OrderRequest {
  stock_code: string;
  order_type: 'BUY' | 'SELL';
  price_type: 'MARKET' | 'LIMIT';
  quantity: number;
  price?: number;
  mode: 'paper' | 'real';
}

// 시뮬레이터 (일시불)
interface LumpsumRequest {
  tickers: string[];
  buy_date: string;
  sell_date: string;
  amount_krw: number;
}

interface LumpsumResult {
  ticker: string;
  name: string;
  shares: number;
  buy_price: number;
  sell_price: number;
  buy_value_krw: number;
  sell_value_krw: number;
  profit_krw: number;
  return_pct: number;
  chart_data: { date: string; return_pct: number }[];
}
```

---

## 7. 개발 단계 (Phase)

양쪽이 같은 Phase를 동시에 진행. Phase 완료 시 `dev` 브랜치에 merge.

### Phase 1 — MVP (1~2주차)
| hygrenn (백엔드) | seogu-Jeong (프론트) |
|---|---|
| Docker Compose 환경 (PostgreSQL + Redis + Celery) | Vite + TypeScript + shadcn 프로젝트 초기 세팅 |
| Alembic 마이그레이션 초기 셋업 | 반응형 레이아웃 (Sidebar + Header + MainPanel) |
| FastAPI 앱 구조 + JWT 인증 | 로그인/회원가입 화면 |
| Rate Limiting 미들웨어 | Zustand 스토어 + Axios JWT interceptor |
| `/auth`, `/stocks` API | LandingPage, Mock 데이터로 차트탭 UI |

### Phase 2 — 실시간 시세 + 차트 (3~4주차)
| hygrenn (백엔드) | seogu-Jeong (프론트) |
|---|---|
| KIS API 연동 (실거래/모의 전환) | Lightweight Charts 캔들스틱 구현 |
| pykrx 시세 수집 + Redis 캐싱 | 기술적 지표 서브차트 (RSI, MACD) |
| KIS WebSocket 풀 관리 | 실시간 시세 WebSocket 연동 |
| `/stocks/{code}/chart`, `/stocks/{code}/orderbook` | 종목 검색 + 상세 |

### Phase 3 — AI 기능 (5~6주차)
| hygrenn (백엔드) | seogu-Jeong (프론트) |
|---|---|
| 피처 엔지니어링 (ml/features.py) | AI 예측 오버레이 (3시나리오 점선) |
| LSTM 학습 + 가중치 생성 | 캔들 패턴 배지 + 툴팁 |
| AI 시그널 계산 + `/ai/**` API | AI 시그널 카드 + 점수 분해 |
| 캔들 패턴 인식 | 멀티 타임프레임 패널 |

### Phase 4 — 거래 + 포트폴리오 (7~8주차)
| hygrenn (백엔드) | seogu-Jeong (프론트) |
|---|---|
| 주문 실행 + 리스크 체크 | 주문 모달 + 확인 플로우 |
| Celery + SendGrid 이메일 알림 | 호가창 (10단) |
| 백테스팅 엔진 | 포트폴리오 현황 + 수익률 차트 |
| APScheduler 시그널 갱신 | 리스크 설정 폼 + 백테스팅 UI |
| `/simulate/**` API 완성 | 투자 시뮬레이터 탭 완성 |

---

## 8. 브랜치 전략

```
main          ← 최종 완성본 (직접 push 금지)
dev           ← Phase 완료 시 merge하는 통합 브랜치
hwang         ← 황윤광 작업 브랜치
seogu-Jeong   ← 정석우 작업 브랜치
```

### 작업 흐름
```bash
# 매일 작업 시작 전
git checkout hwang           # (또는 seogu-Jeong)
git pull origin hwang        # 내 브랜치 최신화

# 작업 후 저장
git add .
git commit -m "feat: 작업 내용 설명"
git push origin hwang

# dev에 합칠 때 (GitHub에서 PR)
# hwang → dev  또는  seogu-Jeong → dev
# merge 전 카톡으로 상대방에게 알릴 것
```

### 커밋 메시지 규칙
```
feat:    새 기능 추가
fix:     버그 수정
refactor: 코드 리팩토링
style:   UI/스타일 변경
docs:    문서 수정
chore:   설정, 빌드 관련
```

---

## 9. 환경변수

로컬 개발 시 `.env.example`을 복사해서 `.env` 만들기.
`.env`는 절대 커밋하지 않는다 (`.gitignore`에 포함).

```bash
cp .env.example .env
```

주요 환경변수:
```
DATABASE_URL          PostgreSQL 연결 문자열
REDIS_URL             Redis 연결 문자열
SECRET_KEY            JWT 서명 키
ENCRYPTION_KEY        AES-256 API 키 암호화용
SENDGRID_API_KEY      이메일 발송
DEMO_KIS_PAPER_APP_KEY  데모 모드용 공용 KIS 모의투자 키
VITE_API_BASE         프론트에서 사용할 백엔드 URL
```

---

## 10. 로컬 개발 환경 시작

### 백엔드 (hygrenn)
```bash
cd backend
docker-compose up -d postgres redis   # DB + Redis 먼저 시작
python -m uvicorn main:app --reload --port 8000
```

### 프론트엔드 (seogu-Jeong)
```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

### 전체 실행 (Docker)
```bash
docker-compose up --build
```

`.env.example` 복사해서 `.env` 만들기. `.env`는 절대 커밋 금지.

## KIS API 레퍼런스 코드

> **AI agent 필독:** KIS API 관련 작업 전 반드시 아래 경로의 파일을 `Read` 도구로 직접 읽을 것.
> 추측으로 코드 작성 금지 — 필드명·TR ID·헤더 구조가 틀리면 500 에러 발생.

### 레퍼런스 위치

공식 KIS Open Trading API 전체 저장소가 로컬에 클론되어 있음:
```
references/open-trading-api/
├── examples_llm/          ← AI agent용 예제 (이걸 우선 참고)
│   ├── kis_auth.py        ← 토큰 발급·헤더 구조
│   ├── domestic_stock/    ← 국내주식 전체 기능별 예제
│   ├── auth/              ← REST 토큰 / WebSocket 토큰
│   └── convention.md      ← 네이밍 규칙, 공통 패턴
├── docs/                  ← 기능별 상세 문서
└── kis_devlp.yaml         ← TR ID 목록, 엔드포인트 정의
```

### 작업별 읽어야 할 파일

| 작업 | 읽을 파일 |
|---|---|
| 토큰 발급 / 헤더 구조 | `examples_llm/kis_auth.py` |
| 주식현재가 조회 | `examples_llm/domestic_stock/inquire_price/inquire_price.py` |
| 1분봉 차트 | `examples_llm/domestic_stock/inquire_time_itemchartprice/inquire_time_itemchartprice.py` |
| 일별 차트 | `examples_llm/domestic_stock/inquire_daily_itemchartprice/inquire_daily_itemchartprice.py` |
| 10단 호가 | `examples_llm/domestic_stock/inquire_asking_price_exp_ccn/inquire_asking_price_exp_ccn.py` |
| 체결 내역 | `examples_llm/domestic_stock/inquire_time_itemconclusion/inquire_time_itemconclusion.py` |
| 매수/매도 주문 | `examples_llm/domestic_stock/order_cash/order_cash.py` |
| 주문 정정/취소 | `examples_llm/domestic_stock/order_rvsecncl/order_rvsecncl.py` |
| 잔고 조회 | `examples_llm/domestic_stock/inquire_balance/inquire_balance.py` |
| 체결 가능 수량 | `examples_llm/domestic_stock/inquire_psbl_order/inquire_psbl_order.py` |
| hashkey 생성 | `examples_llm/kis_auth.py` 내 hashkey 섹션 |

### 핵심 주의사항 (실제 개발 중 발견한 gotcha)

```
1. 시세 조회(FHKST*)는 모의투자 서버 미지원
   → FID_COND_MRKT_DIV_CODE 사용 API는 항상 실전 서버(openapi:9443) 호출
   → 모의투자 자격증명으로도 실전 서버 시세 조회 가능

2. inquire-time-itemchartprice (1분봉) rate limit
   → 호출 간격 최소 1.0초 필요 (0.5초 → 4번째 호출부터 500 에러)

3. 1분봉 페이지네이션
   → 1회 30건 반환, 이전 구간은 FID_INPUT_HOUR_1 = earliest - 1분으로 재조회
   → stck_bsop_date 필터로 오늘 데이터만 선별 (이전날 데이터 섞임 주의)

4. 주문 시 hashkey 필수
   → POST /uapi/domestic-stock/v1/trading/order-cash 요청 전
   → POST /uapi/hashkey 로 body 해시 생성 후 헤더에 포함

5. 잔고조회 TR ID
   → 실계좌: TTTC8434R  / 모의투자: VTTC8434R
   → 계좌번호 = CANO(앞 8자리) + ACNT_PRDT_CD(뒤 2자리) 분리해서 파라미터 전달
```
---

## 11. 협업 규칙

1. **merge 전 카톡 알림** — "나 지금 dev에 merge할게" 한마디 필수
2. **2~3일에 한 번 dev merge** — 너무 오래 쌓이면 충돌이 커짐
3. **types/index.ts 수정 시 반드시 상대방 확인** — 양쪽 코드에 즉시 영향
4. **API 명세 변경 시 이 문서 업데이트** — agent가 잘못된 정보로 작업하지 않도록
5. **Phase 완료 기준** — 해당 Phase의 API + UI가 로컬에서 정상 동작할 때

---

## 12. 현재 진행 상태

> 이 섹션은 작업이 진행되면서 업데이트할 것.

- [x] Phase 1 — MVP (프론트엔드 완료 2026-06-01)
- [x] Phase 2 — 실시간 시세 + 차트 (프론트엔드 완료 2026-06-02)
- [x] Phase 3 — AI 기능 (프론트엔드 완료 2026-06-02)
- [x] Phase 4 — 거래 + 포트폴리오 (프론트엔드 완료 2026-06-03)
