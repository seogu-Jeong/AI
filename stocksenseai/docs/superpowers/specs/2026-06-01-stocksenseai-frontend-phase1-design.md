# StockSenseAI Frontend Phase 1 — Design

**Date:** 2026-06-01
**Author:** seogu-Jeong
**Branch:** seogu-Jeong

---

## 1. 목표

CLAUDE.md Phase 1 프론트엔드 요구사항 구현:
- Vite + TypeScript + shadcn/ui 프로젝트 초기 세팅
- 반응형 레이아웃 (데스크탑 3단 / 모바일 상단탭)
- 로그인/회원가입 화면
- Zustand 스토어 + Axios JWT interceptor
- LandingPage + Mock 데이터로 차트탭 UI

---

## 2. 기술 스택

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| React | 18 | UI 프레임워크 |
| TypeScript | 5 | 타입 안전성 |
| Vite | 5 | 빌드 도구 |
| shadcn/ui | latest | 컴포넌트 |
| Tailwind CSS | v4 | 스타일링 |
| Zustand | 4 | 상태 관리 |
| Axios | 1.6 | HTTP + JWT interceptor |
| Lightweight Charts | 4 | 캔들스틱 차트 |
| Recharts | 2.x | 비교 차트 |
| lucide-react | latest | 아이콘 |

---

## 3. 디렉토리 구조

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/                  # shadcn/ui 컴포넌트
│   │   ├── Layout/
│   │   │   ├── MainLayout.tsx   # 전체 레이아웃 쉘
│   │   │   ├── Header.tsx       # 상단 헤더 (검색, 다크모드 토글, 프로필)
│   │   │   └── MobileTabBar.tsx # 모바일 상단 스크롤 탭
│   │   ├── Sidebar/
│   │   │   ├── Sidebar.tsx      # 좌측 사이드바 컨테이너
│   │   │   ├── StockGroup.tsx   # 종목 그룹 목록
│   │   │   └── StockList.tsx    # 종목 리스트
│   │   ├── MainPanel/
│   │   │   ├── MainPanel.tsx    # 탭 라우팅
│   │   │   └── ChartTab/
│   │   │       └── ChartTab.tsx # Mock 캔들스틱 차트
│   │   ├── WatchlistPanel/
│   │   │   └── WatchlistPanel.tsx # 하단 고정 워치리스트
│   │   └── auth/
│   │       ├── LoginModal.tsx
│   │       └── RegisterModal.tsx
│   ├── store/
│   │   ├── authStore.ts         # 인증 상태
│   │   ├── stockStore.ts        # 선택 종목, 관심종목
│   │   └── uiStore.ts           # darkMode, activeTab, sidebarOpen
│   ├── lib/
│   │   ├── api.ts               # Axios 인스턴스 + JWT interceptor
│   │   └── utils.ts             # shadcn/ui cn() 유틸
│   ├── pages/
│   │   └── LandingPage.tsx
│   ├── types/
│   │   └── index.ts             # 공동 관리 타입 (CLAUDE.md 기준)
│   └── main.tsx
├── index.html
├── vite.config.ts
├── tailwind.config.ts
└── package.json
```

---

## 4. 레이아웃

### 데스크탑 (≥768px) — 3단 분할

```
┌─────────────────────────────────────────────────┐
│  Header (로고 | 검색 | 다크모드 토글 | 프로필)      │
├───────────┬─────────────────────┬───────────────┤
│           │  탭: 차트|AI|시뮬|   │               │
│  Sidebar  │  포트|스크리너|백테  │  우측 패널     │
│  (종목    │                     │  (Phase 4에서  │
│   그룹 +  │  MainPanel          │   호가창 +     │
│   리스트) │  (차트탭 등)         │   주문창 구현) │
│           │                     │               │
├───────────┴─────────────────────┴───────────────┤
│  WatchlistPanel (하단 고정 — 실시간 시세 티커)     │
└─────────────────────────────────────────────────┘
```

### 모바일 (<768px) — 상단 스크롤 탭

```
┌─────────────────────┐
│  Header (로고 | 검색) │
├─────────────────────┤
│ [차트][AI][시뮬][포트] │  ← 가로 스크롤 탭
├─────────────────────┤
│                     │
│   MainPanel         │
│   (전체 화면)        │
│                     │
└─────────────────────┘
```

- 모바일에서 Sidebar는 숨김. Header의 햄버거 버튼으로 드로어로 열림.
- WatchlistPanel은 모바일에서 숨김.

---

## 5. 인증 플로우

```
LandingPage
  ↓ (로그인 버튼)
LoginModal
  → POST /auth/login
  → access_token을 메모리(Zustand)에 저장
  → refresh_token은 httpOnly cookie (백엔드 처리)
  → 성공 시 MainLayout으로 이동

RegisterModal
  → POST /auth/register
  → POST /auth/verify-email (이메일 인증)
  → 완료 후 LoginModal로 이동
```

Axios interceptor:
- 모든 요청에 `Authorization: Bearer <access_token>` 헤더 자동 첨부
- 401 응답 시 → POST /auth/refresh → 성공 시 원래 요청 재시도
- refresh 실패 시 → authStore 초기화 + LandingPage로 리다이렉트

---

## 6. Zustand 스토어

```typescript
// authStore
{
  user: User | null
  accessToken: string | null
  login: (email, password) => Promise<void>
  logout: () => void
  setUser: (user: User) => void
}

// stockStore
{
  selectedStock: { code: string; name: string } | null
  watchlist: string[]  // stock codes
  setSelectedStock: (stock) => void
  addToWatchlist: (code) => void
  removeFromWatchlist: (code) => void
}

// uiStore
{
  darkMode: boolean
  activeTab: 'chart' | 'ai' | 'simulator' | 'portfolio' | 'screener' | 'backtest'
  sidebarOpen: boolean
  toggleDarkMode: () => void
  setActiveTab: (tab) => void
  toggleSidebar: () => void
}
```

---

## 7. Mock 데이터 (Phase 1)

백엔드 미완성 상태에서 차트탭 UI 개발을 위해 mock 데이터 사용.

- `src/lib/mockData.ts`에 mock candle 데이터 정의
- Lightweight Charts에 mock OHLCV 데이터 렌더링
- 백엔드 완성 후 `GET /stocks/{code}/chart` API 호출로 교체

---

## 8. 다크모드

- Tailwind `dark:` 클래스 사용
- `<html>` 태그에 `dark` 클래스 토글
- uiStore.darkMode 상태와 동기화
- 기본값: dark

---

## 9. 환경변수

```
VITE_API_BASE=http://localhost:8000
```

---

## 10. 구현 순서

1. `frontend/` 디렉토리 Vite 프로젝트 초기 세팅
2. shadcn/ui + Tailwind CSS v4 설정
3. 라이브러리 설치 (Zustand, Axios, Lightweight Charts 등)
4. types/index.ts — CLAUDE.md 기준 공통 타입 정의
5. lib/api.ts — Axios 인스턴스 + JWT interceptor
6. Zustand 스토어 3개 (auth, stock, ui)
7. MainLayout + Header + Sidebar + WatchlistPanel
8. MobileTabBar (모바일 상단 스크롤 탭)
9. LandingPage + LoginModal + RegisterModal
10. MainPanel 탭 라우팅 + ChartTab (Mock 캔들스틱)
