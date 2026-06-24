# 2026-06-19 seogu-Jeong 수정 내역 #2

브랜치: `seogu-Jeong` → `dev` 머지

---

## 백테스트 자산추이 차트 라인 미표시 버그

### 증상
백테스트 결과 화면에서 "자산 추이" 차트의 축과 그리드는 정상 표시되나,
실제 데이터 라인(선)이 전혀 보이지 않는 문제.

### 원인
**파일:** `frontend/src/components/MainPanel/BacktestTab/BacktestTab.tsx:415`

Recharts `<Line>` 컴포넌트의 `stroke` 속성에 CSS 변수를 사용:
```tsx
stroke="hsl(var(--primary))"
```

`--primary` 변수가 shadcn/ui Tailwind v4 기준으로 `oklch()` 포맷으로 정의됨:
```css
--primary: oklch(0.922 0 0);
```

따라서 실제 해석되는 값이 `hsl(oklch(0.922 0 0))`이 되어 **유효하지 않은 CSS** → SVG `stroke` 속성이 무효화되어 라인이 렌더링되지 않음.

Recharts의 `<Line>` 컴포넌트는 SVG `stroke` 어트리뷰트로 색상을 지정하는데, SVG 어트리뷰트는 CSS custom property(`var()`)의 oklch 포맷을 올바르게 해석하지 못함.

### 수정
```tsx
// 수정 전
<Line ... stroke="hsl(var(--primary))" ... />

// 수정 후
<Line ... stroke="#58a6ff" ... />
```

하드코딩 색상 `#58a6ff`는 `PortfolioTab.tsx`의 자산 추이 차트에서 이미 사용 중인 값과 동일하게 통일.

### 영향 파일
| 파일 | 변경 내용 |
|---|---|
| `frontend/src/components/MainPanel/BacktestTab/BacktestTab.tsx` | Line stroke CSS 변수 → 하드코딩 hex 색상 |
