# Coding Conventions (Cloud Fullstack)

## Naming Conventions
- **Components**: PascalCase (e.g., `GroupHeatmap.tsx`)
- **Services/API**: camelCase (e.g., `supabase-service.ts`)
- **Hooks**: `use` prefix (e.g., `useAuth.ts`, `useEvents.ts`)

## Directory Structure (Next.js/React App Router Style)
- `src/components/common`: 공통 UI (Button, Input, Modal)
- `src/components/calendar`: 캘린더 관련 컴포넌트
- `src/components/auth`: 로그인/회원가입 컴포넌트
- `src/lib/supabase`: Supabase 클라이언트 설정 및 API 호출 로직
- `src/contexts`: AuthContext 등 전역 상태 관리
- `src/types`: DB 스키마와 일치하는 TypeScript 타입

## Supabase Rules
- 모든 DB 요청은 `src/lib/supabase` 내의 서비스 함수를 통해서만 수행
- RLS(Row Level Security)를 활성화하여 본인의 데이터만 수정 가능하도록 설정
- 실시간 동기화(Realtime)가 필요한 경우 `subscribe` 패턴 사용
