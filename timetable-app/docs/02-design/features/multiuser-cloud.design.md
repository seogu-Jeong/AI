# Design: Cloud Multi-User Timetable

## UI/UX Design

### 1. Authentication Pages
- **Login/Signup**: 이메일 기반 심플한 폼
- **Profile**: 닉네임 설정 및 로그아웃

### 2. Dashboard & Navigation
- 좌측 사이드바: 내 캘린더, 소속 그룹 리스트, 그룹 생성 버튼
- 상단바: 현재 뷰 모드(개인/팀), 프로필 정보

### 3. Personal Calendar View
- 기존 30분 단위 주간/월간 뷰 유지
- 데이터는 Supabase `events` 테이블과 동기화

### 4. Team Heatmap View (Core Feature)
- **Grid**: 월~일(7열) x 00:00~24:00(48행, 30분 단위)
- **Visualization**: 
  - 각 셀의 투명도(Opacity) = (해당 시간 바쁜 멤버 수 / 전체 멤버 수)
  - 색상: Indigo (진할수록 바쁨)
- **Interaction**: 셀 클릭 시 해당 시간에 일정이 있는 멤버 이름 팝업 노출

## Component Architecture
- `App`: 메인 레이아웃 및 Auth 상태 분기
- `AuthContainer`: 로그인/회원가입 로직
- `CalendarContainer`: 개인/팀 뷰 전환 및 데이터 페칭
- `HeatmapGrid`: 팀원들의 일정을 집계하여 렌더링하는 핵심 로직

## Data Flow
1. 사용자가 로그인하면 `AuthContext`에 세션 정보 저장
2. `useEvents` 훅이 현재 사용자의 일정 또는 소속 그룹원들의 일정을 Supabase에서 실시간 구독(Subscribe)
3. 데이터 변경 시 `events` 상태가 업데이트되며 UI 자동 리렌더링
