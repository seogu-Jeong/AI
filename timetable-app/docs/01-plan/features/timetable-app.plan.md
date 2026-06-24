# Plan: Timetable & Calendar App

## Overview
학부 수업, 공부 시간, 수면 시간을 관리할 수 있는 30분 단위 그리드 기반 캘린더 애플리케이션입니다.

## Goals
- 학부 수업(Class), 공부(Study), 수면(Sleep) 시간의 시각적 구분
- 월간(Monthly) 및 주간(Weekly) 뷰 제공
- 30분 단위의 상세한 주간 시간표 지원
- 모던하고 깔끔한 UI/UX 제공

## Target Features
1. **Event Management**
   - 수업, 공부, 수면 등 일정 추가/수정/삭제
   - 일정 타입에 따른 자동 색상 지정
2. **Calendar Views**
   - 월간 뷰: 전체적인 일정 흐름 파악
   - 주간 뷰: 오늘 포함 향후 7일간의 30분 단위 상세 일정
3. **Responsive UI**
   - Vanilla CSS를 사용한 깔끔한 레이아웃

## Technical Stack
- Frontend: React (TypeScript) + Vite
- Styling: Vanilla CSS
- Libraries: date-fns (날짜 계산), lucide-react (아이콘)
- Data Persistence: LocalStorage (Prototype)
