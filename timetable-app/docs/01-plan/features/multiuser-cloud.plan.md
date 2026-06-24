# Plan: Cloud Multi-User Timetable

## Overview
기존 로컬(LocalStorage) 기반의 개인 캘린더를 클라우드 기반의 멀티유저 시스템으로 업그레이드합니다.
핵심은 팀원들이 각자의 일정을 관리하고, **모든 팀원의 시간표를 겹쳐 보아 미팅 가능한 '공강' 시간을 쉽게 찾는 것**입니다.

## Goals
- **접근성**: 브라우저만 있으면 기기(PC, Mobile)에 상관없이 접속 가능 (Vercel 배포)
- **개인화**: 로그인 시스템을 도입하여 개인의 시간표와 데이터를 안전하게 관리
- **협업**: 특정 그룹(팀/동아리)을 만들고 멤버를 초대하여 서로의 시간표를 공유
- **통합 뷰(Heatmap Grid)**: 날짜 지정 없이 요일(월~일) 기준으로 모든 멤버의 일정을 반투명 블록으로 겹쳐 표시. 
  (색이 진할수록 바쁜 시간이 많음 = 색이 없는 빈 칸이 최적의 미팅 시간)

## Target Features
1. **Auth & Identity**
   - 이메일/비밀번호 기반 회원가입 및 로그인
   - 닉네임 설정
2. **Group Management**
   - 그룹(Workspace) 생성 및 고유 초대 링크(또는 코드) 복사
   - 그룹 멤버 리스트 확인
3. **Cloud Event Sync**
   - 개인 일정 CRUD(생성/읽기/수정/삭제)를 클라우드 DB(BaaS)에 실시간 동기화
4. **Team Availability View (Heatmap)**
   - 개인 뷰(내 캘린더)와 팀 뷰(통합 시간표) 토글 스위치
   - 특정 시간대를 클릭하면 "누가 바쁜지" 툴팁으로 표시

## Technical Stack
- **Frontend**: React (TypeScript), Vite, TailwindCSS (Heatmap UI 구현의 편의를 위해 도입 고려)
- **Backend / Database**: Supabase 또는 Firebase (Authentication, Realtime Database)
- **Hosting**: Vercel
