# Schema & Terminology (Cloud Version)

## Terms

| Term | Definition | Example |
|------|------------|---------|
| User | 서비스에 가입한 개별 사용자 | student@univ.edu |
| Group | 일정을 공유하는 팀/동아리 단위 | "캡스톤 디자인 1조" |
| Event | 개인 캘린더에 등록된 개별 일정 | "운영체제 수업" |
| Heatmap | 여러 일정을 겹쳐 시각화한 통합 뷰 | 겹친 블록 (진한 색 = 바쁨) |

## Entity Definitions

```typescript
type EventType = 'CLASS' | 'STUDY' | 'SLEEP' | 'OTHER';

interface User {
  id: string; // UUID
  email: string;
  nickname: string;
  createdAt: string;
}

interface Group {
  id: string; // UUID
  name: string;
  inviteCode: string;
  ownerId: string;
  createdAt: string;
}

interface GroupMember {
  groupId: string;
  userId: string;
  joinedAt: string;
}

interface TimetableEvent {
  id: string; // UUID
  userId: string; // 일정을 생성한 주체
  title: string;
  type: EventType;
  // 통합 뷰(Heatmap) 계산을 위해 반복 요일 데이터 추가 고려
  dayOfWeek?: number; // 0(일) ~ 6(토)
  startTime: string;  // "09:00"
  endTime: string;    // "10:30"
  specificDate?: string; // "2026-03-11" (단발성 일정인 경우)
}
```

## Relationship Diagram

```
User (1) ──── (N) GroupMember (N) ──── (1) Group
  │
  └── (N) TimetableEvent
```
