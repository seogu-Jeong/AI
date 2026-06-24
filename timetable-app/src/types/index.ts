export type EventType = 'CLASS' | 'STUDY' | 'SLEEP' | 'OTHER';

export interface TimetableEvent {
  id: string;
  title: string;
  type: EventType;
  start: Date;
  end: Date;
  color?: string;
  description?: string;
}

export interface CalendarState {
  currentDate: Date;
  viewMode: 'MONTH' | 'WEEK';
  events: TimetableEvent[];
}
