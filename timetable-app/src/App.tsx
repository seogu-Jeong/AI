import React, { useState, useEffect } from 'react';
import { format, addMonths, subMonths, addDays, subDays } from 'date-fns';
import { ChevronLeft, ChevronRight, Users, Calendar as CalendarIcon, LogOut, Plus } from 'lucide-react';
import PersonalCalendar from './PersonalCalendar'; // 기존 캘린더 로직 분리
import Heatmap from './components/calendar/Heatmap';
import './App.css';

const App: React.FC = () => {
  const [user, setUser] = useState<any>(null); // null이면 로그인 전
  const [view, setView] = useState<'PERSONAL' | 'TEAM'>('PERSONAL');
  const [currentDate, setCurrentDate] = useState(new Date());

  // 임시 로그인 처리 (실제 구현에서는 Supabase Auth 사용)
  const mockLogin = () => {
    setUser({ id: 'user1', name: '홍길동' });
  };

  // 모든 멤버의 더미 데이터 (히트맵 테스트용)
  const dummyTeamEvents = [
    { dayOfWeek: 1, startHour: 10, startMin: 0, endHour: 12, endMin: 0 }, // 월요일 10-12시
    { dayOfWeek: 1, startHour: 11, startMin: 0, endHour: 13, endMin: 0 }, // 월요일 11-13시 (중첩)
    { dayOfWeek: 3, startHour: 14, startMin: 0, endHour: 16, endMin: 0 }, // 수요일 14-16시
  ];

  if (!user) {
    return (
      <div className="login-screen">
        <div className="login-card">
          <h1>Smart Timetable</h1>
          <p>팀원들과 시간표를 공유하고 미팅 시간을 잡아보세요.</p>
          <button className="login-btn" onClick={mockLogin}>시작하기 (Guest)</button>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <header className="calendar-header">
        <div className="header-left">
          <div className="view-switcher">
            <button className={view === 'PERSONAL' ? 'active' : ''} onClick={() => setView('PERSONAL')}>
              <CalendarIcon size={18} /> 내 시간표
            </button>
            <button className={view === 'TEAM' ? 'active' : ''} onClick={() => setView('TEAM')}>
              <Users size={18} /> 팀 통합 뷰
            </button>
          </div>
        </div>
        
        <div className="header-center">
          <h2>{view === 'PERSONAL' ? format(currentDate, 'yyyy년 M월') : '팀 공강 시간 확인 (Heatmap)'}</h2>
        </div>

        <div className="header-right">
          <span className="user-info">{user.name}님 환영합니다</span>
          <button className="logout-btn" onClick={() => setUser(null)}><LogOut size={18} /></button>
        </div>
      </header>

      <main className="calendar-content">
        {view === 'PERSONAL' ? (
          <PersonalCalendar /> 
        ) : (
          <Heatmap allMemberEvents={dummyTeamEvents} memberCount={5} />
        )}
      </main>
    </div>
  );
};

export default App;
