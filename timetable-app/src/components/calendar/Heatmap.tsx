import React, { useState, useEffect } from 'react';
import { format, startOfWeek, addDays, isSameDay } from 'date-fns';
import './Heatmap.css';

interface HeatmapProps {
  allMemberEvents: any[]; // 모든 팀원의 일정 데이터
  memberCount: number;
}

const Heatmap: React.FC<HeatmapProps> = ({ allMemberEvents, memberCount }) => {
  const hours = Array.from({ length: 48 }, (_, i) => i);
  const days = ['월', '화', '수', '목', '금', '토', '일'];

  // 각 셀당 바쁜 사람 수 계산
  const getBusyCount = (dayIndex: number, hourIndex: number) => {
    // 실제 구현에서는 요일(dayOfWeek)과 시간(startTime/endTime)을 비교하여 계산
    // 여기서는 예시 로직만 포함
    return allMemberEvents.filter(e => {
      const startSlot = e.startHour * 2 + (e.startMin >= 30 ? 1 : 0);
      const endSlot = e.endHour * 2 + (e.endMin >= 30 ? 1 : 0);
      return e.dayOfWeek === dayIndex && hourIndex >= startSlot && hourIndex < endSlot;
    }).length;
  };

  return (
    <div className="heatmap-container">
      <div className="heatmap-grid">
        <div className="time-col">
          <div className="cell-header"></div>
          {hours.map(h => (
            <div key={h} className="time-label">
              {h % 2 === 0 ? `${Math.floor(h / 2)}:00` : ''}
            </div>
          ))}
        </div>
        
        {days.map((day, dIdx) => (
          <div key={day} className="day-col">
            <div className="cell-header">{day}</div>
            {hours.map(h => {
              const busyCount = getBusyCount(dIdx + 1, h); // 1(월)~7(일)
              const opacity = busyCount > 0 ? (busyCount / memberCount) * 0.8 + 0.1 : 0;
              
              return (
                <div 
                  key={h} 
                  className="heatmap-cell"
                  style={{ backgroundColor: busyCount > 0 ? `rgba(79, 70, 229, ${opacity})` : 'transparent' }}
                  title={`${busyCount}명 바쁨`}
                ></div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Heatmap;
