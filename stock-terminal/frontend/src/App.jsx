import { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:8000/api';

export default function App() {
  const [activePage, setActivePage] = useState('Dashboard');
  const [marketOverview, setMarketOverview] = useState(null);
  const [stocks, setStocks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [currentTime, setCurrentTime] = useState(new Date());

  const fetchMarketData = async () => {
    try {
      const moRes = await fetch(`${API_BASE}/market-overview`);
      const moData = await moRes.json();
      setMarketOverview(moData);

      const stRes = await fetch(`${API_BASE}/stocks`);
      const stData = await stRes.json();
      // Sort by absolute change pct for top movers naturally or keep random order
      setStocks(stData);
      setLoading(false);
    } catch (e) {
      console.error('Failed to fetch data:', e);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMarketData();
    const dataInterval = setInterval(() => {
      fetchMarketData(); // Auto refresh every 30s
    }, 30000);
    
    const timeInterval = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => {
      clearInterval(dataInterval);
      clearInterval(timeInterval);
    };
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw' }}>
      {/* Top Bar */}
      <header style={{ height: '40px', backgroundColor: 'var(--sidebar-bg)', borderBottom: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 20px', fontSize: '12px' }}>
        <div style={{ color: 'var(--accent-blue)', fontWeight: 'bold', letterSpacing: '1px' }}>STOCK TERMINAL</div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <span style={{ backgroundColor: 'var(--card-bg)', padding: '4px 10px', borderRadius: '12px', border: '1px solid var(--border-color)' }}>S&P500</span>
          <span style={{ backgroundColor: 'var(--card-bg)', padding: '4px 10px', borderRadius: '12px', border: '1px solid var(--border-color)' }}>{stocks.length || 50} 종목</span>
          <span style={{ backgroundColor: 'rgba(0, 214, 143, 0.1)', color: 'var(--green)', padding: '4px 10px', borderRadius: '12px', border: '1px solid var(--green)' }}>AI 분석 완료</span>
        </div>
        <div className="mono" style={{ color: 'var(--text-secondary)' }}>
          {currentTime.toLocaleDateString()} {currentTime.toLocaleTimeString()}
        </div>
      </header>

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Sidebar */}
        <aside style={{ width: '220px', backgroundColor: 'var(--sidebar-bg)', borderRight: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column' }}>
          <nav style={{ padding: '20px 0', flex: 1 }}>
            {['Dashboard', '종목 스크리너', '포트폴리오', '뉴스 & 감성', 'AI 분석'].map(item => (
              <div 
                key={item} 
                onClick={() => setActivePage(item)}
                style={{ 
                  padding: '12px 20px', 
                  cursor: 'pointer', 
                  color: activePage === item ? 'var(--text-primary)' : 'var(--text-secondary)',
                  borderLeft: `3px solid ${activePage === item ? 'var(--accent-blue)' : 'transparent'}`,
                  backgroundColor: activePage === item ? 'rgba(21, 101, 255, 0.1)' : 'transparent',
                  transition: 'all 0.2s',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px'
                }}
              >
                {/* Mock Icons via emoji or text for simplicity since no lib is specified */}
                {item === 'Dashboard' && '📊'}
                {item === '종목 스크리너' && '🔍'}
                {item === '포트폴리오' && '💼'}
                {item === '뉴스 & 감성' && '📰'}
                {item === 'AI 분석' && '🧠'}
                <span style={{ marginLeft: '5px' }}>{item}</span>
              </div>
            ))}
          </nav>
          <div style={{ padding: '15px', borderTop: '1px solid var(--border-color)', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--green)', boxShadow: '0 0 5px var(--green)' }}></div>
            System Online
          </div>
        </aside>

        {/* Main Content */}
        <main style={{ flex: 1, padding: '24px', overflowY: 'auto', backgroundColor: 'var(--bg-color)' }}>
          {loading ? (
            <div style={{ color: 'var(--text-secondary)', display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
              데이터를 불러오는 중...
            </div>
          ) : (
            <>
              {activePage === 'Dashboard' && <Dashboard marketOverview={marketOverview} stocks={stocks} />}
              {activePage === '종목 스크리너' && <Screener stocks={stocks} />}
              {activePage === '뉴스 & 감성' && <News />}
              {activePage === 'AI 분석' && <AIAnalysis stocks={stocks} />}
              {activePage === '포트폴리오' && <div style={{ color: 'var(--text-secondary)' }}>포트폴리오 서비스 준비 중...</div>}
            </>
          )}
        </main>
      </div>
    </div>
  );
}

function Dashboard({ marketOverview, stocks }) {
  if (!marketOverview) return null;
  
  // Sort for top movers (by absolute change %)
  const topMovers = [...stocks].sort((a, b) => Math.abs(b.change_pct) - Math.abs(a.change_pct)).slice(0, 10);
  
  // Calc AI signal distribution
  const signals = { BUY: 0, SELL: 0, HOLD: 0 };
  stocks.forEach(s => signals[s.signal]++);
  const total = stocks.length || 1;
  const buyPct = (signals.BUY / total) * 100;
  const holdPct = (signals.HOLD / total) * 100;

  return (
    <div className="fade-in">
      <h2 style={{ marginBottom: '24px', fontWeight: '500' }}>Market Overview</h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginBottom: '32px' }}>
        {Object.entries(marketOverview).map(([name, data]) => {
          const isPos = data.change >= 0;
          return (
            <div key={name} style={{ backgroundColor: 'var(--card-bg)', padding: '20px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '12px' }}>{name}</div>
              <div className="mono" style={{ fontSize: '24px', fontWeight: 'bold' }}>{data.value.toLocaleString(undefined, {minimumFractionDigits: 2})}</div>
              <div className={`mono ${isPos ? 'text-green' : 'text-red'}`} style={{ fontSize: '14px', marginTop: '8px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                {isPos ? '▲' : '▼'} {Math.abs(data.change)} ({isPos ? '+' : ''}{data.change_pct}%)
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
        <div>
          <h3 style={{ marginBottom: '16px', fontWeight: '500' }}>Top Movers</h3>
          <div style={{ backgroundColor: 'var(--card-bg)', border: '1px solid var(--border-color)', borderRadius: '8px', overflow: 'hidden' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-secondary)', fontSize: '13px', backgroundColor: 'var(--sidebar-bg)' }}>
                  <th style={{ padding: '12px 16px' }}>Symbol</th>
                  <th style={{ padding: '12px 16px', textAlign: 'right' }}>Price</th>
                  <th style={{ padding: '12px 16px', textAlign: 'right' }}>Change %</th>
                  <th style={{ padding: '12px 16px', textAlign: 'center' }}>AI Signal</th>
                </tr>
              </thead>
              <tbody>
                {topMovers.map(s => (
                  <tr key={s.symbol} className="table-row">
                    <td style={{ padding: '12px 16px', fontWeight: 'bold' }}>{s.symbol}</td>
                    <td className="mono" style={{ padding: '12px 16px', textAlign: 'right' }}>${s.price.toFixed(2)}</td>
                    <td className={`mono ${s.change_pct >= 0 ? 'text-green' : 'text-red'}`} style={{ padding: '12px 16px', textAlign: 'right' }}>{s.change_pct >= 0 ? '+' : ''}{s.change_pct}%</td>
                    <td style={{ padding: '12px 16px', textAlign: 'center' }}><SignalBadge signal={s.signal} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <h3 style={{ marginBottom: '16px', fontWeight: '500' }}>AI Signal Distribution</h3>
          <div style={{ backgroundColor: 'var(--card-bg)', padding: '24px', borderRadius: '8px', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            {/* CSS Only Stacked Bar Chart */}
            <div style={{ width: '100%', height: '30px', borderRadius: '4px', overflow: 'hidden', display: 'flex' }}>
              <div style={{ width: `${buyPct}%`, backgroundColor: 'var(--green)', transition: 'width 1s' }}></div>
              <div style={{ width: `${holdPct}%`, backgroundColor: 'var(--yellow)', transition: 'width 1s' }}></div>
              <div style={{ flex: 1, backgroundColor: 'var(--red)', transition: 'width 1s' }}></div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--green)' }}></div>
                  <span>BUY</span>
                </div>
                <span className="mono">{signals.BUY} ({Math.round(buyPct)}%)</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--yellow)' }}></div>
                  <span>HOLD</span>
                </div>
                <span className="mono">{signals.HOLD} ({Math.round(holdPct)}%)</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--red)' }}></div>
                  <span>SELL</span>
                </div>
                <span className="mono">{signals.SELL} ({Math.round(100 - buyPct - holdPct)}%)</span>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}

function Screener({ stocks }) {
  const [sortConfig, setSortConfig] = useState({ key: 'ai_score', direction: 'desc' });
  const [filterSignal, setFilterSignal] = useState('ALL');

  const sortedStocks = [...stocks]
    .filter(s => filterSignal === 'ALL' || s.signal === filterSignal)
    .sort((a, b) => {
      if (a[sortConfig.key] < b[sortConfig.key]) return sortConfig.direction === 'asc' ? -1 : 1;
      if (a[sortConfig.key] > b[sortConfig.key]) return sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    });

  const requestSort = (key) => {
    let direction = 'desc';
    if (sortConfig.key === key && sortConfig.direction === 'desc') {
      direction = 'asc';
    }
    setSortConfig({ key, direction });
  };

  const getSortIcon = (key) => {
    if (sortConfig.key !== key) return ' ↕';
    return sortConfig.direction === 'asc' ? ' ↑' : ' ↓';
  };

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h2 style={{ fontWeight: '500', margin: 0 }}>종목 스크리너</h2>
        
        <div style={{ display: 'flex', gap: '10px' }}>
          <span style={{ color: 'var(--text-secondary)', alignSelf: 'center', fontSize: '14px' }}>Signal Filter:</span>
          {['ALL', 'BUY', 'HOLD', 'SELL'].map(sig => (
            <button
              key={sig}
              onClick={() => setFilterSignal(sig)}
              style={{
                background: filterSignal === sig ? 'var(--accent-blue)' : 'var(--card-bg)',
                color: filterSignal === sig ? '#fff' : 'var(--text-secondary)',
                border: '1px solid var(--border-color)',
                padding: '6px 16px',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '13px'
              }}
            >
              {sig}
            </button>
          ))}
        </div>
      </div>

      <div style={{ backgroundColor: 'var(--card-bg)', border: '1px solid var(--border-color)', borderRadius: '8px', overflow: 'auto', flex: 1 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: '800px' }}>
          <thead style={{ position: 'sticky', top: 0, backgroundColor: 'var(--sidebar-bg)', zIndex: 1, boxShadow: '0 1px 0 var(--border-color)' }}>
            <tr style={{ textAlign: 'left', color: 'var(--text-secondary)', fontSize: '13px' }}>
              <th onClick={() => requestSort('symbol')} style={{ padding: '16px', cursor: 'pointer' }}>Symbol{getSortIcon('symbol')}</th>
              <th onClick={() => requestSort('name')} style={{ padding: '16px', cursor: 'pointer' }}>Name{getSortIcon('name')}</th>
              <th onClick={() => requestSort('price')} style={{ padding: '16px', textAlign: 'right', cursor: 'pointer' }}>Price{getSortIcon('price')}</th>
              <th onClick={() => requestSort('change_pct')} style={{ padding: '16px', textAlign: 'right', cursor: 'pointer' }}>Change %{getSortIcon('change_pct')}</th>
              <th onClick={() => requestSort('ai_score')} style={{ padding: '16px', cursor: 'pointer' }}>AI Score{getSortIcon('ai_score')}</th>
              <th onClick={() => requestSort('signal')} style={{ padding: '16px', textAlign: 'center', cursor: 'pointer' }}>Signal{getSortIcon('signal')}</th>
              <th onClick={() => requestSort('volume')} style={{ padding: '16px', textAlign: 'right', cursor: 'pointer' }}>Volume{getSortIcon('volume')}</th>
              <th onClick={() => requestSort('market_cap')} style={{ padding: '16px', textAlign: 'right', cursor: 'pointer' }}>Market Cap{getSortIcon('market_cap')}</th>
            </tr>
          </thead>
          <tbody>
            {sortedStocks.map(s => (
              <tr key={s.symbol} className="table-row">
                <td style={{ padding: '14px 16px', fontWeight: 'bold' }}>{s.symbol}</td>
                <td style={{ padding: '14px 16px', color: 'var(--text-secondary)', fontSize: '13px' }}>{s.name}</td>
                <td className="mono" style={{ padding: '14px 16px', textAlign: 'right' }}>${s.price.toFixed(2)}</td>
                <td className={`mono ${s.change_pct >= 0 ? 'text-green' : 'text-red'}`} style={{ padding: '14px 16px', textAlign: 'right' }}>{s.change_pct >= 0 ? '+' : ''}{s.change_pct}%</td>
                <td style={{ padding: '14px 16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{ width: '60px', height: '6px', backgroundColor: 'var(--bg-color)', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{ width: `${s.ai_score}%`, height: '100%', backgroundColor: s.ai_score > 70 ? 'var(--green)' : s.ai_score < 40 ? 'var(--red)' : 'var(--yellow)' }}></div>
                    </div>
                    <span className="mono" style={{ fontSize: '13px' }}>{s.ai_score}</span>
                  </div>
                </td>
                <td style={{ padding: '14px 16px', textAlign: 'center' }}><SignalBadge signal={s.signal} /></td>
                <td className="mono" style={{ padding: '14px 16px', textAlign: 'right', color: 'var(--text-secondary)' }}>{s.volume}</td>
                <td className="mono" style={{ padding: '14px 16px', textAlign: 'right', color: 'var(--text-secondary)' }}>{s.market_cap}</td>
              </tr>
            ))}
            {sortedStocks.length === 0 && (
              <tr>
                <td colSpan="8" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>매칭되는 종목이 없습니다.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function News() {
  const [news, setNews] = useState([]);
  const symbols = ['NVDA', 'AAPL', 'MSFT', 'META', 'AMZN'];
  const [activeSymbol, setActiveSymbol] = useState('NVDA');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/news/${activeSymbol}`)
      .then(r => r.json())
      .then(d => {
        setNews(d);
        setLoading(false);
      });
  }, [activeSymbol]);

  return (
    <div className="fade-in" style={{ maxWidth: '900px' }}>
      <h2 style={{ marginBottom: '24px', fontWeight: '500' }}>뉴스 & 감성 분석</h2>
      
      <div style={{ display: 'flex', gap: '12px', marginBottom: '24px', overflowX: 'auto', paddingBottom: '8px' }}>
        {symbols.map(sym => (
          <div 
            key={sym} 
            onClick={() => setActiveSymbol(sym)}
            style={{ 
              padding: '8px 20px', 
              borderRadius: '20px', 
              border: `1px solid ${activeSymbol === sym ? 'var(--accent-blue)' : 'var(--border-color)'}`,
              backgroundColor: activeSymbol === sym ? 'rgba(21, 101, 255, 0.15)' : 'var(--card-bg)',
              color: activeSymbol === sym ? 'var(--text-primary)' : 'var(--text-secondary)',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500',
              transition: 'all 0.2s',
              whiteSpace: 'nowrap'
            }}
          >
            {sym}
          </div>
        ))}
      </div>

      {loading ? (
        <div style={{ color: 'var(--text-secondary)', padding: '40px 0' }}>뉴스를 불러오는 중...</div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {news.map(n => (
            <div key={n.id} style={{ 
              backgroundColor: 'var(--card-bg)', 
              padding: '24px', 
              borderRadius: '8px', 
              border: '1px solid var(--border-color)',
              transition: 'transform 0.2s',
              cursor: 'pointer'
            }} onMouseOver={e => e.currentTarget.style.transform = 'translateY(-2px)'} onMouseOut={e => e.currentTarget.style.transform = 'translateY(0)'}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                <h3 style={{ fontSize: '18px', margin: 0, fontWeight: '500', lineHeight: '1.4' }}>{n.headline}</h3>
                <span style={{ 
                  fontSize: '12px', 
                  padding: '6px 12px', 
                  borderRadius: '20px', 
                  backgroundColor: n.sentiment === 'Positive' ? 'rgba(0, 214, 143, 0.1)' : n.sentiment === 'Negative' ? 'rgba(255, 77, 106, 0.1)' : 'rgba(123, 141, 176, 0.1)',
                  color: n.sentiment === 'Positive' ? 'var(--green)' : n.sentiment === 'Negative' ? 'var(--red)' : 'var(--text-secondary)',
                  fontWeight: 'bold',
                  border: `1px solid ${n.sentiment === 'Positive' ? 'rgba(0, 214, 143, 0.2)' : n.sentiment === 'Negative' ? 'rgba(255, 77, 106, 0.2)' : 'rgba(123, 141, 176, 0.2)'}`
                }}>
                  {n.sentiment}
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px', color: 'var(--text-secondary)', fontSize: '13px' }}>
                <span style={{ backgroundColor: 'var(--sidebar-bg)', padding: '4px 8px', borderRadius: '4px' }}>{n.source}</span>
                <span>•</span>
                <span>{n.time}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function AIAnalysis({ stocks }) {
  const [selectedSymbol, setSelectedSymbol] = useState(stocks[0]?.symbol || 'NVDA');
  const s = stocks.find(st => st.symbol === selectedSymbol) || stocks[0];
  
  if (!s) return null;

  return (
    <div className="fade-in">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h2 style={{ fontWeight: '500', margin: 0 }}>심층 AI 분석</h2>
        <select 
          value={selectedSymbol} 
          onChange={(e) => setSelectedSymbol(e.target.value)}
          style={{
            backgroundColor: 'var(--card-bg)',
            color: 'var(--text-primary)',
            border: '1px solid var(--border-color)',
            padding: '8px 16px',
            borderRadius: '4px',
            outline: 'none',
            fontSize: '14px',
            cursor: 'pointer'
          }}
        >
          {stocks.map(st => <option key={st.symbol} value={st.symbol}>{st.symbol} - {st.name}</option>)}
        </select>
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '24px' }}>
        <div style={{ backgroundColor: 'var(--card-bg)', padding: '32px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
            <h3 style={{ fontSize: '18px', fontWeight: '500', margin: 0 }}>Model Confidence Scores</h3>
            <span style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>Updated 2 mins ago</span>
          </div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <ScoreBar label="LSTM Price Prediction (시계열)" score={s.lstm_score} color="var(--accent-blue)" />
            <ScoreBar label="CNN Pattern Recognition (차트 패턴)" score={s.cnn_score} color="#9d4edd" />
            <ScoreBar label="Transformer (거시경제 복합)" score={s.transformer_score} color="#ff9e00" />
            <ScoreBar label="FinBERT Sentiment (뉴스/재무제표)" score={s.sentiment_score} color="#00b4d8" />
          </div>
          
          <div style={{ marginTop: '40px', padding: '20px', backgroundColor: 'var(--sidebar-bg)', borderRadius: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', border: '1px solid rgba(21, 101, 255, 0.2)' }}>
            <div>
              <div style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '4px' }}>Composite AI Score</div>
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Weighted average of 4 deep learning models</div>
            </div>
            <div className="mono" style={{ fontSize: '36px', fontWeight: 'bold', color: s.ai_score > 70 ? 'var(--green)' : s.ai_score < 40 ? 'var(--red)' : 'var(--yellow)', textShadow: '0 0 10px rgba(0,0,0,0.5)' }}>
              {s.ai_score} <span style={{ fontSize: '18px', color: 'var(--text-secondary)' }}>/100</span>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div style={{ backgroundColor: 'var(--card-bg)', padding: '32px', borderRadius: '8px', border: '1px solid var(--border-color)', flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ color: 'var(--text-secondary)', marginBottom: '16px', fontSize: '14px' }}>7-Day Target Price</div>
            <div className="mono" style={{ fontSize: '42px', fontWeight: 'bold' }}>${(s.price * (1 + (s.ai_score-50)/500)).toFixed(2)}</div>
            <div className={`mono ${s.ai_score >= 50 ? 'text-green' : 'text-red'}`} style={{ marginTop: '12px', fontSize: '16px', padding: '4px 12px', backgroundColor: s.ai_score >= 50 ? 'rgba(0, 214, 143, 0.1)' : 'rgba(255, 77, 106, 0.1)', borderRadius: '20px' }}>
              {s.ai_score >= 50 ? '▲' : '▼'} {Math.abs((s.ai_score-50)/5).toFixed(1)}% Expected
            </div>
          </div>
          
          <div style={{ backgroundColor: 'var(--card-bg)', padding: '32px', borderRadius: '8px', border: '1px solid var(--border-color)', flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
             <div style={{ color: 'var(--text-secondary)', marginBottom: '20px', fontSize: '14px' }}>Consensus Action Signal</div>
             <SignalBadge signal={s.signal} large={true} />
             <div style={{ marginTop: '20px', fontSize: '13px', color: 'var(--text-secondary)' }}>System Confidence: <span className="mono text-primary">{(s.lstm_score * 0.4 + s.sentiment_score * 0.6).toFixed(1)}%</span></div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ScoreBar({ label, score, color }) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px', fontSize: '14px' }}>
        <span>{label}</span>
        <span className="mono" style={{ fontWeight: 'bold' }}>{score}</span>
      </div>
      <div style={{ height: '8px', backgroundColor: 'var(--bg-color)', borderRadius: '4px', overflow: 'hidden' }}>
        <div style={{ width: `${score}%`, height: '100%', backgroundColor: color, borderRadius: '4px', transition: 'width 1s cubic-bezier(0.4, 0, 0.2, 1)' }}></div>
      </div>
    </div>
  );
}

function SignalBadge({ signal, large=false }) {
  let bg, color, border;
  if (signal === 'BUY') { 
    bg = 'rgba(0, 214, 143, 0.15)'; 
    color = 'var(--green)'; 
    border = 'rgba(0, 214, 143, 0.3)';
  } else if (signal === 'SELL') { 
    bg = 'rgba(255, 77, 106, 0.15)'; 
    color = 'var(--red)'; 
    border = 'rgba(255, 77, 106, 0.3)';
  } else { 
    bg = 'rgba(255, 179, 0, 0.15)'; 
    color = 'var(--yellow)'; 
    border = 'rgba(255, 179, 0, 0.3)';
  }

  return (
    <span style={{ 
      backgroundColor: bg, 
      color: color, 
      border: `1px solid ${border}`,
      padding: large ? '12px 40px' : '6px 14px', 
      borderRadius: '24px', 
      fontSize: large ? '24px' : '12px', 
      fontWeight: 'bold',
      letterSpacing: '1px',
      display: 'inline-block',
      textAlign: 'center',
      minWidth: large ? 'auto' : '64px',
      boxShadow: large ? `0 0 20px ${bg}` : 'none'
    }}>
      {signal}
    </span>
  );
}
