import { useState } from 'react'
import { useStockStore } from '@/store/stockStore'
import api from '@/lib/api'

interface ScreenerResult {
  code: string
  name: string
  signal: 'BUY' | 'HOLD' | 'SELL'
  signal_score: number
  tech_score: number | null
  lstm_score: number | null
  financial_score: number | null
  financial_grade: string | null
  financial_risk: boolean | null
  per: number | null
  pbr: number | null
  roe: number | null
  eps: number | null
  dividend_yield: number | null
}

interface ScreenerResponse {
  results: ScreenerResult[]
  matched: number
  scanned: number
}

interface Filters {
  signals: { BUY: boolean; HOLD: boolean; SELL: boolean }
  min_score: string
  grades: { 우수: boolean; 양호: boolean; 보통: boolean; 위험: boolean }
  max_per: string
  max_pbr: string
  min_roe: string
  exclude_risk: boolean
  foreign_net_buy: boolean
  institution_net_buy: boolean
  sort_by: 'signal_score' | 'financial_score' | 'per' | 'pbr'
}

const DEFAULT_FILTERS: Filters = {
  signals: { BUY: false, HOLD: false, SELL: false },
  min_score: '',
  grades: { 우수: false, 양호: false, 보통: false, 위험: false },
  max_per: '',
  max_pbr: '',
  min_roe: '',
  exclude_risk: false,
  foreign_net_buy: false,
  institution_net_buy: false,
  sort_by: 'signal_score',
}

const SIGNAL_STYLE: Record<string, string> = {
  BUY:  'text-green-400 bg-green-400/10 border border-green-400/30',
  SELL: 'text-red-400 bg-red-400/10 border border-red-400/30',
  HOLD: 'text-yellow-400 bg-yellow-400/10 border border-yellow-400/30',
}

const GRADE_STYLE: Record<string, string> = {
  우수: 'text-green-400',
  양호: 'text-blue-400',
  보통: 'text-yellow-400',
  위험: 'text-red-400',
}

function CheckGroup<K extends string>({
  label,
  keys,
  values,
  onChange,
}: {
  label: string
  keys: K[]
  values: Record<K, boolean>
  onChange: (k: K, v: boolean) => void
}) {
  return (
    <div>
      <div className="text-[11px] text-muted-foreground mb-1">{label}</div>
      <div className="flex flex-wrap gap-1.5">
        {keys.map((k) => (
          <label key={k} className="flex items-center gap-1 cursor-pointer">
            <input
              type="checkbox"
              checked={values[k]}
              onChange={(e) => onChange(k, e.target.checked)}
              className="w-3 h-3 accent-primary"
            />
            <span className="text-xs">{k}</span>
          </label>
        ))}
      </div>
    </div>
  )
}

function NumInput({
  label, value, onChange, placeholder,
}: {
  label: string; value: string; onChange: (v: string) => void; placeholder?: string
}) {
  return (
    <div>
      <div className="text-[11px] text-muted-foreground mb-1">{label}</div>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder ?? '제한 없음'}
        className="w-full bg-muted text-xs px-2 py-1.5 rounded border border-border focus:outline-none focus:border-primary"
      />
    </div>
  )
}

export function ScreenerTab() {
  const { setSelectedStock } = useStockStore()
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS)
  const [data, setData] = useState<ScreenerResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [selectedCode, setSelectedCode] = useState<string | null>(null)

  const setSignal = (k: keyof typeof filters.signals, v: boolean) =>
    setFilters((f) => ({ ...f, signals: { ...f.signals, [k]: v } }))

  const setGrade = (k: keyof typeof filters.grades, v: boolean) =>
    setFilters((f) => ({ ...f, grades: { ...f.grades, [k]: v } }))

  const handleSearch = async () => {
    setLoading(true)
    setData(null)
    try {
      const params: Record<string, string> = {
        sort_by: filters.sort_by,
        exclude_risk: String(filters.exclude_risk),
      }
      const sigs = Object.entries(filters.signals).filter(([, v]) => v).map(([k]) => k)
      if (sigs.length) params.signals = sigs.join(',')
      if (filters.min_score) params.min_score = filters.min_score
      const gs = Object.entries(filters.grades).filter(([, v]) => v).map(([k]) => k)
      if (gs.length) params.grades = gs.join(',')
      if (filters.max_per) params.max_per = filters.max_per
      if (filters.max_pbr) params.max_pbr = filters.max_pbr
      if (filters.min_roe) params.min_roe = filters.min_roe
      if (filters.foreign_net_buy) params.foreign_net_buy = 'true'
      if (filters.institution_net_buy) params.institution_net_buy = 'true'

      const { data: res } = await api.get<ScreenerResponse>('/analysis/screener', { params })
      setData(res)
    } catch {
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFilters(DEFAULT_FILTERS)
    setData(null)
    setSelectedCode(null)
  }

  const handleRowClick = (item: ScreenerResult) => {
    setSelectedCode(item.code)
    setSelectedStock({ code: item.code, name: item.name })
  }

  return (
    <div className="h-full overflow-hidden flex">
      {/* 좌측 필터 패널 */}
      <div className="w-52 shrink-0 border-r border-border flex flex-col overflow-y-auto p-3 gap-3 bg-card">
        <div className="text-sm font-semibold">필터 조건</div>

        <CheckGroup
          label="AI 신호"
          keys={['BUY', 'HOLD', 'SELL'] as const}
          values={filters.signals}
          onChange={setSignal}
        />

        <NumInput
          label="AI 점수 최솟값"
          value={filters.min_score}
          onChange={(v) => setFilters((f) => ({ ...f, min_score: v }))}
          placeholder="예: 60"
        />

        <CheckGroup
          label="재무 등급"
          keys={['우수', '양호', '보통', '위험'] as const}
          values={filters.grades}
          onChange={setGrade}
        />

        <NumInput
          label="PER 최댓값"
          value={filters.max_per}
          onChange={(v) => setFilters((f) => ({ ...f, max_per: v }))}
          placeholder="예: 15"
        />

        <NumInput
          label="PBR 최댓값"
          value={filters.max_pbr}
          onChange={(v) => setFilters((f) => ({ ...f, max_pbr: v }))}
          placeholder="예: 2"
        />

        <NumInput
          label="ROE 최솟값 (%)"
          value={filters.min_roe}
          onChange={(v) => setFilters((f) => ({ ...f, min_roe: v }))}
          placeholder="예: 10"
        />

        <div>
          <div className="text-[11px] text-muted-foreground mb-1">정렬 기준</div>
          <select
            value={filters.sort_by}
            onChange={(e) => setFilters((f) => ({ ...f, sort_by: e.target.value as Filters['sort_by'] }))}
            className="w-full bg-muted text-xs px-2 py-1.5 rounded border border-border focus:outline-none"
          >
            <option value="signal_score">AI 점수 높은 순</option>
            <option value="financial_score">재무 점수 높은 순</option>
            <option value="per">PER 낮은 순</option>
            <option value="pbr">PBR 낮은 순</option>
          </select>
        </div>

        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={filters.exclude_risk}
            onChange={(e) => setFilters((f) => ({ ...f, exclude_risk: e.target.checked }))}
            className="w-3 h-3 accent-primary"
          />
          <span className="text-xs">재무 위험 종목 제외</span>
        </label>

        <div>
          <div className="text-[11px] text-muted-foreground mb-1">투자자 동향 <span className="text-[10px] opacity-60">(느릴 수 있음)</span></div>
          <div className="flex flex-col gap-1">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={filters.foreign_net_buy}
                onChange={(e) => setFilters((f) => ({ ...f, foreign_net_buy: e.target.checked }))}
                className="w-3 h-3 accent-blue-400"
              />
              <span className="text-xs text-blue-400">외국인 5일 순매수</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={filters.institution_net_buy}
                onChange={(e) => setFilters((f) => ({ ...f, institution_net_buy: e.target.checked }))}
                className="w-3 h-3 accent-violet-400"
              />
              <span className="text-xs text-violet-400">기관 5일 순매수</span>
            </label>
          </div>
        </div>

        <div className="flex flex-col gap-1.5 mt-auto pt-2">
          <button
            onClick={handleSearch}
            disabled={loading}
            className="w-full py-1.5 text-xs rounded bg-primary text-primary-foreground font-medium disabled:opacity-50 hover:opacity-90 transition-opacity"
          >
            {loading ? '스캔 중…' : '검색'}
          </button>
          <button
            onClick={handleReset}
            className="w-full py-1.5 text-xs rounded border border-border text-muted-foreground hover:text-foreground transition-colors"
          >
            초기화
          </button>
        </div>
      </div>

      {/* 우측 결과 테이블 */}
      <div className="flex-1 min-w-0 overflow-y-auto">
        {/* 상태 바 */}
        <div className="sticky top-0 bg-card border-b border-border px-4 py-2 flex items-center gap-2">
          <span className="text-sm font-semibold">스크리너 결과</span>
          {data && (
            <span className="text-xs text-muted-foreground">
              {data.scanned}종목 스캔 · <span className="text-foreground font-medium">{data.matched}개</span> 조건 충족
            </span>
          )}
          {!data && !loading && (
            <span className="text-xs text-muted-foreground">조건을 설정하고 검색 버튼을 누르세요.</span>
          )}
        </div>

        {loading && (
          <div className="text-xs text-muted-foreground text-center py-12 space-y-2">
            <div>전체 종목 스캔 중…</div>
            <div className="text-[11px]">
              {(filters.foreign_net_buy || filters.institution_net_buy)
                ? '투자자 동향 필터 적용 중 — 추가 시간이 소요됩니다.'
                : '첫 번째 검색은 30-60초 소요됩니다.\n이후에는 캐시로 즉시 응답합니다.'
              }
            </div>
          </div>
        )}

        {!loading && data && data.results.length === 0 && (
          <div className="text-xs text-muted-foreground text-center py-12">
            조건에 맞는 종목이 없습니다. 필터를 완화해 보세요.
          </div>
        )}

        {!loading && data && data.results.length > 0 && (
          <table className="w-full text-xs">
            <thead className="sticky top-[37px] bg-card border-b border-border">
              <tr>
                <th className="px-3 py-2 text-left text-muted-foreground font-medium w-8">#</th>
                <th className="px-3 py-2 text-left text-muted-foreground font-medium">종목</th>
                <th className="px-3 py-2 text-center text-muted-foreground font-medium w-14">신호</th>
                <th className="px-3 py-2 text-right text-muted-foreground font-medium w-14">AI점수</th>
                <th className="px-3 py-2 text-center text-muted-foreground font-medium w-14">재무등급</th>
                <th className="px-3 py-2 text-right text-muted-foreground font-medium w-14">재무점수</th>
                <th className="px-3 py-2 text-right text-muted-foreground font-medium w-14">PER</th>
                <th className="px-3 py-2 text-right text-muted-foreground font-medium w-14">PBR</th>
                <th className="px-3 py-2 text-right text-muted-foreground font-medium w-14">ROE%</th>
              </tr>
            </thead>
            <tbody>
              {data.results.map((item, idx) => (
                <tr
                  key={item.code}
                  onClick={() => handleRowClick(item)}
                  className={`border-b border-border/50 cursor-pointer transition-colors ${
                    selectedCode === item.code ? 'bg-primary/10' : 'hover:bg-accent/50'
                  }`}
                >
                  <td className="px-3 py-2 text-muted-foreground">{idx + 1}</td>
                  <td className="px-3 py-2">
                    <div className="font-medium">{item.name}</div>
                    <div className="text-[10px] text-muted-foreground">{item.code}</div>
                  </td>
                  <td className="px-3 py-2 text-center">
                    <span className={`px-1.5 py-0.5 rounded text-[11px] font-semibold ${SIGNAL_STYLE[item.signal] ?? ''}`}>
                      {item.signal}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right font-medium">{item.signal_score.toFixed(0)}</td>
                  <td className="px-3 py-2 text-center">
                    <span className={`font-medium ${GRADE_STYLE[item.financial_grade ?? ''] ?? 'text-muted-foreground'}`}>
                      {item.financial_grade ?? '-'}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right text-muted-foreground">
                    {item.financial_score != null ? item.financial_score.toFixed(1) : '-'}
                  </td>
                  <td className="px-3 py-2 text-right text-muted-foreground">
                    {item.per != null ? item.per.toFixed(1) : '-'}
                  </td>
                  <td className="px-3 py-2 text-right text-muted-foreground">
                    {item.pbr != null ? item.pbr.toFixed(2) : '-'}
                  </td>
                  <td className="px-3 py-2 text-right text-muted-foreground">
                    {item.roe != null ? item.roe.toFixed(1) : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
