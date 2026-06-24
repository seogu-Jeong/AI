import { getServerSession } from "next-auth"
import { redirect } from "next/navigation"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import Image from "next/image"
import TopicCard from "@/components/TopicCard"
import PricingSection from "@/components/PricingSection"
import UpgradeToast from "@/components/UpgradeToast"
import { TOPICS } from "@/lib/topics"

const FREE_VIEW_LIMIT = 5

async function getUserViewCount(userId: string): Promise<number> {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL
  if (!apiUrl) return 0
  try {
    const res = await fetch(`${apiUrl}/users/${userId}`, {
      headers: { "X-Internal-Secret": process.env.INTERNAL_SECRET! },
      cache: "no-store",
    })
    if (!res.ok) return 0
    const user = await res.json()
    return user.view_count ?? 0
  } catch {
    return 0
  }
}

export default async function Dashboard({
  searchParams,
}: {
  searchParams: Promise<{ upgraded?: string }>
}) {
  const session = await getServerSession(authOptions)
  if (!session) redirect("/")

  const params = await searchParams
  const justUpgraded = params.upgraded === "true"
  const isPro = session.user.isPro

  const viewCount = !isPro && session.user.id
    ? await getUserViewCount(session.user.id)
    : 0
  const viewsRemaining = Math.max(0, FREE_VIEW_LIMIT - viewCount)
  const viewsUsedPercent = Math.min(100, (viewCount / FREE_VIEW_LIMIT) * 100)

  return (
    <div style={{ minHeight: "calc(100vh - 28px)", padding: "12px", display: "flex", flexDirection: "column", gap: "10px" }}>
      {justUpgraded && <UpgradeToast />}

      {/* My Computer Window */}
      <div className="win98-window" style={{ width: "100%", maxWidth: "860px", margin: "0 auto" }}>
        {/* Title Bar */}
        <div className="win98-title">
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span>🖥️</span>
            <span>DeepLearn95 — 내 강의실</span>
          </div>
          <div style={{ display: "flex", gap: "2px" }}>
            <span className="win98-title-btn">_</span>
            <span className="win98-title-btn">□</span>
            <span className="win98-title-btn" style={{ fontFamily: "serif" }}>✕</span>
          </div>
        </div>

        {/* Menu bar */}
        <div style={{
          display: "flex",
          gap: "0",
          padding: "2px 4px",
          fontSize: "11px",
          borderBottom: "1px solid var(--w98-dark)",
          background: "var(--w98-bg)",
        }}>
          {["파일(F)", "편집(E)", "보기(V)", "이동(G)", "즐겨찾기(A)", "도구(T)", "도움말(H)"].map((item) => (
            <span
              key={item}
              style={{ padding: "2px 8px", cursor: "default" }}
              className="win98-menu-item"
            >
              {item}
            </span>
          ))}
        </div>

        {/* Toolbar */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "4px",
          padding: "3px 6px",
          borderBottom: "1px solid var(--w98-dark)",
          background: "var(--w98-bg)",
        }}>
          {[
            { icon: "◀", label: "뒤로" },
            { icon: "▶", label: "앞으로" },
            { icon: "↑", label: "위로" },
            { icon: "🔍", label: "검색" },
            { icon: "📁", label: "폴더" },
          ].map((btn) => (
            <button key={btn.label} className="win98-btn" style={{ fontSize: "10px", padding: "1px 8px", minWidth: "auto" }}>
              {btn.icon} {btn.label}
            </button>
          ))}
          <div style={{ flex: 1 }} />
          {/* User info */}
          <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px" }}>
            {session.user?.image ? (
              <Image
                src={session.user.image}
                alt={session.user.name ?? "User"}
                width={18}
                height={18}
                style={{ borderRadius: "50%", border: "1px solid var(--w98-dark)" }}
              />
            ) : (
              <span>👤</span>
            )}
            <span>{session.user?.name?.split(" ")[0]}</span>
            {isPro && (
              <span style={{ background: "#ffcc00", color: "#000", fontSize: "9px", padding: "0 4px", borderRadius: "2px" }}>
                PRO
              </span>
            )}
          </div>
        </div>

        {/* Address bar */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "6px",
          padding: "3px 6px",
          borderBottom: "1px solid var(--w98-dark)",
          background: "var(--w98-bg)",
          fontSize: "11px",
        }}>
          <span style={{ color: "var(--w98-dark)" }}>주소:</span>
          <div className="win98-sunken" style={{ flex: 1, padding: "1px 6px", fontSize: "11px" }}>
            C:\DeepLearn95\Week5\강의실
          </div>
          <button className="win98-btn" style={{ fontSize: "10px", padding: "1px 10px" }}>이동</button>
        </div>

        {/* Main content area */}
        <div style={{ display: "flex", minHeight: "480px" }}>
          {/* Left panel — folder tree */}
          <div className="win98-sunken" style={{
            width: "160px",
            minWidth: "160px",
            padding: "4px",
            borderRight: "none",
            fontSize: "11px",
            overflow: "auto",
          }}>
            <div style={{ fontWeight: "bold", marginBottom: "6px", padding: "2px" }}>폴더</div>
            {[
              { icon: "🖥️", label: "내 컴퓨터", indent: 0 },
              { icon: "💾", label: "C: 드라이브", indent: 1 },
              { icon: "📁", label: "DeepLearn95", indent: 2 },
              { icon: "📂", label: "Week5 강의실", indent: 3, active: true },
              { icon: "📁", label: "Week4", indent: 2 },
              { icon: "📁", label: "Week3", indent: 2 },
              { icon: "🗑️", label: "휴지통", indent: 0 },
            ].map((item) => (
              <div
                key={item.label}
                style={{
                  paddingLeft: `${item.indent * 10 + 4}px`,
                  paddingTop: "2px",
                  paddingBottom: "2px",
                  background: item.active ? "var(--w98-navy)" : "transparent",
                  color: item.active ? "white" : "inherit",
                  cursor: "default",
                  display: "flex",
                  alignItems: "center",
                  gap: "4px",
                }}
              >
                <span>{item.icon}</span>
                <span>{item.label}</span>
              </div>
            ))}
          </div>

          {/* Right panel — file icons */}
          <div style={{ flex: 1, padding: "8px", background: "white", overflow: "auto" }}>
            {/* View count banner */}
            {!isPro && (
              <div style={{
                marginBottom: "10px",
                padding: "6px 10px",
                border: `1px solid ${viewsRemaining === 0 ? "#c0392b" : viewsRemaining <= 2 ? "#e67e22" : "var(--w98-dark)"}`,
                background: viewsRemaining === 0 ? "#fce4e4" : viewsRemaining <= 2 ? "#fef3e2" : "#f0f0f0",
                fontSize: "11px",
                display: "flex",
                alignItems: "center",
                gap: "10px",
              }}>
                <span style={{ fontSize: "14px" }}>
                  {viewsRemaining === 0 ? "🔴" : viewsRemaining <= 2 ? "🟡" : "🟢"}
                </span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: "bold", marginBottom: "3px" }}>
                    {viewsRemaining === 0
                      ? "⚠️ 무료 조회 횟수 소진 — Pro 업그레이드 필요"
                      : `무료 조회 ${viewCount}/${FREE_VIEW_LIMIT}회 사용 · 잔여 ${viewsRemaining}회`}
                  </div>
                  {/* Win98 progress bar */}
                  <div className="win98-progress-track" style={{ height: "12px" }}>
                    <div
                      className="win98-progress-bar"
                      style={{
                        width: `${viewsUsedPercent}%`,
                        background: viewsRemaining === 0
                          ? "repeating-linear-gradient(90deg,#c0392b 0,#c0392b 8px,#e74c3c 8px,#e74c3c 16px)"
                          : viewsRemaining <= 2
                          ? "repeating-linear-gradient(90deg,#e67e22 0,#e67e22 8px,#f39c12 8px,#f39c12 16px)"
                          : "repeating-linear-gradient(90deg,#000080 0,#000080 8px,#1565C0 8px,#1565C0 16px)",
                      }}
                    />
                  </div>
                </div>
                {viewsRemaining <= 3 && (
                  <a href="#pricing" className="win98-btn" style={{ fontSize: "10px", textDecoration: "none", whiteSpace: "nowrap" }}>
                    Pro 업그레이드
                  </a>
                )}
              </div>
            )}

            {/* File grid */}
            <div style={{
              fontSize: "11px",
              marginBottom: "6px",
              color: "var(--w98-dark)",
              borderBottom: "1px solid var(--w98-dark)",
              paddingBottom: "4px",
            }}>
              5개 개체 · {isPro ? "Pro 플랜 — 무제한 접근" : `무료 ${viewsRemaining}회 조회 가능`}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))", gap: "8px" }}>
              {TOPICS.map((topic) => (
                <TopicCard
                  key={topic.slug}
                  slug={topic.slug}
                  icon={topic.icon}
                  title={topic.title}
                  subtitle={topic.subtitle}
                  tags={topic.tags}
                  isFree={true}
                  description={topic.description}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Status bar */}
        <div className="win98-statusbar" style={{ fontSize: "10px" }}>
          <div className="win98-status-panel" style={{ flex: 1 }}>
            5개 개체
          </div>
          <div className="win98-status-panel" style={{ flex: 2 }}>
            C:\DeepLearn95\Week5\강의실
          </div>
          <div className="win98-status-panel">
            {isPro ? "Pro 플랜" : `잔여 ${viewsRemaining}회`}
          </div>
        </div>
      </div>

      {/* Pricing section */}
      {!isPro && (
        <div style={{ maxWidth: "860px", margin: "0 auto", width: "100%" }} id="pricing">
          <PricingSection isPro={isPro} />
        </div>
      )}
    </div>
  )
}
