import { getServerSession } from "next-auth"
import { redirect, notFound } from "next/navigation"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import { getTopicBySlug, TOPICS } from "@/lib/topics"
import Link from "next/link"
import ViewLimitWall from "@/components/ViewLimitWall"

export async function generateStaticParams() {
  return TOPICS.map((t) => ({ slug: t.slug }))
}

async function recordView(userId: string): Promise<{
  allowed: boolean
  view_count: number
  views_remaining: number
  is_pro: boolean
} | null> {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL
  if (!apiUrl) return null
  try {
    const res = await fetch(`${apiUrl}/users/${userId}/view`, {
      method: "POST",
      headers: { "X-Internal-Secret": process.env.INTERNAL_SECRET! },
      cache: "no-store",
    })
    if (!res.ok) return null
    return res.json()
  } catch {
    return null
  }
}

export default async function TopicPage({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const session = await getServerSession(authOptions)
  if (!session) redirect("/")

  const { slug } = await params
  const topic = getTopicBySlug(slug)
  if (!topic) notFound()

  const userId = session.user.id
  let viewData = null
  if (userId) {
    viewData = await recordView(userId)
  }

  const isAllowed = session.user.isPro || !viewData || viewData.allowed

  const topicIndex = TOPICS.findIndex((t) => t.slug === slug)
  const prevTopic = topicIndex > 0 ? TOPICS[topicIndex - 1] : null
  const nextTopic = topicIndex < TOPICS.length - 1 ? TOPICS[topicIndex + 1] : null

  return (
    <div style={{ minHeight: "calc(100vh - 28px)", padding: "12px" }}>
      <div style={{ maxWidth: "820px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "8px" }}>

        {/* Breadcrumb / Back */}
        <div style={{ fontSize: "11px", display: "flex", alignItems: "center", gap: "4px", color: "var(--w98-dark)" }}>
          <Link href="/dashboard" style={{ color: "var(--w98-navy)", textDecoration: "underline", fontSize: "11px" }}>
            🖥️ 내 강의실
          </Link>
          <span>▶</span>
          <span>{topic.subtitle} — {topic.title}</span>
        </div>

        {/* Notepad window */}
        <div className="win98-window">
          {/* Title bar */}
          <div className="win98-title">
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <span>📄</span>
              <span>{topic.subtitle} — {topic.title} - 메모장</span>
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
          }}>
            {["파일(F)", "편집(E)", "서식(O)", "보기(V)", "도움말(H)"].map((item) => (
              <span key={item} style={{ padding: "2px 8px", cursor: "default" }}>
                {item}
              </span>
            ))}
          </div>

          {!isAllowed ? (
            <ViewLimitWall />
          ) : (
            <div style={{ padding: "16px", background: "white" }}>
              {/* View count warning banner */}
              {!session.user.isPro && viewData && viewData.views_remaining >= 0 && (
                <div style={{
                  marginBottom: "14px",
                  padding: "6px 10px",
                  border: `1px solid ${viewData.views_remaining <= 1 ? "#c0392b" : "var(--w98-dark)"}`,
                  background: viewData.views_remaining <= 1 ? "#fce4e4" : "#f5f5f5",
                  fontSize: "11px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: "8px",
                }}>
                  <span>
                    {viewData.views_remaining === 0
                      ? "⚠️ 마지막 무료 조회였습니다. 다음 방문부터는 Pro 플랜이 필요합니다."
                      : `ℹ️ 무료 조회 ${viewData.views_remaining}회 남음 (총 5회 제공)`}
                  </span>
                  {viewData.views_remaining <= 2 && (
                    <Link href="/dashboard#pricing" className="win98-btn" style={{ fontSize: "10px", textDecoration: "none", whiteSpace: "nowrap" }}>
                      Pro 보기
                    </Link>
                  )}
                </div>
              )}

              {/* Topic header */}
              <div style={{ marginBottom: "16px", borderBottom: "1px solid #ccc", paddingBottom: "12px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
                  <span style={{ fontSize: "36px" }}>{topic.icon}</span>
                  <div>
                    <div style={{ fontSize: "10px", color: "var(--w98-dark)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      {topic.subtitle}
                    </div>
                    <h1 style={{ fontSize: "18px", fontWeight: "bold", margin: "2px 0" }}>{topic.title}</h1>
                  </div>
                </div>
                <p style={{ fontSize: "12px", lineHeight: "1.6", color: "#333", marginBottom: "8px" }}>
                  {topic.description}
                </p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "3px" }}>
                  {topic.tags.map((tag) => (
                    <span
                      key={tag}
                      className="win98-raised"
                      style={{ fontSize: "10px", padding: "1px 6px", background: "var(--w98-bg)" }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>

              {/* Content sections */}
              <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                {topic.content.sections.map((section, i) => (
                  <div key={section.heading} className="win98-sunken" style={{ padding: "10px 12px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
                      <span style={{
                        display: "inline-flex",
                        alignItems: "center",
                        justifyContent: "center",
                        width: "18px",
                        height: "18px",
                        background: "var(--w98-navy)",
                        color: "white",
                        fontSize: "10px",
                        fontWeight: "bold",
                        flexShrink: 0,
                      }}>
                        {i + 1}
                      </span>
                      <h2 style={{ fontSize: "12px", fontWeight: "bold", margin: 0 }}>{section.heading}</h2>
                    </div>
                    <p style={{ fontSize: "11px", lineHeight: "1.7", color: "#333", margin: 0 }}>
                      {section.body}
                    </p>
                  </div>
                ))}
              </div>

              {/* Code block */}
              {topic.content.code && (
                <div style={{ marginTop: "16px" }}>
                  {/* Code window title bar */}
                  <div className="win98-window" style={{ marginBottom: "0" }}>
                    <div className="win98-title" style={{ fontSize: "11px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                        <span>🐍</span>
                        <span>{topic.content.codeFile}</span>
                      </div>
                      <span style={{ fontSize: "9px", background: "#4CAF50", color: "white", padding: "0 4px" }}>Python</span>
                    </div>
                    <pre className="win98-code">
                      <code>{topic.content.code}</code>
                    </pre>
                  </div>
                </div>
              )}

              {/* Code explanations */}
              {topic.content.codeExplanations && topic.content.codeExplanations.length > 0 && (
                <div style={{ marginTop: "12px" }}>
                  <div style={{
                    fontSize: "12px",
                    fontWeight: "bold",
                    marginBottom: "8px",
                    padding: "4px 8px",
                    background: "var(--w98-navy)",
                    color: "white",
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                  }}>
                    <span>📝</span>
                    <span>코드 상세 해설</span>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                    {topic.content.codeExplanations.map((exp, i) => (
                      <div key={i} className="win98-raised" style={{ padding: "8px 10px" }}>
                        <div style={{ fontSize: "11px", fontWeight: "bold", marginBottom: "4px", display: "flex", alignItems: "center", gap: "6px" }}>
                          <span style={{ color: "var(--w98-navy)" }}>▶</span>
                          <code style={{ background: "#e8e8e8", padding: "0 4px", fontSize: "10px" }}>{exp.line}</code>
                          <span>{exp.title}</span>
                        </div>
                        <p style={{ fontSize: "11px", lineHeight: "1.6", margin: 0, color: "#333" }}>
                          {exp.explanation}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Navigation */}
              <div style={{
                marginTop: "20px",
                paddingTop: "10px",
                borderTop: "1px solid #ccc",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                fontSize: "11px",
              }}>
                <div>
                  {prevTopic && (
                    <Link href={`/dashboard/topics/${prevTopic.slug}`} className="win98-btn" style={{ textDecoration: "none", fontSize: "11px" }}>
                      ◀ {prevTopic.title}
                    </Link>
                  )}
                </div>
                <Link href="/dashboard" className="win98-btn" style={{ textDecoration: "none", fontSize: "11px" }}>
                  🏠 강의실로
                </Link>
                <div>
                  {nextTopic && (
                    <Link href={`/dashboard/topics/${nextTopic.slug}`} className="win98-btn" style={{ textDecoration: "none", fontSize: "11px" }}>
                      {nextTopic.title} ▶
                    </Link>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Status bar */}
          <div className="win98-statusbar" style={{ fontSize: "10px" }}>
            <div className="win98-status-panel" style={{ flex: 1 }}>
              {topic.subtitle} · {topic.tags.join(", ")}
            </div>
            <div className="win98-status-panel">
              {session.user.isPro ? "Pro" : viewData ? `조회 ${viewData.view_count}/5` : "Free"}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
