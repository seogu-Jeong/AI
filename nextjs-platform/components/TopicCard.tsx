"use client"

import { useRouter } from "next/navigation"

interface TopicCardProps {
  slug: string
  icon: string
  title: string
  subtitle: string
  tags: string[]
  isFree: boolean
  description: string
}

export default function TopicCard({ slug, icon, title, subtitle, tags, isFree, description }: TopicCardProps) {
  const router = useRouter()

  return (
    <div
      className="win98-window"
      onClick={() => router.push(`/dashboard/topics/${slug}`)}
      style={{ cursor: "pointer" }}
    >
      {/* Title bar */}
      <div className="win98-title" style={{ fontSize: "11px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          <span>{icon}</span>
          <span>{title}</span>
        </div>
        <div style={{ display: "flex", gap: "2px" }}>
          {!isFree && <span style={{ fontSize: "9px", background: "#ffcc00", color: "#000", padding: "0 4px", borderRadius: "2px" }}>PRO</span>}
          <span className="win98-title-btn" style={{ fontFamily: "serif" }}>✕</span>
        </div>
      </div>

      {/* Content */}
      <div style={{ padding: "10px 12px" }}>
        {/* File info row */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
          <span style={{ fontSize: "32px" }}>{icon}</span>
          <div>
            <div style={{ fontSize: "11px", fontWeight: "bold" }}>{subtitle}</div>
            <div style={{ fontSize: "10px", color: "var(--w98-dark)" }}>
              {isFree ? "🔓 무료 접근 가능" : "🔒 Pro 필요"}
            </div>
          </div>
        </div>

        <div className="win98-sep" />

        {/* Description */}
        <div style={{ fontSize: "11px", margin: "6px 0", lineHeight: "1.5", color: "var(--w98-text)" }}>
          {description}
        </div>

        {/* Tags */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: "3px", marginTop: "6px" }}>
          {tags.map((tag) => (
            <span
              key={tag}
              className="win98-raised"
              style={{ fontSize: "10px", padding: "1px 5px", background: "var(--w98-bg)" }}
            >
              {tag}
            </span>
          ))}
        </div>
      </div>

      {/* Status bar */}
      <div className="win98-statusbar" style={{ fontSize: "10px" }}>
        <div className="win98-status-panel" style={{ flex: 1 }}>
          📄 {subtitle}
        </div>
        <div className="win98-status-panel">
          {isFree ? "무료" : "Pro"}
        </div>
      </div>
    </div>
  )
}
