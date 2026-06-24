"use client"

import Link from "next/link"
import { signOut } from "next-auth/react"
import type { Session } from "next-auth"

function Clock() {
  // Static time for SSR safety — updates client-side
  return (
    <span suppressHydrationWarning>
      {typeof window !== "undefined"
        ? new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" })
        : "00:00"}
    </span>
  )
}

export default function Navbar({ session }: { session: Session | null }) {
  return (
    <div className="win98-taskbar">
      {/* Start button */}
      <Link href="/" className="win98-start" style={{ textDecoration: "none", color: "inherit" }}>
        <span style={{ fontSize: "14px" }}>🪟</span>
        <span>시작</span>
      </Link>

      <div className="win98-sep" style={{ width: "1px", height: "20px", margin: "0 2px" }} />

      {/* Active windows */}
      {session && (
        <button
          className="win98-raised"
          style={{
            background: "var(--w98-bg)",
            padding: "2px 10px",
            fontSize: "11px",
            height: "22px",
            display: "flex",
            alignItems: "center",
            gap: "6px",
            cursor: "default",
            border: "none",
          }}
        >
          <span>📚</span>
          <span>DeepLearn95 - 대시보드</span>
        </button>
      )}

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* System tray */}
      <div
        className="win98-sunken"
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          padding: "2px 6px",
          height: "22px",
          fontSize: "11px",
        }}
      >
        {session ? (
          <>
            <span title={session.user?.email ?? ""}>👤 {session.user?.name?.split(" ")[0]}</span>
            <div className="win98-sep" style={{ width: "1px", height: "14px", margin: "0" }} />
            <button
              onClick={() => signOut({ callbackUrl: "/" })}
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                fontSize: "11px",
                padding: 0,
                fontFamily: "inherit",
                color: "var(--w98-text)",
              }}
            >
              로그아웃
            </button>
            <div className="win98-sep" style={{ width: "1px", height: "14px", margin: "0" }} />
          </>
        ) : null}
        <span>🔊</span>
        <span>🌐</span>
        <div className="win98-sep" style={{ width: "1px", height: "14px", margin: "0" }} />
        <Clock />
      </div>
    </div>
  )
}
