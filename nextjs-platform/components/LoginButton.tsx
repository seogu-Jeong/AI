"use client"

import { signIn } from "next-auth/react"

export default function LoginButton() {
  return (
    <div style={{ display: "flex", justifyContent: "center", gap: "8px" }}>
      <button
        onClick={() => signIn("google", { callbackUrl: "/dashboard" })}
        className="win98-btn win98-btn-primary"
        style={{ minWidth: "140px", fontSize: "12px", padding: "6px 16px" }}
      >
        🔑 Google로 로그인
      </button>
      <button className="win98-btn" onClick={() => {}}>
        취소
      </button>
    </div>
  )
}
