"use client"

import { signOut } from "next-auth/react"

export default function LogoutButton() {
  return (
    <button
      onClick={() => signOut({ callbackUrl: "/" })}
      className="win98-btn"
      style={{ minWidth: "80px" }}
    >
      로그아웃
    </button>
  )
}
