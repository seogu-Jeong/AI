"use client"

import { useState } from "react"
import { useToast } from "@/components/Toast"

export function useCheckout() {
  const [loading, setLoading] = useState(false)
  const { toast } = useToast()

  async function startCheckout() {
    setLoading(true)
    try {
      const res = await fetch("/api/checkout", { method: "POST" })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error ?? "결제 요청 실패")
      window.location.href = data.url
    } catch (err) {
      toast(err instanceof Error ? err.message : "결제 중 오류가 발생했습니다.", "error")
      setLoading(false)
    }
  }

  return { loading, startCheckout }
}
