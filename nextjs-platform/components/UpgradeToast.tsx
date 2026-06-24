"use client"

import { useEffect } from "react"
import { useToast } from "@/components/Toast"
import { useRouter } from "next/navigation"

export default function UpgradeToast() {
  const { toast } = useToast()
  const router = useRouter()

  useEffect(() => {
    toast("결제 완료! Pro 콘텐츠가 활성화되었습니다 🎉", "success")
    router.replace("/dashboard")
  }, [toast, router])

  return null
}
