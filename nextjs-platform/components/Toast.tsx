"use client"

import { createContext, useContext, useState, useCallback, useEffect } from "react"

type ToastType = "success" | "error" | "info"

interface Toast {
  id: string
  message: string
  type: ToastType
}

interface ToastContextValue {
  toast: (message: string, type?: ToastType) => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])

  const toast = useCallback((message: string, type: ToastType = "info") => {
    const id = Math.random().toString(36).slice(2)
    setToasts((prev) => [...prev, { id, message, type }])
  }, [])

  const remove = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      {/* Win98-style notification popups */}
      <div style={{
        position: "fixed",
        bottom: "36px",
        right: "12px",
        zIndex: 100,
        display: "flex",
        flexDirection: "column",
        gap: "6px",
        pointerEvents: "none",
      }}>
        {toasts.map((t) => (
          <ToastItem key={t.id} toast={t} onRemove={remove} />
        ))}
      </div>
    </ToastContext.Provider>
  )
}

function ToastItem({ toast, onRemove }: { toast: Toast; onRemove: (id: string) => void }) {
  useEffect(() => {
    const timer = setTimeout(() => onRemove(toast.id), 4000)
    return () => clearTimeout(timer)
  }, [toast.id, onRemove])

  const titleMap: Record<ToastType, string> = {
    success: "✅ 완료",
    error: "⛔ 오류",
    info: "ℹ️ 알림",
  }

  return (
    <div
      className="win98-window"
      style={{
        pointerEvents: "all",
        minWidth: "260px",
        maxWidth: "320px",
        boxShadow: "4px 4px 8px rgba(0,0,0,0.4)",
      }}
    >
      {/* Title bar */}
      <div className="win98-title" style={{ fontSize: "11px" }}>
        <span>{titleMap[toast.type]}</span>
        <button
          onClick={() => onRemove(toast.id)}
          className="win98-title-btn"
          style={{ fontFamily: "serif", cursor: "pointer", border: "none", background: "none", color: "white" }}
        >
          ✕
        </button>
      </div>

      {/* Content */}
      <div style={{ padding: "8px 12px", fontSize: "11px", display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{ fontSize: "18px" }}>
          {toast.type === "success" ? "✅" : toast.type === "error" ? "🚫" : "📢"}
        </span>
        <span>{toast.message}</span>
      </div>

      {/* Status bar */}
      <div className="win98-statusbar" style={{ fontSize: "10px" }}>
        <div className="win98-status-panel" style={{ flex: 1 }}>
          4초 후 자동 닫힘
        </div>
      </div>
    </div>
  )
}

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error("useToast must be used within ToastProvider")
  return ctx
}
