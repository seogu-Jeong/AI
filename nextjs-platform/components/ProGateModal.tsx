"use client"

import { useCheckout } from "@/hooks/useCheckout"

interface ProGateModalProps {
  isOpen: boolean
  onClose: () => void
  topicTitle: string
}

export default function ProGateModal({ isOpen, onClose, topicTitle }: ProGateModalProps) {
  const { loading, startCheckout } = useCheckout()

  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center px-4"
      onClick={onClose}
    >
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />

      <div
        className="relative w-full max-w-md rounded-2xl border border-white/10 bg-gray-900 p-8 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute right-4 top-4 rounded-full p-1.5 text-white/40 transition-colors hover:bg-white/10 hover:text-white/80"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        <div className="mb-6 flex justify-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-purple-500/10 ring-1 ring-purple-500/20">
            <svg className="h-8 w-8 text-purple-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
            </svg>
          </div>
        </div>

        <h2 className="mb-2 text-center text-xl font-bold text-white">Pro 전용 콘텐츠</h2>
        <p className="mb-1 text-center text-sm text-white/60">
          <span className="font-medium text-white/80">{topicTitle}</span>은 Pro 플랜 전용입니다.
        </p>
        <p className="mb-8 text-center text-sm text-white/40">
          Pro로 업그레이드하면 나머지 3개 토픽 전체에 접근할 수 있습니다.
        </p>

        <ul className="mb-8 space-y-2.5">
          {["🖼️ Data Augmentation", "🧠 Transfer Learning", "✍️ MNIST CNN 실습"].map((item) => (
            <li key={item} className="flex items-center gap-3 text-sm text-white/70">
              <svg className="h-4 w-4 shrink-0 text-purple-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              {item}
            </li>
          ))}
        </ul>

        <button
          onClick={startCheckout}
          disabled={loading}
          className="w-full rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 py-3.5 text-sm font-semibold text-white shadow-lg shadow-violet-500/20 transition-all hover:shadow-violet-500/40 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:scale-100"
        >
          {loading ? "결제 페이지 이동 중..." : "Pro로 업그레이드"}
        </button>

        <p className="mt-4 text-center text-xs text-white/30">
          Polar.sh 결제 · 안전한 보안 연결
        </p>
      </div>
    </div>
  )
}
